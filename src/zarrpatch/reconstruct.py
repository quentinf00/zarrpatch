"""Rebuild a datacube from (overlapping) patch predictions.

Typical inference flow: scan the domain exhaustively with a
:class:`~zarrpatch.samplers.GridSampler` (stride < patch size for overlap),
run the model on each patch, and accumulate the predictions back onto the
scanned dims with a position-dependent weight — by default a Hann window, so
patch centers dominate and overlapping patches blend smoothly at the seams::

    ds = ZarrPatchDataset(path, patch_size, sampler=GridSampler(stride=...),
                          shuffle=False, return_corners=True)
    rec = Reconstructor.from_dataset(ds)
    for batch in DataLoader(ds, batch_size=32, num_workers=4):
        rec.add(batch["corner"], model(batch["sst"]))
    cube = rec.result()   # (*extra_dims, *scanned_dims), NaN where uncovered

The prediction may have extra dims the input does not (channels, ensemble
members, ...): any axes *before* the scanned patch axes are carried through to
the output unchanged. Only the scanned dims — the trailing axes of the added
values — are reconstructed.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


def _window1d(kind: str, n: int) -> np.ndarray:
    """Strictly positive 1-D weight window (no zero weight sums at edges)."""
    if kind == "uniform":
        return np.ones(n)
    if kind == "triangle":
        return np.bartlett(n + 2)[1:-1]
    if kind == "hann":
        return np.hanning(n + 2)[1:-1]
    raise ValueError(
        f"unknown weight {kind!r}, expected 'hann', 'triangle', 'uniform' "
        "or an array"
    )


class Reconstructor:
    """Weighted-overlap reconstruction of scanned dims from patch values.

    Parameters
    ----------
    dim_sizes:
        Dim name -> size of the domain being rebuilt (extra keys are ignored,
        so a dataset's dim-size dict can be passed as is). When the dataset
        uses ``dim_slices``, these are the *sliced* sizes — corners and the
        reconstruction then both live in the sliced index space.
    patch_size:
        Dim name -> patch length of the *values* passed to :meth:`add`. If a
        model changes the scanned extent (e.g. center-crops its output), pass
        the output extent here and shift the corners accordingly yourself.
    weight:
        ``"hann"`` (default), ``"triangle"``, ``"uniform"``, or an array of
        shape ``(*patch_size)`` (in ``dims`` order) of non-negative weights.
        The named windows are strictly positive, so any covered cell has a
        nonzero weight sum. Overlapping patches are averaged with these
        per-position weights.
    dims:
        Order of the trailing (patch) axes of the values passed to
        :meth:`add`, and of the scanned axes of the output. Defaults to
        ``sorted(patch_size)``; :meth:`from_dataset` sets it to the on-disk
        variable order (e.g. ``("time", "lat", "lon")``).

    Corners may be dicts (or lists of dicts), or int arrays/tensors of shape
    ``(k,)`` or ``(batch, k)`` with columns ordered like ``sorted(patch_size)``
    — the convention of the dataset's ``"corner"`` key, so
    ``rec.add(batch["corner"], prediction)`` works directly. CUDA tensors must
    be moved to CPU first.

    Under DDP each rank sees a disjoint subset of the patches: build one
    Reconstructor per rank, gather them and :meth:`merge` (or all-reduce the
    ``sum`` and ``weight_sum`` arrays) before calling :meth:`result`.
    """

    def __init__(
        self,
        dim_sizes: Dict[str, int],
        patch_size: Dict[str, int],
        weight: Union[str, np.ndarray] = "hann",
        dims: Optional[Sequence[str]] = None,
    ):
        self.dims: List[str] = list(dims) if dims is not None else sorted(patch_size)
        if sorted(self.dims) != sorted(patch_size):
            raise ValueError(
                f"dims {self.dims} must be a permutation of the patch dims "
                f"{sorted(patch_size)}"
            )
        missing = [d for d in self.dims if d not in dim_sizes]
        if missing:
            raise ValueError(f"dim_sizes missing scanned dims {missing}")
        self.patch_size = {d: int(patch_size[d]) for d in self.dims}
        self.shape: Tuple[int, ...] = tuple(int(dim_sizes[d]) for d in self.dims)
        pshape = tuple(self.patch_size[d] for d in self.dims)

        if isinstance(weight, str):
            w = np.ones(pshape)
            for axis, n in enumerate(pshape):
                shape1 = [1] * len(pshape)
                shape1[axis] = n
                w = w * _window1d(weight, n).reshape(shape1)
        else:
            w = np.asarray(weight, dtype=np.float64)
            if w.shape != pshape:
                raise ValueError(
                    f"weight shape {w.shape} != patch shape {pshape} "
                    f"(dims order {self.dims})"
                )
            if (w < 0).any() or not (w > 0).any():
                raise ValueError("weights must be non-negative and not all zero")
        self._w = w
        self._sum: Optional[np.ndarray] = None  # (*extra, *self.shape), lazy
        self._wsum = np.zeros(self.shape)

    @classmethod
    def from_dataset(cls, dataset, weight: Union[str, np.ndarray] = "hann"
                     ) -> "Reconstructor":
        """Reconstructor for a dataset's scanned dims, in on-disk axis order."""
        v = dataset.variables[0]
        dims = [d for d in dataset.reader.dims(v) if d in dataset.patch_size]
        return cls(dataset._dim_sizes, dataset.patch_size, weight=weight,
                   dims=dims)

    # -- accumulation --------------------------------------------------------

    def _corner_array(self, corners) -> np.ndarray:
        """Normalize to an int (batch, k) array with columns in dims order."""
        if isinstance(corners, dict):
            corners = [corners]
        if (isinstance(corners, (list, tuple)) and corners
                and isinstance(corners[0], dict)):
            return np.array([[c[d] for d in self.dims] for c in corners],
                            dtype=np.int64)
        arr = np.asarray(corners)
        if arr.ndim == 1:
            arr = arr[None]
        if arr.ndim != 2 or arr.shape[1] != len(self.dims):
            raise ValueError(
                f"corners must be (batch, {len(self.dims)}) or "
                f"({len(self.dims)},), got shape {np.asarray(corners).shape}"
            )
        # array corners follow the dataset convention: sorted(patch_size) order
        perm = [sorted(self.dims).index(d) for d in self.dims]
        return arr[:, perm].astype(np.int64)

    def add(self, corners, values) -> None:
        """Accumulate patch ``values`` at ``corners`` (batched or single).

        ``values`` has shape ``(batch, *extra, *patch)`` for batched corners
        (a list of dicts or a 2-D array/tensor) or ``(*extra, *patch)`` for a
        single corner (one dict or a 1-D array). Extra dims must be identical
        across calls.
        """
        single = isinstance(corners, dict) or (
            not isinstance(corners, (list, tuple))
            and np.asarray(corners).ndim == 1
        )
        arr = self._corner_array(corners)
        values = np.asarray(values)
        if single:
            values = values[None]
        k = len(self.dims)
        pshape = tuple(self.patch_size[d] for d in self.dims)
        if (values.ndim < 1 + k or values.shape[0] != len(arr)
                or values.shape[-k:] != pshape):
            raise ValueError(
                f"values shape {values.shape} does not match {len(arr)} "
                f"corner(s) with trailing patch shape {pshape} "
                f"(dims order {self.dims})"
            )
        extra = values.shape[1:-k]
        if self._sum is None:
            self._sum = np.zeros(extra + self.shape)
        elif self._sum.shape != extra + self.shape:
            raise ValueError(
                f"extra dims {extra} differ from earlier adds "
                f"(accumulator shape {self._sum.shape}, domain {self.shape})"
            )
        for b in range(len(arr)):
            sl = []
            for c, d, size in zip(arr[b], self.dims, self.shape):
                p = self.patch_size[d]
                if not 0 <= c <= size - p:
                    raise ValueError(
                        f"corner {int(c)} out of bounds for dim {d!r} "
                        f"(size {size}, patch {p})"
                    )
                sl.append(slice(int(c), int(c) + p))
            sl = tuple(sl)
            self._sum[(Ellipsis,) + sl] += values[b] * self._w
            self._wsum[sl] += self._w

    def merge(self, other: "Reconstructor") -> None:
        """Add another Reconstructor's accumulators (e.g. from another rank)."""
        if other.shape != self.shape or other.patch_size != self.patch_size:
            raise ValueError("cannot merge Reconstructors with different "
                             "domains or patch sizes")
        if other._sum is not None:
            if self._sum is None:
                self._sum = np.zeros(other._sum.shape)
            self._sum += other._sum
        self._wsum += other._wsum

    # -- results -------------------------------------------------------------

    @property
    def sum(self) -> Optional[np.ndarray]:
        """Weighted value accumulator, ``(*extra, *scanned)`` (None before add)."""
        return self._sum

    @property
    def weight_sum(self) -> np.ndarray:
        """Accumulated weights over the scanned dims (0 where uncovered)."""
        return self._wsum

    def result(self, fill_value: float = np.nan) -> np.ndarray:
        """Weighted average, ``fill_value`` where no patch contributed."""
        if self._sum is None:
            raise ValueError("no patches added")
        out = np.full(self._sum.shape, fill_value, dtype=self._sum.dtype)
        np.divide(self._sum, self._wsum, out=out, where=self._wsum > 0)
        return out
