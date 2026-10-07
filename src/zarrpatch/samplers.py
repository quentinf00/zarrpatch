"""Patch corner samplers.

A *corner* is a dict mapping dimension name -> start index; together with the
dataset-level ``patch_size`` (dim -> length) it defines a patch. Samplers only
produce corners in index space; ``coords_to_indices`` converts physical
coordinate values (e.g. lat/lon) to indices.

All samplers guarantee ``corner[d] + patch_size[d] <= dim_size[d]``.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

Corner = Dict[str, int]


class Sampler:
    def sample(self, dim_sizes: Dict[str, int], patch_size: Dict[str, int],
               rng: np.random.Generator) -> List[Corner]:
        raise NotImplementedError


def _check_fits(dim_sizes, patch_size):
    for d, p in patch_size.items():
        if d not in dim_sizes:
            raise ValueError(f"patch dim {d!r} not in data dims {sorted(dim_sizes)}")
        if p > dim_sizes[d]:
            raise ValueError(f"patch size {p} exceeds dim {d!r} of size {dim_sizes[d]}")


def _build_sat(field: np.ndarray) -> np.ndarray:
    """Zero-padded summed-area table of ``field``.

    ``_sat[i1..ik] = field[:i1, .., :ik].sum()`` — a cumulative sum along every
    axis, then a leading zero padded on each so a box sum reads clean starts and
    stops. Shared by :class:`MaskedSampler` and :class:`WeightedSampler`.
    """
    table = np.asarray(field).cumsum(axis=0)
    for axis in range(1, table.ndim):
        table = table.cumsum(axis=axis)
    return np.pad(table, [(1, 0)] * table.ndim)


def _box_mean(sat: np.ndarray, starts: List[np.ndarray],
              lens: List[int]) -> np.ndarray:
    """Mean of the summed field over each corner's patch footprint.

    ``sat`` is a zero-padded summed-area table (see :func:`_build_sat`);
    ``starts`` gives the per-axis corner starts (one array per axis, all the
    same length) and ``lens`` the patch length along each axis. Returns the
    per-corner box mean, O(1) per corner via inclusion–exclusion.
    """
    n = len(starts[0]) if starts else 1
    total = np.zeros(n, dtype=sat.dtype)
    for bits in np.ndindex(*(2,) * len(starts)):
        idx = tuple(s + b * ln for s, b, ln in zip(starts, bits, lens))
        total = total + (-1) ** (len(starts) - sum(bits)) * sat[idx]
    return total / np.prod(lens)


class RandomSampler(Sampler):
    """``num_patches`` uniformly random corners, redrawn each epoch."""

    def __init__(self, num_patches: int):
        self.num_patches = num_patches

    def sample(self, dim_sizes, patch_size, rng):
        _check_fits(dim_sizes, patch_size)
        dims = list(patch_size)
        highs = [dim_sizes[d] - patch_size[d] + 1 for d in dims]
        draws = [rng.integers(0, h, size=self.num_patches) for h in highs]
        return [dict(zip(dims, (int(v[i]) for v in draws)))
                for i in range(self.num_patches)]


class GridSampler(Sampler):
    """All patches on a regular grid with the given stride per dim.

    ``stride=None`` means non-overlapping tiling (stride = patch size). The last
    patch along each dim is clamped to the boundary so the whole cube is covered.
    """

    def __init__(self, stride: Optional[Dict[str, int]] = None):
        self.stride = stride

    def sample(self, dim_sizes, patch_size, rng):
        _check_fits(dim_sizes, patch_size)
        dims = list(patch_size)
        axes = []
        for d in dims:
            step = (self.stride or {}).get(d, patch_size[d])
            if step < 1:
                raise ValueError(f"stride for {d!r} must be >= 1, got {step}")
            last = dim_sizes[d] - patch_size[d]
            starts = list(range(0, last + 1, step))
            if starts[-1] != last:  # clamp a final patch to the edge
                starts.append(last)
            axes.append(starts)
        mesh = np.meshgrid(*axes, indexing="ij")
        return [dict(zip(dims, (int(m.flat[i]) for m in mesh)))
                for i in range(mesh[0].size)]


class ListSampler(Sampler):
    """Patches at an explicit list of corners (index space).

    Every corner must specify exactly the patch dims.
    """

    def __init__(self, corners: Sequence[Corner]):
        self.corners = [dict(c) for c in corners]

    def sample(self, dim_sizes, patch_size, rng):
        _check_fits(dim_sizes, patch_size)
        for c in self.corners:
            if set(c) != set(patch_size):
                raise ValueError(
                    f"corner {c} must specify exactly the patch dims "
                    f"{sorted(patch_size)} (got {sorted(c)})"
                )
            for d, start in c.items():
                if not 0 <= start <= dim_sizes[d] - patch_size[d]:
                    raise ValueError(f"corner {c} out of bounds for dim {d!r}")
        return list(self.corners)


class MaskedSampler(Sampler):
    """Filter another sampler's corners through a validity mask.

    ``mask`` is an array over ``dims`` (a subset of the patch dims) where
    nonzero/True marks valid cells — e.g. an ocean mask, or
    ``np.isfinite(field).mean("time") > 0.9``. A corner is kept when the
    invalid fraction of its patch footprint (over ``dims``) is at most
    ``max_invalid_frac``. Footprint sums are O(1) per corner via a
    summed-area table.

    When the base sampler has a fixed ``num_patches`` (e.g.
    :class:`RandomSampler`), rejected corners are topped up by redrawing, up
    to ``max_draws`` rounds; grid/list corners are simply filtered.
    """

    def __init__(self, base: Sampler, mask: np.ndarray, dims: Sequence[str],
                 max_invalid_frac: float = 0.0, max_draws: int = 100):
        self.base = base
        self.dims = list(dims)
        self.max_invalid_frac = float(max_invalid_frac)
        self.max_draws = max_draws
        invalid = ~(np.asarray(mask) != 0)
        if invalid.ndim != len(self.dims):
            raise ValueError(
                f"mask has {invalid.ndim} dims but {len(self.dims)} dim "
                f"names given: {self.dims}"
            )
        self.mask_shape = invalid.shape
        self._sat = _build_sat(invalid.astype(np.int64))

    def _invalid_frac(self, corners: List[Corner],
                      patch_size: Dict[str, int]) -> np.ndarray:
        starts = [np.array([c[d] for c in corners]) for d in self.dims]
        lens = [patch_size[d] for d in self.dims]
        return _box_mean(self._sat, starts, lens)

    def sample(self, dim_sizes, patch_size, rng):
        for d in self.dims:
            if d not in patch_size:
                raise ValueError(f"mask dim {d!r} is not a patch dim")
            if dim_sizes[d] != self.mask_shape[self.dims.index(d)]:
                raise ValueError(
                    f"mask size {self.mask_shape[self.dims.index(d)]} != "
                    f"data size {dim_sizes[d]} for dim {d!r}"
                )

        def keep(corners: List[Corner]) -> List[Corner]:
            if not corners:
                return []
            frac = self._invalid_frac(corners, patch_size)
            return [c for c, f in zip(corners, frac)
                    if f <= self.max_invalid_frac]

        kept = keep(self.base.sample(dim_sizes, patch_size, rng))
        target = getattr(self.base, "num_patches", None)
        draws = 1
        while target is not None and len(kept) < target:
            if draws >= self.max_draws:
                raise ValueError(
                    f"only {len(kept)}/{target} corners pass the mask after "
                    f"{draws} draws; loosen max_invalid_frac or the mask"
                )
            kept += keep(self.base.sample(dim_sizes, patch_size, rng))
            draws += 1
        return kept[:target] if target is not None else kept


class WeightedSampler(Sampler):
    """Draw corners with replacement, weighted by a per-cell weight array.

    ``weight`` is an array over ``dims`` (a subset of the patch dims, exactly
    like :class:`MaskedSampler`'s mask). Each valid corner is scored by the
    **mean of ``weight`` over its patch footprint** on ``dims`` — patch
    integrated via a summed-area table, so a patch spanning high-weight rows
    inherits their weight from all of them, not just its top-left corner (and
    it degenerates to corner-value weighting when the patch is size 1 on a
    dim). ``num_patches`` corners are drawn with replacement, corner
    probability proportional to that footprint mean; patch dims not in ``dims``
    are drawn uniformly, like :class:`RandomSampler`.

    Typical uses: latitude oversampling (``weight = 1 + (f-1)*(|lat|/90)**2``
    over a 1-D ``lat``) or per-store cluster resampling over a virtual
    ``store`` dim.

    Notes
    -----
    Corners are enumerated jointly over the weight ``dims`` only, so the memory
    cost is ``prod(dim_sizes[d] - patch_size[d] + 1 for d in dims)`` float64
    weights plus one table of ``weight.shape``. Keep the weighted ``dims`` to
    1–2; do not weight all patch dims.

    Because corners are drawn with replacement, the dataset's "every patch
    exactly once per epoch" guarantee becomes "every drawn corner exactly
    once"; duplicate corners are allowed and cheap — they fall in the same
    block group and are sliced/yielded once each.
    """

    def __init__(self, num_patches: int, weight: np.ndarray,
                 dims: Sequence[str]):
        self.num_patches = num_patches
        self.dims = list(dims)
        weight = np.asarray(weight, dtype=np.float64)
        if weight.ndim != len(self.dims):
            raise ValueError(
                f"weight has {weight.ndim} dims but {len(self.dims)} dim "
                f"names given: {self.dims}"
            )
        if not np.all(np.isfinite(weight)):
            raise ValueError("weight must be finite (no NaN/inf)")
        if np.any(weight < 0):
            raise ValueError("weight must be nonnegative")
        self.weight = weight
        self.weight_shape = weight.shape
        self._sat = _build_sat(weight)

    def sample(self, dim_sizes, patch_size, rng):
        _check_fits(dim_sizes, patch_size)
        for i, d in enumerate(self.dims):
            if d not in patch_size:
                raise ValueError(f"weight dim {d!r} is not a patch dim")
            if dim_sizes[d] != self.weight_shape[i]:
                raise ValueError(
                    f"weight size {self.weight_shape[i]} != "
                    f"data size {dim_sizes[d]} for dim {d!r}"
                )

        # joint grid of valid corners over the weight dims only
        n_valid = [dim_sizes[d] - patch_size[d] + 1 for d in self.dims]
        grids = np.meshgrid(*[np.arange(n) for n in n_valid], indexing="ij")
        starts = [g.ravel() for g in grids]
        lens = [patch_size[d] for d in self.dims]
        probs = _box_mean(self._sat, starts, lens)
        total = probs.sum()
        if not total > 0:
            raise ValueError(
                "total weight over valid corners is 0; all footprints are "
                "empty or the weight is all-zero"
            )
        probs = probs / total

        chosen = rng.choice(len(probs), size=self.num_patches, replace=True,
                            p=probs)
        weighted_starts = [s[chosen] for s in starts]

        # non-weighted patch dims drawn uniformly, like RandomSampler
        other_dims = [d for d in patch_size if d not in self.dims]
        other_highs = [dim_sizes[d] - patch_size[d] + 1 for d in other_dims]
        other_draws = [rng.integers(0, h, size=self.num_patches)
                       for h in other_highs]

        corners: List[Corner] = []
        for i in range(self.num_patches):
            corner: Corner = {d: int(ws[i])
                              for d, ws in zip(self.dims, weighted_starts)}
            for d, draw in zip(other_dims, other_draws):
                corner[d] = int(draw[i])
            corners.append(corner)
        return corners


def coords_to_indices(coord_values: Dict[str, np.ndarray],
                      points: Iterable[Dict[str, float]],
                      patch_size: Dict[str, int],
                      center: bool = True) -> List[Corner]:
    """Convert physical coordinate points to index-space corners.

    ``coord_values`` maps dim name -> 1D array of coordinate values (must be
    monotonically increasing or decreasing). Each point is snapped to the
    nearest index; with ``center=True`` the patch is centered on it (clamped to
    stay in bounds), otherwise the point is the corner.
    """
    corners = []
    for pt in points:
        corner: Corner = {}
        for d, value in pt.items():
            coords = np.asarray(coord_values[d])
            idx = int(np.abs(coords - value).argmin())
            if center:
                idx -= patch_size.get(d, 1) // 2
            idx = max(0, min(idx, len(coords) - patch_size.get(d, 1)))
            corner[d] = idx
        corners.append(corner)
    return corners
