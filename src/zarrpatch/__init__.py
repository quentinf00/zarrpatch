"""zarrpatch — fast random-patch dataloading from zarr datacubes.

See the README for usage and chunk/shard layout guidelines.

The PyTorch dataset adapters are imported lazily, so the NumPy-based readers,
samplers, reconstruction, and layout helpers remain usable without PyTorch.
"""

from .bench import BenchResult, benchmark, print_results, results_to_markdown
from .guidelines import LayoutRecommendation, recommend_layout
from .profiling import JsonlProfiler, format_summary, read_trace, summarize_trace
from .readers import (
    CubeReader,
    KerchunkReader,
    MultiCubeReader,
    NetCDFReader,
    TensorStoreReader,
    ZarrReader,
    expand_units,
    open_cube,
)
from .reconstruct import Reconstructor
from .samplers import (
    GridSampler,
    ListSampler,
    MaskedSampler,
    RandomSampler,
    Sampler,
    WeightedSampler,
    coords_to_indices,
)
from .shuffle import shuffled

try:
    from .dataset import ZarrPatchDataset, group_corners_by_block, worker_init_fn
    from .materialize import (
        HDF5PatchDataset,
        MaterializedValidation,
        materialize,
        validate_materialized,
    )
    from .planning import ExecutionPlan, analyze_layout
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None

__version__ = "0.1.0"

_TORCH_EXPORTS = {
    "ZarrPatchDataset",
    "group_corners_by_block",
    "worker_init_fn",
    "HDF5PatchDataset",
    "materialize",
    "MaterializedValidation",
    "validate_materialized",
    "ExecutionPlan",
    "analyze_layout",
}


def __getattr__(name: str):
    """Explain how to enable unavailable PyTorch integrations."""
    if name not in _TORCH_EXPORTS or _TORCH_IMPORT_ERROR is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    raise ImportError(
        f"{name} requires the optional PyTorch integration; install "
        "`zarrpatch[torch]`, or select `pytorch-cpu`/`pytorch-gpu` in "
        "the consuming Pixi environment."
    ) from _TORCH_IMPORT_ERROR


__all__ = [
    "BenchResult",
    "CubeReader",
    "GridSampler",
    "HDF5PatchDataset",
    "ExecutionPlan",
    "LayoutRecommendation",
    "MaterializedValidation",
    "ListSampler",
    "MaskedSampler",
    "KerchunkReader",
    "JsonlProfiler",
    "MultiCubeReader",
    "NetCDFReader",
    "RandomSampler",
    "Reconstructor",
    "Sampler",
    "TensorStoreReader",
    "WeightedSampler",
    "ZarrPatchDataset",
    "ZarrReader",
    "benchmark",
    "analyze_layout",
    "coords_to_indices",
    "expand_units",
    "format_summary",
    "group_corners_by_block",
    "materialize",
    "open_cube",
    "print_results",
    "recommend_layout",
    "read_trace",
    "results_to_markdown",
    "shuffled",
    "summarize_trace",
    "validate_materialized",
    "worker_init_fn",
]
