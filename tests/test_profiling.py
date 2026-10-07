import json

from torch.utils.data import DataLoader

from zarrpatch import (
    JsonlProfiler,
    ListSampler,
    ZarrPatchDataset,
    format_summary,
    read_trace,
    summarize_trace,
)


PATCH = {"time": 4, "lat": 16, "lon": 16}


def _dataset(path, trace, *, detailed=False):
    return ZarrPatchDataset(
        path,
        PATCH,
        variables=["sst"],
        sampler=ListSampler([
            {"time": 0, "lat": 0, "lon": 0},
            {"time": 8, "lat": 20, "lon": 20},
        ]),
        shuffle=False,
        prefetch=0,
        profiler=JsonlProfiler(trace, detailed=detailed),
    )


def test_profiler_lifecycle_and_summary(zarr3_plain, tmp_path):
    trace = tmp_path / "trace"
    assert len(list(_dataset(zarr3_plain, trace))) == 2

    records = read_trace(trace)
    names = [record["event"] for record in records]
    assert names.count("store_open_start") == 1
    assert names.count("store_open_end") == 1
    assert "dataset_iter_start" in names
    assert "dataset_plan_end" in names
    assert "first_patch" in names
    assert "dataset_iter_end" in names
    assert "variable_read_start" not in names

    summary = summarize_trace(trace)
    assert summary["processes"] == 1
    assert summary["stores"] == 1
    assert summary["counts"]["first_patch"] == 1
    assert "store_open" in summary["elapsed_seconds"]
    assert "unique stores: 1" in format_summary(summary)


def test_detailed_profiler_records_reads(zarr3_plain, tmp_path):
    trace = tmp_path / "trace"
    assert len(list(_dataset(zarr3_plain, trace, detailed=True))) == 2
    records = read_trace(trace)
    reads = [r for r in records if r["event"] == "variable_read_end"]
    assert reads
    assert all(r["bytes"] > 0 and r["elapsed_ns"] >= 0 for r in reads)
    assert any(r["event"] == "group_read_end" for r in records)


def test_profiler_is_dataloader_process_safe(zarr3_plain, tmp_path):
    trace = tmp_path / "trace"
    loader = DataLoader(
        _dataset(zarr3_plain, trace), batch_size=None, num_workers=2
    )
    assert len(list(loader)) == 2
    summary = summarize_trace(trace)
    assert summary["processes"] >= 2
    assert summary["counts"]["dataset_iter_start"] == 2


def test_trace_is_jsonl(zarr3_plain, tmp_path):
    trace = tmp_path / "trace"
    list(_dataset(zarr3_plain, trace))
    for path in trace.glob("*.jsonl"):
        for line in path.read_text().splitlines():
            assert "event" in json.loads(line)
