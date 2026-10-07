"""Opt-in, process-safe tracing for zarrpatch readers and datasets.

The profiler deliberately writes one file per process.  DataLoader workers do
not need a shared queue or lock, and traces remain useful if a worker is killed.
"""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, Optional


class JsonlProfiler:
    """Write timestamped zarrpatch events to per-process JSONL files.

    Parameters
    ----------
    directory:
        Trace directory. Files are named ``trace-<pid>.jsonl``.
    detailed:
        Include high-frequency ``detail=True`` events such as individual
        variable and block-group reads. Lifecycle, store-open, cache, and
        first-output events are always recorded.

    The object is safe to pickle into DataLoader workers. Each event is emitted
    with a single append write, and each process has its own output file.
    """

    def __init__(self, directory: os.PathLike[str] | str, detailed: bool = False):
        self.directory = str(directory)
        self.detailed = bool(detailed)
        Path(self.directory).mkdir(parents=True, exist_ok=True)

    def event(self, name: str, *, detail: bool = False, **fields: Any) -> None:
        if detail and not self.detailed:
            return
        record = {
            "event": name,
            "monotonic_ns": time.monotonic_ns(),
            "time_ns": time.time_ns(),
            "pid": os.getpid(),
            **fields,
        }
        path = Path(self.directory) / f"trace-{os.getpid()}.jsonl"
        line = json.dumps(record, separators=(",", ":"), default=str) + "\n"
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            os.write(fd, line.encode())
        finally:
            os.close(fd)

    @contextmanager
    def span(self, name: str, *, detail: bool = False,
             **fields: Any) -> Iterator[None]:
        start = time.monotonic_ns()
        self.event(f"{name}_start", detail=detail, **fields)
        try:
            yield
        except BaseException as exc:
            self.event(
                f"{name}_error", detail=detail,
                elapsed_ns=time.monotonic_ns() - start,
                error_type=type(exc).__name__, **fields,
            )
            raise
        else:
            self.event(
                f"{name}_end", detail=detail,
                elapsed_ns=time.monotonic_ns() - start, **fields,
            )

    def __getstate__(self) -> Dict[str, Any]:
        return {"directory": self.directory, "detailed": self.detailed}


def emit(profiler: Optional[JsonlProfiler], name: str, *,
         detail: bool = False, **fields: Any) -> None:
    """Emit an event when profiling is enabled."""
    if profiler is not None:
        profiler.event(name, detail=detail, **fields)


def read_trace(directory: os.PathLike[str] | str) -> list[dict]:
    """Read and time-sort all profiler records in *directory*."""
    records = []
    for path in sorted(Path(directory).glob("trace-*.jsonl")):
        with path.open() as f:
            records.extend(json.loads(line) for line in f if line.strip())
    return sorted(records, key=lambda r: (r["time_ns"], r["pid"]))


def summarize_trace(directory: os.PathLike[str] | str) -> dict:
    """Return compact counts and elapsed-time totals for a trace directory."""
    records = read_trace(directory)
    counts: Dict[str, int] = {}
    elapsed_ns: Dict[str, int] = {}
    pids = set()
    stores = set()
    for record in records:
        name = record["event"]
        counts[name] = counts.get(name, 0) + 1
        pids.add(record["pid"])
        if name.endswith("_end") or name.endswith("_error"):
            base = name.rsplit("_", 1)[0]
            elapsed_ns[base] = elapsed_ns.get(base, 0) + record.get("elapsed_ns", 0)
        if "path" in record:
            stores.add(record["path"])
    return {
        "processes": len(pids),
        "stores": len(stores),
        "counts": counts,
        "elapsed_seconds": {k: v / 1e9 for k, v in elapsed_ns.items()},
        "wall_seconds": (
            (records[-1]["time_ns"] - records[0]["time_ns"]) / 1e9
            if len(records) > 1 else 0.0
        ),
    }


def format_summary(summary: dict) -> str:
    """Format :func:`summarize_trace` for a terminal report."""
    lines = [
        f"processes: {summary['processes']}",
        f"unique stores: {summary['stores']}",
        f"trace wall time: {summary['wall_seconds']:.3f} s",
        "",
        "elapsed totals:",
    ]
    lines.extend(
        f"  {name}: {seconds:.3f} s"
        for name, seconds in sorted(
            summary["elapsed_seconds"].items(), key=lambda item: -item[1]
        )
    )
    lines.append("")
    lines.append("event counts:")
    lines.extend(
        f"  {name}: {count}"
        for name, count in sorted(summary["counts"].items())
    )
    return "\n".join(lines)


def main() -> None:
    """CLI entry point: ``python -m zarrpatch.profiling TRACE_DIR``."""
    import argparse

    parser = argparse.ArgumentParser(description="Summarize zarrpatch traces")
    parser.add_argument("trace_directory")
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()
    summary = summarize_trace(args.trace_directory)
    print(json.dumps(summary, indent=2) if args.as_json else format_summary(summary))


if __name__ == "__main__":  # pragma: no cover
    main()
