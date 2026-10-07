"""Small utilities to benchmark patch dataloading throughput."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional

import numpy as np

MiB = 1024 * 1024


def _item_bytes(item) -> int:
    if isinstance(item, dict):
        return sum(_item_bytes(v) for v in item.values())
    if isinstance(item, (list, tuple)):
        return sum(_item_bytes(v) for v in item)
    if hasattr(item, "nbytes"):  # numpy array / torch tensor
        return int(item.nbytes if isinstance(item, np.ndarray)
                   else item.element_size() * item.nelement())
    return 0


@dataclass
class BenchResult:
    name: str
    n_items: int
    wall_s: float
    first_item_s: float
    bytes_total: int

    @property
    def items_per_s(self) -> float:
        return self.n_items / self.wall_s if self.wall_s else float("inf")

    @property
    def mib_per_s(self) -> float:
        return self.bytes_total / MiB / self.wall_s if self.wall_s else float("inf")


def benchmark(
    make_iter: Callable[[], Iterable],
    name: str = "loader",
    max_items: Optional[int] = None,
    repeats: int = 1,
) -> BenchResult:
    """Time the consumption of an iterable of samples/batches.

    ``make_iter`` is called once per repeat and must return a fresh iterator
    (e.g. ``lambda: DataLoader(ds, ...)``). With ``repeats > 1`` the reported
    wall time is the best (minimum) run, which suppresses filesystem-cache and
    scheduler noise.
    """
    best: Optional[BenchResult] = None
    for _ in range(repeats):
        n, nbytes = 0, 0
        t0 = time.perf_counter()
        first = None
        for item in make_iter():
            if first is None:
                first = time.perf_counter() - t0
            n += 1
            nbytes += _item_bytes(item)
            if max_items is not None and n >= max_items:
                break
        wall = time.perf_counter() - t0
        res = BenchResult(name, n, wall, first or 0.0, nbytes)
        if best is None or res.wall_s < best.wall_s:
            best = res
    assert best is not None
    return best


def print_results(results: List[BenchResult], title: str = "dataloading benchmark",
                  baseline: Optional[str] = None) -> None:
    """Pretty-print results with rich; speedup column is vs ``baseline`` name
    (default: the first result)."""
    from rich.console import Console
    from rich.table import Table

    base = next((r for r in results if r.name == baseline), results[0])
    table = Table(title=title)
    for col in ("loader", "items", "wall [s]", "first item [s]",
                "items/s", "MiB/s", "speedup"):
        table.add_column(col, justify="right")
    table.columns[0].justify = "left"
    for r in results:
        table.add_row(
            r.name, str(r.n_items), f"{r.wall_s:.2f}", f"{r.first_item_s:.3f}",
            f"{r.items_per_s:.1f}", f"{r.mib_per_s:.1f}",
            f"{r.items_per_s / base.items_per_s:.2f}x",
        )
    Console().print(table)


def results_to_markdown(results: List[BenchResult],
                        baseline: Optional[str] = None) -> str:
    base = next((r for r in results if r.name == baseline), results[0])
    lines = [
        "| loader | items | wall [s] | first item [s] | items/s | MiB/s | speedup |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        lines.append(
            f"| {r.name} | {r.n_items} | {r.wall_s:.2f} | {r.first_item_s:.3f} "
            f"| {r.items_per_s:.1f} | {r.mib_per_s:.1f} "
            f"| {r.items_per_s / base.items_per_s:.2f}x |"
        )
    return "\n".join(lines)
