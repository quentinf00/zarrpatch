"""Streaming shuffle buffer.

Block-grouped reading (one storage block -> many patches) is fast but emits
patches in nearly block order, which correlates neighbouring samples within a
batch. Piping the stream through a shuffle buffer of size >> patches-per-block
restores sample-level randomness at a small, bounded memory cost.
"""

from __future__ import annotations

from typing import Iterable, Iterator, TypeVar

import numpy as np

T = TypeVar("T")


def shuffled(items: Iterable[T], buffer_size: int,
             rng: np.random.Generator) -> Iterator[T]:
    """Yield ``items`` in approximately random order using a reservoir buffer.

    Fills a buffer of ``buffer_size`` items, then for each new item yields a
    random buffer slot and replaces it. ``buffer_size <= 1`` is a passthrough.
    """
    if buffer_size <= 1:
        yield from items
        return

    buffer: list = []
    for item in items:
        if len(buffer) < buffer_size:
            buffer.append(item)
            continue
        idx = int(rng.integers(0, buffer_size))
        buffer[idx], item = item, buffer[idx]
        yield item

    rng.shuffle(buffer)  # drain
    yield from buffer
