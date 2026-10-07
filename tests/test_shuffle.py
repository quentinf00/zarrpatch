import numpy as np

from zarrpatch.shuffle import shuffled


def test_yields_all_items_exactly_once():
    items = list(range(1000))
    out = list(shuffled(items, 64, np.random.default_rng(0)))
    assert sorted(out) == items


def test_actually_shuffles():
    items = list(range(1000))
    out = list(shuffled(items, 256, np.random.default_rng(0)))
    assert out != items


def test_deterministic_given_seed():
    items = list(range(200))
    a = list(shuffled(items, 32, np.random.default_rng(5)))
    b = list(shuffled(items, 32, np.random.default_rng(5)))
    c = list(shuffled(items, 32, np.random.default_rng(6)))
    assert a == b
    assert a != c


def test_buffer_leq_one_is_passthrough():
    items = list(range(50))
    assert list(shuffled(items, 1, np.random.default_rng(0))) == items
    assert list(shuffled(items, 0, np.random.default_rng(0))) == items


def test_buffer_larger_than_stream():
    items = list(range(10))
    out = list(shuffled(items, 1000, np.random.default_rng(0)))
    assert sorted(out) == items


def test_no_item_emitted_before_it_could_have_arrived():
    # the item emitted at output position pos was drawn from a buffer that has
    # only seen the first pos + buffer_size input items
    n, size = 2000, 100
    out = list(shuffled(range(n), size, np.random.default_rng(1)))
    for pos, item in enumerate(out):
        assert item <= pos + size
