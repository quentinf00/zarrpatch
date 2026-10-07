import numpy as np

from zarrpatch.bench import benchmark, print_results, results_to_markdown


def make_stream():
    return ({"x": np.zeros((4, 16, 16), dtype=np.float32)} for _ in range(20))


def test_benchmark_counts_items_and_bytes():
    res = benchmark(make_stream, name="fake")
    assert res.n_items == 20
    assert res.bytes_total == 20 * 4 * 16 * 16 * 4
    assert res.wall_s > 0
    assert res.items_per_s > 0
    assert res.first_item_s <= res.wall_s


def test_benchmark_max_items():
    res = benchmark(make_stream, max_items=5)
    assert res.n_items == 5


def test_benchmark_repeats_keeps_best():
    res = benchmark(make_stream, repeats=3)
    assert res.n_items == 20


def test_markdown_and_print(capsys):
    results = [benchmark(make_stream, name="a"), benchmark(make_stream, name="b")]
    md = results_to_markdown(results, baseline="a")
    assert md.count("\n") == 3
    assert "| a |" in md and "1.00x" in md
    print_results(results, baseline="a")
    assert "a" in capsys.readouterr().out
