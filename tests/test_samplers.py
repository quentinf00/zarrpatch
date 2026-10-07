import numpy as np
import pytest

from zarrpatch.samplers import (
    GridSampler,
    ListSampler,
    MaskedSampler,
    RandomSampler,
    WeightedSampler,
    coords_to_indices,
)

DIMS = {"time": 24, "lat": 60, "lon": 80}
PATCH = {"lat": 16, "lon": 16}


def rng():
    return np.random.default_rng(0)


def check_in_bounds(corners, patch=PATCH):
    for c in corners:
        for d, p in patch.items():
            assert 0 <= c[d] <= DIMS[d] - p


def test_random_sampler_count_and_bounds():
    corners = RandomSampler(500).sample(DIMS, PATCH, rng())
    assert len(corners) == 500
    check_in_bounds(corners)
    # not all identical
    assert len({tuple(sorted(c.items())) for c in corners}) > 1


def test_random_sampler_deterministic_per_seed():
    a = RandomSampler(50).sample(DIMS, PATCH, np.random.default_rng(7))
    b = RandomSampler(50).sample(DIMS, PATCH, np.random.default_rng(7))
    c = RandomSampler(50).sample(DIMS, PATCH, np.random.default_rng(8))
    assert a == b
    assert a != c


def test_grid_sampler_tiles_whole_cube():
    corners = GridSampler().sample(DIMS, PATCH, rng())
    check_in_bounds(corners)
    lat_starts = sorted({c["lat"] for c in corners})
    # 60/16 -> starts 0,16,32 plus clamped 44 to reach the boundary
    assert lat_starts == [0, 16, 32, 44]
    covered = np.zeros((DIMS["lat"], DIMS["lon"]), dtype=bool)
    for c in corners:
        covered[c["lat"]:c["lat"] + 16, c["lon"]:c["lon"] + 16] = True
    assert covered.all()


def test_grid_sampler_stride():
    corners = GridSampler(stride={"lat": 8, "lon": 8}).sample(DIMS, PATCH, rng())
    lat_starts = sorted({c["lat"] for c in corners})
    assert lat_starts == [0, 8, 16, 24, 32, 40, 44]
    check_in_bounds(corners)


def test_list_sampler_passthrough_and_validation():
    good = [{"lat": 0, "lon": 0}, {"lat": 44, "lon": 64}]
    assert ListSampler(good).sample(DIMS, PATCH, rng()) == good
    with pytest.raises(ValueError):
        ListSampler([{"lat": 45, "lon": 0}]).sample(DIMS, PATCH, rng())


def test_patch_larger_than_dim_raises():
    with pytest.raises(ValueError):
        RandomSampler(1).sample(DIMS, {"lat": 61}, rng())


def test_coords_to_indices_centered_and_clamped():
    lat = np.linspace(-30, 30, 61)  # 1 degree spacing, lat[30] == 0
    corners = coords_to_indices(
        {"lat": lat}, [{"lat": 0.0}, {"lat": -30.0}], {"lat": 10}
    )
    assert corners[0] == {"lat": 25}  # centered on index 30
    assert corners[1] == {"lat": 0}  # clamped at the lower edge


def test_coords_to_indices_descending():
    lat = np.linspace(30, -30, 61)
    (corner,) = coords_to_indices({"lat": lat}, [{"lat": 30.0}], {"lat": 10})
    assert corner == {"lat": 0}


def test_list_sampler_rejects_missing_or_extra_dims():
    with pytest.raises(ValueError, match="exactly the patch dims"):
        ListSampler([{"lat": 0}]).sample(DIMS, PATCH, rng())
    with pytest.raises(ValueError, match="exactly the patch dims"):
        ListSampler([{"time": 0, "lat": 0, "lon": 0}]).sample(DIMS, PATCH, rng())


def test_masked_sampler_filters_grid_corners():
    mask = np.zeros(DIMS["lat"])
    mask[:32] = 1  # valid lat rows 0..31
    ms = MaskedSampler(GridSampler(), mask, ["lat"])
    corners = ms.sample(DIMS, PATCH, rng())
    assert sorted({c["lat"] for c in corners}) == [0, 16]
    # every lon start survives for the valid lat rows
    all_corners = GridSampler().sample(DIMS, PATCH, rng())
    assert len(corners) == sum(c["lat"] in (0, 16) for c in all_corners)


def test_masked_sampler_invalid_fraction_threshold():
    mask = np.zeros(DIMS["lat"])
    mask[:40] = 1
    ms = MaskedSampler(GridSampler(), mask, ["lat"], max_invalid_frac=0.5)
    # start 32 has 8/16 invalid rows (== 0.5, kept); start 44 is fully invalid
    assert sorted({c["lat"] for c in ms.sample(DIMS, PATCH, rng())}) == [0, 16, 32]


def test_masked_sampler_tops_up_random_draws():
    mask = np.zeros((DIMS["lat"], DIMS["lon"]))
    mask[:40, :40] = 1
    ms = MaskedSampler(RandomSampler(200), mask, ["lat", "lon"])
    corners = ms.sample(DIMS, PATCH, rng())
    assert len(corners) == 200
    for c in corners:
        assert c["lat"] <= 24 and c["lon"] <= 24  # patch stays in the valid box


def test_masked_sampler_raises_when_mask_excludes_everything():
    mask = np.zeros(DIMS["lat"])
    with pytest.raises(ValueError, match="pass the mask"):
        MaskedSampler(RandomSampler(10), mask, ["lat"], max_draws=5).sample(
            DIMS, PATCH, rng())


def test_masked_sampler_validates_shape_and_dims():
    mask = np.ones(DIMS["lat"])
    with pytest.raises(ValueError, match="not a patch dim"):
        MaskedSampler(GridSampler(), mask, ["time"]).sample(DIMS, PATCH, rng())
    with pytest.raises(ValueError, match="data size"):
        MaskedSampler(GridSampler(), np.ones(10), ["lat"]).sample(
            DIMS, PATCH, rng())


def test_weighted_sampler_count_and_bounds():
    weight = 1 + (np.abs(np.linspace(-30, 30, DIMS["lat"])) / 30) ** 2
    corners = WeightedSampler(500, weight, ["lat"]).sample(DIMS, PATCH, rng())
    assert len(corners) == 500
    check_in_bounds(corners)
    assert len({tuple(sorted(c.items())) for c in corners}) > 1


def test_weighted_sampler_biases_toward_high_weight():
    weight = np.ones(DIMS["lat"])
    weight[:30] = 0  # zero weight on the first 30 lat rows
    corners = WeightedSampler(2000, weight, ["lat"]).sample(DIMS, PATCH, rng())
    # no corner has its whole lat footprint inside the zero-weight band
    assert all(c["lat"] + PATCH["lat"] > 30 for c in corners)
    assert np.mean([c["lat"] for c in corners]) > 30


def test_weighted_sampler_patch_integrated():
    # one-hot weight on a single lat index -> every corner whose footprint
    # covers that index is eligible (corner-value weighting would allow only 1)
    weight = np.zeros(DIMS["lat"])
    weight[40] = 1.0
    corners = WeightedSampler(3000, weight, ["lat"]).sample(DIMS, PATCH, rng())
    lat_starts = {c["lat"] for c in corners}
    assert lat_starts == set(range(40 - PATCH["lat"] + 1, 41))
    assert len(lat_starts) == PATCH["lat"]


def test_weighted_sampler_uniform_on_unweighted_dims():
    weight = np.ones(DIMS["lat"])
    corners = WeightedSampler(2000, weight, ["lat"]).sample(DIMS, PATCH, rng())
    lon_starts = sorted({c["lon"] for c in corners})
    # lon is drawn uniformly over its full valid range 0..64
    assert min(lon_starts) <= 2 and max(lon_starts) >= DIMS["lon"] - PATCH["lon"] - 2


def test_weighted_sampler_deterministic_per_seed():
    w = 1 + np.linspace(0, 1, DIMS["lat"])
    a = WeightedSampler(50, w, ["lat"]).sample(DIMS, PATCH, np.random.default_rng(7))
    b = WeightedSampler(50, w, ["lat"]).sample(DIMS, PATCH, np.random.default_rng(7))
    c = WeightedSampler(50, w, ["lat"]).sample(DIMS, PATCH, np.random.default_rng(8))
    assert a == b
    assert a != c


def test_weighted_sampler_composes_with_masked():
    weight = 1 + np.linspace(0, 1, DIMS["lat"])
    mask = np.zeros((DIMS["lat"], DIMS["lon"]))
    mask[:40, :40] = 1
    ms = MaskedSampler(WeightedSampler(200, weight, ["lat"]), mask, ["lat", "lon"])
    corners = ms.sample(DIMS, PATCH, rng())
    assert len(corners) == 200
    for c in corners:
        assert c["lat"] <= 24 and c["lon"] <= 24


def test_weighted_sampler_errors():
    with pytest.raises(ValueError):  # all-zero weight
        WeightedSampler(10, np.zeros(DIMS["lat"]), ["lat"]).sample(DIMS, PATCH, rng())
    with pytest.raises(ValueError):  # negative weight
        w = np.ones(DIMS["lat"]); w[0] = -1
        WeightedSampler(10, w, ["lat"]).sample(DIMS, PATCH, rng())
    with pytest.raises(ValueError):  # non-finite weight
        w = np.ones(DIMS["lat"]); w[0] = np.inf
        WeightedSampler(10, w, ["lat"])
    with pytest.raises(ValueError):  # wrong ndim
        WeightedSampler(10, np.ones((DIMS["lat"], 3)), ["lat"])
    with pytest.raises(ValueError, match="not a patch dim"):
        WeightedSampler(10, np.ones(DIMS["time"]), ["time"]).sample(
            DIMS, PATCH, rng())
    with pytest.raises(ValueError, match="data size"):
        WeightedSampler(10, np.ones(10), ["lat"]).sample(DIMS, PATCH, rng())
