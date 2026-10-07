import numpy as np
import pytest
from torch.utils.data import DataLoader

from zarrpatch import GridSampler, Reconstructor, ZarrPatchDataset

PATCH = {"time": 4, "lat": 16, "lon": 16}


def test_uniform_overlap_blend_and_fill():
    rec = Reconstructor({"x": 8}, {"x": 4}, weight="uniform", dims=["x"])
    rec.add({"x": 0}, np.zeros(4))
    rec.add({"x": 2}, np.ones(4))
    out = rec.result()
    np.testing.assert_array_equal(out[:2], [0.0, 0.0])
    np.testing.assert_array_equal(out[2:4], [0.5, 0.5])  # overlap, equal weights
    np.testing.assert_array_equal(out[4:6], [1.0, 1.0])
    assert np.isnan(out[6:]).all()  # never covered


def test_hann_weights_favor_patch_centers():
    rec = Reconstructor({"x": 8}, {"x": 4}, weight="hann", dims=["x"])
    rec.add({"x": 0}, np.zeros(4))
    rec.add({"x": 2}, np.ones(4))
    out = rec.result()
    # x=2 is near the first patch's center but the second's edge, and vice
    # versa at x=3: the blend leans toward whichever patch is centered there
    assert out[2] < 0.5 < out[3]


def test_batched_add_with_extra_dims_is_identity():
    rng = np.random.default_rng(0)
    truth = rng.normal(size=(3, 10, 12))  # (channel, y, x)
    rec = Reconstructor({"y": 10, "x": 12}, {"y": 4, "x": 6}, dims=["y", "x"])
    corners = [{"y": y, "x": x} for y in (0, 3, 6) for x in (0, 4, 6)]
    values = np.stack(
        [truth[:, c["y"]:c["y"] + 4, c["x"]:c["x"] + 6] for c in corners]
    )
    rec.add(corners, values)
    out = rec.result()
    assert out.shape == (3, 10, 12)
    np.testing.assert_allclose(out, truth, atol=1e-12)


def test_array_corners_use_sorted_dim_order():
    # output/value dims order (time, lat), so array corner columns — which
    # follow the dataset's sorted(patch_size) convention — are (lat, time)
    truth = np.arange(30, dtype=float).reshape(6, 5)  # (time, lat)
    rec = Reconstructor({"time": 6, "lat": 5}, {"time": 3, "lat": 5},
                        weight="uniform", dims=["time", "lat"])
    rec.add(np.array([[0, 0], [0, 3]]), np.stack([truth[0:3], truth[3:6]]))
    np.testing.assert_allclose(rec.result(), truth)


def test_merge_matches_single_reconstructor():
    rng = np.random.default_rng(1)
    truth = rng.normal(size=(9,))
    corners = [{"x": i} for i in (0, 2, 4, 5)]
    patches = [truth[c["x"]:c["x"] + 4] for c in corners]
    full = Reconstructor({"x": 9}, {"x": 4}, dims=["x"])
    a = Reconstructor({"x": 9}, {"x": 4}, dims=["x"])
    b = Reconstructor({"x": 9}, {"x": 4}, dims=["x"])
    for c, p in zip(corners, patches):
        full.add(c, p)
    for c, p in zip(corners[:2], patches[:2]):
        a.add(c, p)
    for c, p in zip(corners[2:], patches[2:]):
        b.add(c, p)
    a.merge(b)
    np.testing.assert_allclose(a.result(), full.result())


def test_validation_errors():
    with pytest.raises(ValueError, match="unknown weight"):
        Reconstructor({"x": 8}, {"x": 4}, weight="boxcar")
    with pytest.raises(ValueError, match="no patches added"):
        Reconstructor({"x": 8}, {"x": 4}).result()
    with pytest.raises(ValueError, match="permutation"):
        Reconstructor({"x": 8, "y": 8}, {"x": 4, "y": 4}, dims=["x"])
    rec = Reconstructor({"x": 8}, {"x": 4})
    with pytest.raises(ValueError, match="does not match"):
        rec.add({"x": 0}, np.zeros(5))
    with pytest.raises(ValueError, match="out of bounds"):
        rec.add({"x": 6}, np.zeros(4))
    rec2 = Reconstructor({"x": 8}, {"x": 4})
    rec2.add({"x": 0}, np.zeros((2, 4)))
    with pytest.raises(ValueError, match="extra dims"):
        rec2.add({"x": 0}, np.zeros((3, 4)))


def test_custom_weight_array():
    w = np.array([1.0, 2.0, 2.0, 1.0])
    rec = Reconstructor({"x": 6}, {"x": 4}, weight=w, dims=["x"])
    rec.add({"x": 0}, np.zeros(4))
    rec.add({"x": 2}, np.ones(4))
    out = rec.result()
    assert out[2] == pytest.approx(1.0 / (2.0 + 1.0))  # w[2]=2 vs w[0]=1
    with pytest.raises(ValueError, match="weight shape"):
        Reconstructor({"x": 6}, {"x": 4}, weight=np.ones(3), dims=["x"])


def test_overlapping_grid_scan_reconstructs_cube(zarr3_sharded, dataset):
    ds = ZarrPatchDataset(
        zarr3_sharded, PATCH, variables=["sst"],
        sampler=GridSampler(stride={"time": 2, "lat": 8, "lon": 8}),
        shuffle=False, return_corners=True,
    )
    rec = Reconstructor.from_dataset(ds)
    for p in ds:
        rec.add(p["corner"], p["sst"])
    assert not np.isnan(rec.result()).any()
    np.testing.assert_allclose(rec.result(), dataset["sst"].values, atol=1e-5)


def test_reconstruct_from_dataloader_batches(zarr3_sharded, dataset):
    ds = ZarrPatchDataset(
        zarr3_sharded, PATCH, variables=["sst"], sampler=GridSampler(),
        shuffle=False, return_corners=True,
    )
    rec = Reconstructor.from_dataset(ds, weight="triangle")
    for batch in DataLoader(ds, batch_size=8, num_workers=2):
        rec.add(batch["corner"], batch["sst"])  # torch tensors
    np.testing.assert_allclose(rec.result(), dataset["sst"].values, atol=1e-5)


def test_reconstruct_val_split(zarr3_sharded, dataset):
    ds = ZarrPatchDataset(
        zarr3_sharded, PATCH, variables=["sst"], sampler=GridSampler(),
        shuffle=False, return_corners=True, dim_slices={"time": (16, 24)},
    )
    rec = Reconstructor.from_dataset(ds)
    for p in ds:
        rec.add(p["corner"], p["sst"])
    np.testing.assert_allclose(rec.result(), dataset["sst"].values[16:24],
                               atol=1e-5)
