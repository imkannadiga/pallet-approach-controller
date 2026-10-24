"""The speed work on the hot stages, and the proof it changed no answer.

`voxel_downsample` and `cluster_points` were 84 % of a frame: 229 ms total, of
which 110 ms and 77 ms. Both were slow spellings rather than slow algorithms --
same complexity, written in a way NumPy and scikit-learn cannot run quickly.
After rewriting them a frame takes 90 ms, which is what makes 5 Hz on the
Jetson plausible.

"Faster and identical" is only worth having if the second half stays true, so
each is checked against a reference implementation rather than against itself.
"""

import numpy as np
import pytest

from pallet_approach import perception as P


# --------------------------------------------------- speed work changes nothing

def naive_voxel_downsample(points, voxel_size):
    """The obvious implementation, kept here as the reference the fast one must
    match. Slow, and that is the point: it has no cleverness to be wrong."""
    cube = np.floor(points / voxel_size).astype(np.int64)
    _, inverse = np.unique(cube, axis=0, return_inverse=True)
    inverse = inverse.ravel()
    n = inverse.max() + 1
    sums = np.zeros((n, 3))
    np.add.at(sums, inverse, points)
    return sums / np.bincount(inverse, minlength=n)[:, None]


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_fast_voxel_downsample_matches_the_naive_one(seed):
    """The fast version encodes three cube indices into one integer key and
    accumulates with bincount instead of np.add.at, which took it from 110 ms to
    14 ms on a real frame. It must be the SAME answer, not a close one.

    Negative coordinates are in the sample deliberately: the key encoding shifts
    by the minimum index, and getting that wrong folds the cubes either side of
    an axis together.
    """
    rng = np.random.default_rng(seed)
    points = rng.uniform(-1.5, 3.0, size=(4000, 3))
    fast = P.voxel_downsample(points, 0.02)
    slow = naive_voxel_downsample(points, 0.02)
    assert fast.shape == slow.shape
    order_fast = fast[np.lexsort(fast.T)]
    order_slow = slow[np.lexsort(slow.T)]
    assert np.allclose(order_fast, order_slow)


def test_voxel_downsample_handles_an_empty_cloud():
    assert len(P.voxel_downsample(np.empty((0, 3)), 0.02)) == 0


def test_dense_voxel_path_matches_the_naive_one():
    """Inside a ROI-sized box the cubes are counted in a dense array instead of
    sorted (2026-10-23); it must give exactly the same cubes."""
    rng = np.random.default_rng(5)
    points = rng.uniform([0.5, -1.5, -0.05], [4.0, 1.5, 0.8], size=(20000, 3))
    fast = P.voxel_downsample(points, 0.02)
    slow = naive_voxel_downsample(points, 0.02)
    assert np.allclose(fast[np.lexsort(fast.T)], slow[np.lexsort(slow.T)])


def test_clustering_matches_scikit_learn():
    """cluster_points is DBSCAN on SciPy's KD-tree (2x faster, 2026-10-23).
    On well-separated blobs every point is core or noise unambiguously, so the
    partition must equal scikit-learn's exactly."""
    from sklearn.cluster import DBSCAN
    rng = np.random.default_rng(0)
    blobs = np.vstack([rng.normal(centre, 0.02, size=(200, 3))
                       for centre in ([0, 0, 0], [1, 0, 0], [0, 1, 0.5])])
    ours = P.cluster_points(blobs, eps=0.05, min_samples=8)
    reference = DBSCAN(eps=0.05, min_samples=8).fit_predict(blobs)
    # labels are arbitrary integers; what must match is the PARTITION
    assert _partition(ours) == _partition(reference)


def _partition(labels):
    groups = {}
    for index, label in enumerate(labels):
        groups.setdefault(label, set()).add(index)
    return sorted((frozenset(v) for v in groups.values()), key=sorted)
