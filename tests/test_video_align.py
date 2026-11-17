"""The alignment checker must say no to a one-frame slip, and yes to the aligned case."""

import numpy as np
import pytest

from mtah.video import align


def _scene(n=400, seed=0, fp=3, miss=0.2, noise=1.5):
    """A ball moving 6-14 px per frame, a detector that sees it with noise, misses some frames
    and adds `fp` random false positives per frame."""
    rng = np.random.default_rng(seed)
    v = rng.uniform(6, 14, size=(n, 2)) * rng.choice([-1, 1], size=(n, 2))
    traj = 960 + np.cumsum(v, axis=0) % 800
    dets = []
    for t in range(n):
        pts = [rng.uniform(0, 1920, size=2) for _ in range(fp)]
        if rng.random() > miss:
            pts.append(traj[t] + rng.normal(0, noise, 2))
        dets.append(np.array(pts))
    return traj, dets


def _refs(traj, slip=0):
    """Reference at shift k for frame t = trajectory at t + k + slip (NaN past the ends)."""
    n = len(traj)
    out = {}
    for k in range(-3, 4):
        idx = np.arange(n) + k + slip
        r = np.full((n, 2), np.nan)
        ok = (idx >= 0) & (idx < n)
        r[ok] = traj[idx[ok]]
        out[k] = r
    return out


def test_aligned_passes():
    traj, dets = _scene()
    res = align.check(_refs(traj), dets)
    assert res["aligned"] and res["best"] == 0
    assert res["hit_rate"][0] > 0.7 and res["n_informative"] > 100


@pytest.mark.parametrize("slip", [1, -1, 2])
def test_one_frame_slip_fails(slip):
    traj, dets = _scene(seed=1)
    res = align.check(_refs(traj, slip=slip), dets)
    assert not res["aligned"]
    assert res["best"] == -slip


def test_no_signal_fails():
    """Detections unrelated to the reference: nothing to align, so not 'aligned'."""
    traj, _ = _scene(seed=2)
    _, dets = _scene(seed=3)
    assert not align.check(_refs(traj), dets)["aligned"]


def test_missing_shift_raises():
    traj, dets = _scene(n=20)
    refs = _refs(traj)
    del refs[1]
    with pytest.raises(ValueError):
        align.check(refs, dets)


def test_sign_test_large_n():
    assert align._sign_test_p(3000, 2000) < 1e-20
    assert align._sign_test_p(2000, 2000) > 0.4
