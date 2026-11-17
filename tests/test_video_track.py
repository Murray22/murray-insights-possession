"""The ball tracker must follow a moving ball through clutter, refuse an isolated blip, and refuse
to jump faster than a ball can travel."""

import numpy as np

from mtah.video import track


def _c(*pts):
    return np.array(pts, dtype=float).reshape(-1, 3)


def test_follows_ball_through_clutter():
    rng = np.random.default_rng(0)
    cands, truth = [], []
    for t in range(30):
        ball = (500 + 8 * t, 400)
        fps = [(rng.uniform(0, 1920), rng.uniform(0, 1080), 0.4) for _ in range(3)]
        cands.append(_c(*fps, (ball[0], ball[1], 0.3)))
        truth.append(3)
    path = track.viterbi(cands, np.full(30, 10.0))
    assert (path == np.array(truth)).mean() > 0.9


def test_isolated_blip_is_none():
    cands = [_c() for _ in range(10)]
    cands[5] = _c((800, 500, 0.3))
    path = track.viterbi(cands, np.full(10, 10.0))
    assert (path == -1).all()


def test_teleport_refused():
    # a confident ball on frames 0-9 at x=500, then 600 px away on 10-19: cannot be one track,
    # so the tracker must not link them through a jump (each side may be kept, the jump never)
    cands = [_c((500, 500, 0.9)) for _ in range(10)] + [_c((1100, 500, 0.9)) for _ in range(10)]
    path = track.viterbi(cands, np.full(20, 10.0))
    assert path[9] == -1 or path[10] == -1


def test_support_lengths():
    assert track.support(np.array([-1, 0, 0, 0, -1, 2, -1])).tolist() == [0, 3, 3, 3, 0, 1, 0]
