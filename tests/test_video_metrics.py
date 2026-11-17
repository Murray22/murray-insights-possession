"""Geometry round trip, candidate filtering, recall and frame-to-frame linking on a synthetic
camera (a pure scaling homography: 10 px per metre, pitch at the image origin)."""

import numpy as np

from mtah.video import geometry, metrics
from mtah.video.detectors import BALL, PERSON

PITCH = geometry.PITCH_M
H = np.diag([PITCH[0] * 10.0, PITCH[1] * 10.0, 1.0])     # normalised pitch -> px, 10 px/m


def _cache(n, ball):
    return dict(H=np.repeat(H[None], n, 0), ball=np.asarray(ball, float),
                xy=np.full((n, 2, 2), np.nan))


def test_project_unproject_round_trip():
    Hp = np.array([[1822.7, -511.2, 44.5], [1.7, 169.3, 255.7], [-0.019, -0.527, 1.0]])
    x, y = np.array([0.1, 0.5, 0.9]), np.array([0.2, 0.5, 0.8])
    px, py = geometry.project(Hp, x, y)
    bx, by = geometry.unproject(Hp, px, py)
    assert np.allclose(bx, x) and np.allclose(by, y)


def test_px_per_metre():
    assert np.isclose(geometry.px_per_metre(H, 0.5, 0.5), 10.0)


def test_candidates_drop_off_pitch_low_conf_and_persons():
    d = np.array([[500, 300, 3, 3, 0.9, BALL],        # on pitch
                  [500, 300, 3, 3, 0.01, BALL],       # below threshold
                  [1200, 300, 3, 3, 0.9, BALL],       # 15 m beyond the goal line
                  [500, 300, 20, 60, 0.9, PERSON]], dtype=np.float32)
    pix, ground = metrics.candidates(d, H, conf=0.1)
    assert len(pix) == 1 and np.allclose(pix[0, :2], [500, 300])


def test_visible_recall_and_fp():
    ball = [[0.5, 0.5], [0.5, 0.5], [np.nan, np.nan]]
    c = _cache(3, ball)
    ref = np.array(geometry.project(H, 0.5, 0.5))
    dets = [np.array([[ref[0] + 2, ref[1], 3, 3, 0.8, BALL], [100, 100, 3, 3, 0.9, BALL]]),
            np.array([[ref[0] + 30, ref[1], 3, 3, 0.8, BALL]]),
            np.zeros((0, 6))]
    rows = metrics.per_frame(c, [0, 1, 2], dets, conf=0.1)
    m = metrics.visible_metrics(rows)
    assert m["frames"] == 2
    assert m["recall_px"] == 0.5          # frame 0 hit (not top-1), frame 1 30 px off
    assert m["top1_px"] == 0.0
    assert m["fp_per_frame"] == 1.0       # one FP on each visible frame


def test_linked_respects_speed_and_clip_boundaries():
    # clip_len 2: frames 0,1 are one clip, 2,3 another. 1 frame at 40 m/s = 1.6 m = 16 px (+3).
    pts = [(500, 300), (515, 300), (530, 300), (700, 300)]
    dets = [np.array([[x, y, 3, 3, 0.9, BALL]]) for x, y in pts]
    c = _cache(4, [[np.nan, np.nan]] * 4)
    rows = metrics.per_frame(c, [0, 1, 2, 3], dets, conf=0.1)
    lk = metrics.linked(rows, 2, c["H"])
    assert lk.tolist() == [1.0, 1.0, 0.0, 0.0]   # 1->2 is 15 px but crosses clips; 2->3 is 170 px
