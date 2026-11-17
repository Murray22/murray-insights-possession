"""The video ball source fills only ATD gaps, never overwrites the ATD, and flags provenance; the
full-run planner covers every gap frame plus its window and only graded frames."""

import numpy as np

from mtah.frames import FrameTable
from mtah.video import fullrun, source


def _table(n=10):
    ball = np.full((n, 2), 0.5, np.float32)
    ball[3:7] = np.nan
    return FrameTable(frame=np.arange(n), period=np.ones(n, np.int8), ball=ball,
                      xy=np.full((n, 2, 2), np.nan, np.float32), slot_team=np.array([0, 1], np.int8))


def _track(n=10):
    t = dict(processed=np.ones(n, bool), ball=np.full((n, 2), 0.2, np.float32),
             support=np.full(n, 10, np.int32), config=np.array("x"))
    t["support"][5] = 2                                    # too short to trust
    return t


def test_fills_gaps_only_and_flags():
    out = source.with_video_ball(_table(), track=_track())
    src = out.meta["ball_source"]
    assert src.tolist() == [0, 0, 0, 1, 1, -1, 1, 0, 0, 0]
    assert np.allclose(out.ball[[0, 1, 2, 7, 8, 9]], 0.5)          # ATD untouched
    assert np.allclose(out.ball[[3, 4, 6]], 0.2)
    assert np.isnan(out.ball[5]).all()


def test_original_table_unchanged():
    t = _table()
    source.with_video_ball(t, track=_track())
    assert np.isnan(t.ball[3:7]).all()


def test_frames_to_process_window_and_graded():
    n = 200
    c = dict(period=np.zeros(n, np.int8), ball=np.zeros((n, 2)))
    c["period"][20:180] = 1
    c["ball"][100:110] = np.nan            # one gap in play
    c["ball"][0:10] = np.nan               # a gap outside the graded halves
    m = fullrun.frames_to_process(c, "gaps", window=5)
    assert np.flatnonzero(m).tolist() == list(range(95, 115))
    assert sum(n for _, n in fullrun.chunks(m, size=8)) == 20
