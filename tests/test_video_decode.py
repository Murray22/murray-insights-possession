"""Frame-exact decoding: the pts check refuses shifted or short clips; on the real video (skipped
when it is absent) a seek lands on the same pixels as decoding forward, and a one-frame offset does
not."""

import os

import numpy as np
import pytest

from mtah.video import decode


def test_check_pts_accepts_exact_run():
    decode.check_pts([5000, 6000, 7000], start=5, n=3)


@pytest.mark.parametrize("pts", [[6000, 7000, 8000],      # one frame late
                                 [4000, 5000, 6000],      # one frame early
                                 [5000, 6000],            # short
                                 [5000, 5000, 6000],      # duplicated frame
                                 [5000, 7000, 8000]])     # dropped frame
def test_check_pts_refuses(pts):
    with pytest.raises(decode.FrameIndexError):
        decode.check_pts(pts, start=5, n=3)


def test_out_of_range_refused():
    with pytest.raises(ValueError):
        decode.read_clip(decode.N_FRAMES - 1, 2)


needs_video = pytest.mark.skipif(not os.path.isfile(decode.video_path()),
                                 reason="tactical video not present")


@needs_video
def test_seek_matches_forward_decode():
    a = decode.read_clip(100030, 4, scale=0.25)
    b = decode.read_clip(100000, 40, scale=0.25)          # forward from the keyframe before
    assert np.array_equal(a, b[30:34])
    assert not np.array_equal(a, b[31:35])                # a one-frame slip is visible
    assert not np.array_equal(a, b[29:33])


@needs_video
def test_last_frame_decodes():
    a = decode.read_clip(decode.N_FRAMES - 2, 2, scale=0.25)
    assert a.shape[0] == 2
