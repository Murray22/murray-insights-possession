"""Our baseline port must match the organiser's starter/baseline.py logic."""

import os
import sys

import numpy as np
import pytest

from mtah import baseline, paths
from mtah.frames import AWAY, HOME, FrameTable


@pytest.fixture(scope="module")
def organiser_baseline():
    sys.path.insert(0, os.path.join(paths.REF_DIR, "starter"))
    import baseline as ref_baseline  # ref/starter/baseline.py
    return ref_baseline


@pytest.mark.parametrize("window", [0, 25, 50, 125])
def test_smoothing_identical_to_organiser(organiser_baseline, window):
    rng = np.random.default_rng(window)
    codes = rng.choice(np.array([-1, 0, 1], dtype=np.int8), 20000, p=[0.4, 0.3, 0.3])
    codes[5000:6000] = -1                                 # a long gap, D in every variant
    ours = baseline.smooth(codes, window)
    theirs = organiser_baseline.smooth(codes, window)
    np.testing.assert_array_equal(ours, theirs)


def _table(ball, players, teams):
    n = len(ball)
    return FrameTable(np.arange(n), np.ones(n, dtype=np.int8), np.array(ball, dtype=np.float32),
                      np.array(players, dtype=np.float32), np.array(teams, dtype=np.int8),
                      (105.0, 68.0))


def test_nearest_player_in_metres_and_missing_ball():
    nan = np.nan
    # frame 0: home player 0.05 away along x (5.25 m), away player 0.06 along y (4.08 m):
    #   in normalised units home looks nearer, in metres away is -> A
    # frame 1: ball missing -> NOBODY; frame 2: away slot empty -> home
    ball = [[0.5, 0.5], [nan, nan], [0.5, 0.5]]
    players = [[[0.55, 0.5], [0.5, 0.56]],
               [[0.55, 0.5], [0.5, 0.56]],
               [[0.55, 0.5], [nan, nan]]]
    codes = baseline.nearest_player_codes(_table(ball, players, [HOME, AWAY]))
    assert codes.tolist() == [AWAY, baseline.NOBODY, HOME]


def test_four_variants():
    ball = [[0.5, 0.5]] * 10
    players = [[[0.5, 0.5], [0.9, 0.9]]] * 10
    preds = baseline.predict_all(_table(ball, players, [HOME, AWAY]))
    assert [n for n, _ in baseline.VARIANTS] == list(preds)
    assert all((p == "H").all() for p in preds.values())
