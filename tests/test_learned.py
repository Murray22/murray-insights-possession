"""Learned emissions: home/away symmetry, the leave-one-game-out fold, label swapping."""

import numpy as np

from mtah.frames import FrameTable
from mtah.models import learned


def tiny_table(game=None, kind="degraded"):
    n = 10
    return FrameTable(np.arange(n), np.ones(n, dtype=np.int8), np.zeros((n, 2), np.float32),
                      np.zeros((n, 4, 2), np.float32), np.array([0, 0, 1, 1], np.int8),
                      (105.0, 68.0), None, {"game": game, "kind": kind})


def test_fold_excludes_the_test_game():
    assert learned.fold_games(tiny_table(2)) == (1, 3)
    assert learned.fold_games(tiny_table(1)) == (2, 3)
    assert learned.fold_games(tiny_table(None, "atd")) == (1, 2, 3)
    assert learned.fold_games(tiny_table(3, "clean")) == (1, 2, 3)


def test_swap_labels_and_table():
    lab = np.array(["H", "A", "D", "X", "H"])
    assert list(learned.swap_labels(lab)) == ["A", "H", "D", "X", "A"]
    t = tiny_table(1)
    s = learned.swapped(t)
    assert list(s.slot_team) == [1, 1, 0, 0] and list(t.slot_team) == [0, 0, 1, 1]


def test_emissions_symmetric_prior_and_scale():
    P = np.array([[0.7, 0.2, 0.1], [0.2, 0.7, 0.1], [0.1, 0.1, 0.8]])
    prior = np.array([0.35, 0.25, 0.4])                 # asymmetric training prior
    e = learned.emissions_from_P(P, prior, 1.0)
    # H and A are judged against the same (averaged) prior: a mirrored P gives mirrored evidence
    assert np.isclose(e[0, 0], e[1, 1]) and np.isclose(e[0, 1], e[1, 0])
    assert np.allclose(learned.emissions_from_P(P, prior, 0.5), 0.5 * e)


def test_feature_sets_differ_as_documented():
    f = {k: None for k in ("tt_past25", "gap_len", "pspd_p80_3s_z", "pspd_p80_3s_r", "bx",
                           "pspd_p80_3s")}
    assert learned.feature_names(f, "gbm1") == ["gap_len", "pspd_p80_3s_z", "tt_past25"]
    assert learned.feature_names(f, "gbm1_novis") == ["pspd_p80_3s_z", "tt_past25"]
    assert learned.feature_names(f, "gbm2") == ["gap_len", "pspd_p80_3s_r", "tt_past25"]
    assert learned.feature_names(f, "gbm3") == ["pspd_p80_3s_r", "tt_past25"]


# -- Phase D: possession-length weighting ----------------------------------------------------------

def test_possession_lengths_cut_where_the_scorer_cuts():
    lab = np.array(list("HHHAADDHHX"))
    assert list(learned.possession_lengths(lab)) == [3, 3, 3, 2, 2, 0, 0, 2, 2, 0]


def test_row_weights_favour_short_possessions_and_keep_the_class_balance():
    L = np.array([25, 25, 250, 250, 250, 250, 0, 0])      # two short-possession rows, four long, two D
    w = learned.row_weights(L, 1.0)
    assert np.isclose(w[L > 0].mean(), 1.0) and np.all(w[L == 0] == 1.0)
    assert np.isclose(w[0] / w[2], 10.0)
    # gamma 0 is the unweighted fit; a 1-frame possession is weighted like a 1-second one
    assert np.all(learned.row_weights(L, 0.0) == 1.0)
    w2 = learned.row_weights(np.array([1, 25, 250]), 1.0)
    assert np.isclose(w2[0], w2[1])


def test_training_weights_refuse_rows_that_do_not_line_up(monkeypatch):
    """The weights are recomputed from the row draw; if that draw ever drifts from _train_part's,
    the check against the cached labels must fail rather than weight the wrong rows."""
    gt = np.array(list("HHHHAADDDDHHHHHAAAAA"))
    rows = np.arange(0, 20, 2)
    monkeypatch.setattr(learned, "TRAIN_SEEDS", (0,))
    monkeypatch.setattr(learned, "_part_rows", lambda g, m, s, k, mixes: (rows, gt))
    code = {"H": 0, "A": 1, "D": 2}
    y = np.array([code[c] for c in gt[rows]], dtype=np.int8)
    ysw = np.where(y == 0, 1, np.where(y == 1, 0, y)).astype(np.int8)
    good = np.concatenate([y, ysw])
    w = learned.training_weights((1,), "touch", {"m": {}}, 1.0, good)
    assert len(w) == 2 * len(rows)
    bad = np.concatenate([np.roll(y, 1), ysw])
    import pytest
    with pytest.raises(ValueError):
        learned.training_weights((1,), "touch", {"m": {}}, 1.0, bad)
    with pytest.raises(ValueError):
        learned.training_weights((1,), "touch", {"m": {}}, 1.0, np.concatenate([good, good[:2]]))
