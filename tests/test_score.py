"""The scorer wrapper must reproduce the organiser metric's known behaviour (EVALUATION.md table)."""

import os

import numpy as np
import pytest

from mtah import paths, score
from conftest import needs_sample


def synthetic_gt(seed=0, n_poss=2000, median=130, sigma=1.25, min_len=6):
    """A match-like GT: possessions with lognormal lengths, some separated by dead stretches.

    median 130 frames / sigma 1.25 is the lognormal whose shifted-prediction scores reproduce the
    EVALUATION.md table (0.8933 at 8 frames, 0.6539 at 25), found by a grid search over the
    analytic formula in spell_shift_expectation(). The organiser's table was measured on the
    BAR-ATH truth, which is why the table is matched with a distribution and not a constant.
    min_len 6: the table's "1 frame late is free" means no true possession is under ~6 frames
    (a 1-frame shift of a 5-frame spell has IoU 4/6 < 0.7).
    """
    rng = np.random.default_rng(seed)
    parts = [np.array(["X"] * 100)]
    team = "H"
    lengths = []
    for _ in range(n_poss):
        L = int(max(min_len, round(rng.lognormal(np.log(median), sigma))))
        lengths.append(L)
        parts.append(np.array([team] * L))
        if rng.random() < 0.35:
            parts.append(np.array(["D"] * int(max(50, rng.lognormal(np.log(500), 0.6)))))
            team = "H" if rng.random() < 0.5 else "A"
        else:
            team = "A" if team == "H" else "H"
    parts.append(np.array(["X"] * 100))
    return np.concatenate(parts)


def spell_shift_expectation(gt, k):
    """spellF1 of a k-frame-late copy if every spell matches only its own shifted self."""
    runs = score.pm.valid_runs(score.codes(gt))
    spells = score.pm.possession_spells(score.codes(gt), runs)
    L = np.array([b - a for a, b, _ in spells], dtype=np.float64)
    iou = np.clip((L - k) / (L + k), 0, None)
    return float(np.mean([(iou >= t).mean() for t in score.pm.SPELL_IOUS]))


def shift(labels, k):
    """Prediction = GT delayed by k frames (graded frames only; X stays X)."""
    out = labels.copy()
    graded = labels != "X"
    idx = np.flatnonzero(graded)
    out[idx[k:]] = labels[idx[:-k]]
    out[idx[:k]] = labels[idx[0]]
    return out


def test_uses_organiser_module_unmodified():
    assert os.path.dirname(os.path.abspath(score.pm.__file__)) == os.path.abspath(paths.REF_EVAL_DIR)


def test_perfect_prediction_scores_one():
    gt = synthetic_gt()
    r = score.evaluate(gt, gt)
    assert r["spell_f1_mean"] == pytest.approx(1.0)
    assert r["macro_f1"] == pytest.approx(1.0)


def test_one_frame_late_is_free():
    gt = synthetic_gt()
    assert score.evaluate(gt, shift(gt, 1))["spell_f1_mean"] == pytest.approx(1.0)


@pytest.mark.parametrize("k", [3, 8, 25, 50])
def test_shift_matches_analytic_iou(k):
    # Independent arithmetic: a k-late copy of a spell of length L has IoU (L-k)/(L+k).
    gt = synthetic_gt()
    assert score.evaluate(gt, shift(gt, k))["spell_f1_mean"] == pytest.approx(
        spell_shift_expectation(gt, k), abs=0.01)


@pytest.mark.parametrize("k, expected", [(3, 0.9818), (8, 0.8933), (25, 0.6539)])
def test_shift_matches_evaluation_table(k, expected):
    vals = [score.evaluate(gt, shift(gt, k))["spell_f1_mean"] for gt in
            (synthetic_gt(s) for s in range(3))]
    assert np.mean(vals) == pytest.approx(expected, abs=0.03)


def test_shift_can_fail():
    # The same check on a GT of long possessions does NOT give the table's 1 s value.
    gt = synthetic_gt(median=500, sigma=0.5)
    assert score.evaluate(gt, shift(gt, 25))["spell_f1_mean"] > 0.85


@pytest.mark.parametrize("const", ["H", "A", "D"])
def test_constant_guess_scores_zero(const):
    gt = synthetic_gt()
    pred = np.where(gt == "X", "X", const)
    assert score.evaluate(gt, pred)["spell_f1_mean"] == 0.0


def test_swapped_teams_score_zero():
    gt = synthetic_gt()
    pred = np.where(gt == "H", "A", np.where(gt == "A", "H", gt))
    assert score.evaluate(gt, pred)["spell_f1_mean"] == 0.0


def test_pooling_keeps_games_apart():
    a, b = synthetic_gt(1), synthetic_gt(2)
    # make the boundary labels equal so a naive concatenation would merge two possessions
    a[-1], b[0] = "H", "H"
    pooled = score.pool([a, b])
    runs = score.pm.valid_runs(pooled)
    n_sep = sum(score.pm.possession_spells(score.codes(x), score.pm.valid_runs(score.codes(x))).__len__()
                for x in (a, b))
    assert len(score.pm.possession_spells(pooled, runs)) == n_sep
    res = score.score_games({1: a, 2: b}, {1: a, 2: b})
    assert res["pooled"]["spell_f1_mean"] == pytest.approx(1.0)
    assert res["pooled"]["n_gt_spells"] == res[1]["n_gt_spells"] + res[2]["n_gt_spells"]


def test_bootstrap_interval_brackets_point():
    gt = synthetic_gt()
    r = score.evaluate(gt, shift(gt, 25), bootstrap_n=200)
    assert r["spell_bootstrap_low"] <= r["spell_f1_mean"] <= r["spell_bootstrap_high"]
    assert r["spell_bootstrap_high"] - r["spell_bootstrap_low"] > 0


@needs_sample
def test_shift_on_sample_gt():
    """Our sample GTs have longer possessions than BAR-ATH's: 1 s late scores ~0.86, not 0.65."""
    from mtah import sample_gt
    gt = sample_gt.sample_game_gt(1).labels
    assert score.evaluate(gt, gt)["spell_f1_mean"] == pytest.approx(1.0)
    assert score.evaluate(gt, shift(gt, 25))["spell_f1_mean"] == pytest.approx(
        spell_shift_expectation(gt, 25), abs=0.01)
