"""Score predictions with the organiser's own metric code, on any game, not only BAR-ATH.

possession_metrics.py is imported from ref/evaluation via sys.path, never copied or edited, so a
score here means exactly what the organiser's board means. Its file-level entry point
(validate_submission) insists on 199,319 rows; for the sample games we call evaluate() and the
bootstrap functions directly on label arrays of any length.

Pooling several games: the arrays are concatenated with a one-frame X separator, so
possession_metrics.valid_runs() keeps every game (and every half) a separate run and no possession
or transition can straddle two games.
"""

import os
import sys

import numpy as np

from . import paths

if paths.REF_EVAL_DIR not in sys.path:
    sys.path.insert(0, paths.REF_EVAL_DIR)
import possession_metrics as pm  # noqa: E402  (organiser code, ref/evaluation)

HEADLINE_KEYS = ("spell_f1_mean", "macro_f1", "spell_bootstrap_low", "spell_bootstrap_high",
                 "bootstrap_low", "bootstrap_high", "n_gt_spells", "n_pred_spells",
                 "mean_spell_duration_s", "gt_mean_spell_duration_s", "alive_dead_f1",
                 "team_acc_both_alive", "fragmentation")


def codes(labels):
    """Label strings (or a frame,label DataFrame) -> organiser codes 0..3."""
    if isinstance(labels, np.ndarray) and labels.dtype.kind in "iu":
        return labels.astype(np.int8)
    return pm.label_codes(np.asarray(labels))


def evaluate(gt, pred, bootstrap_n=0):
    """pm.evaluate on one pair of label arrays; returns the organiser's full result dict."""
    gt_c, pred_c = codes(gt), codes(pred)
    if len(gt_c) != len(pred_c):
        raise ValueError("length mismatch: gt {} vs pred {}".format(len(gt_c), len(pred_c)))
    return pm.evaluate(gt_c, pred_c, bootstrap_n=bootstrap_n)


def pool(arrays):
    """Concatenate per-game label arrays with an X frame between them."""
    parts = []
    for k, a in enumerate(arrays):
        if k:
            parts.append(np.array([pm.X], dtype=np.int8))
        parts.append(codes(a))
    return np.concatenate(parts)


def headline(result):
    return {k: result.get(k) for k in HEADLINE_KEYS}


def score_games(gts, preds, bootstrap_n=0):
    """{game: headline, ..., 'pooled': headline} for parallel dicts {game: labels}."""
    games = list(gts)
    out = {}
    for g in games:
        out[g] = headline(evaluate(gts[g], preds[g], bootstrap_n))
    out["pooled"] = headline(pm.evaluate(pool([gts[g] for g in games]),
                                         pool([preds[g] for g in games]),
                                         bootstrap_n=bootstrap_n))
    return out
