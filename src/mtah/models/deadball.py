"""Dead-ball detector from tracking only (no events, no Smart Tagging required).

A per-frame log-likelihood ratio, dead versus alive, summed from a few cues with coarse weights,
then a two-state Viterbi with a switching cost so stoppages come out as whole stretches:

- motion: a whole-pitch speed collapse. The 80th percentile of player speeds over +/- 1.5 s, and
  the median over +/- 4 s. Players walk during stoppages and run during play; this is the cue that
  does not depend on the ball being seen.
- ball off the pitch: a visible ball more than 1 m outside the lines.
- ball still: a visible ball that has barely moved over +/- 1 s (placed for a restart).
- ball fast: a visible ball moving fast is in play (negative evidence).
- long ball gaps: a ball unseen for over 10 s, weak evidence on its own (the tracker also loses
  live balls), which the motion cue has to confirm.
- restart shapes: a kick-off layout (each team inside its own half), and a player standing off the
  pitch (a throw-in taker).
- optional Smart-Tagging-like intervals: a set-piece interval puts dead evidence on the frames
  just before its start. The sample games have none; on BAR-ATH this is where they plug in.

All weights are round numbers set from the feature distributions on degraded sample game 1 (dead
versus alive medians, docs/model_log.md) and then left alone; the switching cost is the one knob
chosen on the bench, from a coarse grid.
"""

import numpy as np

from .. import features as F
from . import hmm

DEFAULTS = dict(
    bias=-0.5,                 # prior: about 40% of graded frames are dead
    w_motion=2.0, ref_motion=2.2,      # per m/s of pspd_p80_3s below 2.2 m/s
    w_motion8=1.5, ref_motion8=1.7,    # per m/s of pspd_8s below 1.7 m/s
    w_out=3.0, out_m=1.0,
    w_still=1.0,
    w_fast=-1.5, fast_mps=8.0,
    w_gap10=1.0, w_gap20=0.5,
    w_kickoff=2.0,
    w_offpitch=0.5,
    w_tag=2.0, tag_lead_s=3.0,
    switch_cost=15.0,
    # "raw": thresholds in m/s (v1); "game": on per-game standardised speeds (v2), where the dead
    # and alive medians sit near -0.5..-0.7 and +0.25..+0.45 in every degrader variant
    motion_norm="raw",
    ref_z=-0.2, w_motion_z=3.0, w_motion8_z=2.0,
    # "range" (v3): (x - q05) / (q75 - q05) over graded frames. Unlike median / IQR it does not
    # assume how much of the game is dead or slow: q05 is standing still plus noise, q75 is
    # active play. Dead medians sit at 0.15-0.27 and alive 10th percentiles at 0.38-0.49 in every
    # game and noise level tried (docs/model_log.md); a slow build-up at half the speed of play
    # lands near 0.4 (tests/test_models.py).
    ref_r=0.45, w_motion_r=5.0, w_motion8_r=5.0,       # grid {0.33, 0.4, 0.45} x {5, 8}
)


def range_norm(x, mask, lo=5, hi=75):
    v = x[mask & ~np.isnan(x)]
    if len(v) < 10:
        return np.full_like(x, np.nan)
    a, b = np.percentile(v, [lo, hi])
    return (x - a) / max(b - a, 1e-6)


def dead_llr(feats, intervals=None, n=None, **params):
    p = dict(DEFAULTS, **params)
    f = feats
    n = len(f["bvis"]) if n is None else n
    llr = np.full(n, p["bias"])
    if p["motion_norm"] == "range":
        graded = f["period"] > 0 if "period" in f else np.ones(n, dtype=bool)
        m = range_norm(f["pspd_p80_3s"], graded)
        m8 = range_norm(f["pspd_8s"], graded)
        m = np.nan_to_num(m, nan=p["ref_r"])
        m8 = np.nan_to_num(m8, nan=p["ref_r"])
        llr += np.clip(p["w_motion_r"] * (p["ref_r"] - m), -3.0, 3.0)
        llr += np.clip(p["w_motion8_r"] * (p["ref_r"] - m8), -2.0, 2.0)
    elif p["motion_norm"] == "game":
        m = np.nan_to_num(f["pspd_p80_3s_z"], nan=p["ref_z"])
        llr += np.clip(p["w_motion_z"] * (p["ref_z"] - m), -3.0, 3.0)
        m8 = np.nan_to_num(f["pspd_8s_z"], nan=p["ref_z"])
        llr += np.clip(p["w_motion8_z"] * (p["ref_z"] - m8), -2.0, 2.0)
    else:
        m = np.nan_to_num(f["pspd_p80_3s"], nan=p["ref_motion"])
        llr += np.clip(p["w_motion"] * (p["ref_motion"] - m), -3.0, 3.0)
        m8 = np.nan_to_num(f["pspd_8s"], nan=p["ref_motion8"])
        llr += np.clip(p["w_motion8"] * (p["ref_motion8"] - m8), -2.0, 2.0)
    vis = f["bvis"] > 0.5
    with np.errstate(invalid="ignore"):
        llr += p["w_out"] * (vis & (np.nan_to_num(f["bout"]) > p["out_m"]))
        llr += p["w_still"] * (vis & (np.nan_to_num(f["bstill_2s"]) > 0.5))
        llr += p["w_fast"] * (vis & (np.nan_to_num(f["bspeed_1s"]) > p["fast_mps"]))
    gap = np.where(vis, 0.0, f["gap_len"])
    llr += p["w_gap10"] * (gap > 250) + p["w_gap20"] * (gap > 500)
    llr += p["w_kickoff"] * (np.nan_to_num(f["separation_1s"]) > 0.95)
    llr += p["w_offpitch"] * (f["n_off"] >= 1)
    if intervals:
        lead = int(round(p["tag_lead_s"] * F.FPS))
        for iv in intervals:
            if iv.get("kind") == "set_piece":
                a = max(0, int(iv["start"]) - lead)
                llr[a:int(iv["start"]) + 1] += p["w_tag"]
    return llr


def predict_dead(table, intervals=None, feats=None, **params):
    """(n,) bool dead mask (False outside graded runs), and the per-frame LLR."""
    p = dict(DEFAULTS, **params)
    f = feats if feats is not None else F.cached(table)
    llr = dead_llr(f, intervals, table.n_frames, **params)
    emit = np.stack([-0.5 * llr, 0.5 * llr], axis=1)          # [alive, dead]
    c = p["switch_cost"]
    path, _ = hmm.decode(emit, np.array([[0.0, c], [c, 0.0]]), table.period)
    return path == 1, llr
