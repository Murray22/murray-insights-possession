"""Named model versions, as scored in docs/model_log.md. Each entry: FrameTable -> labels.

    from mtah import registry
    labels = registry.get("seg_v1")(table)            # (n,) array of 'H' 'A' 'D' 'X'
"""

import numpy as np

from . import baseline
from . import features as F
from .models import deadball, learned, segment


def _period_mask(table, labels):
    return np.where(table.period > 0, labels, "X")


def smooth50(table, **_):
    return _period_mask(table, baseline.smooth(baseline.nearest_player_codes(table), 50))


def dead_v1_s50(table, intervals=None, **params):
    """Dead-ball detector v1 for D; smooth50 nearest-team vote (dead frames excluded) for H / A."""
    dead, _ = deadball.predict_dead(table, intervals, **params)
    codes = baseline.nearest_player_codes(table)
    codes = np.where(dead, baseline.NOBODY, codes)
    lab = baseline.smooth(codes, 50)
    # alive frames with no visible ball in the window: carry the last team
    alive = ~dead
    lab = np.where(dead, "D", lab)
    team = np.where(lab == "D", "", lab)
    last = ""
    out = lab.copy()
    for t in np.flatnonzero(alive):
        if team[t]:
            last = team[t]
        elif last:
            out[t] = last
    return _period_mask(table, out)


def seg(table, intervals=None, **params):
    return segment.predict(table, intervals, **params)


def dead_v2_s50(table, intervals=None, **params):
    return dead_v1_s50(table, intervals, **dict(dict(motion_norm="game"), **params))


SEG2 = dict(team_mode="touch", fast_mps=8.0, fast_weight=0.5, w_team=4.0, cost_team=10.0)


def seg_v3(table, intervals=None, **kw):
    """seg_v2 with the v2 dead-ball detector (per-game standardised motion)."""
    p = dict(SEG2, dead=dict(motion_norm="game"))
    p.update(kw)
    return segment.predict(table, intervals, **p)


def dead_v3_s50(table, intervals=None, **params):
    return dead_v1_s50(table, intervals, **dict(dict(motion_norm="range"), **params))


def seg_v4(table, intervals=None, **kw):
    """seg_v2 with the v3 dead-ball detector (per-game range-normalised motion)."""
    p = dict(SEG2, dead=dict(motion_norm="range"))
    p.update(kw)
    return segment.predict(table, intervals, **p)


MODELS = {
    "smooth50": smooth50,
    "dead_v1_s50": dead_v1_s50,
    "seg_v1": seg,
    "seg_v2": lambda table, intervals=None, **kw: segment.predict(
        table, intervals, **dict(dict(team_mode="touch", fast_mps=8.0, fast_weight=0.5), **kw)),
    "dead_v2_s50": dead_v2_s50,
    "seg_v3": seg_v3,
    "dead_v3_s50": dead_v3_s50,
    "seg_v4": seg_v4,
    "gbm_v1": lambda table, intervals=None, **kw: learned.predict(table, intervals, **kw),
    "gbm_v1_novis": lambda table, intervals=None, **kw: learned.predict(
        table, intervals, **dict(dict(tag="gbm1_novis"), **kw)),
    "gbm_v2": lambda table, intervals=None, **kw: learned.predict(
        table, intervals, **dict(dict(tag="gbm2"), **kw)),
    "gbm_v3": lambda table, intervals=None, **kw: learned.predict(
        table, intervals, **dict(dict(tag="gbm3"), **kw)),
    "gbm_v3_ctrl": lambda table, intervals=None, **kw: learned.predict(
        table, intervals, **dict(dict(tag="gbm3", gt_key="controlled"), **kw)),
    "gbm_v4": lambda table, intervals=None, **kw: learned.predict(
        table, intervals, **dict(dict(tag="gbm4"), **kw)),
    # Phase B2: trained on the degrader calibrated to the real ATD (modelbench.ATD_CAL mixture)
    "gbm_v5a": lambda table, intervals=None, **kw: learned.predict(
        table, intervals, **dict(dict(tag="gbm5a"), **kw)),
    "gbm_v5": lambda table, intervals=None, **kw: learned.predict(
        table, intervals, **dict(dict(tag="gbm5"), **kw)),
    "gbm_v5_nop": lambda table, intervals=None, **kw: learned.predict(
        table, intervals, **dict(dict(tag="gbm5_nop"), **kw)),
    "gbm_v5_nogap": lambda table, intervals=None, **kw: learned.predict(
        table, intervals, **dict(dict(tag="gbm5_nogap"), **kw)),
    # Phase D: gbm_v5_nogap with training rows weighted by possession length (gamma 0.5 / 1)
    "gbm_v6_w05": lambda table, intervals=None, **kw: learned.predict(
        table, intervals, **dict(dict(tag="gbm6_w05"), **kw)),
    "gbm_v6_w1": lambda table, intervals=None, **kw: learned.predict(
        table, intervals, **dict(dict(tag="gbm6_w1"), **kw)),
    # Phase D: plus the touch-kink features (features_d), unweighted and weighted (gamma 0.5)
    "gbm_v7": lambda table, intervals=None, **kw: learned.predict(
        table, intervals, **dict(dict(tag="gbm7"), **kw)),
    "gbm_v7_w05": lambda table, intervals=None, **kw: learned.predict(
        table, intervals, **dict(dict(tag="gbm7_w05"), **kw)),
    # Phase E: gbm_v5_nogap's and gbm_v7's designs retrained on atd_cal2 (the ball-shape fit)
    "gbm_v5_nogap_c2": lambda table, intervals=None, **kw: learned.predict(
        table, intervals, **dict(dict(tag="gbm5n_c2"), **kw)),
    "gbm_v7_c2": lambda table, intervals=None, **kw: learned.predict(
        table, intervals, **dict(dict(tag="gbm7_c2"), **kw)),
    # Phase E: plus the video ball channel (features_v), trained on atd_cal2v. They read
    # table.vball; on the competition match mtah.predict attaches the saved video track
    "gbm_v8": lambda table, intervals=None, **kw: learned.predict(
        table, intervals, **dict(dict(tag="gbm8"), **kw)),
    "gbm_v8n": lambda table, intervals=None, **kw: learned.predict(
        table, intervals, **dict(dict(tag="gbm8n"), **kw)),
}
# models that read the video channel: the competition table needs it attached (mtah.predict)
VIDEO_MODELS = frozenset({"gbm_v8", "gbm_v8n"})


def get(name):
    if name in MODELS:
        return MODELS[name]
    raise KeyError("unknown model {!r}; have {}".format(name, sorted(MODELS)))
