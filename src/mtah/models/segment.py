"""Segment model: a three-state Viterbi decoder over {H, A, D}.

Emissions per frame (log-evidence, energies):
- D versus alive: half the dead-ball detector's LLR (mtah.models.deadball) each way.
- H versus A, ball visible: which team's nearest player is closer, squashed (tanh of the metre
  difference over 2 m), weighted up when the ball is close to someone (a player on the ball) and
  down when the ball is moving fast (a pass or a shot in flight: the nearest player along the way
  is mostly not touching it).
- H versus A, ball missing: nothing by default. The switching cost carries the last team through
  the gap, and the frames either side of it decide where a hand-over lands.
- optional intervals with a team (Smart-Tagging phases): a weak push towards that team.

Transitions: a cost for H <-> A (the fragmentation knob) and one for alive <-> D.

Abstention (optional, `abstain_below`): forward-backward posteriors; a frame whose decoded label
has posterior below the threshold becomes X, but only strictly inside a decoded run (never its
first frame), so the scorer's forward fill restores exactly the decoded label and spellF1 is
unchanged. It only trades macroF1 (an X is a miss, a wrong label is a miss plus a false positive).
"""

import numpy as np

from .. import features as F
from . import deadball, hmm

LABELS = np.array(["H", "A", "D"])
DEFAULTS = dict(
    w_team=2.0,            # max log-evidence per frame for the nearer team
    team_scale_m=2.0,      # metres of nearest-distance difference for half of it (tanh)
    close_m=2.0,           # ball this close to the nearest player: full weight, else half
    far_weight=0.5,
    fast_mps=6.0,          # ball faster than this (1 s mean): weight x fast_weight
    fast_weight=0.25,
    cost_team=10.0,        # H <-> A switching cost
    cost_dead=12.0,        # alive <-> D switching cost
    dead_scale=0.5,        # share of the dead LLR put on each side
    w_tag_team=0.0,        # per-frame push from a team-tagged interval (off without intervals)
    abstain_below=0.0,     # 0 = never abstain
    team_mode="v1",        # "v1": tanh of the distance difference; "touch": touch-like frames only
    touch_r=1.0,           # "touch": ball within this many metres of the nearest player
    margin_m=1.0,          # "touch": full evidence when the other team is this much further away
    dead=None,             # dict of dead-ball detector params
    dead_mode="llr",       # "llr": dead LLR as emission; "mask": detector's decoded mask fixed
)


def team_evidence(f, p):
    """(n,) log-evidence for H over A (positive = home), 0 where the ball is missing."""
    if p["team_mode"] == "touch":
        return touch_evidence(f, p)
    dd = np.nan_to_num(f["ddiff"], nan=0.0)
    e = p["w_team"] * np.tanh(dd / p["team_scale_m"])
    close = np.nan_to_num(f["dmin"], nan=np.inf) < p["close_m"]
    e *= np.where(close, 1.0, p["far_weight"])
    fast = np.nan_to_num(f["bspeed_1s"], nan=0.0) > p["fast_mps"]
    e *= np.where(fast, p["fast_weight"], 1.0)
    e[f["bvis"] < 0.5] = 0.0
    return e


def touch_evidence(f, p):
    """Evidence only on touch-like frames: the ball within touch_r of the nearest player.

    On degraded sample game 1 the nearest team is right 95% of the time when the ball is within
    1 m of that player, and 53-65% (near chance) beyond 1 m, so frames with the ball further away
    carry nothing and the switching cost bridges them. The margin over the other team's nearest
    player scales the evidence (two players at the ball is a duel, not a touch).
    """
    dd = np.nan_to_num(f["ddiff"], nan=0.0)
    dmin = np.nan_to_num(f["dmin"], nan=np.inf)
    e = p["w_team"] * np.sign(dd) * np.clip(np.abs(dd) / p["margin_m"], 0.0, 1.0)
    e *= np.where(dmin < p["touch_r"], 1.0, 0.0)
    fast = np.nan_to_num(f["bspeed_1s"], nan=0.0) > p["fast_mps"]
    e *= np.where(fast, p["fast_weight"], 1.0)
    e[f["bvis"] < 0.5] = 0.0
    return e


def emissions(table, f, intervals=None, p=None):
    p = dict(DEFAULTS, **(p or {}))
    n = table.n_frames
    dp = p["dead"] or {}
    if p["dead_mode"] == "mask":
        dead_mask, _ = deadball.predict_dead(table, intervals, feats=f, **dp)
        llr = np.where(dead_mask, 30.0, -30.0)
    else:
        llr = deadball.dead_llr(f, intervals, n, **dp)
    te = team_evidence(f, p)
    if intervals and p["w_tag_team"]:
        for iv in intervals:
            if iv.get("team") in ("H", "A"):
                sgn = 1.0 if iv["team"] == "H" else -1.0
                te[int(iv["start"]):int(iv["end"])] += sgn * p["w_tag_team"]
    ds = p["dead_scale"]
    emit = np.stack([0.5 * te - ds * llr, -0.5 * te - ds * llr, ds * llr], axis=1)
    return emit


def cost_matrix(p):
    ct, cd = p["cost_team"], p["cost_dead"]
    return np.array([[0.0, ct, cd], [ct, 0.0, cd], [cd, cd, 0.0]])


def abstain(labels, path, post, threshold):
    """X where the decoded state's posterior < threshold, never on a run's first frame."""
    out = labels.copy()
    n = len(path)
    if threshold <= 0 or post is None:
        return out
    idx = np.arange(n)
    ok = path >= 0
    conf = np.where(ok, post[idx, np.clip(path, 0, None)], 1.0)
    first = np.ones(n, dtype=bool)
    first[1:] = path[1:] != path[:-1]
    sel = ok & (conf < threshold) & ~first
    out[sel] = "X"
    return out


def predict(table, intervals=None, feats=None, **params):
    p = dict(DEFAULTS, **params)
    f = feats if feats is not None else F.cached(table)
    emit = emissions(table, f, intervals, p)
    want = p["abstain_below"] > 0
    path, post = hmm.decode(emit, cost_matrix(p), table.period, want_post=want)
    labels = np.where(path >= 0, LABELS[np.clip(path, 0, 2)], "X")
    return abstain(labels, path, post, p["abstain_below"])
