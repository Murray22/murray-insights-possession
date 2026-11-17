"""Phase D features: which touches change possession? The ball's velocity change at a touch.

The Phase D diagnosis (docs/phaseD_segmentation.md) found that the models miss short possessions
because their emissions never favour the short possession's team, not because the decoder
suppresses them. The touch features (features.tt*) see a nearby player of the possession's team in
most short possessions, but they see as many touches by the team without the ball that change
nothing: a defender running past, a ball rolling by. On the degraded bench, distance to the
nearest player does not tell the two apart (AUC 0.52), and the ball's velocity change at the touch
does (AUC 0.79): a real touch deflects, stops or kicks the ball, a ball passing a player does not.

Columns, all named "d_", all symmetric under a team swap (a touch's sign flips with the teams):

- `d_kink4`, `d_kink8`: |v(t, t + k) - v(t - k, t)| in m/s, the ball's velocity change over k
  frames either side (NaN where either side is missing).
- `d_ttk`: the 1.5 m touch sign (+1 home, -1 away) times the clipped kink: a signed touch strength.
- `d_ttk_{past,next}{12,25,75}`: its windowed sums.
- strong touches (a touch with a kink of 2 m/s or more): the team of the last / next one, frames
  since / until the last / next one by each team.

Tracking only, computed from the table alone (it does not need features.compute's columns).
"""

import numpy as np

from . import features as F
from .frames import AWAY, HOME

EXTRA_VERSION = 1
TOUCH_R = 1.5            # m, as features.tt15
STRONG_KINK = 2.0        # m/s
KINK_CLIP = 15.0


def touch_sign(table, r=TOUCH_R):
    """+1 / -1 where the ball is within r of a home / away player and nearer that team; 0 else."""
    L, W = table.pitch
    sc = np.array([L, W], dtype=np.float64)
    ball = table.ball.astype(np.float64) * sc
    xy = table.xy.astype(np.float64) * sc
    dist = np.linalg.norm(xy - ball[:, None, :], axis=2)
    dist = np.where(np.isnan(dist), np.inf, dist)
    team = table.slot_team
    dh = dist[:, team == HOME].min(axis=1) if (team == HOME).any() else np.full(len(ball), np.inf)
    da = dist[:, team == AWAY].min(axis=1) if (team == AWAY).any() else np.full(len(ball), np.inf)
    dmin = np.minimum(dh, da)
    s = np.where(dh < da, 1.0, np.where(da < dh, -1.0, 0.0))
    return np.where(dmin < r, s, 0.0)


def kink(table, k):
    """|ball velocity over (t, t + k) - velocity over (t - k, t)| in m/s; NaN if a side is missing."""
    L, W = table.pitch
    b = table.ball.astype(np.float64) * np.array([L, W])
    n = len(b)
    v = (b[k:] - b[:-k]) * F.FPS / k                     # v[i] = velocity from i to i + k
    after = np.full((n, 2), np.nan)
    before = np.full((n, 2), np.nan)
    after[:n - k] = v
    before[k:] = v
    return np.linalg.norm(after - before, axis=1)


def compute_extra(table):
    n = table.n_frames
    out = {}
    tt = touch_sign(table)
    k4, k8 = kink(table, 4), kink(table, 8)
    out["d_kink4"], out["d_kink8"] = k4, k8
    ttk = tt * np.minimum(np.nan_to_num(k8, nan=0.0), KINK_CLIP)
    out["d_ttk"] = ttk
    c = np.concatenate(([0.0], np.cumsum(ttk)))
    idx = np.arange(n)
    for w in (12, 25, 75):
        lo = np.clip(idx - w, 0, n)
        hi = np.clip(idx + w + 1, 0, n)
        out["d_ttk_past{}".format(w)] = c[idx + 1] - c[lo]
        out["d_ttk_next{}".format(w)] = c[hi] - c[idx]
    strong = (tt != 0) & (np.nan_to_num(k8, nan=0.0) >= STRONG_KINK)
    out["d_st_last"] = np.nan_to_num(F.ffill_values(tt, strong), nan=0.0)
    out["d_st_next"] = np.nan_to_num(F.bfill_values(tt, strong), nan=0.0)
    for sgn, tag in ((1.0, "h"), (-1.0, "a")):
        m = strong & (tt == sgn)
        out["d_st_since_" + tag] = F.since_last(m)
        out["d_st_until_" + tag] = F.until_next(m)
    return {k: np.asarray(v, dtype=np.float64) for k, v in out.items()}
