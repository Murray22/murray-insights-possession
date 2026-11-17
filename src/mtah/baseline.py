"""The organiser baseline (ref/starter/baseline.py) on our common FrameTable.

Same two steps: (1) on every frame where the ball is visible, the team of the nearest tracked slot
(distances in metres, so the non-square pitch does not stretch the short axis); NOBODY where the
ball is missing; (2) a centred majority vote over +/- window frames, ties to home, D where the
window holds no answer at all. Ported line for line from the organiser's nearest_player_labels()
and smooth(); test_baseline.py checks the smoothing against the organiser function itself.
"""

import numpy as np

from .frames import AWAY, HOME

NOBODY = -1
VARIANTS = (("baseline_raw", 0), ("baseline_smooth25", 25),
            ("baseline_smooth50", 50), ("baseline_smooth125", 125))


def nearest_player_codes(table):
    """(n,) int8: 0 home / 1 away for the nearest present slot to a visible ball, else NOBODY."""
    length, width = table.pitch
    scale = np.array([length, width], dtype=np.float64)
    ball = table.ball.astype(np.float64) * scale            # (n, 2)
    xy = table.xy.astype(np.float64) * scale                # (n, S, 2)
    d2 = ((xy - ball[:, None, :]) ** 2).sum(-1)             # (n, S), NaN where missing
    d2 = np.where(np.isnan(d2), np.inf, d2)
    nearest = np.argmin(d2, axis=1)
    best = d2[np.arange(len(d2)), nearest]
    out = table.slot_team[nearest].astype(np.int8)
    out[~np.isfinite(best)] = NOBODY                         # no ball, or no player at all
    return out


def smooth(codes, window):
    """Identical to ref/starter/baseline.py smooth(): returns an array of 'H' 'A' 'D'."""
    if window <= 0:
        return np.where(codes == HOME, "H", np.where(codes == AWAY, "A", "D"))
    home_cumulative = np.concatenate(([0], np.cumsum(codes == HOME)))
    away_cumulative = np.concatenate(([0], np.cumsum(codes == AWAY)))
    index = np.arange(len(codes))
    low = np.maximum(0, index - window)
    high = np.minimum(len(codes), index + window + 1)
    home = home_cumulative[high] - home_cumulative[low]
    away = away_cumulative[high] - away_cumulative[low]
    labels = np.full(len(codes), "D", dtype="<U1")
    seen = (home + away) > 0
    labels[seen] = np.where(home[seen] >= away[seen], "H", "A")
    return labels


def predict_all(table):
    """{variant name: labels} for the four published baselines."""
    raw = nearest_player_codes(table)
    return {name: smooth(raw, w) for name, w in VARIANTS}
