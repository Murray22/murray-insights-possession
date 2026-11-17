"""Phase B2 features for long ball gaps: player-only evidence and the ball at both ends of a gap.

The real ATD loses the ball for long stretches (median 268 frames, 41% of graded time), so most
of the match has no ball to read. Two kinds of evidence survive a gap:

1. Players. The team in possession advances and spreads; the other team retreats, compacts and
   presses. Everything here is relative to each team's attacking direction (inferred per half from
   the kick-off layout, features.attack_sign), never absolute, so nothing encodes the half or the
   game: centroid advance and its velocity, forward running, team length / width / compactness,
   share of the team in the opponent half, nearest-opponent distance (pressing), and smoothed
   versions over 2 and 5 s.
2. The ball at both ends of the gap. Labelling is offline, so the frame where the ball reappears is
   as available as the one where it vanished: where (relative to home's attack), how fast, how far
   from the lines, which team was nearest, and how far the ball travelled across the gap.

Returned as extra columns next to features.compute(); names are prefixed "b2_".
"""

import warnings

import numpy as np

from . import features as F
from .frames import AWAY, HOME

EXTRA_VERSION = 1


def _team_arrays(table):
    L, W = table.pitch
    xy = table.xy.astype(np.float64) * np.array([L, W])
    return xy[:, :, 0], xy[:, :, 1], L, W


def _vel(x, lag=10):
    """Centred difference per frame (units per second), NaN-padded."""
    h = lag // 2
    out = np.full_like(x, np.nan)
    out[h:len(x) - h] = (x[2 * h:] - x[:-2 * h]) * F.FPS / (2 * h)
    return out


def compute_extra(table, f):
    """dict of extra features; `f` is features.compute(table) (reads attack sign, ball columns)."""
    x, y, L, W = _team_arrays(table)
    team = table.slot_team
    s = f["home_attack_sign"]
    n = table.n_frames
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        vx = np.full_like(x, np.nan)
        vx[5:-5] = (x[10:] - x[:-10]) * F.FPS / 10.0
        vx = np.clip(vx, -12, 12)
        cents = {}
        for code, tag, sign in ((HOME, "h", s), (AWAY, "a", -s)):
            tx, ty = x[:, team == code], y[:, team == code]
            cx = np.nanmean(tx, axis=1)
            cents[tag] = (cx, np.nanmean(ty, axis=1))
            out["b2_adv_" + tag] = (cx - L / 2) * sign                       # + = in opponent half
            out["b2_cvel_" + tag] = _vel(cx) * sign                          # centroid forward speed
            out["b2_fwd_" + tag] = np.nanmean(vx[:, team == code], axis=1) * sign
            out["b2_len_" + tag] = F.row_quantile(tx, 0.9) - F.row_quantile(tx, 0.1)
            out["b2_wid_" + tag] = F.row_quantile(ty, 0.9) - F.row_quantile(ty, 0.1)
            out["b2_comp_" + tag] = np.nanmean(np.hypot(tx - cx[:, None], ty - cents[tag][1][:, None]), axis=1)
            out["b2_opphalf_" + tag] = np.nanmean(
                np.where(np.isnan(tx), np.nan, ((tx - L / 2) * sign[:, None] > 0).astype(float)), axis=1)
            # deepest outfield-ish player: 2nd most backward, relative to own goal line
            back = np.sort(np.where(np.isnan(tx), np.nan, (tx - L / 2) * sign[:, None]), axis=1)
            out["b2_line_" + tag] = back[:, 1] if back.shape[1] > 1 else np.full(n, np.nan)
        # pressing: mean distance from each team's players to the nearest opponent
        ph = np.stack([x[:, team == HOME], y[:, team == HOME]], axis=2)
        pa = np.stack([x[:, team == AWAY], y[:, team == AWAY]], axis=2)
        d = np.full((n,), np.nan)
        dha = np.full((n,), np.nan)
        for c0 in range(0, n, 20000):
            a = ph[c0:c0 + 20000]
            b = pa[c0:c0 + 20000]
            dd = np.linalg.norm(a[:, :, None, :] - b[:, None, :, :], axis=3)
            d[c0:c0 + 20000] = np.nanmean(np.nanmin(dd, axis=2), axis=1)     # home to nearest away
            dha[c0:c0 + 20000] = np.nanmean(np.nanmin(dd, axis=1), axis=1)   # away to nearest home
        out["b2_press_h"] = d
        out["b2_press_a"] = dha
    for k in ("adv", "cvel", "fwd", "len", "wid", "comp", "opphalf", "line", "press"):
        out["b2_{}_d".format(k)] = out["b2_{}_h".format(k)] - out["b2_{}_a".format(k)]
    for k in ("cvel_d", "fwd_d", "adv_d", "opphalf_d", "press_d", "comp_d", "len_d", "line_d"):
        out["b2_{}_2s".format(k)] = F.rolling_nanmean(out["b2_" + k], 50)
        out["b2_{}_5s".format(k)] = F.rolling_nanmean(out["b2_" + k], 125)
    out["b2_cvel_d_past3s"] = F.window_nanmean(out["b2_cvel_d"], -75, 0)
    out["b2_cvel_d_next3s"] = F.window_nanmean(out["b2_cvel_d"], 0, 75)

    # the ball at both ends of a gap
    bvis = f["bvis"] > 0.5
    bx, by = f["bx"], f["by"]
    dline = np.minimum.reduce([np.abs(bx), np.abs(L - bx), np.abs(by), np.abs(W - by)])
    for side, fill in (("last", F.ffill_values), ("next", F.bfill_values)):
        out["b2_{}_by_c".format(side)] = np.abs(fill(by, bvis) - W / 2)
        out["b2_{}_dline".format(side)] = fill(dline, bvis)
        out["b2_{}_bspeed".format(side)] = fill(np.nan_to_num(f["bspeed_1s"], nan=0.0), bvis)
        out["b2_{}_dmin".format(side)] = fill(np.nan_to_num(f["dmin"], nan=30.0), bvis)
    lx, ly = F.ffill_values(bx, bvis), F.ffill_values(by, bvis)
    nx, ny = F.bfill_values(bx, bvis), F.bfill_values(by, bvis)
    gap = ~bvis
    out["b2_gap_disp"] = np.where(gap, np.hypot(nx - lx, ny - ly), 0.0)
    out["b2_gap_dx_att"] = np.where(gap, (nx - lx) * s, 0.0)
    gl = np.maximum(f["gap_len"], 1.0)
    out["b2_gap_frac"] = np.where(gap, f["since_vis"] / gl, 0.0)
    # uncapped gap length in seconds (features.gap_len caps at 30 s)
    since = _since_uncapped(bvis)
    until = _since_uncapped(bvis[::-1])[::-1]
    out["b2_gap_s"] = np.where(gap, (since + until) / F.FPS, 0.0)
    for k, v in out.items():
        out[k] = np.asarray(v, dtype=np.float64)
    return out


def _since_uncapped(mask):
    n = len(mask)
    idx = np.where(mask, np.arange(n), -1)
    last = np.maximum.accumulate(idx)
    out = (np.arange(n) - last).astype(np.float64)
    out[last < 0] = np.nan
    return out


def compute(table):
    """features.compute(table) plus the B2 extras."""
    f = F.compute(table)
    f.update(compute_extra(table, f))
    return f
