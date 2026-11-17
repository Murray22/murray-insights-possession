"""Per-frame features from a FrameTable, for the dead-ball detector, the segment model and the learned
emissions. Tracking only: ball position (or its absence), anonymous slot positions and slot teams.

Nothing here reads player_id, the frame's ground truth or anything outside the table, and every
feature uses the whole table (past and future), because labelling is offline.

Units: metres and metres per second (the table's normalised coordinates times its pitch). A NaN
means "not observable here"; the gradient-boosting model reads NaN natively and the hand-built
models treat it explicitly.
"""

import numpy as np

from .frames import AWAY, HOME

FPS = 25
FEATURE_VERSION = 4
CAP_GAP = 750            # frames; time-to-ball features are capped at 30 s
PLAYER_LAG = 10          # frames between the two positions of a player speed (0.4 s)
BALL_LAG = 4             # frames for ball speed (0.16 s)
SPEED_CLIP = 12.0        # m/s; a re-used slot can jump, so clip
BALL_SPEED_CLIP = 40.0


# -- small helpers ---------------------------------------------------------------------------------

def rolling_nanmean(x, half, weights=None):
    """Centred mean over +/- half frames ignoring NaN (NaN where the window has nothing)."""
    x = np.asarray(x, dtype=np.float64)
    ok = ~np.isnan(x)
    v = np.where(ok, x, 0.0)
    c = np.concatenate(([0.0], np.cumsum(v)))
    k = np.concatenate(([0], np.cumsum(ok)))
    n = len(x)
    i = np.arange(n)
    lo, hi = np.maximum(0, i - half), np.minimum(n, i + half + 1)
    cnt = k[hi] - k[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (c[hi] - c[lo]) / cnt
    out[cnt == 0] = np.nan
    return out


def window_nanmean(x, lo_off, hi_off):
    """Mean of x over [t + lo_off, t + hi_off] ignoring NaN (one-sided windows allowed)."""
    x = np.asarray(x, dtype=np.float64)
    ok = ~np.isnan(x)
    c = np.concatenate(([0.0], np.cumsum(np.where(ok, x, 0.0))))
    k = np.concatenate(([0], np.cumsum(ok)))
    n = len(x)
    i = np.arange(n)
    lo = np.clip(i + lo_off, 0, n)
    hi = np.clip(i + hi_off + 1, 0, n)
    cnt = k[hi] - k[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (c[hi] - c[lo]) / cnt
    out[cnt <= 0] = np.nan
    return out


def since_last(mask, cap=CAP_GAP):
    """Frames since the last True (0 on a True frame), capped."""
    n = len(mask)
    idx = np.where(mask, np.arange(n), -10 ** 9)
    last = np.maximum.accumulate(idx)
    return np.minimum(np.arange(n) - last, cap).astype(np.float64)


def until_next(mask, cap=CAP_GAP):
    return since_last(mask[::-1], cap)[::-1]


def ffill_values(values, mask):
    """values at the last frame where mask is True (NaN before the first)."""
    n = len(mask)
    idx = np.where(mask, np.arange(n), -1)
    last = np.maximum.accumulate(idx)
    out = np.full(n, np.nan)
    ok = last >= 0
    out[ok] = values[last[ok]]
    return out


def bfill_values(values, mask):
    return ffill_values(values[::-1], mask[::-1])[::-1]


def shift(x, k):
    """x[t - k] (k > 0: the past), NaN-padded."""
    out = np.full(len(x), np.nan)
    if k > 0:
        out[k:] = x[:-k]
    elif k < 0:
        out[:k] = x[-k:]
    else:
        out[:] = x
    return out


def standardise(x, mask):
    """(x - median) / IQR, both over the mask's non-NaN values (0-safe)."""
    v = x[mask & ~np.isnan(x)]
    if len(v) < 10:
        return np.zeros_like(x)
    q25, q50, q75 = np.percentile(v, [25, 50, 75])
    return (x - q50) / max(q75 - q25, 1e-6)


def range_scale(x, mask, lo=5, hi=75):
    """(x - q05) / (q75 - q05) over the mask: standing still plus noise to active play. Less
    dependent on how much of the game is dead than median / IQR (tests/test_models.py)."""
    v = x[mask & ~np.isnan(x)]
    if len(v) < 10:
        return np.full_like(x, np.nan)
    a, b = np.percentile(v, [lo, hi])
    return (x - a) / max(b - a, 1e-6)


def row_quantile(a, q):
    """Per-row quantile ignoring NaN (linear interpolation, like np.nanquantile), via one sort."""
    srt = np.sort(a, axis=1)                       # NaN sorts last
    cnt = (~np.isnan(a)).sum(1)
    pos = q * np.maximum(cnt - 1, 0)
    lo = np.floor(pos).astype(np.int64)
    hi = np.minimum(lo + 1, np.maximum(cnt - 1, 0))
    r = np.arange(len(a))
    w = pos - lo
    out = srt[r, lo] * (1 - w) + srt[r, hi] * w
    out[cnt == 0] = np.nan
    return out


def runs(mask):
    m = np.asarray(mask, dtype=np.int8)
    e = np.diff(np.concatenate(([0], m, [0])))
    return list(zip(np.flatnonzero(e == 1).tolist(), np.flatnonzero(e == -1).tolist()))


# -- the feature table -------------------------------------------------------------------------------

def compute(table):
    """dict name -> (n,) float64 array. See the module docstring."""
    L, W = table.pitch
    scale = np.array([L, W], dtype=np.float64)
    ball = table.ball.astype(np.float64) * scale
    xy = table.xy.astype(np.float64) * scale
    n, S = xy.shape[:2]
    team = table.slot_team
    present = ~np.isnan(xy[:, :, 0])
    bvis = ~np.isnan(ball[:, 0])
    f = {}

    # ball
    f["bvis"] = bvis.astype(np.float64)
    f["bx"], f["by"] = ball[:, 0], ball[:, 1]
    with np.errstate(invalid="ignore"):
        out = np.maximum.reduce([np.zeros(n), -ball[:, 0], ball[:, 0] - L, -ball[:, 1],
                                 ball[:, 1] - W])
    out[~bvis] = np.nan
    f["bout"] = out
    h = BALL_LAG // 2
    bsp = np.full(n, np.nan)
    d = np.linalg.norm(ball[2 * h:] - ball[:-2 * h], axis=1) * FPS / (2 * h)
    bsp[h:n - h] = d
    bsp = np.minimum(bsp, BALL_SPEED_CLIP)
    f["bspeed"] = bsp
    f["bspeed_1s"] = rolling_nanmean(bsp, 12)
    with np.errstate(invalid="ignore"):
        still = np.where(np.isnan(bsp), np.nan, (bsp < 0.7).astype(np.float64))
    f["bstill_2s"] = rolling_nanmean(still, 25)
    f["bvis_5s"] = rolling_nanmean(f["bvis"], 62)
    f["bvis_20s"] = rolling_nanmean(f["bvis"], 250)

    dist = np.linalg.norm(xy - ball[:, None, :], axis=2)               # (n, S), NaN if missing
    dist = np.where(np.isnan(dist), np.inf, dist)
    dh = dist[:, team == HOME].min(axis=1)
    da = dist[:, team == AWAY].min(axis=1)
    dh[~np.isfinite(dh)] = np.nan
    da[~np.isfinite(da)] = np.nan
    f["dh"], f["da"] = dh, da
    f["dmin"] = np.fmin(dh, da)
    f["ddiff"] = np.clip(da - dh, -15, 15)          # > 0: home closer
    f["nh3"] = np.where(bvis, (dist[:, team == HOME] < 3.0).sum(1), np.nan)
    f["na3"] = np.where(bvis, (dist[:, team == AWAY] < 3.0).sum(1), np.nan)
    f["since_vis"] = since_last(bvis)
    f["until_vis"] = until_next(bvis)
    f["gap_len"] = np.where(bvis, 0.0, np.minimum(f["since_vis"] + f["until_vis"], CAP_GAP))
    f["last_ddiff"] = ffill_values(f["ddiff"], bvis & ~np.isnan(f["ddiff"]))
    f["next_ddiff"] = bfill_values(f["ddiff"], bvis & ~np.isnan(f["ddiff"]))
    f["last_bx"] = ffill_values(ball[:, 0], bvis)
    f["last_by"] = ffill_values(ball[:, 1], bvis)
    f["next_bx"] = bfill_values(ball[:, 0], bvis)
    f["next_by"] = bfill_values(ball[:, 1], bvis)

    # players: speed per slot over PLAYER_LAG frames (both ends present), clipped
    h = PLAYER_LAG // 2
    spd = np.full((n, S), np.nan)
    dd = np.linalg.norm(xy[2 * h:] - xy[:-2 * h], axis=2) * FPS / (2 * h)
    spd[h:n - h] = dd
    spd = np.minimum(spd, SPEED_CLIP)
    f["pspd"] = row_quantile(spd, 0.5)
    f["pspd_h"] = row_quantile(spd[:, team == HOME], 0.5)
    f["pspd_a"] = row_quantile(spd[:, team == AWAY], 0.5)
    f["pspd_p80"] = row_quantile(spd, 0.8)
    cnt = (~np.isnan(spd)).sum(1)
    with np.errstate(invalid="ignore"):
        f["pslow"] = np.where(cnt > 0, (spd < 1.0).sum(1) / np.maximum(cnt, 1), np.nan)
        f["pfast"] = np.where(cnt > 0, (spd > 4.0).sum(1) / np.maximum(cnt, 1), np.nan)
    for name, half in (("1s", 12), ("3s", 37), ("8s", 100)):
        f["pspd_" + name] = rolling_nanmean(f["pspd"], half)
    f["pspd_p80_3s"] = rolling_nanmean(f["pspd_p80"], 37)
    f["pfast_3s"] = rolling_nanmean(f["pfast"], 37)
    f["pspd_past4s"] = window_nanmean(f["pspd"], -100, 0)
    f["pspd_next4s"] = window_nanmean(f["pspd"], 0, 100)
    f["pspd_past10s"] = window_nanmean(f["pspd"], -250, 0)
    f["pspd_next10s"] = window_nanmean(f["pspd"], 0, 250)
    f["n_present"] = present.sum(1).astype(np.float64)

    x, y = xy[:, :, 0], xy[:, :, 1]
    with np.errstate(invalid="ignore"):
        off = (x < -0.5) | (x > L + 0.5) | (y < -0.5) | (y > W + 0.5)
        f["n_off"] = (off & present).sum(1).astype(np.float64)
        f["n_touchline"] = (((y < 1.0) | (y > W - 1.0)) & present).sum(1).astype(np.float64)
        cm = np.full(n, np.inf)
        for cx, cy in ((0, 0), (0, W), (L, 0), (L, W)):
            dc = np.hypot(x - cx, y - cy)
            cm = np.minimum(cm, np.where(present, dc, np.inf).min(axis=1))
        f["corner_min"] = np.where(np.isfinite(cm), cm, np.nan)
        f["n_centre"] = (((x - L / 2) ** 2 + (y - W / 2) ** 2 < 9.15 ** 2) & present).sum(1).astype(float)
        in_box_l = (x < 16.5) & (np.abs(y - W / 2) < 20.16)
        in_box_r = (x > L - 16.5) & (np.abs(y - W / 2) < 20.16)
        f["n_box"] = np.maximum((in_box_l & present).sum(1), (in_box_r & present).sum(1)).astype(float)

    # team shape
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for code, tag in ((HOME, "h"), (AWAY, "a")):
            tx, ty = x[:, team == code], y[:, team == code]
            f["cx_" + tag] = np.nanmean(tx, axis=1)
            f["cy_" + tag] = np.nanmean(ty, axis=1)
            f["sx_" + tag] = np.nanstd(tx, axis=1)
            f["sy_" + tag] = np.nanstd(ty, axis=1)
            f["left_" + tag] = np.nanmean(np.where(np.isnan(tx), np.nan, (tx < L / 2).astype(float)), axis=1)
    f["dcx"] = f["cx_h"] - f["cx_a"]
    f["spread_diff"] = (f["sx_h"] * f["sy_h"]) - (f["sx_a"] * f["sy_a"])
    f["separation"] = np.fmax(f["left_h"] * (1 - f["left_a"]), (1 - f["left_h"]) * f["left_a"])
    f["separation_1s"] = rolling_nanmean(f["separation"], 12)
    # direction-free "who is higher up": positive when home's centroid is closer to the away goal.
    # The attacking direction is not in the table, so it is inferred per half from the kick-off
    # layout: the team whose centroid is on the left at the start of a half attacks right.
    f["home_attack_sign"] = attack_sign(table, f)
    f["home_adv"] = f["dcx"] * f["home_attack_sign"]
    # touch signal: +1 / -1 where the ball is within r of a home / away player (and nearer to that
    # team), 0 elsewhere; windowed sums and the last / next touch, for the learned emissions
    for tag, r in (("", 1.0), ("15", 1.5)):
        dd = np.nan_to_num(f["ddiff"], nan=0.0)
        tt = np.where(np.nan_to_num(f["dmin"], nan=np.inf) < r, np.sign(dd), 0.0)
        f["tt" + tag] = tt
        c = np.concatenate(([0.0], np.cumsum(tt)))
        idx = np.arange(n)
        for k in (12, 25, 75, 250):
            lo = np.clip(idx - k, 0, n)
            hi = np.clip(idx + k + 1, 0, n)
            f["tt{}_past{}".format(tag, k)] = c[idx + 1] - c[lo]
            f["tt{}_next{}".format(tag, k)] = c[hi] - c[idx]
        touch = tt != 0
        f["tt{}_last".format(tag)] = np.nan_to_num(ffill_values(tt, touch), nan=0.0)
        f["tt{}_next".format(tag)] = np.nan_to_num(bfill_values(tt, touch), nan=0.0)
        f["tt{}_since".format(tag)] = since_last(touch)
        f["tt{}_until".format(tag)] = until_next(touch)
    for k in (12, 25):
        f["ddiff_m{}".format(k)] = shift(f["ddiff"], k)
        f["ddiff_p{}".format(k)] = shift(f["ddiff"], -k)
        f["bspeed_m{}".format(k)] = shift(f["bspeed_1s"], k)
        f["bspeed_p{}".format(k)] = shift(f["bspeed_1s"], -k)
    # ball position relative to home's attacking direction (x towards the away goal)
    f["bx_att"] = (f["bx"] - L / 2) * f["home_attack_sign"]
    f["by_c"] = np.abs(f["by"] - W / 2)
    f["last_bx_att"] = (f["last_bx"] - L / 2) * f["home_attack_sign"]
    f["next_bx_att"] = (f["next_bx"] - L / 2) * f["home_attack_sign"]
    if table.period is not None:
        f["period"] = table.period.astype(np.float64)
    # per-game standardised motion: (x - median) / IQR over graded frames. Removes the level that
    # position noise adds to every speed (a 0.4 m jitter lifts the 80th percentile by ~0.9 m/s)
    graded = table.period > 0 if table.period is not None else np.ones(n, dtype=bool)
    for k in ("pspd", "pspd_1s", "pspd_3s", "pspd_8s", "pspd_p80_3s", "pspd_past4s",
              "pspd_next4s", "pspd_past10s", "pspd_next10s", "pfast_3s"):
        f[k + "_z"] = standardise(f[k], graded)
        f[k + "_r"] = range_scale(f[k], graded)
    for k, v in f.items():
        f[k] = np.asarray(v, dtype=np.float64)
    return f


def attack_sign(table, f=None):
    """(n,) +1 where home attacks towards x = L (right), -1 where it attacks left, per half.

    Phase B2: read from the whole half, not from its first seconds. Each team's deepest and
    furthest-forward players (goalkeepers and the last line) sit towards the goal it defends, so
    the per-frame sum (away min x - home min x) + (away max x - home max x) is positive while home
    defends the left goal; its median over the half decides. On the real ATD the second half's
    first 10 s are not a kick-off layout (15 tracked objects, home centroid 51 m, away 78 m) and the
    Phase C rule (who stands on the left in the first 10 s) gave the second half the wrong
    direction; the median agrees with that rule on every sample-game half (docs/phaseB2_report.md).
    Falls back to the first-10-s layout when the median is undefined.
    """
    L, W = table.pitch
    x = table.xy[:, :, 0].astype(np.float64) * L
    team = table.slot_team
    out = np.zeros(table.n_frames)
    import warnings
    for k in np.unique(table.period):
        if k <= 0:
            continue
        idx = np.flatnonzero(table.period == k)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            xh, xa = x[idx][:, team == HOME], x[idx][:, team == AWAY]
            stat = (np.nanmin(xa, 1) - np.nanmin(xh, 1)) + (np.nanmax(xa, 1) - np.nanmax(xh, 1))
            med = np.nanmedian(stat) if np.isfinite(stat).any() else np.nan
            if np.isfinite(med) and med != 0:
                out[idx] = 1.0 if med > 0 else -1.0
                continue
            head = idx[:250]
            ch = np.nanmean(x[head][:, team == HOME])
            ca = np.nanmean(x[head][:, team == AWAY])
            if np.isnan(ch) or np.isnan(ca):
                ch = np.nanmean(x[idx][:, team == HOME])
                ca = np.nanmean(x[idx][:, team == AWAY])
        out[idx] = 1.0 if ch < ca else -1.0
    return out


def matrix(feats, names=None):
    """(n, k) float32 design matrix in a fixed column order."""
    names = names or sorted(feats)
    return np.stack([feats[k] for k in names], axis=1).astype(np.float32), list(names)


def cached(table, cache_dir=None):
    """compute(table), cached on disk for bench tables (meta variant / seed / game set)."""
    import os
    from . import paths
    m = table.meta
    if not all(k in m for k in ("variant", "seed", "game")) or os.environ.get("MTAH_FEATURE_CACHE") == "0":
        return compute(table)       # MTAH_FEATURE_CACHE=0: no cache (84 MB per bench table)
    d = cache_dir or os.path.join(paths.RUNS_DIR, "cache", "features")
    path = os.path.join(d, "{}_s{}_g{}_v{}.npz".format(m.get("variant_key", m["variant"]), m["seed"],
                                                        m["game"], FEATURE_VERSION))
    if os.path.isfile(path):
        z = np.load(path)
        return {k: z[k] for k in z.files}
    f = compute(table)
    os.makedirs(d, exist_ok=True)
    tmp = "{}.{}.tmp.npz".format(path, os.getpid())   # per process: two bench workers can build
    np.savez(tmp, **{k: v.astype(np.float32) for k, v in f.items()})   # the same table at once
    os.replace(tmp, path)
    z = np.load(path)
    return {k: z[k] for k in z.files}
