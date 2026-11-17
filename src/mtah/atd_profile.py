"""Distribution profile of a tracking table: the statistics the calibrated degrader has to reproduce.

    PYTHONPATH=src .venv/bin/python -m mtah.atd_profile --variants default atd_cal --seeds 2

One function, `profile(table)`, runs on the real ATD and on degraded sample games alike, so the
two can be put side by side (`compare`). Everything is a distribution or a share over graded
frames: no frame of BAR-ATH is singled out, and no label is read. Smart Tagging (automatic, an
allowed input) is used only to measure ball visibility around tagged set pieces, which is the one
label-free view we have of how the tracker behaves at stoppages.

Units: metres, metres per second, frames at 25 fps.

What is measured
- ball: visible share; missing-run and visible-run lengths; the context at the last frame before
  each gap (ball speed, nearest player, players within 3 m, distance to a line, distance from the
  halfway line) against the same quantities over all visible frames, as onset hazard ratios per
  bin; where the ball reappears relative to where it vanished; implausible jumps; jitter.
- players: tracked objects per frame (all, per team); tracklet lengths per slot; gaps between a
  slot's tracklets; re-acquisition (a new tracklet of the same team starting near where one ended);
  near-duplicates (two same-team slots within 0.5 m); jitter; teleports.
- set pieces (when intervals are given): the share of missing-ball frames inside the tagged
  windows, the missing share inside versus outside them, and the visible share in one-second bins
  around each tag's start.
"""

import argparse
import json
import os
import time
import uuid
import warnings

import numpy as np

from . import paths
from .features import runs as _runs

FPS = 25
Q = (5, 10, 25, 50, 75, 90, 95, 99)


def _q(x, qs=Q):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if not len(x):
        return {"n": 0}
    out = {"n": int(len(x)), "mean": float(x.mean())}
    out.update({"p{}".format(q): float(v) for q, v in zip(qs, np.percentile(x, qs))})
    return out


def _graded_blocks(table):
    """[(start, stop)] of contiguous graded runs (period > 0)."""
    return _runs(table.period > 0)


def _block_runs(mask, blocks):
    """Runs of mask inside each block, with whether the run touches the block edge."""
    out = []
    for a, b in blocks:
        for s, e in _runs(mask[a:b]):
            out.append((a + s, a + e, s == 0 or a + e == b))
    return out


def _metric(table):
    L, W = table.pitch
    scale = np.array([L, W], dtype=np.float64)
    return table.ball.astype(np.float64) * scale, table.xy.astype(np.float64) * scale, L, W


# -- ball --------------------------------------------------------------------------------------------

BINS = {
    "speed": [0, 2, 5, 10, 15, 20, 30, 1e9],          # m/s, ball speed over the last 4 frames
    "dmin": [0, 1, 2, 4, 8, 15, 1e9],                 # m, nearest player
    "n3": [0, 1, 2, 3, 4, 6, 99],                     # players within 3 m
    "dline": [0, 3, 8, 15, 1e9],                      # m, nearest touchline / goal line
    "dhalf": [0, 15, 30, 45, 1e9],                    # m, |x - L / 2|
}


def ball_context(ball, xy, L, W):
    """Per-frame context of the visible ball (NaN where not computable)."""
    n = len(ball)
    sp = np.full(n, np.nan)
    sp[4:] = np.linalg.norm(ball[4:] - ball[:-4], axis=1) * FPS / 4.0
    d = np.linalg.norm(xy - ball[:, None, :], axis=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        dmin = np.nanmin(d, axis=1)
    n3 = (np.nan_to_num(d, nan=np.inf) < 3.0).sum(1).astype(np.float64)
    bx, by = ball[:, 0], ball[:, 1]
    dline = np.minimum.reduce([np.abs(bx), np.abs(L - bx), np.abs(by), np.abs(W - by)])
    dhalf = np.abs(bx - L / 2)
    vis = ~np.isnan(bx)
    n3[~vis] = np.nan
    return {"speed": sp, "dmin": dmin, "n3": n3, "dline": dline, "dhalf": dhalf}


def ball_profile(table):
    ball, xy, L, W = _metric(table)
    graded = table.period > 0
    vis = ~np.isnan(ball[:, 0])
    blocks = _graded_blocks(table)
    gaps = _block_runs(~vis, blocks)
    visr = _block_runs(vis, blocks)
    n_graded = int(graded.sum())
    out = {"visible_share": float(vis[graded].mean()), "graded_frames": n_graded,
           "gap_len": _q([e - s for s, e, _ in gaps]),
           "gap_runs_per_hour": len(gaps) / (n_graded / FPS / 3600.0),
           "visible_len": _q([e - s for s, e, _ in visr])}
    lens = np.array([e - s for s, e, _ in gaps])
    if len(lens):
        out["missing_in_runs_ge25"] = float(lens[lens >= 25].sum() / lens.sum())
        out["missing_in_runs_ge250"] = float(lens[lens >= 250].sum() / lens.sum())
        out["gap_share_lt25"] = float((lens < 25).mean())
    ctx = ball_context(ball, xy, L, W)
    # onset: the last visible frame before an interior gap
    onset = np.array([s - 1 for s, e, edge in gaps if not edge], dtype=np.int64)
    reapp = np.array([e for s, e, edge in gaps if not edge], dtype=np.int64)
    base = vis & graded
    haz = {}
    for k, edges in BINS.items():
        v = ctx[k]
        hb = np.histogram(v[base & np.isfinite(v)], edges)[0].astype(np.float64)
        ho = np.histogram(v[onset][np.isfinite(v[onset])], edges)[0].astype(np.float64)
        rate = ho.sum() / max(1.0, hb.sum())
        with np.errstate(invalid="ignore", divide="ignore"):
            hr = (ho / hb) / rate
        haz[k] = {"edges": [float(x) for x in edges], "visible_share": (hb / max(1, hb.sum())).tolist(),
                  "onset_share": (ho / max(1, ho.sum())).tolist(),
                  "hazard_ratio": [None if not np.isfinite(x) else float(x) for x in hr]}
        out["onset_" + k] = _q(v[onset])
        out["visible_" + k] = _q(v[base])
    out["onset_hazard"] = haz
    if len(onset):
        disp = np.linalg.norm(ball[reapp] - ball[onset], axis=1)
        glen = reapp - onset
        out["reappear_disp_m"] = _q(disp)
        out["reappear_disp_short_m"] = _q(disp[glen < 100])
        out["reappear_disp_long_m"] = _q(disp[glen >= 500])
        out["reappear_speed"] = _q(ctx["speed"][np.minimum(reapp + 4, len(ball) - 1)])
        dl = ctx["dline"]
        out["onset_out_of_pitch"] = float(np.mean(_outside(ball[onset], L, W)))
        out["reappear_out_of_pitch"] = float(np.mean(_outside(ball[reapp], L, W)))
        out["onset_near_line_3m"] = float(np.mean(dl[onset] < 3))
    out["visible_out_of_pitch"] = float(np.mean(_outside(ball[base], L, W)))
    # implausible jumps between consecutive visible frames (> 1.6 m per frame = 40 m/s)
    step = np.linalg.norm(ball[1:] - ball[:-1], axis=1)
    both = base[1:] & base[:-1]
    out["ball_jumps_per_min"] = float((step[both] > 1.6).sum() / (n_graded / FPS / 60.0))
    out["ball_step_m"] = _q(step[both])
    out["ball_jitter_m"] = _jitter(ball[:, None, :], graded)
    out.update(ball_shape_profile(table))
    return out


# -- ball-track shape (Phase E): how the ball's velocity bends -------------------------------------

def ball_shape_profile(table):
    """Velocity-change ("kink") distributions of the visible ball on graded frames.

    kink_k = |v(t, t + k) - v(t - k, t)| (m/s, features_d.kink). Its low percentiles measure the
    track's wobble between touches, its upper tail how sharply a kick or a deflection shows up
    after the tracker's smoothing. A strong touch is a run of frames with the ball within 1.5 m of
    a player and kink4 >= 2 m/s (features_d's touch sign), counted per graded minute, as the
    Phase D review addendum counted them. The real ATD and atd_cal differ here (Phase D addendum)
    though their jitter and speed rows looked close.
    """
    from . import features_d as D
    graded = table.period > 0
    out = {}
    k4 = None
    for k in (2, 4, 8):
        kk = D.kink(table, k)
        out["kink{}".format(k)] = _q(kk[graded])
        if k == 4:
            k4 = kk
    tt = D.touch_sign(table)
    touch = graded & (tt != 0)
    out["kink4_touch"] = _q(k4[touch])
    strong = touch & (np.nan_to_num(k4, nan=0.0) >= 2.0)
    minutes = graded.sum() / FPS / 60.0
    out["strong_touch_runs_per_min"] = float(len(_block_runs(strong, _graded_blocks(table))) / max(1e-9, minutes))
    return out


def _outside(p, L, W):
    return (p[:, 0] < -1) | (p[:, 0] > L + 1) | (p[:, 1] < -1) | (p[:, 1] > W + 1)


def _jitter(xyz, graded, half=4):
    """sd (m) of the residual from a centred (2 half + 1)-frame mean, where the window is full."""
    res = []
    k = 2 * half + 1
    for j in range(xyz.shape[1]):
        x = xyz[:, j, 0]
        ok = ~np.isnan(x) & graded
        if ok.sum() < k:
            continue
        for c in (0, 1):
            v = np.where(ok, xyz[:, j, c], 0.0)
            cs = np.concatenate(([0.0], np.cumsum(v)))
            cnt = np.concatenate(([0], np.cumsum(ok)))
            i = np.arange(half, len(x) - half)
            full = (cnt[i + half + 1] - cnt[i - half]) == k
            m = (cs[i + half + 1] - cs[i - half]) / k
            r = (xyz[i, j, c] - m)[full]
            res.append(r)
    if not res:
        return None
    r = np.concatenate(res)
    return float(np.sqrt(np.mean(r * r)))


# -- players -----------------------------------------------------------------------------------------

def player_profile(table, reacq_window=75, reacq_radius=3.0):
    ball, xy, L, W = _metric(table)
    graded = table.period > 0
    present = ~np.isnan(xy[:, :, 0])
    team = table.slot_team
    blocks = _graded_blocks(table)
    n_graded = int(graded.sum())
    minutes = n_graded / FPS / 60.0
    cnt = present[graded].sum(1)
    out = {"objects": _q(cnt), "objects_home_mean": float(present[graded][:, team == 0].sum(1).mean()),
           "objects_away_mean": float(present[graded][:, team == 1].sum(1).mean()),
           "frames_over_22": float((cnt > 22).mean()),
           "frames_team_over_11": float(((present[graded][:, team == 0].sum(1) > 11)
                                         | (present[graded][:, team == 1].sum(1) > 11)).mean()),
           "slots_used": int(present[graded].any(0).sum())}
    tl, sg = [], []
    ends, starts = {0: [], 1: []}, {0: [], 1: []}
    for j in range(xy.shape[1]):
        rr = _block_runs(present[:, j], blocks)
        for s, e, edge in rr:
            tl.append(e - s)
            if s > 0 and graded[s - 1]:
                starts[int(team[j])].append((s, xy[s, j, 0], xy[s, j, 1]))
            if e < len(xy) and graded[min(e, len(xy) - 1)]:
                ends[int(team[j])].append((e, xy[e - 1, j, 0], xy[e - 1, j, 1]))
        for (s0, e0, _), (s1, e1, _) in zip(rr[:-1], rr[1:]):
            if any(a <= s0 < b and a <= s1 < b for a, b in blocks):
                sg.append(s1 - e0)
    out["tracklet_len"] = _q(tl)
    out["tracklets_per_min"] = len(tl) / minutes
    out["slot_gap_len"] = _q(sg)
    # re-acquisition: for each tracklet end, the first new tracklet start of the same team within
    # reacq_window frames and reacq_radius metres of where it ended
    delays, found = [], 0
    n_end = 0
    for tm in (0, 1):
        st = np.array(starts[tm]) if starts[tm] else np.zeros((0, 3))
        if not len(st):
            continue
        order = np.argsort(st[:, 0])
        st = st[order]
        for e, x, y in ends[tm]:
            n_end += 1
            lo = np.searchsorted(st[:, 0], e - 1, side="left")
            hi = np.searchsorted(st[:, 0], e + reacq_window, side="right")
            c = st[lo:hi]
            if not len(c):
                continue
            d = np.hypot(c[:, 1] - x, c[:, 2] - y)
            ok = np.flatnonzero(d < reacq_radius)
            if len(ok):
                found += 1
                delays.append(c[ok[0], 0] - e)
    out["reacq_share"] = found / max(1, n_end)
    out["reacq_delay"] = _q(delays)
    out["reacq_within_3"] = float(np.mean(np.array(delays) <= 3) * found / max(1, n_end)) if delays else 0.0
    # near-duplicates: two present same-team slots within 0.5 m
    dup = np.zeros(len(xy), dtype=bool)
    gi = np.flatnonzero(graded)
    for tm in (0, 1):
        idx = np.flatnonzero(team == tm)
        for c0 in range(0, len(gi), 20000):
            rows = gi[c0:c0 + 20000]
            p = xy[rows][:, idx]
            d = np.linalg.norm(p[:, :, None, :] - p[:, None, :, :], axis=3)
            iu = np.triu_indices(len(idx), 1)
            close = np.nan_to_num(d[:, iu[0], iu[1]], nan=np.inf) < 0.5
            dup[rows] |= close.any(1)
    out["dup_frames_share"] = float(dup[graded].mean())
    out["dup_bursts_per_min"] = len(_block_runs(dup, blocks)) / minutes
    # motion
    step = np.linalg.norm(xy[1:] - xy[:-1], axis=2)
    both = present[1:] & present[:-1] & graded[1:, None]
    out["player_step_m"] = _q(step[both][::7])
    out["teleports_per_min"] = float((step[both] > 1.0).sum() / minutes)
    out["player_jitter_m"] = _jitter(xy, graded)
    return out


# -- Smart Tagging set pieces ------------------------------------------------------------------------

def setpiece_profile(table, intervals, before_s=30, after_s=40):
    graded = table.period > 0
    vis = table.ball_visible()
    sp = [iv for iv in intervals if iv.get("kind") == "set_piece"]
    inside = np.zeros(table.n_frames, dtype=bool)
    for iv in sp:
        inside[max(0, iv["start"]):iv["end"]] = True
    miss = ~vis & graded
    out = {"n_set_pieces": len(sp),
           "window_frames_share_of_graded": float(inside[graded].mean()),
           "missing_frames_in_windows": float((miss & inside).sum() / max(1, miss.sum())),
           "missing_share_inside": float(miss[inside & graded].mean()) if (inside & graded).any() else None,
           "missing_share_outside": float(miss[~inside & graded].mean())}
    # visible share and player motion in 1 s bins around tag start (mean over tags)
    _, xy, _, _ = _metric(table)
    spd = np.full(table.n_frames, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        st = np.linalg.norm(xy[10:] - xy[:-10], axis=2) * FPS / 10.0
        spd[5:-5] = np.nanpercentile(np.minimum(st, 12.0), 80, axis=1)
    bins = np.arange(-before_s, after_s)
    vprof = np.full((len(sp), len(bins)), np.nan)
    mprof = np.full((len(sp), len(bins)), np.nan)
    for i, iv in enumerate(sp):
        for k, b in enumerate(bins):
            a = iv["start"] + b * FPS
            if a < 0 or a + FPS > table.n_frames or not graded[a:a + FPS].all():
                continue
            vprof[i, k] = vis[a:a + FPS].mean()
            mprof[i, k] = np.nanmean(spd[a:a + FPS])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out["visible_by_second"] = dict(zip(bins.tolist(), np.nanmean(vprof, 0).round(3).tolist()))
        out["p80_speed_by_second"] = dict(zip(bins.tolist(), np.nanmean(mprof, 0).round(2).tolist()))
    return out


def profile(table, intervals=None):
    out = {"ball": ball_profile(table), "players": player_profile(table)}
    if intervals:
        out["set_pieces"] = setpiece_profile(table, intervals)
    return out


def real_gap_lengths(cache=True):
    """The real ATD's missing-ball run lengths on graded frames (atd_io.ball_gap_lengths), cached
    in runs/cache. A distribution (one number per run, no positions), for the degrader."""
    from . import atd_io
    path = os.path.join(paths.RUNS_DIR, "cache", "atd_gap_lengths.npy")
    if cache and os.path.isfile(path):
        return np.load(path)
    out = atd_io.ball_gap_lengths(atd_io.load_competition())
    if cache:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # one temporary file per writer: training workers can build this cache at the same moment
        tmp = "{}.{}.{}.tmp.npy".format(path[:-4], os.getpid(), uuid.uuid4().hex)
        np.save(tmp, out)
        os.replace(tmp, path)
    return out


# -- side-by-side table ------------------------------------------------------------------------------

ROWS = [
    ("ball visible share (graded)", ("ball", "visible_share"), "{:.3f}"),
    ("missing runs per hour", ("ball", "gap_runs_per_hour"), "{:.0f}"),
    ("missing run p10 / p50 / p90 (frames)", ("ball", "gap_len", ("p10", "p50", "p90")), "{:.0f}"),
    ("missing run p99 / mean", ("ball", "gap_len", ("p99", "mean")), "{:.0f}"),
    ("missing frames in runs >= 25", ("ball", "missing_in_runs_ge25"), "{:.3f}"),
    ("missing frames in runs >= 250", ("ball", "missing_in_runs_ge250"), "{:.3f}"),
    ("visible run p10 / p50 / p90", ("ball", "visible_len", ("p10", "p50", "p90")), "{:.0f}"),
    ("onset ball speed p25 / p50 / p75 (m/s)", ("ball", "onset_speed", ("p25", "p50", "p75")), "{:.1f}"),
    ("visible ball speed p25 / p50 / p75", ("ball", "visible_speed", ("p25", "p50", "p75")), "{:.1f}"),
    ("onset nearest player p25 / p50 / p75 (m)", ("ball", "onset_dmin", ("p25", "p50", "p75")), "{:.1f}"),
    ("visible nearest player p25 / p50 / p75", ("ball", "visible_dmin", ("p25", "p50", "p75")), "{:.1f}"),
    ("onset players within 3 m, mean", ("ball", "onset_n3", ("mean",)), "{:.2f}"),
    ("visible players within 3 m, mean", ("ball", "visible_n3", ("mean",)), "{:.2f}"),
    ("onset distance to line p25 / p50", ("ball", "onset_dline", ("p25", "p50")), "{:.1f}"),
    ("visible distance to line p25 / p50", ("ball", "visible_dline", ("p25", "p50")), "{:.1f}"),
    ("onset within 3 m of a line", ("ball", "onset_near_line_3m"), "{:.3f}"),
    ("reappear displacement p25 / p50 / p75 (m)", ("ball", "reappear_disp_m", ("p25", "p50", "p75")), "{:.1f}"),
    ("  gaps >= 500 frames: p50", ("ball", "reappear_disp_long_m", ("p50",)), "{:.1f}"),
    ("ball visible outside the pitch", ("ball", "visible_out_of_pitch"), "{:.4f}"),
    ("ball jumps > 40 m/s per min", ("ball", "ball_jumps_per_min"), "{:.2f}"),
    ("ball jitter (m, 9-frame residual)", ("ball", "ball_jitter_m"), "{:.3f}"),
    ("kink2 p25 / p50 / p75 / p90 / p99 (m/s)", ("ball", "kink2", ("p25", "p50", "p75", "p90", "p99")), "{:.2f}"),
    ("kink4 p25 / p50 / p75 / p90 / p99 (m/s)", ("ball", "kink4", ("p25", "p50", "p75", "p90", "p99")), "{:.2f}"),
    ("kink8 p25 / p50 / p75 / p90 / p99 (m/s)", ("ball", "kink8", ("p25", "p50", "p75", "p90", "p99")), "{:.2f}"),
    ("kink4 on touch frames p50 / p90", ("ball", "kink4_touch", ("p50", "p90")), "{:.2f}"),
    ("strong touches (kink4 >= 2) per min", ("ball", "strong_touch_runs_per_min"), "{:.1f}"),
    ("objects per frame mean / p5 / p50 / p95", ("players", "objects", ("mean", "p5", "p50", "p95")), "{:.1f}"),
    ("frames over 22 objects", ("players", "frames_over_22"), "{:.4f}"),
    ("tracklets per minute", ("players", "tracklets_per_min"), "{:.1f}"),
    ("tracklet length p25 / p50 / p75 / p90", ("players", "tracklet_len", ("p25", "p50", "p75", "p90")), "{:.0f}"),
    ("slot gap p25 / p50 / p75", ("players", "slot_gap_len", ("p25", "p50", "p75")), "{:.0f}"),
    ("re-acquired within 3 s, 3 m", ("players", "reacq_share"), "{:.3f}"),
    ("re-acquisition delay p25 / p50 / p75", ("players", "reacq_delay", ("p25", "p50", "p75")), "{:.0f}"),
    ("near-duplicate frames (same team < 0.5 m)", ("players", "dup_frames_share"), "{:.4f}"),
    ("player teleports > 1 m/frame per min", ("players", "teleports_per_min"), "{:.2f}"),
    ("player jitter (m)", ("players", "player_jitter_m"), "{:.3f}"),
    ("player step p50 / p95 (m/frame)", ("players", "player_step_m", ("p50", "p95")), "{:.3f}"),
]


# The calibration gate: the statistics a calibrated degrader must reproduce, with tolerances.
# (path, kind, tolerance): kind "abs" = |deg - real| <= tol; "rel" = |deg / real - 1| <= tol;
# "ratio" = deg / real <= tol (an upper bound only, for jitter).
CAL_RULES = [
    (("ball", "visible_share"), "abs", 0.02),
    (("ball", "gap_len", "p50"), "rel", 0.25),
    (("ball", "gap_len", "p90"), "rel", 0.25),
    (("ball", "missing_in_runs_ge25"), "abs", 0.02),
    (("ball", "missing_in_runs_ge250"), "abs", 0.08),
    (("ball", "visible_len", "p50"), "rel", 0.5),
    (("ball", "onset_dline", "p50"), "rel", 0.4),
    (("ball", "ball_jitter_m"), "ratio", 3.0),
    (("players", "objects", "mean"), "abs", 1.0),
    (("players", "objects", "p5"), "abs", 3.0),
    (("players", "tracklet_len", "p50"), "rel", 0.5),
    (("players", "player_jitter_m"), "ratio", 3.0),
]
# Phase E: the shape of the ball track (Phase D review addendum). atd_cal passes CAL_RULES_B2 and
# fails these: its ball is too still between touches and too sharp at them.
CAL_RULES_B2 = list(CAL_RULES)
CAL_RULES_SHAPE = [
    (("ball", "kink4", "p25"), "rel", 0.3),
    (("ball", "kink4", "p50"), "rel", 0.3),
    (("ball", "kink4", "p90"), "rel", 0.3),
    (("ball", "kink4", "p99"), "rel", 0.3),
    (("ball", "kink8", "p50"), "rel", 0.3),
    (("ball", "kink4_touch", "p90"), "rel", 0.3),
    (("ball", "strong_touch_runs_per_min"), "rel", 0.3),
    (("ball", "visible_speed", "p25"), "rel", 0.3),
]
CAL_RULES = CAL_RULES_B2 + CAL_RULES_SHAPE


def check_calibration(real, deg, rules=CAL_RULES):
    """[(statistic, real, degraded, rule)] for every rule the degraded profile breaks."""
    bad = []
    for path, kind, tol in rules:
        r = _get(real, path)[0] if len(path) == 2 else _get(real, (path[0], path[1], (path[2],)))[0]
        d = _get(deg, path)[0] if len(path) == 2 else _get(deg, (path[0], path[1], (path[2],)))[0]
        name = ".".join(path)
        if r is None or d is None:
            bad.append((name, r, d, "missing"))
            continue
        ok = {"abs": abs(d - r) <= tol,
              "rel": r != 0 and abs(d / r - 1.0) <= tol,
              "ratio": r > 0 and d / r <= tol}[kind]
        if not ok:
            bad.append((name, r, d, "{} {}".format(kind, tol)))
    return bad


def _get(prof, path):
    sec, key = path[0], path[1]
    v = prof.get(sec, {}).get(key)
    if len(path) == 2:
        return [v]
    return [None if not isinstance(v, dict) else v.get(k) for k in path[2]]


def compare(columns):
    """Markdown table: one row per statistic, one column per (name, profile)."""
    head = "| statistic | " + " | ".join(n for n, _ in columns) + " |"
    lines = [head, "|---|" + "---|" * len(columns)]
    for label, path, fmt in ROWS:
        cells = []
        for _, prof in columns:
            vals = _get(prof, path)
            cells.append(" / ".join("-" if v is None else fmt.format(v) for v in vals))
        lines.append("| {} | {} |".format(label, " | ".join(cells)))
    return "\n".join(lines)


def mean_profiles(profs):
    """Element-wise mean of numeric leaves (for averaging games / seeds)."""
    if not profs:
        return {}
    if isinstance(profs[0], dict):
        keys = set(profs[0])
        return {k: mean_profiles([p[k] for p in profs if isinstance(p, dict) and k in p]) for k in keys}
    if isinstance(profs[0], (int, float)) and not isinstance(profs[0], bool):
        vals = [p for p in profs if isinstance(p, (int, float))]
        return float(np.mean(vals)) if vals else None
    return profs[0]


def smart_tagging_path(source="tactical", root=None):
    from . import atd_io
    games = atd_io._games_module()
    p = games.asset_paths(games.GAME, atd_io.data_dir(root), source)
    for k, v in p.items():
        if "tag" in k:
            return v
    raise KeyError("no smart tagging path in games.asset_paths: {}".format(sorted(p)))


def main(argv=None):
    from . import atd_io, smart_tagging, modelbench
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--variants", nargs="*", default=["default"])
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--games", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--no-real", action="store_true")
    ap.add_argument("--out", default=os.path.join(paths.RUNS_DIR, "phaseB2"))
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    cols = []
    t0 = time.time()
    real_path = os.path.join(a.out, "profile_real.json")
    if not a.no_real:
        t = atd_io.load_competition()
        iv = smart_tagging.load(smart_tagging_path())
        real = profile(t, iv)
        with open(real_path, "w") as fh:
            json.dump(real, fh, indent=1)
        print("real profiled at {:.0f} s".format(time.time() - t0), flush=True)
    if os.path.isfile(real_path):
        with open(real_path) as fh:
            cols.append(("real ATD", json.load(fh)))
    for v in a.variants:
        profs = []
        for s in range(a.seeds):
            for g in a.games:
                t, _ = modelbench.degraded_table(v, s, g)
                profs.append(profile(t))
        m = mean_profiles(profs)
        with open(os.path.join(a.out, "profile_{}.json".format(v)), "w") as fh:
            json.dump(m, fh, indent=1)
        cols.append((v, m))
        print("{} profiled at {:.0f} s".format(v, time.time() - t0), flush=True)
    print(compare(cols))
    if os.path.isfile(real_path):
        for name, prof in cols[1:]:
            bad = check_calibration(cols[0][1], prof)
            print("\ncalibration gate, {}: {}".format(name, "PASS" if not bad else "FAIL"))
            for b in bad:
                print("  {}: real {} degraded {} ({})".format(*b))


if __name__ == "__main__":
    main()
