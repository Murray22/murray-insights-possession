"""Label-free properties of a video ball track beside an ATD-shaped table (Phase E).

    PYTHONPATH=src .venv/bin/python -m mtah.video.props            # the real track, JSON + table

One function, `profile(table, vball, processed)`, runs on the real match (the saved full-run
track) and on a degraded sample game whose degrader simulated a video track
(`degrade.DegradeConfig.video_fill_share`), so the two can be compared row by row and the
simulation calibrated. Everything is a distribution or a share over graded frames. No label is
read, no frame is singled out.

Frames:
- gap frames: graded, ATD ball missing. "filled" = the video track has a ball there.
- window frames: graded, ATD ball visible, processed by the detector (the run covered 25 frames
  either side of every gap so the tracker could start from a visible ball). On these the ATD
  ball is a reference: video-vs-ATD error, the lag between the two, and the video coverage by the
  ATD ball's speed (how often the detector keeps a ball moving at a given speed).

What is measured
- fill share on gap frames: overall; by gap length; by the players' median speed; by the ATD
  ball's last-seen speed before the gap; by position in the gap (first / last second, middle).
- fill bursts inside gaps (runs of filled frames): length distribution, bursts per gap, the
  delay from a gap's start to its first fill.
- the filled ball: its speed, how often it is static, how often it sits off the pitch or within
  a metre of a line with no player within 5 m ("spare-like": what a spare ball looks like).
- window frames: coverage by ATD speed bin; error vector (video - ATD, metres) distribution by
  ATD speed bin, the error's within-run autocorrelation; the shift of the ATD that best matches
  the video (lag, in frames, with linear interpolation).
"""

import argparse
import json
import os
import warnings

import numpy as np

from .. import paths
from ..features import runs as _runs

FPS = 25
Q = (5, 10, 25, 50, 75, 90, 95, 99)
SPEED_EDGES = [0, 1, 3, 6, 10, 15, 1e9]                      # m/s, ATD ball speed bins
GAP_LEN_EDGES = [0, 100, 250, 800, 1e9]                      # frames
PSPD_EDGES = [0, 1, 2, 3, 1e9]                               # m/s, players' median speed
STATIC_MPS = 0.5
SPARE_LINE_M = 1.0                                           # inside the line by at most this
SPARE_PLAYER_M = 5.0
WINDOW = 25


def _q(x, qs=Q):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if not len(x):
        return {"n": 0}
    out = {"n": int(len(x)), "mean": float(x.mean())}
    out.update({"p{}".format(q): float(v) for q, v in zip(qs, np.percentile(x, qs))})
    return out


def _metric(a, pitch):
    return np.asarray(a, dtype=np.float64) * np.asarray(pitch, dtype=np.float64)


def speed(pos_m, lag=4):
    """(n,) speed over `lag` frames, centred (NaN where either end is missing)."""
    n = len(pos_m)
    out = np.full(n, np.nan)
    h = lag // 2
    if n > lag:
        out[h:n - lag + h] = np.linalg.norm(pos_m[lag:] - pos_m[:-lag], axis=1) * FPS / lag
    return out


def players_median_speed(table, step=10):
    xy = _metric(table.xy, table.pitch)
    n = len(xy)
    v = np.full(xy.shape[:2], np.nan)
    h = step // 2
    v[h:n - step + h] = np.linalg.norm(xy[step:] - xy[:-step], axis=2) * FPS / step
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(np.minimum(v, 12.0), axis=1)


def line_distance(p_m, pitch):
    """Signed distance (m) to the nearest touchline / goal line, negative outside the pitch."""
    L, W = pitch
    x, y = p_m[:, 0], p_m[:, 1]
    return np.minimum.reduce([x, L - x, y, W - y])


def nearest_player(table, p_m):
    xy = _metric(table.xy, table.pitch)
    d = np.linalg.norm(xy - p_m[:, None, :], axis=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmin(np.where(np.isnan(d), np.inf, d), axis=1)


def _binned_share(values, mask, hit, edges):
    """[(lo, hi, frames, share of hit)] over mask, binned by values."""
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = mask & np.isfinite(values) & (values >= lo) & (values < hi)
        out.append({"lo": float(lo), "hi": float(hi), "frames": int(m.sum()),
                    "share": float(hit[m].mean()) if m.any() else None})
    return out


def gap_runs(table):
    """[(start, stop)] of ATD-missing runs inside graded blocks."""
    graded = table.period > 0
    miss = np.isnan(table.ball[:, 0]) & graded
    out = []
    for a, b in _runs(graded):
        out.extend((a + s, a + e) for s, e in _runs(miss[a:b]))
    return out


def profile(table, vball, processed=None, truth=None):
    """Properties of the video track `vball` ((n, 2) normalised, NaN where none) on `table`.
    `truth` (bench only): the true ball (normalised), for the error of filled frames against it."""
    pitch = table.pitch
    graded = table.period > 0
    atd = _metric(table.ball, pitch)
    vb = _metric(vball, pitch)
    atd_vis = ~np.isnan(atd[:, 0])
    vis = ~np.isnan(vb[:, 0])
    if processed is None:
        processed = np.ones(len(vb), dtype=bool)
    gap = graded & ~atd_vis
    fill = gap & vis
    out = {"gap_frames": int(gap.sum()), "filled_frames": int(fill.sum()),
           "fill_share": float(fill.sum() / max(1, gap.sum())),
           "processed_share_of_gap": float((processed & gap).sum() / max(1, gap.sum()))}
    gaps = gap_runs(table)
    glen = np.zeros(len(vb))
    pos = np.full(len(vb), -1)                    # 0 first second, 1 middle, 2 last second
    for s, e in gaps:
        glen[s:e] = e - s
        pos[s:e] = 1
        pos[s:min(e, s + FPS)] = 0
        pos[max(s, e - FPS):e] = 2
    out["fill_by_gap_len"] = _binned_share(glen, gap, vis, GAP_LEN_EDGES)
    psp = players_median_speed(table)
    out["fill_by_player_speed"] = _binned_share(psp, gap, vis, PSPD_EDGES)
    # the ATD ball's speed over the last 4 visible frames before each gap, spread over the gap
    asp = speed(atd)
    last_sp = np.full(len(vb), np.nan)
    for s, e in gaps:
        if s >= 1 and np.isfinite(asp[s - 3 if s >= 3 else 0]):
            last_sp[s:e] = asp[max(0, s - 3)]
    out["fill_by_last_seen_speed"] = _binned_share(last_sp, gap, vis, SPEED_EDGES)
    out["fill_by_gap_position"] = {k: (float(vis[gap & (pos == i)].mean()) if (gap & (pos == i)).any()
                                       else None) for i, k in enumerate(("first_s", "middle", "last_s"))}
    # bursts inside gaps
    blen, per_gap, first_delay, off_len = [], [], [], []
    for s, e in gaps:
        r = _runs(vis[s:e])
        per_gap.append(len(r))
        blen.extend(b - a for a, b in r)
        if r:
            first_delay.append(r[0][0])
            off_len.extend(c - b for (_, b), (c, _) in zip(r[:-1], r[1:]))
    out["burst_len"] = _q(blen)
    out["bursts_per_gap"] = _q(per_gap)
    out["bursts_per_gap_hour"] = float(len(blen) / max(1e-9, graded.sum() / FPS / 3600.0))
    out["first_fill_delay"] = _q(first_delay)
    out["between_bursts"] = _q(off_len)
    # the filled ball
    vsp = speed(np.where(vis[:, None], vb, np.nan))
    out["filled_speed"] = _q(vsp[fill])
    out["filled_static_share"] = float((vsp[fill] < STATIC_MPS).mean()) if fill.any() else None
    out["filled_fast_share"] = float((vsp[fill] > 15).mean()) if fill.any() else None
    dl = line_distance(vb, pitch)
    near = nearest_player(table, vb)
    out["filled_offpitch_share"] = float((dl[fill] < 0).mean()) if fill.any() else None
    out["filled_line_dist"] = _q(dl[fill])
    out["filled_nearest_player"] = _q(near[fill])
    with np.errstate(invalid="ignore"):
        spare = fill & (dl < SPARE_LINE_M) & (near > SPARE_PLAYER_M) & (np.nan_to_num(vsp, nan=0.0) < STATIC_MPS)
    out["spare_like_share_of_gap"] = float(spare.sum() / max(1, gap.sum()))
    out["spare_like_share_of_filled"] = float(spare.sum() / max(1, fill.sum()))
    out["spare_like_burst_len"] = _q([e - s for s, e in _runs(spare)])
    out["filled_near_player_2m"] = float((near[fill] < 2).mean()) if fill.any() else None
    # window frames: the ATD ball as a reference
    win = graded & atd_vis & processed
    both = win & vis
    out["window_frames"] = int(win.sum())
    out["window_coverage"] = float(both.sum() / max(1, win.sum()))
    out["window_coverage_by_speed"] = _binned_share(asp, win, vis, SPEED_EDGES)
    err = vb - atd
    en = np.linalg.norm(err, axis=1)
    out["window_err_m"] = _q(en[both])
    out["window_err_dx"] = _q(err[both, 0])
    out["window_err_dy"] = _q(err[both, 1])
    out["window_err_by_speed"] = [
        dict(lo=float(lo), hi=float(hi), **{k: v for k, v in _q(en[both & (asp >= lo) & (asp < hi)],
                                                                 (50, 75, 90)).items()})
        for lo, hi in zip(SPEED_EDGES[:-1], SPEED_EDGES[1:])]
    out["window_err_over_2m"] = float((en[both] > 2).mean()) if both.any() else None
    out["window_err_over_5m"] = float((en[both] > 5).mean()) if both.any() else None
    # temporal structure of the error inside runs of `both`: per-run mean and the residual
    bias, resid, ac = [], [], []
    for s, e in _runs(both):
        if e - s < 5:
            continue
        r = err[s:e]
        m = r.mean(0)
        bias.append(np.linalg.norm(m))
        d = r - m
        resid.append(np.sqrt((d ** 2).sum(1).mean()))
        x = d[:, 0]
        if x.std() > 0:
            ac.append(float(np.corrcoef(x[1:], x[:-1])[0, 1]))
    out["window_err_run_bias_m"] = _q(bias)
    out["window_err_run_resid_m"] = _q(resid)
    out["window_err_lag1_autocorr"] = float(np.nanmedian(ac)) if ac else None
    out["lag"] = lag_scan(atd, vb, both)
    if truth is not None:
        tr = _metric(truth, pitch)
        ok = fill & ~np.isnan(tr[:, 0])
        out["filled_err_vs_truth_m"] = _q(np.linalg.norm(vb[ok] - tr[ok], axis=1))
    return out


# The video-simulation gate: a simulated track must look like the real one on these
# (key, kind, tolerance); "abs" |sim - real| <= tol, "rel" |sim / real - 1| <= tol, "max" sim <= tol.
VIDEO_RULES = [
    (("fill_share",), "abs", 0.03),
    (("filled_static_share",), "abs", 0.10),
    (("filled_fast_share",), "max", 0.05),
    (("burst_len", "p50"), "rel", 0.5),
    (("burst_len", "p90"), "rel", 0.6),
    (("spare_like_share_of_gap",), "abs", 0.02),
    (("filled_near_player_2m",), "abs", 0.10),
    (("filled_offpitch_share",), "abs", 0.06),
]


def check_video_calibration(real, sim, rules=VIDEO_RULES):
    """[(statistic, real, simulated, rule)] for every rule the simulated profile breaks."""
    bad = []
    for path, kind, tol in rules:
        r = real.get(path[0])
        d = sim.get(path[0])
        if len(path) == 2:
            r = r.get(path[1]) if isinstance(r, dict) else None
            d = d.get(path[1]) if isinstance(d, dict) else None
        name = ".".join(path)
        if d is None or (r is None and kind != "max"):
            bad.append((name, r, d, "missing"))
            continue
        ok = {"abs": lambda: abs(d - r) <= tol, "rel": lambda: r != 0 and abs(d / r - 1) <= tol,
              "max": lambda: d <= tol}[kind]()
        if not ok:
            bad.append((name, r, d, "{} {}".format(kind, tol)))
    return bad


def lag_scan(atd_m, vb_m, both, shifts=np.arange(-3.0, 3.01, 0.25)):
    """Median |video(f) - ATD(f + k)| for fractional k (linear interpolation of the ATD); the k
    with the smallest median is the ATD's lag behind the video (positive: ATD late)."""
    idx = np.flatnonzero(both)
    n = len(atd_m)
    res = {}
    for k in shifts:
        lo = np.floor(k).astype(int)
        w = k - lo
        i0, i1 = idx + lo, idx + lo + 1
        ok = (i0 >= 0) & (i1 < n)
        a = (1 - w) * atd_m[np.clip(i0, 0, n - 1)] + w * atd_m[np.clip(i1, 0, n - 1)]
        d = np.linalg.norm(vb_m[idx] - a, axis=1)
        d = d[ok & np.isfinite(d)]
        res[float(k)] = float(np.median(d)) if len(d) else None
    good = {k: v for k, v in res.items() if v is not None}
    best = min(good, key=good.get) if good else None
    return {"median_err_by_shift": res, "best_shift": best}


CALIB_CACHE = os.path.join(paths.RUNS_DIR, "cache", "video_calibration.npz")


def calibration_from(table, vball, processed):
    """What the degrader's video simulation draws from: the lengths of the fill bursts inside gaps
    and the video-minus-ATD error vectors (metres) on window frames, binned by the ATD ball's
    speed (SPEED_EDGES). Distributions with no frame index: one number or vector per burst / frame."""
    pitch = table.pitch
    graded = table.period > 0
    atd = _metric(table.ball, pitch)
    vb = _metric(vball, pitch)
    vis = ~np.isnan(vb[:, 0])
    atd_vis = ~np.isnan(atd[:, 0])
    blen = []
    for s, e in gap_runs(table):
        blen.extend(b - a for a, b in _runs(vis[s:e]))
    both = graded & atd_vis & processed & vis
    asp = speed(atd)
    err = (vb - atd)
    out = {"burst_len": np.array(blen, dtype=np.int64), "speed_edges": np.array(SPEED_EDGES[:-1])}
    for i, (lo, hi) in enumerate(zip(SPEED_EDGES[:-1], SPEED_EDGES[1:])):
        m = both & (asp >= lo) & (asp < hi)
        out["err{}".format(i)] = err[m].astype(np.float32)
    # shuffle the vectors so nothing about their order (time) is kept
    rng = np.random.default_rng(0)
    for k in list(out):
        if k.startswith("err") or k == "burst_len":
            out[k] = out[k][rng.permutation(len(out[k]))]
    return out


def real_video_calibration(cache=True):
    """calibration_from() on the competition match and the saved video track, cached in runs/cache
    (a distribution, like atd_profile.real_gap_lengths)."""
    if cache and os.path.isfile(CALIB_CACHE):
        z = np.load(CALIB_CACHE)
        return {k: z[k] for k in z.files}
    out = calibration_from(*real_track())
    if cache:
        os.makedirs(os.path.dirname(CALIB_CACHE), exist_ok=True)
        np.savez(CALIB_CACHE + ".tmp.npz", **out)
        os.replace(CALIB_CACHE + ".tmp.npz", CALIB_CACHE)
    return out


def real_track():
    """(table, vball, processed) for the competition match from the saved full-run track."""
    from .. import atd_io
    from . import source
    t = atd_io.load_competition()
    vb, info = source.video_ball(t)
    return t, vb, info["processed"]


ROWS = [
    ("fill share of gap frames", ("fill_share",), "{:.3f}"),
    ("fill share, gaps < 100 / 100-250 / 250-800 / >= 800 frames", ("fill_by_gap_len",), "{:.2f}"),
    ("fill share by players' median speed < 1 / 1-2 / 2-3 / >= 3 m/s", ("fill_by_player_speed",), "{:.2f}"),
    ("fill share by last-seen ball speed 0-1 / 1-3 / 3-6 / 6-10 / 10-15 / 15+", ("fill_by_last_seen_speed",), "{:.2f}"),
    ("fill share: first second / middle / last second of a gap", ("fill_by_gap_position",), "{:.2f}"),
    ("burst length p25 / p50 / p75 / p90 (frames)", ("burst_len", ("p25", "p50", "p75", "p90")), "{:.0f}"),
    ("bursts per gap, mean", ("bursts_per_gap", ("mean",)), "{:.2f}"),
    ("frames between bursts p25 / p50 / p75", ("between_bursts", ("p25", "p50", "p75")), "{:.0f}"),
    ("first fill delay p25 / p50 / p75 (frames)", ("first_fill_delay", ("p25", "p50", "p75")), "{:.0f}"),
    ("filled ball speed p25 / p50 / p75 / p95 (m/s)", ("filled_speed", ("p25", "p50", "p75", "p95")), "{:.1f}"),
    ("filled ball static (< 0.5 m/s)", ("filled_static_share",), "{:.3f}"),
    ("filled ball faster than 15 m/s", ("filled_fast_share",), "{:.3f}"),
    ("filled ball off the pitch", ("filled_offpitch_share",), "{:.3f}"),
    ("filled ball within 2 m of a player", ("filled_near_player_2m",), "{:.3f}"),
    ("filled nearest player p25 / p50 / p75 (m)", ("filled_nearest_player", ("p25", "p50", "p75")), "{:.1f}"),
    ("spare-like share of gap frames", ("spare_like_share_of_gap",), "{:.4f}"),
    ("spare-like burst p50 / p90", ("spare_like_burst_len", ("p50", "p90")), "{:.0f}"),
    ("window coverage", ("window_coverage",), "{:.3f}"),
    ("window coverage by ATD speed 0-1 / 1-3 / 3-6 / 6-10 / 10-15 / 15+", ("window_coverage_by_speed",), "{:.2f}"),
    ("window error p25 / p50 / p75 / p90 (m)", ("window_err_m", ("p25", "p50", "p75", "p90")), "{:.2f}"),
    ("window error over 2 m / 5 m", ("window_err_over_2m", "window_err_over_5m"), "{:.3f}"),
    ("window error p50 by ATD speed bin", ("window_err_by_speed",), "{:.2f}"),
    ("window run bias p50 / resid p50 (m)", ("window_err_run_bias_m", "window_err_run_resid_m"), "{:.2f}"),
    ("error lag-1 autocorrelation", ("window_err_lag1_autocorr",), "{:.2f}"),
    ("best ATD shift against the video (frames)", ("lag",), "{:+.2f}"),
    ("bench: filled error vs the true ball p25 / p50 / p75 / p90 (m)", ("filled_err_vs_truth_m", ("p25", "p50", "p75", "p90")), "{:.2f}"),
]


def _cell(prof, key, fmt):
    if len(key) == 2 and isinstance(key[1], tuple):
        v = prof.get(key[0]) or {}
        return " / ".join("-" if v.get(k) is None else fmt.format(v[k]) for k in key[1])
    vals = []
    for k in key:
        v = prof.get(k)
        if isinstance(v, list):
            vals.extend("-" if (b.get("share", b.get("p50")) is None)
                        else fmt.format(b.get("share", b.get("p50"))) for b in v)
        elif isinstance(v, dict) and "best_shift" in v:
            vals.append("-" if v["best_shift"] is None else fmt.format(v["best_shift"]))
        elif isinstance(v, dict) and "p50" in v:
            vals.append(fmt.format(v["p50"]))
        elif isinstance(v, dict):
            vals.extend("-" if x is None else fmt.format(x) for x in v.values())
        else:
            vals.append("-" if v is None else fmt.format(v))
    return " / ".join(vals)


def compare(columns):
    lines = ["| statistic | " + " | ".join(n for n, _ in columns) + " |",
             "|---|" + "---|" * len(columns)]
    for label, key, fmt in ROWS:
        lines.append("| {} | {} |".format(label, " | ".join(_cell(p, key, fmt) for _, p in columns)))
    return "\n".join(lines)


def simulated(variant, seeds=(0,), games=(1, 2, 3)):
    """Mean profile of the simulated video channel on degraded sample games (truth = clean ball)."""
    from .. import atd_profile, modelbench as mb
    profs = []
    for s in seeds:
        for g in games:
            t, _ = mb.degraded_table(variant, s, g)
            clean = mb.game_truth(g)[0]
            vb = t.vball if t.vball is not None else np.full((t.n_frames, 2), np.nan, np.float32)
            profs.append(profile(t, vb, None, clean.ball))
    return atd_profile.mean_profiles(profs)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=os.path.join(paths.RUNS_DIR, "phaseE", "video_props_real.json"))
    ap.add_argument("--sim", nargs="*", default=[], help="bench variants with a simulated video channel")
    ap.add_argument("--seeds", type=int, default=1)
    a = ap.parse_args(argv)
    t, vb, proc = real_track()
    prof = profile(t, vb, proc)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(prof, fh, indent=1)
    cols = [("real", prof)]
    for v in a.sim:
        m = simulated(v, tuple(range(a.seeds)))
        with open(os.path.join(os.path.dirname(a.out), "video_props_{}.json".format(v)), "w") as fh:
            json.dump(m, fh, indent=1)
        cols.append((v, m))
    print(compare(cols))
    for name, m in cols[1:]:
        bad = check_video_calibration(prof, m)
        print("\nvideo gate, {}: {}".format(name, "PASS" if not bad else "FAIL"))
        for b in bad:
            print("  {}: real {} simulated {} ({})".format(*b))


if __name__ == "__main__":
    main()
