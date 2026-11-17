"""Score pilot detections (runs/video/pilot/*.npz) and run the alignment check.

    python -m mtah.video.evaluate [--configs a,b] [--thresholds 0.05,0.1,...]

Writes runs/video/pilot_eval.json and prints a table. Uses ATD and homography only; no labels.
"""

import argparse
import glob
import json
import os

import numpy as np

from . import align, cache, geometry, metrics
from .detectors import BALL, PERSON
from .pilot import OUT_DIR, load_dets


def alignment(c, frames, dets, conf, shifts=range(-3, 4), radius=8.0):
    """Ball: ATD ball of frame f+k projected with H[f], against ball detections on frame f."""
    frames = np.asarray(frames)
    ball_dets = [d[(d[:, 5] == BALL) & (d[:, 4] >= conf)] for d in dets]
    refs = {}
    for k in shifts:
        g = np.clip(frames + k, 0, len(c["ball"]) - 1)
        b = c["ball"][g]
        px, py = geometry.project(c["H"][frames], b[:, 0], b[:, 1])
        refs[k] = np.column_stack([px, py])
    return align.check(refs, ball_dets, radius=radius)


def player_alignment(c, frames, dets, shifts=(-5, -2, -1, 0, 1, 2, 5), radius=6.0, cap=20.0,
                     min_motion=1.5):
    """Same check with every projected ATD player as an item and person-box foot points (bottom
    centre) as detections. Players move ~0.2 m (2-6 px) per frame, so the threshold is lower."""
    items_ref = {k: [] for k in shifts}
    items_det = []
    for f, d in zip(frames, dets):
        p = d[d[:, 5] == PERSON]
        if len(p) == 0:
            continue
        feet = np.column_stack([p[:, 0], p[:, 1] + p[:, 3] / 2])
        H = c["H"][f]
        u0, v0 = geometry.project(H, c["xy"][f][:, 0], c["xy"][f][:, 1])
        inside = np.isfinite(u0) & (u0 >= 0) & (u0 < 1920) & (v0 >= 0) & (v0 < 1080)
        for k in shifts:
            g = min(max(f + k, 0), len(c["xy"]) - 1)
            u, v = geometry.project(H, c["xy"][g][:, 0], c["xy"][g][:, 1])
            items_ref[k].append(np.column_stack([u, v])[inside])
        items_det += [feet] * int(inside.sum())
    refs = {k: np.concatenate(v) for k, v in items_ref.items()}
    return align.check(refs, items_det, radius=radius, cap=cap, min_motion=min_motion)


def evaluate(configs=None, thresholds=(0.05, 0.1, 0.2, 0.3, 0.5)):
    c = cache.load()
    res = {}
    for path in sorted(glob.glob(os.path.join(OUT_DIR, "*.npz"))):
        which, cfg = os.path.basename(path)[:-4].split("_", 1)
        if configs and cfg not in configs:
            continue
        frames, dets, L = load_dets(path)
        info = json.load(open(path[:-4] + ".json"))
        entry = res.setdefault(cfg, {})
        entry[which + "_info"] = info
        for thr in thresholds:
            rows = metrics.per_frame(c, frames, dets, thr)
            if which == "vis":
                m = metrics.visible_metrics(rows)
                m.update(metrics.visible_linked_precision(c, rows, L))
            else:
                m = metrics.gap_metrics(c, rows, L)
            m.update(metrics.tracked_metrics(c, rows, L))
            entry.setdefault(which, {})[str(thr)] = m
        if cfg != "motion_blob":
            entry[which + "_breakdown"] = gap_breakdown(c, frames, dets, L)
        if which == "vis" and cfg != "motion_blob":
            entry["alignment"] = {str(thr): alignment(c, frames, dets, thr) for thr in (0.1, 0.3)}
            if any((d[:, 5] == PERSON).any() for d in dets):
                entry["player_alignment"] = player_alignment(c, frames, dets)
    return res


def gap_breakdown(c, frames, dets, clip_len, conf=0.1, min_support=5):
    """What the decoded gap frames look like, as shares of gap frames.

    stoppage-like: median ATD player speed < 1 m/s (gap_profile.player_speed). Within each, the
    tracked video ball (support >= min_support) is 'with a player' (ground point within 2 m of an
    ATD player), 'loose' (on the pitch, further), or absent. 'fast' = tracked ground speed above
    15 m/s between consecutive tracked frames (a kicked or lofted ball; for a lofted ball the
    ground back-projection itself is wrong, so this is a lower bound on balls in the air).
    Also: off-pitch candidates (3-15 m outside the lines) and ATD players on the frame.
    """
    from .gap_profile import player_speed
    speed = player_speed(c)
    rows = metrics.per_frame(c, frames, dets, conf, margin_m=3.0)
    chosen, sup = metrics.tracked(c, rows, clip_len, min_support=min_support)
    has = np.isfinite(chosen[:, 0])
    ground = np.full((len(rows), 2), np.nan)
    for i, r in enumerate(rows):
        if has[i]:
            gx, gy = geometry.unproject(c["H"][r["frame"]], *chosen[i])
            ground[i] = (gx, gy)
    near = np.full(len(rows), np.nan)
    for i, r in enumerate(rows):
        if has[i]:
            xy = c["xy"][r["frame"]]
            xy = xy[np.isfinite(xy[:, 0])]
            if len(xy):
                near[i] = geometry.metres_between(xy, ground[i][None, :]).min()
    fast = np.zeros(len(rows), bool)
    for i in range(1, len(rows)):
        if i % clip_len and has[i] and has[i - 1]:
            fast[i] = geometry.metres_between(ground[i], ground[i - 1]) * 25 > 15
    # static: the tracked run this frame belongs to stays within 0.3 m of its median for >= 10
    # frames (a ball placed for a restart, or a pitch marking / spare ball mistaken for one)
    static = np.zeros(len(rows), bool)
    t = 0
    while t < len(rows):
        if not has[t]:
            t += 1
            continue
        u = t
        while u < len(rows) and has[u] and (u == t or u % clip_len):
            u += 1
        if u - t >= 10:
            med = np.median(ground[t:u], axis=0)
            if geometry.metres_between(ground[t:u], med[None, :]).max() < 0.3:
                static[t:u] = True
        t = u
    stop = np.array([speed[r["frame"]] < 1.0 for r in rows])
    out = {"frames": len(rows), "stoppage_like": float(stop.mean())}
    for name, m in (("play", ~stop), ("stoppage", stop)):
        k = max(1, m.sum())
        out[name] = dict(share=float(m.mean()),
                         ball_with_player=float((m & has & (near <= 2)).sum() / k),
                         ball_loose=float((m & has & ~(near <= 2)).sum() / k),
                         ball_fast=float((m & fast).sum() / k),
                         ball_static=float((m & static).sum() / k),
                         no_ball=float((m & ~has).sum() / k))
    off = []
    for f, d in zip(frames, dets):
        b = d[(d[:, 5] == 0) & (d[:, 4] >= conf)]
        if len(b) == 0:
            off.append(False)
            continue
        gx, gy = geometry.unproject(c["H"][f], b[:, 0], b[:, 1])
        out_in = geometry.on_pitch(gx, gy, 15 / 105.3) & ~geometry.on_pitch(gx, gy, 3 / 105.3)
        off.append(bool(out_in.any()))
    out["offpitch_candidate_share"] = float(np.mean(off))
    out["atd_players_median"] = float(np.median([np.isfinite(c["xy"][f][:, 0]).sum() for f in frames]))
    return out


def _fmt(x):
    return "  -  " if x is None or (isinstance(x, float) and np.isnan(x)) else "{:5.3f}".format(x)


def _print_align(name, al):
    print("   alignment {}: aligned={} best={} informative={}/{} mean_px {} sign-test {}".format(
        name, al["aligned"], al["best"], al["n_informative"], al["n"],
        {k: round(v, 2) for k, v in al["mean_dist"].items()},
        {k: (p["n01"], p["n10"], "{:.1g}".format(p["p"])) for k, p in al["pair"].items()}))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default=None)
    ap.add_argument("--thresholds", default="0.05,0.1,0.2,0.3,0.5")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(OUT_DIR), "pilot_eval.json"))
    a = ap.parse_args(argv)
    res = evaluate(a.configs.split(",") if a.configs else None,
                   tuple(float(t) for t in a.thresholds.split(",")))
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1, default=float)
    print("{:16s} {:>5s} | {:>5s} {:>5s} {:>5s} {:>5s} {:>6s} {:>5s} | {:>5s} {:>5s} | {:>5s} {:>6s} {:>5s} {:>5s} | {:>6s}".format(
        "config", "thr", "rec10", "rec25", "recCor", "top1", "FP/fr", "prec", "trkR", "trkAcor",
        "gapDet", "gapLnk", "gTrk", "near2", "fps"))
    for cfg, e in res.items():
        for thr in a.thresholds.split(","):
            v = e.get("vis", {}).get(str(float(thr)), {})
            g = e.get("gap", {}).get(str(float(thr)), {})
            print("{:16s} {:>5s} | {} {} {} {} {:6.2f} {} | {} {} | {} {} {} {} | {:6.2f}".format(
                cfg, thr, _fmt(v.get("recall_px")), _fmt(v.get("recall_25px")), _fmt(v.get("recall_corridor")),
                _fmt(v.get("top1_px")), v.get("fp_per_frame", float("nan")), _fmt(v.get("precision")),
                _fmt(v.get("track_rate")), _fmt(v.get("track_agree_corridor")),
                _fmt(g.get("det_rate")), _fmt(g.get("linked_rate")), _fmt(g.get("track_rate")),
                _fmt(g.get("near_player_2m_frac_of_dets")),
                e.get("vis_info", {}).get("detect_fps", float("nan"))))
        if "alignment" in e:
            for thr, al in e["alignment"].items():
                _print_align("ball thr " + thr, al)
        if "player_alignment" in e:
            _print_align("players", e["player_alignment"])


if __name__ == "__main__":
    main()
