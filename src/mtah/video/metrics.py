"""Pilot metrics for ball detections on sampled clips.

Reference on visible frames is the ATD ball projected with the homography. The ATD is itself an
automatic output, so "recall" here is agreement with the ATD, not truth; tuning a confidence
threshold against it is algorithmic self-consistency (no labels, no ground truth).

Candidate filter applied to every detector alike: kind == BALL, conf >= threshold, and the
ground-plane back-projection inside the pitch widened by `margin_m`. A ball in the air
back-projects to the wrong ground point (further from the camera), so metre distances understate
agreement for lofted balls; pixel distances do not have that bias.
"""

import numpy as np

from . import geometry
from .detectors import BALL

MAX_SPEED_MPS = 40.0      # faster than any kicked ball; frame-to-frame link limit
FPS = 25


def candidates(dets, H, conf, margin_m=3.0, pitch=geometry.PITCH_M):
    """Filter one frame's detections; returns (k, 4): px, py, conf, and ground x, y (k, 2)."""
    d = dets[(dets[:, 5] == BALL) & (dets[:, 4] >= conf)]
    if len(d) == 0:
        return np.zeros((0, 3)), np.zeros((0, 2))
    gx, gy = geometry.unproject(H, d[:, 0], d[:, 1])
    m = margin_m / pitch[0]
    ok = geometry.on_pitch(gx, gy, m)
    d = d[ok]
    order = np.argsort(-d[:, 4])
    return d[order][:, [0, 1, 4]], np.column_stack([gx[ok], gy[ok]])[order]


def per_frame(c, frames, dets, conf, margin_m=3.0):
    """Per-frame candidate lists plus the ATD reference in pixels and on the pitch."""
    rows = []
    for f, d in zip(frames, dets):
        H = c["H"][f]
        pix, ground = candidates(d, H, conf, margin_m)
        b = c["ball"][f]
        ref = np.array(geometry.project(H, b[0], b[1])) if np.isfinite(b).all() else None
        rows.append(dict(frame=int(f), pix=pix, ground=ground, ref_pix=ref,
                         ref_ground=b if ref is not None else None))
    return rows


def visible_metrics(rows, radius_px=10.0, radius_m=1.0, pitch=geometry.PITCH_M):
    """Agreement with the projected ATD ball on frames where it is visible."""
    n = tp_any = tp_top = tp_m = n_det = n_tp_det = n_fp = 0
    for r in rows:
        if r["ref_pix"] is None:
            continue
        n += 1
        k = len(r["pix"])
        n_det += k
        if k == 0:
            continue
        dpx = np.hypot(r["pix"][:, 0] - r["ref_pix"][0], r["pix"][:, 1] - r["ref_pix"][1])
        dm = geometry.metres_between(r["ground"], r["ref_ground"][None, :], pitch)
        hit = dpx <= radius_px
        tp_any += bool(hit.any())
        tp_top += bool(hit[0])
        tp_m += bool((dm <= radius_m).any())
        n_tp_det += int(hit.any())          # at most one true positive per frame
        n_fp += int(k - hit.any())
    out = dict(frames=n, recall_px=tp_any / n if n else np.nan, top1_px=tp_top / n if n else np.nan,
                recall_1m=tp_m / n if n else np.nan, fp_per_frame=n_fp / n if n else np.nan,
                precision=n_tp_det / n_det if n_det else np.nan,
                det_rate=sum(len(r["pix"]) > 0 for r in rows if r["ref_pix"] is not None) / n if n else np.nan)
    # recall at several radii: the ATD ball is itself an automatic estimate and is often 10-40 px
    # from the ball visible in the frame (docs/video_pilot.md), so one tight radius understates
    near_px, near_m = [], []
    for r in rows:
        if r["ref_pix"] is None:
            continue
        if len(r["pix"]):
            near_px.append(np.hypot(r["pix"][:, 0] - r["ref_pix"][0], r["pix"][:, 1] - r["ref_pix"][1]).min())
            near_m.append(geometry.metres_between(r["ground"], r["ref_ground"][None, :], pitch).min())
        else:
            near_px.append(np.inf)
            near_m.append(np.inf)
    near_px, near_m = np.array(near_px), np.array(near_m)
    # vertical corridor: the ATD ball is a ground position (x, y, speed; no height), so a ball in
    # the air is seen ABOVE it in the image (measured: 82-84% of 25-400 px disagreements between the
    # ATD and two agreeing detectors are straight up). Accept |dx| <= 10 px and -300 <= dy <= 10.
    corr = 0
    for r in rows:
        if r["ref_pix"] is None or not len(r["pix"]):
            continue
        dx = r["pix"][:, 0] - r["ref_pix"][0]
        dy = r["pix"][:, 1] - r["ref_pix"][1]
        corr += bool(((np.abs(dx) <= 10) & (dy >= -300) & (dy <= 10)).any())
    out["recall_corridor"] = corr / n if n else np.nan
    for rad in (5, 10, 25, 50):
        out["recall_{}px".format(rad)] = float((near_px <= rad).mean()) if n else np.nan
    for rad in (1, 2, 3):
        out["recall_{}m".format(rad)] = float((near_m <= rad).mean()) if n else np.nan
    return out


def linked(rows, clip_len, H_all, pitch=geometry.PITCH_M):
    """For each frame, does its top candidate have a candidate on a neighbouring frame of the same
    clip within MAX_SPEED_MPS (pixel displacement scaled by the local ground scale)?

    Returns (n,) float: 1 linked, 0 not linked, NaN no candidate.
    """
    out = np.full(len(rows), np.nan)
    for i, r in enumerate(rows):
        if len(r["pix"]) == 0:
            continue
        pos = i % clip_len
        top = r["pix"][0]
        g = r["ground"][0]
        scale = float(geometry.px_per_metre(H_all[r["frame"]], g[0], g[1]))
        lim = MAX_SPEED_MPS / FPS * scale + 3.0
        ok = False
        for j in (i - 1, i + 1):
            if 0 <= j < len(rows) and (j % clip_len) == pos + (j - i) and len(rows[j]["pix"]):
                p = rows[j]["pix"]
                if np.hypot(p[:, 0] - top[0], p[:, 1] - top[1]).min() <= lim:
                    ok = True
        out[i] = float(ok)
    return out


def nearest_player_m(c, rows, pitch=geometry.PITCH_M):
    """Ground distance (m) from each frame's top candidate to the nearest ATD player (NaN if none)."""
    out = np.full(len(rows), np.nan)
    for i, r in enumerate(rows):
        if len(r["ground"]) == 0:
            continue
        xy = c["xy"][r["frame"]]
        xy = xy[np.isfinite(xy[:, 0])]
        if len(xy):
            out[i] = geometry.metres_between(xy, r["ground"][0][None, :], pitch).min()
    return out


def gap_metrics(c, rows, clip_len):
    has = np.array([len(r["pix"]) > 0 for r in rows])
    lk = linked(rows, clip_len, c["H"])
    near = nearest_player_m(c, rows)
    return dict(frames=len(rows), det_rate=float(has.mean()),
                linked_rate=float(np.nansum(lk) / len(rows)),
                linked_frac_of_dets=float(np.nanmean(lk)) if has.any() else np.nan,
                near_player_2m_frac_of_dets=float(np.nanmean(near <= 2.0)) if has.any() else np.nan,
                median_player_dist_m=float(np.nanmedian(near)) if has.any() else np.nan)


def visible_linked_precision(c, rows, clip_len, radius_px=10.0):
    """On visible frames: among top candidates that are linked, the share that agree with the ATD.

    This is the calibration that lets a linked detection in a gap be read as 'probably the ball'.
    """
    lk = linked(rows, clip_len, c["H"])
    agree, n = 0, 0
    for r, l in zip(rows, lk):
        if l == 1 and r["ref_pix"] is not None:
            n += 1
            agree += np.hypot(*(r["pix"][0, :2] - r["ref_pix"])) <= radius_px
    return dict(linked_top=n, linked_top_agree=agree / n if n else np.nan,
                linked_rate=float(np.nansum(lk) / max(1, sum(r["ref_pix"] is not None for r in rows))))


def tracked(c, rows, clip_len, min_support=5, **kw):
    """Run the Viterbi tracker per clip. Returns (chosen pixel (n, 2) NaN if none, support (n,))."""
    from . import track
    n = len(rows)
    chosen = np.full((n, 2), np.nan)
    sup = np.zeros(n, int)
    for s in range(0, n, clip_len):
        clip = rows[s:s + clip_len]
        cands = [r["pix"] for r in clip]
        scales = []
        for r in clip:
            g = r["ground"][0] if len(r["ground"]) else np.array([0.5, 0.5])
            scales.append(float(geometry.px_per_metre(c["H"][r["frame"]], g[0], g[1])))
        path = track.viterbi(cands, np.array(scales), **kw)
        sp = track.support(path)
        for i, p in enumerate(path):
            if p >= 0 and sp[i] >= min_support:
                chosen[s + i] = clip[i]["pix"][p, :2]
                sup[s + i] = sp[i]
    return chosen, sup


def tracked_metrics(c, rows, clip_len, radii=(10, 25, 50), **kw):
    """Share of frames the tracker keeps a ball on, and (where the ATD ball exists) its agreement."""
    chosen, sup = tracked(c, rows, clip_len, **kw)
    has = np.isfinite(chosen[:, 0])
    out = dict(track_rate=float(has.mean()))
    ref = np.array([r["ref_pix"] if r["ref_pix"] is not None else (np.nan, np.nan) for r in rows])
    both = has & np.isfinite(ref[:, 0])
    if both.any():
        d = np.hypot(*(chosen[both] - ref[both]).T)
        dx, dy = (chosen[both] - ref[both]).T
        corr = (np.abs(dx) <= 10) & (dy >= -300) & (dy <= 10)
        out["track_agree_corridor"] = float(corr.mean())
        out["track_recall_corridor"] = float(corr.sum() / np.isfinite(ref[:, 0]).sum())
        for rad in radii:
            out["track_agree_{}px".format(rad)] = float((d <= rad).mean())
            out["track_recall_{}px".format(rad)] = float((d <= rad).sum() / np.isfinite(ref[:, 0]).sum())
    near = nearest_player_m(c, [dict(r, ground=r["ground"]) for r in rows])
    return out
