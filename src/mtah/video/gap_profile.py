"""What are the ATD missing-ball gaps? Distributions from the ATD, homography and the automatic
Smart Tagging, no video and no labels. Every number is an aggregate over graded frames.

    python -m mtah.video.gap_profile      # prints JSON

Signals:
- players' median speed (m/s, from ATD positions over 5 frames): play stopped vs running;
- the last ATD ball position before a gap and the first after it: near a touchline or goal line
  (the ball may have gone out), and how far the ball "jumped" across the gap;
- camera framing: share of the pitch area inside the image (from the homography);
- where the last-seen ball lies relative to the image at the gap's middle frame (left the frame?);
- Smart Tagging: share of frames inside a SET PIECES interval or inside any phase.
"""

import json
import os

import numpy as np

from .. import paths
from . import cache, geometry
from .pilot import runs

W, HGT = 1920, 1080


def player_speed(c, step=5):
    xy = c["xy"] * np.asarray(c["pitch"], dtype=np.float32)
    v = np.full(xy.shape[:2], np.nan, dtype=np.float32)
    v[step:] = np.sqrt(((xy[step:] - xy[:-step]) ** 2).sum(-1)) * (25.0 / step)
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return np.nanmedian(v, axis=1)


def visible_pitch_share(H, n=21):
    g = np.linspace(0, 1, n)
    gx, gy = np.meshgrid(g, g)
    px, py = geometry.project(H, gx.ravel(), gy.ravel())
    return float(((px >= 0) & (px < W) & (py >= 0) & (py < HGT)).mean())


def edge_m(p, pitch=geometry.PITCH_M):
    """Distance (m) from a normalised point to the nearest pitch boundary line (negative = outside)."""
    x, y = p[0] * pitch[0], p[1] * pitch[1]
    return min(x, pitch[0] - x, y, pitch[1] - y)


def profile(c=None, tagging_path=None):
    c = c or cache.load()
    graded = c["period"] > 0
    vis = ~np.isnan(c["ball"][:, 0])
    speed = player_speed(c)
    out = {}
    out["median_player_speed_mps"] = {
        "visible": float(np.nanmedian(speed[graded & vis])),
        "gap": float(np.nanmedian(speed[graded & ~vis]))}
    out["share_frames_player_speed_below_1mps"] = {
        "visible": float(np.nanmean(speed[graded & vis] < 1.0)),
        "gap": float(np.nanmean(speed[graded & ~vis] < 1.0))}
    rng = np.random.default_rng(0)
    for name, m in (("visible", graded & vis), ("gap", graded & ~vis)):
        idx = rng.choice(np.flatnonzero(m), 2000, replace=False)
        out.setdefault("visible_pitch_share_median", {})[name] = float(
            np.median([visible_pitch_share(c["H"][f]) for f in idx]))
    gaps = runs(graded & ~vis)
    gaps = [(s, ln) for s, ln in gaps if s > 0 and graded[s - 1]]   # gaps that start in play
    near_line_before, jump, left_frame, lengths = [], [], [], []
    for s, ln in gaps:
        before = c["ball"][s - 1]
        after = c["ball"][s + ln] if s + ln < len(vis) and vis[s + ln] else None
        lengths.append(ln)
        near_line_before.append(edge_m(before))
        if after is not None:
            jump.append(float(geometry.metres_between(before, after)))
        mid = s + ln // 2
        px, py = geometry.project(c["H"][mid], before[0], before[1])
        left_frame.append(not (0 <= px < W and 0 <= py < HGT))
    lengths = np.array(lengths)
    nl = np.array(near_line_before)
    w = lengths / lengths.sum()
    out["gaps_starting_in_play"] = len(gaps)
    out["gap_last_ball_within_3m_of_a_line"] = {
        "share_of_gaps": float((nl < 3).mean()), "share_of_gap_frames": float(w[nl < 3].sum())}
    out["gap_ball_jump_m_median_p90"] = [float(np.median(jump)), float(np.percentile(jump, 90))]
    lf = np.array(left_frame)
    out["last_seen_ball_outside_image_at_gap_middle"] = {
        "share_of_gaps": float(lf.mean()), "share_of_gap_frames": float(w[lf].sum())}
    tagging_path = tagging_path or os.path.join(
        os.environ.get("MTAH_DATA_DIR") or os.path.join(paths.ROOT, "data"),
        "tactical", "BAR-ATH_tactical_smart_tagging.xml")
    if os.path.isfile(tagging_path):
        from ..smart_tagging import load
        iv = load(tagging_path)
        sp = np.zeros(len(vis), bool)
        ph = np.zeros(len(vis), bool)
        for i in iv:
            (sp if i["kind"] == "set_piece" else ph)[max(0, i["start"]):i["end"]] = True
        out["smart_tagging_set_piece_share"] = {"visible": float(sp[graded & vis].mean()),
                                                "gap": float(sp[graded & ~vis].mean())}
        out["smart_tagging_phase_share"] = {"visible": float(ph[graded & vis].mean()),
                                            "gap": float(ph[graded & ~vis].mean())}
        # gap frames within 10 s before a set-piece start (a restart follows a stoppage)
        pre = np.zeros(len(vis), bool)
        for i in iv:
            if i["kind"] == "set_piece":
                pre[max(0, i["start"] - 250):i["start"]] = True
        out["within_10s_before_set_piece"] = {"visible": float(pre[graded & vis].mean()),
                                              "gap": float(pre[graded & ~vis].mean())}
    return out


if __name__ == "__main__":
    print(json.dumps(profile(), indent=1))
