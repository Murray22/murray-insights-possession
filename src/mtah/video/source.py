"""The video ball track as a separate, flagged ball source for a FrameTable.

Nothing here changes the ATD ball. `video_ball(table)` returns the video track on the table's
frames; `with_video_ball(table)` returns a copy whose ball is the ATD ball where the ATD has one and
the video ball (support >= min_support) only where the ATD is missing, with the provenance in
meta["ball_source"] (0 ATD, 1 video, -1 none) and the raw video arrays in meta["video_ball"].

Caveat carried in meta: the video position is the detection's ground-plane back-projection, which
is right for a ball on the ground and wrong for a ball in the air (it lands too far from the
camera). The ATD ball, by contrast, is a ground position (docs/video_pilot.md).

numpy only; reads runs/video/full/ball_track_<config>.npz written by mtah.video.fullrun track.
"""

import os

import numpy as np

from .fullrun import FULL_DIR

ATD, VIDEO, NONE = 0, 1, -1


def load_track(path=None, config="soccana_ov"):
    path = path or os.path.join(FULL_DIR, "ball_track_{}.npz".format(config))
    z = np.load(path)
    return {k: z[k] for k in z.files}


def video_ball(table, track=None, min_support=5, **kw):
    """(ball (n, 2) float32 normalised pitch, NaN where no kept video ball; info dict) on table.frame."""
    track = track or load_track(**kw)
    f = np.asarray(table.frame)
    n_video = len(track["processed"])
    ok_idx = (f >= 0) & (f < n_video)
    ball = np.full((len(f), 2), np.nan, np.float32)
    sup = np.zeros(len(f), np.int32)
    proc = np.zeros(len(f), bool)
    ball[ok_idx] = track["ball"][f[ok_idx]]
    sup[ok_idx] = track["support"][f[ok_idx]]
    proc[ok_idx] = track["processed"][f[ok_idx]]
    ball[sup < min_support] = np.nan
    return ball, dict(support=sup, processed=proc, min_support=min_support,
                      config=str(track.get("config", "")))


def with_video_ball(table, track=None, min_support=5, **kw):
    """Copy of `table` with ATD-missing frames filled from the video track, provenance flagged."""
    vb, info = video_ball(table, track, min_support, **kw)
    out = table.copy()
    atd = ~np.isnan(table.ball[:, 0])
    fill = ~atd & np.isfinite(vb[:, 0])
    out.ball[fill] = vb[fill]
    src = np.full(len(atd), NONE, np.int8)
    src[atd] = ATD
    src[fill] = VIDEO
    out.meta = dict(out.meta)
    out.meta["ball_source"] = src
    out.meta["video_ball"] = dict(ball=vb, **info)
    out.meta["video_ball_caveat"] = "ground back-projection; wrong for a ball in the air"
    return out


def with_video_channel(table, track=None, min_support=5, **kw):
    """Copy of `table` with the video ball as a separate channel (FrameTable.vball) on the frames
    where the ATD ball is missing; `ball` is not changed. meta["ball_source"] as in
    with_video_ball. This is what the Phase E models read (features_v)."""
    vb, info = video_ball(table, track, min_support, **kw)
    out = table.copy()
    atd = ~np.isnan(table.ball[:, 0])
    vb = vb.copy()
    vb[atd] = np.nan
    out.vball = vb.astype(np.float32)
    src = np.full(len(atd), NONE, np.int8)
    src[atd] = ATD
    src[np.isfinite(vb[:, 0])] = VIDEO
    out.meta = dict(out.meta)
    out.meta["ball_source"] = src
    out.meta["video_track_config"] = info["config"]
    out.meta["video_ball_caveat"] = "ground back-projection; wrong for a ball in the air"
    return out
