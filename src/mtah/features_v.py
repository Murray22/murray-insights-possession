"""Phase E features: the video ball as a separate channel (FrameTable.vball), never merged.

The video ball exists only where the ATD ball is missing (mtah.video.source.with_video_channel on
the real match; degrade.DegradeConfig.video_fill_share on the bench). Every column here is named
"v_" and reads `vball` only; the ATD-ball features (features, features_b2, features_d) are
computed from `ball` exactly as before, so they keep their ATD meaning.

What the video sees inside a gap, and what it is evidence of (docs/video_pilot.md):
- a ball at a player's feet: team evidence, through the nearest player. The video position is a
  ground back-projection, wrong for a ball in the air, so the touch radius is wider (2 m) than
  the ATD's 1-1.5 m;
- a still ball: mostly a dead ball being placed or held for a restart;
- a still ball by or beyond a line with nobody near: a spare ball or a ball out of play, D
  evidence at most, never team evidence;
- nothing: a flight, an occlusion, or a failure; the flag says which source a frame has.

Columns (all symmetric under a team swap: signed ones flip, per-team ones swap):
- `v_src`: 1 where the video ball is present (the source flag; 0 where the ATD has the ball or
  nobody has), and its shares over +/- 2 s and +/- 10 s; frames since / until a video sighting;
- `v_ddiff`, `v_dmin`: nearest-player distance difference (> 0 home closer) and nearest distance;
- `v_tt`: touch sign within 2 m and its windowed sums (1 s, 3 s) past and next; the last / next
  video touch's team;
- `v_speed`, `v_still`, `v_still_2s`: the video ball's speed and stillness;
- `v_dline`: signed distance to the nearest line (negative outside), `v_spare`: still, within 1 m
  of a line or outside it, nobody within 5 m;
- `v_bx_att`: position along home's attack; `v_last_ddiff`, `v_next_ddiff`.
"""

import numpy as np

from . import features as F
from .frames import AWAY, HOME

EXTRA_VERSION = 1
TOUCH_R = 2.0


def compute_extra(table, attack=None):
    n = table.n_frames
    L, W = table.pitch
    out = {}
    if table.vball is None:
        vb = np.full((n, 2), np.nan)
    else:
        vb = table.vball.astype(np.float64) * np.array([L, W])
    # only where the ATD ball is missing: the channel never overlaps the ATD ball
    vb[~np.isnan(table.ball[:, 0])] = np.nan
    src = ~np.isnan(vb[:, 0])
    out["v_src"] = src.astype(np.float64)
    out["v_src_2s"] = F.rolling_nanmean(out["v_src"], 50)
    out["v_src_10s"] = F.rolling_nanmean(out["v_src"], 250)
    out["v_since"] = F.since_last(src)
    out["v_until"] = F.until_next(src)
    xy = table.xy.astype(np.float64) * np.array([L, W])
    d = np.linalg.norm(xy - vb[:, None, :], axis=2)
    d = np.where(np.isnan(d), np.inf, d)
    team = table.slot_team
    dh = d[:, team == HOME].min(axis=1) if (team == HOME).any() else np.full(n, np.inf)
    da = d[:, team == AWAY].min(axis=1) if (team == AWAY).any() else np.full(n, np.inf)
    dh[~np.isfinite(dh)] = np.nan
    da[~np.isfinite(da)] = np.nan
    ddiff = np.clip(da - dh, -15, 15)
    out["v_ddiff"] = ddiff
    out["v_dmin"] = np.fmin(dh, da)
    tt = np.where(np.nan_to_num(out["v_dmin"], nan=np.inf) < TOUCH_R,
                  np.sign(np.nan_to_num(ddiff, nan=0.0)), 0.0)
    out["v_tt"] = tt
    c = np.concatenate(([0.0], np.cumsum(tt)))
    idx = np.arange(n)
    for k in (25, 75):
        lo = np.clip(idx - k, 0, n)
        hi = np.clip(idx + k + 1, 0, n)
        out["v_tt_past{}".format(k)] = c[idx + 1] - c[lo]
        out["v_tt_next{}".format(k)] = c[hi] - c[idx]
    touch = tt != 0
    out["v_tt_last"] = np.nan_to_num(F.ffill_values(tt, touch), nan=0.0)
    out["v_tt_next"] = np.nan_to_num(F.bfill_values(tt, touch), nan=0.0)
    out["v_tt_since"] = F.since_last(touch)
    out["v_tt_until"] = F.until_next(touch)
    sp = np.full(n, np.nan)
    if n > 4:
        sp[2:n - 2] = np.linalg.norm(vb[4:] - vb[:-4], axis=1) * F.FPS / 4.0
    out["v_speed"] = np.minimum(sp, F.BALL_SPEED_CLIP)
    with np.errstate(invalid="ignore"):
        still = np.where(np.isnan(sp), np.nan, (sp < 0.5).astype(np.float64))
    out["v_still"] = still
    out["v_still_2s"] = F.rolling_nanmean(still, 50)
    dline = np.minimum.reduce([vb[:, 0], L - vb[:, 0], vb[:, 1], W - vb[:, 1]])
    out["v_dline"] = dline
    with np.errstate(invalid="ignore"):
        spare = (np.nan_to_num(still, nan=0.0) > 0) & (dline < 1.0) & (np.nan_to_num(out["v_dmin"], nan=np.inf) > 5.0)
    out["v_spare"] = np.where(src, spare.astype(np.float64), np.nan)
    sign = F.attack_sign(table) if attack is None else attack
    out["v_bx_att"] = (vb[:, 0] - L / 2) * sign
    ok = src & ~np.isnan(ddiff)
    out["v_last_ddiff"] = F.ffill_values(ddiff, ok)
    out["v_next_ddiff"] = F.bfill_values(ddiff, ok)
    return {k: np.asarray(v, dtype=np.float64) for k, v in out.items()}
