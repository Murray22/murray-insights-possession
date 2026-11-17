"""Turn clean sample-game tracking into something shaped like the competition's automatic tracking.

DATA.md describes the ATD this has to resemble: no player identities (46 anonymous track slots,
each owned by one team, that "appear, vanish and swap identity"), about 20.5 tracked objects of
22 on graded frames, the odd duplicate, and the ball missing on 41% of graded frames, "occlusions,
crowded boxes and balls in the air" most of all. Clean Metrica sample tracking has none of that,
and one more difference matters a lot: its ball is NaN almost exactly when the ball is dead (it is
corrected data, and nobody tracks a ball that is out of play). Left alone, that hands any model
the dead-ball answer for free, so the degrader also gives the dead-ball stretches a plausible
ball some of the time.

Every step is seeded and parameterised in DegradeConfig. The ball burst-length distribution can
be replaced by an empirical one (`ball_burst_lengths`): once the real ATD arrives, its ball
visibility run lengths (a distribution only, never labels) go there.

Steps, in order (all positions in metres internally, normalised 0-1 on output):
1. Identity swaps: when two same-team players come within `swap_distance_m`, with probability
   `swap_prob` the tracker swaps them at the closest frame, so every later slot follows the other
   person.
2. Tracklets: each tracker identity is cut into tracklets by a per-frame end hazard (raised when
   another player is within `crowd_radius_m`), each followed by a gap with no detection. Gap
   lengths are lognormal; a share `reacquire_prob` of breaks are re-acquired within 1-3 frames
   (a slot change with almost no gap). The hazard is solved so the mean number of tracked objects
   on graded frames lands on `target_present_mean`.
3. Duplicates: ghost copies of a tracked player, offset by ~`dup_offset_sd_m`, for short bursts.
4. Slots: every tracklet takes a random free slot of its team's pool (`slots_per_team`), so
   slots end, restart and are re-used. Slot team never changes, as in the ATD metadata.
5. Positional noise: AR(1) noise of sd `pos_noise_m` (players) and `ball_noise_m` (ball).
6. Ball, dead stretches: inside every interior NaN run of the clean ball, a synthetic ball sits
   where it was last seen, is carried to where it reappears, and waits there. By default
   (`dead_ball_pattern="ends"`) each stoppage then shows it at its start and end only and hides
   the middle, about `dead_ball_visible` of the frames visible; "bursts" hides it in short bursts
   instead.
7. Ball, live play: bursts of missing ball start with a per-frame rate weighted by context (fast
   ball as an in-the-air proxy, crowding, near a line), with the base rate solved so that the
   ball is missing on `target_ball_missing` of graded frames overall.
"""

from dataclasses import asdict, dataclass, field, replace
from typing import Optional

import numpy as np
from scipy.signal import lfilter

from .frames import AWAY, HOME, FrameTable


@dataclass
class DegradeConfig:
    seed: int = 0
    slots_per_team: int = 23                 # 46 slots in all, as in the BAR-ATH tactical ATD
    # ball
    target_ball_missing: float = 0.41        # share of graded frames with no ball (DATA.md)
    # ASSUMPTION, the one DATA.md cannot pin down: share of dead-ball frames showing a ball.
    # 0.25 was picked from the coarse grid {0, 0.25, 0.5, 0.75} as the only point where the
    # organiser baselines' macroF1 behaves like their published full-match numbers (rises from
    # raw to smooth25/50, drops at smooth125, levels within ~0.05). Public aggregates, not ground
    # truth; replace with a measurement from the real ATD (docs/phaseA_report.md).
    dead_ball_visible: float = 0.25
    # "ends": each stoppage shows the ball at its start (going out) and end (placed for the
    # restart) and hides the middle; "bursts": hidden in the same short bursts as live play.
    dead_ball_pattern: str = "ends"
    dead_visible_concentration: float = 4.0  # Beta concentration of the per-stoppage visible share
    # "linear": held where last seen, carried in a straight line between 35% and 65% of the
    # stoppage, held where it reappears (Phase A). "random": per stoppage one of three shapes with
    # random timings (straight line, carried by the player who restarts, wandering through random
    # waypoints), so a learned model cannot key on one synthetic trajectory (phaseA_report.md,
    # found-not-fixed 3).
    dead_ball_path: str = "linear"
    # True: before that path, the ball rolls on from its last live velocity and slows to rest over
    # 0.5-3 s. Without it the synthetic ball stops dead on the first dead frame, a frame-exact
    # "play has stopped" signal no real tracking gives (docs/model_log.md, gbm boundary check).
    dead_ball_rollout: bool = False
    ball_burst_median: float = 20.0          # frames; lognormal burst lengths ...
    ball_burst_sigma: float = 1.0            # ... (mean = median * exp(sigma^2 / 2) ~ 33 frames)
    ball_burst_lengths: Optional[np.ndarray] = None   # empirical lengths, overrides the lognormal
    air_speed_mps: float = 15.0              # ball faster than this counts as "in the air"
    air_weight: float = 3.0
    crowd_ball_radius_m: float = 3.0
    crowd_ball_min: int = 3                  # at least this many players near the ball = crowded
    crowd_weight: float = 2.0
    line_distance_m: float = 3.0             # ball this close to a touchline or goal line
    line_weight: float = 2.0
    ball_noise_m: float = 0.20
    # Phase B2 ("joint", the atd_cal variant): one ball-gap process over all graded time instead
    # of live bursts plus a separate dead-ball pattern. The real ATD's gaps are long (median 268
    # frames) and run across stoppages and restarts (docs/phaseB2_report.md), so a gap starts at
    # a visible frame with a context-weighted hazard, takes its length from `ball_burst_lengths`
    # (the real gap lengths), and, with probability restart_reacq_prob, ends a lognormal delay
    # after the next restart it runs into. Two base rates are solved: dead-stretch onsets so that
    # 1 - dead_ball_visible of dead frames are hidden, and live onsets for target_ball_missing.
    ball_mode: str = "split"
    dead_onset_head_frames: int = 50         # dead onsets are more likely just after play stops
    dead_onset_head_weight: float = 3.0
    restart_reacq_prob: float = 0.8
    restart_reacq_median: float = 75.0       # frames after the restart
    restart_reacq_sigma: float = 0.6
    slow_mps: float = 2.0                    # live onset weights (joint mode only)
    slow_weight: float = 1.0
    fast_mps: float = 10.0
    fast_weight: float = 1.0
    deep_m: float = 45.0                     # |x - halfway| beyond this, near a goal line
    deep_weight: float = 1.0
    # players
    # mass dropouts (atd_cal): a share of the tracked players vanish together for a while (the
    # real ATD has 4% of graded frames with under 14 objects, in runs of median 67 frames)
    mass_drop_rate_per_min: float = 0.0
    mass_drop_median: float = 67.0
    mass_drop_sigma: float = 1.5
    mass_drop_frac_lo: float = 0.3
    mass_drop_frac_hi: float = 0.7
    mass_drop_dead_weight: float = 4.0       # rate multiplier inside dead stretches
    slot_policy: str = "random"              # "lowest": recycle the lowest free slot (atd_cal)
    smooth_frames: int = 1                   # centred moving average of the clean input first
    noise_smooth_frames: int = 1             # >1: smooth the AR(1) error (slow drift, no jitter)
    min_gap_frames: int = 1                  # joint mode: a gap cut shorter than this by a cap is dropped
    # Phase E (atd_cal2): the shape of the shown ball's error. The real ATD ball bends less at a
    # touch (kink4 p99 5.9 m/s against atd_cal's 10.5) and wobbles more between touches (kink4 p50
    # 0.91 against 0.36), so the shown ball gets a further zero-phase smoothing (a centred moving
    # average of ball_post_smooth_frames, applied ball_post_smooth_passes times) and a
    # mid-frequency wobble (white noise smoothed the same way over ball_wobble_frames, rescaled to
    # sd ball_wobble_m per axis), and may lag the truth by ball_lag_frames (linear interpolation;
    # the video alignment puts it at 0). The players are untouched. Own random stream, so every
    # earlier variant draws exactly what it drew before.
    ball_post_smooth_frames: int = 1
    ball_post_smooth_passes: int = 1
    ball_wobble_m: float = 0.0
    ball_wobble_frames: int = 1
    ball_wobble_passes: int = 1
    ball_lag_frames: float = 0.0
    # Phase E: a simulated video ball track, a separate flagged channel (FrameTable.vball) shown
    # only where the ATD ball is hidden, as our own detector fills the real ATD's gaps
    # (docs/phaseE_report.md). True-ball fills come in bursts with the real fill bursts' lengths,
    # start with a hazard weighted by the true ball's speed (video_speed_weights per bin of
    # video_speed_edges m/s; never above the last edge), stop when the ball gets fast or the gap
    # ends, and carry a per-burst error drawn from the real video-minus-ATD error vectors of the
    # same speed bin (lofted balls back-project too far, so fast bins carry the big errors) plus an
    # AR(1) residual. False "spare balls": static balls just inside or outside a line (up to
    # video_spare_margin_m outside, the detector's 1.94 m touchline margin) with no player within
    # 5 m, on video_spare_share of hidden frames. video_fill_share is the share of hidden graded
    # frames given a true ball. "real" calibration = video.props.real_video_calibration() (a
    # cached distribution); a dict with the same keys can be passed instead (tests).
    video_fill_share: float = 0.0
    video_spare_share: float = 0.0
    video_speed_edges: tuple = (0.5, 3.0, 8.0, 15.0)
    video_speed_weights: tuple = (1.0, 0.5, 0.3, 0.15)
    video_error_scale: float = 1.0
    video_resid_m: float = 0.3
    video_resid_rho: float = 0.999
    video_spare_margin_m: float = 1.94
    video_spare_player_m: float = 5.0
    video_margin_m: float = 3.0              # the detector keeps candidates within this (x) of the
                                             # pitch, normalised so y gets margin * width / length
    video_calibration: object = "real"
    dead_out_weight: float = 1.0             # joint mode: dead onsets with the ball off the pitch
    target_present_mean: float = 20.5        # tracked objects per graded frame (DATA.md)
    gap_median: float = 25.0                 # frames, lognormal gap after a lost track
    gap_sigma: float = 1.0
    reacquire_prob: float = 0.3              # breaks re-acquired within 1-3 frames
    crowd_radius_m: float = 1.5              # another player this close raises the end hazard
    crowd_hazard_boost: float = 2.0
    swap_distance_m: float = 1.5
    swap_prob: float = 0.3
    dup_rate_per_min: float = 1.5            # duplicate bursts per graded minute, both teams
    dup_median: float = 25.0
    dup_sigma: float = 0.8
    dup_offset_sd_m: float = 0.8
    pos_noise_m: float = 0.15
    noise_rho: float = 0.9                   # AR(1) coefficient per frame

    def as_dict(self):
        d = asdict(self)
        if d["ball_burst_lengths"] is not None:
            d["ball_burst_lengths"] = "empirical[{}]".format(len(self.ball_burst_lengths))
        if not isinstance(d["video_calibration"], str):
            d["video_calibration"] = "custom"
        return d


def make_config(overrides=None, **kw):
    """DegradeConfig() with overrides. `ball_burst_lengths="atd"` is resolved to the real ATD's
    missing-ball run lengths (atd_profile.real_gap_lengths: a cached distribution, never frames)."""
    o = dict(overrides or {})
    o.update(kw)
    if isinstance(o.get("ball_burst_lengths"), str):
        if o["ball_burst_lengths"] != "atd":
            raise ValueError("ball_burst_lengths must be an array or 'atd'")
        from .atd_profile import real_gap_lengths
        o["ball_burst_lengths"] = real_gap_lengths()
    return replace(DegradeConfig(), **o)


@dataclass
class DegradeReport:
    ball_missing_graded: float = 0.0
    ball_missing_alive: float = 0.0          # on frames where the clean ball is visible
    ball_visible_dead: float = 0.0           # on frames where the clean ball is NaN (interior)
    present_mean_graded: float = 0.0
    frames_over_22: float = 0.0
    n_tracklets: int = 0
    n_swaps: int = 0
    n_duplicates: int = 0
    n_dropped_tracklets: int = 0
    alive_target: float = 0.0
    extra: dict = field(default_factory=dict)


# -- helpers ---------------------------------------------------------------------------------------

def _runs(mask):
    """[(start, stop)] of True runs."""
    m = np.asarray(mask, dtype=np.int8)
    edges = np.diff(np.concatenate(([0], m, [0])))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def _lognormal_int(rng, median, sigma, size, lo=1):
    return np.maximum(lo, np.round(rng.lognormal(np.log(median), sigma, size))).astype(np.int64)


def _ar1(rng, shape, sd, rho):
    eps = rng.standard_normal(shape)
    return lfilter([sd * np.sqrt(1.0 - rho * rho)], [1.0, -rho], eps, axis=0)


def _noise(rng, shape, sd, cfg):
    """AR(1) position error; with noise_smooth_frames > 1 also smoothed by a centred moving
    average and rescaled to sd, so the error is a slow drift with almost no frame-to-frame jitter
    (the real ATD's jitter is under 0.05 m). Same draws either way."""
    e = _ar1(rng, shape, sd, cfg.noise_rho)
    k = cfg.noise_smooth_frames
    if k > 1 and sd > 0:
        c = np.cumsum(np.concatenate([np.zeros((1,) + e.shape[1:]), e]), axis=0)
        n, h = e.shape[0], k // 2
        i = np.arange(n)
        lo, hi = np.maximum(0, i - h), np.minimum(n, i + h + 1)
        m = (c[hi] - c[lo]) / (hi - lo).reshape((-1,) + (1,) * (e.ndim - 1))
        e = m * (sd / max(1e-12, float(m.std())))
    return e


def _run_end_index(eligible):
    """For each frame, the stop index of the eligible run containing it (n where ineligible)."""
    n = len(eligible)
    stop = np.full(n, n, dtype=np.int64)
    for s, e in _runs(eligible):
        stop[s:e] = e
    return stop


def _burst_mask(eligible, weights, lengths, u, base):
    """Missing mask from bursts that start at eligible frame t when u[t] < base * w[t].

    Bursts never leave the eligible run they start in, and a start inside an earlier burst is
    ignored. u and lengths are drawn once, so the result is deterministic and monotone in base.
    """
    n = len(eligible)
    out = np.zeros(n, dtype=bool)
    starts = np.flatnonzero(eligible & (u < base * weights))
    if not len(starts):
        return out
    stop_of = _run_end_index(eligible)
    covered = -1
    for t in starts.tolist():
        if t < covered:
            continue
        e = min(t + int(lengths[t]), int(stop_of[t]))
        out[t:e] = True
        covered = e
    return out


def _calibrated_bursts(eligible, weights, lengths, u, target, count_mask=None):
    """Bisect the base rate so the missing share over count_mask & eligible hits target."""
    cm = eligible if count_mask is None else (eligible & count_mask)
    denom = cm.sum()
    if denom == 0 or target <= 0:
        return np.zeros(len(eligible), dtype=bool), 0.0
    lo, hi = 0.0, 1.0 / max(1e-9, float(weights[eligible].min()))
    best, best_err = None, np.inf
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        m = _burst_mask(eligible, weights, lengths, u, mid)
        share = m[cm].sum() / denom
        if abs(share - target) < best_err:
            best, best_err = (m, mid), abs(share - target)
        if best_err < 5e-4:
            break
        if share < target:
            lo = mid
        else:
            hi = mid
    return best


# -- player side -----------------------------------------------------------------------------------

def _team_swaps(xy_m, team_idx, cfg, rng):
    """Chronological swap list [(frame, player_a, player_b)] within one team (player indices)."""
    swaps = []
    k = len(team_idx)
    for a in range(k):
        for b in range(a + 1, k):
            d = np.linalg.norm(xy_m[:, team_idx[a]] - xy_m[:, team_idx[b]], axis=1)
            close = np.nan_to_num(d, nan=np.inf) < cfg.swap_distance_m
            for s, e in _runs(close):
                if rng.random() < cfg.swap_prob:
                    f = s + int(np.argmin(d[s:e]))
                    swaps.append((f, team_idx[a], team_idx[b]))
    swaps.sort()
    return swaps


def _identity_tracks(xy_m, players, swaps):
    """(n, len(players), 2) positions per tracker identity after applying swaps in order."""
    n = xy_m.shape[0]
    ident = list(players)                    # identity j currently follows player ident[j]
    pos = {p: j for j, p in enumerate(players)}
    out = np.empty((n, len(players), 2), dtype=np.float64)
    cursor = 0
    for f, pa, pb in swaps + [(n, None, None)]:
        for j, p in enumerate(ident):
            out[cursor:f, j] = xy_m[cursor:f, p]
        cursor = f
        if pa is None:
            break
        ja, jb = pos[pa], pos[pb]
        ident[ja], ident[jb] = pb, pa
        pos[pa], pos[pb] = jb, ja
    return out


def _tracklets(track, hazard, cfg, rng):
    """[(start, stop)] tracked spans of one identity track (n, 2)."""
    present = ~np.isnan(track[:, 0])
    out = []
    cum = np.concatenate(([0.0], np.cumsum(hazard)))
    for s, e in _runs(present):
        t = s
        while t < e:
            need = rng.exponential(1.0)
            # first frame where accumulated hazard since t exceeds `need`
            end = int(np.searchsorted(cum, cum[t] + need, side="left"))
            end = min(max(end, t + 1), e)
            out.append((t, end))
            if end >= e:
                break
            if rng.random() < cfg.reacquire_prob:
                gap = int(rng.integers(1, 4))
            else:
                gap = int(_lognormal_int(rng, cfg.gap_median, cfg.gap_sigma, 1)[0])
            t = end + gap
    return out


def _assign_slots(tracklets, n_slots, rng, policy="random"):
    """tracklets: [(start, stop, payload)] -> [(slot, start, stop, payload)], dropped count.

    policy "random": any free slot (Phase A); "lowest": the lowest-numbered free slot, as a
    tracker that recycles ids does (slots then rest for short stretches, as in the real ATD)."""
    free_from = np.zeros(n_slots, dtype=np.int64)
    out, dropped = [], 0
    for start, stop, payload in sorted(tracklets, key=lambda x: (x[0], x[1])):
        free = np.flatnonzero(free_from <= start)
        if not len(free):
            dropped += 1
            continue
        slot = int(free[0]) if policy == "lowest" else int(rng.choice(free))
        free_from[slot] = stop + 1           # a slot rests one frame before re-use
        out.append((slot, start, stop, payload))
    return out, dropped


# -- main --------------------------------------------------------------------------------------------

def degrade(clean, graded=None, config=None):
    """Degrade a clean FrameTable. `graded` (bool, n) marks the frames the targets refer to.

    Returns (FrameTable with 2 * slots_per_team anonymous slots, DegradeReport).
    """
    cfg = config or DegradeConfig()
    rng = np.random.default_rng(cfg.seed)
    n = clean.n_frames
    graded = np.ones(n, dtype=bool) if graded is None else np.asarray(graded, dtype=bool)
    length, width = clean.pitch
    scale = np.array([length, width])
    xy_m = clean.xy.astype(np.float64) * scale
    ball_m = clean.ball.astype(np.float64) * scale
    if cfg.smooth_frames > 1:
        # the real ATD is smooth frame to frame (jitter under 0.05 m); the sample tracking's own
        # ball jitter is about 0.18 m, so smooth it first and add only low-frequency error after
        xy_m = _smooth_nan(xy_m, cfg.smooth_frames)
        ball_m = _smooth_nan(ball_m, cfg.smooth_frames)
    rep = DegradeReport()

    # crowding per player (any other player within crowd_radius_m), for the end hazard
    present = ~np.isnan(xy_m[:, :, 0])
    crowd = np.zeros(present.shape, dtype=bool)
    for i in range(xy_m.shape[1]):
        d = np.linalg.norm(xy_m - xy_m[:, i:i + 1], axis=2)
        d[:, i] = np.inf
        crowd[:, i] = (np.nan_to_num(d, nan=np.inf) < cfg.crowd_radius_m).any(axis=1)

    # players: duplicates add to the count, so solve the missing share net of them
    graded_min = graded.sum() / 25.0 / 60.0
    dup_mean_len = cfg.dup_median * np.exp(cfg.dup_sigma ** 2 / 2)
    n_dups_expected = cfg.dup_rate_per_min * graded_min
    dup_per_frame = n_dups_expected * dup_mean_len / max(1, graded.sum())
    on_pitch = present[graded].sum(axis=1).mean() if graded.any() else 22.0
    mass_len_mean = cfg.mass_drop_median * np.exp(cfg.mass_drop_sigma ** 2 / 2)
    mass_share = (cfg.mass_drop_rate_per_min / (25.0 * 60.0) * mass_len_mean
                  * 0.5 * (cfg.mass_drop_frac_lo + cfg.mass_drop_frac_hi))
    miss = np.clip((on_pitch + dup_per_frame - cfg.target_present_mean) / on_pitch - mass_share,
                   0.0, 0.9)
    gap_mean = (cfg.reacquire_prob * 2.0
                + (1 - cfg.reacquire_prob) * cfg.gap_median * np.exp(cfg.gap_sigma ** 2 / 2))
    h_mean = miss / max(1e-9, (1.0 - miss) * gap_mean)

    slots_xy = np.full((n, 2 * cfg.slots_per_team, 2), np.nan, dtype=np.float64)
    slot_team = np.array([HOME] * cfg.slots_per_team + [AWAY] * cfg.slots_per_team, dtype=np.int8)
    for team in (HOME, AWAY):
        team_idx = [int(i) for i in np.flatnonzero(clean.slot_team == team)]
        swaps = _team_swaps(xy_m, team_idx, cfg, rng)
        rep.n_swaps += len(swaps)
        tracks = _identity_tracks(xy_m, team_idx, swaps)
        # identity j's crowd flag follows whichever player it is on; approximate with player j's
        w = 1.0 + (cfg.crowd_hazard_boost - 1.0) * crowd[:, team_idx]
        w_mean = w[graded].mean() if graded.any() else w.mean()
        items = []
        for j in range(len(team_idx)):
            hz = h_mean * w[:, j] / w_mean
            for s, e in _tracklets(tracks[:, j], hz, cfg, rng):
                items.append((s, e, ("real", j, None)))
        # duplicates: Poisson count per team, half the total rate each
        n_dup = rng.poisson(n_dups_expected / 2.0)
        cand = np.flatnonzero(graded)
        for _ in range(n_dup):
            if not len(cand):
                break
            t0 = int(rng.choice(cand))
            here = np.flatnonzero(~np.isnan(tracks[t0, :, 0]))
            if not len(here):
                continue
            j = int(rng.choice(here))
            L = int(_lognormal_int(rng, cfg.dup_median, cfg.dup_sigma, 1)[0])
            t1 = min(n, t0 + L)
            off = rng.normal(0.0, cfg.dup_offset_sd_m, 2)
            items.append((t0, t1, ("dup", j, off)))
            rep.n_duplicates += 1
        base = 0 if team == HOME else cfg.slots_per_team
        assigned, dropped = _assign_slots(items, cfg.slots_per_team, rng, cfg.slot_policy)
        rep.n_dropped_tracklets += dropped
        for slot, s, e, (kind, j, off) in assigned:
            seg = tracks[s:e, j].copy()
            if kind == "dup":
                seg = seg + off
            else:
                rep.n_tracklets += 1
            slots_xy[s:e, base + slot] = seg

    if cfg.mass_drop_rate_per_min > 0:
        _mass_dropouts(slots_xy, graded, np.isnan(ball_m[:, 0]), cfg,
                       np.random.default_rng([cfg.seed, 13]))
    slots_xy += _noise(rng, slots_xy.shape, cfg.pos_noise_m, cfg)

    # ball, dead stretches: synthetic path inside interior NaN runs of the clean ball
    vis = ~np.isnan(ball_m[:, 0])
    ball_syn = ball_m.copy()
    if cfg.dead_ball_path not in ("linear", "random"):
        raise ValueError("dead_ball_path must be 'linear' or 'random'")
    # its own stream, so "linear" (the Phase A default) draws exactly what it always drew
    path_rng = np.random.default_rng([cfg.seed, 7])
    roll_rng = np.random.default_rng([cfg.seed, 11])
    dead_eligible = np.zeros(n, dtype=bool)
    for s, e in _runs(~vis):
        if s == 0 or e >= n:
            continue                          # before the first / after the last sighting
        p0, p1 = ball_m[s - 1], ball_m[e]
        s0 = s
        if cfg.dead_ball_rollout:
            # the ball keeps rolling past the moment play stops, slowing to rest, instead of
            # freezing on the frame the clean tracking drops it
            w0 = max(0, s - 6)
            v = (ball_m[s - 1] - ball_m[w0]) / max(1, s - 1 - w0)
            if not np.isfinite(v).all():
                v = np.zeros(2)
            R = min(int(roll_rng.uniform(0.5, 3.0) * 25), (e - s) // 2)
            if R > 0:
                k = 3.0 / R
                tt = np.arange(1, R + 1, dtype=np.float64)[:, None]
                ball_syn[s:s + R] = p0 + v * (1.0 - np.exp(-k * tt)) / k
                p0 = ball_syn[s + R - 1].copy()
                s0 = s + R
        L = e - s0
        if L > 0:
            if cfg.dead_ball_path == "random":
                ball_syn[s0:e] = _random_dead_path(p0, p1, s0, e, xy_m, path_rng)
            else:
                a, b = int(0.35 * L), int(0.65 * L)
                frac = np.clip((np.arange(L) - a) / max(1, b - a), 0.0, 1.0)[:, None]
                ball_syn[s0:e] = p0 + (p1 - p0) * frac
        dead_eligible[s:e] = True
    if cfg.ball_mode == "joint":
        hidden = _joint_gaps(vis, dead_eligible, graded, ball_m, xy_m, cfg, length, width,
                             np.random.default_rng([cfg.seed, 17]), rep)
        ball_out = _ball_shape(ball_syn, cfg) + _noise(rng, ball_syn.shape, cfg.ball_noise_m, cfg)
        ball_out = ball_out + _wobble(cfg, ball_syn.shape)
        ball_out[hidden] = np.nan
        vball = None
        if cfg.video_fill_share > 0 or cfg.video_spare_share > 0:
            vball = _video_fill(ball_syn, hidden & graded, slots_xy, cfg, length, width, rep)
        return _finish(clean, cfg, rep, slots_xy, slot_team, ball_out, hidden, graded,
                       dead_eligible, scale, vball)
    if cfg.ball_mode != "split":
        raise ValueError("ball_mode must be 'split' or 'joint'")
    if cfg.dead_ball_pattern == "ends":
        dead_missing = _dead_ends_mask(_runs(dead_eligible), n, cfg, rng)
    elif cfg.dead_ball_pattern == "bursts":
        u_dead = rng.random(n)
        len_dead = _burst_len_array(rng, cfg, n)
        dead_missing, _ = _calibrated_bursts(dead_eligible, np.ones(n), len_dead, u_dead,
                                             1.0 - cfg.dead_ball_visible, graded)
    else:
        raise ValueError("dead_ball_pattern must be 'ends' or 'bursts'")
    dead_hidden = (~vis & ~dead_eligible) | dead_missing

    # ball, live play: context-weighted bursts calibrated to the overall graded target
    weights = _ball_context_weights(ball_m, xy_m, cfg, length, width)
    live = vis.copy()
    n_graded = graded.sum()
    already = (dead_hidden & graded).sum()
    need = cfg.target_ball_missing * n_graded - already
    alive_target = float(np.clip(need / max(1, (live & graded).sum()), 0.0, 0.95))
    rep.alive_target = alive_target
    u_live = rng.random(n)
    len_live = _burst_len_array(rng, cfg, n)
    live_missing, _ = _calibrated_bursts(live, weights, len_live, u_live, alive_target, graded)

    ball_out = _ball_shape(ball_syn, cfg) + _noise(rng, ball_syn.shape, cfg.ball_noise_m, cfg)
    ball_out = ball_out + _wobble(cfg, ball_syn.shape)
    hidden = dead_hidden | live_missing
    ball_out[hidden] = np.nan

    # report
    rep.ball_missing_alive = float(live_missing[live & graded].mean()) if (live & graded).any() else 0.0
    rep.ball_visible_dead = (float(1.0 - dead_missing[dead_eligible & graded].mean())
                             if (dead_eligible & graded).any() else 0.0)
    return _finish(clean, cfg, rep, slots_xy, slot_team, ball_out, hidden, graded, None, scale)


def _finish(clean, cfg, rep, slots_xy, slot_team, ball_out, hidden, graded, dead, scale, vball=None):
    n_graded = graded.sum()
    out_present = ~np.isnan(slots_xy[:, :, 0])
    rep.ball_missing_graded = float(hidden[graded].mean()) if n_graded else 0.0
    if dead is not None:
        live = ~dead & ~np.isnan(clean.ball[:, 0])
        rep.ball_missing_alive = float(hidden[live & graded].mean()) if (live & graded).any() else 0.0
        rep.ball_visible_dead = (float(1.0 - hidden[dead & graded].mean())
                                 if (dead & graded).any() else 0.0)
    counts = out_present[graded].sum(axis=1)
    rep.present_mean_graded = float(counts.mean()) if n_graded else 0.0
    rep.frames_over_22 = float((counts > 22).mean()) if n_graded else 0.0

    table = FrameTable(clean.frame.copy(), clean.period.copy(),
                       (ball_out / scale).astype(np.float32),
                       (slots_xy / scale).astype(np.float32), slot_team, tuple(clean.pitch),
                       None, {"game": clean.meta.get("game"), "kind": "degraded",
                              "config": cfg.as_dict()},
                       None if vball is None else (vball / scale).astype(np.float32))
    return table, rep


def _ma_passes(a, k, passes):
    """Centred k-frame moving average along axis 0, `passes` times (NaN-aware, NaN stays NaN)."""
    if k <= 1:
        return a
    for _ in range(max(1, passes)):
        a = _smooth_nan(a, k)
    return a


def _ball_shape(ball_syn, cfg):
    """The shown ball before noise: further smoothing and a lag (identity with the defaults)."""
    out = _ma_passes(ball_syn, cfg.ball_post_smooth_frames, cfg.ball_post_smooth_passes)
    lag = float(cfg.ball_lag_frames)
    if lag:
        n = len(out)
        t = np.arange(n, dtype=np.float64) - lag
        i0 = np.clip(np.floor(t).astype(np.int64), 0, n - 1)
        i1 = np.clip(i0 + 1, 0, n - 1)
        w = np.clip(t - np.floor(t), 0.0, 1.0)[:, None]
        out = (1 - w) * out[i0] + w * out[i1]
    return out


def _wobble(cfg, shape):
    """Mid-frequency ball error: smoothed white noise with sd ball_wobble_m per axis (0 = none)."""
    if cfg.ball_wobble_m <= 0:
        return 0.0
    e = np.random.default_rng([cfg.seed, 23]).standard_normal(shape)
    e = _ma_passes(e, cfg.ball_wobble_frames, cfg.ball_wobble_passes)
    return e * (cfg.ball_wobble_m / max(1e-12, float(e.std())))


def _random_dead_path(p0, p1, s, e, xy_m, rng):
    """(e - s, 2) synthetic dead-ball path from p0 (last seen) to p1 (where it reappears)."""
    L = e - s
    t = np.arange(L, dtype=np.float64)
    mode = rng.choice(3, p=(0.3, 0.4, 0.3))
    if mode == 0 or L < 10:                                      # straight, random timings
        a = rng.uniform(0.0, 0.6) * L
        b = a + rng.uniform(0.1, 1.0) * (L - a)
        frac = np.clip((t - a) / max(1.0, b - a), 0.0, 1.0)[:, None]
        return p0 + (p1 - p0) * frac
    if mode == 1:                                                # carried by the restart taker
        d = np.linalg.norm(xy_m[min(e, len(xy_m) - 1)] - p1, axis=1)
        d = np.nan_to_num(d, nan=np.inf)
        j = int(np.argmin(d))
        taker = xy_m[s:e, j].copy()
        ok = ~np.isnan(taker[:, 0])
        if not ok.any():
            taker[:] = p1
        else:
            idx = np.flatnonzero(ok)
            for c in range(2):
                taker[:, c] = np.interp(t, idx, taker[idx, c])
        taker += rng.normal(0.0, 0.5, 2)
        a = rng.uniform(0.0, 0.4) * L                             # still where it went out
        b = a + rng.uniform(0.05, 0.3) * L                        # picked up by b
        c_ = b + rng.uniform(0.3, 1.0) * (L - b)                  # placed for the restart by c_
        out = np.empty((L, 2))
        w = np.clip((t - a) / max(1.0, b - a), 0.0, 1.0)[:, None]
        out[:] = p0 + (taker - p0) * w
        put = t >= c_
        w2 = np.clip((t - c_) / max(1.0, 0.1 * L), 0.0, 1.0)[:, None]
        out[put] = (taker + (p1 - taker) * w2)[put]
        return out
    k = int(rng.integers(1, 4))                                  # wandering through waypoints
    pts = [p0] + [p0 + (p1 - p0) * rng.uniform(0, 1) + rng.normal(0.0, 6.0, 2)
                  for _ in range(k)] + [p1]
    knots = np.sort(rng.uniform(0, L, len(pts)))
    knots[0], knots[-1] = rng.uniform(0, 0.3) * L, L - rng.uniform(0, 0.3) * L
    knots = np.maximum.accumulate(knots)
    pts = np.array(pts)
    return np.stack([np.interp(t, knots, pts[:, c]) for c in range(2)], axis=1)


def _dead_ends_mask(runs, n, cfg, rng):
    """Hidden mask for "ends": per stoppage a visible head and tail, hidden middle.

    The visible share of each stoppage is Beta-distributed around dead_ball_visible, then all
    shares are scaled together (clipped to [0, 1]) so the frame-weighted share over all stoppages
    is dead_ball_visible, and each is split between head and tail by a uniform 20-80% draw.
    """
    hidden = np.zeros(n, dtype=bool)
    if not runs:
        return hidden
    v = float(np.clip(cfg.dead_ball_visible, 0.0, 1.0))
    k = cfg.dead_visible_concentration
    lengths = np.array([e - s for s, e in runs], dtype=np.float64)
    if 0.0 < v < 1.0:
        f = rng.beta(v * k, (1.0 - v) * k, len(runs))
        lo, hi = 0.0, 1.0 / max(1e-9, f.min()) + 1.0
        for _ in range(60):
            c = 0.5 * (lo + hi)
            share = (np.clip(f * c, 0, 1) * lengths).sum() / lengths.sum()
            lo, hi = (c, hi) if share < v else (lo, c)
        f = np.clip(f * 0.5 * (lo + hi), 0, 1)
    else:
        f = np.full(len(runs), v)
    split = rng.uniform(0.2, 0.8, len(runs))
    for (s, e), fi, sp in zip(runs, f, split):
        n_vis = int(round(fi * (e - s)))
        head = int(round(sp * n_vis))
        tail = n_vis - head
        hidden[s + head:e - tail] = True
    return hidden


def _burst_len_array(rng, cfg, n):
    if cfg.ball_burst_lengths is not None:
        pool = np.asarray(cfg.ball_burst_lengths, dtype=np.int64)
        return rng.choice(pool[pool > 0], n)
    return _lognormal_int(rng, cfg.ball_burst_median, cfg.ball_burst_sigma, n)


def _ball_context_weights(ball_m, xy_m, cfg, length, width):
    n = len(ball_m)
    vel = np.full(n, 0.0)
    v = np.linalg.norm(ball_m[2:] - ball_m[:-2], axis=1) * 25.0 / 2.0
    vel[1:-1] = np.nan_to_num(v, nan=0.0)
    vel = np.convolve(vel, np.ones(5) / 5.0, mode="same")
    air = vel > cfg.air_speed_mps
    d = np.linalg.norm(xy_m - ball_m[:, None, :], axis=2)
    crowded = (np.nan_to_num(d, nan=np.inf) < cfg.crowd_ball_radius_m).sum(axis=1) >= cfg.crowd_ball_min
    bx, by = ball_m[:, 0], ball_m[:, 1]
    with np.errstate(invalid="ignore"):
        near = ((by < cfg.line_distance_m) | (by > width - cfg.line_distance_m)
                | (bx < cfg.line_distance_m) | (bx > length - cfg.line_distance_m))
    near = np.nan_to_num(near, nan=False).astype(bool)
    return (cfg.air_weight ** air) * (cfg.crowd_weight ** crowded) * (cfg.line_weight ** near)


# -- Phase B2: joint ball gaps and mass player dropouts ------------------------------------------

def _gap_mask(onset_ok, weights, lengths, u, base, cap, min_len=1):
    """Gaps starting at frame t where onset_ok[t] and u[t] < base * w[t]; a gap starting at t
    ends at min(t + lengths[t], cap[t]). Starts inside an earlier gap are ignored."""
    n = len(onset_ok)
    out = np.zeros(n, dtype=bool)
    starts = np.flatnonzero(onset_ok & (u < base * weights))
    covered = -1
    for t in starts.tolist():
        if t < covered:
            continue
        e = min(t + int(lengths[t]), int(cap[t]))
        if e - t >= min(min_len, int(lengths[t])):
            out[t:e] = True
            covered = e
    return out


def _solve_base(fn, measure, target, hi):
    """Bisect base in [0, hi] so measure(fn(base)) ~ target (measure increases with base)."""
    lo = 0.0
    best, best_err = fn(0.0), abs(measure(fn(0.0)) - target)
    for _ in range(36):
        mid = 0.5 * (lo + hi)
        m = fn(mid)
        v = measure(m)
        if abs(v - target) < best_err:
            best, best_err = m, abs(v - target)
        if best_err < 5e-4:
            break
        lo, hi = (mid, hi) if v < target else (lo, mid)
    return best


def _smooth_nan(a, k):
    """Centred k-frame mean along axis 0 over runs of non-NaN values (NaN stays NaN)."""
    ok = ~np.isnan(a)
    v = np.where(ok, a, 0.0)
    c = np.concatenate([np.zeros((1,) + a.shape[1:]), np.cumsum(v, axis=0)])
    cn = np.concatenate([np.zeros((1,) + a.shape[1:]), np.cumsum(ok, axis=0)])
    n, h = a.shape[0], k // 2
    i = np.arange(n)
    lo, hi = np.maximum(0, i - h), np.minimum(n, i + h + 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (c[hi] - c[lo]) / (cn[hi] - cn[lo])
    return np.where(ok, out, np.nan)


def _joint_gaps(vis, dead, graded, ball_m, xy_m, cfg, length, width, rng, rep):
    """Hidden-ball mask for ball_mode="joint" (see DegradeConfig)."""
    n = len(vis)
    blocks = _runs(graded)
    block_end = np.full(n, 0, dtype=np.int64)
    for a, b in blocks:
        block_end[a:b] = b
    # restart caps: a gap running into a restart ends a delay after it (per stoppage, a coin flip)
    cap = block_end.copy()
    nxt = np.full(n + 1, np.iinfo(np.int64).max, dtype=np.int64)
    for s, e in _runs(dead):
        if rng.random() < cfg.restart_reacq_prob:
            d = int(_lognormal_int(rng, cfg.restart_reacq_median, cfg.restart_reacq_sigma, 1)[0])
            nxt[s:e] = e + d                 # a gap starting inside the stoppage
            nxt[e] = min(nxt[e], e + d)
    # a gap starting in live play before a stoppage runs into that stoppage's restart cap
    run_min = nxt[::-1].copy()
    np.minimum.accumulate(run_min, out=run_min)
    cap = np.minimum(cap, run_min[::-1][:n])
    lengths = _burst_len_array(rng, cfg, n)
    u_dead, u_live = rng.random(n), rng.random(n)
    # dead onsets: anywhere in a stoppage, more likely in its first frames
    w_dead = np.ones(n)
    for s, e in _runs(dead):
        w_dead[s:min(e, s + cfg.dead_onset_head_frames)] = cfg.dead_onset_head_weight
    with np.errstate(invalid="ignore"):
        outside = ((ball_m[:, 0] < -1) | (ball_m[:, 0] > length + 1)
                   | (ball_m[:, 1] < -1) | (ball_m[:, 1] > width + 1))
    w_dead = w_dead * np.where(outside, cfg.dead_out_weight, 1.0)
    dead_ok = dead & graded
    # live onsets: context weights measured on the real ATD (docs/phaseB2_report.md)
    w_live = _joint_live_weights(ball_m, xy_m, cfg, length, width)
    live_ok = vis & ~dead & graded
    g = graded
    dg = dead & graded
    target_dead = 1.0 - cfg.dead_ball_visible
    live_mask = np.zeros(n, dtype=bool)
    dead_mask = np.zeros(n, dtype=bool)
    for _ in range(2):
        def dead_fn(b):
            return _gap_mask(dead_ok & ~live_mask, w_dead, lengths, u_dead, b, cap,
                             cfg.min_gap_frames)
        dead_mask = (_solve_base(dead_fn, lambda m: (m | live_mask)[dg].mean(), target_dead, 1.0)
                     if dg.any() else dead_mask)

        def live_fn(b):
            return _gap_mask(live_ok & ~dead_mask, w_live, lengths, u_live, b, cap,
                             cfg.min_gap_frames)
        live_mask = _solve_base(live_fn, lambda m: (m | dead_mask)[g].mean(),
                                cfg.target_ball_missing, 1.0 / max(1e-9, w_live[live_ok].min()))
    hidden = (live_mask | dead_mask) | (~vis & ~dead)
    rep.alive_target = float(hidden[live_ok].mean()) if live_ok.any() else 0.0
    return hidden


def _joint_live_weights(ball_m, xy_m, cfg, length, width):
    n = len(ball_m)
    sp = np.full(n, np.nan)
    sp[4:] = np.linalg.norm(ball_m[4:] - ball_m[:-4], axis=1) * 25.0 / 4.0
    sp = np.nan_to_num(sp, nan=5.0)
    w = np.ones(n)
    w *= np.where(sp < cfg.slow_mps, cfg.slow_weight, 1.0)
    w *= np.where(sp > cfg.fast_mps, cfg.fast_weight, 1.0)
    d = np.linalg.norm(xy_m - ball_m[:, None, :], axis=2)
    crowded = (np.nan_to_num(d, nan=np.inf) < cfg.crowd_ball_radius_m).sum(axis=1) >= cfg.crowd_ball_min
    w *= np.where(crowded, cfg.crowd_weight, 1.0)
    bx, by = ball_m[:, 0], ball_m[:, 1]
    with np.errstate(invalid="ignore"):
        near = ((by < cfg.line_distance_m) | (by > width - cfg.line_distance_m)
                | (bx < cfg.line_distance_m) | (bx > length - cfg.line_distance_m))
        deep = np.abs(bx - length / 2) > cfg.deep_m
    w *= np.where(np.nan_to_num(near, nan=False).astype(bool), cfg.line_weight, 1.0)
    w *= np.where(np.nan_to_num(deep, nan=False).astype(bool), cfg.deep_weight, 1.0)
    return w


def _mass_dropouts(slots_xy, graded, clean_ball_missing, cfg, rng):
    """Blank a random share of the tracked slots together for a lognormal stretch, in place."""
    n = slots_xy.shape[0]
    w = np.where(clean_ball_missing, cfg.mass_drop_dead_weight, 1.0) * graded
    if w.sum() <= 0:
        return 0
    k = rng.poisson(cfg.mass_drop_rate_per_min * graded.sum() / (25.0 * 60.0))
    starts = rng.choice(n, size=k, p=w / w.sum())
    for t0 in np.sort(starts).tolist():
        L = int(_lognormal_int(rng, cfg.mass_drop_median, cfg.mass_drop_sigma, 1)[0])
        here = np.flatnonzero(~np.isnan(slots_xy[t0, :, 0]))
        if not len(here):
            continue
        frac = rng.uniform(cfg.mass_drop_frac_lo, cfg.mass_drop_frac_hi)
        drop = rng.choice(here, size=int(round(frac * len(here))), replace=False)
        slots_xy[t0:t0 + L, drop] = np.nan
    return k


# -- Phase E: simulated video ball (a separate channel) ------------------------------------------

def _video_calibration(cfg):
    c = cfg.video_calibration
    if isinstance(c, str):
        if c != "real":
            raise ValueError("video_calibration must be 'real' or a dict")
        from .video.props import real_video_calibration
        return real_video_calibration()
    return c


def _speed4(p):
    n = len(p)
    sp = np.full(n, np.nan)
    if n > 4:
        sp[2:n - 2] = np.linalg.norm(p[4:] - p[:-4], axis=1) * 25.0 / 4.0
    return sp


def _fill_bursts(eligible, weights, lengths, u, base):
    """Bursts starting at eligible t when u[t] < base * w[t], running lengths[t] frames or until
    the first ineligible frame; starts inside an earlier burst are ignored."""
    n = len(eligible)
    out = np.zeros(n, dtype=bool)
    starts = np.flatnonzero(eligible & (u < base * weights))
    if not len(starts):
        return out
    stop_of = _run_end_index(eligible)
    covered = -1
    for t in starts.tolist():
        if t < covered:
            continue
        e = min(t + int(lengths[t]), int(stop_of[t]))
        out[t:e] = True
        covered = e
    return out


def _video_fill(ball_true, hidden, slots_xy, cfg, length, width, rep):
    """(n, 2) metres: the simulated video ball on hidden graded frames, NaN elsewhere."""
    rng = np.random.default_rng([cfg.seed, 29])
    cal = _video_calibration(cfg)
    n = len(ball_true)
    pool = np.asarray(cal["burst_len"], dtype=np.int64)
    pool = pool[pool > 0] if len(pool) else np.array([10])
    lengths = rng.choice(pool, n)
    u = rng.random(n)
    sp = _speed4(ball_true)
    edges = np.asarray(cfg.video_speed_edges, dtype=np.float64)
    w_bins = np.asarray(cfg.video_speed_weights, dtype=np.float64)
    b = np.searchsorted(edges, np.nan_to_num(sp, nan=0.0), side="right")   # 0..len(edges)
    w = np.where(b < len(w_bins), w_bins[np.minimum(b, len(w_bins) - 1)], 0.0)
    mx = cfg.video_margin_m
    my = cfg.video_margin_m * width / length

    def inside(p):
        with np.errstate(invalid="ignore"):
            return ((p[:, 0] >= -mx) & (p[:, 0] <= length + mx) & (p[:, 1] >= -my)
                    & (p[:, 1] <= width + my))
    eligible = hidden & ~np.isnan(ball_true[:, 0]) & (w > 0) & inside(ball_true)
    denom = max(1, int(hidden.sum()))
    fill = np.zeros(n, dtype=bool)
    if cfg.video_fill_share > 0 and eligible.any():
        fill = _solve_base(lambda base: _fill_bursts(eligible, w, lengths, u, base),
                           lambda m: m.sum() / denom, cfg.video_fill_share,
                           1.0 / max(1e-9, float(w[eligible].min())))
    out = np.full((n, 2), np.nan)
    # per-burst error: a real video-minus-ATD vector from the true speed's bin, plus AR(1) residual
    err_edges = np.asarray(cal["speed_edges"], dtype=np.float64)
    pools = [np.asarray(cal["err{}".format(i)], dtype=np.float64) for i in range(len(err_edges))]
    resid = _ar1(rng, (n, 2), cfg.video_resid_m, cfg.video_resid_rho)
    for s, e in _runs(fill):
        k = int(np.clip(np.searchsorted(err_edges, np.nan_to_num(sp[s], nan=0.0), side="right") - 1,
                        0, len(pools) - 1))
        while k > 0 and not len(pools[k]):
            k -= 1
        bias = pools[k][int(rng.integers(len(pools[k])))] if len(pools[k]) else np.zeros(2)
        out[s:e] = ball_true[s:e] + cfg.video_error_scale * (bias + resid[s:e])
    rep.extra["video_fill"] = float(fill.sum() / denom)
    # false spare balls: static, near a line, no player within video_spare_player_m at the start
    spare = np.zeros(n, dtype=bool)
    if cfg.video_spare_share > 0:
        free = hidden & ~fill
        lens2 = rng.choice(pool, n)
        u2 = rng.random(n)
        pts = _spare_points(rng, n, length, width, cfg.video_spare_margin_m)
        players = np.nan_to_num(np.linalg.norm(slots_xy - pts[:, None, :], axis=2), nan=np.inf).min(1)
        w2 = (players > cfg.video_spare_player_m).astype(np.float64)
        if (free & (w2 > 0)).any():
            spare = _solve_base(lambda base: _fill_bursts(free, w2, lens2, u2, base),
                                lambda m: m.sum() / denom, cfg.video_spare_share, 1.0)
        for s, e in _runs(spare):
            out[s:e] = pts[s] + rng.normal(0.0, 0.05, (e - s, 2))
    rep.extra["video_spare"] = float(spare.sum() / denom)
    out[~hidden | ~inside(out)] = np.nan
    return out


def _spare_points(rng, n, length, width, margin):
    """(n, 2) random points by a touchline or goal line, from `margin` m outside to 1 m inside."""
    a = rng.uniform(0, 2 * length + 2 * width, n)
    d = rng.uniform(-margin, 1.0, n)                     # distance inside the line
    side = np.searchsorted([length, 2 * length, 2 * length + width], a, side="right")
    x = np.select([side == 0, side == 1, side == 2], [a, a - length, d], length - d)
    y = np.select([side == 0, side == 1, side == 2], [d, width - d, a - 2 * length],
                  a - 2 * length - width)
    return np.stack([x, y], axis=1)
