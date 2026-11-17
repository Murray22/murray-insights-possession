"""Phase E: ball-track shape calibration (atd_cal2), the simulated video channel, the video
features, the video-simulation gate and the paired bootstrap. Each gate is fed the case it exists
to catch."""

import json
import os

import numpy as np
import pytest

from mtah import atd_profile, features_v, paired, paths, score
from mtah.degrade import DegradeConfig, _ball_shape, _wobble, degrade
from mtah.frames import AWAY, HOME, FrameTable
from mtah.video import props, source
from conftest import needs_sample
from test_degrade import synthetic_clean

JOINT = dict(ball_mode="joint", ball_burst_lengths=np.full(50, 300), target_ball_missing=0.40,
             dead_ball_visible=0.3, min_gap_frames=25, smooth_frames=9, noise_rho=0.99,
             noise_smooth_frames=25, ball_noise_m=0.6)
SHAPE = dict(ball_post_smooth_frames=13, ball_post_smooth_passes=2, ball_wobble_m=0.25,
             ball_wobble_frames=21, ball_wobble_passes=2)


@pytest.fixture(scope="module")
def clean():
    return synthetic_clean(n=30000)


def _shape_rules():
    return [r for r in atd_profile.CAL_RULES_SHAPE if "touch" not in r[0][1]]


# -- Part 1: the shape rules and the ball-error options --------------------------------------------

def test_shape_gate_fails_unshaped_ball_and_passes_another_seed(clean):
    """The kink rules say no to the atd_cal ball error (no further smoothing, no wobble) when the
    reference has the fitted shape, and yes to another seed of the reference."""
    ref, _ = degrade(clean, None, DegradeConfig(seed=10, **JOINT, **SHAPE))
    same, _ = degrade(clean, None, DegradeConfig(seed=11, **JOINT, **SHAPE))
    old, _ = degrade(clean, None, DegradeConfig(seed=11, **JOINT))
    pr, ps, po = ({"ball": atd_profile.ball_profile(t)} for t in (ref, same, old))
    rules = _shape_rules()
    assert atd_profile.check_calibration(pr, ps, rules) == []
    bad = {b[0] for b in atd_profile.check_calibration(pr, po, rules)}
    assert "ball.kink4.p50" in bad and "ball.kink4.p99" in bad


def test_shape_rules_are_in_the_gate():
    names = {".".join(r[0]) for r in atd_profile.CAL_RULES}
    assert {"ball.kink4.p50", "ball.kink4.p99", "ball.strong_touch_runs_per_min"} <= names
    assert len(atd_profile.CAL_RULES) == len(atd_profile.CAL_RULES_B2) + len(atd_profile.CAL_RULES_SHAPE)


@needs_sample
def test_real_profile_fails_atd_cal_on_shape():
    """On the real numbers (runs/phaseE/profile_real.json, written by atd_profile's main), the
    current atd_cal fails the kink rules; atd_cal2 passes every kink4 rule and fails fewer."""
    path = os.path.join(paths.RUNS_DIR, "phaseE", "profile_real.json")
    if not os.path.isfile(path):
        pytest.skip("no real profile in runs/phaseE")
    from mtah import modelbench as mb
    with open(path) as fh:
        real = json.load(fh)
    profs = {}
    for v in ("atd_cal", "atd_cal2"):
        t, _ = mb.degraded_table(v, 0, 1)
        profs[v] = {"ball": atd_profile.ball_profile(t)}
    bad = {b[0] for b in atd_profile.check_calibration(real, profs["atd_cal"], atd_profile.CAL_RULES_SHAPE)}
    assert "ball.kink4.p50" in bad and "ball.kink4.p99" in bad
    bad2 = {b[0] for b in atd_profile.check_calibration(real, profs["atd_cal2"], atd_profile.CAL_RULES_SHAPE)}
    assert not any(b.startswith("ball.kink4.") for b in bad2) and len(bad2) < len(bad)


def test_ball_shape_defaults_are_identity_and_lag_shifts():
    b = np.cumsum(np.ones((50, 2)), axis=0)
    assert np.array_equal(_ball_shape(b, DegradeConfig()), b)
    assert _wobble(DegradeConfig(), b.shape) == 0.0
    lagged = _ball_shape(b, DegradeConfig(ball_lag_frames=2))
    assert np.allclose(lagged[10:], b[8:-2])                     # the shown ball is 2 frames late


def test_wobble_and_smoothing_change_only_the_ball(clean):
    a, _ = degrade(clean, None, DegradeConfig(seed=4, **JOINT))
    b, _ = degrade(clean, None, DegradeConfig(seed=4, **JOINT, **SHAPE))
    assert np.array_equal(np.isnan(a.ball[:, 0]), np.isnan(b.ball[:, 0]))
    np.testing.assert_array_equal(a.xy, b.xy)
    assert not np.allclose(a.ball, b.ball, equal_nan=True)


# -- Part 2: the simulated video channel -------------------------------------------------------------

def _cal(bias=(0.0, 0.0)):
    out = {"burst_len": np.array([10, 20, 30]), "speed_edges": np.array([0.0, 1, 3, 6, 10, 15])}
    for i in range(6):
        out["err{}".format(i)] = np.tile(np.asarray(bias, dtype=np.float32), (5, 1))
    return out


VIDEO = dict(video_fill_share=0.4, video_spare_share=0.0, video_resid_m=0.0,
             video_calibration=None)


def _vcfg(seed=1, **kw):
    d = dict(JOINT, **VIDEO)
    d.update(kw)
    if d["video_calibration"] is None:
        d["video_calibration"] = _cal()
    return DegradeConfig(seed=seed, **d)


def test_video_fill_only_in_gaps_and_atd_part_unchanged(clean):
    a, _ = degrade(clean, None, DegradeConfig(seed=5, **JOINT))
    b, rep = degrade(clean, None, _vcfg(seed=5))
    np.testing.assert_array_equal(a.ball, b.ball)
    np.testing.assert_array_equal(a.xy, b.xy)
    assert a.vball is None and b.vball is not None
    vis = ~np.isnan(b.ball[:, 0])
    assert np.isnan(b.vball[vis]).all()                           # never where the ATD has the ball
    share = np.isfinite(b.vball[~vis, 0]).mean()
    assert share == pytest.approx(0.4, abs=0.05)
    assert rep.extra["video_fill"] == pytest.approx(0.4, abs=0.01)


def test_video_never_fills_fast_balls_and_prefers_still_ones():
    clean = synthetic_clean(n=30000, seed=2)
    n = clean.n_frames
    # alternate 500-frame stretches of a still ball and a ball crossing the pitch at 20 m/s
    t = np.arange(n)
    fast = (t // 500) % 3 == 1
    medium = (t // 500) % 3 == 2
    x = np.where(fast, 0.5 + 0.4 * np.sin(t * 20.0 / 25 / 105 * 2), 0.5)
    x = np.where(medium, 0.5 + 0.3 * np.sin(t * 2.0 / 25 / 30), x)
    ball = np.stack([x, np.full(n, 0.5)], axis=1).astype(np.float32)
    ball[np.isnan(clean.ball[:, 0])] = np.nan
    c = FrameTable(clean.frame, clean.period, ball, clean.xy, clean.slot_team, clean.pitch,
                   clean.player_id, dict(clean.meta))
    out, _ = degrade(c, None, _vcfg(seed=3, smooth_frames=1, video_fill_share=0.3,
                                    video_speed_weights=(1.0, 0.2, 0.1, 0.05)))
    hidden = np.isnan(out.ball[:, 0]) & ~np.isnan(ball[:, 0])
    filled = np.isfinite(out.vball[:, 0])
    sp = props.speed(ball.astype(np.float64) * np.array(c.pitch))
    assert not (filled & hidden & (sp > 16)).any()
    still = hidden & (sp < 0.3)
    moving = hidden & (sp > 1) & (sp < 14)
    assert filled[still].mean() > 2 * filled[moving].mean()


def test_video_error_comes_from_the_calibration_pool(clean):
    out, _ = degrade(clean, None, _vcfg(seed=6, smooth_frames=1, video_calibration=_cal((3.0, 0.0))))
    f = np.isfinite(out.vball[:, 0]) & ~np.isnan(clean.ball[:, 0])
    d = (out.vball[f] - clean.ball[f]) * np.array(clean.pitch)
    assert f.sum() > 100
    assert np.allclose(d[:, 0], 3.0, atol=1e-3) and np.allclose(d[:, 1], 0.0, atol=1e-3)


def test_spare_balls_are_static_by_the_lines_away_from_players(clean):
    out, rep = degrade(clean, None, _vcfg(seed=7, video_fill_share=0.0, video_spare_share=0.05))
    L, W = clean.pitch
    f = np.isfinite(out.vball[:, 0])
    assert rep.extra["video_spare"] == pytest.approx(0.05, abs=0.01)
    p = out.vball[f] * np.array([L, W])
    dl = props.line_distance(p, (L, W))
    assert (dl < 1.0 + 0.3).all() and (dl > -3.0 - 0.3).all()
    near = props.nearest_player(out, out.vball.astype(np.float64) * np.array([L, W]))
    starts = np.flatnonzero(f & ~np.concatenate(([False], f[:-1])))
    assert (near[starts] > 4.5).all()


def test_video_props_measure_fill_and_lag():
    n = 3000
    L, W = 100.0, 50.0
    t = np.arange(n)
    ball = np.stack([0.2 + 0.6 * (t % 500) / 500, np.full(n, 0.5)], axis=1).astype(np.float32)
    atd = ball.copy()
    atd[1000:1600] = np.nan
    xy = np.full((n, 2, 2), 0.1, np.float32)
    tab = FrameTable(t, np.ones(n, np.int8), atd, xy, np.array([0, 1], np.int8), (L, W))
    vb = np.full((n, 2), np.nan, np.float32)
    vb[1100:1400] = ball[1100:1400]
    vb[900:1000] = ball[901:1001]                       # video one frame ahead of the ATD on the window
    p = props.profile(tab, vb, np.ones(n, bool))
    assert p["fill_share"] == pytest.approx(300 / 600)
    assert p["burst_len"]["p50"] == 300
    assert p["lag"]["best_shift"] == pytest.approx(1.0)      # the scan finds a 1-frame shift
    vb[900:1000] = ball[900:1000]
    assert props.profile(tab, vb, np.ones(n, bool))["lag"]["best_shift"] == pytest.approx(0.0)


def test_video_gate_fails_a_moving_simulation():
    real = {"fill_share": 0.4, "filled_static_share": 0.66, "filled_fast_share": 0.01,
            "burst_len": {"p50": 9, "p90": 50}, "spare_like_share_of_gap": 0.04,
            "filled_near_player_2m": 0.33, "filled_offpitch_share": 0.13}
    good = dict(real, filled_static_share=0.6)
    bad = dict(real, filled_static_share=0.2, filled_fast_share=0.2)
    assert props.check_video_calibration(real, good) == []
    names = {b[0] for b in props.check_video_calibration(real, bad)}
    assert names == {"filled_static_share", "filled_fast_share"}


def test_with_video_channel_keeps_the_atd_ball():
    n = 10
    ball = np.full((n, 2), 0.5, np.float32)
    ball[3:7] = np.nan
    t = FrameTable(np.arange(n), np.ones(n, np.int8), ball, np.full((n, 2, 2), np.nan, np.float32),
                   np.array([0, 1], np.int8))
    tr = dict(processed=np.ones(n, bool), ball=np.full((n, 2), 0.2, np.float32),
              support=np.full(n, 10, np.int32), config=np.array("x"))
    out = source.with_video_channel(t, track=tr)
    np.testing.assert_array_equal(out.ball, t.ball)
    assert np.isfinite(out.vball[:, 0]).tolist() == [False] * 3 + [True] * 4 + [False] * 3
    assert out.meta["ball_source"].tolist() == [0, 0, 0, 1, 1, 1, 1, 0, 0, 0]


def test_vball_survives_save_load_and_copy(tmp_path):
    n = 5
    t = FrameTable(np.arange(n), np.ones(n, np.int8), np.zeros((n, 2), np.float32),
                   np.zeros((n, 2, 2), np.float32), np.array([0, 1], np.int8),
                   vball=np.full((n, 2), 0.3, np.float32))
    p = str(tmp_path / "t.npz")
    t.save_npz(p)
    u = FrameTable.load_npz(p)
    np.testing.assert_array_equal(u.vball, t.vball)
    assert t.copy().vball is not t.vball
    with pytest.raises(ValueError):
        FrameTable(np.arange(n), np.ones(n, np.int8), np.zeros((n, 2), np.float32),
                   np.zeros((n, 2, 2), np.float32), np.array([0, 1], np.int8),
                   vball=np.zeros((n + 1, 2), np.float32))


# -- video features ----------------------------------------------------------------------------------

def _vtable(n=400):
    L, W = 105.0, 68.0
    xy = np.full((n, 2, 2), np.nan, np.float32)
    xy[:, 0] = (30 / L, 30 / W)                 # home player
    xy[:, 1] = (70 / L, 30 / W)                 # away player
    ball = np.full((n, 2), np.nan, np.float32)
    ball[:100] = (50 / L, 30 / W)
    vb = np.full((n, 2), np.nan, np.float32)
    vb[150:200] = (31 / L, 30 / W)               # at the home player's feet
    vb[50:60] = (69 / L, 30 / W)                 # where the ATD has the ball: must be ignored
    return FrameTable(np.arange(n), np.ones(n, np.int8), ball, xy, np.array([HOME, AWAY], np.int8),
                      (L, W), vball=vb)


def test_video_features_read_the_channel_only_in_gaps_and_flip_with_teams():
    from mtah.models.learned import swapped
    t = _vtable()
    f = features_v.compute_extra(t, np.ones(t.n_frames))
    assert f["v_src"][50:60].sum() == 0 and f["v_src"][150:200].all()
    assert (f["v_tt"][150:200] == 1).all() and f["v_ddiff"][160] > 0
    s = features_v.compute_extra(swapped(t), -np.ones(t.n_frames))
    np.testing.assert_allclose(s["v_tt"], -f["v_tt"])
    np.testing.assert_allclose(np.nan_to_num(s["v_ddiff"]), -np.nan_to_num(f["v_ddiff"]))
    np.testing.assert_allclose(np.nan_to_num(s["v_bx_att"]), -np.nan_to_num(f["v_bx_att"]))
    t.vball = None
    g = features_v.compute_extra(t, np.ones(t.n_frames))
    assert g["v_src"].sum() == 0 and np.isnan(g["v_ddiff"]).all()


# -- the paired bootstrap ----------------------------------------------------------------------------

def _labels(spells, n):
    out = np.full(n, "D", dtype="<U1")
    for a, b, c in spells:
        out[a:b] = c
    return out


def test_paired_f1_equals_the_organiser_and_detects_a_better_model():
    rng = np.random.default_rng(0)
    n, spells, t = 40000, [], 0
    while t < n - 400:
        L = int(rng.integers(30, 300))
        spells.append((t, t + L, "H" if len(spells) % 2 else "A"))
        t += L + int(rng.integers(0, 40))
    gt = _labels(spells, n)
    good = _labels([(a + int(rng.integers(-3, 4)), b + int(rng.integers(-3, 4)), c) for a, b, c in spells], n)
    bad = good.copy()
    for a, b, c in spells[::3]:
        bad[a:b] = "A" if c == "H" else "H"                       # a third of possessions wrong
    g, pg, pb = (score.codes(x) for x in (gt, good, bad))
    fl, npred = paired.spell_flags(g, pg)
    want = score.pm.spell_scores(g, score.pm.fill_abstentions(pg, score.pm.valid_runs(g)),
                                 score.pm.valid_runs(g))["mean_f1"]
    assert paired.f1_from(fl, npred) == pytest.approx(want)
    fb, nb = paired.spell_flags(g, pb)
    rs = np.random.RandomState(0)
    diffs = [paired.f1_from(fl, npred, p) - paired.f1_from(fb, nb, p)
             for p in (rs.randint(0, fl.shape[1], fl.shape[1]) for _ in range(200))]
    assert np.percentile(diffs, 2.5) > 0                          # good beats bad on every resample
    same = [paired.f1_from(fl, npred, p) - paired.f1_from(fl, npred, p)
            for p in (rs.randint(0, fl.shape[1], fl.shape[1]) for _ in range(20))]
    assert np.allclose(same, 0)
