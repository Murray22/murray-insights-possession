"""Phase B2: the calibrated degrader (joint ball gaps, mass dropouts, smooth error, slot recycling),
the calibration gate, the long-gap features and the label-free diagnostics. Each gate is shown to
fire on the case it exists to catch."""

import numpy as np
import pytest

from mtah import atd_profile, diagnostics, features as F, features_b2
from mtah.degrade import DegradeConfig, degrade, make_config
from mtah.frames import FrameTable
from test_degrade import synthetic_clean

JOINT = dict(ball_mode="joint", ball_burst_lengths=np.full(50, 300), target_ball_missing=0.40,
             dead_ball_visible=0.3, min_gap_frames=25)


@pytest.fixture(scope="module")
def clean():
    return synthetic_clean(n=30000)


def _runs_len(mask):
    e = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return np.flatnonzero(e == -1) - np.flatnonzero(e == 1)


def test_joint_mode_hits_targets_and_keeps_long_gaps(clean):
    t, rep = degrade(clean, None, DegradeConfig(seed=1, **JOINT))
    miss = np.isnan(t.ball[:, 0])
    assert miss.mean() == pytest.approx(0.40, abs=0.02)
    dead = np.isnan(clean.ball[:, 0])
    assert (~miss[dead]).mean() == pytest.approx(0.3, abs=0.12)
    assert np.median(_runs_len(miss)) >= 200          # gaps keep their drawn length (300)


def test_joint_gaps_run_across_stoppages(clean):
    t, _ = degrade(clean, None, DegradeConfig(seed=2, **JOINT))
    miss = np.isnan(t.ball[:, 0])
    dead = np.isnan(clean.ball[:, 0])
    e = np.diff(np.concatenate(([0], miss.astype(np.int8), [0])))
    spans = [(a, b) for a, b in zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1))]
    crossing = [1 for a, b in spans if dead[a:b].any() and (~dead[a:b]).any()]
    assert len(crossing) >= 3


def test_restart_cap_ends_gaps_after_restarts(clean):
    """With restart_reacq_prob 1 and a short delay, no gap that started inside a stoppage runs
    more than the delay past its restart; with 0, some do."""
    dead = np.isnan(clean.ball[:, 0])
    starts_dead = lambda m: [(a, b) for a, b in zip(
        np.flatnonzero(np.diff(np.concatenate(([0], m.astype(np.int8), [0]))) == 1),
        np.flatnonzero(np.diff(np.concatenate(([0], m.astype(np.int8), [0]))) == -1)) if dead[a]]
    over = []
    for p in (1.0, 0.0):
        t, _ = degrade(clean, None, DegradeConfig(seed=3, restart_reacq_prob=p, restart_reacq_median=5,
                                                  restart_reacq_sigma=0.01, **JOINT))
        m = np.isnan(t.ball[:, 0])
        n_over = 0
        for a, b in starts_dead(m):
            r = a + int(np.argmax(~dead[a:])) if (~dead[a:]).any() else len(dead)
            n_over += b > r + 8
        over.append(n_over)
    assert over[0] == 0 and over[1] > 0


def test_mass_dropouts_make_low_count_frames(clean):
    a, _ = degrade(clean, None, DegradeConfig(seed=0))
    b, _ = degrade(clean, None, DegradeConfig(seed=0, mass_drop_rate_per_min=2.0))
    ca, cb = a.present().sum(1), b.present().sum(1)
    assert np.percentile(cb, 5) <= np.percentile(ca, 5) - 4


def test_smoothed_error_has_little_jitter():
    from mtah.degrade import _noise
    shape = (20000, 4, 2)
    a = _noise(np.random.default_rng(0), shape, 0.3, DegradeConfig(noise_rho=0.99))
    b = _noise(np.random.default_rng(0), shape, 0.3, DegradeConfig(noise_rho=0.99, noise_smooth_frames=25))
    g = np.ones(shape[0], dtype=bool)
    assert b.std() == pytest.approx(0.3, rel=0.05)                     # same size of error ...
    assert atd_profile._jitter(b, g) < 0.2 * atd_profile._jitter(a, g)   # ... far less jitter


def test_lowest_slot_policy_recycles_slots_sooner(clean):
    gaps = {}
    for pol in ("random", "lowest"):
        t, _ = degrade(clean, None, DegradeConfig(seed=0, slot_policy=pol))
        p = atd_profile.player_profile(t)
        gaps[pol] = p["slot_gap_len"]["p50"]
    assert gaps["lowest"] < gaps["random"]


def test_make_config_defaults_and_rejects_unknown_lengths():
    assert make_config({}) == DegradeConfig()
    with pytest.raises(ValueError):
        make_config({"ball_burst_lengths": "bogus"})
    with pytest.raises(ValueError):
        degrade(synthetic_clean(n=3000), None, DegradeConfig(ball_mode="mixed"))


def test_phase_c_variants_keep_their_cache_names():
    from mtah import modelbench as mb
    assert "default" in mb.PHASE_C_NAMES and "rollout" in mb.PHASE_C_NAMES
    assert "atd_cal" not in mb.PHASE_C_NAMES


def test_calibration_gate_fails_split_and_passes_joint(clean):
    """The gate (atd_profile.check_calibration) says no to a degrader whose gaps are the Phase A
    lognormal bursts when the reference has long gaps, and yes to another seed of the reference."""
    ref, _ = degrade(clean, None, DegradeConfig(seed=10, **JOINT))
    same, _ = degrade(clean, None, DegradeConfig(seed=11, **JOINT))
    old, _ = degrade(clean, None, DegradeConfig(seed=11, target_ball_missing=0.40))
    rules = [r for r in atd_profile.CAL_RULES if r[0][0] == "ball" and "onset" not in r[0][1]]
    pr, ps, po = (atd_profile.profile(t) for t in (ref, same, old))
    assert atd_profile.check_calibration(pr, ps, rules) == []
    bad = {b[0] for b in atd_profile.check_calibration(pr, po, rules)}
    assert "ball.gap_len.p50" in bad and "ball.missing_in_runs_ge250" in bad


def test_profile_counts_gaps_and_onset_context():
    n = 2000
    ball = np.tile([[0.5, 0.5]], (n, 1)).astype(np.float32)
    ball[100:400] = np.nan
    ball[1000:1050] = np.nan
    xy = np.full((n, 2, 2), 0.4, dtype=np.float32)
    t = FrameTable(np.arange(n), np.ones(n, dtype=np.int8), ball, xy, np.array([0, 1], np.int8),
                   (100.0, 50.0))
    p = atd_profile.ball_profile(t)
    assert p["gap_len"]["n"] == 2 and p["gap_len"]["p50"] == pytest.approx(175)
    assert p["visible_share"] == pytest.approx(1 - 350 / n)
    assert p["reappear_disp_m"]["p50"] == pytest.approx(0.0, abs=1e-6)


def _team_table(n=1500, home_forward=True):
    rng = np.random.default_rng(0)
    L, W = 105.0, 68.0
    xy = np.zeros((n, 22, 2))
    base = rng.uniform([20, 10], [85, 58], (22, 2))
    drift = np.linspace(0, 20, n)[:, None]
    # home attacks right (starts on the left at kick-off), away attacks left
    base[:11, 0] = rng.uniform(10, 45, 11)
    base[11:, 0] = rng.uniform(60, 95, 11)
    xy[:] = base
    if home_forward:
        xy[300:, :, 0] += drift[:n - 300] * 1.0          # both teams move right: home advances
    ball = np.full((n, 2), np.nan)
    ball[:200] = [40, 30]
    ball[1300:] = [80, 30]
    return FrameTable(np.arange(n), np.ones(n, dtype=np.int8), (ball / [L, W]).astype(np.float32),
                      (xy / [L, W]).astype(np.float32), np.array([0] * 11 + [1] * 11, np.int8),
                      (L, W))


def test_b2_player_features_follow_who_advances_and_flip_with_teams():
    t = _team_table()
    f = F.compute(t)
    e = features_b2.compute_extra(t, f)
    mid = slice(500, 1200)
    assert np.nanmean(e["b2_cvel_d"][mid]) > 0.5             # home moving towards its attack
    from mtah.models.learned import swapped
    ts = swapped(t)
    fs = F.compute(ts)
    es = features_b2.compute_extra(ts, fs)
    assert np.nanmean(es["b2_cvel_d"][mid]) < -0.5


def test_b2_gap_features_use_both_ends():
    t = _team_table()
    f = F.compute(t)
    e = features_b2.compute_extra(t, f)
    k = 700                                                  # inside the 1,100-frame gap
    # since + until counts the gap plus one frame, as features.gap_len does (which caps at 30 s)
    assert e["b2_gap_s"][k] == pytest.approx(1101 / 25)
    assert e["b2_gap_dx_att"][k] == pytest.approx(40.0, abs=0.5)   # ball moved 40 m towards home's attack
    assert e["b2_gap_frac"][k] == pytest.approx(f["since_vis"][k] / f["gap_len"][k])


def test_diagnostics_possessions_and_shares():
    lab = np.array(list("HHHXXAAADDHHH") + ["X"] * 3)
    graded = np.ones(len(lab), dtype=bool)
    graded[-3:] = False
    lens = diagnostics.possessions(lab, graded)
    assert sorted(lens.tolist()) == [3, 3, 5]                # X inside H's run counts for H
    vis = np.ones(len(lab), dtype=bool)
    vis[8:10] = False
    s = diagnostics.summarise_labels(lab, graded, vis)
    assert s["D_share"] == pytest.approx(2 / 13)
    assert s["D_share_ball_missing"] == pytest.approx(1.0)
    assert s["home_share_of_HA"] == pytest.approx(6 / 9)
    iv = [{"code": "PROGRESSION", "team": "H", "start": 0, "end": 3},
          {"code": "SET PIECES", "team": "A", "start": 0, "end": 13}]
    a = diagnostics.smart_tag_agreement(lab, iv, graded)
    assert a["agree_share"] == pytest.approx(1.0) and a["tagged_frames"] == 3


def test_attack_sign_survives_a_scrambled_half_start():
    """Home defends the left goal all half (keeper near x = 3) but the half's first 10 s show home
    on the right (players walking out). The Phase C first-10-s rule gets it wrong; the whole-half
    extremes rule does not."""
    rng = np.random.default_rng(1)
    n, L, W = 3000, 105.0, 68.0
    xy = np.zeros((n, 22, 2))
    xy[:, :11, 0] = rng.uniform(15, 60, (n, 11))
    xy[:, 0, 0] = 3.0                                       # home keeper
    xy[:, 11:, 0] = rng.uniform(45, 90, (n, 11))
    xy[:, 11, 0] = 102.0                                    # away keeper
    xy[:250, :11, 0] = 80.0
    xy[:250, 11:, 0] = 30.0
    xy[:, :, 1] = rng.uniform(5, 63, (n, 22))
    t = FrameTable(np.arange(n), np.ones(n, dtype=np.int8), np.full((n, 2), np.nan, np.float32),
                   (xy / [L, W]).astype(np.float32), np.array([0] * 11 + [1] * 11, np.int8), (L, W))
    assert (F.attack_sign(t) == 1.0).all()


@pytest.mark.skipif(not __import__("conftest").sample_available(3), reason="sample games not present")
def test_attack_sign_matches_the_kickoff_rule_on_sample_games():
    from mtah import sample_io
    for g in (1, 2, 3):
        t = sample_io.clean_tracking(g)
        x = t.xy[:, :, 0]
        for k in (1, 2):
            idx = np.flatnonzero(t.period == k)
            head = idx[:250]
            ch = np.nanmean(x[head][:, t.slot_team == 0])
            ca = np.nanmean(x[head][:, t.slot_team == 1])
            assert F.attack_sign(t)[idx[0]] == (1.0 if ch < ca else -1.0)


def test_slow_share_and_mixture_arithmetic():
    pspd = np.array([0.5, 0.5, 2.0, 2.0, 0.2])
    m = np.array([True, True, True, True, False])
    assert diagnostics.slow_share(pspd, m) == pytest.approx(0.5)
    assert diagnostics.slow_share(pspd, np.zeros(5, dtype=bool)) is None
