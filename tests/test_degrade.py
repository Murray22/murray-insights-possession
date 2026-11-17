"""The degrader is deterministic per seed and hits the targets it is calibrated to."""

import numpy as np
import pytest

from mtah.degrade import DegradeConfig, degrade
from mtah.frames import AWAY, HOME, FrameTable
from conftest import needs_sample


def synthetic_clean(n=20000, seed=0):
    """22 random-walking players, a ball that follows one of them, and dead stretches with no ball."""
    rng = np.random.default_rng(seed)
    xy = np.empty((n, 22, 2))
    xy[0] = rng.uniform(0.1, 0.9, (22, 2))
    steps = rng.normal(0, 0.002, (n, 22, 2))
    for t in range(1, n):
        xy[t] = np.clip(xy[t - 1] + steps[t], 0.02, 0.98)
    carrier = np.repeat(rng.integers(0, 22, n // 100 + 1), 100)[:n]
    ball = xy[np.arange(n), carrier] + rng.normal(0, 0.003, (n, 2))
    dead = np.zeros(n, dtype=bool)
    for s in range(1500, n - 1000, 3000):
        dead[s:s + int(rng.integers(300, 900))] = True
    ball[dead] = np.nan
    team = np.array([HOME] * 11 + [AWAY] * 11, dtype=np.int8)
    return FrameTable(np.arange(n), np.ones(n, dtype=np.int8), ball.astype(np.float32),
                      xy.astype(np.float32), team, (105.0, 68.0), np.arange(22).astype(str),
                      {"game": "synthetic"})


@pytest.fixture(scope="module")
def clean():
    return synthetic_clean()


def test_deterministic_per_seed(clean):
    a, ra = degrade(clean, None, DegradeConfig(seed=3))
    b, rb = degrade(clean, None, DegradeConfig(seed=3))
    np.testing.assert_array_equal(a.ball, b.ball)
    np.testing.assert_array_equal(a.xy, b.xy)
    assert ra == rb
    c, _ = degrade(clean, None, DegradeConfig(seed=4))
    assert not np.array_equal(np.isnan(a.ball[:, 0]), np.isnan(c.ball[:, 0]))


@pytest.mark.parametrize("target", [0.30, 0.41, 0.50])
def test_hits_ball_missing_target(clean, target):
    table, rep = degrade(clean, None, DegradeConfig(seed=1, target_ball_missing=target))
    measured = np.isnan(table.ball[:, 0]).mean()
    assert measured == pytest.approx(target, abs=0.01)
    assert rep.ball_missing_graded == pytest.approx(measured)


def test_target_only_counts_graded_frames(clean):
    graded = np.zeros(clean.n_frames, dtype=bool)
    graded[2000:15000] = True
    table, _ = degrade(clean, graded, DegradeConfig(seed=1))
    assert np.isnan(table.ball[graded, 0]).mean() == pytest.approx(0.41, abs=0.01)


def test_ball_missing_comes_in_bursts(clean):
    table, _ = degrade(clean, None, DegradeConfig(seed=2, dead_ball_visible=0.0))
    live = ~np.isnan(clean.ball[:, 0])
    miss = np.isnan(table.ball[:, 0]) & live
    starts = np.sum(miss[1:] & ~miss[:-1])
    assert miss.sum() / max(1, starts) > 5          # mean burst well above one frame


def test_empirical_burst_lengths_are_used(clean):
    cfg = DegradeConfig(seed=2, dead_ball_visible=0.0, target_ball_missing=0.2,
                        ball_burst_lengths=np.full(100, 4))
    table, _ = degrade(clean, None, cfg)
    live = ~np.isnan(clean.ball[:, 0])
    miss = (np.isnan(table.ball[:, 0]) & live).astype(np.int8)
    edges = np.diff(np.concatenate(([0], miss, [0])))
    lengths = np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)
    assert np.median(lengths) == 4


def test_dead_ball_visibility_is_parameterised(clean):
    dead = np.isnan(clean.ball[:, 0])
    for v in (0.0, 0.5, 1.0):
        table, rep = degrade(clean, None, DegradeConfig(seed=5, dead_ball_visible=v))
        shown = (~np.isnan(table.ball[dead, 0])).mean()
        assert shown == pytest.approx(v, abs=0.12)


def test_players_anonymous_slots_and_counts(clean):
    table, rep = degrade(clean, None, DegradeConfig(seed=0))
    assert table.player_id is None
    assert table.n_slots == 46
    assert (table.slot_team[:23] == HOME).all() and (table.slot_team[23:] == AWAY).all()
    counts = table.present().sum(axis=1)
    assert counts.mean() == pytest.approx(20.5, abs=0.3)
    assert (counts > 22).any()                       # the odd duplicate
    # slots end and restart: far more tracklets than slots
    pres = table.present().astype(np.int8)
    n_starts = int((np.diff(pres, axis=0) == 1).sum() + pres[0].sum())
    assert n_starts > 3 * table.n_slots
    # team membership is preserved: each home slot sits on a home player's track
    home_clean = clean.xy[:, :11]
    t = 5000
    hs = table.xy[t, :23][~np.isnan(table.xy[t, :23, 0])]
    d = np.linalg.norm(hs[:, None, :] - home_clean[t][None], axis=2).min(axis=1)
    assert np.median(d) < 0.02


def test_identity_swaps_happen(clean):
    _, rep = degrade(clean, None, DegradeConfig(seed=0, swap_prob=1.0, swap_distance_m=3.0))
    _, rep0 = degrade(clean, None, DegradeConfig(seed=0, swap_prob=0.0))
    assert rep.n_swaps > 0 and rep0.n_swaps == 0


@needs_sample
def test_sample_game_targets():
    from mtah import sample_gt, sample_io
    clean = sample_io.clean_tracking(2)
    graded = sample_gt.sample_game_gt(2).labels != "X"
    table, rep = degrade(clean, graded, DegradeConfig(seed=0))
    assert np.isnan(table.ball[graded, 0]).mean() == pytest.approx(0.41, abs=0.01)
    assert table.present()[graded].sum(axis=1).mean() == pytest.approx(20.5, abs=0.3)


def test_random_dead_path_changes_dead_stretches_only(clean):
    """The randomised dead-ball path (for learned models) moves the synthetic ball inside dead
    stretches and nowhere else; the default "linear" path is untouched by its existence."""
    lin, _ = degrade(clean, None, DegradeConfig(seed=4))
    lin2, _ = degrade(clean, None, DegradeConfig(seed=4, dead_ball_path="linear"))
    rnd, _ = degrade(clean, None, DegradeConfig(seed=4, dead_ball_path="random"))
    rnd2, _ = degrade(clean, None, DegradeConfig(seed=4, dead_ball_path="random"))
    np.testing.assert_array_equal(lin.ball, lin2.ball)
    np.testing.assert_array_equal(rnd.ball, rnd2.ball)
    np.testing.assert_array_equal(lin.xy, rnd.xy)
    dead = np.isnan(clean.ball[:, 0])
    both = ~np.isnan(lin.ball[:, 0]) & ~np.isnan(rnd.ball[:, 0])
    np.testing.assert_array_equal(np.isnan(lin.ball), np.isnan(rnd.ball))
    np.testing.assert_allclose(lin.ball[both & ~dead], rnd.ball[both & ~dead])
    diff = np.abs(lin.ball[both & dead] - rnd.ball[both & dead]).max(axis=1)
    assert (diff > 1e-3).mean() > 0.1      # visible ends often sit at the same held points


def test_unknown_dead_path_rejected(clean):
    with pytest.raises(ValueError):
        degrade(clean, None, DegradeConfig(dead_ball_path="spiral"))


def test_rollout_keeps_the_ball_moving_after_play_stops(clean):
    """Without rollout the synthetic ball freezes on the first dead frame; with it, it keeps moving
    for a while in its last direction. Live frames are untouched."""
    a, _ = degrade(clean, None, DegradeConfig(seed=5, ball_noise_m=0.0, pos_noise_m=0.0,
                                              dead_ball_visible=1.0))
    b, _ = degrade(clean, None, DegradeConfig(seed=5, ball_noise_m=0.0, pos_noise_m=0.0,
                                              dead_ball_visible=1.0, dead_ball_rollout=True))
    dead = np.isnan(clean.ball[:, 0])
    starts = [s for s in np.flatnonzero(np.diff(dead.astype(int)) == 1) + 1 if s > 6]
    moved_a = [np.abs(a.ball[s + 5] - a.ball[s]).max() for s in starts]
    moved_b = [np.abs(b.ball[s + 5] - b.ball[s]).max() for s in starts]
    assert max(moved_a) < 1e-6 and np.median(moved_b) > 1e-4
    live = ~dead
    np.testing.assert_array_equal(a.ball[live], b.ball[live])
