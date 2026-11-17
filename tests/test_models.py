"""Decoders, the dead-ball detector and the segment model: each test can fail."""

import itertools

import numpy as np
import pytest

from mtah import features as F
from mtah.models import deadball, hmm, segment
from synth import phases_table


# -- decoder -----------------------------------------------------------------------------------------

def brute_force(emit, cost):
    n, K = emit.shape
    best, arg = -np.inf, None
    for path in itertools.product(range(K), repeat=n):
        s = sum(emit[t, path[t]] for t in range(n))
        s -= sum(cost[path[t - 1], path[t]] for t in range(1, n))
        if s > best:
            best, arg = s, path
    return np.array(arg)


@pytest.mark.parametrize("seed", range(5))
def test_viterbi_matches_brute_force(seed):
    rng = np.random.default_rng(seed)
    emit = rng.normal(0, 2, (7, 3))
    cost = rng.uniform(0, 3, (3, 3))
    np.fill_diagonal(cost, 0)
    assert np.array_equal(hmm.viterbi(emit, cost), brute_force(emit, cost))


def test_posteriors_sum_to_one_and_follow_evidence():
    emit = np.zeros((50, 2))
    emit[:25, 0] = 3
    emit[25:, 1] = 3
    post = hmm.posteriors(emit, np.array([[0, 5.0], [5.0, 0]]))
    assert np.allclose(post.sum(1), 1)
    assert post[5, 0] > 0.9 and post[45, 1] > 0.9


def count_runs(path, states=(0, 1)):
    b = np.flatnonzero(np.diff(path) != 0) + 1
    starts = np.concatenate(([0], b))
    return int(np.isin(path[starts], states).sum())


def test_switching_cost_reduces_possessions_on_noisy_sequence():
    rng = np.random.default_rng(1)
    lengths = rng.integers(50, 250, 40)
    truth = np.repeat(np.arange(40) % 2, lengths)                  # 40 alternating spells
    flip = rng.random(len(truth)) < 0.3                             # 30% of frames lie
    obs = np.where(flip, 1 - truth, truth)
    emit = np.zeros((len(truth), 3))
    emit[np.arange(len(truth)), obs] = 1.0
    emit[:, 2] = -5.0
    counts = []
    for c in (0.2, 2.0, 16.0):
        cost = np.array([[0, c, 20], [c, 0, 20], [20, 20, 0]], dtype=float)
        counts.append(count_runs(hmm.viterbi(emit, cost)))
    assert counts[0] > counts[1] > counts[2]
    assert counts[0] > 200            # no cost: follows the noise
    assert counts[2] <= 60            # a real cost: close to the 40 true spells


def test_decode_respects_graded_runs():
    emit = np.zeros((10, 3))
    emit[:, 1] = 1
    period = np.array([0, 1, 1, 1, 0, 0, 2, 2, 2, 0])
    path, _ = hmm.decode(emit, np.zeros((3, 3)), period)
    assert (path[period == 0] == -1).all() and (path[period > 0] == 1).all()


def test_abstain_never_on_first_frame_and_fill_restores_labels():
    rng = np.random.default_rng(0)
    path = np.repeat([0, 1, 2, 0, 1], 20)
    labels = segment.LABELS[path]
    post = rng.uniform(0, 1, (100, 3))
    out = segment.abstain(labels, path, post, 0.9)
    assert (out == "X").sum() > 30
    first = np.r_[True, path[1:] != path[:-1]]
    assert not (out[first] == "X").any()
    filled = out.copy()
    for t in range(1, len(filled)):           # the scorer's forward fill
        if filled[t] == "X":
            filled[t] = filled[t - 1]
    assert np.array_equal(filled, labels)


# -- dead-ball detector --------------------------------------------------------------------------------

PHASES = [("play", 60, 4.0), ("dead", 30, 0.6), ("buildup", 60, 2.0), ("play", 40, 4.0),
          ("dead", 25, 0.7), ("play", 45, 4.0)]


@pytest.mark.parametrize("norm", ["raw", "range"])
def test_dead_detector_flags_out_of_play_not_slow_buildup(norm):
    table, kinds = phases_table(PHASES, seed=3)
    dead, llr = deadball.predict_dead(table, feats=F.compute(table), motion_norm=norm)
    assert dead[kinds == "dead"].mean() > 0.8
    assert dead[kinds == "buildup"].mean() < 0.1
    assert dead[kinds == "play"].mean() < 0.05


def test_game_normalised_motion_mistakes_a_long_slow_buildup_for_dead_ball():
    """Why v2 (median / IQR per game) was dropped: with a quarter of the sequence slow build-up it
    calls the build-up dead. Kept as a test so the failure stays documented."""
    table, kinds = phases_table(PHASES, seed=3)
    dead, _ = deadball.predict_dead(table, feats=F.compute(table), motion_norm="game")
    assert dead[kinds == "buildup"].mean() > 0.5


def test_dead_detector_ball_out_cue_matters():
    """Players slow down without the ball leaving the pitch: the evidence must be weaker."""
    table, kinds = phases_table(PHASES, seed=3)
    f = F.compute(table)
    _, llr_out = deadball.predict_dead(table, feats=f)
    f2 = dict(f)
    f2["bout"] = np.where(np.isnan(f["bout"]), np.nan, 0.0)
    llr_in = deadball.dead_llr(f2, None, table.n_frames)
    d = kinds == "dead"
    assert llr_out[d].mean() > llr_in[d].mean() + 0.5


def test_set_piece_interval_adds_dead_evidence():
    table, kinds = phases_table(PHASES, seed=3)
    f = F.compute(table)
    base = deadball.dead_llr(f, None, table.n_frames)
    tagged = deadball.dead_llr(f, [{"kind": "set_piece", "start": 2000, "end": 2100}],
                               table.n_frames)
    assert (tagged[1925:2001] > base[1925:2001]).all() and (tagged[:1900] == base[:1900]).all()


# -- segment model ------------------------------------------------------------------------------------

def test_segment_model_follows_possession_and_cost_controls_count():
    table, kinds = phases_table(PHASES, seed=5)
    f = F.compute(table)
    lab = segment.predict(table, feats=f, team_mode="touch", cost_team=10.0,
                          dead=dict(motion_norm="range"))
    assert set(np.unique(lab)) <= {"H", "A", "D", "X"}
    # the build-up phase is one home possession
    b = kinds == "buildup"
    assert (lab[b] == "H").mean() > 0.9
    lo = segment.predict(table, feats=f, team_mode="touch", cost_team=1.0)
    hi = segment.predict(table, feats=f, team_mode="touch", cost_team=30.0)

    def n_poss(x):
        x = x[x != "X"]
        st = np.r_[True, x[1:] != x[:-1]]
        return int(np.isin(x[st], ["H", "A"]).sum())
    assert n_poss(lo) > n_poss(hi)


def test_touch_feature_sign():
    table, kinds = phases_table([("buildup", 10, 2.0)], seed=1)
    f = F.compute(table)
    near = f["tt"] != 0
    assert near.mean() > 0.5 and (f["tt"][near] > 0).mean() > 0.95      # home has the ball


def test_viterbi_per_frame_cost_moves_and_refuses():
    """A per-frame cost (n, K, K) lets a switch happen only where it is cheap; a wrong shape fails."""
    from mtah.models import hmm
    emit = np.zeros((10, 2))
    emit[:5, 0] = 1.2                    # state 0 is better before frame 5 (total 6)
    emit[5:, 1] = 1.0                    # state 1 is better from frame 5 on (total 5)
    fixed = np.array([[0.0, 6.0], [6.0, 0.0]])
    assert list(hmm.viterbi(emit, fixed)) == [0] * 10          # too costly to switch at all
    tv = np.broadcast_to(fixed, (10, 2, 2)).copy()
    tv[5] = [[0.0, 1.0], [1.0, 0.0]]
    assert list(hmm.viterbi(emit, tv)) == [0] * 5 + [1] * 5
    import pytest
    with pytest.raises(ValueError):
        hmm.viterbi(emit, tv[:9])
    with pytest.raises(ValueError):
        hmm.decode(emit, tv, np.ones(10), want_post=True)
