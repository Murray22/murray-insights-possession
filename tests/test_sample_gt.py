"""The event-to-label rules on small hand-made sequences, plus a cross-check on real sample data."""

import numpy as np
import pandas as pd
import pytest

from mtah import sample_gt
from mtah.sample_gt import GTConfig, build_labels
from conftest import needs_sample


def ev(team, type_, start, end=None, sub=(), period=1):
    return {"team": team, "type": type_, "subtypes": frozenset(sub), "period": period,
            "start": start, "end": start if end is None else end, "player": None,
            "start_x": np.nan, "start_y": np.nan, "end_x": np.nan, "end_y": np.nan}


def frame(rows):
    df = pd.DataFrame(rows)
    df["order"] = np.arange(len(df))
    return df


def labels(rows, n=1000, **cfg):
    return build_labels(frame(rows), n, GTConfig(**cfg) if cfg else None).labels


def runs(lab):
    """Run-length encoding as [(label, start, stop)] for readable asserts."""
    b = np.flatnonzero(lab[1:] != lab[:-1]) + 1
    bounds = np.concatenate(([0], b, [len(lab)]))
    return [(str(lab[bounds[i]]), int(bounds[i]), int(bounds[i + 1])) for i in range(len(bounds) - 1)]


KICK = ev("H", "SET PIECE", 50, sub=("KICK OFF",))


def end_half(at, team="H"):
    return ev(team, "BALL LOST", at, sub=("END HALF",))


def test_dead_between_ball_out_and_restart():
    lab = labels([KICK, ev("H", "PASS", 50, 80), ev("H", "BALL OUT", 90, 100),
                  ev("A", "SET PIECE", 200, sub=("THROW IN",)), ev("A", "PASS", 200, 230),
                  end_half(300, "A")])
    assert runs(lab) == [("X", 0, 50), ("H", 50, 100), ("D", 100, 200), ("A", 200, 301),
                         ("X", 301, 1000)]


def test_loose_ball_stays_with_last_team():
    lab = labels([KICK, ev("H", "PASS", 50, 70), ev("H", "BALL LOST", 100, 140),
                  ev("A", "RECOVERY", 160), ev("A", "PASS", 160, 190), end_half(300, "A")])
    # 100..159: ball loose after H lost it -> still H; A only from its controlled regain
    assert runs(lab)[1:3] == [("H", 50, 160), ("A", 160, 301)]


def test_touch_without_control_does_not_change_possession():
    lab = labels([KICK, ev("H", "PASS", 50, 70), ev("H", "BALL LOST", 100, 130),
                  ev("A", "RECOVERY", 130, sub=("INTERCEPTION",)),
                  ev("A", "BALL LOST", 130, 150, sub=("INTERCEPTION",)),  # one-touch, lost at once
                  ev("H", "RECOVERY", 150), ev("H", "PASS", 150, 170), end_half(300)])
    assert runs(lab)[1] == ("H", 50, 301)


def test_deflection_out_stays_with_attacking_team():
    lab = labels([KICK, ev("H", "PASS", 50, 70), ev("H", "BALL LOST", 100, 130),
                  ev("A", "RECOVERY", 130), ev("A", "BALL OUT", 130, 150),
                  ev("H", "SET PIECE", 250, sub=("CORNER KICK",)), ev("H", "PASS", 250, 270),
                  end_half(400)])
    assert runs(lab)[1:4] == [("H", 50, 150), ("D", 150, 250), ("H", 250, 401)]


def test_aerial_duel_won_without_control_is_ignored():
    lab = labels([KICK, ev("H", "PASS", 50, 100), ev("A", "CHALLENGE", 100, sub=("AERIAL", "WON")),
                  ev("H", "CHALLENGE", 100, sub=("AERIAL", "LOST")), ev("H", "RECOVERY", 110),
                  ev("H", "PASS", 110, 140), end_half(300)])
    assert runs(lab)[1] == ("H", 50, 301)


def test_hold_threshold_decides_control():
    rows = [KICK, ev("H", "PASS", 50, 70), ev("H", "BALL LOST", 90, 100), ev("A", "RECOVERY", 100),
            ev("A", "BALL LOST", 130, 150), ev("H", "RECOVERY", 150), ev("H", "PASS", 150, 170),
            end_half(300)]
    # A held it 30 frames: a possession at the 25-frame default, a touch at 50
    assert runs(labels(rows))[1:4] == [("H", 50, 100), ("A", 100, 150), ("H", 150, 301)]
    assert runs(labels(rows, min_control_frames=50))[1] == ("H", 50, 301)


def test_soft_stoppage_foul_dead_until_free_kick():
    lab = labels([KICK, ev("H", "PASS", 50, 90), ev("A", "FAULT RECEIVED", 120),
                  ev("H", "CHALLENGE", 120, sub=("FAULT", "LOST")), ev("H", "CARD", 160, sub=("YELLOW",)),
                  ev("A", "SET PIECE", 300, sub=("FREE KICK",)), ev("A", "PASS", 300, 330),
                  end_half(500, "A")])
    assert runs(lab)[1:4] == [("H", 50, 120), ("D", 120, 300), ("A", 300, 501)]


def test_play_on_after_foul_moves_dead_start():
    # advantage / late whistle: H keeps playing after the foul, then BALL LOST FORCED
    lab = labels([KICK, ev("H", "PASS", 50, 90), ev("H", "FAULT RECEIVED", 100),
                  ev("A", "CHALLENGE", 100, sub=("FAULT", "LOST")), ev("H", "PASS", 110, 130),
                  ev("H", "BALL LOST", 140, sub=("FORCED",)),
                  ev("H", "SET PIECE", 300, sub=("FREE KICK",)), end_half(500)])
    assert runs(lab)[1:4] == [("H", 50, 140), ("D", 140, 300), ("H", 300, 501)]


def test_goal_kick_and_goalkeeper_pickup():
    lab = labels([KICK, ev("H", "PASS", 50, 80), ev("H", "SHOT", 100, 120, sub=("OFF TARGET", "OUT")),
                  ev("A", "RECOVERY", 150),               # keeper fetches the ball: dead time
                  ev("A", "RECOVERY", 400), ev("A", "PASS", 400, 450, sub=("GOAL KICK",)),
                  end_half(600, "A")])
    assert runs(lab)[1:4] == [("H", 50, 120), ("D", 120, 400), ("A", 400, 601)]


def test_untagged_restart_after_ball_out():
    lab = labels([KICK, ev("H", "PASS", 50, 80), ev("H", "BALL OUT", 90, 110),
                  ev("A", "RECOVERY", 380), ev("A", "PASS", 400, 450), end_half(600, "A")])
    # the pass is the untagged restart, pulled back to the recovery 20 frames before it
    assert runs(lab)[1:4] == [("H", 50, 110), ("D", 110, 380), ("A", 380, 601)]


def test_goal_then_kickoff():
    lab = labels([KICK, ev("H", "PASS", 50, 80), ev("H", "SHOT", 100, 110, sub=("ON TARGET", "GOAL")),
                  ev("A", "SET PIECE", 700, sub=("KICK OFF",)), ev("A", "PASS", 700, 720),
                  end_half(900, "A")])
    assert runs(lab)[1:4] == [("H", 50, 110), ("D", 110, 700), ("A", 700, 901)]


def test_two_halves_and_ungraded_gap():
    rows = [KICK, ev("H", "PASS", 50, 80), end_half(200),
            ev("A", "SET PIECE", 400, sub=("KICK OFF",), period=2), ev("A", "PASS", 400, 420, period=2),
            ev("A", "BALL OUT", 500, 520, period=2)]
    lab = labels(rows)
    # no END HALF in period 2: graded to the last event's end frame, inclusive
    assert runs(lab) == [("X", 0, 50), ("H", 50, 201), ("X", 201, 400), ("A", 400, 520),
                         ("D", 520, 521), ("X", 521, 1000)]


def test_rules_can_say_no():
    # Guard against a vacuous builder: changing one event must change the labels.
    base = [KICK, ev("H", "PASS", 50, 80), ev("H", "BALL OUT", 90, 100),
            ev("A", "SET PIECE", 200, sub=("THROW IN",)), end_half(300, "A")]
    other = list(base)
    other[3] = ev("H", "SET PIECE", 200, sub=("THROW IN",))
    assert not np.array_equal(labels(base), labels(other))


def _tracking_agreement(game, config):
    from mtah import sample_io
    events = sample_io.normalised_events(game)
    table = sample_io.clean_tracking(game)
    lab = build_labels(events, table.n_frames, config).labels
    vis = table.ball_visible()
    g = lab != "X"
    return ((lab[g] == "D") == ~vis[g]).mean()


@needs_sample
@pytest.mark.parametrize("game", [1, 2, 3])
def test_dead_ball_agrees_with_clean_tracking(game):
    """Independent check: clean Metrica tracking mostly drops the ball when it is out of play.

    Our D comes from events only; agreement with the tracking's ball-absent frames on graded
    frames is evidence the dead/alive rules are right. Measured 0.933 / 0.976 / 0.997 for games
    1-3 (game 1's tracking keeps the ball through several goal-kick waits and one long stoppage,
    where the events are right and the ball flag is not a dead flag).
    """
    assert _tracking_agreement(game, GTConfig()) > 0.92


@needs_sample
@pytest.mark.parametrize("game", [1, 2, 3])
def test_tracking_agreement_can_fail(game):
    # Dropping the soft-stoppage rule (fouls, offsides) must cost at least 5 points of agreement.
    good = _tracking_agreement(game, GTConfig())
    broken = _tracking_agreement(game, GTConfig(soft_stoppages=False))
    assert good - broken > 0.05


@needs_sample
def test_sample_gt_shapes():
    for game in (1, 2, 3):
        res = sample_gt.sample_game_gt(game)
        stats = sample_gt.label_stats(res.labels)
        assert 150 < stats["n_possessions"] < 400
        assert 0.2 < stats["dead_share"] < 0.55
        assert set(np.unique(res.labels)) <= {"H", "A", "D", "X"}
