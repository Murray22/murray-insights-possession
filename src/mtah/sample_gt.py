"""Possession ground truth for Metrica's sample games, derived from their event data.

One label per tracking frame: H (home in possession, ball alive), A (away), D (dead ball) and X
(ungraded: before the kick-off of a half, after its last event, and between halves). The rules,
and every judgement call behind them, are written up in docs/sample_gt_rules.md; the short form:

1. Graded time. A half is graded from its first KICK OFF to the end frame of its END HALF event
   (or, without one, of its last event). Everything else is X.
2. Dead ball.
   - Hard stoppage: the ball physically leaves play. BALL OUT; a SHOT tagged OUT or GOAL. Dead
     from the event's end frame (the ball crossing the line / entering the goal).
   - Soft stoppage: the referee stops play (fouls, offside, hand ball, injuries ...). The events
     do not say when the whistle went, and play often continues for a second or two after a foul
     (advantage, delayed whistle, BALL LOST FORCED). So a SET PIECE that is not preceded by a hard
     stoppage closes a dead stretch that starts at the latest end frame of the live events before it.
   - Restart: a SET PIECE, or an event tagged GOAL KICK (goal kicks are PASS / BALL LOST in this
     data, not SET PIECE). After a hard stoppage with no tagged restart, the first ball-playing
     event (PASS, BALL LOST, SHOT, BALL OUT, CARRY, CHALLENGE) is an untagged restart, pulled back
     to a RECOVERY that leads into it within `pickup_link_frames`. A lone RECOVERY in dead time is a
     player (usually the goalkeeper) picking the ball up, not a restart.
3. Possession while alive. The restarting team has the ball from the restart frame. Events are
   grouped into runs of consecutive on-ball events (PASS, SHOT, BALL LOST, RECOVERY, BALL OUT,
   CARRY, SET PIECE) by one team; CHALLENGE, CARD and FAULT RECEIVED never change possession on
   their own. A run by the other team takes possession, from its first event, only if it shows
   control: a completed PASS, a CARRY, a SHOT or a SET PIECE, or a hold of at least
   `min_control_frames` (to the start of its losing event, or else to the other team's next on-ball
   event or the stoppage). Otherwise it was a touch, a deflection or a lost duel and possession
   stays with the last team. Loose balls (between a BALL LOST and the next RECOVERY) stay with the
   last team.

This is ground truth for a test bench on OTHER games. It is not, and cannot be, the organiser's
derivation (theirs comes from corrected tracking); docs/sample_gt_rules.md lists where it differs.
"""

from dataclasses import dataclass, field

import numpy as np

H, A, D, X = "H", "A", "D", "X"

ON_BALL = frozenset({"PASS", "SHOT", "BALL LOST", "RECOVERY", "BALL OUT", "CARRY", "SET PIECE"})
CONTROL_TYPES = frozenset({"PASS", "SHOT", "SET PIECE"})
LOSS_TYPES = frozenset({"BALL LOST", "BALL OUT"})
UNTAGGED_RESTART_TYPES = frozenset({"PASS", "BALL LOST", "SHOT", "BALL OUT", "CARRY", "CHALLENGE"})
IGNORED_TYPES = frozenset({"CARD"})


@dataclass
class GTConfig:
    # A run by the non-possessing team needs this many frames of hold, unless it completes a pass,
    # carries, shoots or takes a set piece. 25 frames = 1 s: the low end of CHALLENGE.md's "keeps
    # the ball for a few seconds", chosen before any scoring and never tuned (see the rules doc).
    min_control_frames: int = 25
    # A RECOVERY this close before an untagged restart is the start of that restart.
    pickup_link_frames: int = 50
    # Off only to show the tracking cross-check can fail (tests/test_sample_gt.py).
    soft_stoppages: bool = True


@dataclass
class GTResult:
    labels: np.ndarray            # (n_frames,) array of 'H' 'A' 'D' 'X'
    graded: list                  # [(first, last)] inclusive frame spans per half
    stoppages: list               # [(dead_start, restart, kind)] kind in hard/soft/end
    counts: dict = field(default_factory=dict)


def _tokens(event):
    return event["subtypes"]


def is_kickoff(e):
    return e["type"] == "SET PIECE" and "KICK OFF" in _tokens(e)


def is_end_half(e):
    return "END HALF" in _tokens(e)


def is_tagged_restart(e):
    return e["type"] == "SET PIECE" or "GOAL KICK" in _tokens(e)


def is_hard_trigger(e):
    s = _tokens(e)
    if e["type"] == "BALL OUT":
        return True
    if e["type"] == "SHOT" and ("OUT" in s or "GOAL" in s):
        return True
    return False


def _period_events(events, period):
    sub = events[(events["period"] == period) & (~events["type"].isin(IGNORED_TYPES))]
    sub = sub.sort_values(["start", "order"], kind="stable")
    return [row for _, row in sub.iterrows()]


def _graded_span(evs):
    kick = next((e for e in evs if is_kickoff(e)), None)
    if kick is None:
        return None
    end_half = [e for e in evs if is_end_half(e) and e["start"] >= kick["start"]]
    last = int(end_half[0]["end"]) if end_half else int(max(e["end"] for e in evs))
    return int(kick["start"]), last


def _stoppages(evs, first, last, cfg, counts):
    """Dead spans [dead_start, restart) inside a graded half, plus the restart team hints."""
    out = []
    state = "alive"
    span_max_end = first          # latest end frame of a live event in the current alive span
    dead_start = None
    i = 0
    live = [e for e in evs if first <= e["start"] <= last]
    while i < len(live):
        e = live[i]
        if state == "alive":
            if e["start"] == first and is_kickoff(e):
                i += 1
                continue
            if is_end_half(e):
                break
            if e["type"] == "SET PIECE" and cfg.soft_stoppages:
                # Soft stoppage: whistle not in the data; dead from the last live event's end.
                ds = int(span_max_end)
                if ds < e["start"]:
                    out.append((ds, int(e["start"]), "soft"))
                    counts["soft_stoppages"] += 1
                else:
                    counts["set_piece_without_gap"] += 1
                span_max_end = int(e["end"])
                i += 1
                continue
            span_max_end = max(span_max_end, int(e["end"]))
            if is_hard_trigger(e):
                state, dead_start = "dead", int(e["end"])
                counts["hard_stoppages"] += 1
            i += 1
            continue

        # dead
        if e["start"] <= dead_start:
            i += 1                 # straggler from the play that led to the stoppage
            continue
        if is_end_half(e):
            out.append((dead_start, last + 1, "end"))
            state = "ended"
            break
        if is_tagged_restart(e):
            restart = int(e["start"])
            out.append((dead_start, restart, "hard"))
            state, span_max_end = "alive", int(e["end"])
            i += 1
            continue
        if e["type"] in UNTAGGED_RESTART_TYPES:
            restart = int(e["start"])
            j = i - 1
            while (j >= 0 and live[j]["type"] == "RECOVERY" and live[j]["start"] > dead_start
                   and restart - live[j]["start"] <= cfg.pickup_link_frames):
                restart = int(live[j]["start"])
                j -= 1
            out.append((dead_start, restart, "hard"))
            counts["untagged_restarts"] += 1
            state, span_max_end = "alive", restart
            continue               # re-process e as a live event (it may itself be a trigger)
        counts["dead_time_events_ignored"] += 1
        i += 1
    if state == "dead":
        out.append((dead_start, last + 1, "end"))
        counts["dead_until_half_end"] += 1
    return out


def _possession_changes(span_events, start, stop, cfg, counts, fallback_owner):
    """[(frame, team)] owner changes inside one alive span [start, stop)."""
    on_ball = [e for e in span_events if e["type"] in ON_BALL and e["team"] in (H, A)]
    if not on_ball:
        counts["alive_spans_without_on_ball_event"] += 1
        return [(start, fallback_owner)] if fallback_owner else [(start, None)]

    runs = []
    for e in on_ball:
        if runs and runs[-1]["team"] == e["team"]:
            runs[-1]["events"].append(e)
        else:
            runs.append({"team": e["team"], "events": [e]})

    owner = runs[0]["team"]
    changes = [(start, owner)]
    for k in range(1, len(runs)):
        run = runs[k]
        if run["team"] == owner:
            continue
        evs = run["events"]
        first = int(evs[0]["start"])
        if any(e["type"] in CONTROL_TYPES for e in evs):
            controlled = True
        else:
            losses = [e for e in evs if e["type"] in LOSS_TYPES]
            if losses:
                # Measured to the moment the ball is released. The last loss, not the last event:
                # game 3 appends a 1-frame CARRY after a touch at the same start frame.
                hold = int(losses[-1]["start"]) - first
            else:
                nxt = int(runs[k + 1]["events"][0]["start"]) if k + 1 < len(runs) else stop
                hold = nxt - first
            controlled = hold >= cfg.min_control_frames
        if controlled:
            owner = run["team"]
            changes.append((max(first, start), owner))
            counts["possession_changes"] += 1
        else:
            counts["uncontrolled_runs_ignored"] += 1
    return changes


def build_labels(events, n_frames, config=None):
    """Per-frame H/A/D/X labels for one sample game from its normalised events."""
    cfg = config or GTConfig()
    counts = {k: 0 for k in ("hard_stoppages", "soft_stoppages", "set_piece_without_gap",
                             "untagged_restarts", "dead_time_events_ignored",
                             "dead_until_half_end", "alive_spans_without_on_ball_event",
                             "possession_changes", "uncontrolled_runs_ignored")}
    labels = np.full(n_frames, X, dtype="<U1")
    graded, all_stops = [], []
    for period in sorted(events["period"].unique()):
        evs = _period_events(events, period)
        span = _graded_span(evs)
        if span is None:
            continue
        first, last = span
        last = min(last, n_frames - 1)
        graded.append((first, last))
        stops = _stoppages(evs, first, last, cfg, counts)
        all_stops.extend(stops)

        # alive spans are the complement of the dead spans inside [first, last]
        alive, cursor = [], first
        for ds, rs, _ in stops:
            if ds > cursor:
                alive.append((cursor, ds))
            labels[ds:rs] = D
            cursor = max(cursor, rs)
        if cursor <= last:
            alive.append((cursor, last + 1))

        owner = None
        for a0, a1 in alive:
            span_events = [e for e in evs if a0 <= e["start"] < a1]
            changes = _possession_changes(span_events, a0, a1, cfg, counts, owner)
            for idx, (f, team) in enumerate(changes):
                f1 = changes[idx + 1][0] if idx + 1 < len(changes) else a1
                labels[f:f1] = team if team else D
                owner = team or owner
    return GTResult(labels, graded, all_stops, counts)


def sample_game_gt(game, config=None):
    """Convenience: labels for sample game 1, 2 or 3 aligned to its clean tracking frames."""
    from . import sample_io
    events = sample_io.normalised_events(game)
    n = sample_io.clean_tracking(game).n_frames
    return build_labels(events, n, config)


def label_stats(labels, fps=25):
    """Possessions, mean possession length and class shares, computed per graded run of frames."""
    labels = np.asarray(labels)
    graded = labels != X
    edges = np.diff(np.concatenate(([0], graded.astype(np.int8), [0])))
    runs = list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))
    lengths = []
    for s, e in runs:
        w = labels[s:e]
        b = np.flatnonzero(w[1:] != w[:-1]) + 1
        bounds = np.concatenate(([0], b, [len(w)]))
        for k in range(len(bounds) - 1):
            if w[bounds[k]] in (H, A):
                lengths.append(bounds[k + 1] - bounds[k])
    n_graded = int(graded.sum())
    return {
        "n_graded": n_graded,
        "H": int(np.sum(labels == H)), "A": int(np.sum(labels == A)),
        "D": int(np.sum(labels == D)),
        "dead_share": float(np.sum(labels == D)) / n_graded if n_graded else 0.0,
        "n_possessions": len(lengths),
        "mean_possession_s": float(np.mean(lengths)) / fps if lengths else 0.0,
        "median_possession_s": float(np.median(lengths)) / fps if lengths else 0.0,
        "possessions_per_graded_minute": len(lengths) / (n_graded / fps / 60.0) if n_graded else 0.0,
    }
