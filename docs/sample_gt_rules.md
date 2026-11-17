# Possession ground truth for the Metrica sample games: rules and judgement calls

This is the ground truth our test bench scores models against. It labels **other games**, Metrica
Sports' public sample games 1-3, never BAR-ATH. It is derived by code (`src/mtah/sample_gt.py`)
from Metrica's event data. Nobody labelled anything by hand. RULES.md allows derived data about
other games and asks that it be documented; this file is that documentation, and it goes into the
technical document.

Source: Metrica Sports sample data, <https://github.com/metrica-sports/sample-data> (games 1 and 2
in Metrica CSV format, game 3 in FIFA EPTS with JSON events). Event and tracking frames are
synchronised in that data; we index both as `tracking frame - 1`, the same convention as the ATD.

## What the labels mean

One label per tracking frame, using the competition alphabet:

| label | meaning here |
|---|---|
| `H` | home team in possession, ball alive |
| `A` | away team in possession, ball alive |
| `D` | ball dead (between a stoppage and its restart) |
| `X` | ungraded: before a half's kick-off, after its last event, and between halves |

## The rules

### 1. Graded time

A half is graded from the start frame of its first `SET PIECE / KICK OFF` to the end frame of its
`END HALF` event, inclusive. A half without an `END HALF` event (game 1's second half) is graded to
the end frame of its last event. Everything else is `X`.

### 2. Dead ball

**Hard stoppages: the ball physically leaves play.** These are `BALL OUT` (any subtype) and `SHOT`
tagged `OUT` or `GOAL`. The ball is dead from the event's **end** frame, the moment it crosses the
line or enters the goal. Events that start on or before that frame are still part of the play that
led to it. A `CHALLENGE` coded while the ball is rolling out is one example.

**Soft stoppages: the referee stops play.** These are fouls, offsides, hand balls and injuries.
The event data never says when the whistle went. Many fouls have no `FAULT RECEIVED` event at all:
game 3 has 12 of them for 28 free kicks, and other stoppages show up only as `CHALLENGE ... FAULT`,
an offside pass or `BALL LOST / FORCED`. Sometimes play goes on after the foul. In 4 of the 42
`FAULT RECEIVED` events in games 1-2, a regain and a `BALL LOST / FORCED` come before the free
kick. So a `FAULT RECEIVED` does not stop play by itself. Instead, **every `SET PIECE` that is not preceded by a hard
stoppage closes a dead stretch**. That stretch starts at the latest end frame of the live events
before the set piece.

**Restarts.** A restart is a `SET PIECE` (kick-off, throw-in, free kick, corner, penalty, including
`RETAKEN`), or **any event tagged `GOAL KICK`**. In this data, goal kicks are `PASS / GOAL KICK` or
`BALL LOST / GOAL KICK`, not set pieces. The dead stretch runs up to the restart's start frame.

**Dead-time events.** After a hard stoppage, a lone `RECOVERY` is a player, usually the goalkeeper,
collecting the ball for the restart. It does not restart play. Example: game 1 frame 118959, where
the keeper recovers 1.4 s after a shot goes wide and the goal kick follows 31 s later.

**Untagged restarts.** If a hard stoppage is followed by ball-playing events (`PASS`, `BALL LOST`,
`SHOT`, `BALL OUT`, `CARRY`, `CHALLENGE`) before any tagged restart, play must have restarted. The
first such event is taken as an untagged restart. It is pulled back to a `RECOVERY` that starts no
more than `pickup_link_frames` = 50 frames (2 s) before it, because that recovery is the start of
the restart. This happened 3 / 3 / 1 times in games 1 / 2 / 3.

If a half ends while the ball is dead, the half stays `D` to its last graded frame.

### 3. Possession while the ball is alive

- **Restart.** The team of the first on-ball event after the restart has the ball from the restart
  frame. That is normally the set-piece taker.
- **Runs.** On-ball events are `PASS`, `SHOT`, `BALL LOST`, `RECOVERY`, `BALL OUT`, `CARRY` and
  `SET PIECE`. Consecutive on-ball events by one team form a run. `CHALLENGE`, `CARD` and
  `FAULT RECEIVED` never change possession and do not break runs.
- **Control.** A run by the team without the ball takes possession, from the start frame of its
  first event, only if it shows control:
  - it contains a completed `PASS` (in Metrica data an unsuccessful pass is `BALL LOST`), a `SHOT`
    or a `SET PIECE`, or
  - it holds the ball for at least `min_control_frames` = **25 frames (1 s)**. Hold runs from the
    run's first event to the start of its last losing event (`BALL LOST` / `BALL OUT`). If there is
    no losing event, it runs to the other team's next on-ball event or the stoppage.
- **No control.** A run that fails the control test was a touch, a deflection or a duel won
  without control, and possession stays with the last team (CHALLENGE.md "What does not count as
  a change of possession"). Examples: an interception that goes straight out
  (`RECOVERY` + `BALL OUT` on the same frame), and a one-touch clearance straight back to the
  opponent.
- **Loose balls.** A loose ball, from a `BALL LOST` to the next controlled regain, stays with the
  team that lost it ("loose and contested balls stay with the last team").
- **Duels.** An aerial or ground duel (`CHALLENGE ... WON`) changes nothing on its own. Only a
  following controlled run does.

## Judgement calls, and why

1. **Whistle timing for soft stoppages** comes from the latest live event, not from the foul
   frame (see section 2). The evidence is the `BALL LOST / FORCED` events. They are
   Metrica's code for "play stopped while this player had the ball", and they come after the foul.
2. **`CARRY` does not count as control on its own; it only adds to the hold.** Games 1 and 2 have
   no `CARRY` events at all. Game 3 appends a `CARRY` to nearly every touch, including 1-frame
   carries straight after a one-touch interception. If carries counted, game 3 would have 381
   possessions against 227 and 222. With carries counted only through the hold, it has 248. The
   rule should not depend on an event type that only one of the three games uses.
3. **The hold runs to the last losing event, not to the last event.** Game 3 codes the 1-frame
   carry after the `BALL LOST` at the same start frame. This was a bug in the first version and
   is fixed and tested.
4. **The 25-frame control threshold** is the low end of CHALLENGE.md's proxy "keeps the ball for
   a few seconds". It was chosen before any scoring and has not been tuned. We take the low end
   because a Metrica `RECOVERY` already means a human annotator judged a regain. The alternative
   proxy, "more than two consecutive passes", would turn many genuine regains (recover, run,
   lose it) into non-events. Sensitivity of the possession count (game 1 / 2 / 3):
   0 frames: 379 / 362 / 405; 12: 234 / 224 / 256; **25: 227 / 222 / 248**; 50: 212 / 215 / 242.
5. **A single completed pass is enough** to count as control, not "more than two passes". A
   Metrica `PASS` is a successful pass, which already means the team had the ball under control.
6. **The change of possession happens at the regain frame**: the first event of the controlling
   run, not the moment the other team lost the ball. The gap between them is loose ball and stays
   with the last team.
7. **Shots off the woodwork** (`SHOT / WOODWORK` without `OUT`) stay in play. `BALL OUT / WOODWORK
   / GOAL` (game 1) is a hard stoppage like any `BALL OUT`.
8. **`BALL LOST / REFEREE HIT`** (once in game 1, once in game 3) is not a stoppage. In both cases
   the events show play going on.
9. **Retaken set pieces** are separate restarts. Game 2's second half starts with a kick-off
   tagged `RETAKEN`, then `BALL LOST / FORCED`, then a second kick-off 17 s later. We take that
   literally: alive, then a soft stoppage, then a restart.
10. **Stoppages without a set piece or ball-out event** cannot be detected. Examples are an injury
    stoppage restarted by a drop ball, or a referee's whistle that is not coded. Those stretches
    are labelled alive. This is the conservative reading because it invents no stoppages.
11. **Who owns a dead ball is not recorded**: dead is `D`, as in the competition.

## A second definition: "touch"

`GTConfig(min_control_frames=0)` uses the same dead-ball rules, but **any** on-ball event by the
other team takes possession, including a one-touch deflection. We report it beside the main
("controlled") definition for one reason. The organiser's own spellF1 table in EVALUATION.md (a
perfect prediction 8 frames late scores 0.8933, 1 s late scores 0.6539) is a property of the
BAR-ATH ground truth's possession lengths. A spell of length L delayed by k frames has IoU
(L-k)/(L+k), so those two numbers imply that roughly a third of BAR-ATH's true possessions are
shorter than about 3 s. A lognormal fit gives a median near 5 s. Our controlled definition has
median possessions of 9.7-10.9 s, only 10-12% under 3 s, and gives 0.982 / 0.856 on the same shift
test. The touch definition has medians of 4.1-4.6 s, 37-38% under 3 s, and gives 0.855 / 0.605.
So the organiser's ground truth sits between the two and closer to "touch". This fits a
mechanical tracking-based owner that flips on brief touches.

**We did not change the main rules to match this.** The touch definition was added after we saw
the comparison, as a second yardstick, so that model choices can be checked against both. It is
not a replacement.

## Cross-check against the clean tracking

Metrica's clean sample tracking stops tracking the ball while it is out of play. That gives an
independent check of the dead/alive rules, which use events only. On graded frames, our `D` agrees
with "ball absent from the clean tracking" on **0.933 / 0.976 / 0.997** of frames (games 1 / 2 /
3). Without the soft-stoppage rule, agreement falls to 0.805 / 0.826 / 0.905. That is the
`test_tracking_agreement_can_fail` test. The remaining disagreement in game 1 is mostly the
tracking keeping the ball through several goal-kick waits and one 160-second stoppage (frames
38477-42475). In those cases the events are right and the clean ball is not a dead flag.

Ball visibility in the clean tracking by our label: `H` / `A` frames 0.99-1.00, `D` frames 0.14 /
0.04 / 0.003. Because clean sample tracking marks dead time this way, the degrader must give dead
stretches a ball some of the time. Otherwise any model gets the dead-ball answer for free (see
docs/phaseA_report.md).

## Resulting statistics

| | game 1 | game 2 | game 3 | public BAR-ATH window (sanity only) |
|---|---|---|---|---|
| graded minutes | 94.7 | 93.4 | 95.1 | 5.0 |
| stoppages (hard / soft) | 99 (65 / 34) | 99 (65 / 34) | 100 (72 / 28) | n/a |
| mean stoppage | 25.1 s | 23.5 s | 18.0 s | n/a |
| dead share | 0.437 | 0.415 | 0.316 | 0.196 (1,469 / 7,500) |
| **controlled**: possessions | 227 | 222 | 248 | 17 |
| mean / median possession | 14.1 / 9.7 s | 14.8 / 10.4 s | 15.7 / 10.9 s | 14.2 s mean |
| possessions per graded minute | 2.4 | 2.4 | 2.6 | 3.4 |
| **touch**: possessions | 379 | 362 | 405 | |
| mean / median possession | 8.4 / 4.1 s | 9.1 / 4.5 s | 9.6 / 4.6 s | |
| home share of alive frames | 0.518 | 0.529 | 0.518 | 0.573 |

The public window column is the organiser's published summary of the 5-minute public ground truth
(`ref/evaluation/gt/BAR-ATH_possession_gt_public_meta.json`). It is printed for comparison only.
**None of our rules or constants were tuned to it**, and RULES.md forbids it as a tuning target. It
is one 5-minute window from the start of a match, so it is not expected to match full-game
statistics. Its dead share in particular covers 5 minutes, not 95.
