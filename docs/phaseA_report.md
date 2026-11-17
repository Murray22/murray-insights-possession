# Phase A report: the validation harness

Date: 2026-11-03. Everything here runs on Metrica Sports' public sample games 1-3
(<https://github.com/metrica-sports/sample-data>). No BAR-ATH data was used or looked for. The
only BAR-ATH material used is the organiser's own public text: the published full-match baseline
scores and the spellF1 shift table in EVALUATION.md, plus the public window's summary in its meta
JSON. We used it as a realism check and printed it for comparison. No labels from the public
window were read.

Reproduce: `PYTHONPATH=src .venv/bin/python -m mtah.bench --seeds 5 --sweep-seeds 3 --bootstrap
1000 --workers 6` (8 minutes on 6 cores). It writes `runs/phaseA/fidelity.json`, which is
gitignored. Tests: `.venv/bin/python -m pytest`.

## What was built (`src/mtah/`)

| module | job |
|---|---|
| `frames.py` | `FrameTable`, the one in-memory format for sample games and the real ATD: frame, period, ball x/y (NaN if missing), per-slot x/y, fixed slot team, pitch size, normalised 0-1 coordinates |
| `sample_io.py` | readers for sample games 1-2 (Metrica CSV) and 3 (EPTS through kloppy, JSON events) into normalised events and a clean `FrameTable`; cache in `runs/cache` |
| `sample_gt.py` | per-frame H/A/D/X ground truth from events (rules in `docs/sample_gt_rules.md`) |
| `degrade.py` | clean tracking to ATD-shaped tracking, seeded, every knob in `DegradeConfig` |
| `score.py` | the organiser's `possession_metrics.py`, imported from `ref/evaluation` (not copied), per game and pooled, with its bootstrap |
| `baseline.py` | the organiser baseline on a `FrameTable` (nearest slot in metres, majority vote over +/-0/25/50/125) |
| `bench.py` | the fidelity check below |

## Label rules, summary

The full rules and all 11 judgement calls are in `docs/sample_gt_rules.md`.

- **Dead (D).** Dead from a `BALL OUT`, or a `SHOT` tagged OUT or GOAL, at the event's end frame,
  until the restart. The restart is a `SET PIECE` or a `GOAL KICK`-tagged pass. A `SET PIECE` with
  no ball-out before it (a foul, offside or hand ball) closes a dead stretch that starts at the
  last live event. The whistle is not in the data, and play sometimes goes on after the foul.
  Goalkeeper pickups in dead time are ignored. After a ball-out with no tagged restart, the first
  ball-playing event counts as an untagged restart (7 cases over the three games).
- **Possession (H / A).** The restarting team has the ball first. A run of on-ball events by the
  other team takes it over only with control: a completed pass, a shot, a set piece, or a hold of
  25 frames or more. Touches, deflections, duels and loose balls leave possession with the last
  team.
- **Graded (X outside).** From each half's first kick-off to its `END HALF` (or last event).
- **Second definition, "touch".** Same dead-ball rules, but every on-ball event by the other team
  flips possession. It was added after we compared our possession lengths with the organiser's
  shift table (below). It is not a replacement.
- **Independent check.** Clean sample tracking drops the ball when it is dead. Our event-only `D`
  agrees with "ball absent" on 0.933 / 0.976 / 0.997 of graded frames. Removing the
  soft-stoppage rule drops this to 0.805 / 0.826 / 0.905, and a test asserts that drop.

## Sample-game statistics

| | game 1 | game 2 | game 3 | public BAR-ATH 5 min (sanity only) |
|---|---|---|---|---|
| graded minutes | 94.7 | 93.4 | 95.1 | 5.0 |
| stoppages (hard / soft), mean length | 99 (65 / 34), 25.1 s | 99 (65 / 34), 23.5 s | 100 (72 / 28), 18.0 s | n/a |
| dead share | 0.437 | 0.415 | 0.316 | 0.196 |
| controlled: possessions, mean / median | 227, 14.1 / 9.7 s | 222, 14.8 / 10.4 s | 248, 15.7 / 10.9 s | 17, 14.2 s mean |
| touch: possessions, mean / median | 379, 8.4 / 4.1 s | 362, 9.1 / 4.5 s | 405, 9.6 / 4.6 s | |
| share of possessions under 3 s (controlled / touch) | 0.12 / 0.37 | 0.12 / 0.38 | 0.10 / 0.37 | |
| home share of alive frames | 0.518 | 0.529 | 0.518 | 0.573 |

The public-window column is the organiser's published summary and is printed for comparison only.
Nothing was tuned to it. It is one 5-minute stretch, so its dead share is not comparable to a
95-minute game's.

**What the organiser's own shift table says about their ground truth.** EVALUATION.md reports
that a perfect prediction scores 0.8933 when 8 frames late and 0.6539 when 1 s late. That
depends only on the true possession lengths: a spell of length L delayed by k frames has IoU
(L-k)/(L+k). The organiser's numbers fit a lognormal with a median near 5 s, with about a third
of possessions under 3 s. The same test on our ground truth gives 0.982 / 0.856 under
"controlled" and 0.855 / 0.605 under "touch". The official truth sits between our two
definitions, closer to "touch". `tests/test_score.py` checks both the arithmetic and the
reproduction.

## Degrader settings (defaults, `DegradeConfig()`)

| what | setting | basis |
|---|---|---|
| slots | 23 per team (46), fixed team, random free slot on each new tracklet | DATA.md (46 slots) |
| tracked objects | 20.5 per graded frame; achieved 20.49-20.50 | DATA.md |
| tracklet breaks | per-frame end hazard, x2 when another player is within 1.5 m; gaps lognormal (median 25 frames, sigma 1); 30% re-acquired within 1-3 frames; about 7,400-7,600 tracklets per game | guess, recalibrate on ATD slot runs |
| identity swaps | same-team pairs within 1.5 m swap with p = 0.3 per encounter (175-490 per game) | guess |
| duplicates | 1.5 bursts per graded minute, lognormal length (median 25 frames), offset sd 0.8 m; 0.6-0.8% of frames show more than 22 objects | DATA.md "occasionally" |
| position noise | AR(1), rho 0.9, sd 0.15 m (players), 0.20 m (ball) | guess |
| ball missing | **41.0%** of graded frames in every game and seed | DATA.md |
| ball in live play | lognormal bursts (median 20 frames, sigma 1), start rate x3 when the ball moves faster than 15 m/s (in-the-air proxy), x2 when 3+ players are within 3 m, x2 within 3 m of a line; base rate solved to hit 41% overall, which lands at 18-25% of live frames | DATA.md shape, guessed numbers |
| ball in dead stretches | synthetic ball: held where last seen, carried to where it reappears, held there. Visible only at the start and end of each stoppage, on **25%** of dead frames (`dead_ball_visible`) | **chosen from a grid against public aggregates, see below** |
| burst-length override | `ball_burst_lengths=<array>` replaces the lognormal with an empirical distribution (tested) | for ATD recalibration |

`dead_ball_visible` is the one assumption DATA.md cannot pin down, and it matters most for
macroF1. We picked 0.25 from the coarse grid {0, 0.25, 0.5, 0.75}. It was the only point where the
baselines' macroF1 moves the way the published full-match numbers do: up from raw to smooth25 /
50 and down at smooth125, with levels within about 0.05. The full sweep is below. This is a coarse
calibration against public aggregate results, not against any ground truth. It should be replaced
by a measurement from the real ATD once it arrives (see the next-phase notes).

## Baseline fidelity

Pooled games 1-3; interval = organiser block bootstrap (1000 draws, 5-95%). Degraded = mean over 5
seeds, with the [min, max] over seeds in brackets after the bootstrap interval (bootstrap of seed 0).

### Main table

| baseline | **organiser full match** spellF1 / macroF1 | clean, controlled | clean, touch | degraded, controlled | **degraded, touch** |
|---|---|---|---|---|---|
| raw | 0.0486 / 0.6532 | 0.059 [0.055, 0.063] / 0.834 [0.817, 0.849] | 0.104 [0.099, 0.109] / 0.836 | 0.019 [0.018, 0.022] {0.018-0.020} / 0.637 [0.623, 0.651] | 0.043 [0.041, 0.048] {0.040-0.045} / 0.638 [0.624, 0.652] |
| smooth25 | 0.1996 / 0.6752 | 0.327 [0.315, 0.338] / 0.847 [0.827, 0.862] | 0.365 [0.349, 0.379] / 0.846 | 0.175 [0.170, 0.189] {0.165-0.180} / 0.710 [0.690, 0.726] | 0.214 [0.210, 0.231] {0.204-0.221} / 0.710 [0.690, 0.727] |
| smooth50 | **0.2433** / 0.6724 | 0.434 [0.419, 0.448] / 0.835 [0.815, 0.850] | 0.412 [0.393, 0.431] / 0.830 | 0.286 [0.282, 0.309] {0.275-0.296} / 0.711 [0.688, 0.727] | **0.292** [0.283, 0.315] {0.284-0.299} / 0.708 [0.685, 0.723] |
| smooth125 | 0.2242 / 0.6356 | 0.398 [0.377, 0.418] / 0.760 [0.738, 0.778] | 0.298 [0.278, 0.317] / 0.746 | **0.353** [0.339, 0.379] {0.336-0.366} / 0.652 [0.626, 0.670] | 0.276 [0.254, 0.295] {0.266-0.281} / 0.641 [0.614, 0.657] |

Possessions predicted / true (pooled): clean raw 6,773, s25 2,036, s50 1,500, s125 1,123
against 697 (controlled) or 1,146 (touch). Degraded: 8,785 / 2,739 / 1,809 / 1,032.

Rankings (best first):

| | spellF1 | macroF1 |
|---|---|---|
| **organiser full match** | s50 > s125 > s25 > raw | s25 > s50 > raw > s125 |
| clean, controlled | s50 > s125 > s25 > raw | s25 > s50 > raw > s125 |
| clean, touch | s50 > s25 > s125 > raw | s25 > raw > s50 > s125 |
| degraded, controlled | **s125** > s50 > s25 > raw | s50 > s25 > s125 > raw |
| **degraded, touch (default bench)** | **s50 > s125 > s25 > raw** | s25 > s50 > s125 > raw (s125 vs raw 0.641 vs 0.638) |

Per game, degraded touch, spellF1 s50 / s125: game 1 0.303 / 0.275, game 2 0.306 / 0.274,
game 3 0.272 / 0.278. Two of three games put s50 first; game 3 is a tie.

### Sensitivity sweep (3 seeds each; pooled spellF1 of raw / s25 / s50 / s125, then the order)

| degrader variant | GT controlled | GT touch |
|---|---|---|
| default (dead_vis 0.25, ends) | 0.019 / 0.175 / 0.286 / 0.353: s125 first | 0.043 / 0.214 / 0.292 / 0.276: **s50 first** |
| dead_ball_visible 0.00 | 0.036 / 0.238 / 0.347 / 0.365: s125 | 0.068 / 0.278 / 0.344 / 0.283: s50 |
| dead_ball_visible 0.50 | 0.011 / 0.131 / 0.241 / 0.322: s125 | 0.028 / 0.165 / 0.252 / 0.256: s125, tie |
| dead_ball_visible 0.75 | 0.005 / 0.099 / 0.199 / 0.299: s125 | 0.016 / 0.131 / 0.216 / 0.243: s125 |
| dead ball hidden in short bursts | 0.017 / 0.143 / 0.255 / 0.321: s125 | 0.035 / 0.180 / 0.260 / 0.251: s50, tie |
| short ball bursts (median 5 frames) | 0.004 / 0.211 / 0.325 / 0.377: s125 | 0.015 / 0.252 / 0.323 / 0.285: s50 |
| long ball bursts (median 75 frames) | 0.028 / 0.175 / 0.261 / 0.315: s125 | 0.056 / 0.216 / 0.267 / 0.251: s50 |
| no context weights | 0.018 / 0.169 / 0.281 / 0.352: s125 | 0.042 / 0.211 / 0.287 / 0.277: s50 |
| ball missing 30% | 0.031 / 0.223 / 0.336 / 0.371: s125 | 0.060 / 0.265 / 0.332 / 0.282: s50 |
| ball missing 50% | 0.012 / 0.142 / 0.252 / 0.332: s125 | 0.030 / 0.179 / 0.264 / 0.267: s125, tie |
| players not degraded | 0.025 / 0.201 / 0.339 / 0.386: s125 | 0.050 / 0.239 / 0.335 / 0.291: s50 |

macroF1 for the dead-ball variants (raw / s25 / s50 / s125): dead_vis 0: 0.780 / 0.813 / 0.808 /
0.744. dead_vis 0.5: 0.494 / 0.597 / 0.602 / 0.551. dead_vis 0.75: 0.348 / 0.472 / 0.486 / 0.465.
Dead hidden in short bursts: 0.637 / 0.506 / 0.462 / 0.436, where smoothing lowers macroF1, the
opposite of the organiser's result. All other variants stay at 0.63-0.74 with the organiser's
macroF1 shape.

### Verdict

- **Does the bench rank the baselines like the full match?** Yes, with the "touch" ground truth.
  Degraded tracking with touch GT reproduces the published spellF1 order exactly (s50 > s125 >
  s25 > raw) and the macroF1 order up to a 0.003 swap of raw and s125, which is noise. Its
  levels are within 0.002-0.05 of the published ones. smooth50 is 0.292 against 0.243, so the
  bench is slightly easier, as expected from cleaner-than-real tracking under the noise.
- **What changes the ranking: the ground-truth definition, not the degrader.** Under
  "controlled", heavier smoothing wins on spellF1 in every degrader variant (s125 first, by 0.02-
  0.10). Under "touch", s50 is first or tied in every variant. The organiser's ground truth has a
  third of its possessions under 3 s (shift table), which is why one window wins over the other.
  Selecting a model's smoothing or switching cost on the controlled GT alone would over-smooth.
- **What changes the levels: how visible the dead ball is.** macroF1 runs from 0.35 to 0.81 over
  the plausible range of `dead_ball_visible`, and hiding the dead ball in short bursts reverses
  the effect of smoothing on macroF1. spellF1 levels move by up to about 0.15 but keep their order
  within a GT definition, except the s50/s125 near-ties at dead_vis >= 0.5 or 50% missing.
- Clean tracking is far too easy (macroF1 0.83-0.85, because the clean ball is nearly a dead-ball
  oracle). Never select a model on clean tracking.

## Tests

`.venv/bin/python -m pytest`: **57 passed** (2026-11-03, 40 s). The ones that show a
gate can say no:

- `test_score.py`: a perfect prediction scores spellF1 1.0 and a 1-frame offset is free. A shift
  of k frames equals the analytic IoU formula for k = 3, 8, 25, 50. On a GT with the organiser's
  implied length distribution, the EVALUATION.md table is reproduced (0.9818 / 0.8933 / 0.6539
  within 0.03). On long-possession GT, the same 1 s shift scores above 0.85, so the check can
  fail. A constant H, A or D guess scores 0, and so do swapped teams. Pooling keeps games apart.
  The bootstrap brackets the point. The scorer is the unmodified file in `ref/evaluation`.
- `test_sample_gt.py`: 13 hand-made event sequences cover ball out to restart, a loose ball kept
  by the last team, a one-touch interception, a deflection out, an aerial duel, the hold
  threshold at 25 vs 50, a foul to free kick, play on after a foul, a goal kick with a keeper
  pickup, an untagged restart, a goal to kick-off, two halves with an ungraded gap, and a check
  that one changed event changes the labels. On real data, the tracking agreement is above 0.92,
  and dropping the soft-stoppage rule must cost at least 5 points.
- `test_degrade.py`: the degrader is deterministic per seed and different across seeds. Ball
  missing hits 0.30 / 0.41 / 0.50 within 0.01, and the target counts graded frames only. Missing
  ball comes in bursts. Empirical burst lengths are honoured. `dead_ball_visible` 0 / 0.5 / 1 is
  honoured. Slots are anonymous, 46 in all, with fixed teams. The mean count is 20.5 +/- 0.3, the
  odd frame has more than 22, slots are re-used, and slot positions sit on their own team's
  players. Swaps happen only when enabled. Sample game 2 hits both targets.
- `test_baseline.py`: smoothing is identical to `ref/starter/baseline.py` `smooth()` at all four
  windows. Nearest player is computed in metres, and a missing ball gives no answer.
- `test_bench.py`: both GT variants are scored, a constant guess is distinguished from a perfect
  one, and the summaries and orderings are right.

## Found, not fixed

1. **Our main GT rules are less touchy than the organiser's.** The shift-table analysis puts the
   official truth near the touch definition (median possession about 5 s). We kept the rules
   written from CHALLENGE.md and added "touch" beside them, rather than tuning. A middle setting,
   such as a hold threshold of a few frames, was not tried, on purpose.
2. **`dead_ball_visible` = 0.25 is a coarse fit to public aggregates.** It needs a measurement.
   On the real ATD, ball visibility inside Smart Tagging set-piece windows, or in long periods of
   near-zero team speed, would estimate it without any labels.
3. **The synthetic dead-ball path is an artefact**: the ball is held, moved in a straight line
   and held again, with AR(1) noise. A learned model could key on it, for example by learning
   "constant velocity means dead". Learned emissions need leave-one-game-out validation and a
   feature check, and the path should be randomised further before Phase C3.
4. **Tracklet and slot dynamics are uncalibrated.** Mean tracklet length is about 15 s, swap and
   duplicate rates are guesses, and there are no wrong-team slots, ghost balls or referee tracks.
   Measure slot run lengths, count distributions and ball-jump statistics on the real ATD.
5. **The in-the-air proxy is ball speed above 15 m/s.** The sample data has no ball height.
6. **The GT cannot see stoppages without a set piece or ball-out event** (injury and drop ball).
   Those stretches are labelled alive.
7. **Game 1's clean tracking keeps the ball through some goal-kick waits and one 160 s
   stoppage**, so "clean ball absent" is not a perfect dead flag. The degrader's dead-ball
   synthesis only touches interior NaN runs, so those stretches keep a real visible ball and are
   slightly easier.
8. **Bootstrap intervals resample possessions within our three games.** They do not include
   between-game variance. Per-game spread is about 0.03-0.05 in spellF1, comparable to the
   intervals.
9. **Game 3's dead share (0.32) is lower than games 1-2 (0.42-0.44).** Tracking agreement is
   0.997, so it looks real, but it means the pooled numbers mix different game rhythms.
10. Game 3's event coding (CARRY, 1-frame carries after touches) differs from games 1-2. The rules
    neutralise it (see judgement calls 2-3), but further event-type differences may exist that
    we did not notice.

## Notes for the next phase (models)

- **Score every model on degraded tracking against both GT definitions**, and prefer choices that
  hold under both. If they disagree, "touch" is the better proxy for the official truth. The
  switching cost in an HMM/Viterbi model is exactly the knob the two definitions pull in opposite
  directions: about a third of official possessions are under 3 s.
- **The dead-ball detector carries macroF1.** In the bench, `D` is where the baseline loses most
  (degraded `D` F1 is about 0.65 at best for smooth50). It also decides whether smoothing helps.
  Ball absence is only weak evidence on the real ATD. Use player kinematics (a team-wide speed
  collapse, restart shapes), the ball off the pitch, and Smart Tagging set pieces.
- Vary `dead_ball_visible` (0 to 0.5) and the burst lengths whenever comparing models. A model
  whose lead disappears inside that range is not robustly better. Re-run with the ATD-measured
  burst distribution (`ball_burst_lengths`) as soon as the data lands.
- Learned emissions: train on degraded games with leave-one-game-out (three games), use several
  degrader seeds as augmentation, and never train on clean tracking, where ball absence leaks the
  dead-ball label.
- Differences under about 0.03-0.04 in pooled spellF1 are inside the bootstrap and seed spread
  here, the same as the organiser's noise band.
