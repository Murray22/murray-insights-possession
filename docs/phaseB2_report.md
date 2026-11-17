# Phase B2 report: a bench that looks like the real ATD, and models rebuilt on it

Date: 2026-11-05. Inputs: the competition ATD and its automatic Smart Tagging, measured as
distributions only; Metrica sample games 1-3 for everything scored. No label of BAR-ATH was read.
The full-match ground truth is not ours to have. The public 5-minute window was used once, at the
very end, as a pipeline check (last section), and nothing changed after it.

## Summary

- **The calibrated degrader (`atd_cal`) reproduces the real ATD's gap structure.** Missing-ball
  runs: real p10 / p50 / p90 = 12 / 268 / 845 frames, degraded 49 / 256 / 803 (Phase A degrader:
  4 / 22 / 176). 99.8% of missing frames sit in runs of 25+ frames (real 99.9%, Phase A 92%).
  Player tracking is smooth (jitter 0.011 m against real 0.007, Phase A 0.060) with long tracklets
  and mass dropouts (objects p5 12 against real 14, Phase A 18). A calibration gate
  (`atd_profile.check_calibration`) passes `atd_cal` and fails the Phase A degrader on 10 of 12
  rules.
- **The bench got harder in the right direction.** On `atd_cal` smooth50 scores 0.262 touch
  spellF1. The organiser's full-match number is 0.2433; the Phase A bench gave 0.292.
- **Current best: `gbm_v5_nogap`**: 0.588 [0.558, 0.613] touch spellF1, 0.734 [0.715, 0.763]
  controlled, macroF1 0.908, against gbm_v4 0.499 / 0.604 and smooth50 0.262 / 0.270 on the same
  tables. Two steps get there, each about +0.044 touch:
  1. retraining gbm_v4's design on the calibrated degrader (gbm_v5a);
  2. adding player-only team evidence inside gaps.

  The ball-at-both-ends and visibility features add nothing (gbm_v5, which has them, ties it in
  every sweep column). The lead over gbm_v4 holds in all 9 calibrated settings (+0.083 to +0.134
  touch), including 3 held out of training.
- **A real transfer bug, found by the label-free diagnostics and fixed:** the attack direction
  was read from each half's first 10 s, and the real second half does not open in a kick-off
  layout, so every attack-relative feature had home attacking the wrong way for 45 minutes.
  `gbm_v5`'s agreement with Smart Tagging's attacking-phase team in the second half went from
  0.628 to 0.785. The direction now comes from the whole half (keepers and last lines). It agrees
  with the old rule on every sample-game half, so no bench number moves.
- **D is 43-45% of the real match for every learned model, 83-87% of ball-missing frames.** A
  label-free check from the players' speeds puts real gaps at about 91% dead (86-100% across
  thresholds and assumptions). So this is probably not an overcall, contrary to Phase B's reading
  and the video pilot's. It rests on the sample games' players moving like BAR-ATH's. Nothing was
  tuned to it.

## Task 1: what the real ATD looks like (distributions only)

`PYTHONPATH=src .venv/bin/python -m mtah.atd_profile --variants default atd_cal --seeds 2`
profiles the real ATD and six degraded tables per variant with one function (`atd_profile.profile`)
and prints the table below. The real profile goes to `runs/phaseB2/profile_real.json`.

**Ball.**

- Visible on 59.3% of graded frames; 150 missing runs (85 per graded hour).
- Run lengths p10 12, median 268, p90 845, p99 2,619 frames. 85% of missing frames sit in runs of
  10 s or more.
- **The ball vanishes when it is slow, near a line and near players, not when it flies.** At the
  last frame before a gap, the ball's speed has a median of 2.0 m/s (5.1 m/s over all visible
  frames). Onset hazard ratios against visible frames:
  - ball under 2 m/s: 2.35; 10-15 m/s: 0.14; 15-20 m/s: 0.24.
  - within 3 m of a touchline or goal line: 2.53; 15 m or more from every line: 0.55.
  - more than 45 m from halfway (the last 7.5 m before a goal line): 3.18.
  - 2 or more players within 3 m: 1.5-2.7; nearest player 8 m or more away: 1.5-2.6.

  The Phase A guess ("balls in the air", fast ball x3) was the wrong way round. Slow balls near the
  lines are what a ball going out of play looks like.
- **Reappearance:** median 10.2 m from where it vanished (15.1 m for gaps of 20 s or more), so a
  long gap usually ends near where it began.
- Visible off the pitch on 2.7% of visible frames. Position jitter (residual from a 9-frame mean)
  is 0.047 m: the ATD is smoothed.

**Stoppages, through Smart Tagging.** 59 automatic SET PIECES tags: 26 throw-ins, 12 corners, 10
goal kicks, 6 kick-offs, 5 free kicks. Each is a 35 s window (5 s for kick-offs). We averaged ball
visibility and players' 80th-percentile speed in 1 s bins around each tag's start, over all 59
(one curve, no single tag inspected):

| seconds from tag start | -30 to -20 | -20 to -10 | -10 to -5 | -5 to 0 | 0 to 3 | 3 to 6 | 6 to 10 | 10 to 20 | 20 to 40 |
|---|---|---|---|---|---|---|---|---|---|
| ball visible | 0.56 | 0.43 | 0.36 | 0.27 | 0.35 | 0.50 | 0.77 | 0.79 | 0.73 |
| player p80 speed (m/s) | 2.31 | 2.27 | 2.12 | 1.72 | 1.57 | 1.79 | 2.59 | 3.22 | 2.71 |

Play is stopped around the tag start (players slowest at 0-3 s) and resumes 3-6 s after it.
During the stoppage the ball is visible about 30% of the time. Once play resumes it is visible
75-80% of the time. This gives the degrader's dead-ball visibility, 0.30, and implies that live
play loses the ball about 25% of the time. Inside the 35 s windows (mostly play after the
restart) 32% of frames miss the ball, against 44% outside them; 22% of all missing frames fall
inside tagged windows.

**Players.**

- 20.5 objects per graded frame, median 22, 5th percentile 14. 75% of frames have 21-22 objects
  and 4.4% have fewer than 14, in runs of median 67 frames: a mass dropout, not a steady trickle.
- Tracklets are long (median 888 frames, 11.5 per minute) and rarely re-acquired: 20% of
  tracklet ends are followed within 3 s by a new tracklet of the same team within 3 m. A slot
  rests a median 188 frames before re-use.
- Two same-team slots within 0.5 m on 3.5% of frames. More than 11 objects for one team on 12.9%
  of frames (not modelled: referees or duplicates, found-not-fixed 3).
- Jitter 0.007 m and a median step of 0.054 m/frame (1.35 m/s): smooth tracking.

## Task 2: the calibrated degrader (`atd_cal`)

New options in `degrade.DegradeConfig`. Every old variant is untouched: its random draws and
outputs are unchanged, and its cache keys are kept.

| option | what | atd_cal | from |
|---|---|---|---|
| `ball_mode="joint"` | one gap process over all graded time. A gap starts at a visible frame and takes its length from the real gap lengths (`ball_burst_lengths="atd"`, resolved at run time from the ATD, cached in runs/). It can run across stoppages and restarts. Two base rates are solved: dead onsets so 70% of dead frames are hidden, and live onsets for 40.7% hidden overall | yes | gap lengths, visibility |
| `dead_ball_visible` | dead frames with a visible ball | 0.30 | set-piece profile |
| `restart_reacq_prob`, `restart_reacq_median` | a gap that runs into a restart ends a lognormal delay (median 3 s) after it, with this probability | 0.2, 75 frames | visibility at 3-10 s after tag start |
| `min_gap_frames` | a gap cut shorter than this by a restart is dropped | 25 | no short gaps |
| live onset weights | ball under 2 m/s x1, faster than 10 m/s x0.2, 2+ players within 3 m x3, within 3 m of a line x2, over 45 m from halfway x2 | | onset hazard ratios |
| `dead_out_weight`, `dead_onset_head_*` | dead onsets x30 with the ball off the pitch, x3 in a stoppage's first 2 s | 30, 3 | ball rarely seen off the pitch |
| `smooth_frames`, `noise_smooth_frames` | 9-frame mean of the clean input, then AR(1) error smoothed over 25 frames: a slow drift of ball 0.6 m, players 0.25 m, almost no jitter | 9, 25 | jitter, nearest-player distances |
| `mass_drop_*` | 0.7 per minute (x4 in dead time), a lognormal length (median 67 frames), 50-90% of tracked players removed together | | objects p5, low-count runs |
| `slot_policy="lowest"` | a new tracklet takes the lowest free slot | | slot rest times |
| tracklets | lost-track gaps median 250 frames, no 1-3 frame re-acquisition, objects target 20.0 | | tracklet lengths, re-acquisition |

The live onset weights started from the measured hazard ratios and were adjusted over four
simulate-and-compare rounds on sample game 1 (`runs/phaseB2/`, calibration only, no model
scores looked at), then checked on all three games.

### Real against degraded (sample games 1-3 x 2 seeds)

| statistic | real ATD | Phase A default | atd_cal |
|---|---|---|---|
| ball visible share (graded) | 0.593 | 0.590 | 0.594 |
| missing runs per hour | 85 | 487 | 99 |
| missing run p10 / p50 / p90 (frames) | 12 / 268 / 845 | 4 / 22 / 176 | 49 / 256 / 803 |
| missing run p99 / mean | 2619 / 431 | 991 / 77 | 1458 / 373 |
| missing frames in runs >= 25 | 0.999 | 0.920 | 0.998 |
| missing frames in runs >= 250 | 0.852 | 0.576 | 0.823 |
| visible run p10 / p50 / p90 | 54 / 345 / 1542 | 10 / 77 / 252 | 28 / 271 / 1427 |
| onset ball speed p25 / p50 / p75 (m/s) | 0.8 / 2.0 / 5.0 | 2.2 / 5.7 / 12.0 | 0.5 / 1.1 / 2.9 |
| visible ball speed p25 / p50 / p75 | 2.4 / 5.1 / 9.5 | 1.9 / 4.6 / 10.0 | 1.5 / 4.3 / 9.4 |
| onset nearest player p25 / p50 / p75 (m) | 0.6 / 1.5 / 3.3 | 0.4 / 1.7 / 3.9 | 1.2 / 2.7 / 5.6 |
| visible nearest player p25 / p50 / p75 | 0.9 / 1.9 / 4.1 | 0.4 / 1.2 / 3.5 | 0.8 / 1.6 / 3.8 |
| onset players within 3 m, mean | 1.05 | 0.93 | 0.81 |
| visible players within 3 m, mean | 0.84 | 0.92 | 0.89 |
| onset distance to line p25 / p50 | 2.6 / 6.5 | 3.8 / 11.2 | 2.2 / 5.4 |
| visible distance to line p25 / p50 | 6.3 / 13.0 | 5.2 / 12.2 | 5.3 / 12.2 |
| onset within 3 m of a line | 0.265 | 0.220 | 0.350 |
| reappear displacement p25 / p50 / p75 (m) | 4.7 / 10.2 / 22.5 | 2.2 / 5.2 / 10.5 | 2.4 / 12.0 / 30.2 |
|   gaps >= 500 frames: p50 | 15.1 | 9.3 | 33.9 |
| ball visible outside the pitch | 0.0268 | 0.0359 | 0.0563 |
| ball jumps > 40 m/s per min | 2.04 | 1.39 | 0.68 |
| ball jitter (m, 9-frame residual) | 0.047 | 0.159 | 0.084 |
| objects per frame mean / p5 / p50 / p95 | 20.5 / 14.0 / 22.0 / 22.0 | 20.5 / 18.0 / 21.0 / 22.0 | 20.2 / 12.0 / 21.0 / 22.0 |
| frames over 22 objects | 0.0236 | 0.0072 | 0.0171 |
| tracklets per minute | 11.5 | 80.6 | 16.5 |
| tracklet length p25 / p50 / p75 / p90 | 198 / 888 / 3633 / 7116 | 103 / 260 / 532 / 889 | 260 / 1070 / 2646 / 4854 |
| slot gap p25 / p50 / p75 | 90 / 188 / 646 | 112 / 316 / 667 | 45 / 105 / 272 |
| re-acquired within 3 s, 3 m | 0.198 | 0.786 | 0.297 |
| re-acquisition delay p25 / p50 / p75 | 24 / 47 / 59 | 2 / 8 / 21 | 14 / 28 / 47 |
| near-duplicate frames (same team < 0.5 m) | 0.0347 | 0.0571 | 0.0524 |
| player teleports > 1 m/frame per min | 2.83 | 3.41 | 2.55 |
| player jitter (m) | 0.007 | 0.060 | 0.011 |
| player step p50 / p95 (m/frame) | 0.054 / 0.188 | 0.104 / 0.235 | 0.058 / 0.181 |

**Calibration gate:** `atd_profile.check_calibration` (12 rules: visible share; gap p50 and p90;
missing frames in runs of 25+ and 250+; visible-run median; onset line distance; ball and player
jitter; objects mean and p5; tracklet median). `atd_cal` passes. The Phase A degrader fails 10
rules. `tests/test_b2.py` shows the gate failing on a degrader with Phase A's bursts and passing
another seed of the reference.

What still does not match (found-not-fixed 1):

- the ball vanishes 2.7 m from the nearest player in the degrader, 1.5 m in the real file;
- long gaps end 34 m from where they began in the degrader, 15 m in the real file;
- the ball is visible off the pitch 2x too often;
- 16.5 tracklets per minute against 11.5; slots are re-used sooner (105 frames against 188).

## Task 3: models on the calibrated bench

### Main table (`atd_cal`, pooled games 1-3, mean of 3 degrader seeds, bootstrap 5-95 on seed 0)

| model | touch spellF1 [boot 5-95] (seeds min-max) | touch macroF1 | controlled spellF1 [boot] | controlled macroF1 | pred poss (touch GT 1146, controlled 697) | dead F1 | touch spellF1 game 1 / 2 / 3 |
|---|---|---|---|---|---|---|---|
| smooth50 | 0.262 [0.244, 0.277] (0.258-0.269) | 0.595 | 0.270 [0.253, 0.286] | 0.598 | 1465 | 0.548 | 0.281 / 0.263 / 0.244 |
| seg_v4 | 0.357 [0.334, 0.378] (0.355-0.360) | 0.776 | 0.440 [0.415, 0.462] | 0.785 | 806 | 0.861 | 0.366 / 0.344 / 0.360 |
| gbm_v4 | 0.499 [0.471, 0.525] (0.496-0.501) | 0.863 | 0.604 [0.581, 0.630] | 0.873 | 766 | 0.911 | 0.528 / 0.529 / 0.444 |
| gbm_v5a | 0.543 [0.512, 0.565] (0.537-0.551) | 0.896 | 0.693 [0.676, 0.725] | 0.908 | 697 | 0.943 | 0.562 / 0.549 / 0.520 |
| gbm_v5_nop | 0.544 [0.505, 0.559] (0.530-0.554) | 0.897 | 0.702 [0.676, 0.727] | 0.910 | 686 | 0.946 | 0.562 / 0.552 / 0.518 |
| gbm_v5 | 0.587 [0.558, 0.611] (0.582-0.593) | 0.909 | 0.739 [0.719, 0.762] | 0.920 | 719 | 0.950 | 0.599 / 0.610 / 0.556 |
| gbm_v5_nogap | 0.588 [0.558, 0.613] (0.582-0.595) | 0.908 | 0.734 [0.715, 0.763] | 0.918 | 723 | 0.947 | 0.599 / 0.601 / 0.565 |

How to read it. smooth50, seg_v4 and gbm_v4 were run as they are. gbm_v4 loses 0.161 touch
spellF1 against its Phase C bench, and seg_v4 loses 0.125. Two steps win it back, each just
above the 0.03-0.04 noise band in touch and well above it in controlled:

1. **Retraining on the calibrated degrader** (gbm_v5a): +0.044 touch, +0.089 controlled. The model
   stops learning that a long gap means a stoppage.
2. **Player-only team evidence** (gbm_v5 against gbm_v5_nop, which differs only in those
   features): +0.043 touch, +0.037 controlled. This is the long-gap model the task asked for.
   Inside a gap, the team that is advancing, running forwards, stretched long and wide, and less
   pressed is the team in possession.

**The ball-at-both-ends features and the ball-visibility features add nothing.** gbm_v5_nogap
has neither and scores 0.588 / 0.734 against gbm_v5's 0.587 / 0.739. The last-seen and next-seen
touch features from Phase C (`tt_last`, `tt_next`, `last_ddiff`, `next_ddiff`, `last_bx_att`,
`next_bx_att`) were already in every model. They carry what the gap ends know, and the new
gap-end columns duplicate them.

Decode settings: the alpha x H<->A cost grid is flat in touch spellF1 for both gbm_v5a
(0.517-0.546) and gbm_v5 (0.564-0.591). Phase C's alpha 0.5 and cost 8 stay.

### Robustness sweep around the calibrated settings

Each column changes one setting of `atd_cal`. gbm_v5 / v5a / v5_nogap trained on vis15, vis50,
noisy, noctx and restart60 (other seeds). They never saw miss35, miss47 (ball missing 35% / 47%)
or nomass (no mass dropouts).

#### pooled spellF1, GT touch, mean over 3 seeds

| model | atd_cal | cal_vis15 | cal_vis50 | cal_miss35 | cal_miss47 | cal_noisy | cal_noctx | cal_restart60 | cal_nomass | min | max |
|---|---|---|---|---|---|---|---|---|---|---|---|
| smooth50 | 0.262 | 0.265 | 0.225 | 0.272 | 0.236 | 0.240 | 0.263 | 0.261 | 0.244 | 0.225 | 0.272 |
| seg_v4 | 0.357 | 0.358 | 0.348 | 0.366 | 0.343 | 0.284 | 0.360 | 0.369 | 0.354 | 0.284 | 0.369 |
| gbm_v4 | 0.499 | 0.500 | 0.495 | 0.508 | 0.484 | 0.433 | 0.499 | 0.511 | 0.505 | 0.433 | 0.511 |
| gbm_v5a | 0.543 | 0.537 | 0.534 | 0.549 | 0.529 | 0.515 | 0.543 | 0.551 | 0.539 | 0.515 | 0.551 |
| gbm_v5 | 0.587 | 0.582 | 0.583 | 0.591 | 0.579 | 0.573 | 0.588 | 0.592 | 0.591 | 0.573 | 0.592 |
| gbm_v5_nogap | 0.588 | 0.582 | 0.586 | 0.591 | 0.581 | 0.567 | 0.588 | 0.595 | 0.589 | 0.567 | 0.595 |

#### pooled spellF1, GT controlled

| model | atd_cal | cal_vis15 | cal_vis50 | cal_miss35 | cal_miss47 | cal_noisy | cal_noctx | cal_restart60 | cal_nomass | min | max |
|---|---|---|---|---|---|---|---|---|---|---|---|
| smooth50 | 0.270 | 0.269 | 0.227 | 0.281 | 0.240 | 0.242 | 0.271 | 0.265 | 0.248 | 0.227 | 0.281 |
| seg_v4 | 0.440 | 0.450 | 0.434 | 0.451 | 0.431 | 0.361 | 0.446 | 0.462 | 0.447 | 0.361 | 0.462 |
| gbm_v4 | 0.604 | 0.606 | 0.602 | 0.614 | 0.589 | 0.512 | 0.605 | 0.626 | 0.612 | 0.512 | 0.626 |
| gbm_v5a | 0.693 | 0.693 | 0.681 | 0.702 | 0.676 | 0.662 | 0.694 | 0.704 | 0.692 | 0.662 | 0.704 |
| gbm_v5 | 0.739 | 0.731 | 0.739 | 0.744 | 0.733 | 0.709 | 0.740 | 0.743 | 0.742 | 0.709 | 0.744 |
| gbm_v5_nogap | 0.734 | 0.727 | 0.732 | 0.738 | 0.726 | 0.705 | 0.734 | 0.743 | 0.732 | 0.705 | 0.743 |

#### pooled macroF1, GT touch

| model | atd_cal | cal_vis15 | cal_vis50 | cal_miss35 | cal_miss47 | cal_noisy | cal_noctx | cal_restart60 | cal_nomass | min | max |
|---|---|---|---|---|---|---|---|---|---|---|---|
| smooth50 | 0.595 | 0.635 | 0.515 | 0.608 | 0.564 | 0.584 | 0.594 | 0.602 | 0.583 | 0.515 | 0.635 |
| seg_v4 | 0.776 | 0.777 | 0.764 | 0.783 | 0.762 | 0.741 | 0.776 | 0.781 | 0.776 | 0.741 | 0.783 |
| gbm_v4 | 0.863 | 0.860 | 0.858 | 0.867 | 0.852 | 0.835 | 0.863 | 0.870 | 0.864 | 0.835 | 0.870 |
| gbm_v5a | 0.896 | 0.895 | 0.890 | 0.898 | 0.890 | 0.886 | 0.895 | 0.898 | 0.894 | 0.886 | 0.898 |
| gbm_v5 | 0.909 | 0.908 | 0.905 | 0.911 | 0.905 | 0.903 | 0.909 | 0.911 | 0.909 | 0.903 | 0.911 |
| gbm_v5_nogap | 0.908 | 0.906 | 0.904 | 0.909 | 0.904 | 0.902 | 0.907 | 0.909 | 0.907 | 0.902 | 0.909 |

#### dead F1 (alive / dead, touch GT)

| model | atd_cal | cal_vis15 | cal_vis50 | cal_miss35 | cal_miss47 | cal_noisy | cal_noctx | cal_restart60 | cal_nomass | min | max |
|---|---|---|---|---|---|---|---|---|---|---|---|
| smooth50 | 0.548 | 0.630 | 0.424 | 0.561 | 0.530 | 0.548 | 0.549 | 0.555 | 0.548 | 0.424 | 0.630 |
| seg_v4 | 0.861 | 0.860 | 0.858 | 0.861 | 0.858 | 0.860 | 0.859 | 0.863 | 0.862 | 0.858 | 0.863 |
| gbm_v4 | 0.911 | 0.910 | 0.912 | 0.914 | 0.903 | 0.904 | 0.911 | 0.923 | 0.915 | 0.903 | 0.923 |
| gbm_v5a | 0.943 | 0.942 | 0.942 | 0.945 | 0.939 | 0.941 | 0.943 | 0.946 | 0.943 | 0.939 | 0.946 |
| gbm_v5 | 0.950 | 0.948 | 0.947 | 0.951 | 0.947 | 0.948 | 0.949 | 0.952 | 0.950 | 0.947 | 0.952 |
| gbm_v5_nogap | 0.947 | 0.946 | 0.945 | 0.948 | 0.944 | 0.945 | 0.946 | 0.949 | 0.947 | 0.944 | 0.949 |

Reading the sweep:

- **gbm_v5 and gbm_v5_nogap lead gbm_v4 in every column**, by 0.083-0.134 touch spellF1 and
  0.117-0.197 controlled. The bootstrap intervals do not overlap on `atd_cal`, and the worst
  gbm_v5_nogap seed (0.582) beats the best gbm_v4 seed (0.501).
- **They tie each other everywhere:** touch differences of -0.006 to +0.003, controlled 0.000 to
  0.007 in gbm_v5's favour. Both are inside the noise band.
- **The lead holds on the held-out settings:** ball missing 35% and 47%, and no mass dropouts.
- **Heavy position noise costs the most again:** -0.021 for gbm_v5_nogap and -0.066 for gbm_v4.
  gbm_v4 was never trained on smooth drift.

## Task 4: label-free diagnostics on the real match

`PYTHONPATH=src .venv/bin/python -m mtah.diagnostics --models smooth50 seg_v4 gbm_v4 gbm_v5a gbm_v5
gbm_v5_nogap` runs every model on the competition ATD, with the fixed attack direction. These are
sanity checks, not scores. The bench GT columns are the sample games' truths.

| statistic | bench GT touch | bench GT controlled | smooth50 | seg_v4 | gbm_v4 | gbm_v5a | gbm_v5 | gbm_v5_nogap |
|---|---|---|---|---|---|---|---|---|
| D share (graded) | 0.389 | 0.389 | 0.323 | 0.391 | 0.454 | 0.439 | 0.430 | 0.431 |
| D share, ball missing | - | - | 0.795 | 0.781 | 0.870 | 0.843 | 0.834 | 0.830 |
| D share, ball visible | - | - | 0.000 | 0.124 | 0.168 | 0.161 | 0.153 | 0.157 |
| stoppages per hour | 63 | 63 | 75 | 70 | 70 | 67 | 64 | 66 |
| stoppage median (s) | 16.9 | 16.9 | 7.6 | 12.6 | 14.7 | 16.2 | 18.6 | 17.8 |
| possessions per hour | 243 | 148 | 337 | 185 | 159 | 142 | 146 | 147 |
| possession p25 / p50 / p75 (s) | 1.8 / 4.4 / 12.5 | 4.8 / 10.3 / 19.9 | 1.4 / 4.6 / 9.2 | 3.6 / 7.9 / 16.8 | 3.0 / 6.4 / 18.5 | 3.5 / 8.1 / 22.2 | 3.6 / 7.7 / 20.6 | 3.7 / 7.5 / 21.0 |
| possessions under 3 s | 0.37 | 0.11 | 0.39 | 0.21 | 0.25 | 0.19 | 0.19 | 0.18 |
| home share of H + A | 0.516 | 0.521 | 0.570 | 0.656 | 0.623 | 0.629 | 0.615 | 0.615 |
| agreement with Smart Tagging attacking-phase team | - | - | 0.654 | 0.692 | 0.763 | 0.774 | 0.780 | 0.779 |

What the diagnostics say:

- **Possession structure transfers.** The learned models predict 146-159 possessions per hour,
  with a median of 6.4-8.1 s. Their bench predictions have 697-785 against the touch GT's 1,146 and
  the controlled GT's 697. So they sit at the same place relative to the truths on the real match
  as on the bench: near the controlled GT's count, merging the touch GT's short possessions
  (Phase C found-not-fixed 2, still open).
- **H / A balance: Barcelona 61-63% of H + A for every learned model.** That is plausible for this
  fixture. seg_v4 (66%) and smooth50 (57%) bracket it. Nothing is symmetric-broken: the models are
  exactly team-symmetric by construction.
- **Stoppages: 64-70 per hour with a median of 15-19 s**, against 63 per hour and 16.9 s in the
  sample games' truth.
- **Agreement with Smart Tagging's attacking-phase team: 0.76-0.78 for the learned models**,
  0.69 for seg_v4 and 0.65 for smooth50. This is how the attack-direction bug was found (below).
- **D share: 43-45% of graded frames, 83-87% of ball-missing frames, 15-17% of visible ones.**
  Phase B read gbm_v4's 89% in gaps as a symptom. The bench's own numbers say most of it is not:
  - On `atd_cal`, 55-69% of gap frames are truly dead (per game), with 39% dead overall.
  - The players' speed is an independent, label-free view (`diagnostics.speed_mixture`). On the
    bench, the players' median speed is under 1 m/s on 55% of dead gap frames and 9% of live gap
    frames. On the real ATD it is under 1 m/s on 51% of gap frames. Solving the mixture gives
    **91% dead inside the real gaps**. The estimate is 100% at a 0.7 m/s threshold and 94% at
    1.5 m/s.
  - The estimate leans on the sample games' players moving like BAR-ATH's. The real match's
    visible-ball frames are a little slower than the bench's live frames (11% against 9% under
    1 m/s, with some dead frames in there). If real live play were as slow as 15% or 25% under
    1 m/s, the estimate would drop to 90% or 86%.
  - So 83-87% D inside gaps is consistent with the players. The video pilot's reading of the same
    51% ("gaps are only about half stoppages") assumes every dead frame has slow players, and on
    the bench only about half of them do.
  - The overall 43% D is above the sample games' 39%. LaLiga's ball-in-play time is usually about
    55% of the clock, so it may be right. The public window's 20% covers 5 minutes and is no
    guide. No constant was tuned to any of this.

### The attack-direction bug (found by the diagnostics, fixed)

Before the fix, `gbm_v5`'s Smart Tagging agreement was 0.775 in the first half and 0.628 in the
second, and gbm_v4's home share was 0.651 / 0.596. The second half's first 10 s are not a kick-off
layout on the real ATD (15 objects, home centroid 51 m, away 78 m), so every attack-relative
feature had home attacking the wrong way after half time. `features.attack_sign` now takes the
median over the whole half of (away min x - home min x) + (away max x - home max x). It agrees
with the old rule on every half of sample games 1-3 (0.96-1.00 of frames on one side), so bench
numbers and trained models are unchanged. Second-half agreement after the fix: gbm_v4 0.766,
gbm_v5a 0.778, gbm_v5 0.785. This was the one pathology the diagnostics showed, and the fix is a
general rule with nothing tied to BAR-ATH's frames.

## Current best and its exact command

**`gbm_v5_nogap`**: learned emissions (gbm4's features plus the B2 player-only team features, no
ball-visibility or gap features), trained on the `atd_cal` mixture, decoded by the three-state
Viterbi (alpha 0.5, H<->A cost 8, alive<->D cost 12). On `atd_cal` it scores 0.588 [0.558, 0.613]
touch / 0.734 [0.715, 0.763] controlled spellF1, macroF1 0.908. gbm_v5 ties it in every sweep
column (largest gap 0.006 touch). We keep the one that assumes least, as in Phase C. It does not
read where the ball is hidden, which is the part of the degrader with the known residual
mismatches (onset distance, reappearance distance). It also will not need retraining for a
meaning change when video fills part of the gaps.

```bash
# train once (the B2 training cache is shared with gbm5: about 20-50 min on 6 cores, depending on load)
PYTHONPATH=src .venv/bin/python -m mtah.models.learned --gt touch --tag gbm5_nogap --workers 6

# labels for the competition ATD; runs the organiser's validate_submission
PYTHONPATH=src .venv/bin/python -m mtah.predict --model gbm_v5_nogap --competition \
    --out runs/submission/MurrayInsights.csv
```

- The file has 199,319 rows: H 55,616 / A 34,817 / D 68,424 / X 40,462 (X outside graded time).
  `validate_submission: OK`.
- A second run in a separate process gave a byte-identical file (SHA-256 `2a7c130b...`).
- The degrader needs the real gap lengths, which `atd_profile.real_gap_lengths()` reads from the
  ATD on first use and caches in `runs/cache/atd_gap_lengths.npy`. So training needs the bundle in
  `data/`, which the competition prediction needs anyway.

Bench: `PYTHONPATH=src .venv/bin/python -m mtah.modelbench --models smooth50 seg_v4 gbm_v4 gbm_v5a
gbm_v5 gbm_v5_nogap --robust-b2 --seeds 3 --bootstrap 300 --workers 6`. The sweep was run variant
by variant (`runs/phaseB2/sweep*.sh`) to prune the 84 MB-per-table feature cache. Tables:
`PYTHONPATH=src .venv/bin/python -m mtah.report --main atd_cal runs/phaseC/sweep_*.json
runs/phaseC/sweep_nogap_*.json`.

## How a video-ball variant would plug in (the pilot's track is not here yet)

`mtah.video.source.with_video_ball(table)` will fill the ATD's gaps on about 39% of gap frames and
flag each frame's source (atd / video / none). It will never track fast balls (over 15 m/s); it
sees still balls at restarts on 21-31% of decoded gap frames and spare balls near the lines.

1. **Features: a parallel channel, not a merged ball.** Keep every ATD-ball feature as is. Add
   the same touch / nearest-team / speed / still features computed from the video ball (`vtt_*`,
   `vball_still`, `vball_offpitch`), plus the source flag and time since the last video sighting.
   A still video ball near a line is weak D evidence (spare balls). A video ball at a player's feet
   is team evidence. The video ball is not a ground position for lofted balls, so its touch radius
   should not be the ATD's 1 m.
2. **Degrader: simulate the partial, flagged fill.** Inside the hidden frames of `atd_cal`,
   reveal the clean ball as "video":
   - in bursts, on a share drawn around 39%;
   - never when the clean ball moves faster than 15 m/s;
   - more often when it is still (dead-time restarts);
   - with extra slow-drift error and a height-like offset for fast or lofted balls;
   - plus false positives: a still ball off the pitch near a line during stoppages.

   Calibrate the fill share, burst lengths and noise against the video track's own distributions
   with `atd_profile` (one more column in the calibration table and gate).
3. **Retrain (gbm6) and re-bench** with the fill share swept (0, 0.2, 0.39, 0.6), so a wrong fill
   rate is visible as a robustness column, not a hidden assumption.

## Public window: the one-time pipeline check

Run once, after the choice above was made and the file written, with the organiser's scorer
(`cd ref/evaluation && python run_evaluation.py --results <dir with the file> --gt public`).
**gbm_v5_nogap on the public 5 minutes: spellF1 0.619, macroF1 0.898** (14 possessions against
17; alive/dead F1 0.971; home share 0.580 against 0.573). For reference, the organiser's smooth50
scores 0.326 there. This confirms the pipeline is aligned (frames, teams, dead ball) and nothing
more: it is 17 possessions, and EVALUATION.md shows public and full-match rankings can disagree.
Nothing was changed after it.

## Found, not fixed

1. **Calibration residuals.** The degraded ball vanishes further from players (2.7 m median
   against 1.5 m). Long gaps end far from where they began (34 m against 15 m). The ball is
   visible off the pitch twice as often. Tracklets are shorter (16.5 per minute against 11.5).
   Slots are re-used sooner. All are visible in the calibration table. The gap features that
   would be most exposed to the first two are not in the chosen model.
2. **The dead-share prior comes from the sample games (39%).** The real share is unknown. The
   speed mixture says gaps are about 90% dead on the real match. That is consistent with the
   models, but it rests on the sample games' players moving like BAR-ATH's.
3. **More than 11 objects for one team on 12.9% of real frames** (referees tracked as a team?
   duplicates?). Not in the degrader. A referee next to the ball would bias nearest-team touches.
   Worth one look at where those extra objects sit relative to the ball.
4. **Possession count below the touch GT** (697-785 predicted against 1,146 on the bench; 146-159
   per hour on the real match). The models merge short possessions. Phase C found-not-fixed 2,
   still open.
5. **Smart Tagging's attacking phases agree with the learned models on 76-78% of tagged frames.**
   They are an allowed automatic input and could be team evidence. They are untested as evidence
   because the sample games have no tags.
6. **`FEATURE_VERSION` was not bumped for the attack-sign change.** That is safe because the sign
   is identical on every sample table, so cached bench features and trained models stay valid.
   Any feature cache built from a non-sample table under the old rule would be stale (none exists
   in runs/).
7. **The previous `runs/submission/MurrayInsights.csv` (gbm_v4) predates the attack-sign fix.**
   It has been overwritten by the command above. A copy named
   `ref/evaluation/gt/MurrayInsights_gbm_v4.csv` sits inside the organiser clone's gt folder
   (gitignored, not ours; created before this phase). It should be moved out so nobody scores it
   as ground truth.
8. **The live onset weights were tuned on sample game 1** over four simulate-and-compare rounds
   (calibration of the degrader, never a model score), then checked on all three games.
9. **Training is slow under load:** 20-50 min per tag with the video job running. The training
   cache stores all columns, so ablations (gbm5_nop, gbm5_nogap) need only the fits.

## Tests

`tests/test_b2.py` (16). Each gate is fed the case it exists to catch:

- the calibration gate fails a Phase A-style degrader against a long-gap reference and passes
  another seed of the reference;
- the attack direction survives a half that opens in a scrambled layout (the old rule fails it),
  and equals the old rule on all sample-game halves;
- joint gaps hit the missing target and the dead visibility, keep their drawn length, and run
  across stoppages;
- the restart cap stops gaps a set delay after the restart (with probability 1, never past it;
  with 0, some run past);
- mass dropouts create low-count frames;
- smoothed error keeps its size with under a fifth of the jitter;
- "lowest" slot recycling shortens slot rests;
- `make_config` rejects unknown length sources, and `ball_mode` rejects unknown modes;
- Phase C variant names keep their caches;
- the profile counts gaps exactly;
- the B2 player features follow who advances and flip with the teams;
- the gap features read both ends uncapped;
- the diagnostics count possessions with X forward-filled, D shares and tag agreement exactly.

Full suite: `.venv/bin/python -m pytest` 135 passed (including the video pilot's tests, which this phase did not touch).

