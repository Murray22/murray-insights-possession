# Phase E: a ball track shaped like the real one, and the video ball as an input

Date: 2026-11-05. Inputs: the competition ATD, the saved full-run video ball track
(`runs/video/full/ball_track_soccana_ov.npz`) and Metrica sample games 1-3. Only distributions of
the real ATD and of the video track were measured. No label of BAR-ATH was read, and the
full-match ground truth is not ours. Nothing is tied to a BAR-ATH frame. Bench protocol as in
Phases B2 and D: sample games 1-3, leave one game out, degrader seeds 0-2 for scoring and 10000+
for training, both GTs. Noise band 0.03-0.04 touch spellF1 pooled.

## Summary

- **The gate now covers the ball track's shape, and atd_cal fails it.** Eight kink rules were
  added to `atd_profile.check_calibration`: kink 4 / 8 percentiles, kink at touches, strong touches
  per minute, and slow-ball speed. atd_cal fails all eight. The real ball wobbles between touches
  (kink4 p50 0.91 m/s against 0.37) and bends far less at them (p99 5.9 against 10.2).
- **atd_cal2 fits it.** It adds a further zero-phase smoothing of the shown ball (13-frame mean,
  twice) and a 0.25 m mid-frequency wobble. There is no lag: the frame-exact video puts the ATD at
  0.00 frames. It passes 7 of 8 shape rules; kink8 p50 is 30.2% low against a 30% tolerance.
- **The kink features stop helping once the inputs match.**
  - gbm_v7's +0.018 touch lead on atd_cal is real by a paired bootstrap ([+0.010, +0.027]).
  - On atd_cal2 it is +0.000 (15 of 27 cells).
  - Retrained on atd_cal2 (gbm_v7_c2), it is +0.005 over gbm_v5_nogap_c2: 22 of 27 cells, paired
    [-0.000, +0.010].
  - The shipped gbm_v5_nogap drops from 0.588 to 0.581 on atd_cal2, and retraining does not beat
    it.
- **The real video track, label-free:**
  - fills 40.5% of gap frames, in bursts (median 9 frames, about 5 per gap);
  - the filled ball is mostly still (66.5% under 0.5 m/s, 1.3% over 15 m/s);
  - error against the ATD is median 1.05 m on window frames, growing with ball speed and pointing
    away from the camera (lofted balls);
  - 4.3% of gap frames look like spare balls.
- **The simulation, `atd_cal2v`:**
  - reveals the true ball in gaps in real-length bursts, weighted to still balls, never above
    15 m/s;
  - gives each burst a real error vector of its speed bin;
  - adds false spare balls by the lines;
  - matches the fill share, burst lengths, spare share and off-pitch share. It fails one video
    gate rule: its fills are less still (0.56 against 0.67) and nearer players, because the bench's
    gaps hold fewer dead balls than the real ones.
- **Video features as a separate channel help, and gbm_v8n meets the ship rule.**
  - On the calibrated sweep it leads the shipped model by +0.021 touch in 27 of 27 cells (paired
    [+0.014, +0.029]) and by +0.030 controlled.
  - With no video it ties the shipped model (+0.001). With half the video it still leads (+0.015).
  - The kink features add nothing on top of it (gbm_v8: -0.001 against gbm_v8n).
  - On the real match its possession count, lengths, D share and home share are plausible and
    close to the shipped file's. It changes 3.3% of graded frames, as on the bench (2.7-3.4%).
- **Shipped as a new file, not over the old one:** `runs/submission/MurrayInsights_gbm_v8n.csv`
  (SHA-256 `899be8ee...`). It is byte-identical on a rerun and `validate_submission: OK`.
  `MurrayInsights.csv` (gbm_v5_nogap, `2a7c130b...`) is untouched.
- **Worries, in order:**
  1. The lead is below the 0.03-0.04 band. It passes on consistency plus the paired bootstrap,
     and both share three sample games.
  2. The real video is weaker team evidence than the simulated video (about 20% fewer touches,
     further from players), so the real gain is probably smaller.
  3. The file now depends on our own detector run. Whether that run reproduces byte for byte from
     the video has not been tested, and its environment is not pinned in the repo.


## Part 1: the ball track's shape

### 1.1 What was missing from the gate

`atd_profile.ball_shape_profile` adds, for the visible ball on graded frames:

- `kink_k = |v(t, t + k) - v(t - k, t)|` for k = 2, 4, 8 (`features_d.kink`), percentiles;
- kink4 on touch frames (ball within 1.5 m of a player);
- strong touches per minute: runs of touch frames with kink4 >= 2 m/s. This is the Phase D
  addendum's count; it reproduces its 9.4 (real) and 13.5-16.4 (atd_cal g1-g3).

Eight rules joined `CAL_RULES` as `CAL_RULES_SHAPE`, each within 30% relative: kink4 p25 / p50 /
p90 / p99, kink8 p50, kink4 on touches p90, strong touches per minute, visible ball speed p25.
The Phase B2 rules are kept as `CAL_RULES_B2`. `atd_cal` passes the B2 rules and fails all eight
new ones. Tests (`tests/test_phaseE.py`):

- `test_shape_gate_fails_unshaped_ball_and_passes_another_seed`: on synthetic tracking, the gate
  fails atd_cal's ball error (kink4 p50 and p99 among the failures) against a shaped reference,
  and passes another seed of the reference.
- `test_real_profile_fails_atd_cal_on_shape`: against the real profile, `atd_cal` fails the kink4
  rules and `atd_cal2` passes every kink4 rule and fails fewer rules.

### 1.2 The fit (`mtah.ballfit`)

The real track is smoother at a touch and wobblier between touches. Its kink grows about linearly
with the window (p50 0.45 / 0.91 / 1.76 m/s at k = 2 / 4 / 8). That is a low-frequency wobble,
not white jitter. `atd_cal`'s kink barely grows (0.24 / 0.37 / 0.67), and its upper tail is twice
the real one. Three options were added to `DegradeConfig`. They draw from their own random
stream, and with the defaults `atd_cal`'s tables are byte-identical to the cached ones (checked
on games 1 and 3):

- `ball_post_smooth_frames` / `_passes`: a further centred moving average of the shown ball.
- `ball_wobble_m` / `_frames` / `_passes`: white noise smoothed by the same moving average,
  rescaled to sd `ball_wobble_m` per axis.
- `ball_lag_frames`: a lag of the shown ball. It was not fitted. The video is frame-exact, and the
  lag scan of the ATD against it puts the best shift at 0.00 frames (median distance 1.046 m at 0,
  1.048 m at +0.25, 1.068 m at -0.25). A centred smoother has no lag.

Two grids (70 + 27 configurations, `runs/phaseE/ballfit_grid1.json`, `ballfit.json`) scored
games 1-3 at seed 0. Each was scored by the mean squared log ratio of 22 ball statistics against
the real ones. No model score was looked at. The minimum: a further 13-frame mean applied twice,
and a 0.25 m wobble smoothed over 21 frames twice. That is `atd_cal2` = `atd_cal` + those five
settings (`modelbench.BALL_SHAPE`). Players, gaps and everything else are as in `atd_cal`.

Real against degraded (`python -m mtah.atd_profile --variants atd_cal atd_cal2 --seeds 2 --out
runs/phaseE`, sample games 1-3 x 2 seeds). Ball rows only. Every player row and every gap row is
identical between atd_cal and atd_cal2; the full table is in `runs/phaseE/profile.log`.

| statistic | real ATD | atd_cal | atd_cal2 |
|---|---|---|---|
| ball visible share (graded) | 0.593 | 0.594 | 0.594 |
| missing run p10 / p50 / p90 (frames) | 12 / 268 / 845 | 49 / 256 / 803 | 49 / 256 / 803 |
| onset ball speed p25 / p50 / p75 (m/s) | 0.8 / 2.0 / 5.0 | 0.5 / 1.1 / 2.9 | 0.8 / 1.4 / 3.2 |
| visible ball speed p25 / p50 / p75 | 2.4 / 5.1 / 9.5 | 1.5 / 4.3 / 9.4 | 1.8 / 4.6 / 8.8 |
| ball jitter (m, 9-frame residual) | 0.047 | 0.084 | 0.042 |
| ball jumps > 40 m/s per min | 2.04 | 0.68 | 0.09 |
| kink2 p25 / p50 / p75 / p90 / p99 (m/s) | 0.18 / 0.45 / 0.94 / 1.46 / 2.94 | 0.15 / 0.24 / 0.42 / 2.34 / 5.93 | 0.29 / 0.49 / 0.88 / 1.53 / 2.86 |
| kink4 p25 / p50 / p75 / p90 / p99 (m/s) | 0.35 / 0.91 / 1.87 / 2.89 / 5.92 | 0.22 / 0.37 / 0.93 / 4.30 / 10.18 | 0.43 / 0.75 / 1.63 / 2.93 / 5.50 |
| kink8 p25 / p50 / p75 / p90 / p99 (m/s) | 0.69 / 1.76 / 3.61 / 5.58 / 11.62 | 0.35 / 0.67 / 2.80 / 7.29 / 15.74 | 0.66 / 1.23 / 3.14 / 5.45 / 10.01 |
| kink4 on touch frames p50 / p90 | 0.85 / 2.60 | 0.46 / 4.93 | 0.79 / 2.81 |
| strong touches (kink4 >= 2) per min | 9.4 | 15.2 | 11.7 |
| **shape rules failed (of 8)** | - | **8** | **1** (kink8 p50, 1.23 against 1.76: 30.2% off, tolerance 30%) |

Reading it:

- **atd_cal2 is much closer to the real track on every kink row.** The kink4 median goes from
  0.37 to 0.75 m/s (real 0.91), p99 from 10.2 to 5.5 (real 5.9), and touch-frame p90 from 4.9 to
  2.8 (real 2.6). Strong touches fall from 15.2 to 11.7 per minute (real 9.4).
- **One rule still fails, and I did not widen the gate to pass it.** kink8 p50 is 30.2% low
  against a 30% tolerance. With two parameters the wobble cannot match the median at k = 8
  without overshooting the low end at k = 4: kink4 p25 is already 0.43 against 0.35. The loss
  minimum was kept as fitted.
- **Big ball jumps vanish (0.09 per minute against 2.04).** The extra smoothing removes the
  degrader's few jumps, and the real ones (re-acquisitions after a short loss) are not modelled.
  This is listed in found-not-fixed.


### 1.3 Results on atd_cal2

Commands:

```bash
PYTHONPATH=src .venv/bin/python -m mtah.models.learned --gt touch --tag gbm5n_c2 gbm7_c2 --workers 6
PYTHONPATH=src .venv/bin/python -m mtah.modelbench --models gbm_v5_nogap gbm_v7 smooth50 --variants atd_cal2 \
    --seeds 3 --bootstrap 300 --tag e_asis_main
runs/phaseE/sweep_c2.sh       # gbm_v5_nogap_c2 / gbm_v7_c2 main, then all four on the 8 cal2 settings
PYTHONPATH=src .venv/bin/python -m mtah.paired --a gbm_v7_c2 --b gbm_v5_nogap_c2 --variants atd_cal2 \
    cal2_vis15 cal2_vis50 cal2_miss35 cal2_miss47 cal2_noisy cal2_noctx cal2_restart60 cal2_nomass --seeds 3
```

`gbm5n_c2` and `gbm7_c2` are gbm_v5_nogap's and gbm_v7's designs, retrained on the same
six-setting mixture built on atd_cal2 instead of atd_cal.

**Main table (atd_cal2, mean of 3 seeds, bootstrap 5-95 on seed 0)**

| model | touch spellF1 [boot] (seeds min-max) | touch macroF1 | controlled spellF1 [boot] | pred poss (GT 1146) | dead F1 |
|---|---|---|---|---|---|
| smooth50 | 0.253 [0.235, 0.269] (0.247-0.261) | 0.590 | 0.258 [0.243, 0.274] | 1501 | 0.548 |
| gbm_v5_nogap (as is, shipped) | 0.581 [0.547, 0.603] (0.572-0.593) | 0.905 | 0.728 [0.704, 0.748] | 724 | 0.945 |
| gbm_v7 (as is) | 0.580 [0.550, 0.602] (0.574-0.585) | 0.905 | 0.728 [0.705, 0.749] | 725 | 0.946 |
| gbm_v5_nogap_c2 (retrained) | 0.579 [0.555, 0.608] (0.571-0.586) | 0.904 | 0.733 [0.713, 0.761] | 720 | 0.945 |
| gbm_v7_c2 (retrained) | 0.585 [0.556, 0.608] (0.578-0.596) | 0.906 | 0.736 [0.713, 0.760] | 713 | 0.947 |

**Robustness sweep (pooled spellF1, mean of 3 seeds)**

touch:

| model | atd_cal2 | miss35 | miss47 | noctx | noisy | nomass | restart60 | vis15 | vis50 | min | max |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gbm_v5_nogap | 0.581 | 0.584 | 0.574 | 0.581 | 0.563 | 0.583 | 0.587 | 0.576 | 0.577 | 0.563 | 0.587 |
| gbm_v7 | 0.580 | 0.586 | 0.571 | 0.580 | 0.565 | 0.586 | 0.584 | 0.578 | 0.577 | 0.565 | 0.586 |
| gbm_v5_nogap_c2 | 0.579 | 0.582 | 0.571 | 0.579 | 0.565 | 0.583 | 0.584 | 0.575 | 0.577 | 0.565 | 0.584 |
| gbm_v7_c2 | 0.585 | 0.590 | 0.575 | 0.586 | 0.568 | 0.584 | 0.589 | 0.582 | 0.579 | 0.568 | 0.590 |

controlled:

| model | atd_cal2 | miss35 | miss47 | noctx | noisy | nomass | restart60 | vis15 | vis50 | min | max |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gbm_v5_nogap | 0.728 | 0.732 | 0.721 | 0.729 | 0.699 | 0.726 | 0.739 | 0.725 | 0.728 | 0.699 | 0.739 |
| gbm_v7 | 0.728 | 0.733 | 0.723 | 0.729 | 0.706 | 0.732 | 0.738 | 0.722 | 0.728 | 0.706 | 0.738 |
| gbm_v5_nogap_c2 | 0.733 | 0.736 | 0.724 | 0.734 | 0.716 | 0.729 | 0.741 | 0.728 | 0.732 | 0.716 | 0.741 |
| gbm_v7_c2 | 0.736 | 0.741 | 0.726 | 0.738 | 0.710 | 0.737 | 0.742 | 0.731 | 0.733 | 0.710 | 0.742 |

macroF1 (touch) 0.899-0.908 and dead F1 0.941-0.949 for all four. The dead-ball side does not move.

**Paired bootstrap** (`mtah.paired`, 1,000 draws). Each draw resamples the touch GT's 1,146
possessions (games 1-3 pooled). The same resample is applied to both models, every seed and every
setting. The draw's value is the mean A - B over the cells. "Cells" are seed x setting.

| A - B | GT | setting(s) | observed | cells > 0 | worst cell | 2.5-97.5 | draws > 0 |
|---|---|---|---|---|---|---|---|
| gbm_v7 - gbm_v5_nogap | touch | atd_cal (Phase D bench) | +0.018 | 3 / 3 | +0.012 | [+0.010, +0.027] | 1.000 |
| gbm_v7 - gbm_v5_nogap | controlled | atd_cal | +0.008 | 3 / 3 | +0.002 | [+0.000, +0.017] | 0.978 |
| gbm_v7 - gbm_v5_nogap (as is) | touch | 9 cal2 settings | +0.000 | 15 / 27 | -0.008 | [-0.006, +0.007] | 0.479 |
| gbm_v7_c2 - gbm_v5_nogap_c2 | touch | 9 cal2 settings | +0.005 | 22 / 27 | -0.013 | [-0.000, +0.010] | 0.967 |
| gbm_v7_c2 - gbm_v5_nogap_c2 | controlled | 9 cal2 settings | +0.002 | 18 / 27 | -0.013 | [-0.003, +0.008] | 0.818 |
| gbm_v7_c2 - gbm_v5_nogap (shipped) | touch | 9 cal2 settings | +0.003 | 22 / 27 | -0.008 | [-0.002, +0.009] | 0.874 |
| gbm_v7_c2 - gbm_v5_nogap (shipped) | controlled | 9 cal2 settings | +0.007 | 24 / 27 | -0.004 | [+0.002, +0.012] | 0.999 |

**Do the kink features still help once the inputs match? Barely, and not enough to ship.**

- On the Phase D bench, gbm_v7 led by +0.018 touch, with a paired interval that excludes 0. That
  part of the addendum's question has a yes: the Phase D lead was not resampling noise.
- On a ball shaped like the real one, the as-is gbm_v7 ties gbm_v5_nogap (+0.000, 15 of 27
  cells). Its Phase D lead came from the sharp kinks of the old degrader's ball, and those are not
  in the real track.
- Retrained on atd_cal2, the kink features keep +0.005 touch. That is 22 of 27 cells, worst
  -0.013, and the interval touches 0. Against the shipped model the lead is +0.003 touch and
  +0.007 controlled.
- This is a tenth of the noise band and not consistent in every cell. gbm_v7 and gbm_v7_c2 are
  not shipped.
- The as-is shipped model loses 0.007 touch going from atd_cal to atd_cal2 (0.588 to 0.581).
  Retraining on atd_cal2 does not win it back (gbm_v5_nogap_c2 0.579). The bench simply got a
  little harder; the shipped model is not mis-trained for this ball.


## Part 2: the video ball

### 2.1 What the real video track looks like (label-free, `mtah.video.props`)

`PYTHONPATH=src .venv/bin/python -m mtah.video.props --sim atd_cal2v cal2v_f20 cal2v_f60 --seeds 2`
profiles the saved track (support >= 5) beside the ATD. It writes `runs/phaseE/video_props_*.json`.

- Gap frames: graded, with the ATD ball missing (64,643).
- Window frames: graded, ATD visible, processed by the detector (the 25 frames either side of each
  gap; 7,198, of which 5,575 have a video ball).

**Fill.**

- 40.5% of gap frames are filled (26,189).
- By gap length: under 100 frames 0.64 (only 75 frames), 100-250 0.33, 250-800 0.34, 800+ 0.50.
- By the players' median speed: under 1 m/s 0.46 (stoppage-like), 1-2 0.35, 2-3 0.31, 3+ 0.32.
- By the ATD ball's last-seen speed before the gap: no clear trend (0.14-0.52 across bins).
- By position in the gap: first second 0.31, middle 0.42, last second 0.31.

**Bursts.**

- 5.2 bursts per gap.
- Burst length p25 / p50 / p75 / p90 is 6 / 9 / 18 / 50 frames (p99 392).
- Between bursts: p50 17 frames.
- First fill: a median 12 frames into the gap.

**The filled ball.**

- It is mostly still: speed p50 0.1 m/s, 66.5% under 0.5 m/s, 1.3% over 15 m/s.
- 13.0% of filled frames are off the pitch.
- 33.5% are within 2 m of a player. The nearest player is p50 4.7 m away.
- "Spare-like" (still, within 1 m of a line or outside it, nobody within 5 m): 4.3% of gap frames,
  10.5% of filled frames. This includes real dead balls lying out of play as well as spare balls.
  The two cannot be told apart without labels.

**Video against the ATD on the window frames.**

- Coverage is 0.78. It is flat across the ATD ball's speed (0.69-0.87, including 0.82 above
  15 m/s). Fast balls are tracked next to a gap, though almost never inside one.
- Error p25 / p50 / p75 / p90 is 0.40 / 1.05 / 3.78 / 11.2 m. 37% of frames are over 2 m and 20%
  over 5 m.
- The error grows with ball speed (p50 0.41 m under 1 m/s, 3.9 m above 15 m/s). This is the
  lofted-ball back-projection.
- The error is one-sided: mean dy -3.2 m, p5 -17 m. The ball lands on the far side of where the
  ATD has it, away from the camera.
- Within a tracked run, the error is mostly a constant offset (run bias p50 1.19 m, residual
  0.53 m, lag-1 autocorrelation 0.98).
- **Lag:** the best ATD shift against the video is 0.00 frames (scan -3 to +3 in quarter frames).

### 2.2 The simulated video channel (`degrade.DegradeConfig.video_*`, variant `atd_cal2v`)

The video ball goes in a new, separate field, `FrameTable.vball`. It is only ever set on frames
where the ATD ball is hidden, and it never changes `ball`. It draws from its own random stream, so
the ATD part of an `atd_cal2v` table is byte-identical to the `atd_cal2` table with the same seed
(a test checks ball and players). The simulation, inside the hidden graded frames:

- **True-ball fills in bursts.** A burst starts at a hidden frame with a hazard weighted by the
  true ball's speed: 1 / 0.04 / 0.02 / 0.01 for under 0.5 / 0.5-3 / 3-8 / 8-15 m/s, and never
  above 15 m/s. It runs for a length drawn from the real burst lengths (785 values,
  `runs/cache/video_calibration.npz`), and it stops early at a fast ball or at the gap's end. The
  base rate is solved for `video_fill_share` = 0.41 of hidden frames.
- **Matched error.** Each burst gets one error vector drawn from the real video-minus-ATD vectors
  in the true speed's bin, so fast and lofted balls get the large, far-side errors. An AR(1) drift
  (0.3 m, rho 0.9998) is added on top. As on the real track, the error is a per-run offset with
  little frame-to-frame jitter.
- **Detector margin.** Positions more than 3 m beyond a goal line or 1.94 m beyond a touchline are
  dropped, as the detector's candidate filter drops them.
- **False spare balls.** Still balls with 5 cm jitter, at a random point from 1.94 m outside to
  1 m inside a line, with no player within 5 m at the start. They come in real-length bursts, on
  `video_spare_share` = 0.02 of hidden frames.

Bench variants:

- `atd_cal2v`: the calibrated setting.
- `cal2v_f00` / `f20` / `f60`: fill 0 / 0.2 / 0.6, spares scaled with it.
- `cal2v_err2`: video error x2.
- `cal2v_nospare`, `cal2v_spare3`: no spares / 3x spares.
- `cal2v_<x>`: the eight atd_cal sweep settings on top of atd_cal2v.

The calibration was four simulate-and-compare rounds on games 1-3. Only the video statistics below
were looked at, never a model score.

| statistic | real track | atd_cal2v (2 seeds) | cal2v_f20 | cal2v_f60 |
|---|---|---|---|---|
| fill share of gap frames | 0.405 | 0.396 | 0.193 | 0.591 |
| fill share by players' median speed < 1 / 1-2 / 2-3 / >= 3 m/s | 0.46 / 0.35 / 0.31 / 0.32 | 0.57 / 0.31 / 0.16 / 0.20 | 0.30 / 0.16 / 0.04 / 0.06 | 0.72 / 0.51 / 0.44 / 0.40 |
| fill share: first second / middle / last second of a gap | 0.31 / 0.42 / 0.31 | 0.34 / 0.40 / 0.39 | 0.12 / 0.20 / 0.19 | 0.51 / 0.60 / 0.59 |
| burst length p25 / p50 / p75 / p90 (frames) | 6 / 9 / 18 / 50 | 7 / 12 / 26 / 60 | 6 / 10 / 18 / 43 | 6 / 11 / 28 / 101 |
| bursts per gap, mean | 5.23 | 5.47 | 3.31 | 5.74 |
| first fill delay p25 / p50 / p75 (frames) | 0 / 12 / 37 | 2 / 13 / 60 | 12 / 43 / 126 | 0 / 4 / 24 |
| filled ball speed p25 / p50 / p75 / p95 (m/s) | 0.1 / 0.1 / 0.5 / 4.3 | 0.1 / 0.3 / 1.2 / 8.8 | 0.1 / 0.2 / 0.5 / 6.0 | 0.1 / 0.4 / 2.1 / 10.1 |
| filled ball static (< 0.5 m/s) | 0.665 | 0.557 | 0.625 | 0.480 |
| filled ball faster than 15 m/s | 0.013 | 0.005 | 0.002 | 0.006 |
| filled ball off the pitch | 0.130 | 0.182 | 0.206 | 0.162 |
| filled ball within 2 m of a player | 0.335 | 0.430 | 0.433 | 0.444 |
| filled nearest player p25 / p50 / p75 (m) | 1.2 / 4.7 / 14.0 | 1.1 / 2.6 / 5.9 | 1.1 / 2.6 / 6.4 | 1.1 / 2.5 / 5.6 |
| spare-like share of gap frames | 0.0427 | 0.0366 | 0.0216 | 0.0472 |
| error (m) p25 / p50 / p75 / p90: real vs the ATD on window frames; bench vs the true ball | 0.40 / 1.05 / 3.78 / 11.2 | 0.41 / 0.79 / 2.50 / 36.5 | 0.44 / 0.76 / 17.6 / 53.7 | 0.45 / 0.82 / 2.30 / 16.9 |

**Video gate** (`props.check_video_calibration`, 8 rules: fill share +/-0.03, static share
+/-0.10, fast share <= 0.05, burst p50 / p90 within 50% / 60%, spare-like share +/-0.02,
within 2 m of a player +/-0.10, off-pitch share +/-0.06). `atd_cal2v` fails one rule:

- **static share, 0.557 against 0.665 (tolerance 0.10).** Strengthening the still weighting from
  1/0.5/0.3/0.15 to 1/0.04/0.02/0.01 moved it from 0.41 to 0.56 and then it saturated. The
  limit is the bench's truth. Real gaps are about 91% dead by the speed mixture, the bench's 55-69%,
  and the degrader's synthetic dead-ball paths move (carried, wandering). So a simulated fill sees
  a moving ball more often than the real detector does. The same cause puts simulated fills nearer
  players (p50 2.6 m against 4.7 m).
- Off-pitch (0.18 against 0.13) and near-player (0.43 against 0.34) pass, but at the edge.
- f20 / f60 fail on fill share by design.

The bench error column is not comparable to the real one row for row. The bench measures against
the true ball on live frames only (the clean ball is undefined in stoppages). Fills favour still,
dead balls, so on live frames the spare balls, which are 30-50 m from the real ball, are a larger
share. That makes the bench's tail longer than the real window error's.


### 2.3 Video features: a separate channel (`features_v`, 25 columns)

Every column is named `v_` and reads only `vball` (and only where the ATD ball is missing). The
ATD-ball features keep their ATD meaning. The columns:

- the source flag and its 2 s / 10 s shares;
- frames since / until a video sighting;
- the nearest-player distance difference and the nearest distance;
- a 2 m touch sign (wider than the ATD's 1-1.5 m, because the position is a ground
  back-projection) with 1 s / 3 s windowed sums, the last / next video touch's team, and since /
  until;
- speed and stillness;
- the signed distance to the nearest line;
- a spare-like flag;
- position along home's attack;
- the last / next video ddiff.

All are team-symmetric (a test swaps the teams). A merged ATD+video ball variant was not built
(found-not-fixed).

**gbm8** = gbm_v7's design (gbm_v5_nogap features + the kink features) + `features_v`. It is
trained on the atd_cal2v mixture: the six Phase B2 mixture settings on atd_cal2, each with its
own video setting, so training sees fill shares 0.10-0.55, video error x1.5 and spare share 0.05:

| mixture entry | video setting |
|---|---|
| cal_default | fill 0.41, spares 0.02 |
| cal_vis15 | fill 0.25 |
| cal_vis50 | fill 0.55 |
| cal_noisy | error x1.5 |
| cal_noctx | fill 0.10 |
| cal_restart60 | fill 0.45, spares 0.05 |

Held out of training: fill 0 / 0.2 / 0.6 at the default settings, error x2, no spares, 3x spares,
and the three held-out atd_cal settings (miss35, miss47, nomass). **gbm8n** is the same without
the kink features (an ablation). Training reuses the cached atd_cal2 feature parts and adds only
the video columns: video options never change the ATD part, and the part loader checks that the
rows' labels line up.

### 2.4 The ship test, fixed before the sweep was read

Written at 14:20, after the main atd_cal2v column and the f00 / f20 / f60 / err2 / nospare cells
had printed, and before the eight calibrated sweep columns and spare3. The reference is the
shipped model, `gbm_v5_nogap` as is. Its tables are the atd_cal2 tables with the same seed: the
same ATD part, with no video.

1. **Clear lead.** Pooled over the 27 cells (atd_cal2v and its eight calibrated sweep settings, x 3
   seeds), the touch difference is either above 0.03, or positive in all 27 cells with the paired
   bootstrap's 2.5% bound above 0.
2. **Controlled GT.** The pooled controlled difference is at least -0.005, and no cell is below
   -0.02.
3. **Video robustness.** On fill 0.2 / 0.6, error x2, no spares and 3x spares, no loss against the
   reference (touch difference >= 0). On fill 0 (no video at all), within -0.005.
4. **Transfer** (label-free, real match):
   - the video inputs' distributions look like the bench's;
   - possessions per hour, median length, D share and home share stay plausible;
   - the frames where the candidate differs from the shipped file look like the bench's.

**Added at 15:10, before gbm_v8n's sweep was complete.** gbm_v8n (gbm8 without the kink features)
tied gbm_v8 on the main column (0.603 against 0.602 touch). If gbm_v8n passes the same four tests,
it is the one shipped, because it assumes less. The kink features read exactly the part of the
ball track where atd_cal2 still misses (kink8 p50), and Part 1 found they add about +0.005 at
most. If gbm_v8n does not pass and gbm_v8 does, gbm_v8 ships.

### 2.5 Results

Commands:

```bash
PYTHONPATH=src .venv/bin/python -m mtah.models.learned --gt touch --tag gbm8 gbm8n --workers 3
MTAH_FEATURE_CACHE=0 PYTHONPATH=src .venv/bin/python -m mtah.modelbench --models gbm_v8 --variants atd_cal2v \
    --seeds 3 --bootstrap 300 --tag e_v8_main          # and gbm_v8n, --tag e_v8n_main
runs/phaseE/sweep_c2v.sh; runs/phaseE/sweep_c2v8n.sh  # 14 sweep settings each, bootstrap 0
PYTHONPATH=src .venv/bin/python -m mtah.paired --a gbm_v8n --b gbm_v5_nogap \
    --variants atd_cal2v cal2v_vis15 cal2v_vis50 cal2v_miss35 cal2v_miss47 cal2v_noisy cal2v_noctx cal2v_restart60 cal2v_nomass \
    --b-variants atd_cal2 cal2_vis15 cal2_vis50 cal2_miss35 cal2_miss47 cal2_noisy cal2_noctx cal2_restart60 cal2_nomass --seeds 3
```

The shipped model's scores come from the atd_cal2 tables with the same seeds. Their ATD part is
byte-identical to atd_cal2v's; it simply has no video channel.

**Main table (atd_cal2v; mean of 3 seeds, bootstrap 5-95 on seed 0)**

| model | touch spellF1 [boot] (seeds min-max) | touch macroF1 | controlled spellF1 [boot] | pred poss (GT 1146) | dead F1 | touch game 1 / 2 / 3 |
|---|---|---|---|---|---|---|
| gbm_v5_nogap (shipped, no video) | 0.581 [0.547, 0.603] (0.572-0.593) | 0.905 | 0.728 [0.704, 0.748] | 724 | 0.945 | 0.592 / 0.595 / 0.557 |
| gbm_v7_c2 (no video) | 0.585 [0.556, 0.608] (0.578-0.596) | 0.906 | 0.736 [0.713, 0.760] | 713 | 0.947 | 0.596 / 0.610 / 0.553 |
| gbm_v8 (kink + video) | 0.602 [0.569, 0.621] (0.592-0.612) | 0.916 | 0.756 [0.729, 0.774] | 731 | 0.958 | 0.616 / 0.618 / 0.575 |
| **gbm_v8n (video, no kink)** | **0.603 [0.567, 0.623] (0.593-0.611)** | 0.915 | **0.755 [0.732, 0.777]** | 736 | 0.957 | 0.623 / 0.614 / 0.573 |

**Robustness, pooled spellF1 touch (mean of 3 seeds).** The shipped model's column is its
atd_cal2 counterpart. For the video-only settings, that counterpart is atd_cal2 itself.

| model | cal2v | vis15 | vis50 | miss35 | miss47 | noisy | noctx | restart60 | nomass | f00 | f20 | f60 | err2 | nospare | spare3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gbm_v5_nogap | 0.581 | 0.576 | 0.577 | 0.584 | 0.574 | 0.563 | 0.581 | 0.587 | 0.583 | 0.581 | 0.581 | 0.581 | 0.581 | 0.581 | 0.581 |
| gbm_v8 | 0.602 | 0.603 | 0.598 | 0.607 | 0.596 | 0.584 | 0.604 | 0.601 | 0.606 | 0.581 | 0.595 | 0.610 | 0.596 | 0.603 | 0.601 |
| gbm_v8n | 0.603 | 0.600 | 0.598 | 0.607 | 0.598 | 0.588 | 0.603 | 0.597 | 0.601 | 0.581 | 0.595 | 0.608 | 0.599 | 0.603 | 0.603 |

controlled:

| model | cal2v | vis15 | vis50 | miss35 | miss47 | noisy | noctx | restart60 | nomass | f00 | f20 | f60 | err2 | nospare | spare3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gbm_v5_nogap | 0.728 | 0.725 | 0.728 | 0.732 | 0.721 | 0.699 | 0.729 | 0.739 | 0.726 | 0.728 | 0.728 | 0.728 | 0.728 | 0.728 | 0.728 |
| gbm_v8 | 0.756 | 0.756 | 0.755 | 0.757 | 0.754 | 0.728 | 0.756 | 0.758 | 0.757 | 0.729 | 0.746 | 0.770 | 0.749 | 0.756 | 0.755 |
| gbm_v8n | 0.755 | 0.756 | 0.762 | 0.761 | 0.755 | 0.735 | 0.757 | 0.756 | 0.758 | 0.730 | 0.746 | 0.763 | 0.749 | 0.755 | 0.754 |

macroF1 (touch): gbm_v8 0.904-0.919, gbm_v8n 0.904-0.917; shipped 0.899-0.907. Dead F1: 0.944-0.959
for both against 0.941-0.947.

Reading it:

- **The video adds about +0.02 touch and +0.03 controlled, and scales with the fill.** With no
  video at all (f00), gbm_v8 / v8n score exactly what the shipped model scores (0.581 / 0.729-0.730).
  At fill 0.2 / 0.4 / 0.6 they score 0.595 / 0.602-0.603 / 0.608-0.610.
- **Doubling the video error or tripling the spare balls costs little**: 0.596-0.603 touch.
- **The kink features add nothing on top of the video.** gbm_v8n - gbm_v8 is -0.001 touch (10
  of 27 cells), paired [-0.006, +0.005]; and +0.002 controlled.
- Predicted possessions rise slightly (731-736 against 724), still far below the touch GT's 1,146.
  The video is fixing who has the ball inside gaps; it does not find the short possessions
  (Phase D found-not-fixed 1, still open).

### 2.6 Paired bootstrap against the shipped model

Same method as 1.3: 1,000 draws, touch-GT possessions resampled once per draw and shared by both
models, every seed and every setting.

| A - B | GT | settings | observed | cells > 0 | worst cell | 2.5-97.5 | 5-95 | draws > 0 |
|---|---|---|---|---|---|---|---|---|
| gbm_v8n - gbm_v5_nogap | touch | 9 calibrated (27 cells) | **+0.021** | **27 / 27** | +0.003 | **[+0.014, +0.029]** | [+0.015, +0.027] | 1.000 |
| gbm_v8n - gbm_v5_nogap | controlled | 9 calibrated | +0.030 | 27 / 27 | +0.004 | [+0.022, +0.037] | [+0.023, +0.036] | 1.000 |
| gbm_v8n - gbm_v5_nogap | touch | 6 video settings (18 cells) | +0.018 | 17 / 18 | -0.005 (f00) | [+0.009, +0.026] | | 1.000 |
| gbm_v8n - gbm_v5_nogap | controlled | 6 video settings | +0.021 | 17 / 18 | -0.011 (f00) | [+0.012, +0.030] | | 1.000 |
| gbm_v8 - gbm_v5_nogap | touch | 9 calibrated | +0.022 | 27 / 27 | +0.009 | [+0.014, +0.029] | [+0.016, +0.028] | 1.000 |
| gbm_v8 - gbm_v5_nogap | controlled | 9 calibrated | +0.028 | 27 / 27 | +0.007 | [+0.021, +0.036] | [+0.022, +0.034] | 1.000 |
| gbm_v8 - gbm_v5_nogap | touch | 6 video settings | +0.017 | 17 / 18 | -0.003 (f00) | [+0.009, +0.025] | | 1.000 |
| gbm_v8n - gbm_v8 | touch | 9 calibrated | -0.001 | 10 / 27 | -0.008 | [-0.006, +0.005] | | 0.415 |

Per video setting, gbm_v8n - gbm_v5_nogap touch:

| setting | f00 | f20 | f60 | err2 | nospare | spare3 |
|---|---|---|---|---|---|---|
| observed | +0.001 | +0.015 | +0.027 | +0.018 | +0.022 | +0.022 |
| paired 2.5-97.5 | [-0.008, +0.009] | [+0.006, +0.023] | [+0.018, +0.038] | [+0.009, +0.027] | [+0.013, +0.032] | [+0.013, +0.031] |

The weakest calibrated setting is `cal2v_restart60` (+0.010, worst cell +0.003). When the ATD
re-acquires the ball soon after restarts, gaps are shorter and there is less for the video to add.

What the bootstrap does and does not cover:

- It resamples ground-truth possessions with the predictions held fixed, as the organiser's own
  interval does. It does not cover the variance of retraining, or a bench that differs from the
  real match.
- The 27 cells share three sample games and their training data, so they are not 27 independent
  tests.
- The lead is below the 0.03-0.04 band, so the case rests on the second route: positive in every
  cell, with a paired interval that excludes 0.

### 2.7 Transfer checks on the real match (label-free)

**Inputs.** `diagnostics.input_shift` (`runs/phaseE/input_shift.json`) compares graded frames,
video columns on ATD-missing frames, real match with its saved track against atd_cal2v seed 0,
games 1-3:

| input | real | atd_cal2v | reading |
|---|---|---|---|
| video present on gap frames (`v_src`) | 0.405 | 0.395 | matched |
| video touch within 2 m (`v_tt` != 0), share of gap frames | 0.136 | 0.166 | bench has about 20% more video touches |
| video ball's nearest player p10 / p50 / p90 (m) | 0.66 / 4.7 / 28.9 | 0.59 / 2.7 / 14.5 | real fills sit further from players |
| video ball still, share of filled frames with a speed | 0.74 | 0.64 | real fills are stiller |
| video ball line distance p10 / p50 / p90 (m) | -0.4 / 5.3 / 32.9 | -0.9 / 5.4 / 27.8 | close |
| spare-like flag | 0.081 | 0.087 | matched |
| video speed p50 / p90 (m/s) | 0.15 / 2.0 | 0.28 / 5.3 | real slower |
| kink4 p10 / p50 / p90 (m/s), all graded frames | 0.13 / 0.91 / 2.89 | 0.25 / 0.75 / 2.94 | close (atd_cal: 0.13 / 0.37 / 4.32) |

The real video is a little weaker team evidence than the simulated one: fewer touches, further
from players, stiller (more dead balls). The f20 column bounds the damage. Even with half the
simulated video, gbm_v8n leads by +0.015 in every cell.

**Labels** (`mtah.diagnostics`, `runs/phaseE/diag/`, `runs/phaseE/diag_v8n/`):

| statistic | bench GT touch | bench GT controlled | gbm_v5_nogap (shipped) | gbm_v7_c2 | gbm_v8 | gbm_v8n |
|---|---|---|---|---|---|---|
| D share (graded) | 0.389 | 0.389 | 0.431 | 0.435 | 0.433 | 0.434 |
| D share, ball missing / visible | - | - | 0.830 / 0.157 | 0.836 / 0.160 | 0.831 / 0.160 | 0.833 / 0.161 |
| stoppages per hour, median (s) | 63, 16.9 | 63, 16.9 | 66, 17.8 | 63, 18.7 | 65, 18.5 | 65, 18.5 |
| possessions per hour | 243 | 148 | 147 | 144 | 144 | 146 |
| possession p25 / p50 / p75 (s) | 1.8 / 4.4 / 12.5 | 4.8 / 10.3 / 19.9 | 3.7 / 7.5 / 21.0 | 3.7 / 7.8 / 20.0 | 3.7 / 8.2 / 20.7 | 3.6 / 7.4 / 19.9 |
| possessions under 3 s | 0.37 | 0.11 | 0.18 | 0.18 | 0.19 | 0.19 |
| home share of H + A | 0.516 | 0.521 | 0.615 | 0.619 | 0.619 | 0.611 |
| agreement with Smart Tagging attacking-phase team | - | - | 0.779 | 0.779 | 0.777 | 0.782 |

All plausible and close to the shipped model. The speed-mixture estimate of the dead share in
real gaps is unchanged (0.909).

**Where the candidate differs from the shipped file** (share of graded frames whose label changes;
real: the two files; bench: atd_cal2v against atd_cal2, 3 seeds x 3 games):

| | all graded | ATD visible | gap | gap with video | gap without video |
|---|---|---|---|---|---|
| real, gbm_v8n vs shipped | 0.033 | 0.027 | 0.043 | 0.048 | 0.039 |
| bench, range of 9 tables | 0.027-0.034 | 0.017-0.023 | 0.037-0.052 | 0.034-0.066 | 0.033-0.050 |
| real, gbm_v8 vs shipped | 0.037 | 0.035 | 0.040 | 0.041 | 0.040 |

The real match changes about as much, and in the same places, as the bench does. Visible-ball
frames change a little more on the real match (0.027 against at most 0.023).

## Ship decision

**gbm_v8n replaces gbm_v5_nogap**, by the test in 2.4:

1. **Clear lead.** +0.021 touch pooled, positive in 27 of 27 calibrated cells (worst +0.003),
   paired bootstrap [+0.014, +0.029]. This is below the 0.03-0.04 band; it passes on the
   every-cell-plus-paired-bootstrap route.
2. **Controlled GT.** +0.030, 27 of 27 cells. No loss.
3. **Video robustness.**
   - No loss at fill 0.2 / 0.6, error x2, no spares or 3x spares (+0.015 to +0.027).
   - With no video at all: +0.001.
4. **Transfer.**
   - The inputs match on fill share, spare-like share and line distance. The real video is
     somewhat weaker evidence than the simulated video (see 2.7).
   - Possession count, length, D share and home share are plausible and close to the shipped
     model's.
   - The frames that change look like the bench's.
   - gbm_v8 passes all four tests too. gbm_v8n is shipped by the rule fixed at 15:10: it ties
     gbm_v8 and assumes less.

**It is written under a new name. `runs/submission/MurrayInsights.csv` is untouched**
(SHA-256 `2a7c130b...`). Renaming the file for upload is Steve's call.

- `runs/submission/MurrayInsights_gbm_v8n.csv`: 199,319 rows, H 54,856 / A 34,993 / D 69,008 /
  X 40,462 (X outside graded time); `validate_submission: OK`; video channel on 26,189 frames.
- A second run in a separate process (`--out runs/phaseE/MurrayInsights_gbm_v8n_rerun.csv`)
  gave a byte-identical file. SHA-256
  `899be8eedc8d75aff28636fc6c1b1e92043b24087b62d148a91ee462ffd3a382`.
- For comparison, gbm_v8's file `runs/submission/MurrayInsights_gbm_v8.csv` is also validated and
  byte-identical on a rerun (SHA-256 `877d9818...`).

**Exact commands**

```bash
# 0. the video ball track (once; detector environment ~/.cache/wt-mtah-video/venv, see below)
# 1. train (builds the atd_cal2 parts first; gbm5n_c2 makes them, gbm8n adds the video columns)
PYTHONPATH=src .venv/bin/python -m mtah.models.learned --gt touch --tag gbm5n_c2 gbm8n --workers 6
# 2. labels for the competition ATD, the saved video track attached as a separate channel
PYTHONPATH=src .venv/bin/python -m mtah.predict --model gbm_v8n --competition \
    --out runs/submission/MurrayInsights_gbm_v8n.csv
```

Training does not need step 1's gbm5n_c2 models, only its cached atd_cal2 parts. A fresh clone
running `--tag gbm8n` alone computes the full features: slower, with the same numbers, because
the part loader checks labels line up. The competition model is
`runs/models/gbm8n_touch_123_v4b21v1.pkl`. The degrader needs two cached distributions from the
match:

- `runs/cache/atd_gap_lengths.npy`, from `atd_profile.real_gap_lengths()`;
- `runs/cache/video_calibration.npz`, from `video.props.real_video_calibration()`. These are the
  real fill-burst lengths and video-minus-ATD error vectors, shuffled. Both are rebuilt from
  `data/` and the saved track on first use.

**Reproduction now also needs the video detector run**, because the predict step reads
`runs/video/full/ball_track_soccana_ov.npz`. The organisers must be able to regenerate it:

```bash
# detector environment (Python 3.12.13: torch 2.15 CPU, ultralytics 8.4.202, opencv 5.0,
# openvino 2026.4); weights by sha256 from mtah.video.weights (soccana, Adit-jain/soccana on HF)
DET=~/.cache/wt-mtah-video/venv/bin/python
cd src
../.venv/bin/python -m mtah.video.cache                        # runs/video/atd_cache.npz
$DET -m mtah.video.weights                                      # fetch + verify weights
$DET -m mtah.video.fullrun detect --config soccana_ov --scope gaps --window 25
$DET -m mtah.video.fullrun track --config soccana_ov            # -> ball_track_soccana_ov.npz
```

- That is about 6 h of CPU on this laptop.
- The project has no requirements file for the detector environment yet. One must be written,
  with exact pins, before submission.
- The full run used `~/.cache/wt-mtah-video/run_full.sh` (the two fullrun lines above, at nice 19,
  with `YOLO_OFFLINE=1`). That script lives outside the repo and should be copied in.
- On the pilot, torch and OpenVINO agreed within 2 px on every detection. Whether a fresh
  detection run gives a byte-identical track, on this machine or another, was not tested (see
  found-not-fixed). If it does not, the submitted file is reproducible only from the saved track.

## Public window, once, after the decision

The organiser's scorer was run once, after the ship decision above was written, on the gbm_v8n
file (`cd ref/evaluation && ../../.venv/bin/python run_evaluation.py --results
../../runs/phaseE/publiccheck --gt public`).

**gbm_v8n: spellF1 0.593, macroF1 0.886.** 12 possessions against 17; alive/dead F1 0.967; home
share 0.569 against 0.573. For comparison, gbm_v5_nogap scored 0.619 / 0.898 there (Phase B2)
and gbm_v7 0.621 / 0.886 (Phase D).

- The file is aligned: frames, teams and the dead ball are right.
- It is 0.026 below the shipped file, on 17 possessions. One matched possession at one IoU is
  worth about 0.07 here. EVALUATION.md shows that the public window picks the wrong model at
  this scale, so this says nothing about which model is better.
- Nothing was changed after it.

## Found, not fixed

1. **atd_cal2 still fails one shape rule:** kink8 p50 is 1.23 against 1.76, 30.2% low against a
   30% tolerance. The two-parameter wobble cannot match the k = 8 median without overshooting
   kink4 p25. Real big ball jumps (2.0 per minute) are absent from atd_cal2 (0.09).
2. **The video simulation fails one gate rule:** the static share of fills is 0.557 against
   0.665. Simulated fills also sit nearer players (p50 2.6 m against 4.7 m), and the real video
   touch rate is about 20% below the bench's. The cause is the bench's truth: its gaps are
   55-69% dead against about 91% on the real match, and its synthetic dead-ball paths move. The
   real video is therefore probably weaker team evidence than on the bench, and the real gain is
   probably smaller than +0.021. f20 (+0.015) is the closest bound we have.
3. **Fresh-detection reproducibility is untested.** The predict path is byte-identical from the
   saved track. Rerunning the detector from the video on another machine (thread count,
   OpenVINO version, CPU instruction set) may change detections and hence the file. This needs
   checking before 11-17. The fallback is to document the saved track as an intermediate
   produced by the detector command, if the licence allows sharing it.
4. **The detector environment has no pinned requirements file** (AGPL-3.0 ultralytics; soccana
   weights with no card licence). Both must be in the public repo and the technical document.
5. **No merged-ball variant was tested.** The ATD features keep their ATD meaning, and the video
   is a separate channel only.
6. **The video does not find short possessions.** Predicted possessions are 736 against the touch
   GT's 1,146. The gain is in who has the ball inside gaps.
7. **The degraded-table cache write was not atomic.** Two bench workers building the same table
   could hand a reader a half-written file (`BadZipFile` on the first cal2_vis15 run). It now
   writes to a per-process temporary name and renames. cal2_vis15 was rerun; no result was mixed.
8. **The paired bootstrap's limits** are listed in 2.6.
9. **Window coverage of fast balls (0.82 above 15 m/s next to gaps) is not simulated.** The
   bench never shows the video ball outside gaps, and neither do the models' inputs, so it does
   not matter for these models. It would matter for a merged-ball variant.

## Tests

`tests/test_phaseE.py` (15). Each gate is fed the case it exists to catch:

- the shape rules fail atd_cal's ball error against a shaped reference and pass another seed of
  it;
- against the real profile, atd_cal fails the kink4 rules and atd_cal2 passes them;
- the ball-shape options are the identity at their defaults; the lag shifts the ball by the set
  frames; the wobble and smoothing change the ball only;
- the video fill:
  - appears only in gaps and leaves the ATD part byte-identical;
  - hits its share;
  - never fills a ball faster than 15 m/s;
  - fills still balls more than twice as often as moving ones;
  - takes its error from the calibration pool exactly;
- spare balls sit by the lines, away from players, at their share;
- the video profile finds a 1-frame lag that was put in, and 0 when there is none;
- the video gate fails a moving simulation on exactly the two rules it breaks;
- `with_video_channel` never changes the ATD ball;
- `vball` survives save, load and copy, and a wrong shape raises;
- the video features ignore frames where the ATD has the ball and flip with the teams;
- the paired F1 equals the organiser's `spell_scores`, a better model wins on every resample,
  and identical models give 0.

Full suite: `.venv/bin/python -m pytest` 157 passed, 142 before this phase.


