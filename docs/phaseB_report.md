# Phase B report: the match data arrives

Date: 2026-11-05. The bundle is in `data/` (gitignored), laid out as `ref/data/README.md` asks:
the tactical video, ATD, homography and Smart Tagging, plus the broadcast ATD and homography
(the broadcast video was not downloaded). Nothing here uses the full-match ground truth. The
public five minutes are not used in this report.

## Checks

| check | result |
|---|---|
| `git check-ignore` on the bundle | every file ignored (`.gitignore` line `data/*`); `git status` clean |
| organiser `starter/baseline.py` on our copy | all four baselines byte-identical to `ref/evaluation/results/` |
| our loader (`atd_io.load_competition`) + our port (`baseline.predict_all`) | all four byte-identical |
| `load_competition` strict checks | passed: 199,319 records, ids contiguous from 1, every record in a format spec, Barcelona home; load 9.4 s |
| `predict --model gbm_v4 --competition` | 199,319 rows, `validate_submission: OK`, 31 s |

## Profile of the tactical ATD (distributions only, no labels)

- **Graded frames:** 158,857. Ball visible on 59.3% of them (56.9% over the whole video, as
  `DATA.md` says).
- **Missing ball comes in long blocks, not flicker.** On graded frames there are 150 missing-ball
  runs, with lengths in frames of median 268, 75th percentile 527, 90th 845, 99th 2,619 and max 5,537.
  Runs of 25 frames or more hold 99.9% of the missing frames. Visible runs: 149, median 345.
- **Players:** median 22 tracked per graded frame (11 per team), 5th percentile 14. 46 slots,
  24 in the first team and 22 in the second.

## What this changes

The Phase A degrader hides the ball during live play in lognormal bursts with a median of 20
frames (mean about 33). Long hidden stretches happen only at stoppages. The real file has no short
bursts at all. Its gaps are about 13 times longer and make up 41% of graded time.

The bench therefore taught the learned model that a long stretch without the ball is a stoppage.
On the real match `gbm_v4` labels 46.9% of graded frames `D`: 17.9% where the ball is visible and
**89.3% where it is missing**. smooth50 gives 32.3% overall and 79.5% where the ball is missing.
Both numbers are a symptom of the gap model, not a score; no labels were consulted.

Next:

1. **Degrader recalibration.** Feed the empirical run lengths into `ball_burst_lengths`, and find
   out from the automatic Smart Tagging how gaps relate to set pieces. Then retrain and re-bench.
2. **Video ball detection pilot.** The gaps are long enough that filling them from the tactical
   video, with the 100% usable homography, may be worth more than any model change.
