# Phase D: short possessions

Date: 2026-11-05. Bench only: Metrica sample games 1-3 degraded with `atd_cal`, leave one game
out, degrader seeds 0-2 for scoring and 10000+ for training, both GT variants. No BAR-ATH label
was read; the full-match GT is not ours. The public 5-minute window was run once, at the end, on
the final pick (last section), and nothing changed after it.

## Summary

- **The misses are in the emissions, not the decoder.** 37% of the touch GT's possessions are under
  3 s and gbm_v5_nogap matches 1-16% of them. 85% of all missed possessions are not matched by any
  decoding of its probabilities, down to the per-frame argmax: inside a missed short possession the
  model gives its team 0.31-0.45. Lower switching costs, a touch-gated cost and a re-gridded
  alpha x cost all leave touch spellF1 where it was (0.567-0.592), while the predicted count moves
  from 637 to 854.
- **Why the emissions miss them:** on the degraded bench, touches by the team without the ball
  that change nothing outnumber touches at real hand-overs (7,868 against 5,241 frames on seed 0),
  so a single touch is weak evidence. The ball's velocity change at the touch separates the two
  (AUC 0.79), and no existing feature measured it.
- **gbm_v7 (the touch-kink features, `features_d`) is the bench leader but not a clear win.**
  Touch spellF1 0.606 [0.572, 0.623] against gbm_v5_nogap's 0.588 [0.558, 0.613], controlled 0.743
  against 0.734, macroF1 0.911 against 0.908. It is ahead on every seed (+0.012 / +0.021 / +0.020
  touch) and in all 9 calibrated settings in both GTs (touch +0.010 to +0.021, controlled +0.002
  to +0.009), but every gap is inside the 0.03-0.04 noise band and the bootstrap intervals overlap.
  Its real-match possession structure is unchanged (150 per hour against 147, median 7.6 s).
- **Weighting training rows by possession length (gbm_v6_w05 / w1, gbm_v7_w05) trades the two GTs
  against each other**: touch +0.015 to +0.024, controlled -0.012 to -0.048, more predicted
  possessions (779-841). Not kept.
- **The submission file is not replaced.** No candidate clears the noise band. gbm_v7's file is
  written under a separate name (`runs/submission/MurrayInsights_gbm_v7.csv`), validated, and
  byte-identical on a rerun. Whether to switch is Steve's call; the case for it is that gbm_v7 is
  ahead in every column of both GTs, the case against is that "ahead by less than the noise band
  everywhere" is still inside the noise band.
- **The short-possession gap is mostly not closed.** gbm_v7 matches 50.5% of GT possessions at IoU
  0.5 (gbm_v5_nogap 48.7%) and predicts 738 against 1,146. Possessions under 3 s go from 1.2% / 15.7%
  matched to 2.1% / 18.9%.

## 1. Diagnosis: which possessions are missed, and why

`PYTHONPATH=src .venv/bin/python -m mtah.segdiag` (gbm_v5_nogap, `atd_cal`, seeds 0-2, 3,438 touch-GT
possessions over the three seeds; decode alpha 0.5, H<->A cost 8). For each GT possession it
records the scorer's one-to-one IoU match, length, ball visibility, what surrounds it, the model's
probabilities inside it, and whether the same probabilities decoded with a lower switching cost (4,
2, 1, 0) or the per-frame argmax would have matched it.

**By length** (hit = matched at IoU 0.5; p_own = mean model probability of the possession's team;
frac_fav = share of its frames where that team is the argmax):

| length | GT spells | share | hit@0.3 | hit@0.5 | hit@0.7 | ball visible | p_own | frac_fav | hit@0.5 at cost 4 | cost 2 | cost 0 | argmax |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| <1 s | 432 | 0.126 | 0.025 | 0.012 | 0.002 | 0.73 | 0.31 | 0.28 | 0.012 | 0.021 | 0.032 | 0.030 |
| 1-3 s | 852 | 0.248 | 0.228 | 0.157 | 0.088 | 0.67 | 0.45 | 0.45 | 0.195 | 0.212 | 0.222 | 0.229 |
| 3-6 s | 705 | 0.205 | 0.618 | 0.502 | 0.325 | 0.65 | 0.64 | 0.68 | 0.519 | 0.523 | 0.523 | 0.530 |
| 6-12 s | 564 | 0.164 | 0.835 | 0.715 | 0.550 | 0.70 | 0.82 | 0.86 | 0.725 | 0.725 | 0.754 | 0.759 |
| 12-30 s | 663 | 0.193 | 0.911 | 0.851 | 0.735 | 0.73 | 0.91 | 0.94 | 0.863 | 0.875 | 0.906 | 0.906 |
| >=30 s | 222 | 0.065 | 0.991 | 0.959 | 0.842 | 0.84 | 0.94 | 0.96 | 0.964 | 0.959 | 0.964 | 0.950 |

- **Possessions under 3 s are 37% of the touch GT and are matched 1-16% of the time.** Longer
  than 12 s, 85-96%. The ball is visible on 67-73% of a short possession's frames, about as often
  as on long ones, so this is not a ball-gap problem.
- **It is the emissions, not the decoder.** Of the 1,765 missed possessions (IoU 0.5), 3.8% are
  recovered by halving the switching cost to 4, another 2.0% at cost 2, 6.5% only at cost 0-1 and
  2.8% only by the per-frame argmax. **85% are not matched by any decoding of these
  probabilities.** Inside a missed short possession the model gives its team 0.31-0.45 on average
  (0.90 in matched ones). Summing alpha · log(P_team / P_other) over a missed possession's frames,
  the evidence points the wrong way for 50% of misses, is positive but below the 16 a cost-8
  insertion needs for 12%, and is above it for 38% (mostly 3-12 s possessions missed for D or
  boundary reasons).
- **Context.** 54% of GT possessions sit between two runs of the other team (A-B-A); those are
  matched 42% of the time (median 3.9 s). Possessions next to a D are matched 53-75%.
- **Miss modes** (IoU 0.5): absent, no same-team predicted run overlaps at all, 49% (median
  1.4 s); swallowed into a same-team run over twice as long, 40% (median 2.8 s: the other half of an
  A-B-A whose B was missed, or a same-team pair across a missed D); boundary, overlap with IoU too
  low, 11%.
- **Hand-over timing is loose.** Where a GT H<->A hand-over has a predicted one of the same
  direction within 4 s, the predicted one is a median 2 frames late, IQR -9 to +21, p10-p90 -32 to
  +48. The per-frame crossing of P(H) and P(A) has the same spread (-7 / +21), so this too is in the
  emissions. On a 2 s possession, 20 frames at each end is the difference between IoU 0.5 and 0.3.
- **The information is partly there.** A touch (ball within 1.5 m, `tt15`) by the possession's own
  team is seen in 55% of 1-3 s possessions on the degraded tables (78% when the ball is visible;
  98% on the clean tables), yet only 16% of those are matched. The reason is the other half of the
  picture: on the degraded tables, touch frames by the team without the ball that change nothing
  (7,868 on seed 0) outnumber touch frames at a real hand-over (5,241). A single touch is not
  evidence of a hand-over, and the model has learned to wait for more.
- **What separates the two is the ball, not the player.** At those touch frames, the ball's
  velocity change over 8 frames either side separates real hand-overs from passing touches with
  AUC 0.79 (median 5.1 against 0.85 m/s; over 4 frames, AUC 0.77). The distance to the nearest
  player does not (AUC 0.52). No current feature measures a direction change: `bspeed` and its
  shifts see speed only.

So the fix belongs in the emissions. Two candidates follow from this: weight the training rows so
short possessions count (the loss counts frames, the metric counts possessions), and give the
model the ball's velocity change at a touch.

## 2. What was tried

Every row is pooled over sample games 1-3, mean of degrader seeds 0-2 on `atd_cal`, touch GT
unless stated. Decoder experiments reuse gbm_v5_nogap's cached probabilities, so they cost no
retraining.

| version | idea | touch spellF1 | controlled spellF1 | pred poss (GT 1146) | kept? |
|---|---|---|---|---|---|
| gbm_v5_nogap | Phase B2 best, alpha 0.5, H<->A cost 8 | 0.588 | 0.734 | 723 | reference |
| decode grid | alpha {0.35, 0.5, 0.75, 1} x H<->A cost {4, 8, 12, 16} on gbm_v5_nogap | 0.567-0.590 | 0.680-0.757 | 637-854 | dropped: flat; more possessions buy nothing in touch and cost controlled |
| touch-gated cost | H<->A cost lowered to 2-4 within 12-25 frames of a 1 m / 1.5 m touch by the incoming team, base 8-16 (8 settings, `hmm.viterbi` now takes a per-frame cost) | 0.587-0.592 | - | 2174-2324 over 3 seeds | dropped: +0.004 at best, as the diagnosis predicted (6% of misses are within reach of a lower cost) |
| gbm_v6_w05 | gbm_v5_nogap's features and rows, training rows weighted by (1 / max(L, 25))^0.5, L the row's GT possession length; possession rows keep their total weight, D rows weigh 1 | 0.603 | 0.714 | 779 | dropped: +0.015 touch, -0.020 controlled |
| gbm_v6_w1 | as above, gamma 1 (every possession weighs the same) | 0.602 | 0.686 | 841 | dropped: -0.048 controlled |
| gbm_v6 decode grids | the same alpha x cost grid on both | 0.583-0.607 | 0.616-0.757 | 657-1032 | flat in touch again |
| **gbm_v7** | gbm_v5_nogap plus 15 touch-kink features (`features_d`): the ball's velocity change over 4 and 8 frames either side, the 1.5 m touch sign times the kink and its windowed sums (0.5 / 1 / 3 s), the team of the last / next strong touch (kink 2 m/s or more), frames since / until each team's strong touch | **0.606** | **0.743** | 738 | **bench leader, inside the noise band; not shipped** |
| gbm_v7_w05 | gbm_v7 with the gamma 0.5 weighting | 0.612 | 0.722 | 780 | dropped: same trade as gbm_v6_w05 |

The weighting is the direct translation of "spellF1 counts possessions, the loss counts frames",
and it did what it says: more short possessions predicted, more of the touch GT matched. But the
extra possessions are about half right, so touch spellF1 moves by less than the noise band and the
controlled GT, which has few short possessions, falls. When the two GTs disagree the brief says
trust touch; here the touch gain is itself inside the noise, so the controlled loss decides it.

## 3. Results

Commands:

```bash
PYTHONPATH=src .venv/bin/python -m mtah.models.learned --gt touch --tag gbm6_w05 gbm6_w1 gbm7 gbm7_w05 --folds 12 13 23
PYTHONPATH=src .venv/bin/python -m mtah.modelbench --models gbm_v5_nogap gbm_v6_w05 gbm_v6_w1 \
    --variants atd_cal --seeds 3 --bootstrap 300 --workers 6 --tag d_main
PYTHONPATH=src .venv/bin/python -m mtah.modelbench --models gbm_v7 gbm_v7_w05 --variants atd_cal \
    --seeds 3 --bootstrap 300 --workers 6 --tag d_gbm7
runs/phaseD/sweep.sh            # the 8 other calibrated variants, one at a time, bootstrap 0
PYTHONPATH=src .venv/bin/python -m mtah.report --main atd_cal runs/phaseC/sweep_*.json \
    runs/phaseC/d_main.json runs/phaseC/d_gbm7.json runs/phaseC/d_sweep_*.json
```

### Main table (`atd_cal`, mean of 3 seeds, bootstrap 5-95 on seed 0)

| model | touch spellF1 [boot 5-95] (seeds min-max) | touch macroF1 | controlled spellF1 [boot] | controlled macroF1 | pred poss (touch GT 1146, controlled 697) | dead F1 | touch spellF1 game 1 / 2 / 3 |
|---|---|---|---|---|---|---|---|
| smooth50 | 0.262 [0.244, 0.277] (0.258-0.269) | 0.595 | 0.270 [0.253, 0.286] | 0.598 | 1465 | 0.548 | 0.281 / 0.263 / 0.244 |
| gbm_v4 | 0.499 [0.471, 0.525] (0.496-0.501) | 0.863 | 0.604 [0.581, 0.630] | 0.873 | 766 | 0.911 | 0.528 / 0.529 / 0.444 |
| gbm_v5_nogap | 0.588 [0.558, 0.613] (0.582-0.595) | 0.908 | 0.734 [0.715, 0.763] | 0.918 | 723 | 0.947 | 0.599 / 0.601 / 0.565 |
| gbm_v6_w05 | 0.603 [0.577, 0.627] (0.598-0.607) | 0.908 | 0.714 [0.695, 0.735] | 0.916 | 779 | 0.947 | 0.623 / 0.618 / 0.570 |
| gbm_v6_w1 | 0.602 [0.564, 0.614] (0.587-0.618) | 0.905 | 0.686 [0.667, 0.706] | 0.910 | 841 | 0.947 | 0.629 / 0.617 / 0.564 |
| gbm_v7 | 0.606 [0.572, 0.623] (0.594-0.615) | 0.911 | 0.743 [0.727, 0.770] | 0.921 | 738 | 0.948 | 0.616 / 0.622 / 0.581 |
| gbm_v7_w05 | 0.612 [0.581, 0.634] (0.604-0.625) | 0.911 | 0.722 [0.713, 0.753] | 0.918 | 780 | 0.948 | 0.627 / 0.631 / 0.583 |

Per seed, touch spellF1 (predicted possessions): gbm_v5_nogap 0.582 (717) / 0.586 (722) / 0.595
(729); gbm_v7 0.594 (741) / 0.607 (720) / 0.615 (753); gbm_v7_w05 0.604 (780) / 0.609 (769) /
0.625 (790). The touch GT has 1,146 possessions on every seed.

### Robustness sweep (mean of 3 seeds; gbm_v5_nogap's columns are Phase B2's runs)

pooled spellF1, GT touch:

| model | atd_cal | cal_miss35 | cal_miss47 | cal_noctx | cal_noisy | cal_nomass | cal_restart60 | cal_vis15 | cal_vis50 | min | max |
|---|---|---|---|---|---|---|---|---|---|---|---|
| smooth50 | 0.262 | 0.272 | 0.236 | 0.263 | 0.240 | 0.244 | 0.261 | 0.265 | 0.225 | 0.225 | 0.272 |
| gbm_v5_nogap | 0.588 | 0.591 | 0.581 | 0.588 | 0.567 | 0.589 | 0.595 | 0.582 | 0.586 | 0.567 | 0.595 |
| gbm_v7 | 0.606 | 0.609 | 0.593 | 0.605 | 0.577 | 0.610 | 0.610 | 0.599 | 0.601 | 0.577 | 0.610 |
| gbm_v7_w05 | 0.612 | 0.617 | 0.603 | 0.612 | 0.585 | 0.617 | 0.614 | 0.606 | 0.611 | 0.585 | 0.617 |

pooled spellF1, GT controlled:

| model | atd_cal | cal_miss35 | cal_miss47 | cal_noctx | cal_noisy | cal_nomass | cal_restart60 | cal_vis15 | cal_vis50 | min | max |
|---|---|---|---|---|---|---|---|---|---|---|---|
| smooth50 | 0.270 | 0.281 | 0.240 | 0.271 | 0.242 | 0.248 | 0.265 | 0.269 | 0.227 | 0.227 | 0.281 |
| gbm_v5_nogap | 0.734 | 0.738 | 0.726 | 0.734 | 0.705 | 0.732 | 0.743 | 0.727 | 0.732 | 0.705 | 0.743 |
| gbm_v7 | 0.743 | 0.746 | 0.734 | 0.741 | 0.708 | 0.741 | 0.747 | 0.738 | 0.743 | 0.708 | 0.747 |
| gbm_v7_w05 | 0.722 | 0.726 | 0.713 | 0.722 | 0.693 | 0.726 | 0.724 | 0.716 | 0.724 | 0.693 | 0.726 |

pooled macroF1, GT touch: gbm_v5_nogap 0.902-0.909, gbm_v7 0.904-0.914, gbm_v7_w05 0.904-0.913.
Dead F1: 0.944-0.949, 0.945-0.951, 0.945-0.950. The dead-ball side does not move.

Reading it:

- gbm_v7 is ahead of gbm_v5_nogap in all 9 columns of both GTs. The touch lead is +0.010 to
  +0.021 and the controlled lead +0.002 to +0.009. Each is inside the noise band; the consistency
  is the only argument for it, and the columns share training data and sample games, so they are
  not 9 independent tests.
- gbm_v7_w05 has the highest touch number in every column and the lowest controlled number of the
  three in every column. Same trade as gbm_v6.
- Held-out settings (cal_miss35, cal_miss47, cal_nomass): the same order.

### Where gbm_v7's gain comes from (`segdiag --tag gbm7`)

Matched at IoU 0.5: 0.505 of GT possessions (gbm_v5_nogap 0.487). By length: under 1 s 0.021
(0.012), 1-3 s 0.189 (0.157), 3-6 s 0.523 (0.502), 6-12 s 0.736 (0.715), 12-30 s 0.854 (0.851), 30 s
and over 0.968 (0.959). A-B-A possessions: 0.443 (0.422). 84% of the remaining misses are still not
matched by any decoding of the probabilities: the kink features help the emissions a little, they
do not solve them.

## 4. Real-match diagnostics (label-free)

`PYTHONPATH=src .venv/bin/python -m mtah.diagnostics --models gbm_v5_nogap gbm_v7 gbm_v7_w05 --out
runs/phaseD`. Sanity checks, not scores; no label of BAR-ATH is read.

| statistic | bench GT touch | bench GT controlled | gbm_v5_nogap | gbm_v7 | gbm_v7_w05 |
|---|---|---|---|---|---|
| D share (graded) | 0.389 | 0.389 | 0.431 | 0.428 | 0.427 |
| D share, ball missing | - | - | 0.830 | 0.832 | 0.832 |
| D share, ball visible | - | - | 0.157 | 0.151 | 0.149 |
| stoppages per hour | 63 | 63 | 66 | 63 | 66 |
| stoppage median (s) | 16.9 | 16.9 | 17.8 | 18.2 | 17.2 |
| possessions per hour | 243 | 148 | 147 | 150 | 160 |
| possession p25 / p50 / p75 (s) | 1.8 / 4.4 / 12.5 | 4.8 / 10.3 / 19.9 | 3.7 / 7.5 / 21.0 | 3.5 / 7.6 / 20.2 | 3.3 / 7.1 / 18.1 |
| possessions under 3 s | 0.37 | 0.11 | 0.18 | 0.20 | 0.21 |
| home share of H + A | 0.516 | 0.521 | 0.615 | 0.617 | 0.609 |
| agreement with Smart Tagging attacking-phase team | - | - | 0.779 | 0.771 | 0.757 |

Nothing blows up. gbm_v7 predicts 150 possessions per hour on the real match (gbm_v5_nogap 147)
and keeps the D share, stoppage count and home share. On the bench it predicts 738 against 723, so
the real match moves the same small amount. The weighted model goes to 160 per hour, still near the
controlled GT's rate and far below the touch GT's 243. Smart Tagging agreement falls slightly with
each step towards shorter possessions (0.779, 0.771, 0.757); short possessions inside a tagged
attacking phase are by construction against the tagged team, so this is expected and is not a
score. The speed-mixture estimate of the dead share inside real gaps is unchanged (0.91).

## 5. Current best and its commands

**Current best for the submission: unchanged, `gbm_v5_nogap`** (`runs/submission/MurrayInsights.csv`,
sha256 2a7c130b..., untouched). Its commands are in `docs/phaseB2_report.md`.

**Bench leader, not shipped: `gbm_v7`**, tag gbm7: gbm_v5_nogap's features plus `features_d`,
trained on the same `atd_cal` mixture and rows, unweighted, decoded by the same three-state Viterbi
(alpha 0.5, H<->A cost 8, alive<->D cost 12).

```bash
# train (the gbm7 training cache adds the features_d columns to the cached b2 parts:
# about 10 min to build, then 13-18 min per fit on this machine under load)
PYTHONPATH=src .venv/bin/python -m mtah.models.learned --gt touch --tag gbm7 --workers 6

# labels for the competition ATD (written under a separate name; runs validate_submission)
PYTHONPATH=src .venv/bin/python -m mtah.predict --model gbm_v7 --competition \
    --out runs/submission/MurrayInsights_gbm_v7.csv
```

- `runs/submission/MurrayInsights_gbm_v7.csv`: 199,319 rows, H 55,993 / A 34,822 / D 68,042 /
  X 40,462 (X outside graded time); `validate_submission: OK`.
- A second run in a separate process (`--out runs/phaseD/...`) gave a byte-identical file,
  SHA-256 `94b5a7196096bf1e44cdf3bab586cf8137c4e3c72b7c61fd0429a863ffcee4d7`.
- `runs/submission/MurrayInsights.csv` (gbm_v5_nogap) is unchanged: SHA-256 `2a7c130b...`.
- To ship gbm_v7 instead, rerun the predict command with `--out runs/submission/MurrayInsights.csv`
  (the competition model, trained on all three games, is `runs/models/gbm7_touch_123_v4b21d1.pkl`).

## 6. Public window, once

Run once, after the tables above and the ship decision were written, with the organiser's scorer
(`cd ref/evaluation && python run_evaluation.py --results ../../runs/phaseD/publiccheck --gt
public`) on the gbm_v7 candidate file. **spellF1 0.621, macroF1 0.886** (12 possessions against 17;
alive/dead F1 0.965; home share 0.556 against 0.573). gbm_v5_nogap scored 0.619 / 0.898 there in
Phase B2. This confirms the gbm_v7 file is aligned (frames, teams, dead ball). It is 17
possessions and says nothing about which model is better; nothing was changed after it.

## Found, not fixed

1. **Short possessions are still mostly missed.** gbm_v7 matches 2% of GT possessions under 1 s
   and 19% of 1-3 s ones; it predicts 738 possessions against the touch GT's 1,146. The diagnosis
   says the remaining misses are in the emissions (84% not matched by any decoding). The next lever
   is better touch evidence: a learned touch / hand-over detector trained on the bench's event
   frames (rather than kink features fed to a frame classifier), or the video ball, which is the
   only source of new information about touches inside the ATD's gaps.
2. **The bench's touch evidence depends on the degrader's ball error model.** The kink features
   read the ball's velocity change, and `atd_cal`'s ball error is a 9-frame mean plus an AR(1)
   drift fitted to the real ATD's jitter (0.084 m against 0.047 m real; Phase B2 table). A real
   ATD ball that is smoothed differently around touches would change what a kink looks like. No
   label-free check of kink distributions on the real ATD was run; that is the first thing to do
   before shipping gbm_v7.
3. **The noise band decides the ship call, and it is a rule of thumb.** gbm_v7 is ahead in every
   one of 27 seed x setting cells we can look at, by less than the band. A paired test over GT
   possessions (the same bootstrap, both models scored on each resample) would say whether that
   consistency is real; it was not run.
4. **Feature-cache race fixed in passing.** Two bench workers building the same table's feature
   cache wrote to one temporary file name, and one crashed (`cal_vis15` on the first sweep run).
   The temporary name now carries the process id. That variant was rerun; no result was lost or
   mixed.
5. **Training-cache dependency.** The gbm7 training parts are built from the cached gbm5 (b2)
   parts plus newly computed `features_d` columns. A check that the shortcut equals a full
   recompute passed on one part (192 columns, all equal), and every part checks its labels line
   up, but a fresh clone builds the full features (slower, same numbers).
6. **Smart Tagging agreement falls slightly** as predicted possessions get shorter (0.779, 0.771,
   0.757). Expected, not a score, but worth watching if a later model goes much further.

## Tests

- `tests/test_learned.py`: possession lengths cut where the scorer cuts; the row weights favour
  short possessions while keeping the class balance (gamma 0 is unweighted, under 1 s is clipped);
  **the weight check refuses rows that do not line up** with the cached labels (a shifted label
  array and a length mismatch both raise).
- `tests/test_features_d.py`: a deflection is a kink and a roll past a player is not (both are
  "touches" to the old features); a team swap flips the signed columns and swaps the per-team
  ones; a missing ball gives no kink and no touch.
- `tests/test_models.py`: the per-frame switching cost moves a switch to the frame where it is
  cheap, and a wrong cost shape or posteriors with a per-frame cost raise.

Full suite: `.venv/bin/python -m pytest` 142 passed (135 before this phase, 7 new).

## Review addendum (coordinator, 2026-11-05): gbm_v7 not shipped

**Paired differences (gbm_v7 minus gbm_v5_nogap, spellF1).** Taken from the per-run result files over the 27 seed x variant cells.

| GT | pooled games: cells positive | min | mean |
|---|---|---|---|
| touch | 27 of 27 | +0.008 | +0.016 |
| controlled | 21 of 27 | -0.003 | +0.008 |

On controlled, game 1 is a coin flip (14 of 27 positive, mean -0.001). The lead is consistent but small.

**Touch-kink transfer check (label-free).** Graded frames with the ball visible, real ATD against `atd_cal` seed 0:

| | kink4 p25 / p50 / p75 / p90 / p99 (m/s) | kink4 on touch frames, p50 / p90 | strong-touch events per minute |
|---|---|---|---|
| real BAR-ATH | 0.35 / 0.91 / 1.87 / 2.89 / 5.92 | 0.85 / 2.60 | 9.4 |
| atd_cal g1-g3 | 0.21-0.23 / 0.35-0.39 / 0.71-1.15 / 4.06-4.52 / 9.5-10.9 | 0.41-0.50 / 4.6-5.3 | 13.5-16.4 |

The real ball track has more low-level velocity noise and a compressed upper tail, which suggests heavier smoothing. Its touches deflect it less sharply, and its strong touches fire about 40% less often. gbm7's kink features would run outside their training distribution. With the bench gain inside the noise band, the submission stays gbm_v5_nogap.

**Before kink features can ship:** add the kink distribution to `atd_profile.check_calibration`, with a test showing it fails on the current degrader. Then fit the degrader's ball-error model (smoothing plus jitter) to match it, retrain gbm7, and re-run the protocol.
