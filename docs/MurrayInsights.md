# Murray Insights: team possession, frame by frame, from automatic tracking

More Than a Hack 2027, Initial Challenge. Team: Murray Insights.
Code: <https://github.com/Murray22/murray-insights-possession>

## 1. What we submitted

Every one of the 199,319 frames is labelled `H`, `A` or `D` inside graded time, and `X` outside it.

The file comes from `gbm_v5_nogap`:
- a gradient-boosted classifier gives each frame a probability of home possession, away possession and dead ball;
- a three-state Viterbi decoder turns those probabilities into possessions.

It reads only the tactical automatic tracking (ATD). No person labelled or corrected anything in this match.

## 2. What the real data looks like

We measured the competition ATD as distributions only. No label and no individual frame was used for calibration.
- **The ball is missing on 41% of graded frames.** It goes missing in long blocks, not brief flickers: 150 missing
  runs, median 268 frames (about 11 s), 90th percentile 845 frames.
- **The ball tends to vanish when it is slow, near a line and near players,** which is what a ball going out of play
  looks like. A ball under 2 m/s is 2.35 times as likely to drop out as an average visible frame. One within 3 m of a
  touchline or goal line is 2.53 times as likely.
- **When the ball comes back, it is usually near where it vanished** (median 10.2 m away).
- **Player tracking is smooth** (jitter 0.007 m), with long tracklets.
- **Sometimes most of the players drop out together.** On 4.4% of frames fewer than 14 objects are tracked, in runs
  with a median of 67 frames.

So the hard part is not the visible-ball frames. It is deciding who has the ball, and whether play is alive at all,
through long stretches where the ball isn't there.

## 3. Approach, and why

**No leaderboard, so we built our own bench.** The scorer gives no feedback before the deadline, so every choice was
made on Metrica Sports' three public sample games. Those games come with event data, from which possession labels can
be derived. We degraded their clean tracking until it looked like the competition ATD: the same ball-gap lengths,
the same contexts for gaps starting, the same smoothing, mass dropouts and tracklet behaviour. Then we scored
models with the organiser's own `possession_metrics.py`, leaving one game out each time, over three degradation seeds,
with the organiser's block bootstrap.

**Calibrating the bench mattered more than any model change.** Our first bench hid the ball in short bursts. On the
real match, the model trained on it called 89% of missing-ball frames dead: it had learned that "long gap = stoppage".
After calibration:
- the organiser's `smooth50` baseline scores 0.262 spellF1 on our bench, against its published full-match 0.2433;
- the uncalibrated bench gave it 0.292.

**The model.** Per frame, a histogram gradient-boosting classifier (scikit-learn, 300 iterations) estimates
P(home), P(away) and P(dead) from 118 tracking features:
- the ball's speed and position relative to home's attack;
- the nearest player of each team, and the margin between them;
- recent and upcoming touches;
- each team's shape (centroid, length, width, compactness, last line, share in the opponent's half), and the speed
  and direction of its movement;
- whole-pitch movement, scaled per game;
- restart layouts.

A three-state Viterbi decoder works on `0.5 * log(P / prior)`, with a cost of 8 to switch between home and away and
12 to switch between alive and dead. The model is trained with each table as is and with the teams swapped, and the
two readings are averaged at prediction time, so it treats home and away exactly alike.

**Why this and not something else:**

| approach | touch spellF1 on the calibrated bench |
|---|---|
| organiser baseline (`smooth50`) | 0.262 |
| our best hand-built rules (dead-ball detector plus nearest-player evidence at touches) | 0.357 |
| gradient boosting trained on the uncalibrated bench | 0.499 |
| gradient boosting trained on the calibrated bench (submitted) | 0.588 |

Learned probabilities beat our rules by a wide margin in every setting we tried. An end-to-end model on the video
was out of reach: there are no possession labels for this match, and the rules rightly forbid making any. We used
the video for one narrow job only, finding the ball for `gbm_v8n`, which is not submitted (section 8), plus a short
pose-estimation pilot on players near the ball, for round 2; not used in any model.

## 4. Assets: what we used and what we dropped

| asset | how we used it |
|---|---|
| Tactical ATD | the only model input for the submitted file |
| Homography (tactical) | to project the ATD into the video for the video ball (`gbm_v8n` only, not submitted) and to check alignment: ATD frame id - 1 = video frame = homography row, confirmed by a paired-distance test that rejects deliberate one-frame slips |
| Tactical video | our own ball detection in the ball gaps (`gbm_v8n` only, not submitted); a short pose-estimation pilot on players near the ball, for round 2, not used in any model |
| Smart Tagging | **not a model input.** We used its 59 set-piece tags, averaged together, to measure how often the ball is visible during a stoppage (about 30%) and how soon play resumes after one. Both set values in the degrader. We also used its attacking phases as a label-free check (section 6). |
| Broadcast video | dropped, never downloaded (13 GB). The tactical camera already covers the pitch and is frame-aligned with the ATD. |
| Broadcast ATD and homography | downloaded; measured once whether its ball could fill the tactical ball gaps (+0.004 to +0.006 on the bench, dropped); not a model input |
| Public 5-minute ground truth | used once per candidate, after the choice was made, to check the pipeline end to end. Nothing was changed after looking at it. |

## 5. Training data

**Source.** Metrica Sports' public sample games 1-3 (<https://github.com/metrica-sports/sample-data>). Together
they hold 283 graded minutes, with tracking and event data.

**Who labelled it.** The event data was annotated by Metrica Sports. Our possession labels are derived from those
events by code (`src/mtah/sample_gt.py`), and nobody on our team labelled a frame. The rules are in section 7 and in
full in `docs/sample_gt_rules.md`. The labels pass a cross-check: our dead-ball frames agree with "ball absent from
Metrica's clean tracking" on 93.3%, 97.6% and 99.7% of graded frames in games 1, 2 and 3.

**How much.**
- Under the definition we train on ("touch"), 1,146 possessions; under the stricter "controlled" definition, 697.
- Each game is degraded under six settings around the calibrated one, with two seeds each, and every table is also
  used with the teams swapped. The varied settings are:
  - how often the ball is visible during stoppages;
  - heavier position noise;
  - no context weighting of gap starts;
  - more frequent re-acquisition of the ball at restarts.
- The submitted model is trained on all three games: 1,019,720 rows. For scoring, each model is trained on two games
  and tested on the third.

**Nothing from the provided match.** The competition ATD was used only as distributions (gap lengths, visibility,
jitter, tracklet lengths, object counts) to calibrate the degradation. Training and scoring use different seeds.

## 6. Missing and noisy tracking

- **Ball gaps.** The model never needs the ball to decide which team is on top. Inside a gap it reads the players:
  - each team's advance relative to its own attack, and its velocity;
  - forward running;
  - the length and width of its shape;
  - the last line;
  - how much of each team is in the opponent's half;
  - distances to the nearest opponent.

  These player features are worth +0.044 spellF1 on their own. Features describing the gap itself (its length, and
  the ball at either end) added nothing in testing, so the submitted model leaves them out.
- **Dead ball.** Stoppages are recognised from whole-pitch movement (players slow down together), the ball leaving
  the pitch or sitting still, and restart layouts. Movement is range-scaled per game, so the model assumes nothing
  about how much of a match is dead. On the real match it calls 43% of graded frames dead, 83% of frames where the
  ball is missing and 16% where it is visible. A separate label-free estimate from players' speeds puts about 91% of
  the real gaps in dead time. So this looks like a fair reading, not an over-call.
- **Noise.** The training mixture includes a heavy-noise setting (0.4 m on players, 1.0 m on the ball). On the bench
  that setting is the one that costs most, but it does not break the model.
- **Dropouts and tracklets.** The degradation removes 50-90% of tracked players together for about 3 s at a time,
  ends tracklets the way the real file does, and reuses slots. The model sees these during training.
- **Attack direction.** Several features are measured relative to home's attack. We first read the direction from
  each half's opening 10 s. The real second half doesn't open in a kick-off layout, so every one of those features
  pointed the wrong way for 45 minutes. We caught it without labels: our model's agreement with Smart Tagging's
  attacking-phase team was 0.775 in the first half and 0.628 in the second. The direction now comes from the whole half (keepers and
  last lines), and second-half agreement is 0.785. The fix leaves every sample-game half unchanged.

## 7. The possession rules we implemented

These rules label the sample games we train and score on. We never label the provided match.

- **Dead ball.** The ball is dead from a hard stoppage (it crosses a line or goes in) or a referee's stoppage until
  the restart.
  - Event data never says when a whistle went, so a referee's stoppage starts at the last live event before the
    set piece.
  - Goal kicks count as restarts. A goalkeeper collecting the ball after it has gone out does not.
  - A dead ball has no owner (`D`).
- **Restarts.** The team that plays the restart has the ball from the restart frame.
- **Challenges.** A duel, a challenge, a card or a foul received changes nothing on its own.
- **Loose balls.** A loose ball stays with the team that lost it until the other team plays it.
- **Where we drew the line on touches.** We built two definitions:
  - **Controlled**, our reading of CHALLENGE.md: a team takes the ball only with a completed pass, a shot, a set
    piece, or one second of holding it.
  - **Touch:** any on-ball action by the other team takes the ball, including a one-touch deflection.

  We train on touch. CHALLENGE.md says "a simple touch of the ball without control" is not a change of possession,
  but also that this definition "is not a technical specification of how the ground truth was produced", and that
  "if your reading of the definition disagrees with it, the ground truth is right". The labels are derived
  mechanically: a live frame "takes the team the tracking names as owning it", and "Nothing is smoothed,
  interpolated or hand-adjusted afterwards". CHALLENGE.md does not say how the tracking decides who owns the ball.
  Nothing between that owner and the labels removes a brief spell, so if the owner changes on a touch, so do the
  labels. Touch is the definition that allows for this; controlled is our reading of the written definition, and we
  report both. A model trained on controlled gained 0.033 spellF1 against controlled labels and lost 0.064 against
  touch labels.

  How we chose: we wrote controlled first. We added touch, and chose it, after the worked examples of shifted
  predictions in EVALUATION.md (a perfect prediction 8 frames late scores 0.8933, 1 s late 0.6539) put the real mix
  of possession lengths between our two definitions and closer to touch, and after our bench reproduced the order of
  the organiser's published full-match baseline scores only under touch. Those examples cannot come from the
  17-possession public window, so they were most likely computed on the full match. Like the baseline scores, they
  are published in EVALUATION.md for every team. The reading above is our justification now. It agrees with that
  choice, but it is not how we made it.

  We did not rewrite the controlled rules to match. Both definitions are reported throughout.

  The organisers' summary of the public window (17 possessions, mean 14.2 s) looks more like controlled than touch,
  but it is 17 possessions from the first five minutes. `docs/phaseA_report.md` prints it for comparison only;
  nothing was tuned to it.
- **`X`.** We output `X` only outside graded time. We tested abstaining (`X` where the decoder is unsure), and it
  changed nothing, because the decoder is rarely unsure.

## 8. Results, and what we tried and dropped

**On the calibrated bench** (touch spellF1, pooled over games 1-3, mean of 3 seeds, bootstrap 5-95% on seed 0):

| model | touch spellF1 | controlled spellF1 | macroF1 (touch) |
|---|---|---|---|
| `smooth50` (organiser baseline) | 0.262 [0.244, 0.277] | 0.270 | 0.595 |
| `gbm_v5_nogap` (submitted) | 0.588 [0.558, 0.613] | 0.734 | 0.908 |

`gbm_v5_nogap` stays between 0.567 and 0.595 across the nine degradation settings we scored, three of which it never
trained on.

**On the real match, without labels:**
- 147 possessions per hour, with a median length of 7.5 s;
- 43% of graded time dead;
- home in possession 61.5% of live time;
- 78% agreement with Smart Tagging's attacking-phase team.

The public five minutes, checked once: spellF1 0.619, macroF1 0.898. Five minutes from the start of a match says
little about the other ninety, so we treat it as a pipeline check, not an estimate.

**The video ball (`gbm_v8n`, not submitted).**
- **The detector.** A public ball detector (soccana, run through Ultralytics and OpenVINO) finds the ball on about
  40% of gap frames, mostly balls standing still or at a player's feet. Fast and lofted balls are not recovered.
- **Simulation.** We simulated that track on the bench. Its burst lengths and error vectors are sampled from the
  detector's track on the provided video, and its other settings (fill rate, stillness weighting, spare-ball rate,
  drift) were tuned over four rounds to match that track (fill: 0.405 of gap frames real, 0.396 simulated).
- **The model.** `gbm_v8n` is `gbm_v5_nogap` plus 25 features from the video ball, kept as a separate, flagged
  channel. It leads `gbm_v5_nogap` by +0.021 touch spellF1 in 27 of 27 bench cells (paired bootstrap
  [+0.014, +0.029]). With half the simulated video it still leads by +0.015, and with none it ties.
- **On the real match** it changes 3.3% of graded frames. Its possession count, lengths and dead share are close to
  `gbm_v5_nogap`'s.
- **Why it is not submitted.** The organisers' clarification email says parameters must be tuned on external data,
  never on the provided video. The simulation above was tuned on the provided video, so `gbm_v8n` is not eligible.
  It was also a weak case:
  - its lead is smaller than our noise band (0.03-0.04);
  - the real video track is weaker team evidence than the simulated one;
  - the submission would depend on a detector run of several hours. Unpinned, Ultralytics 8.4.202 leaves the
    device and precision to OpenVINO on Linux x86. It ran in 32-bit float here; per OpenVINO's documented defaults,
    other CPUs and GPUs may use lower precision, so the file could differ.

**Tried and dropped** (details in `docs/model_log.md`):
- Hand-built dead-ball detectors. One collapses under position noise; one assumes how much of a game is dead.
- Abstention. No effect.
- Training on the controlled definition. Costs 0.064 touch spellF1.
- Weighting short possessions in training. It trades controlled score for touch score.
- Decoder grids and touch-dependent switching costs. Flat.
- Features at the moment of a touch (ball velocity kinks). They helped by +0.018 until the bench's ball track was
  given the real track's shape, then by +0.000.
- Late experiments, 12-13 November, each against a rule written before it ran (touch spellF1 on the calibrated
  bench): learned switching costs, -0.008; a second model stacked on the first, +0.006 [-0.004, +0.015]; inferring
  the ball's position in its gaps, -0.001; lowering the switching cost only at sharp ball turns near the incoming
  team, +0.001 (controlled -0.011); filling ball gaps from the broadcast recording's tracking, +0.004 to +0.006.

## 9. Limitations, and what we would do with more time

- **Short possessions are the biggest miss.** Possessions under 3 s are 37% of the touch labels and we match few of
  them. In 85% of the misses no decoding of our probabilities finds them at all: the per-frame evidence never
  favours the team that briefly had the ball. With more time, we would build better touch evidence from the video,
  near the ball at full resolution.
- **The bench is three games from one provider.** Its players and its event coding are Metrica's. The dead-ball share
  in particular rests on sample-game players moving like this match's players. More games, especially from other
  leagues, would make every number in this document more trustworthy.
- **Some real-data patterns are not modelled.** One team shows more than 11 tracked objects on 12.9% of frames
  (referees or duplicates). Restarts on the bench are a little too easy: the synthetic ball is placed, still and
  visible, until it is kicked.
- **Stoppages with no event can't be learned.** In the sample events, an injury stoppage with no set piece or ball-out
  event can't be seen, so it is labelled alive.
- **We don't know our real score.** The bench predicts the baseline's real score reasonably (0.262 against 0.2433),
  but that is one number, and it says nothing about our model.

## 10. Reproduction

Python 3.14 for the model, CPU only. The match bundle goes in `data/` or `$MTAH_DATA_DIR`, as the organiser's
`data/README.md` describes. It is never in the repository.

```bash
python3.14 -m venv .venv && .venv/bin/pip install -r requirements.txt
scripts/fetch_external.sh        # organiser repo -> ref/, Metrica sample games -> external/ (pinned, hash-checked)
PYTHONPATH=src .venv/bin/python -m pytest -q
PYTHONPATH=src .venv/bin/python -m mtah.models.learned --gt touch --tag gbm5_nogap --workers 6
PYTHONPATH=src .venv/bin/python -m mtah.predict --model gbm_v5_nogap --competition --out runs/submission/MurrayInsights.csv
```

From a fresh clone this took 19 minutes end to end (training about 16 minutes on 6 cores). The output was
byte-identical to our submission (SHA-256 `2a7c130b...`). `mtah.predict` runs the organiser's `validate_submission`
on what it writes. The video-ball model's commands are in the README.

Any row of the model log reproduces with
`PYTHONPATH=src .venv/bin/python -m mtah.modelbench --models <name> --robust --seeds 3 --bootstrap 300`.

## 11. AI tools

We used an AI coding assistant, Anthropic's Claude through Claude Code, during development. It wrote and debugged
most of the code, drafted documentation including this document, and ran commands on our own machine: loading data,
experiments, training and evaluation. We reviewed its work and made the decisions, including which model to submit.
No language model is part of the submitted pipeline. The submission file comes entirely from the code in this
repository and the data it reads.

## Credits

- Metrica Sports, for the challenge repository and the public sample data.
- The soccana ball detector (`Adit-jain/soccana` on Hugging Face), run through Ultralytics and OpenVINO.
