# Phase C report: ATD loader and possession models

Date: 2026-11-03. Everything here runs on Metrica's public sample games 1-3 through the Phase A
bench. No BAR-ATH data was used or looked for, and the public 5-minute window was not used at all.
Every version and its scores are in `docs/model_log.md`; this report is the summary.

## Summary

- **ATD loader done** (`src/mtah/atd_io.py`). It reads a FIFA/EPTS pair into a FrameTable, matches
  kloppy value for value on sample game 3 (143,761 frames, 35 slots, 11 format specifications),
  and is about 1 s per 140k frames instead of kloppy's minutes. It follows `MTAH_DATA_DIR` and
  the organiser's `games.py` paths.
- **Current best: `gbm_v4`**, gradient-boosted per-frame emissions decoded by a three-state
  Viterbi. On the default degrader it scores pooled spellF1 **0.660 [0.641, 0.695] (touch GT) and
  0.790 [0.784, 0.820] (controlled GT)**, macroF1 0.934 / 0.942, against smooth50's 0.292 / 0.283
  (macroF1 0.707 / 0.710). Its lead holds in every degrader setting, every game and both GTs.
- **Best without learning: `seg_v4`**: 0.482 / 0.601 spellF1. **Dead-ball detector alone
  (`dead_v3`)**: alive/dead F1 0.877 for the dead class (0.85-0.89 across settings); the learned
  model's own dead F1 is 0.96-0.97.
- The bench is easier than the real match: smooth50 scores 0.292 here against the published 0.2433.
  Expect real scores well below these. The differences between models are what transfers.

## Task 1: ATD loader

`atd_io.load_epts(meta_xml, raw_txt)` and `atd_io.load_competition()`.

| what | how |
|---|---|
| frame index | `frame = frame_id - 1` (DATA.md). A missing record becomes an all-NaN row (counted in `meta["missing_records"]`). For a duplicated frame id the first record is kept (counted). |
| layout | parsed from the metadata's DataFormatSpecification registers, one spec per frame range, not assumed. The number of fields is checked on every line, because pandas would silently pad a short line with NaN (a test feeds it a short line). |
| slots | one per `<Player>` in file order, `slot_team` 0 for the first `<Team>` (FIFA/EPTS home, as the organiser baseline and kloppy assume). `player_id` stays None; slot names and shirt numbers go to `meta` for inspection only. |
| ball | x, y; NaN when missing or half-missing. A z channel would be ignored. |
| speed | `meta["slot_speed"]` / `meta["ball_speed"]` when a channel is named s / speed / v / velocity (DATA.md lists `1_s`). The models do not use it; they compute speeds from positions on every source alike. |
| period | metadata half parameters when present, or the `halves` argument. `load_competition` passes `games.half_video_blocks()` (published boundaries), so `period > 0` is exactly graded time. |
| pitch | metadata FieldSize, or `games.py`'s 105.3 x 68.0 m in `load_competition`. |
| bundle paths | the organiser's `ref/games.py asset_paths()` under the `root` argument > `$MTAH_DATA_DIR` > `<project>/data`. |
| strict checks (`load_competition`) | 199,319 records, contiguous ids from 1, every record covered by a spec, and Barcelona not in the away slot. Each check raises rather than corrects, as the organiser's `load_atd.py` asks. |

What might differ on the real file, and how it is handled:

- 46 slots named `Track k`, with ShirtNumber equal to the slot index. These are read as
  anonymous slots, and `meta["shirt_is_slot_index"]` records it. Nothing reads shirt numbers.
- No player ids. Nothing needs them, and `player_id` is None.
- A speed channel. It is parsed if named as above. **A speed channel under another name is
  ignored silently.** That is harmless now, because nothing uses speed, but the profile should
  report it.
- Players and ball as `x,y,speed` groups. The register parse handles any group shape the
  metadata declares.
- Several DataFormatSpecifications. They are handled, and sample game 3 has 11.
- Half parameters in the XML on another counter (`games.py` mentions an IdFrame counter).
  `load_competition` ignores them and uses the published video-frame blocks.
- `atd_io.summary(table)` profiles a loaded file (ball visibility, objects per graded frame,
  frames over 22, gap runs) without labels. Run it first when the data lands.

Tests: `tests/test_atd_io.py` (8). They check the frame offset (a row-shifted reader fails), the
NaN patterns per slot, the speed channel, the halves, spec switching, missing and duplicate
records, a short line raising, `MTAH_DATA_DIR` paths, the strict check refusing a 200-record file
and Barcelona as away, and the exact kloppy match on sample game 3.

## Task 2: models

All in `src/mtah/models/`. Each one is `FrameTable (+ optional intervals) -> labels`, registered in
`src/mtah/registry.py`. All are tracking-only: ball position or its absence, anonymous slot
positions and slot teams.

1. **Dead-ball detector** (`deadball.py`). It sums a per-frame log-likelihood ratio from these
   cues: a whole-pitch speed collapse, the ball off the pitch, the ball still, the ball fast
   (alive), long ball gaps, the kick-off layout, a player off the pitch, and optional set-piece
   intervals. A two-state Viterbi then joins the frames into stoppages. v3 scales speeds per game
   from the 5th to the 75th percentile, so position noise and dead share do not move the
   threshold.
2. **Segment model** (`segment.py`, `hmm.py`). A Viterbi over {H, A, D}. The dead-ball LLR is the
   D emission. Team evidence comes only from touch-like frames: the ball within 1 m of the nearest
   player. Within 1 m the nearest team is right 95% of the time, and beyond 1 m it is near
   chance. Switching costs control fragmentation. The optional X is forward-backward abstention,
   inside decoded runs only.
3. **Learned emissions** (`learned.py`). HistGradientBoosting P(H, A, D) per frame from 73 tracking
   features: windowed touch counts, last and next touch, ball speed, ball position
   relative to home's attack, team shape, range-scaled motion and restart shapes. It is decoded
   as alpha · log(P / prior) by the same Viterbi.
   - Training data: degraded sample games only, never clean tracking. Each table uses one of six
     degrader settings, with two seeds, a randomised dead-ball path and the ball rollout on half
     of them.
   - Validation: leave one game out.
   - Symmetry: every table is also used with the teams swapped, and the two readings are
     averaged at prediction, so the model is exactly symmetric in home and away.
   - Settings: alpha 0.5, H<->A cost 8, alive<->D cost 12, chosen on a coarse grid.

### Results (pooled games 1-3, mean of 3 degrader seeds, bootstrap 5-95 on seed 0)

| model | touch spellF1 | touch macroF1 | controlled spellF1 | controlled macroF1 | possessions (GT touch 1146 / controlled 697) | dead F1 |
|---|---|---|---|---|---|---|
| smooth50 (organiser baseline) | 0.292 [0.285, 0.315] | 0.707 | 0.283 [0.282, 0.308] | 0.710 | 1817 | 0.687 |
| dead_v3 + smooth50 (detector alone) | 0.335 [0.322, 0.356] | 0.802 | 0.327 [0.324, 0.353] | 0.804 | 1681 | 0.877 |
| seg_v4 | 0.482 [0.454, 0.503] | 0.840 | 0.601 [0.584, 0.628] | 0.851 | 850 | 0.868 |
| gbm_v3_ctrl (trained on controlled) | 0.599 [0.584, 0.639] | 0.928 | 0.823 [0.807, 0.848] | 0.948 | 677 | 0.971 |
| **gbm_v4** | **0.660 [0.641, 0.695]** | **0.934** | **0.790 [0.784, 0.820]** | **0.942** | 785 | 0.969 |

Robustness: pooled touch / controlled spellF1, min-max over 9 degrader settings. The settings
are the default; dead-ball visibility 0 and 0.5; the dead ball hidden in bursts; short and long
ball bursts; a random dead-ball path; 0.4 m / 0.6 m position noise; and the ball rolling on after
play stops. Stress settings outside the training range come after.

| model | touch min-max | controlled min-max | dead F1 min-max | dead-ball visibility 0.75 (touch / controlled) |
|---|---|---|---|---|
| smooth50 | 0.252-0.344 | 0.241-0.347 | 0.115-0.864 | 0.216 / 0.199 |
| dead_v1 + smooth50 | 0.285-0.363 | 0.276-0.364 | 0.645-0.906 | |
| dead_v3 + smooth50 | 0.311-0.365 | 0.298-0.367 | 0.851-0.887 | |
| seg_v4 | 0.437-0.508 | 0.544-0.621 | 0.847-0.891 | 0.437 / 0.570 |
| gbm_v4 | 0.621-0.670 | 0.744-0.799 | 0.960-0.971 | 0.634 / 0.773 |

Per game, default degrader, seed 0, gbm_v4 touch spellF1 0.660 [0.615, 0.706] / 0.680 [0.634,
0.728] / 0.663 [0.615, 0.707], and seg_v4 0.506 / 0.470 / 0.455. The full per-setting tables,
macroF1 and dead F1 for all nine finalists are in `docs/model_log.md`.

How the versions were chosen. A lead counted only if it held across settings and exceeded about
0.03-0.04. When the GTs disagreed, touch was trusted.

- gbm_v1, gbm_v3 and gbm_v4 are within 0.011 touch spellF1 of each other everywhere, so the
  choice among them is by assumptions. v4 uses no ball-visibility features (an ablation showed
  they add nothing), scales motion by the game's own range, and trains with the ball rollout.
- Training on the controlled GT moves 0.064 touch spellF1 the wrong way, so the model trains on
  touch.
- Abstention had no effect on macroF1 (0.934 vs 0.934 / 0.933), because the posteriors are near
  1, and spellF1 does not change by construction. It is off.

### Current best and the command that produces its labels

```bash
# train once: the 3 leave-one-game-out folds and the all-games model (about 9 min on 6 cores;
# models go to runs/models/, training caches to runs/cache/train/)
PYTHONPATH=src .venv/bin/python -m mtah.models.learned --gt touch --tag gbm4 --workers 6

# labels for the competition ATD (once the bundle is in data/ or $MTAH_DATA_DIR); runs the
# organiser's validate_submission on the file it writes
PYTHONPATH=src .venv/bin/python -m mtah.predict --model gbm_v4 --competition \
    --out runs/submission/MurrayInsights.csv

# labels for a bench table (sample game, degrader setting, seed)
PYTHONPATH=src .venv/bin/python -m mtah.predict --model gbm_v4 --sample-game 3 --variant default \
    --seed 0 --out runs/phaseC/gbm_v4_g3.csv
```

In Python: `from mtah import registry; labels = registry.get("gbm_v4")(table)`, where `table` is any
FrameTable with `period > 0` on graded frames.
- A degraded sample-game table gets its leave-one-game-out fold.
- Any other table (BAR-ATH) gets `runs/models/gbm4_touch_123_v4.pkl`, which is trained on first
  use if it is missing.

**Reproducibility.** Labels are byte-identical across separate processes, with or without caches
and with `OMP_NUM_THREADS` 1 or 8 (gbm_v4 on sample game 3, rollout setting, seed 7: SHA-256
`4c4fc8da...` three times).
- A retrained model gave bit-identical probabilities: gbm1 trained twice in separate runs, with a
  maximum absolute difference of 0.0.
- Training and prediction run on one thread (threadpoolctl), the GBM uses no early stopping, and
  every degrader stream is seeded.

## What remains to calibrate once the real data arrives (distributions only, no labels)

1. **Ball-gap lengths.** `atd_io.ball_gap_lengths(table)` goes into
   `DegradeConfig(ball_burst_lengths=...)`. Keep the gaps inside stretches the tracking-only
   detector calls alive, so dead-ball gaps do not inflate the live bursts. Then re-run the
   robustness set and retrain gbm_v4 on the calibrated mixture.
2. **Dead-ball ball visibility.** Measure ball visibility inside stretches dead_v3 calls dead,
   and in the seconds before Smart Tagging set-piece starts. This replaces Phase A's coarse 0.25.
   gbm_v4 moved only 0.634-0.670 touch spellF1 over visibility 0-0.75, so this matters more for
   the heuristic models and for macroF1.
3. **Position noise.** Compare slot and ball jitter (high-frequency residuals) in the ATD with
   the degrader's 0.15 / 0.20 m. Heavy noise is the setting that costs most (-0.04). The 1 m
   touch radius in seg_v4, and the GBM's distance features, depend on it. Ball height projected
   onto the pitch (lofted balls) is a noise source the degrader does not model.
4. **Smart Tagging.** The models take it as optional `intervals` (`src/mtah/smart_tagging.py`
   loads it). The sample games have none, so its weights are unscored guesses: set-piece
   evidence before a tag's start in the dead-ball detector, and a team push in the segment model.
   The GBM ignores it.
   - Before any use, check label-free agreement on BAR-ATH: do tagged set pieces sit on restarts
     the detector finds?
   - If they do, add tag evidence to the GBM's D emission at decoding with a small fixed weight.
   - Never tune it on the public window.
5. **Ghost balls, wrong-team slots, referee tracks.** These are still absent from the degrader
   (Phase A found-not-fixed 4). Count implausible ball jumps and slot-team oddities on the ATD,
   and add a degrader variant if they are common.
6. **Phase B.** Run `atd_io.summary()` on the bundle. Reproduce `baseline_smooth50.csv` byte for
   byte from `atd_io` plus `baseline.py` (a check of the loader's slot order and team mapping).
   Run gbm_v4 on the public window as a correctness check only (frames aligned, file well-formed).

## Found, not fixed

1. **Restart boundaries are optimistic on the bench.** gbm_v4 ends stoppages within a frame of the
   restart (median 0, IQR 0-1 on the rollout setting), because the synthetic dead ball is placed,
   still and visible until the real ball is kicked. Real throw-ins (ball in hands) or a ball
   hidden by the taker will be later. Starts are realistic since the rollout change: a median
   1 frame late for ball-outs (IQR -3 to 3, the ball seen crossing the line) and 8 for fouls
   (IQR 0-16).
2. **Too few possessions against touch.** Predicted possessions are 785 against 1,146 (touch) and
   697 (controlled): the model merges short touch-possessions. Lowering the H<->A cost raises the
   count but did not raise touch spellF1 on the grid. The organiser's truth has a median near 5 s
   (Phase A), so this is where a better hand-over model would pay off.
3. **Only three training games, one provider.** Leave-one-game-out has three folds, and
   per-game spread is 0.02-0.05. BAR-ATH's style, tracker and camera are different, which is
   untested.
4. **The degrader is the only source of realism for the learned model.** The no-visibility
   ablation and the rollout and stress settings show the lead is not a single degrader artefact,
   but every setting shares the same slot model and the same ball-noise model.
5. **dead_v2's failure is composition-dependence.** Median / IQR scaling calls a long slow
   build-up dead. The only tripwire for v3 is a synthetic test (`tests/test_models.py`); real slow
   phases may still sit near v3's 0.45 threshold.
6. **Smart Tagging hooks are untested on real tags** (see calibration 4). `smart_tagging.load`
   maps `Atletic Club` through an alias, and an unmatched team value gives None.
7. **A speed channel under an unexpected name would be dropped silently** by the ATD loader. It
   is unused today.
8. **Reproducing from scratch needs the sample-data fetch** (external/, gitignored) and the
   organiser clone (ref/, gitignored). The fetch script is not written yet (Phase D). The pinned
   versions that produced these numbers are Python 3.14, numpy 2.5.4, scikit-learn 1.9.1, scipy
   1.18.1, pandas 3.1.0, and kloppy 3.18.0 for sample game 3.
9. **Earlier exploration rows in the model log use 1-2 seeds without bootstrap.** Only the final
   table is the full protocol.

## Tests added

`.venv/bin/python -m pytest`: 91 tests, all passing. Each test that guards a decision can fail:

- `test_atd_io.py` (8): see above.
- `test_models.py` (16): Viterbi equals brute force on random problems. Forward-backward
  posteriors follow the evidence. **The switching cost reduces the predicted possession count on
  a noisy toy sequence** (above 200 runs at cost 0.2, at most 60 at cost 16, 40 true). Decoding
  stays inside graded runs. Abstention never lands on a run's first frame, and forward-filling it
  restores the labels.
  - **The dead-ball detector flags a synthetic out-of-play sequence (above 80%) and not a slow
    build-up (below 10%)**, for v1 and v3. A test documents v2 failing on the same build-up.
  - The ball-out cue raises dead evidence. A set-piece interval adds evidence only before its
    start.
  - The segment model holds one possession through a build-up, and its cost controls the count.
    The touch feature has the right sign.
- `test_learned.py` (4): leave-one-game-out fold selection, team swapping, symmetric emissions,
  feature sets per tag.
- `test_degrade.py` (+3): the random dead-ball path changes dead stretches only and leaves the
  default path untouched. Unknown paths are rejected. The rollout keeps the ball moving after
  play stops and leaves live frames alone.
- `test_smart_tagging.py` (2), `test_predict.py` (1): the time base and team mapping (N/A becomes
  None), and the CLI writes a file the organiser's `validate_submission` accepts, with X outside
  graded time.

## Reproduce

```bash
PYTHONPATH=src .venv/bin/python -m mtah.models.learned --gt touch --tag gbm1 gbm1_novis gbm2 gbm3 gbm4
PYTHONPATH=src .venv/bin/python -m mtah.models.learned --gt controlled --tag gbm3
PYTHONPATH=src .venv/bin/python -m mtah.modelbench --models smooth50 dead_v1_s50 dead_v3_s50 seg_v1 \
    seg_v4 gbm_v1 gbm_v3 gbm_v3_ctrl gbm_v4 --robust --seeds 3 --bootstrap 300 --workers 6 --tag final
PYTHONPATH=src .venv/bin/python -m mtah.report runs/phaseC/final.json
```

The final run takes about 42 minutes on 6 cores with cold caches, and training takes about 9
minutes per tag.
