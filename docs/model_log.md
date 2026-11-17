# Model log (Phase C)

Every version that was scored, what it tried, and whether it was kept. This feeds the technical
document's "tried and dropped" section. All numbers come from the Phase A bench: Metrica sample
games 1-3, degraded to look like the competition ATD, scored with the organiser's
`possession_metrics.py` against two ground truths derived from the sample games' events
("touch": any on-ball event by the other team flips possession; "controlled": the CHALLENGE.md
rules, docs/sample_gt_rules.md). Pooled over the three games. Phase A: when the two disagree,
trust "touch" (the organiser's own shift table puts their truth near it), and pooled spellF1 gaps
under about 0.03-0.04 are noise.

Nothing here touched BAR-ATH data. The public 5-minute window was not used.

Reproduce any row: `PYTHONPATH=src .venv/bin/python -m mtah.modelbench --models <name> [--robust]
--seeds 3 --bootstrap 300`. Result files go to `runs/phaseC/` (gitignored).

## Bench conventions

- Degrader seeds 0-2 for the bench (game g gets seed `100 * s + g`, as in Phase A). Learned models
  train on degrader seeds 10000+, never on bench seeds, and leave one game out.
- The table a model sees has `period = 0` outside graded time, as the real ATD will (its half
  blocks are published in `ref/games.py`). Models never see events, clean tracking or player ids.
- Robustness set (`--robust`): default; dead-ball ball visibility 0 and 0.5 (`dead_vis_*`); dead ball
  hidden in bursts; short (median 5 frames) and long (75) live ball bursts; randomised synthetic
  dead-ball path; heavy position noise (0.4 m players, 0.6 m ball, `noisy_positions`, new in
  Phase C); the ball rolling on after play stops (`rollout`, new in Phase C).

## Versions in the order they were tried

Exploration numbers are degrader "default", 1-2 seeds, no bootstrap; they were used to pick
directions, not to rank finalists. The final table below re-scores the finalists on 3 seeds with
bootstrap intervals.

| version | idea | touch spellF1 | controlled spellF1 | touch macroF1 | dead F1 | kept? |
|---|---|---|---|---|---|---|
| smooth50 | organiser baseline (nearest slot, +/-50 vote) on our FrameTable | 0.295 | 0.285 | 0.707 | 0.686 | reference |
| dead_v1 (+ smooth50 for H/A) | heuristic dead-ball LLR: whole-pitch speed collapse in m/s (p80 over +/-1.5 s, median over +/-4 s), ball off the pitch, ball still, ball fast (alive), long ball gaps, kick-off layout, player off the pitch; two-state Viterbi, switching cost 15 | 0.337 | 0.327 | 0.809 | 0.894 | replaced by v3: dead F1 falls to 0.63-0.65 with 0.4 m position noise, because noise lifts every speed |
| dead_v2 | v1 on per-game median / IQR standardised speeds | 0.336 | 0.329 | 0.804 | 0.882 (noisy 0.886) | dropped: calls a long slow build-up dead when a quarter of the sequence is slow (tests/test_models.py keeps the failure); it assumes how much of a game is dead |
| dead_v3 first try | per-game range scaling (q05 to q75) of speeds, threshold 0.33, weight 8 | 0.337 | 0.332 | 0.776 | 0.824 | re-gridded: precision 0.96, recall 0.72 |
| **dead_v3** | range scaling, threshold 0.45, weight 5 (grid {0.33, 0.4, 0.45} x {5, 8}, default / noisy / dead_vis 0.5) | 0.338 | 0.340 | 0.804 | 0.877 (noisy 0.879) | **kept as the tracking-only dead-ball detector** |
| seg_v1 | 3-state Viterbi {H, A, D}: dead LLR + tanh of the nearest-distance difference, half weight beyond 2 m, quarter weight for a fast ball; costs H<->A 10, alive<->D 12 | 0.432 | 0.532 | 0.829 | 0.893 | replaced by v2 |
| seg_v1 cost grid | H<->A {4, 8, 16, 32} x alive<->D {6, 12, 24} | 0.378-0.436 | 0.379-0.585 | | | touch flat from 8 to 16; controlled wants more cost (Phase A's warning) |
| diagnosis | oracle D into seg_v1: +0.08 touch spellF1; oracle team on alive frames: +0.43. The nearest team is right 95% of the time with the ball within 1 m of that player and 53-65% beyond 1 m | | | | | the team part is where the score is |
| seg_v2 | team evidence only on touch-like frames (ball within 1 m of the nearest player), scaled by the margin to the other team; grid w {1, 2, 4} x cost {5, 10, 20}; chose w 4, cost 10 | 0.491 | 0.605 | 0.855 | 0.892 (noisy 0.633) | replaced: v1 dead detector inside |
| seg_v3 | seg_v2 + dead_v2 | 0.483 | 0.602 | 0.843 | 0.873 | dropped with dead_v2 |
| **seg_v4** | seg_v2 + dead_v3 | 0.483 | 0.605 | 0.841 | 0.868 | **kept: best model without learning** |
| gbm_v1 (tag gbm1) | gradient boosting P(H, A, D) per frame from 79 tracking features (touch windows, last / next touch, ball speed and position relative to home's attack, team shape and width, per-game standardised motion, ball gaps, restart shapes), trained leave-one-game-out on degraded games (6 degrader settings x 2 seeds, randomised dead-ball path, every table also team-swapped), symmetrised at prediction; Viterbi on alpha * log(P / prior); grid alpha {0.25, 0.5, 1} x cost {4, 8, 16}, chose 0.5 / 8 | 0.666 | 0.789 | 0.934 | 0.967 | replaced by v3 / v4 (same skill, fewer assumptions) |
| abstention | X where the decoded label's forward-backward posterior < 0.4 or 0.6, never on a run's first frame | 0.666 | 0.789 | 0.934 / 0.933 | | dropped: no effect (the posteriors are near 1 almost everywhere); spellF1 unchanged by construction |
| gbm_v1_novis | gbm1 without any ball-visibility feature | 0.663 | 0.785 | 0.933 | 0.967 | ablation: the visibility features add nothing, so the skill does not ride on where the degrader hides the ball |
| gbm_v2 | gbm1 with range-scaled instead of median / IQR motion | 0.666 | 0.794 | 0.936 | 0.971 | merged into v3 |
| gbm_v3 | range-scaled motion, no visibility features | 0.663 (final run) | 0.790 | 0.936 | 0.971 | superseded by v4, not worse |
| gbm_v3_ctrl | gbm_v3 trained on the controlled GT | 0.599 (final run) | 0.823 | 0.928 | 0.971 | dropped: -0.064 touch, +0.033 controlled |
| rollout check | gbm_v1 on the new `rollout` degrader variant, never trained on it | 0.662 | 0.782 | 0.931 | 0.964 | the frame-exact dead starts were not the frozen synthetic ball (see below) |
| **gbm_v4** | gbm_v3 features, trained on a mixture with the ball rollout on half the settings | 0.660 (final run) | 0.790 | 0.934 | 0.969 | **kept: current best** |

### The boundary check behind `rollout`

gbm_v1's predicted stoppages start a median 0 frames from the true ones (IQR -4 to +2), which looked
too good. The suspect was the degrader: its synthetic dead ball stopped dead on the first dead
frame, a "play has stopped" signal no real tracking gives. The degrader now has
`dead_ball_rollout` (the ball rolls on from its last velocity and slows to rest over 0.5-3 s).
On rollout tables, never trained on, gbm_v1 keeps 0.662 touch spellF1 and 0.964 dead F1; its starts
are a median 4 frames late for ball-outs (IQR 0-9, the ball seen crossing the line) and 10 for
fouls (IQR 2-16). Restarts stay within a frame (median 0, IQR 0-1): the synthetic ball is placed,
still and visible, until the real ball is kicked. That part is optimistic and is listed in the
Phase C report under "found, not fixed". gbm_v4, trained with the rollout, starts ball-out
stoppages a median 1 frame late (IQR -3 to 3) and fouls 8 (IQR 0-16) on rollout tables.

## Final comparison

`PYTHONPATH=src .venv/bin/python -m mtah.modelbench --models smooth50 dead_v1_s50 dead_v3_s50 seg_v1
seg_v4 gbm_v1 gbm_v3 gbm_v3_ctrl gbm_v4 --robust --seeds 3 --bootstrap 300 --workers 6 --tag final`
(42 minutes on 6 cores; tables from `python -m mtah.report runs/phaseC/final.json`). Pooled over
games 1-3, mean over 3 degrader seeds; bootstrap = the organiser's block bootstrap over GT
possessions (300 draws, 5-95%) on seed 0; "seeds min-max" is the spread over seeds.
`dead_v1_s50` / `dead_v3_s50` are the dead-ball detectors alone (H / A from the smooth50 vote on
alive frames), so their dead F1 is the detector's own alive/dead score.

### default degrader

| model | touch spellF1 [boot 5-95] (seeds min-max) | touch macroF1 | controlled spellF1 [boot] | controlled macroF1 | pred poss (touch GT 1146, controlled 697) | dead F1 | touch spellF1 game 1 / 2 / 3 |
|---|---|---|---|---|---|---|---|
| smooth50 | 0.292 [0.285, 0.315] (0.284-0.299) | 0.707 | 0.283 [0.282, 0.308] | 0.710 | 1817 | 0.687 | 0.306 / 0.300 / 0.273 |
| dead_v1_s50 | 0.335 [0.322, 0.355] (0.332-0.339) | 0.809 | 0.324 [0.319, 0.350] | 0.811 | 1660 | 0.894 | 0.376 / 0.342 / 0.295 |
| dead_v3_s50 | 0.335 [0.322, 0.356] (0.330-0.338) | 0.802 | 0.327 [0.324, 0.353] | 0.804 | 1681 | 0.877 | 0.374 / 0.342 / 0.295 |
| seg_v1 | 0.426 [0.417, 0.463] (0.414-0.439) | 0.827 | 0.529 [0.514, 0.562] | 0.835 | 873 | 0.892 | 0.445 / 0.439 / 0.397 |
| seg_v4 | 0.482 [0.454, 0.503] (0.477-0.490) | 0.840 | 0.601 [0.584, 0.628] | 0.851 | 850 | 0.868 | 0.504 / 0.487 / 0.457 |
| gbm_v1 | 0.660 [0.646, 0.696] (0.648-0.669) | 0.933 | 0.785 [0.781, 0.815] | 0.941 | 797 | 0.967 | 0.677 / 0.687 / 0.620 |
| gbm_v3 | 0.663 [0.648, 0.699] (0.652-0.672) | 0.936 | 0.790 [0.789, 0.824] | 0.944 | 788 | 0.971 | 0.669 / 0.690 / 0.633 |
| gbm_v3_ctrl | 0.599 [0.584, 0.639] (0.588-0.610) | 0.928 | 0.823 [0.807, 0.848] | 0.948 | 677 | 0.971 | 0.607 / 0.615 / 0.578 |
| gbm_v4 | 0.660 [0.641, 0.695] (0.652-0.667) | 0.934 | 0.790 [0.784, 0.820] | 0.942 | 785 | 0.969 | 0.665 / 0.683 / 0.634 |

### pooled spellF1, GT touch, mean over seeds

| model | default | dead_vis_0.00 | dead_vis_0.50 | dead_bursts | short_ball_bursts | long_ball_bursts | random_dead_path | noisy_positions | rollout | min | max |
|---|---|---|---|---|---|---|---|---|---|---|---|
| smooth50 | 0.292 | 0.344 | 0.252 | 0.260 | 0.323 | 0.267 | 0.291 | 0.270 | 0.292 | 0.252 | 0.344 |
| dead_v1_s50 | 0.335 | 0.363 | 0.313 | 0.326 | 0.357 | 0.323 | 0.335 | 0.285 | 0.337 | 0.285 | 0.363 |
| dead_v3_s50 | 0.335 | 0.365 | 0.315 | 0.328 | 0.356 | 0.325 | 0.334 | 0.311 | 0.338 | 0.311 | 0.365 |
| seg_v1 | 0.426 | 0.455 | 0.403 | 0.424 | 0.444 | 0.408 | 0.424 | 0.339 | 0.433 | 0.339 | 0.455 |
| seg_v4 | 0.482 | 0.508 | 0.457 | 0.474 | 0.495 | 0.461 | 0.479 | 0.437 | 0.485 | 0.437 | 0.508 |
| gbm_v1 | 0.660 | 0.670 | 0.641 | 0.648 | 0.666 | 0.643 | 0.656 | 0.617 | 0.655 | 0.617 | 0.670 |
| gbm_v3 | 0.663 | 0.672 | 0.648 | 0.654 | 0.673 | 0.654 | 0.661 | 0.622 | 0.658 | 0.622 | 0.673 |
| gbm_v3_ctrl | 0.599 | 0.605 | 0.592 | 0.585 | 0.609 | 0.592 | 0.596 | 0.559 | 0.595 | 0.559 | 0.609 |
| gbm_v4 | 0.660 | 0.670 | 0.645 | 0.653 | 0.665 | 0.652 | 0.656 | 0.621 | 0.660 | 0.621 | 0.670 |

### pooled spellF1, GT controlled, mean over seeds

| model | default | dead_vis_0.00 | dead_vis_0.50 | dead_bursts | short_ball_bursts | long_ball_bursts | random_dead_path | noisy_positions | rollout | min | max |
|---|---|---|---|---|---|---|---|---|---|---|---|
| smooth50 | 0.283 | 0.347 | 0.241 | 0.255 | 0.325 | 0.261 | 0.282 | 0.258 | 0.282 | 0.241 | 0.347 |
| dead_v1_s50 | 0.324 | 0.364 | 0.302 | 0.324 | 0.357 | 0.321 | 0.324 | 0.276 | 0.326 | 0.276 | 0.364 |
| dead_v3_s50 | 0.327 | 0.367 | 0.306 | 0.329 | 0.358 | 0.325 | 0.326 | 0.298 | 0.328 | 0.298 | 0.367 |
| seg_v1 | 0.529 | 0.562 | 0.517 | 0.533 | 0.555 | 0.510 | 0.527 | 0.438 | 0.538 | 0.438 | 0.562 |
| seg_v4 | 0.601 | 0.620 | 0.581 | 0.591 | 0.621 | 0.566 | 0.598 | 0.544 | 0.608 | 0.544 | 0.621 |
| gbm_v1 | 0.785 | 0.781 | 0.771 | 0.772 | 0.798 | 0.757 | 0.781 | 0.726 | 0.779 | 0.726 | 0.798 |
| gbm_v3 | 0.790 | 0.790 | 0.776 | 0.781 | 0.795 | 0.776 | 0.788 | 0.741 | 0.783 | 0.741 | 0.795 |
| gbm_v3_ctrl | 0.823 | 0.824 | 0.817 | 0.806 | 0.836 | 0.800 | 0.818 | 0.776 | 0.819 | 0.776 | 0.836 |
| gbm_v4 | 0.790 | 0.788 | 0.780 | 0.776 | 0.799 | 0.773 | 0.787 | 0.744 | 0.787 | 0.744 | 0.799 |

### pooled macroF1, GT touch

| model | default | dead_vis_0.00 | dead_vis_0.50 | dead_bursts | short_ball_bursts | long_ball_bursts | random_dead_path | noisy_positions | rollout | min | max |
|---|---|---|---|---|---|---|---|---|---|---|---|
| smooth50 | 0.707 | 0.805 | 0.599 | 0.458 | 0.720 | 0.672 | 0.707 | 0.702 | 0.707 | 0.458 | 0.805 |
| dead_v1_s50 | 0.809 | 0.822 | 0.797 | 0.797 | 0.816 | 0.795 | 0.808 | 0.683 | 0.813 | 0.683 | 0.822 |
| dead_v3_s50 | 0.802 | 0.813 | 0.788 | 0.787 | 0.808 | 0.791 | 0.801 | 0.797 | 0.806 | 0.787 | 0.813 |
| seg_v1 | 0.827 | 0.846 | 0.812 | 0.816 | 0.833 | 0.813 | 0.825 | 0.688 | 0.833 | 0.688 | 0.846 |
| seg_v4 | 0.840 | 0.855 | 0.827 | 0.825 | 0.844 | 0.827 | 0.838 | 0.835 | 0.845 | 0.825 | 0.855 |
| gbm_v1 | 0.933 | 0.935 | 0.926 | 0.927 | 0.935 | 0.922 | 0.931 | 0.915 | 0.930 | 0.915 | 0.935 |
| gbm_v3 | 0.936 | 0.938 | 0.929 | 0.930 | 0.938 | 0.929 | 0.934 | 0.919 | 0.933 | 0.919 | 0.938 |
| gbm_v3_ctrl | 0.928 | 0.929 | 0.923 | 0.922 | 0.929 | 0.921 | 0.926 | 0.911 | 0.925 | 0.911 | 0.929 |
| gbm_v4 | 0.934 | 0.936 | 0.928 | 0.930 | 0.936 | 0.926 | 0.932 | 0.919 | 0.933 | 0.919 | 0.936 |

### dead F1 (alive/dead, touch GT)

| model | default | dead_vis_0.00 | dead_vis_0.50 | dead_bursts | short_ball_bursts | long_ball_bursts | random_dead_path | noisy_positions | rollout | min | max |
|---|---|---|---|---|---|---|---|---|---|---|---|
| smooth50 | 0.687 | 0.864 | 0.471 | 0.115 | 0.699 | 0.647 | 0.687 | 0.687 | 0.687 | 0.115 | 0.864 |
| dead_v1_s50 | 0.894 | 0.906 | 0.887 | 0.873 | 0.894 | 0.887 | 0.893 | 0.645 | 0.900 | 0.645 | 0.906 |
| dead_v3_s50 | 0.877 | 0.887 | 0.868 | 0.851 | 0.877 | 0.875 | 0.876 | 0.882 | 0.883 | 0.851 | 0.887 |
| seg_v1 | 0.892 | 0.917 | 0.876 | 0.876 | 0.894 | 0.885 | 0.891 | 0.627 | 0.900 | 0.627 | 0.917 |
| seg_v4 | 0.868 | 0.891 | 0.857 | 0.847 | 0.869 | 0.865 | 0.865 | 0.884 | 0.875 | 0.847 | 0.891 |
| gbm_v1 | 0.967 | 0.969 | 0.964 | 0.960 | 0.969 | 0.957 | 0.966 | 0.955 | 0.964 | 0.955 | 0.969 |
| gbm_v3 | 0.971 | 0.972 | 0.968 | 0.965 | 0.972 | 0.965 | 0.969 | 0.960 | 0.967 | 0.960 | 0.972 |
| gbm_v3_ctrl | 0.971 | 0.971 | 0.969 | 0.965 | 0.972 | 0.966 | 0.969 | 0.960 | 0.967 | 0.960 | 0.972 |
| gbm_v4 | 0.969 | 0.970 | 0.966 | 0.964 | 0.971 | 0.962 | 0.967 | 0.960 | 0.968 | 0.960 | 0.971 |


### Per game, default degrader, seed 0 (spellF1 [bootstrap 5-95])

| model | GT | game 1 | game 2 | game 3 | macroF1 game 1 / 2 / 3 |
|---|---|---|---|---|---|
| smooth50 | touch | 0.311 [0.285, 0.339] | 0.308 [0.283, 0.337] | 0.282 [0.259, 0.307] | 0.679 / 0.722 / 0.715 |
| smooth50 | controlled | 0.318 [0.295, 0.340] | 0.311 [0.289, 0.336] | 0.266 [0.243, 0.287] | 0.682 / 0.726 / 0.717 |
| seg_v4 | touch | 0.506 [0.466, 0.550] | 0.470 [0.429, 0.514] | 0.455 [0.417, 0.495] | 0.838 / 0.832 / 0.844 |
| seg_v4 | controlled | 0.630 [0.593, 0.668] | 0.613 [0.576, 0.646] | 0.578 [0.537, 0.622] | 0.849 / 0.848 / 0.856 |
| gbm_v4 | touch | 0.660 [0.615, 0.706] | 0.680 [0.634, 0.728] | 0.663 [0.615, 0.707] | 0.936 / 0.940 / 0.933 |
| gbm_v4 | controlled | 0.796 [0.761, 0.823] | 0.818 [0.787, 0.845] | 0.798 [0.769, 0.828] | 0.947 / 0.948 / 0.940 |
| gbm_v3_ctrl | touch | 0.618 [0.575, 0.667] | 0.623 [0.576, 0.675] | 0.590 [0.545, 0.641] | 0.932 / 0.934 / 0.925 |
| gbm_v3_ctrl | controlled | 0.839 [0.797, 0.872] | 0.854 [0.816, 0.887] | 0.798 [0.758, 0.838] | 0.953 / 0.956 / 0.941 |

### Stress settings outside the robustness set (3 seeds, no bootstrap)

`--variants dead_vis_0.75 rollout_random --tag stress`. Dead-ball visibility 0.75 is outside the
learned models' training range (0 to 0.5).

| model | degrader | touch spellF1 | touch macroF1 | controlled spellF1 | dead F1 |
|---|---|---|---|---|---|
| smooth50 | dead_vis_0.75 | 0.216 | 0.484 | 0.199 | 0.217 |
| seg_v4 | dead_vis_0.75 | 0.437 | 0.817 | 0.570 | 0.849 |
| gbm_v4 | dead_vis_0.75 | 0.634 | 0.924 | 0.773 | 0.963 |
| smooth50 | rollout_random | 0.292 | 0.707 | 0.281 | 0.687 |
| seg_v4 | rollout_random | 0.484 | 0.844 | 0.607 | 0.874 |
| gbm_v4 | rollout_random | 0.660 | 0.933 | 0.788 | 0.968 |

### Reading the final tables

- **Learned emissions lead everywhere.** gbm_v1 / v3 / v4 beat seg_v4 by 0.16-0.19 touch spellF1
  and 0.17-0.21 controlled spellF1 in every degrader setting and every game; on the default
  setting the worst gbm_v4 seed (0.652) is far above the best seg_v4 seed (0.490), and the bootstrap
  intervals do not overlap. seg_v4 beats smooth50 by 0.16-0.21 touch spellF1 in every setting.
  Both leads are far outside the 0.03-0.04 noise band.
- **gbm_v1, gbm_v3 and gbm_v4 cannot be told apart** (largest gap 0.011 touch spellF1, 0.019
  controlled). We keep **gbm_v4** because it assumes least: no ball-visibility features (whose
  real pattern is the degrader's least certain assumption), range-scaled motion (no assumption on
  how much of the game is dead), and trained with the ball rolling on after play stops.
- **Training target: touch, not controlled.** gbm_v3_ctrl gains 0.033 controlled spellF1 (borderline)
  and loses 0.064 touch spellF1 (beyond noise) against gbm_v3. The GTs disagree; Phase A says trust
  touch.
- **Heavy position noise is the setting that costs most** (gbm_v4 -0.039 touch spellF1, seg_v4
  -0.045). The heuristic dead_v1 collapses there (dead F1 0.645); dead_v3 holds (0.882).
- The H <-> A switching cost was chosen on gbm_v1's grid and kept for v3 / v4; touch spellF1 is
  flat over cost 4-16 at alpha 0.5 (0.647-0.666), controlled prefers more cost.

| final status | version |
|---|---|
| **current best** | gbm_v4 (alpha 0.5, H<->A cost 8, alive<->D cost 12, trained on touch GT) |
| best without learning | seg_v4 |
| dead-ball detector alone | dead_v3 (inside seg_v4); the GBM's own dead F1 is 0.96-0.97 |
| dropped | dead_v1 (noise), dead_v2 (composition), seg_v1-v3, abstention, gbm_v3_ctrl, gbm_v1 / v1_novis / v2 (superseded, not worse) |

## Phase B2: the calibrated bench (`atd_cal`)

Phase B found the real ATD's ball gaps look nothing like the Phase A degrader's (median 268
frames, not 20). Phase B2 measured the real ATD's distributions (`mtah.atd_profile`; no labels,
nothing frame-specific), added the degrader variant `atd_cal` that reproduces them
(`docs/phaseB2_report.md`, calibration table), and re-scored and retrained on it. Same protocol
as Phase C: sample games 1-3, leave one game out, degrader seeds 0-2 for the bench and 10000+ for
training, bootstrap 300 on seed 0, both GTs, gaps under 0.03-0.04 spellF1 are noise, touch is
trusted when the GTs disagree.

Every number in this section is on `atd_cal` unless the row says otherwise (3 seeds, mean).

| version | idea | touch spellF1 [boot] | controlled spellF1 [boot] | touch macroF1 | dead F1 | kept? |
|---|---|---|---|---|---|---|
| smooth50 | organiser baseline, as is | 0.262 [0.244, 0.277] | 0.270 [0.253, 0.286] | 0.595 | 0.548 | reference (published full match 0.2433) |
| seg_v4 | Phase C best without learning, as is | 0.357 [0.334, 0.378] | 0.440 [0.415, 0.462] | 0.776 | 0.861 | reference |
| gbm_v4 | Phase C best, trained on the Phase A degrader, as is | 0.499 [0.471, 0.525] | 0.604 [0.581, 0.630] | 0.863 | 0.911 | replaced: -0.161 touch against its Phase C bench number; it learned "long gap = stoppage" |
| gbm_v5a (tag gbm5a) | gbm_v4's features and settings, retrained on a mixture around `atd_cal` (default, dead visibility 0.15 / 0.5, noisy, no onset context, restart re-acquisition 0.6; 2 seeds each) | 0.543 [0.512, 0.565] | 0.693 [0.676, 0.725] | 0.896 | 0.943 | replaced by v5 (+0.044 touch, +0.089 controlled over v4: the retraining alone) |
| gbm_v5a decode grid | alpha {0.35, 0.5, 0.75} x H<->A cost {4, 8, 16} | 0.517-0.546 | 0.644-0.715 | | | flat in touch; kept 0.5 / 8 |
| gbm_v5 (tag gbm5) | gbm_v5a plus 57 long-gap features (`features_b2`): player-only team evidence relative to each team's attack (centroid advance and velocity, forward running, length, width, compactness, share in the opponent half, last line, nearest-opponent distance, 2 s / 5 s smoothing) and the ball at both ends of each gap (side-line distance, speed, nearest player, displacement across the gap towards home's attack, uncapped gap length, position in the gap); ball-visibility features back in | 0.587 [0.558, 0.611] | 0.739 [0.719, 0.762] | 0.909 | 0.950 | kept (tie with v5_nogap) |
| gbm_v5 decode grid | as above | 0.564-0.591 | 0.697-0.762 | | | flat in touch; kept 0.5 / 8 |
| gbm_v5_nop (ablation) | gbm_v5 without the B2 player features | 0.544 [0.505, 0.559] | 0.702 [0.676, 0.727] | 0.897 | 0.946 | ablation: the player features carry the whole +0.044 |
| gbm_v5_nogap (ablation) | gbm_v5 without the ball-visibility features and without the B2 gap / gap-end features | 0.588 [0.558, 0.613] | 0.734 [0.715, 0.763] | 0.908 | 0.947 | ablation: the gap features add nothing; see the final choice below |

### Attack direction fix (applies to every attack-relative feature)

`features.attack_sign` used to read each half's direction from its first 10 s (kick-off layout).
On the real ATD the second half's first 10 s show 15 tracked objects with home's centroid at 51 m
and away's at 78 m, so the second half got the wrong direction. The label-free diagnostics caught
it: gbm_v5's agreement with Smart Tagging's attacking-phase team was 0.775 in the first half and
0.628 in the second. The sign now comes from the whole half: the median over frames of
(away min x - home min x) + (away max x - home max x), which is positive while home defends the
left goal (keepers and last lines). On every half of sample games 1-3, clean and degraded, it
equals the old rule (a test asserts it on the clean games; the degraded tables were checked), so
no bench number above moves. On the real match the second-half agreement is now 0.785.

### Robustness sweep and final status (Phase B2)

The full tables are in `docs/phaseB2_report.md` (9 calibrated settings x 3 seeds, both GTs, macroF1,
dead F1). Touch spellF1 min-max over the sweep: smooth50 0.225-0.272, seg_v4 0.284-0.369, gbm_v4
0.433-0.511, gbm_v5a 0.515-0.551, gbm_v5 0.573-0.592, gbm_v5_nogap 0.567-0.595. gbm_v5 and
gbm_v5_nogap tie in every column (touch -0.006 to +0.003).

| final status | version |
|---|---|
| **current best** | gbm_v5_nogap (alpha 0.5, H<->A cost 8, alive<->D cost 12; trained on the atd_cal mixture, touch GT); ties gbm_v5 and assumes less (no ball-visibility or gap features) |
| tied, kept as alternative | gbm_v5 |
| ablation | gbm_v5_nop (the player features are the +0.044) |
| superseded | gbm_v5a (retraining only), gbm_v4 (Phase A degrader) |
| public window, once, after the choice | gbm_v5_nogap spellF1 0.619, macroF1 0.898 (pipeline check only, nothing changed after it) |

## Phase D: short possessions (`docs/phaseD_segmentation.md`)

The models predicted about 720 possessions against the touch GT's 1,146. `mtah.segdiag` showed the
misses are possessions under 3 s (37% of the touch GT, matched 1-16%) and that 85% of misses are
not matched by any decoding of the probabilities, down to the per-frame argmax: the emissions never
favour the short possession's team. Touches by the team without the ball that change nothing
outnumber touches at real hand-overs on the bench; the ball's velocity change at the touch
separates them (AUC 0.79), the player's distance does not (0.52). Same protocol as Phase B2,
`atd_cal`, 3 seeds, both GTs.

| version | idea | touch spellF1 [boot] | controlled spellF1 [boot] | touch macroF1 | pred poss (GT 1146) | kept? |
|---|---|---|---|---|---|---|
| gbm_v5_nogap | Phase B2 best | 0.588 [0.558, 0.613] | 0.734 [0.715, 0.763] | 0.908 | 723 | **submission, unchanged** |
| decode grid | alpha {0.35, 0.5, 0.75, 1} x H<->A cost {4, 8, 12, 16} | 0.567-0.590 | 0.680-0.757 | | 637-854 | dropped: flat in touch |
| touch-gated cost | H<->A cost 2-4 near a touch by the incoming team, base 8-16 (8 settings, one decode each, pooled over 3 seeds) | 0.587-0.592 | | | | dropped: the decoder is not where the misses are |
| gbm_v6_w05 (tag gbm6_w05) | training rows weighted (1 / max(L, 25 frames))^0.5 by GT possession length, possession rows keep their total weight | 0.603 [0.577, 0.627] | 0.714 [0.695, 0.735] | 0.908 | 779 | dropped: +0.015 touch, -0.020 controlled |
| gbm_v6_w1 (tag gbm6_w1) | as above, gamma 1 | 0.602 [0.564, 0.614] | 0.686 [0.667, 0.706] | 0.905 | 841 | dropped: -0.048 controlled |
| **gbm_v7** (tag gbm7) | gbm_v5_nogap plus 15 touch-kink features (`features_d`: ball velocity change at 4 / 8 frames, signed touch strength and its windows, last / next strong touch, since / until each team's strong touch) | 0.606 [0.572, 0.623] | 0.743 [0.727, 0.770] | 0.911 | 738 | **bench leader; ahead in all 9 calibrated settings in both GTs (touch +0.010 to +0.021, controlled +0.002 to +0.009), but inside the noise band: not shipped** |
| gbm_v7_w05 (tag gbm7_w05) | gbm_v7 with the gamma 0.5 weighting | 0.612 [0.581, 0.634] | 0.722 [0.713, 0.753] | 0.911 | 780 | dropped: same trade as gbm_v6_w05 |

Real match (label-free, `mtah.diagnostics`): possessions per hour 147 (gbm_v5_nogap), 150
(gbm_v7), 160 (gbm_v7_w05); D share 0.427-0.431; nothing implausible. gbm_v7's candidate file
`runs/submission/MurrayInsights_gbm_v7.csv` validates and is byte-identical on a rerun (SHA-256
94b5a719...). Public window, once, on that file: spellF1 0.621, macroF1 0.886 (pipeline check only).

| final status | version |
|---|---|
| **submission** | gbm_v5_nogap, unchanged |
| bench leader, ready to ship if Steve prefers it | gbm_v7 |
| dropped | decode grid, touch-gated cost, gbm_v6_w05 / w1, gbm_v7_w05 (possession-length weighting trades controlled for touch) |

## Phase E: a ball track shaped like the real one, and the video ball (`docs/phaseE_report.md`)

The Phase D addendum found that the real ATD ball bends less at touches and wobbles more between
them than `atd_cal`'s ball. Phase E added the kink distribution to the calibration gate and fitted
the degrader's ball error to it (`atd_cal2`). It also measured the saved video ball track and
simulated it as a separate, flagged channel (`atd_cal2v`), and added video features. Same
protocol: sample games 1-3, leave one game out, seeds 0-2 for the bench and 10000+ for training,
both GTs. Paired bootstrap (`mtah.paired`) over the touch GT's possessions, with the same
resample for both models in every cell.

| version | idea | touch spellF1 [boot] | controlled spellF1 [boot] | pred poss | kept? |
|---|---|---|---|---|---|
| gbm_v5_nogap on atd_cal2 | the shipped model, as is, on the fitted ball | 0.581 [0.547, 0.603] | 0.728 [0.704, 0.748] | 724 | reference (0.588 on atd_cal) |
| gbm_v7 on atd_cal2 | Phase D bench leader, as is | 0.580 [0.550, 0.602] | 0.728 [0.705, 0.749] | 725 | dropped: its +0.018 on atd_cal is +0.000 here (15 of 27 cells) |
| gbm_v5_nogap_c2 (tag gbm5n_c2) | gbm_v5_nogap's design retrained on atd_cal2 | 0.579 [0.555, 0.608] | 0.733 [0.713, 0.761] | 720 | dropped: ties the shipped model |
| gbm_v7_c2 (tag gbm7_c2) | gbm_v7's design retrained on atd_cal2 | 0.585 [0.556, 0.608] | 0.736 [0.713, 0.760] | 713 | dropped: +0.005 touch over gbm5n_c2, 22 of 27 cells, paired interval [-0.000, +0.010] |
| gbm_v8 (tag gbm8) | gbm_v7's design + 25 video features (`features_v`, a separate channel read from `FrameTable.vball`), trained on the atd_cal2v mixture (fill shares 0.10-0.55, error x1.5, spares 0.05) | 0.602 [0.569, 0.621] on atd_cal2v | 0.756 [0.729, 0.774] | 731 | passes the ship test; tied with gbm_v8n, not shipped (assumes more) |
| **gbm_v8n** (tag gbm8n) | gbm_v8 without the kink features: gbm_v5_nogap's features + `features_v` | **0.603 [0.567, 0.623]** on atd_cal2v | **0.755 [0.732, 0.777]** | 736 | **written as `MurrayInsights_gbm_v8n.csv`** (not submitted): +0.021 touch over gbm_v5_nogap in 27 / 27 calibrated cells, paired [+0.014, +0.029]; +0.030 controlled; fill 0 ties (+0.001), fill 0.2 +0.015 |

Sweeps (mean of 3 seeds): gbm_v8n touch 0.588-0.607 over the 9 calibrated settings (0.581 with no
video, 0.608 at fill 0.6), gbm_v5_nogap 0.563-0.587 on their atd_cal2 counterparts. Real match (label-free):
gbm_v8n has 146 possessions per hour, median 7.4 s, D 0.434, home share 0.611, Smart Tagging
agreement 0.782. Public window, once, after the choice: spellF1 0.593, macroF1 0.886 (pipeline
check only).

| final status | version |
|---|---|
| not eligible 2026-11-13 (was submission candidate) | gbm_v8n, written to `runs/submission/MurrayInsights_gbm_v8n.csv`; not submitted, not eligible (atd_cal2v's video channel is tuned on the provided video) |
| **2026-11-13: submission locked** | gbm_v5_nogap; gbm_v8n not submitted (README, Status) |
| passes the test, not chosen | gbm_v8 (ties gbm_v8n, has the kink features) |
| dropped | gbm_v7 / gbm_v7_c2 (kink features: +0.000 as is, +0.005 retrained on the fitted ball), gbm_v5_nogap_c2 (ties) |
