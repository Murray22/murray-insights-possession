# More Than a Hack 2027: Initial Challenge (team Murray Insights)

Steve registered 2026-11-03. Organisers: Metrica Sports, Barça Innovation Hub, Mobile World Capital Barcelona.
Questions: hackathon.talentarena@mobileworldcapital.com, or an issue on the organiser repo.

## Dates

| When | What |
|---|---|
| by 2026-11-06 | data access email (within 72 h of registering) |
| 2026-11-10 | every team member registered under the exact team name "Murray Insights" (welcome email) |
| by 2026-11-14 | submission form URL by email |
| **2026-11-17 23:59 CET = 17:59 Ottawa** | possession CSV + public code repo + technical document, one Google Form |
| 2026-11-26 to 2027-01-14 | Innovation Challenge, top 20 teams only, 10 matches of tracking + events |
| 2027-03-02 | final pitch in Barcelona, 5 teams |

## The task (ref/CHALLENGE.md, ref/EVALUATION.md, ref/DATA.md)

One match, Barcelona (H) vs Athletic Club (A), LaLiga 2024-25 MD2. Label every one of 199,319 tactical-video frames
`H`, `A`, `D` (dead ball) or `X` (abstain). Graded: 11185-90651 and 110808-190197 (158,857 frames).

- **Ranking metric spellF1:** possessions (maximal H or A runs) matched one-to-one by temporal IoU, F1 averaged over
  IoU 0.3-0.7. One frame late is free, one second late costs about a third. Fragmentation is the main enemy.
- **Tie-break macroF1** (per-frame, H/A/D averaged).
- **X is forward-filled** for spellF1: free inside a possession, costly across a hand-over.
- **Floor:** `baseline_smooth50`, full-match spellF1 0.2433.
- **No leaderboard, no feedback.** The public 5 minutes (17 possessions) are for correctness only; tuning on them picks
  the wrong model (smooth125 0.424 public vs 0.224 full).
- **Ground truth is mechanical from corrected tracking:** `D` where corrected tracking marks the ball dead, otherwise the
  team the tracking says owns it. Loose or contested balls stay with the last team.

## Rules that can disqualify us (ref/RULES.md)

1. No hand labels or hand corrections on BAR-ATH, for any purpose, including rules or constants tied to its frames.
   Hand-labelled other games are fine and must be documented.
2. No private ground truth, and no human-produced data about BAR-ATH from any source (Opta, StatsBomb, LaLiga...).
3. The public repo must regenerate the submitted file byte for byte.
4. Match data is never committed or published, nor frame-level derivatives (tracking exports, clips, our CSVs).

## Layout

- `ref/`: organiser repo clone (MIT code; f8f2629, 2026-10-23). Scorer verified on a fresh clone: public spellF1
  smooth125 0.4242, smooth50 0.3256, smooth25 0.2868, raw 0.0378, matching EVALUATION.md.
- `data/`: the licensed bundle goes here (`MTAH_DATA_DIR`). Gitignored.
- `external/metrica_sample/`: Metrica's public sample games 1-3 (tracking 25 fps + events; "acknowledge the source").
  Gitignored; the fetch goes into a script before submission.
- `src/`, `tests/`: our code. `runs/`: outputs, gitignored.
- `.venv/`: Python 3, kloppy 3.18.0 (organiser pin), numpy, pandas, scipy, scikit-learn, pytest.

## Plan from 19:00 tonight

**A. Validation harness, no match data needed (do first).** There is no leaderboard, so this is how we choose between
models.
1. Possession ground truth for Metrica sample games 1-3 from their events: `D` between a ball-out or foul and the
   restart, a team's possession from a controlled action, loose balls kept by the last team. Write the rules down; they
   go in the technical document.
2. A degrader that makes clean sample tracking look like the ATD: ball removed on about 41% of in-play frames in
   bursts, identities stripped and slots re-used, positional noise, the odd duplicate. Calibrate burst lengths to the
   real ATD when it arrives (distribution only, no labels).
3. Score with the organiser's `possession_metrics.py` (imported from ref/, not copied), with bootstrap intervals.
   Leave-one-game-out for anything learned.

**B. When the data lands.** Load the tactical ATD, homography and Smart Tagging; profile ball visibility and slot counts;
reproduce `baseline_smooth50.csv` byte for byte from our own loader; run `validate_submission.py`.

**C. Models, in order, each scored on A (public window as a correctness check only).**
1. Dead-ball detector: ball outside the pitch, a whole-team speed collapse, restart shapes, Smart Tagging set pieces as
   evidence.
2. Segment model: HMM or Viterbi over {H, A, D} with emissions from ball-to-nearest-player by team, ball speed and
   direction, team shape and pressing, and Smart Tagging team phases; a switching cost that stops fragmentation.
   Abstain inside long ball gaps, guess at hand-overs.
3. Learned emissions (gradient boosting) trained on the degraded sample games.
4. Only if time: our own ball detection from the video to fill gaps.

**D. Deliverables, from about 11-14.** One command regenerates the file; a test that it is byte-identical on rerun;
technical document (approach, assets used and dropped, training data, possession rules, limitations, reproduction);
make the code repo public with no data; validate the exact file uploaded.

## Open items

- Every team member registers as "Murray Insights" by 11-10.
- A Google account is needed for the submission form.
