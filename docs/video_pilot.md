# Video ball detection pilot

Date: 2026-11-05. Question: can our own ball detection on the tactical video fill the ATD's
missing-ball gaps (41% of graded frames, 150 runs, median 268 frames), and at what CPU cost?
Inputs: the tactical video, its homography, the ATD and the automatic Smart Tagging. No ground
truth was read (nothing under `ref/evaluation/gt/`), and no frame of BAR-ATH was labelled or
corrected by hand. Every number below comes from code in `src/mtah/video/`. Contact sheets were
looked at to understand failure modes. No count or rate in this document comes from looking.

## Verdict

**Go, limited: run it on the gap frames as a separate, flagged ball source; do not build further
on it yet.** The run is cheap (about 5-6 h of otherwise idle CPU, launched 2026-11-05 05:53 at
nice 19, resumable) and purely additive. It gives a tracked ball on about 39% of gap frames.
That includes dead balls the ATD drops (a ball placed or held for a restart) and the ball at a
player's feet. Fast and lofted balls in gaps are not recovered. The detector agrees with the ATD
on visible frames only moderately: 49% of visible frames have a detection within 25 px, and the
tracked ball is within 25 px of the ATD on 68% of the frames it covers. Whether this helps the
possession model can only be shown by the model owner, using label-free diagnostics. The bench
cannot simulate a video source.

## 1. Alignment

**Video frame index.** ffprobe on the mp4 shows 199,319 video packets with pts exactly 1000 * i
for i = 0..199,318, time base 1/25000, constant 25 fps, no B-frames, and a keyframe every 50
frames (3,987 keyframes, every gap 50). So the frame count matches DATA.md and video frame i is
pts 1000 * i.

**Decoder** (`decode.read_clip`). It does not trust mp4 seeking. It seeks to the keyframe at or
before the first wanted frame (an exact time), decodes forward, and reads every delivered frame's
pts back from ffmpeg's `showinfo`. A clip whose pts are not exactly the requested run raises
`FrameIndexError`. Tests: the check refuses a late, early, short, duplicated or dropped run.
On the real file, a seek gives the same pixels as forward decoding from the earlier keyframe,
and a one-frame offset gives different pixels. The last frame (199,318) decodes. Every clip in
this pilot passed the pts check.

**ATD + homography against the pixels** (`align.check`). For shift k, the ATD position of frame
f + k is projected with the homography of frame f. d_k is its distance to the nearest
detection on frame f, capped. The comparison uses only informative items: the reference moves
at least a few pixels to both neighbours and some shift is within reach of a detection. The
decision is a paired sign test of d_0 against d_-1 and against d_+1. Main sample: 1,988 visible
frames (100 clips) with soccana through OpenVINO.

| item | informative | mean px at -5 / -2 / -1 / **0** / +1 / +2 / +5 | sign test vs -1 | vs +1 |
|---|---|---|---|---|
| players (ATD slot vs person-box foot point) | 18,626 of 41,011 | 14.4 / 7.5 / 4.8 / **3.1** / 4.1 / 6.5 / 13.8 | 14,803 vs 3,810, p ~ 0 | 12,190 vs 6,426, p ~ 0 |
| ball (ATD ball vs ball detections >= 0.1) | 333 of 1,976 | (-3) 18.7 / 14.0 / 9.3 / **7.5** / 9.9 / 13.7 / (+3) 17.5 | 224 vs 109, p 1e-10 | 215 vs 117, p 4e-8 |

Aligned at shift 0, as DATA.md says: ATD frame_id - 1 = video frame = homography row. A plain
hit rate cannot see a one-frame slip, because a ball at a player's feet moves a pixel or two per
frame. The screen showed hit rates of 0.187 / 0.183 / 0.173 at -1 / 0 / +1, which is why the
check uses paired distances. `tests/test_video_align.py` feeds the checker slips of +1, -1 and +2
and unrelated detections, and checks that it says no each time.

Overlays (in `runs/video/`, gitignored): `overlay_50000.png` (play), `overlay_130000.png` (a
gap during a corner), and contact sheets `contact_vis_soccana.png`, `contact_disagree.png`,
`contact_static_gap.png`.

## 2. The ATD ball is a ground position, and that sets every "recall" below

The ATD ball has three channels: x, y and speed, with no height. Two independent detectors (COCO
YOLO11s and the football-specific soccana) put their top ball candidate within 4 px of each
other on 162 of 300 visible frames. On those frames the ATD ball is more than 25 px away 34% of
the time and more than 50 px away 25% of the time. The direction of the gap is the tell:

| detector consensus vs ATD | frames | share straight up (dy < 0, \|dy\| > \|dx\|) | median dx, dy (px) |
|---|---|---|---|
| 0-10 px | 66 | 0.29 | 3, -3 |
| 10-25 px | 41 | 0.44 | 0, -8 |
| 25-100 px | 34 | 0.82 | 1, -33 |
| 100-400 px | 19 | 0.84 | -3, -113 |

A ball in the air shows up above the ground point under it. Even close agreement has a -3 px bias,
about one ball radius at that scale. So the ATD tracks the ball's ground position (sensible for
tracking), while a detector sees the ball itself. Agreement with the ATD is therefore a lower
bound on detector quality. The tables report three measures: point recall at 10, 25 and 50 px;
a "corridor" recall (\|dx\| <= 10 px, detection between 300 px above and 10 px below the ATD
point); and metres on the ground (pitch 105.3 x 68 m), which a lofted ball distorts.

Consequence for use: a detection's ground back-projection is right for a ball on the ground and
wrong for a ball in the air, where it lands too far from the camera. The loader says so in its
metadata.

## 3. Detectors

All public, all CPU (i7-1185G7, 4 cores / 8 threads, no GPU). Weights are downloaded by
`weights.fetch`, which checks the sha256 and statically scans the checkpoint's pickle before
loading. All five import only torch, collections, ultralytics and builtins.set. Confidence
threshold 0.1 below. That choice was made against the ATD's own visible ball, which is
self-consistency, not labels. Candidates must back-project within 3 m of the pitch.

Screen: 300 visible frames (15 seeded random clips of 20) and 300 gap frames (15 clips of 20).
"tracked" means kept by the Viterbi tracker (`track.py`) in a run of at least 5 frames.

| detector (input) | licence | fps | rec 10 px | rec 25 px | rec 50 px | corridor | FP / frame | tracked | tracked within 25 / 50 px | gap: any cand. | gap: tracked |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **soccana** YOLO11n football (1920) | AGPL-3.0 (ckpt; no card licence) | 2.04 torch, **3.17 OpenVINO** | 0.25 | 0.41 | 0.48 | 0.45 | 0.91 | 0.56 | 0.67 / 0.77 | 0.56 | 0.43 |
| COCO YOLO11s, sports ball (1920) | AGPL-3.0 | 1.01 | 0.23 | 0.37 | 0.43 | 0.41 | 0.72 | 0.49 | 0.69 / 0.78 | 0.50 | 0.31 |
| COCO YOLO11n, tiled 2x zoom (640 tiles) | AGPL-3.0 | 0.14 | 0.27 | 0.44 | 0.52 | 0.48 | 5.50 | 0.91 | 0.26 / 0.28 | - | - |
| COCO YOLO11n (1920) | AGPL-3.0 | 5.2 | no hit on an 8-frame smoke clip; dropped | | | | | | | | |
| acatorcini YOLOv9 ball (1920) | AGPL-3.0 | 0.52 | 0.17 | 0.30 | 0.35 | 0.37 | 1.62 | 0.34 | 0.52 / 0.63 | - | - |
| uisikdag YOLOv8 football (1920) | none stated | 0.62 | 0.22 | 0.37 | 0.41 | 0.39 | 1.06 | 0.70 | 0.45 / 0.47 | - | - |
| motion blobs, homography-compensated (no learning) | ours | 6.6 | 0.29 | 0.54 | 0.80 | 0.58 | 29.7 | 1.00 | 0.39 / 0.45 | 1.00 | 1.00 |

- The **motion baseline** finds the ball often (recall 0.54 at 25 px) but in 30 blobs a frame, so
  its tracked output is noise. It would need a learned verifier on top; not pursued.
- **Tiling** gives the COCO model a little more recall at 7x the cost and 6x the false
  positives.
- **soccana through OpenVINO** gives the same detections as torch (all 581 ball detections
  >= 0.1 on the 300-frame screen matched within 2 px) at 1.6x the speed. It is the pick.

**Main sample, soccana via OpenVINO, 2,000 visible + 2,000 gap frames (100 clips each, seeded):**

| | value |
|---|---|
| visible: any detection within 10 / 25 / 50 px of the ATD | 0.29 / 0.49 / 0.58 |
| visible: corridor recall; within 1 / 2 / 3 m on the ground | 0.47; 0.40 / 0.52 / 0.55 |
| visible: false positives per frame; detection precision | 0.92; 0.24 |
| visible: tracked-ball rate; tracked ball within 25 / 50 px of the ATD | 0.67; 0.68 / 0.78 |
| **gap: any candidate; linked to a neighbouring frame; tracked** | **0.51; 0.47; 0.39** |
| gap: share of candidates linked frame to frame | 0.91 |
| gap: top candidate within 2 m of an ATD player; median distance | 0.25; 2.6 m |

## 4. What the gaps are

From the ATD, homography and Smart Tagging over all graded frames (`gap_profile.py`; aggregates
only):

| signal | visible-ball frames | gap frames |
|---|---|---|
| median player speed (m/s, median over tracked players) | 1.64 | 0.99 |
| players' median speed < 1 m/s (stoppage-like) | 11% | **51%** |
| share of the pitch inside the image (median) | 0.60 | 0.57 |
| within 10 s before a Smart Tagging set piece starts | 4.9% | 15.1% |
| ATD players on the frame (decoded sample, median) | 22 | 21 |

- **About half the gap time looks like a stoppage**, and gaps lead into set pieces three times
  as often as visible-ball time does.
- **Camera framing is not the cause:** the camera shows as much of the pitch during gaps as during
  visible play, and the ATD keeps tracking 21 players.
- **The ball leaving the image is rare:** the last-seen ATD ball is outside the image at the
  gap's middle frame in 3% of gaps (14% of gap frames, the long ones).
- **The ball usually goes missing away from the lines:** 32% of gaps start within 3 m of a
  touchline or goal line.

On the decoded gap sample (2,000 frames, soccana tracked ball at 0.1):

| decoded gap frames | play-like (51%) | stoppage-like (49%) |
|---|---|---|
| tracked ball within 2 m of a player | 0.16 | 0.16 |
| tracked ball loose on the pitch | 0.23 | 0.24 |
| no tracked ball | 0.62 | 0.60 |
| tracked ball static for >= 10 frames (< 0.3 m; a subset of the two ball rows) | 0.21 | 0.31 |
| tracked ball moving > 15 m/s | 0.00 | 0.00 |

Same measures on the visible sample: static 0.05 (play) and 0.13 (stoppage); fast 0.10 and 0.09.

The video says three things about the gaps:

1. **The ATD drops dead balls.** A static ball is tracked on 21-31% of gap frames against 5-13%
   of visible ones. The contact sheet shows what these are: a ball lying on the halfway line
   during a long stoppage, a ball in a player's hands at a throw-in. They are real balls, and
   they are dead-ball evidence by themselves.
2. **Fast balls are not recovered.** A ball moving at more than 15 m/s is never tracked in a gap,
   against 10% of visible frames. Motion blur at 25 fps and a few-pixel ball defeat the detector,
   and gaps hold more long balls. Flights stay gaps.
3. **The ball is with a player on about 16% of gap frames.** That is the case where the video
   can give a team, through the nearest player, and where the ATD is likely losing the ball in
   occlusion.

## 5. Runtime and cost

Measured on the screen with nothing else running. Contended runs, with the other agent's model
jobs on the same 8 threads, measured 1.6-2.4 fps.

| | frames | soccana OpenVINO, 3.2 fps | soccana torch, 2.0 fps | COCO YOLO11s, 1.0 fps |
|---|---|---|---|---|
| graded gap frames | 64,643 | 5.7 h | 9.0 h | 18 h |
| gaps + 25-frame windows (the job) | 71,841 | 6.3 h | 10 h | 20 h |
| whole graded match | 158,857 | 14 h | 22 h | 44 h |

Decoding costs little: 36 fps sequential and 20-50 fps on random clips, about 10% on top of
detection. Not yet tried: OpenVINO THROUGHPUT mode or INT8, and cropping the stands out of the
frame.

## 6. Licences

- **Code and runtimes:** Ultralytics 8.4 is AGPL-3.0. torch is BSD. OpenVINO is Apache-2.0.
  OpenCV is Apache-2.0. ffmpeg is LGPL/GPL, called as an external binary.
- **Weights:**
  - Ultralytics YOLO11 COCO: AGPL-3.0.
  - soccana (`Adit-jain/soccana` on Hugging Face): the model card states no licence; the
    checkpoint metadata says AGPL-3.0. It was trained on the author's public multi-source
    dataset (`Adit-jain/Soccana_player_ball_detection_v1`).
  - acatorcini: AGPL-3.0 (card).
  - uisikdag: no licence stated.
- **What this means for us:** everything is public and downloadable by the organisers, which is
  what RULES.md requires. AGPL obligations attach to distributing or serving a modified program.
  Our public repo only calls these packages, but Steve should be aware the dependency is AGPL.
  This is not legal advice.

## 7. Environment

The detector stack lives in its own venv, `~/.cache/wt-mtah-video/venv` (Python 3.12.13: torch
2.15 CPU, ultralytics 8.4.202, opencv 5.0, openvino 2026.4, onnxruntime). It is kept off the
project `.venv` so a concurrent agent's environment never sees a numpy or opencv change. The
project has no requirements file. Weights live in `~/.cache/wt-mtah-video/models` and come from
`python -m mtah.video.weights`. Code that reads results (`source.py`, `evaluate.py`,
`gap_profile.py`) runs on the project `.venv` with numpy only.

Commands, run from `src/`:

```
../.venv/bin/python -m mtah.video.cache                 # runs/video/atd_cache.npz (ATD + homography)
DET=~/.cache/wt-mtah-video/venv/bin/python
$DET -m mtah.video.pilot --set vis --clips 100 --len 20 --configs soccana_ov
$DET -m mtah.video.pilot --set gap --clips 100 --len 20 --configs soccana_ov
../.venv/bin/python -m mtah.video.evaluate               # table + runs/video/pilot_eval.json
../.venv/bin/python -m mtah.video.gap_profile
~/.cache/wt-mtah-video/run_full.sh                       # full gap run (resumable), then tracking
```

## 8. The full run and the loader

- **Job.** `fullrun.py detect` covers every graded gap frame plus 25 frames either side, so the
  tracker can start from a visible ball: 71,841 frames in 781 chunks of at most 100. Each chunk
  is pts-verified and written atomically to `runs/video/full/soccana_ov/`, so a restart skips
  finished chunks.
- **Tracking.** `fullrun.py track` runs the Viterbi tracker over each contiguous run and writes
  `runs/video/full/ball_track_soccana_ov.npz`, one row per video frame. It holds the ground
  position, pixel, confidence, run support, candidate count and a processed mask.
- **Status.** Launched 05:53 as a detached process at nice 19, about 4 fps on an idle machine.
  Log: `runs/video/full/run.log`, ending `FULL_RUN_DONE`. To resume after a kill, run
  `run_full.sh` again.
- **Loader** (`source.py`, numpy only, no model or registry change):
  - `video_ball(table)` gives the video ball on the table's frames, NaN below support 5.
  - `with_video_ball(table)` returns a copy. The ATD ball is kept wherever it exists; only
    ATD-missing frames are filled. `meta["ball_source"]` holds 0 ATD / 1 video / -1 none, and
    `meta["video_ball"]` the raw arrays and support.
  - `tests/test_video_source.py` checks that it never overwrites the ATD.

## 9. Worries and found-not-fixed

- **Precision inside gaps is not measured directly.** It cannot be without labels. What
  calibrates it is the visible frames (tracked ball within 50 px of the ATD on 78%) plus
  plausibility: 91% of candidates link frame to frame, and the static balls are real. Gaps hold
  more stoppages and fewer balls at feet than visible play, so the visible-frame calibration may
  not transfer.
- **Spare balls and out-of-play balls.** 48% of gap frames have a ball candidate 3-15 m outside
  the lines; overlay 50000 shows a spare ball by the dugout. Candidates are cut at 3 m beyond the
  lines, so a ball in a player's hands for a throw-in just outside the line can survive. A
  consumer should treat the video ball near the lines as dead-ball evidence, not possession.
- **The models' D rate in gaps.** The Phase B and B2 reports have every model calling 84-89% of
  missing-ball frames D. The ATD's own player speeds say about 51% of gap frames are
  stoppage-like. This pilot can't say which is right, but the gap is large enough to check.
- **Aerial balls** are invisible to this source by construction (section 2). A
  height-aware track would need a camera model; the homography gives only the ground plane.
- **Alignment check on the ball uses the detector it validates.** The players' check uses
  person boxes and is independent of the ball detector, and it gives the same answer.
