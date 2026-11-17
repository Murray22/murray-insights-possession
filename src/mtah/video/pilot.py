"""Pilot: run ball detectors on seeded random clips of the tactical video. Never the whole video.

Two clip sets, both inside the graded halves (published boundaries, period > 0):

- vis: clips whose first frame has the ATD ball visible (recall is measured on every visible frame
  of the clip against the projected ATD ball).
- gap: clips lying wholly inside one ATD missing-ball run.

Clips are consecutive frames so frame-to-frame consistency can be measured. Detections go to
runs/video/pilot/<set>_<config>.npz (gitignored; frame-level output, never committed).

    python -m mtah.video.pilot --set vis --clips 100 --len 20 --configs coco11s_1920 [--first 30]

Run with the detector environment (torch + ultralytics + opencv), see docs/video_pilot.md.
"""

import argparse
import json
import os
import time

import numpy as np

from .. import paths
from . import cache, decode, weights
from . import detectors as D

OUT_DIR = os.path.join(paths.RUNS_DIR, "video", "pilot")

CONFIGS = {
    # name: (weights key, ball classes, person classes, imgsz, zoom)
    "coco11n_1920": ("coco11n", [32], [0], 1920, 1),
    "coco11s_1920": ("coco11s", [32], [0], 1920, 1),
    "coco11n_zoom2": ("coco11n", [32], [0], 640, 2),
    "soccana_1920": ("soccana", [1], [0, 2], 1920, 1),
    "soccana_1280": ("soccana", [1], [0, 2], 1280, 1),
    "acato_1920": ("acatorcini", [0], [], 1920, 1),
    "uisikdag_1920": ("uisikdag", [0], [1, 2, 3], 1920, 1),
    # same soccana network through the OpenVINO CPU runtime, fixed 1088 x 1920 input
    "soccana_ov": ("soccana", [1], [0, 2], (1088, 1920), "openvino"),
}


def runs(mask):
    """(start, length) of each True run."""
    m = np.concatenate([[False], np.asarray(mask, bool), [False]])
    d = np.diff(m.astype(np.int8))
    s, e = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return list(zip(s.tolist(), (e - s).tolist()))


def sample_clips(c, which, n_clips, length, seed=20261105):
    """Non-overlapping clip start frames, seeded. Uses period and ATD ball visibility only."""
    graded = c["period"] > 0
    vis = ~np.isnan(c["ball"][:, 0])
    rng = np.random.default_rng(seed + (0 if which == "vis" else 1))
    if which == "vis":
        ok_block = np.zeros(len(graded), bool)
        for s, ln in runs(graded):
            ok_block[s:s + ln - length + 1] = True
        cand = np.flatnonzero(ok_block & vis)
    elif which == "gap":
        cand = []
        for s, ln in runs(graded & ~vis):
            if ln >= length:
                cand.extend(range(s, s + ln - length + 1))
        cand = np.asarray(cand)
    else:
        raise ValueError(which)
    starts = []
    for f in rng.permutation(cand):
        if all(abs(int(f) - s) >= length for s in starts):
            starts.append(int(f))
        if len(starts) == n_clips:
            break
    return starts                     # sampling order, so --first N is itself a random subset


def make_detector(name):
    if name == "motion_blob":
        return D.MotionBlobDetector()
    key, ball, person, imgsz, zoom = CONFIGS[name]
    if zoom == "openvino":
        return D.YoloDetector(weights.openvino(key, imgsz), ball, person, imgsz=imgsz, name=name)
    return D.YoloDetector(weights.fetch(key), ball, person, imgsz=imgsz, zoom=zoom, name=name)


def run(which, n_clips, length, config, first=None, seed=20261105):
    c = cache.load()
    starts = sample_clips(c, which, n_clips, length, seed)
    if first:
        starts = starts[:first]
    det = make_detector(config)
    frames_out, dets_out, decode_s, detect_s = [], [], 0.0, 0.0
    for s in starts:
        t0 = time.time()
        clip = decode.read_clip(s, length)
        t1 = time.time()
        if config == "motion_blob":
            res = det(clip, c["H"][s:s + length])
        else:
            res = det(clip)
        t2 = time.time()
        decode_s += t1 - t0
        detect_s += t2 - t1
        for i, d in enumerate(res):
            frames_out.append(s + i)
            dets_out.append(d)
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "{}_{}.npz".format(which, config))
    counts = np.array([len(d) for d in dets_out])
    np.savez_compressed(out, frame=np.array(frames_out), count=counts,
                        dets=np.concatenate(dets_out) if counts.sum() else np.zeros((0, 6), np.float32),
                        clip_len=length, starts=np.array(starts))
    info = dict(set=which, config=config, clips=len(starts), length=length, frames=len(frames_out),
                decode_s=round(decode_s, 1), detect_s=round(detect_s, 1),
                detect_fps=round(len(frames_out) / detect_s, 3),
                decode_fps=round(len(frames_out) / decode_s, 2))
    with open(out.replace(".npz", ".json"), "w") as f:
        json.dump(info, f, indent=1)
    return out, info


def load_dets(path):
    z = np.load(path)
    idx = np.concatenate([[0], np.cumsum(z["count"])])
    dets = [z["dets"][idx[i]:idx[i + 1]] for i in range(len(z["count"]))]
    return z["frame"], dets, int(z["clip_len"])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, choices=["vis", "gap"])
    ap.add_argument("--clips", type=int, default=100)
    ap.add_argument("--len", type=int, default=20)
    ap.add_argument("--first", type=int, default=None, help="only the first N sampled clips")
    ap.add_argument("--configs", required=True)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args(argv)
    import torch
    torch.set_num_threads(a.threads)
    for cfg in a.configs.split(","):
        out, info = run(a.set, a.clips, a.len, cfg, a.first)
        print(json.dumps(info), flush=True)


if __name__ == "__main__":
    main()
