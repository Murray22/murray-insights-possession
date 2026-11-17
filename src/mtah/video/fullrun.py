"""Full video ball run: detect on every ATD gap frame (plus a window around each gap for linking),
then track. Resumable; output in runs/video/full/ (gitignored, frame-level, never committed).

    # detection (long; run in the background with the detector environment)
    python -m mtah.video.fullrun detect --config soccana_ov [--scope gaps|graded] [--window 25]
    # tracking + the ball-source file read by mtah.video.source
    python -m mtah.video.fullrun track --config soccana_ov

Detection writes one npz per chunk of CHUNK consecutive frames (atomic rename), so a killed job
restarts where it stopped; existing chunks are skipped. Every chunk is pts-verified by
decode.read_clip.
"""

import argparse
import glob
import json
import os
import time

import numpy as np

from .. import paths
from . import cache, decode, geometry, metrics, track
from .pilot import make_detector, runs

FULL_DIR = os.path.join(paths.RUNS_DIR, "video", "full")
CHUNK = 100


def frames_to_process(c, scope="gaps", window=25):
    graded = c["period"] > 0
    if scope == "graded":
        return graded.copy()
    gap = graded & np.isnan(c["ball"][:, 0])
    want = gap.copy()
    for s, ln in runs(gap):
        want[max(0, s - window):s + ln + window] = True
    return want & graded


def chunks(mask, size=CHUNK):
    out = []
    for s, ln in runs(mask):
        for a in range(s, s + ln, size):
            out.append((a, min(size, s + ln - a)))
    return out


def chunk_path(config, start, n):
    return os.path.join(FULL_DIR, config, "{:06d}_{:03d}.npz".format(start, n))


def detect(config, scope="gaps", window=25, threads=8, limit=None):
    import torch
    torch.set_num_threads(threads)
    c = cache.load()
    todo = chunks(frames_to_process(c, scope, window))
    total = sum(n for _, n in todo)
    pending = [(s, n) for s, n in todo if not os.path.isfile(chunk_path(config, s, n))]
    done_frames = total - sum(n for _, n in pending)
    if limit:
        pending = pending[:limit]
    os.makedirs(os.path.join(FULL_DIR, config), exist_ok=True)
    with open(os.path.join(FULL_DIR, config, "plan.json"), "w") as f:
        json.dump(dict(config=config, scope=scope, window=window, frames=int(total),
                       chunks=len(todo)), f)
    det = make_detector(config)
    t0, n_run = time.time(), 0
    for s, n in pending:
        clip = decode.read_clip(s, n)
        res = det(clip)
        counts = np.array([len(d) for d in res])
        out = chunk_path(config, s, n)
        tmp = out[:-4] + ".tmp.npz"
        np.savez_compressed(tmp, frame=np.arange(s, s + n), count=counts,
                            dets=np.concatenate(res) if counts.sum() else np.zeros((0, 6), np.float32))
        os.replace(tmp, out)
        n_run += n
        done_frames += n
        rate = n_run / (time.time() - t0)
        print("chunk {} +{}  {}/{} frames  {:.2f} fps  eta {:.1f} h".format(
            s, n, done_frames, total, rate, (total - done_frames) / rate / 3600), flush=True)
    print("DETECT_DONE", flush=True)


def load_chunks(config):
    frames, dets = [], []
    for p in sorted(glob.glob(os.path.join(FULL_DIR, config, "*_*.npz"))):
        if p.endswith(".tmp.npz"):
            continue
        z = np.load(p)
        idx = np.concatenate([[0], np.cumsum(z["count"])])
        frames.extend(z["frame"].tolist())
        dets.extend(z["dets"][idx[i]:idx[i + 1]] for i in range(len(z["count"])))
    order = np.argsort(frames)
    return np.asarray(frames)[order], [dets[i] for i in order]


def build_track(config, conf=0.1, margin_m=3.0, out=None):
    """Viterbi over each contiguous processed run; writes the ball-source npz (one row per video
    frame, NaN where not processed or no ball kept)."""
    c = cache.load()
    frames, dets = load_chunks(config)
    n = len(c["ball"])
    res = dict(processed=np.zeros(n, bool), px=np.full((n, 2), np.nan, np.float32),
               ball=np.full((n, 2), np.nan, np.float32), conf=np.full(n, np.nan, np.float32),
               support=np.zeros(n, np.int32), n_cand=np.zeros(n, np.int16))
    res["processed"][frames] = True
    rows = metrics.per_frame(c, frames, dets, conf, margin_m)
    pos = {f: i for i, f in enumerate(frames)}
    for s, ln in runs(res["processed"]):
        seg = [rows[pos[f]] for f in range(s, s + ln)]
        scales = np.array([float(geometry.px_per_metre(
            c["H"][r["frame"]], *(r["ground"][0] if len(r["ground"]) else (0.5, 0.5)))) for r in seg])
        path = track.viterbi([r["pix"] for r in seg], scales)
        sup = track.support(path)
        for i, p in enumerate(path):
            f = s + i
            res["n_cand"][f] = len(seg[i]["pix"])
            if p >= 0:
                res["px"][f] = seg[i]["pix"][p, :2]
                res["ball"][f] = seg[i]["ground"][p]
                res["conf"][f] = seg[i]["pix"][p, 2]
                res["support"][f] = sup[i]
    out = out or os.path.join(FULL_DIR, "ball_track_{}.npz".format(config))
    np.savez_compressed(out, config=config, conf_threshold=conf, margin_m=margin_m, **res)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["detect", "track"])
    ap.add_argument("--config", default="soccana_ov")
    ap.add_argument("--scope", default="gaps", choices=["gaps", "graded"])
    ap.add_argument("--window", type=int, default=25)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="process at most N chunks (testing)")
    a = ap.parse_args(argv)
    if a.step == "detect":
        detect(a.config, a.scope, a.window, a.threads, a.limit)
    else:
        print(build_track(a.config))


if __name__ == "__main__":
    main()
