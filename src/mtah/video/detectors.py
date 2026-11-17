"""Ball (and person) detectors on tactical-video frames. CPU only.

Every detector returns, per frame, a float32 array (k, 6): cx, cy, w, h, conf, kind, in full-frame
pixels, where kind is BALL (0) or PERSON (1). Person boxes are only used for the alignment check
and the near-a-player plausibility signal.

Detectors (licences in docs/video_pilot.md):

- YoloDetector: any Ultralytics checkpoint; `ball_classes` / `person_classes` name which class ids
  mean what (COCO: 32 sports ball, 0 person). `zoom` > 1 cuts the frame into tiles and upsamples
  each, for detectors that never saw a ball a few pixels wide. Weights are third-party pickles:
  docs/video_pilot.md records the static scan made before loading them.
- MotionBlobDetector: classical baseline, no learning. Each frame is compared with its neighbours
  warped through the per-frame homography (H_t inv(H_s) maps the ground plane of frame s onto frame
  t, so the pitch cancels and moving objects remain); small blobs whose size matches a 22 cm ball at
  that pitch position (scale from the homography) are kept.

Ultralytics and OpenCV are imported lazily so the rest of the package needs numpy only.
"""

import numpy as np

from . import geometry

BALL, PERSON = 0, 1
BALL_DIAMETER_M = 0.22


def _empty():
    return np.zeros((0, 6), dtype=np.float32)


class YoloDetector:
    def __init__(self, weights, ball_classes, person_classes=(), imgsz=1920, conf=0.05,
                 zoom=1, tile=640, name=None, max_det=50):
        from ultralytics import YOLO
        self.model = YOLO(weights, task="detect")
        self.ball_classes = tuple(ball_classes)
        self.person_classes = tuple(person_classes)
        self.imgsz, self.conf, self.zoom, self.tile = imgsz, conf, zoom, tile
        self.max_det = max_det
        self.name = name or str(weights)
        # an exported (OpenVINO) model has a fixed batch of 1
        import os
        self.batch = 1 if os.path.isdir(str(weights)) else 64

    def _boxes(self, results, offsets, scale):
        out = []
        for r, (ox, oy) in zip(results, offsets):
            b = r.boxes
            if b is None or len(b) == 0:
                out.append(_empty())
                continue
            xywh = b.xywh.numpy().astype(np.float32) / scale
            xywh[:, 0] += ox
            xywh[:, 1] += oy
            cls = b.cls.numpy().astype(int)
            kind = np.full(len(cls), -1, dtype=np.float32)
            kind[np.isin(cls, self.ball_classes)] = BALL
            kind[np.isin(cls, self.person_classes)] = PERSON
            keep = kind >= 0
            out.append(np.column_stack([xywh, b.conf.numpy().astype(np.float32), kind])[keep])
        return out

    def __call__(self, frames, rois=None):
        """frames: (n, H, W, 3) BGR. rois: optional per-frame (x0, y0, x1, y1) to restrict tiles."""
        classes = list(self.ball_classes + self.person_classes)
        if self.zoom == 1:
            res = []
            for k in range(0, len(frames), self.batch):
                res += self.model.predict(list(frames[k:k + self.batch]), imgsz=self.imgsz,
                                          conf=self.conf, classes=classes, verbose=False,
                                          max_det=self.max_det)
            return self._boxes(res, [(0, 0)] * len(frames), 1.0)
        import cv2
        out = []
        h, w = frames.shape[1:3]
        src = self.tile // self.zoom          # source pixels per tile side
        step = src - 16
        for i, img in enumerate(frames):
            x0, y0, x1, y1 = rois[i] if rois is not None else (0, 0, w, h)
            crops, offs = [], []
            for ty in range(max(0, int(y0)), max(int(y0) + 1, int(y1) - 16), step):
                for tx in range(max(0, int(x0)), max(int(x0) + 1, int(x1) - 16), step):
                    ty_, tx_ = min(ty, h - src), min(tx, w - src)
                    crop = img[ty_:ty_ + src, tx_:tx_ + src]
                    crops.append(cv2.resize(crop, (self.tile, self.tile),
                                            interpolation=cv2.INTER_CUBIC))
                    offs.append((tx_, ty_))
            dets = []
            for k in range(0, len(crops), 16):
                res = self.model.predict(crops[k:k + 16], imgsz=self.tile, conf=self.conf,
                                         classes=classes, verbose=False, max_det=self.max_det)
                dets += self._boxes(res, offs[k:k + 16], float(self.zoom))
            d = np.concatenate(dets) if dets else _empty()
            out.append(_dedupe(d))
        return out


def _dedupe(d, radius=6.0):
    """Greedy same-kind duplicate removal across overlapping tiles (keep highest confidence)."""
    if len(d) < 2:
        return d
    d = d[np.argsort(-d[:, 4])]
    keep = []
    for j in range(len(d)):
        if all(d[j, 5] != d[k, 5] or np.hypot(*(d[j, :2] - d[k, :2])) > max(radius, d[k, 2] / 2)
               for k in keep):
            keep.append(j)
    return d[keep]


class MotionBlobDetector:
    """Homography-compensated three-frame differencing plus a ball-sized blob filter."""

    name = "motion_blob"

    def __init__(self, thresh=18, size_tol=(0.5, 2.6), max_blobs=30):
        self.thresh, self.size_tol, self.max_blobs = thresh, size_tol, max_blobs

    def __call__(self, frames, H):
        """frames: (n, H, W, 3) BGR consecutive frames; H: (n, 3, 3) their homographies.

        Frames 0 and n-1 have only one neighbour and use it alone.
        """
        import cv2
        n, h, w = frames.shape[:3]
        gray = [cv2.GaussianBlur(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (3, 3), 0) for f in frames]
        out = []
        for t in range(n):
            nbrs = [s for s in (t - 2, t + 2) if 0 <= s < n] or [s for s in (t - 1, t + 1) if 0 <= s < n]
            diffs = []
            for s in nbrs:
                M = H[t] @ np.linalg.inv(H[s])           # pixels of s -> pixels of t (ground plane)
                warped = cv2.warpPerspective(gray[s], M, (w, h), flags=cv2.INTER_LINEAR,
                                             borderMode=cv2.BORDER_REPLICATE)
                diffs.append(cv2.absdiff(gray[t], warped))
            diff = np.minimum.reduce(diffs) if len(diffs) > 1 else diffs[0]
            mask = (diff > self.thresh).astype(np.uint8)
            k, lab, stats, cent = cv2.connectedComponentsWithStats(mask, connectivity=8)
            if k <= 1:
                out.append(_empty())
                continue
            cx, cy = cent[1:, 0], cent[1:, 1]
            bw, bh, area = stats[1:, 2], stats[1:, 3], stats[1:, 4]
            px, py = geometry.unproject(H[t], cx, cy)
            scale = geometry.px_per_metre(H[t], px, py)
            d_px = BALL_DIAMETER_M * scale
            lo, hi = self.size_tol
            size = np.maximum(bw, bh)
            ok = ((size >= np.maximum(2, lo * d_px)) & (size <= hi * d_px + 2) &
                  (np.minimum(bw, bh) >= 0.4 * size) & geometry.on_pitch(px, py, 0.02))
            idx = np.flatnonzero(ok)
            if len(idx) == 0:
                out.append(_empty())
                continue
            # confidence: mean contrast of the blob relative to the threshold, capped at 1
            contrast = np.array([diff[lab == i + 1].mean() for i in idx], dtype=np.float32)
            conf = np.clip(contrast / (4.0 * self.thresh), 0, 1)
            d = np.column_stack([cx[idx], cy[idx], bw[idx], bh[idx], conf,
                                 np.full(len(idx), BALL)]).astype(np.float32)
            out.append(d[np.argsort(-d[:, 4])][: self.max_blobs])
        return out
