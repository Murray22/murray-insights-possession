"""Homography: normalised pitch (0-1, the ATD convention) <-> tactical-video pixels.

Same convention as ref/starter/load_homography.py (row-major h00..h22 maps pitch -> pixel; row i
is video frame i). Vectorised over frames here because the pilot projects thousands of frames.
"""

import os

import numpy as np
import pandas as pd

from .. import paths

MATRIX_COLUMNS = ["h{}{}".format(r, c) for r in range(3) for c in range(3)]
PITCH_M = (105.3, 68.0)       # ref/games.py, measured by the organiser


def homography_path(root=None):
    root = root or os.environ.get("MTAH_DATA_DIR") or os.path.join(paths.ROOT, "data")
    return os.path.join(root, "tactical", "BAR-ATH_tactical_homography.csv")


def load_homography(path=None):
    """(H (n,3,3) float64, reprojection_error (n,), valid (n,) bool), indexed by video frame.

    Refuses a file whose `frame` column is not exactly 0..n-1.
    """
    t = pd.read_csv(path or homography_path())
    if not np.array_equal(t["frame"].values, np.arange(len(t))):
        raise ValueError("homography rows are not one per video frame from 0")
    H = t[MATRIX_COLUMNS].values.astype(np.float64).reshape(-1, 3, 3)
    return H, t["reprojection_error"].values.astype(np.float64), t["valid"].values.astype(bool)


def project(H, x, y):
    """Pitch (0-1) -> pixels. H is (3,3) or (k,3,3) broadcast against x, y of shape (k,) or (k,m)."""
    H = np.asarray(H, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if H.ndim == 3:
        extra = x.ndim - 1
        H = H.reshape(H.shape[:1] + (1,) * extra + (3, 3))
    d = H[..., 2, 0] * x + H[..., 2, 1] * y + H[..., 2, 2]
    return ((H[..., 0, 0] * x + H[..., 0, 1] * y + H[..., 0, 2]) / d,
            (H[..., 1, 0] * x + H[..., 1, 1] * y + H[..., 1, 2]) / d)


def unproject(H, px, py):
    """Pixels -> pitch (0-1), assuming the point is on the ground plane."""
    return project(np.linalg.inv(np.asarray(H, dtype=np.float64)), px, py)


def metres_between(a, b, pitch=PITCH_M):
    """Euclidean distance in metres between normalised pitch points (..., 2)."""
    d = (np.asarray(a) - np.asarray(b)) * np.asarray(pitch)
    return np.sqrt((d ** 2).sum(-1))


def px_per_metre(H, x, y, pitch=PITCH_M):
    """Local image scale (pixels per metre on the ground) at pitch point(s) x, y on frame H.

    Geometric mean of the scale along the two pitch axes, from a 1 m finite difference.
    """
    dx, dy = 1.0 / pitch[0], 1.0 / pitch[1]
    u0, v0 = project(H, x, y)
    u1, v1 = project(H, np.asarray(x) + dx, y)
    u2, v2 = project(H, x, np.asarray(y) + dy)
    sx = np.hypot(u1 - u0, v1 - v0)
    sy = np.hypot(u2 - u0, v2 - v0)
    return np.sqrt(sx * sy)


def on_pitch(x, y, margin=0.0):
    """Normalised point inside the pitch, widened by `margin` (normalised units) on each side."""
    x, y = np.asarray(x), np.asarray(y)
    return (x >= -margin) & (x <= 1 + margin) & (y >= -margin) & (y <= 1 + margin)
