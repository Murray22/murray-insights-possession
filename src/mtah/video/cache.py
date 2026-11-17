"""One compact npz with what the video work needs from the ATD and homography.

Built once with the project's loader (atd_io.load_competition, strict) so the detector
environment needs numpy only to read it. Lives in runs/video/ (gitignored): it is frame-level
match data and must never be committed.

    python -m mtah.video.cache          # writes runs/video/atd_cache.npz
"""

import os

import numpy as np

from .. import paths
from . import geometry

CACHE = os.path.join(paths.RUNS_DIR, "video", "atd_cache.npz")


def build(out=CACHE):
    from ..atd_io import load_competition
    t = load_competition("tactical", strict=True)
    H, err, valid = geometry.load_homography()
    if len(H) != t.n_frames:
        raise ValueError("homography {} rows vs ATD {} frames".format(len(H), t.n_frames))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(out, frame=t.frame, period=t.period, ball=t.ball, xy=t.xy,
                        slot_team=t.slot_team, H=H, reproj=err, valid=valid,
                        pitch=np.array(t.pitch))
    return out


def load(path=CACHE):
    z = np.load(path)
    return {k: z[k] for k in z.files}


if __name__ == "__main__":
    print(build())
