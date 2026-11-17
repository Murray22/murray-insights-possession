"""Frame-exact decoding of the tactical video through an ffmpeg pipe.

Measured on the delivered file (docs/video_pilot.md): 199,319 video packets, constant 25 fps,
time base 1/25000, packet pts exactly 1000 * i for i = 0..199,318, no B-frames, and a keyframe
every 50 frames. Video frame i is therefore the frame with pts 1000 * i, which is the
homography row i and ATD frame_id i + 1.

mp4 seeking is a classic source of off-by-N errors, so this reader does not trust it: it seeks to
the keyframe at or before the first wanted frame (an exact keyframe time, where input seeking is
exact), decodes forward, and reads every decoded frame's pts back from ffmpeg's showinfo filter.
A clip whose pts are not exactly the expected run raises FrameIndexError instead of returning
shifted pixels.
"""

import os
import re
import subprocess

import numpy as np

from .. import paths

FPS = 25
GOP = 50                      # keyframe interval measured on the tactical file
PTS_PER_FRAME = 1000          # time base 1/25000 at 25 fps
WIDTH, HEIGHT = 1920, 1080
N_FRAMES = 199319

_PTS_RE = re.compile(r"\bpts:\s*(-?\d+)")


class FrameIndexError(RuntimeError):
    """Decoded frames are not the frames that were asked for."""


def video_path(root=None):
    root = root or os.environ.get("MTAH_DATA_DIR") or os.path.join(paths.ROOT, "data")
    return os.path.join(root, "tactical", "BAR-ATH_tactical.mp4")


def check_pts(pts, start, n, pts_per_frame=PTS_PER_FRAME):
    """Raise FrameIndexError unless pts is exactly start..start+n-1 in frame units."""
    pts = np.asarray(pts, dtype=np.int64)
    expected = (start + np.arange(n, dtype=np.int64)) * pts_per_frame
    if len(pts) != n or not np.array_equal(pts, expected):
        got = (pts[:3] // pts_per_frame).tolist() if len(pts) else []
        raise FrameIndexError("asked for frames {}..{}, decoded {} frames starting {}".format(
            start, start + n - 1, len(pts), got))


def read_clip(start, n, path=None, scale=1.0, threads=2, gray=False):
    """Frames start .. start+n-1 as a uint8 array (n, H, W, 3) BGR (or (n, H, W) if gray).

    `scale` < 1 resizes in ffmpeg (area filter) before the pipe. Raises FrameIndexError if the
    decoded timestamps are not exactly the requested frames.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if start < 0 or start + n > N_FRAMES:
        raise ValueError("frames {}..{} outside 0..{}".format(start, start + n - 1, N_FRAMES - 1))
    path = path or video_path()
    key = (start // GOP) * GOP
    skip = start - key
    w, h = int(round(WIDTH * scale)) // 2 * 2, int(round(HEIGHT * scale)) // 2 * 2
    # select keeps frames by their own index after the keyframe; showinfo then reports the pts of
    # exactly the frames we keep. -copyts keeps the file's pts so they can be checked.
    vf = "select='gte(n\\,{})',showinfo".format(skip)
    if scale != 1.0:
        vf += ",scale={}:{}:flags=area".format(w, h)
    cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "info", "-threads", str(threads),
           "-copyts", "-ss", "{:.6f}".format(key / FPS), "-i", path,
           "-vf", vf, "-fps_mode", "passthrough", "-frames:v", str(n),
           "-f", "rawvideo", "-pix_fmt", "gray" if gray else "bgr24", "-"]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        raise RuntimeError("ffmpeg failed: " + proc.stderr.decode(errors="replace")[-2000:])
    pts = [int(m.group(1)) for line in proc.stderr.decode(errors="replace").splitlines()
           if "Parsed_showinfo" in line for m in [_PTS_RE.search(line)] if m]
    # the filter graph may run one frame past -frames:v before the muxer stops; only the first n
    # reached the pipe
    check_pts(pts[:n], start, n)
    ch = 1 if gray else 3
    frame_bytes = w * h * ch
    if len(proc.stdout) != n * frame_bytes:
        raise FrameIndexError("pipe returned {} bytes for {} frames".format(len(proc.stdout), n))
    arr = np.frombuffer(proc.stdout, dtype=np.uint8)
    return arr.reshape((n, h, w) if gray else (n, h, w, 3))


def stream(start, stop, path=None, scale=1.0, threads=4, gray=False, chunk=None):
    """Yield (frame_index, image) sequentially for start <= frame < stop, pts-checked per chunk.

    For long runs (the full-match job): one ffmpeg process per chunk of `chunk` frames (default
    500), each verified like read_clip.
    """
    chunk = chunk or 500
    f = start
    while f < stop:
        n = min(chunk, stop - f)
        clip = read_clip(f, n, path=path, scale=scale, threads=threads, gray=gray)
        for i in range(n):
            yield f + i, clip[i]
        f += n
