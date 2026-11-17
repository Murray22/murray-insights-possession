"""Fit the degrader's ball-error shape (atd_cal2) to the real ATD's ball-track distributions.

    PYTHONPATH=src .venv/bin/python -m mtah.ballfit --workers 6

Grid search over the Phase E ball options of `degrade.DegradeConfig` (further zero-phase smoothing
of the shown ball, a mid-frequency wobble, its smoothing), on top of `atd_cal`. Each candidate
degrades sample games 1-3 (bench seed 0) and is scored by the mean squared log ratio of its
ball-shape statistics against the real ATD's (kink 2 / 4 / 8 percentiles, kink at touches, strong
touches per minute, visible ball speed, jitter). Distributions only: no frame of BAR-ATH and no
label is read, and no model score is looked at. Writes runs/phaseE/ballfit.json.

The lag option is not searched: the video alignment (mtah.video.props, lag scan on the window
frames) puts the ATD ball at 0.00 frames against the frame-exact video (median error 1.046 m at
0, 1.048 at +0.25, 1.068 at -0.25), so a centred smoother with no lag is the model.
"""

import argparse
import itertools
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from . import atd_profile, paths

TARGETS = [
    ("kink2", ("p25", "p50", "p75", "p90", "p99")),
    ("kink4", ("p25", "p50", "p75", "p90", "p99")),
    ("kink8", ("p25", "p50", "p75", "p90", "p99")),
    ("kink4_touch", ("p50", "p90")),
    ("strong_touch_runs_per_min", None),
    ("visible_speed", ("p25", "p50", "p75")),
    ("ball_jitter_m", None),
]
OUT = os.path.join(paths.RUNS_DIR, "phaseE")


def shape_stats(table):
    """The ball statistics the fit targets (a subset of atd_profile.ball_profile)."""
    import warnings
    from .atd_profile import _metric, ball_context, _q, _jitter, ball_shape_profile
    ball, xy, L, W = _metric(table)
    graded = table.period > 0
    vis = ~np.isnan(ball[:, 0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ctx = ball_context(ball, xy, L, W)
    out = ball_shape_profile(table)
    out["visible_speed"] = _q(ctx["speed"][vis & graded])
    out["ball_jitter_m"] = _jitter(ball[:, None, :], graded)
    return out


def flat(stats):
    vals = []
    for key, qs in TARGETS:
        v = stats[key]
        vals.extend([v[q] for q in qs] if qs else [v])
    return np.array(vals, dtype=np.float64)


def loss(real, deg):
    r, d = flat(real), flat(deg)
    return float(np.mean(np.log(np.maximum(d, 1e-3) / np.maximum(r, 1e-3)) ** 2))


def _eval(args):
    overrides, games = args
    from . import degrade, modelbench as mb
    profs = []
    for g in games:
        clean, _, graded = mb.game_truth(g)
        cfg = degrade.make_config(dict(mb.ATD_CAL, **overrides), seed=g)
        t, _ = degrade.degrade(clean, graded, cfg)
        t.period = np.where(graded, t.period, 0).astype(np.int8)
        profs.append(shape_stats(t))
    return overrides, atd_profile.mean_profiles(profs)


GRID = {
    "ball_post_smooth_frames": [1, 5, 9, 13],
    "ball_post_smooth_passes": [1, 2],
    "ball_wobble_m": [0.0, 0.1, 0.2, 0.3],
    "ball_wobble_frames": [5, 9, 15],
    "ball_wobble_passes": [2],
}


def grid_configs(grid=GRID):
    keys = sorted(grid)
    out = []
    for combo in itertools.product(*[grid[k] for k in keys]):
        o = dict(zip(keys, combo))
        if o["ball_post_smooth_frames"] == 1 and o["ball_post_smooth_passes"] > 1:
            continue
        if o["ball_wobble_m"] == 0 and (o["ball_wobble_frames"] != grid["ball_wobble_frames"][0]):
            continue
        out.append(o)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--games", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--grid", default=None, help="JSON {option: [values]} replacing GRID")
    a = ap.parse_args(argv)
    from . import atd_io
    t0 = time.time()
    real = shape_stats(atd_io.load_competition())
    grid = json.loads(a.grid) if a.grid else GRID
    cfgs = grid_configs(grid)
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        res = list(pool.map(_eval, [(c, a.games) for c in cfgs]))
    rows = sorted(({"overrides": o, "loss": loss(real, s), "stats": s} for o, s in res),
                  key=lambda r: r["loss"])
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "ballfit.json"), "w") as fh:
        json.dump({"real": real, "rows": rows, "elapsed_s": time.time() - t0}, fh, indent=1)
    print("real", np.round(flat(real), 2).tolist())
    for r in rows[:12]:
        print("{:.4f} {} {}".format(r["loss"], r["overrides"], np.round(flat(r["stats"]), 2).tolist()))
    print("{} configs in {:.0f} s".format(len(cfgs), time.time() - t0))


if __name__ == "__main__":
    main()
