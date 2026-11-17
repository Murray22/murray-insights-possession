"""Numeric frame-alignment check: do ATD positions, projected with the homography, land on what a
video detector sees on the same frame, and on no neighbouring frame better?

Items are (frame, object) pairs: the ATD ball on a sampled frame, or each ATD player. For every
candidate shift k the reference is the ATD position of frame f + k projected with the homography
of frame f, and d_k is the distance from that reference to the nearest detection on frame f,
capped at `cap` pixels.

Hit rates are a weak test: when the ball sits at a player's feet it moves a pixel or two per frame
and a one-frame slip still "hits". So the decision uses only informative items, where the
reference moves at least `min_motion` pixels to both neighbouring shifts and some shift comes
within `radius` of a detection, and compares distances pairwise: n01 = items closer at 0 than at k,
n10 = closer at k than at 0. Aligned means shift 0 has the lowest mean capped distance and beats
k = -1 and k = +1 each by a one-sided sign test at `alpha`.
"""

import math

import numpy as np


def nearest(ref, dets, cap):
    """ref: (n, 2) pixels (NaN = none); dets: list of (k_i, >=2). (n,) distance capped at cap."""
    out = np.full(len(ref), np.nan)
    for i, (r, d) in enumerate(zip(ref, dets)):
        if not np.isfinite(r).all():
            continue
        out[i] = min(cap, np.hypot(d[:, 0] - r[0], d[:, 1] - r[1]).min()) if len(d) else cap
    return out


def _sign_test_p(n01, n10):
    """One-sided P(X >= n01) for X ~ Binomial(n01 + n10, 0.5)."""
    n = n01 + n10
    if n == 0:
        return 1.0
    return sum(math.comb(n, j) for j in range(n01, n + 1)) / (1 << n)    # exact for large n


def check(ref_by_shift, dets, radius=8.0, cap=30.0, min_motion=3.0, alpha=0.01):
    """ref_by_shift: {shift: (n, 2) reference pixels}, must contain -1, 0, +1; dets: n arrays.

    Returns per-shift mean capped distance and hit rate (all items), the informative item count,
    paired counts and p-values against +-1, the best shift and `aligned`.
    """
    for k in (-1, 0, 1):
        if k not in ref_by_shift:
            raise ValueError("shift {} missing".format(k))
    d = {k: nearest(r, dets, cap) for k, r in ref_by_shift.items()}
    r0 = ref_by_shift[0]
    move = np.minimum(np.hypot(*(ref_by_shift[1] - r0).T), np.hypot(*(ref_by_shift[-1] - r0).T))
    finite = np.all([np.isfinite(v) for v in d.values()], axis=0)
    close = np.min([v for v in d.values()], axis=0) <= radius
    info = finite & (move >= min_motion) & close
    mean = {k: float(v[info].mean()) if info.any() else float("nan") for k, v in d.items()}
    hit = {k: float(np.nanmean(v <= radius)) if np.isfinite(v).any() else float("nan")
           for k, v in d.items()}
    best = min(mean, key=lambda k: (mean[k] if np.isfinite(mean[k]) else np.inf, abs(k)))
    pair = {}
    for k in (-1, 1):
        n01 = int((d[0][info] < d[k][info]).sum())
        n10 = int((d[0][info] > d[k][info]).sum())
        pair[k] = dict(n01=n01, n10=n10, p=_sign_test_p(n01, n10))
    aligned = bool(info.any() and best == 0 and all(pair[k]["p"] < alpha for k in (-1, 1)))
    return dict(mean_dist=mean, hit_rate=hit, best=best, pair=pair, aligned=aligned,
                n=int(finite.sum()), n_informative=int(info.sum()), radius=radius,
                min_motion=min_motion)
