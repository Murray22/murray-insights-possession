"""Viterbi and forward-backward over a handful of states, per graded run of frames.

Energies, not probabilities, for Viterbi: `emit` is (n, K) log-evidence per frame and state, and
`cost` is a (K, K) matrix of switching penalties (cost[i, j] paid on a frame where the state goes
from i to j; zero on the diagonal). Larger costs give fewer, longer segments, which is exactly the
fragmentation knob spellF1 cares about. Plain float64 loops in a fixed order, so the result is
bit-for-bit reproducible.
"""

import numpy as np


def viterbi(emit, cost):
    """Best state path (n,) int for one contiguous run. `cost` is (K, K), or (n, K, K) for a
    switching cost that varies by frame (cost[t] is paid on the step from t - 1 to t)."""
    emit = np.asarray(emit, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    n, K = emit.shape
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    if cost.ndim == 3 and cost.shape != (n, K, K):
        raise ValueError("per-frame cost shape {} != {}".format(cost.shape, (n, K, K)))
    back = np.empty((n, K), dtype=np.int8)
    score = emit[0].copy()
    neg = -cost                                   # (K from, K to) or (n, K, K)
    tv = cost.ndim == 3
    for t in range(1, n):
        cand = score[:, None] + (neg[t] if tv else neg)   # from i to j
        best = cand.argmax(axis=0)
        score = cand[best, np.arange(K)] + emit[t]
        back[t] = best
    path = np.empty(n, dtype=np.int64)
    path[-1] = int(score.argmax())
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def posteriors(emit, cost):
    """(n, K) state posteriors for one run, reading exp(-cost) as unnormalised transitions."""
    emit = np.asarray(emit, dtype=np.float64)
    n, K = emit.shape
    T = np.exp(-np.asarray(cost, dtype=np.float64))
    T = T / T.sum(axis=1, keepdims=True)
    e = np.exp(emit - emit.max(axis=1, keepdims=True))
    alpha = np.empty((n, K))
    a = e[0] / e[0].sum()
    alpha[0] = a
    for t in range(1, n):
        a = (a @ T) * e[t]
        a /= a.sum()
        alpha[t] = a
    beta = np.ones(K)
    post = np.empty((n, K))
    post[-1] = alpha[-1]
    for t in range(n - 2, -1, -1):
        beta = T @ (e[t + 1] * beta)
        beta /= beta.sum()
        p = alpha[t] * beta
        post[t] = p / p.sum()
    return post


def graded_runs(period):
    """[(start, stop)] of contiguous frames with period > 0 and constant period."""
    p = np.asarray(period)
    out = []
    n = len(p)
    t = 0
    while t < n:
        if p[t] <= 0:
            t += 1
            continue
        s = t
        while t < n and p[t] == p[s]:
            t += 1
        out.append((s, t))
    return out


def decode(emit, cost, period, want_post=False):
    """Viterbi per graded run; -1 outside. Optionally the posteriors too (NaN outside)."""
    n, K = emit.shape
    path = np.full(n, -1, dtype=np.int64)
    post = np.full((n, K), np.nan) if want_post else None
    cost = np.asarray(cost, dtype=np.float64)
    if want_post and cost.ndim == 3:
        raise ValueError("posteriors take a fixed (K, K) cost")
    for s, e in graded_runs(period):
        path[s:e] = viterbi(emit[s:e], cost[s:e] if cost.ndim == 3 else cost)
        if want_post:
            post[s:e] = posteriors(emit[s:e], cost)
    return path, post
