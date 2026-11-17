"""Synthetic FrameTables for model tests: phases of play with controlled player speed and ball."""

import numpy as np

from mtah.frames import FrameTable

L, W = 105.0, 68.0


def walk(rng, n, speed, start):
    """(n, 2) smooth walk at roughly `speed` m/s, kept on the pitch."""
    ang = np.cumsum(rng.normal(0, 0.08, n)) + rng.uniform(0, 2 * np.pi)
    sp = np.clip(rng.normal(speed, 0.2 * speed + 1e-3, n), 0, None) / 25.0
    step = np.stack([np.cos(ang), np.sin(ang)], 1) * sp[:, None]
    pos = start + np.cumsum(step, 0)
    # reflect into the pitch
    for c, lim in ((0, L), (1, W)):
        p = np.mod(pos[:, c], 2 * lim)
        pos[:, c] = np.where(p > lim, 2 * lim - p, p)
    return pos


def phases_table(phases, seed=0, noise=0.1):
    """phases: [(kind, seconds, player_speed)], kind in play / dead / buildup.

    play / buildup: the ball sits within 0.4 m of one player, a new player every 2 s (a pass,
    alternating teams in 'play', same team in 'buildup'). dead: the ball is off the pitch
    (y = W + 2) and still, visible on the first and last 5 s only.
    Returns (table, kinds) with kinds a per-frame array of the phase kind.
    """
    rng = np.random.default_rng(seed)
    n = int(sum(s for _, s, _ in phases) * 25)
    xy = np.zeros((n, 22, 2))
    ball = np.full((n, 2), np.nan)
    kinds = np.empty(n, dtype=object)
    starts = rng.uniform([10, 10], [L - 10, W - 10], (22, 2))
    t = 0
    for kind, secs, speed in phases:
        m = int(secs * 25)
        for j in range(22):
            xy[t:t + m, j] = walk(rng, m, speed, starts[j])
            starts[j] = xy[t + m - 1, j]
        if kind in ("play", "buildup"):
            holder = int(rng.integers(0, 22))
            for k in range(0, m, 50):
                if kind == "play":
                    holder = int(rng.integers(0, 11)) if (k // 50) % 2 == 0 else int(rng.integers(11, 22))
                else:
                    holder = int(rng.integers(0, 11))
                e = min(m, k + 50)
                ball[t + k:t + e] = xy[t + k:t + e, holder] + rng.normal(0, 0.2, 2)
        else:
            ball[t:t + m] = (50.0, W + 2.0)
            ball[t + 125:t + m - 125] = np.nan
        kinds[t:t + m] = kind
        t += m
    xy += rng.normal(0, noise, xy.shape)
    team = np.array([0] * 11 + [1] * 11, dtype=np.int8)
    table = FrameTable(np.arange(n), np.ones(n, dtype=np.int8),
                       (ball / [L, W]).astype(np.float32), (xy / [L, W]).astype(np.float32), team,
                       (L, W), None, {"kind": "synthetic"})
    return table, kinds
