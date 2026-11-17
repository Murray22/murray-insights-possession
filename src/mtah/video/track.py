"""Turn per-frame ball candidates into one ball track: Viterbi over {candidate_1..k, none} per frame.

State costs (lower is better):
- a candidate costs -log(conf) (conf clipped to [0.01, 1]);
- "none" costs `miss_cost`, so a weak, isolated candidate loses to "none";
- moving between candidates on consecutive frames costs (d / step_px)^2, where d is the pixel
  distance and step_px = MAX_SPEED_MPS / fps * local px-per-metre; a jump beyond `max_steps`
  step limits is forbidden;
- entering or leaving "none" costs `switch_cost`, which suppresses one-frame blips;
- across a run of "none" frames the next candidate is compared with the last candidate, its
  step limit scaled by the frames elapsed (handled approximately by the switch cost).

Every parameter is generic (ball physics and a confidence scale); none is fitted to a frame of the
match. The output keeps the chosen candidate, its confidence, and `support`: the number of frames
in the contiguous chosen run it belongs to, so a consumer can demand e.g. support >= 5.
"""

import numpy as np

MAX_SPEED_MPS = 40.0
FPS = 25


def viterbi(cands, scales, miss_cost=2.5, switch_cost=1.5, max_steps=3.0):
    """cands: list over frames of (k_t, 3) arrays px, py, conf; scales: (n,) px per metre.

    Returns (n,) int: index of the chosen candidate per frame, -1 for none.
    """
    n = len(cands)
    if n == 0:
        return np.zeros(0, int)
    INF = 1e18
    costs, backs = [], []
    prev_c = None
    for t in range(n):
        c = cands[t]
        k = len(c)
        node = np.empty(k + 1)
        node[:k] = -np.log(np.clip(c[:, 2], 0.01, 1.0)) if k else []
        node[k] = miss_cost
        if t == 0:
            cost = node.copy()
            back = np.full(k + 1, -1)
        else:
            p = cands[t - 1]
            kp = len(p)
            trans = np.full((kp + 1, k + 1), INF)
            if kp and k:
                step = MAX_SPEED_MPS / FPS * max(scales[t], 1.0) + 3.0
                d = np.hypot(p[:, None, 0] - c[None, :, 0], p[:, None, 1] - c[None, :, 1])
                tr = (d / step) ** 2
                tr[d > max_steps * step] = INF
                trans[:kp, :k] = tr
            trans[:kp, k] = switch_cost          # candidate -> none
            trans[kp, :k] = switch_cost          # none -> candidate
            trans[kp, k] = 0.0
            tot = prev_c[:, None] + trans
            back = tot.argmin(0)
            cost = tot[back, np.arange(k + 1)] + node
        costs.append(cost)
        backs.append(back)
        prev_c = cost
    path = np.empty(n, int)
    s = int(np.argmin(costs[-1]))
    for t in range(n - 1, -1, -1):
        path[t] = s if s < len(cands[t]) else -1
        s = int(backs[t][s]) if t > 0 else s
    return path


def support(path):
    """Length of the contiguous chosen (non -1) run each frame belongs to; 0 where none."""
    out = np.zeros(len(path), int)
    t = 0
    while t < len(path):
        if path[t] < 0:
            t += 1
            continue
        u = t
        while u < len(path) and path[u] >= 0:
            u += 1
        out[t:u] = u - t
        t = u
    return out
