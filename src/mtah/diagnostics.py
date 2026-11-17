"""Label-free diagnostics of a model's labels on the real match (Phase B2, task 4).

    PYTHONPATH=src .venv/bin/python -m mtah.diagnostics --models smooth50 gbm_v4 gbm_v5

These are sanity checks on transfer, not scores: no ground truth is read (we do not have the
full-match truth, and the public window is not used). For each model, on the graded frames of the
competition ATD:

- D share overall, where the ball is missing, and where it is visible;
- predicted possessions (maximal H or A runs after X is forward-filled, as the scorer does): count
  per graded hour and length distribution, beside the same numbers for the bench ground truths
  (sample games 1-3, touch and controlled);
- H / A balance (home share of H + A frames);
- agreement with Smart Tagging's automatic team phases (BUILD UP, PROGRESSION, ...: the team the
  tag names is the one attacking). Smart Tagging is automatic and an allowed input, but it is not
  ground truth either; this is a consistency check only.

Nothing here feeds back into a model's constants.
"""

import argparse
import json
import os
import time

import numpy as np

from . import paths
from .features import runs as _runs

FPS = 25
ATTACK_CODES = {"build up", "progression", "attacking transition", "creating chances",
                "ball in final third", "ball in the box", "long ball"}


def ffill_x(labels):
    out = np.asarray(labels).copy()
    last = None
    for i, c in enumerate(out):
        if c == "X":
            if last is not None:
                out[i] = last
        else:
            last = c
    return out


def possessions(labels, graded):
    """Lengths (frames) of maximal H / A runs on graded frames, after forward-filling X."""
    lab = ffill_x(np.where(graded, labels, "X"))
    lens = []
    for a, b in _runs(graded):
        seg = lab[a:b]
        for c in ("H", "A"):
            lens.extend(e - s for s, e in _runs(seg == c))
    return np.array(lens, dtype=np.int64)


def summarise_labels(labels, graded, vis=None):
    lab = np.asarray(labels)
    g = graded
    n = g.sum()
    lens = possessions(lab, g)
    hours = n / FPS / 3600.0
    out = {"graded_frames": int(n), "D_share": float((lab[g] == "D").mean()),
           "X_share": float((lab[g] == "X").mean()),
           "n_poss": int(len(lens)), "poss_per_hour": float(len(lens) / hours)}
    if len(lens):
        s = lens / FPS
        out.update({"poss_s_p25": float(np.percentile(s, 25)), "poss_s_p50": float(np.median(s)),
                    "poss_s_p75": float(np.percentile(s, 75)), "poss_s_mean": float(s.mean()),
                    "poss_share_lt3s": float((s < 3).mean())})
    ha = lab[g & np.isin(lab, ["H", "A"])]
    out["home_share_of_HA"] = float((ha == "H").mean()) if len(ha) else None
    if vis is not None:
        out["D_share_ball_missing"] = float((lab[g & ~vis] == "D").mean())
        out["D_share_ball_visible"] = float((lab[g & vis] == "D").mean())
    # D runs (stoppages): count and length
    dl = []
    for a, b in _runs(g):
        dl.extend(e - s for s, e in _runs(lab[a:b] == "D"))
    dl = np.array(dl) / FPS
    out["n_dead_runs_per_hour"] = float(len(dl) / hours)
    out["dead_run_s_p50"] = float(np.median(dl)) if len(dl) else None
    return out


def smart_tag_agreement(labels, intervals, graded):
    """Share of H / A frames inside attacking-phase tags whose label is the tagged team."""
    lab = np.asarray(labels)
    agree = total = 0
    n_frames_tagged = 0
    for iv in intervals:
        if iv.get("code", "").lower() not in ATTACK_CODES or iv.get("team") not in ("H", "A"):
            continue
        a, b = max(0, iv["start"]), min(len(lab), iv["end"])
        seg = lab[a:b][graded[a:b]]
        n_frames_tagged += len(seg)
        ha = seg[np.isin(seg, ["H", "A"])]
        agree += int((ha == iv["team"]).sum())
        total += len(ha)
    return {"tagged_frames": n_frames_tagged, "agree_share": agree / total if total else None}


def bench_gt_summary():
    from . import modelbench as mb
    out = {}
    for gk in mb.GT_KEYS:
        rows = []
        for g in (1, 2, 3):
            _, gts, graded = mb.game_truth(g)
            rows.append(summarise_labels(gts[gk], graded))
        out[gk] = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]
                   if isinstance(rows[0][k], (int, float))}
    return out


def slow_share(pspd, mask, slow_mps=1.0):
    """Share of mask frames where the players' median speed is under slow_mps."""
    return float((pspd[mask] < slow_mps).mean()) if mask.any() else None


def speed_mixture(real_table, variant="atd_cal", slow_mps=1.0):
    """Label-free estimate of the dead share inside the real match's ball gaps.

    On the bench (degraded sample games, where the truth is known) measure how often the players'
    median speed is under slow_mps on dead and on live ball-missing frames; on the real ATD measure
    it on ball-missing frames; solve real = d * P(slow | dead) + (1 - d) * P(slow | live) for d.
    A mixture of two bench rates and one real share: no BAR-ATH label is involved. It assumes the
    sample games' players move like BAR-ATH's in stoppages and in play.
    """
    from . import features as F, modelbench as mb
    pd_, pl_, w = [], [], []
    for g in (1, 2, 3):
        t, gts = mb.degraded_table(variant, 0, g)
        f = F.cached(t)
        gt = gts["touch"]
        gap = (gt != "X") & ~t.ball_visible()
        pd_.append(slow_share(f["pspd"], gap & (gt == "D"), slow_mps))
        pl_.append(slow_share(f["pspd"], gap & (gt != "D"), slow_mps))
    fr = F.compute(real_table)
    graded = real_table.period > 0
    vis = real_table.ball_visible()
    real_gap = slow_share(fr["pspd"], graded & ~vis, slow_mps)
    real_vis = slow_share(fr["pspd"], graded & vis, slow_mps)
    a, b = float(np.mean(pd_)), float(np.mean(pl_))
    d = float(np.clip((real_gap - b) / max(1e-9, a - b), 0.0, 1.0))
    return {"bench_slow_given_dead_gap": a, "bench_slow_given_live_gap": b,
            "real_slow_given_gap": real_gap, "real_slow_given_visible": real_vis,
            "estimated_dead_share_in_gaps": d}


def run(models, out_dir):
    from . import atd_io, registry, smart_tagging
    from .atd_profile import smart_tagging_path
    t0 = time.time()
    table = atd_io.load_competition()
    iv = smart_tagging.load(smart_tagging_path())
    graded = table.period > 0
    vis = table.ball_visible()
    res = {"bench_gt": bench_gt_summary(), "models": {}, "speed_mixture": speed_mixture(table)}
    vtable = None
    for m in models:
        if m in registry.VIDEO_MODELS:
            if vtable is None:
                from .video import source
                vtable = source.with_video_channel(table)
            lab = np.asarray(registry.get(m)(vtable))
        else:
            lab = np.asarray(registry.get(m)(table))
        s = summarise_labels(lab, graded, vis)
        s["smart_tag"] = smart_tag_agreement(lab, iv, graded)
        res["models"][m] = s
        print("{} done at {:.0f} s".format(m, time.time() - t0), flush=True)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "diagnostics.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    return res


def table_md(res):
    rows = [("D share (graded)", "D_share", "{:.3f}"),
            ("D share, ball missing", "D_share_ball_missing", "{:.3f}"),
            ("D share, ball visible", "D_share_ball_visible", "{:.3f}"),
            ("stoppages per hour", "n_dead_runs_per_hour", "{:.0f}"),
            ("stoppage median (s)", "dead_run_s_p50", "{:.1f}"),
            ("possessions per hour", "poss_per_hour", "{:.0f}"),
            ("possession p25 / p50 / p75 (s)", ("poss_s_p25", "poss_s_p50", "poss_s_p75"), "{:.1f}"),
            ("possessions under 3 s", "poss_share_lt3s", "{:.2f}"),
            ("home share of H + A", "home_share_of_HA", "{:.3f}")]
    cols = [("bench GT touch", res["bench_gt"]["touch"]),
            ("bench GT controlled", res["bench_gt"]["controlled"])]
    cols += [(m, s) for m, s in res["models"].items()]
    lines = ["| statistic | " + " | ".join(c for c, _ in cols) + " |",
             "|---|" + "---|" * len(cols)]
    for label, key, fmt in rows:
        keys = key if isinstance(key, tuple) else (key,)
        cells = []
        for _, s in cols:
            vals = [s.get(k) for k in keys]
            cells.append(" / ".join("-" if v is None else fmt.format(v) for v in vals))
        lines.append("| {} | {} |".format(label, " | ".join(cells)))
    cells = []
    for _, s in cols:
        st = s.get("smart_tag")
        cells.append("-" if not st or st["agree_share"] is None else "{:.3f}".format(st["agree_share"]))
    lines.append("| agreement with Smart Tagging attacking-phase team | {} |".format(" | ".join(cells)))
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--out", default=os.path.join(paths.RUNS_DIR, "phaseB2"))
    ap.add_argument("--inputs", nargs="*", default=None,
                    help="also compare kink / video inputs, real against these bench variants")
    a = ap.parse_args(argv)
    if a.inputs:
        sh = input_shift(tuple(a.inputs))
        os.makedirs(a.out, exist_ok=True)
        with open(os.path.join(a.out, "input_shift.json"), "w") as fh:
            json.dump(sh, fh, indent=1)
        print(input_shift_md(sh))
    res = run(a.models, a.out)
    print(table_md(res))
    print("speed mixture:", json.dumps(res["speed_mixture"]))


if __name__ == "__main__":
    main()


# -- Phase E: do the model's inputs on the real match look like the bench's? ----------------------

INPUT_COLUMNS = {
    "d": ["d_kink4", "d_kink8", "d_ttk", "d_ttk_past25", "d_st_since_h", "d_st_since_a"],
    "v": ["v_src", "v_src_10s", "v_tt", "v_dmin", "v_still", "v_dline", "v_spare", "v_speed"],
}


def _input_summary(f, graded, miss):
    out = {}
    for fs, cols in INPUT_COLUMNS.items():
        mask = graded & miss if fs == "v" else graded
        for c in cols:
            if c not in f:
                continue
            x = np.asarray(f[c], dtype=np.float64)[mask]
            fin = x[np.isfinite(x)]
            row = {"defined": float(len(fin) / max(1, len(x)))}
            if len(fin):
                row.update({"mean": float(fin.mean()), "nonzero": float((fin != 0).mean())})
                row.update({"p{}".format(q): float(v) for q, v in zip((10, 50, 90), np.percentile(fin, [10, 50, 90]))})
            out[c] = row
    return out


def input_shift(variants=("atd_cal2v",), seed=0):
    """Distributions of the kink (features_d) and video (features_v) inputs on graded frames
    (video columns: on ATD-missing frames) of the real match, with its saved video track, and of
    degraded sample games 1-3. Label-free: no label of either is read."""
    from . import atd_io, features as F, features_d, features_v, modelbench as mb
    from .video import source
    res = {}
    real = source.with_video_channel(atd_io.load_competition())

    def one(t):
        f = dict(features_d.compute_extra(t))
        f.update(features_v.compute_extra(t))
        return _input_summary(f, t.period > 0, ~t.ball_visible())
    res["real"] = one(real)
    for v in variants:
        rows = [one(mb.degraded_table(v, seed, g)[0]) for g in (1, 2, 3)]
        res[v] = {c: {k: float(np.mean([r[c][k] for r in rows if k in r[c]])) for k in rows[0][c]}
                  for c in rows[0]}
    return res


def input_shift_md(res):
    cols = list(res)
    lines = ["| input | stat | " + " | ".join(cols) + " |", "|---|---|" + "---|" * len(cols)]
    for c in res[cols[0]]:
        for k in ("defined", "nonzero", "p10", "p50", "p90"):
            if k not in res[cols[0]][c]:
                continue
            cells = []
            for n in cols:
                v = res[n].get(c, {}).get(k)
                cells.append("-" if v is None else "{:.3g}".format(v))
            lines.append("| {} | {} | {} |".format(c, k, " | ".join(cells)))
    return "\n".join(lines)
