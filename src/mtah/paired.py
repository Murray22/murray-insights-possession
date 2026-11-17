"""Paired bootstrap of the spellF1 difference between two models on the same bench tables.

    PYTHONPATH=src .venv/bin/python -m mtah.paired --a gbm_v8 --b gbm_v5_nogap \
        --variants atd_cal2v cal2v_vis15 ... [--b-variants atd_cal2 ...] --seeds 3 --draws 1000

Both models are scored on the same degraded tables (or, with --b-variants, on tables whose ATD
part is identical, as atd_cal2 and atd_cal2v are). Each draw resamples the ground-truth
possessions with replacement (the organiser's bootstrap_spell_f1 unit, sample games 1-3 pooled)
and applies the SAME resample to both models, every seed and every variant, so the draw's
difference is paired. A draw's difference is the mean over seeds and variants of
spellF1(A) - spellF1(B), each spellF1 computed as possession_metrics does (one-to-one matches at
IoU 0.3-0.7, n_pred held fixed as in bootstrap_spell_f1). Reports the observed mean difference,
the 2.5-97.5 and 5-95 percentiles of the draws and the share of draws above 0, per variant and
over all variants, for both ground truths.
"""

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from . import modelbench as mb, paths, score

pm = score.pm


def _preds(args):
    model, variant, seed, games = args
    from . import registry
    fn = registry.get(model)
    out = {}
    for g in games:
        t, _ = mb.degraded_table(variant, seed, g)
        out[g] = np.asarray(fn(t))
    return (model, variant, seed), out


def spell_flags(gt_codes, pred_codes):
    """(flags (len(SPELL_IOUS), n_gt) matched per GT possession, n_pred), as the organiser."""
    runs = pm.valid_runs(gt_codes)
    filled = pm.fill_abstentions(pred_codes, runs)
    gs = pm.possession_spells(gt_codes, runs)
    ps = pm.possession_spells(filled, runs)
    ov = pm.spell_overlaps(gs, ps)
    flags = np.vstack([pm.matched_gt_flags(ov, len(gs), th) for th in pm.SPELL_IOUS])
    return flags, len(ps)


def f1_from(flags, n_pred, picks=None):
    """Mean F1 over the IoU thresholds; with picks, on the resampled GT possessions."""
    n_gt = flags.shape[1]
    tp = flags.sum(1) if picks is None else flags[:, picks].sum(1)
    fn = n_gt - tp
    fp = np.maximum(0, n_pred - tp)
    with np.errstate(invalid="ignore", divide="ignore"):
        f1 = np.where(tp > 0, 2.0 * tp / (2 * tp + fp + fn), 0.0)
    return float(f1.mean())


def run(a, b, variants, b_variants=None, seeds=(0, 1, 2), games=(1, 2, 3), draws=1000,
        workers=6, seed=0):
    b_variants = list(b_variants or variants)
    if len(b_variants) != len(variants):
        raise ValueError("b_variants must pair one to one with variants")
    jobs = [(a, v, s, list(games)) for v in variants for s in seeds]
    jobs += [(b, v, s, list(games)) for v in b_variants for s in seeds]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        preds = dict(pool.map(_preds, jobs))
    gts = {g: mb.game_truth(g)[1] for g in games}
    out = {"a": a, "b": b, "variants": list(variants), "b_variants": b_variants,
           "seeds": list(seeds), "draws": draws}
    for gk in mb.GT_KEYS:
        gt = score.pool([gts[g][gk] for g in games])
        cells = []                                   # (variant index, seed, flagsA, nA, flagsB, nB)
        for i, (va, vb) in enumerate(zip(variants, b_variants)):
            for s in seeds:
                pa = score.pool([preds[(a, va, s)][g] for g in games])
                pb = score.pool([preds[(b, vb, s)][g] for g in games])
                fa, na = spell_flags(gt, pa)
                fb, nb = spell_flags(gt, pb)
                cells.append((i, s, fa, na, fb, nb))
        n_gt = cells[0][2].shape[1]
        obs = np.array([f1_from(c[2], c[3]) - f1_from(c[4], c[5]) for c in cells])
        rng = np.random.RandomState(seed)
        D = np.empty((draws, len(cells)))
        for d in range(draws):
            picks = rng.randint(0, n_gt, n_gt)
            D[d] = [f1_from(c[2], c[3], picks) - f1_from(c[4], c[5], picks) for c in cells]
        res = {"n_gt": int(n_gt), "per_variant": {}}

        def summ(cols):
            m = D[:, cols].mean(1)
            return {"observed": float(obs[cols].mean()), "cells_positive": int((obs[cols] > 0).sum()),
                    "cells": int(len(cols)), "min_cell": float(obs[cols].min()),
                    "p2.5": float(np.percentile(m, 2.5)), "p97.5": float(np.percentile(m, 97.5)),
                    "p5": float(np.percentile(m, 5)), "p95": float(np.percentile(m, 95)),
                    "share_above_0": float((m > 0).mean())}
        for i, v in enumerate(variants):
            res["per_variant"][v] = summ([k for k, c in enumerate(cells) if c[0] == i])
        res["all"] = summ(list(range(len(cells))))
        out[gk] = res
    return out


def table_md(res):
    lines = ["| GT | variant | A - B observed | cells > 0 | min cell | paired boot 2.5-97.5 | 5-95 | draws > 0 |",
             "|---|---|---|---|---|---|---|---|"]
    for gk in mb.GT_KEYS:
        r = res[gk]
        rows = list(r["per_variant"].items()) + [("all", r["all"])]
        for v, s in rows:
            lines.append("| {} | {} | {:+.3f} | {} / {} | {:+.3f} | [{:+.3f}, {:+.3f}] | [{:+.3f}, {:+.3f}] | {:.3f} |".format(
                gk, v, s["observed"], s["cells_positive"], s["cells"], s["min_cell"], s["p2.5"],
                s["p97.5"], s["p5"], s["p95"], s["share_above_0"]))
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--variants", nargs="+", required=True)
    ap.add_argument("--b-variants", nargs="+", default=None)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--tag", default=None)
    a = ap.parse_args(argv)
    t0 = time.time()
    res = run(a.a, a.b, a.variants, a.b_variants, tuple(range(a.seeds)), draws=a.draws,
              workers=a.workers)
    out = os.path.join(paths.RUNS_DIR, "phaseE", "paired_{}.json".format(a.tag or "{}_vs_{}".format(a.a, a.b)))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1)
    print("A = {}, B = {}".format(a.a, a.b))
    print(table_md(res))
    print("{:.0f} s".format(time.time() - t0))


if __name__ == "__main__":
    main()
