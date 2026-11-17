"""Phase A fidelity check: the four organiser baselines on clean and degraded sample games 1-3.

    .venv/bin/python -m mtah.bench --seeds 5 --out runs/phaseA      (PYTHONPATH=src)

Writes runs/phaseA/fidelity.json (every per-game and pooled score) and prints the tables that go
into docs/phaseA_report.md. The organiser's published FULL-MATCH baseline scores are printed beside
ours as a realism check only: they are public aggregates, not ground truth, and nothing is tuned to
them.
"""

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace

import numpy as np

from . import baseline, degrade, paths, sample_gt, sample_io, score

PUBLISHED_FULL_MATCH = {  # EVALUATION.md, organiser full-match scores
    "baseline_raw": (0.0486, 0.6532),
    "baseline_smooth25": (0.1996, 0.6752),
    "baseline_smooth50": (0.2433, 0.6724),
    "baseline_smooth125": (0.2242, 0.6356),
}

# Degrader variants for the sensitivity sweep. "default" is DegradeConfig() unchanged; the others
# change one thing each (the dead_vis_* grid spans the one assumption DATA.md cannot pin down).
SWEEP = {
    "default": {},
    "dead_vis_0.00": {"dead_ball_visible": 0.0},
    "dead_vis_0.50": {"dead_ball_visible": 0.5},
    "dead_vis_0.75": {"dead_ball_visible": 0.75},
    "dead_bursts": {"dead_ball_pattern": "bursts"},
    "short_ball_bursts": {"ball_burst_median": 5.0, "ball_burst_sigma": 0.7},
    "long_ball_bursts": {"ball_burst_median": 75.0, "ball_burst_sigma": 1.0},
    "no_context_weights": {"air_weight": 1.0, "crowd_weight": 1.0, "line_weight": 1.0},
    "ball_missing_30pct": {"target_ball_missing": 0.30},
    "ball_missing_50pct": {"target_ball_missing": 0.50},
    "players_clean": {"target_present_mean": 22.0, "swap_prob": 0.0, "dup_rate_per_min": 0.0,
                      "pos_noise_m": 0.0},
}

# Two ground-truth definitions, scored side by side. "controlled" is the rule set written from
# CHALLENGE.md (docs/sample_gt_rules.md); "touch" counts every on-ball event by the other team as
# a change of possession. Same dead-ball rules in both.
GT_VARIANTS = {
    "controlled": sample_gt.GTConfig(),
    "touch": sample_gt.GTConfig(min_control_frames=0),
}


def _game_inputs(game):
    clean = sample_io.clean_tracking(game)
    events = sample_io.normalised_events(game)
    gts = {k: sample_gt.build_labels(events, clean.n_frames, c).labels
           for k, c in GT_VARIANTS.items()}
    return clean, gts


def _score_all(gts_by_game, preds_by_game, games, bootstrap_n):
    """{gt_variant: {baseline: {game..., pooled: headline}}}."""
    out = {}
    for gk in GT_VARIANTS:
        gts = {g: gts_by_game[g][gk] for g in games}
        out[gk] = {name: score.score_games(gts, {g: preds_by_game[g][name] for g in games},
                                           bootstrap_n)
                   for name, _ in baseline.VARIANTS}
    return out


def _run_one(args):
    variant, overrides, seed, games, bootstrap_n = args
    cfg = replace(degrade.DegradeConfig(), seed=seed, **overrides)
    gts, preds, reports = {}, {}, {}
    for g in games:
        clean, gts[g] = _game_inputs(g)
        graded = gts[g]["controlled"] != "X"      # identical graded frames in every GT variant
        # distinct but reproducible seed per game
        table, rep = degrade.degrade(clean, graded, replace(cfg, seed=seed * 100 + g))
        preds[g] = baseline.predict_all(table)
        reports[g] = asdict(rep)
    return {"variant": variant, "seed": seed, "config": cfg.as_dict(), "degrader": reports,
            "scores": _jsonable(_score_all(gts, preds, games, bootstrap_n))}


def _run_clean(games, bootstrap_n):
    gts, preds = {}, {}
    for g in games:
        clean, gts[g] = _game_inputs(g)
        preds[g] = baseline.predict_all(clean)
    return _jsonable(_score_all(gts, preds, games, bootstrap_n))


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


def rank_order(scores_by_variant, key):
    """Baseline names sorted best-first by key."""
    return [n for n, _ in sorted(scores_by_variant.items(), key=lambda kv: -kv[1][key])]


def summarise(runs, variant, gt_key):
    """Mean, min, max over seeds of pooled spellF1 / macroF1, plus seed-0 bootstrap intervals."""
    rs = [r for r in runs if r["variant"] == variant]
    out = {}
    for name, _ in baseline.VARIANTS:
        sc = [r["scores"][gt_key][name] for r in rs]
        sp = [s["pooled"]["spell_f1_mean"] for s in sc]
        mf = [s["pooled"]["macro_f1"] for s in sc]
        first = min(rs, key=lambda r: r["seed"])["scores"][gt_key][name]["pooled"]
        out[name] = {
            "spell_mean": float(np.mean(sp)), "spell_min": float(np.min(sp)),
            "spell_max": float(np.max(sp)),
            "macro_mean": float(np.mean(mf)), "macro_min": float(np.min(mf)),
            "macro_max": float(np.max(mf)),
            "spell_boot": (first["spell_bootstrap_low"], first["spell_bootstrap_high"]),
            "macro_boot": (first["bootstrap_low"], first["bootstrap_high"]),
            "n_pred_spells": float(np.mean([s["pooled"]["n_pred_spells"] for s in sc])),
            "n_gt_spells": first["n_gt_spells"],
            "per_game_spell": {g: float(np.mean([s[g]["spell_f1_mean"] for s in sc]))
                               for g in sc[0] if g != "pooled"},
            "per_game_macro": {g: float(np.mean([s[g]["macro_f1"] for s in sc]))
                               for g in sc[0] if g != "pooled"},
        }
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, default=5, help="degrader seeds for the default config")
    ap.add_argument("--sweep-seeds", type=int, default=3, help="seeds for each sweep variant")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--variants", nargs="*", default=list(SWEEP))
    ap.add_argument("--out", default=os.path.join(paths.RUNS_DIR, "phaseA"))
    args = ap.parse_args(argv)
    games = list(paths.SAMPLE_GAMES)
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()

    for g in games:                         # warm the tracking cache before forking workers
        sample_io.clean_tracking(g)

    clean = _run_clean(games, args.bootstrap)
    jobs = []
    for variant in args.variants:
        n_seeds = args.seeds if variant == "default" else args.sweep_seeds
        for seed in range(n_seeds):
            jobs.append((variant, SWEEP[variant], seed, games, args.bootstrap))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        runs = list(pool.map(_run_one, jobs))

    gt_stats = {}
    for g in games:
        _, gts = _game_inputs(g)
        gt_stats[str(g)] = {k: sample_gt.label_stats(v) for k, v in gts.items()}
    result = {"games": games, "gt_stats": gt_stats, "clean": clean, "runs": runs,
              "summary": {v: {gk: summarise(runs, v, gk) for gk in GT_VARIANTS}
                          for v in args.variants},
              "published_full_match": PUBLISHED_FULL_MATCH,
              "elapsed_s": time.time() - t0}
    with open(os.path.join(args.out, "fidelity.json"), "w") as handle:
        json.dump(_jsonable(result), handle, indent=1)
    print_tables(result)
    return result


def _fmt_iv(iv):
    lo, hi = iv
    return "[{:.3f}, {:.3f}]".format(lo, hi) if lo is not None else "-"


def _order(d, key):
    return " > ".join(n.replace("baseline_", "") for n in sorted(d, key=lambda k: -d[k][key]))


def print_tables(result):
    pub = result["published_full_match"]
    print("\norganiser full match: spellF1 " + " > ".join(
        n.replace("baseline_", "") for n in sorted(pub, key=lambda k: -pub[k][0]))
        + " | macroF1 " + " > ".join(
        n.replace("baseline_", "") for n in sorted(pub, key=lambda k: -pub[k][1])))
    for gk in GT_VARIANTS:
        print("\n## Clean tracking, GT '{}' (pooled games 1-3, bootstrap 5-95)\n".format(gk))
        print("| baseline | spellF1 [5, 95] | macroF1 [5, 95] | pred / GT poss | organiser full match |")
        print("|---|---|---|---|---|")
        rows = {}
        for name, _ in baseline.VARIANTS:
            p = result["clean"][gk][name]["pooled"]
            rows[name] = {"s": p["spell_f1_mean"], "m": p["macro_f1"]}
            print("| {} | {:.3f} {} | {:.3f} {} | {} / {} | {:.4f} / {:.4f} |".format(
                name.replace("baseline_", ""), p["spell_f1_mean"],
                _fmt_iv((p["spell_bootstrap_low"], p["spell_bootstrap_high"])),
                p["macro_f1"], _fmt_iv((p["bootstrap_low"], p["bootstrap_high"])),
                p["n_pred_spells"], p["n_gt_spells"], pub[name][0], pub[name][1]))
        print("\nspellF1 " + _order(rows, "s") + " | macroF1 " + _order(rows, "m"))
    for variant, by_gt in result["summary"].items():
        for gk, summ in by_gt.items():
            print("\n## Degraded '{}', GT '{}' (pooled; mean [min, max] over seeds; "
                  "bootstrap of seed 0)\n".format(variant, gk))
            print("| baseline | spellF1 | seeds | boot | macroF1 | seeds | boot | pred poss |")
            print("|---|---|---|---|---|---|---|---|")
            for name, _ in baseline.VARIANTS:
                x = summ[name]
                print("| {} | {:.3f} | [{:.3f}, {:.3f}] | {} | {:.3f} | [{:.3f}, {:.3f}] | {} | {:.0f} |".format(
                    name.replace("baseline_", ""), x["spell_mean"], x["spell_min"], x["spell_max"],
                    _fmt_iv(x["spell_boot"]), x["macro_mean"], x["macro_min"], x["macro_max"],
                    _fmt_iv(x["macro_boot"]), x["n_pred_spells"]))
            print("\nspellF1 " + _order(summ, "spell_mean") + " | macroF1 " + _order(summ, "macro_mean"))


if __name__ == "__main__":
    main()
