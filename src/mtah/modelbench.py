"""Score possession models on the Phase A bench: degraded sample games 1-3, both GT variants.

    PYTHONPATH=src .venv/bin/python -m mtah.modelbench --models seg_v1 --variants default \
        --seeds 3 --bootstrap 300

For every (model, degrader variant, seed) it degrades the three sample games exactly as Phase A
did (seed * 100 + game), runs the model on the degraded FrameTable, and scores the labels with the
organiser's possession_metrics against the "controlled" and "touch" ground truths, per game and
pooled. Results go to runs/phaseC/<model>.json; the printed table is what docs/model_log.md quotes.

The table handed to a model has period = 0 outside the graded spans, as the real ATD will (its
graded half blocks are published in ref/games.py). Models never see the ground truth, the clean
tracking or player identities.
"""

import argparse
import hashlib
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

import numpy as np

from . import bench, degrade, paths, sample_gt, sample_io, score
from .frames import FrameTable

CACHE = os.path.join(paths.RUNS_DIR, "cache", "degraded")
OUT = os.path.join(paths.RUNS_DIR, "phaseC")

VARIANTS = dict(bench.SWEEP)
VARIANTS["random_dead_path"] = {"dead_ball_path": "random"}
VARIANTS["noisy_positions"] = {"pos_noise_m": 0.4, "ball_noise_m": 0.6}
VARIANTS["rollout"] = {"dead_ball_rollout": True}
VARIANTS["rollout_random"] = {"dead_ball_rollout": True, "dead_ball_path": "random"}
VARIANTS["dead_vis_0.75"] = dict(bench.SWEEP["dead_vis_0.75"])
# Phase B2: the degrader calibrated to the real ATD's distributions (docs/phaseB2_report.md).
# Ball gaps from one joint process with the real gap lengths, dead-ball visibility from the ball's
# visibility around Smart Tagging set pieces, live onset weights from the real onset context,
# smooth positions, long tracklets, mass dropouts. Distributions only; nothing frame-specific.
ATD_CAL = {
    "ball_mode": "joint", "ball_burst_lengths": "atd", "target_ball_missing": 0.407,
    "dead_ball_visible": 0.30, "dead_ball_path": "random", "dead_ball_rollout": True,
    "restart_reacq_prob": 0.2, "dead_out_weight": 30.0, "min_gap_frames": 25,
    "slow_weight": 1.0, "fast_weight": 0.2, "crowd_weight": 3.0, "crowd_ball_min": 2,
    "line_weight": 2.0, "deep_weight": 2.0,
    "smooth_frames": 9, "ball_noise_m": 0.6, "pos_noise_m": 0.25, "noise_rho": 0.99,
    "noise_smooth_frames": 25,
    "gap_median": 250.0, "reacquire_prob": 0.0, "dup_rate_per_min": 2.5, "slot_policy": "lowest",
    "mass_drop_rate_per_min": 0.7, "mass_drop_sigma": 1.2, "target_present_mean": 20.0,
    "mass_drop_frac_lo": 0.5, "mass_drop_frac_hi": 0.9,
}
PHASE_C_NAMES = frozenset(VARIANTS)
VARIANTS["atd_cal"] = dict(ATD_CAL)
# robustness sweep around the calibrated settings; the learned gbm5 mixture trains on cal_vis15,
# cal_vis50, cal_noisy, cal_noctx and cal_restart60 (other seeds) and never on cal_miss35,
# cal_miss47 or cal_nomass
CAL_SWEEP = {
    "cal_vis15": {"dead_ball_visible": 0.15},
    "cal_vis50": {"dead_ball_visible": 0.50},
    "cal_miss35": {"target_ball_missing": 0.35},
    "cal_miss47": {"target_ball_missing": 0.47},
    "cal_noisy": {"ball_noise_m": 1.0, "pos_noise_m": 0.4},
    "cal_noctx": {"slow_weight": 1.0, "fast_weight": 1.0, "crowd_weight": 1.0, "line_weight": 1.0,
                  "deep_weight": 1.0, "dead_out_weight": 1.0},
    "cal_restart60": {"restart_reacq_prob": 0.6},
    "cal_nomass": {"mass_drop_rate_per_min": 0.0},
}
for _k, _v in CAL_SWEEP.items():
    VARIANTS[_k] = dict(ATD_CAL, **_v)
ROBUST_B2 = ("atd_cal",) + tuple(CAL_SWEEP)
# Phase E: atd_cal plus the ball-error shape fitted to the real ATD's kink distributions
# (mtah.ballfit, runs/phaseE/ballfit.json: further 13-frame centred mean applied twice, a 0.25 m
# wobble smoothed over 21 frames twice, no lag). Players, gaps and everything else as atd_cal.
BALL_SHAPE = {"ball_post_smooth_frames": 13, "ball_post_smooth_passes": 2, "ball_wobble_m": 0.25,
              "ball_wobble_frames": 21, "ball_wobble_passes": 2, "ball_lag_frames": 0.0}
ATD_CAL2 = dict(ATD_CAL, **BALL_SHAPE)
# Phase E: atd_cal2 plus the simulated video channel, calibrated to the real video track
# (mtah.video.props; docs/phaseE_report.md). The ATD part of every table is identical to
# atd_cal2's (the video draws from its own random stream).
VIDEO_CAL = {"video_fill_share": 0.41, "video_spare_share": 0.02,
             "video_speed_weights": (1.0, 0.04, 0.02, 0.01), "video_resid_m": 0.3,
             "video_resid_rho": 0.9998, "video_error_scale": 1.0, "video_calibration": "real"}
ATD_CAL2V = dict(ATD_CAL2, **VIDEO_CAL)
VARIANTS["atd_cal2"] = dict(ATD_CAL2)
VARIANTS["atd_cal2v"] = dict(ATD_CAL2V)
for _k, _v in CAL_SWEEP.items():
    VARIANTS["cal2_" + _k[4:]] = dict(ATD_CAL2, **_v)
    VARIANTS["cal2v_" + _k[4:]] = dict(ATD_CAL2V, **_v)
# video robustness (all held out of gbm8's training mixture except as noted in learned.VIDEO_MIX)
VIDEO_SWEEP = {
    "cal2v_f00": {"video_fill_share": 0.0, "video_spare_share": 0.0},
    "cal2v_f20": {"video_fill_share": 0.20, "video_spare_share": 0.01},
    "cal2v_f60": {"video_fill_share": 0.60, "video_spare_share": 0.03},
    "cal2v_err2": {"video_error_scale": 2.0},
    "cal2v_nospare": {"video_spare_share": 0.0},
    "cal2v_spare3": {"video_spare_share": 0.06},
}
for _k, _v in VIDEO_SWEEP.items():
    VARIANTS[_k] = dict(ATD_CAL2V, **_v)
ROBUST_C2 = ("atd_cal2",) + tuple("cal2_" + k[4:] for k in CAL_SWEEP)
ROBUST_C2V = ("atd_cal2v",) + tuple("cal2v_" + k[4:] for k in CAL_SWEEP)
# the robustness set every final candidate is repeated on (Phase A notes: dead-ball visibility
# 0 to 0.5, short and long ball bursts), plus the randomised dead-ball path
ROBUST = ("default", "dead_vis_0.00", "dead_vis_0.50", "dead_bursts", "short_ball_bursts",
          "long_ball_bursts", "random_dead_path", "noisy_positions", "rollout")
GT_KEYS = ("controlled", "touch")


def _key(variant, seed, game):
    cfg = json.dumps(VARIANTS[variant], sort_keys=True)
    return "{}_{}_s{}_g{}".format(variant, hashlib.md5(cfg.encode()).hexdigest()[:6], seed, game)


def game_truth(game):
    """(clean table, {gt_key: labels}, graded mask)."""
    clean = sample_io.clean_tracking(game)
    events = sample_io.normalised_events(game)
    gts = {k: sample_gt.build_labels(events, clean.n_frames, bench.GT_VARIANTS[k]).labels
           for k in GT_KEYS}
    return clean, gts, gts["controlled"] != "X"


def degraded_table(variant, seed, game, use_cache=True):
    """The degraded FrameTable for (variant, seed, game), cached under runs/cache/degraded."""
    path = os.path.join(CACHE, _key(variant, seed, game) + ".npz")
    clean, gts, graded = game_truth(game)
    if use_cache and os.path.isfile(path):
        t = FrameTable.load_npz(path)
    else:
        cfg = degrade.make_config(VARIANTS[variant], seed=seed * 100 + game)
        t, _ = degrade.degrade(clean, graded, cfg)
        if use_cache:
            # atomic, per process: two bench workers (different models, same table) can build the
            # same table at once, and a reader must never see a half-written file
            os.makedirs(CACHE, exist_ok=True)
            tmp = "{}.{}.tmp.npz".format(path[:-4], os.getpid())
            t.save_npz(tmp)
            os.replace(tmp, path)
    t.period = np.where(graded, t.period, 0).astype(np.int8)
    t.meta.update({"game": game, "kind": "degraded", "variant": variant, "seed": seed})
    if variant not in PHASE_C_NAMES:
        # caches keyed by variant name (features, learned probabilities) must not outlive a
        # change to the variant's settings; Phase C variants keep their names (and caches)
        t.meta["variant_key"] = _key(variant, seed, game)
    return t, gts


def alive_dead_scores(gt, pred):
    """F1 of the dead class, and of alive, on graded frames (pred X counts as alive)."""
    g = gt != "X"
    gd, pd_ = gt[g] == "D", pred[g] == "D"
    tp = float((gd & pd_).sum())
    prec = tp / max(1.0, pd_.sum())
    rec = tp / max(1.0, gd.sum())
    f1d = 2 * prec * rec / max(1e-9, prec + rec)
    ga, pa = ~gd, ~pd_
    tpa = float((ga & pa).sum())
    pa_, ra_ = tpa / max(1.0, pa.sum()), tpa / max(1.0, ga.sum())
    f1a = 2 * pa_ * ra_ / max(1e-9, pa_ + ra_)
    return {"dead_f1": f1d, "dead_precision": prec, "dead_recall": rec, "alive_f1": f1a,
            "alive_dead_macro": 0.5 * (f1d + f1a)}


def _run(args):
    model_name, variant, seed, games, bootstrap_n, params, label = args
    from . import registry
    fn = registry.get(model_name)
    preds, gts = {}, {}
    t0 = time.time()
    for g in games:
        t, gts[g] = degraded_table(variant, seed, g)
        preds[g] = np.asarray(fn(t, **(params or {})))
        if preds[g].shape != (t.n_frames,):
            raise ValueError("{} returned shape {}".format(model_name, preds[g].shape))
    out = {"model": label, "base_model": model_name, "variant": variant, "seed": seed, "params": params or {},
           "elapsed_s": time.time() - t0, "scores": {}, "deadball": {}}
    for gk in GT_KEYS:
        out["scores"][gk] = bench._jsonable(score.score_games(
            {g: gts[g][gk] for g in games}, preds, bootstrap_n))
    gd = {g: alive_dead_scores(gts[g]["touch"], preds[g]) for g in games}
    allg = np.concatenate([gts[g]["touch"] for g in games])
    allp = np.concatenate([preds[g] for g in games])
    gd["pooled"] = alive_dead_scores(allg, allp)
    out["deadball"] = gd
    return out


def expand_grid(models, grid, params=None):
    """[(label, model, params)] for every combination of the grid values (one model each)."""
    import itertools
    out = []
    for m in models:
        base = dict((params or {}).get(m) or {})
        if not grid:
            out.append((m, m, base))
            continue
        keys = sorted(grid)
        for combo in itertools.product(*[grid[k] for k in keys]):
            p = dict(base)
            p.update(dict(zip(keys, combo)))
            label = m + "[" + ",".join("{}={}".format(k, v) for k, v in zip(keys, combo)) + "]"
            out.append((label, m, p))
    return out


def run(models, variants, seeds, games=paths.SAMPLE_GAMES, bootstrap_n=300, workers=4,
        params=None, grid=None):
    entries = expand_grid(models, grid, params)
    jobs = [(m, v, s, list(games), bootstrap_n, p, label)
            for label, m, p in entries for v in variants for s in seeds]
    # warm caches in the parent so workers only read
    for g in games:
        sample_io.clean_tracking(g)
    if workers <= 1:
        return [_run(j) for j in jobs]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_run, jobs))


def summarise(results, model, variant):
    rs = [r for r in results if r["model"] == model and r["variant"] == variant]
    if not rs:
        return None
    first = min(rs, key=lambda r: r["seed"])
    out = {"n_seeds": len(rs)}
    for gk in GT_KEYS:
        sp = [r["scores"][gk]["pooled"]["spell_f1_mean"] for r in rs]
        mf = [r["scores"][gk]["pooled"]["macro_f1"] for r in rs]
        p = first["scores"][gk]["pooled"]
        out[gk] = {
            "spell": float(np.mean(sp)), "spell_min": float(np.min(sp)), "spell_max": float(np.max(sp)),
            "macro": float(np.mean(mf)), "macro_min": float(np.min(mf)), "macro_max": float(np.max(mf)),
            "spell_boot": (p["spell_bootstrap_low"], p["spell_bootstrap_high"]),
            "macro_boot": (p["bootstrap_low"], p["bootstrap_high"]),
            "n_pred": float(np.mean([r["scores"][gk]["pooled"]["n_pred_spells"] for r in rs])),
            "n_gt": p["n_gt_spells"],
            "per_game_spell": {g: float(np.mean([r["scores"][gk][g]["spell_f1_mean"] for r in rs]))
                               for g in first["scores"][gk] if g != "pooled"},
        }
    out["dead_f1"] = float(np.mean([r["deadball"]["pooled"]["dead_f1"] for r in rs]))
    out["alive_dead_macro"] = float(np.mean([r["deadball"]["pooled"]["alive_dead_macro"] for r in rs]))
    return out


def fmt_row(model, variant, s):
    def iv(x):
        lo, hi = x
        return "[{:.3f}, {:.3f}]".format(lo, hi) if lo is not None else "[-]"
    c, t = s["controlled"], s["touch"]
    return ("| {} | {} | {:.3f} {} | {:.3f} | {:.3f} {} | {:.3f} | {:.0f} / {} | {:.3f} | {} |".format(
        model, variant, t["spell"], iv(t["spell_boot"]), t["macro"], c["spell"], iv(c["spell_boot"]),
        c["macro"], t["n_pred"], t["n_gt"], s["dead_f1"],
        " / ".join("{:.3f}".format(v) for v in t["per_game_spell"].values())))


HEADER = ("| model | degrader | touch spellF1 [boot] | touch macroF1 | controlled spellF1 [boot] | "
          "controlled macroF1 | pred / GT poss (touch) | dead F1 | touch spellF1 per game |\n"
          "|---|---|---|---|---|---|---|---|---|")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--variants", nargs="+", default=["default"])
    ap.add_argument("--robust", action="store_true", help="use the ROBUST variant set")
    ap.add_argument("--robust-b2", action="store_true", help="use the ROBUST_B2 (calibrated) set")
    ap.add_argument("--robust-c2", action="store_true", help="use the ROBUST_C2 (atd_cal2) set")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--bootstrap", type=int, default=300)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--params", default=None, help="JSON {model: {param: value}}")
    ap.add_argument("--grid", default=None, help="JSON {param: [values]} applied to every model")
    ap.add_argument("--tag", default=None, help="output file stem")
    args = ap.parse_args(argv)
    variants = (list(ROBUST) if args.robust else list(ROBUST_B2) if args.robust_b2
                else list(ROBUST_C2) if args.robust_c2 else args.variants)
    seeds = list(range(args.seed_offset, args.seed_offset + args.seeds))
    params = json.loads(args.params) if args.params else None
    t0 = time.time()
    grid = json.loads(args.grid) if args.grid else None
    res = run(args.models, variants, seeds, bootstrap_n=args.bootstrap, workers=args.workers,
              params=params, grid=grid)
    os.makedirs(OUT, exist_ok=True)
    stem = args.tag or "_".join(args.models)
    labels = [e[0] for e in expand_grid(args.models, grid, params)]
    summ = {m: {v: summarise(res, m, v) for v in variants} for m in labels}
    with open(os.path.join(OUT, stem + ".json"), "w") as fh:
        json.dump(bench._jsonable({"runs": res, "summary": summ, "elapsed_s": time.time() - t0}),
                  fh, indent=1)
    print(HEADER)
    for m in labels:
        for v in variants:
            print(fmt_row(m, v, summ[m][v]))
    print("\n{:.0f} s".format(time.time() - t0))
    return summ


if __name__ == "__main__":
    main()
