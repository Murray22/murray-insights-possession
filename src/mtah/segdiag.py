"""Phase D diagnosis: which ground-truth possessions does the model miss, and why?

    PYTHONPATH=src .venv/bin/python -m mtah.segdiag --model gbm_v5_nogap --variant atd_cal --seeds 3

For every GT possession (maximal H or A run, as the organiser's scorer sees it) on the bench it
records the length, whether it was matched (IoU 0.3 / 0.5 / 0.7, one-to-one greedy, the scorer's
own matching), what surrounds it (D, the other team, the same team across a D, A-B-A), how much of
it had the ball visible, and two answers to "emissions or decoder?":

- `p_own`: the mean model probability of the possession's team over its frames, and `frac_fav`, the
  share of its frames where that team beats the other team and D (argmax);
- `hit_at_cost[c]`: whether the possession is matched at IoU 0.5 when the same probabilities are
  decoded with a lower H<->A switching cost c. A miss that a cost of 0-2 recovers is the decoder's;
  a miss that even the per-frame argmax does not recover is the emissions'.

Bench only: degraded sample games 1-3, leave-one-game-out probabilities. Nothing about BAR-ATH.
"""

import argparse
import json
import os

import numpy as np

from . import modelbench as mb
from . import paths, score
from .models import hmm, learned

pm = score.pm
CODE = {"H": 0, "A": 1, "D": 2, "X": 3}


def bench_item(variant, seed, game, tag="gbm5_nogap", gt_key="touch"):
    """(table, gts, P, prior) for one bench table; P from the learned-probability cache."""
    t, gts = mb.degraded_table(variant, seed, game)
    P = learned.probabilities(t, tag, gt_key)
    obj = learned.fit(learned.fold_games(t), gt_key, tag)
    return t, gts, P, obj["prior"]


def decode_labels(P, prior, period, alpha=0.5, cost_team=8.0, cost_dead=12.0):
    emit = learned.emissions_from_P(P, prior, alpha)
    cost = np.array([[0.0, cost_team, cost_dead], [cost_team, 0.0, cost_dead],
                     [cost_dead, cost_dead, 0.0]])
    path, _ = hmm.decode(emit, cost, period)
    return np.where(path >= 0, learned.LABELS[np.clip(path, 0, 2)], "X")


def spells_and_flags(gt, pred, ious=(0.3, 0.5, 0.7)):
    """GT spells, pred spells, {iou: matched flags over GT spells}, best same-team IoU per GT spell."""
    g, p = score.codes(gt), score.codes(pred)
    runs = pm.valid_runs(g)
    p = pm.fill_abstentions(p, runs)
    gs, ps = pm.possession_spells(g, runs), pm.possession_spells(p, runs)
    ov = pm.spell_overlaps(gs, ps)
    flags = {i: pm.matched_gt_flags(ov, len(gs), i) for i in ious}
    best = np.zeros(len(gs))
    longest = np.zeros(len(gs))         # length of the longest same-team pred spell overlapping it
    for iou, gi, pj in ov:
        best[gi] = max(best[gi], iou)
        longest[gi] = max(longest[gi], ps[pj][1] - ps[pj][0])
    return gs, ps, flags, best, longest, runs


def describe(gt, vis, gs, runs):
    """Context of each GT spell: previous / next segment kinds and the A-B-A pattern."""
    g = score.codes(gt)
    segs = pm.segments(g, runs)
    idx = {(a, b): k for k, (a, b, c) in enumerate(segs)}
    rows = []
    for a, b, c in gs:
        k = idx[(a, b)]
        prev = segs[k - 1] if k > 0 and segs[k - 1][1] == a else None
        nxt = segs[k + 1] if k + 1 < len(segs) and segs[k + 1][0] == b else None

        def kind(s, other_side):
            if s is None:
                return "edge"
            if s[2] == 2:
                # across the D: which team comes next on that side?
                j = other_side
                if 0 <= j < len(segs) and segs[j][2] in (0, 1) and (segs[j][0] == s[1] or segs[j][1] == s[0]):
                    return "D_same" if segs[j][2] == c else "D_other"
                return "D_edge"
            return "other"
        pk = kind(prev, k - 2)
        nk = kind(nxt, k + 2)
        aba = pk == "other" and nk == "other" and prev[2] == nxt[2]
        rows.append({"start": a, "len": b - a, "team": c, "prev": pk, "next": nk, "aba": bool(aba),
                     "prev_len": (prev[1] - prev[0]) if prev else 0,
                     "vis": float(vis[a:b].mean())})
    return rows


def diagnose(variant="atd_cal", seeds=(0, 1, 2), tag="gbm5_nogap", alpha=0.5, cost_team=8.0,
             cost_dead=12.0, low_costs=(0.0, 1.0, 2.0, 4.0)):
    rows = []
    for s in seeds:
        for game in paths.SAMPLE_GAMES:
            t, gts, P, prior = bench_item(variant, s, game, tag)
            gt = gts["touch"]
            vis = t.ball_visible()
            pred = decode_labels(P, prior, t.period, alpha, cost_team, cost_dead)
            gs, ps, flags, best, longest, runs = spells_and_flags(gt, pred)
            ctx = describe(gt, vis, gs, runs)
            hits = {}
            for c in low_costs:
                pl = decode_labels(P, prior, t.period, alpha, c, cost_dead)
                hits[c] = spells_and_flags(gt, pl)[2][0.5]
            am = np.nanargmax(np.nan_to_num(P, nan=-1), axis=1)
            am_lab = np.where(t.period > 0, learned.LABELS[am], "X")
            hits["argmax"] = spells_and_flags(gt, am_lab)[2][0.5]
            for i, (a, b, c) in enumerate(gs):
                r = ctx[i]
                pp = P[a:b]
                other = 1 - c
                r.update({
                    "seed": s, "game": game,
                    "hit30": bool(flags[0.3][i]), "hit50": bool(flags[0.5][i]), "hit70": bool(flags[0.7][i]),
                    "best_iou": float(best[i]), "pred_len": float(longest[i]),
                    "p_own": float(np.nanmean(pp[:, c])), "p_other": float(np.nanmean(pp[:, other])),
                    "p_dead": float(np.nanmean(pp[:, 2])),
                    "frac_fav": float(np.mean(np.nanargmax(np.nan_to_num(pp, nan=-1), axis=1) == c)),
                    "max_own_margin": float(np.nanmax(pp[:, c] - pp[:, other])),
                })
                for key, f in hits.items():
                    r["hit_c{}".format(key)] = bool(f[i])
                rows.append(r)
    return rows


def bucket_len(n):
    s = n / 25.0
    for lim, name in ((1, "<1s"), (3, "1-3s"), (6, "3-6s"), (12, "6-12s"), (30, "12-30s")):
        if s < lim:
            return name
    return ">=30s"


BUCKETS = ("<1s", "1-3s", "3-6s", "6-12s", "12-30s", ">=30s")


def summary(rows, low_costs=(0.0, 1.0, 2.0, 4.0)):
    """Printable tables (markdown) of the diagnosis."""
    out = []
    n = len(rows)
    miss = [r for r in rows if not r["hit50"]]
    out.append("GT spells {} (over seeds), matched at IoU 0.5: {:.3f}".format(n, 1 - len(miss) / n))
    out.append("\n| length | GT spells | share | hit@0.3 | hit@0.5 | hit@0.7 | ball visible | p_own | frac_fav "
               "| hit@0.5 cost 4 | cost 2 | cost 0 | argmax |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for bname in BUCKETS:
        rs = [r for r in rows if bucket_len(r["len"]) == bname]
        if not rs:
            continue
        m = lambda k: np.mean([r[k] for r in rs])
        out.append("| {} | {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.2f} | {:.2f} | {:.2f} | {:.3f} | {:.3f} | "
                   "{:.3f} | {:.3f} |".format(bname, len(rs), len(rs) / n, m("hit30"), m("hit50"), m("hit70"),
                                              m("vis"), m("p_own"), m("frac_fav"), m("hit_c4.0"),
                                              m("hit_c2.0"), m("hit_c0.0"), m("hit_cargmax")))
    # context of misses
    out.append("\n| context (prev / next) | GT spells | hit@0.5 | median len (s) | missed |")
    out.append("|---|---|---|---|---|")

    def ctx(r):
        if r["aba"]:
            return "A-B-A (other team both sides)"
        ks = sorted((r["prev"], r["next"]))
        if "D_same" in ks:
            return "same team across a D on one side"
        if ks == ["other", "other"]:
            return "other team both sides (different)"
        if "other" in ks:
            return "other team one side, D / edge the other"
        return "D / edge both sides"
    groups = {}
    for r in rows:
        groups.setdefault(ctx(r), []).append(r)
    for k, rs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        out.append("| {} | {} | {:.3f} | {:.1f} | {} |".format(
            k, len(rs), np.mean([r["hit50"] for r in rs]), np.median([r["len"] for r in rs]) / 25,
            sum(not r["hit50"] for r in rs)))
    # mode of miss
    out.append("\n| miss mode (IoU 0.5) | count | share of misses | median GT len (s) |")
    out.append("|---|---|---|---|")
    modes = {}
    for r in miss:
        if r["best_iou"] == 0:
            m = "absent: no same-team predicted run overlaps"
        elif r["pred_len"] > 2 * r["len"]:
            m = "swallowed: the overlapping run is over 2x longer"
        else:
            m = "boundary: overlap exists, IoU too low"
        modes.setdefault(m, []).append(r)
    for k, rs in sorted(modes.items(), key=lambda kv: -len(kv[1])):
        out.append("| {} | {} | {:.3f} | {:.1f} |".format(k, len(rs), len(rs) / max(1, len(miss)),
                                                          np.median([r["len"] for r in rs]) / 25))
    # emissions vs decoder, misses only
    out.append("\n| missed at cost 8 (IoU 0.5) | count | share |")
    out.append("|---|---|---|")
    cats = {"recovered at cost 4": 0, "recovered at cost 2 (not 4)": 0, "recovered at cost 0-1 only": 0,
            "argmax only": 0, "never (emissions)": 0}
    for r in miss:
        if r["hit_c4.0"]:
            cats["recovered at cost 4"] += 1
        elif r["hit_c2.0"]:
            cats["recovered at cost 2 (not 4)"] += 1
        elif r["hit_c0.0"] or r["hit_c1.0"]:
            cats["recovered at cost 0-1 only"] += 1
        elif r["hit_cargmax"]:
            cats["argmax only"] += 1
        else:
            cats["never (emissions)"] += 1
    for k, v in cats.items():
        out.append("| {} | {} | {:.3f} |".format(k, v, v / max(1, len(miss))))
    fav = [r for r in miss]
    out.append("\nmissed spells with frac_fav >= 0.5: {:.3f}; < 0.2: {:.3f}; median p_own {:.2f} "
               "(matched: {:.2f})".format(np.mean([r["frac_fav"] >= 0.5 for r in fav]),
                                         np.mean([r["frac_fav"] < 0.2 for r in fav]),
                                         np.median([r["p_own"] for r in fav]),
                                         np.median([r["p_own"] for r in rows if r["hit50"]])))
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", default="gbm5_nogap")
    ap.add_argument("--variant", default="atd_cal")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default=os.path.join(paths.RUNS_DIR, "phaseD", "diag.json"))
    a = ap.parse_args(argv)
    rows = diagnose(a.variant, tuple(range(a.seeds)), a.tag)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(rows, fh)
    print(summary(rows))


if __name__ == "__main__":
    main()
