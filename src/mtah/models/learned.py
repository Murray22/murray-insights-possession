"""Learned emissions: gradient boosting P(H, A, D | frame features), decoded by the segment Viterbi.

Training data is degraded sample games only, never clean tracking (where ball absence leaks the
dead-ball label) and never BAR-ATH. Every training table comes from a mixture of degrader settings
(TRAIN_MIX), all with the randomised dead-ball path (degrade.DegradeConfig.dead_ball_path =
"random"), so no single synthetic dead-ball trajectory or visibility level can be learned as "the"
dead-ball signature. Two families are held out of the mixture on purpose (long ball bursts and
heavy position noise) so the bench can see how the model does on settings it never trained on.
Training seeds (10000+) never overlap the bench's seeds (0-9).

Each table is used twice, as is and with the teams swapped (slot teams flipped, H and A labels
swapped), and at prediction time the two readings are averaged, so the model is exactly symmetric
in home and away and cannot learn a home-possession prior from three games.

Leave-one-game-out: `fold_for(table)` picks the model trained without the table's sample game; a
table from any other source (BAR-ATH) gets the model trained on all three sample games.

Reproducibility: HistGradientBoosting with a fixed random_state, no early stopping (which would
draw a random validation split), and one thread (threadpoolctl), so the fitted trees and every
probability are the same on every run.
"""

import hashlib
import json
import os
import pickle
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

import numpy as np

from .. import degrade, paths
from .. import features as F
from . import hmm

MODEL_DIR = os.path.join(paths.RUNS_DIR, "models")
TRAIN_CACHE = os.path.join(paths.RUNS_DIR, "cache", "train")
LABEL_CODE = {"H": 0, "A": 1, "D": 2}
LABELS = np.array(["H", "A", "D"])

TRAIN_MIX = {
    "mix_default": {},
    "mix_vis0": {"dead_ball_visible": 0.0},
    "mix_vis50": {"dead_ball_visible": 0.5},
    "mix_bursts": {"dead_ball_pattern": "bursts"},
    "mix_short": {"ball_burst_median": 5.0, "ball_burst_sigma": 0.7},
    "mix_noise25": {"pos_noise_m": 0.25, "ball_noise_m": 0.35},
}
# gbm4 onwards: the same mixture with the dead-ball rollout (degrade.DegradeConfig) on half of it,
# so the synthetic ball no longer stops dead on the first frame of every stoppage
TRAIN_MIX_ROLL = {
    "mixr_default": {"dead_ball_rollout": True},
    "mix_vis0": {"dead_ball_visible": 0.0},
    "mixr_vis50": {"dead_ball_visible": 0.5, "dead_ball_rollout": True},
    "mixr_bursts": {"dead_ball_pattern": "bursts", "dead_ball_rollout": True},
    "mix_short": {"ball_burst_median": 5.0, "ball_burst_sigma": 0.7},
    "mix_noise25": {"pos_noise_m": 0.25, "ball_noise_m": 0.35},
}
# gbm5 onwards (Phase B2): the degrader calibrated to the real ATD (modelbench.ATD_CAL) and
# variations around it. Held out of training on purpose: ball missing 35% / 47%, no mass player
# dropouts and the Phase A degrader, so the bench sees settings the model never trained on.
CAL_MIX_OVERRIDES = {
    "cal_default": {},
    "cal_vis15": {"dead_ball_visible": 0.15},
    "cal_vis50": {"dead_ball_visible": 0.50},
    "cal_noisy": {"ball_noise_m": 1.0, "pos_noise_m": 0.4},
    "cal_noctx": {"slow_weight": 1.0, "fast_weight": 1.0, "crowd_weight": 1.0, "line_weight": 1.0,
                  "deep_weight": 1.0, "dead_out_weight": 1.0},
    "cal_restart60": {"restart_reacq_prob": 0.6},
}
TAG_MIX = {"gbm4": TRAIN_MIX_ROLL}
CAL_TAGS = {"gbm5", "gbm5a", "gbm5_nop", "gbm5_nogap", "gbm6_w05", "gbm6_w1", "gbm7", "gbm7_w05"}
# Phase E: the same six-setting mixture on atd_cal2 (the ball-shape fit), and on atd_cal2v (plus
# the simulated video channel). The video settings vary with the mixture entry so the model sees
# fill shares from 0.1 to 0.6 and a noisier video track; held out of training: fill shares 0, 0.2
# and 0.6 at the default settings, double video error, no spare balls, triple spare balls.
CAL2_TAGS = {"gbm5n_c2", "gbm7_c2"}
CAL2V_TAGS = {"gbm8", "gbm8n"}
VIDEO_MIX = {
    "cal_default": {},
    "cal_vis15": {"video_fill_share": 0.25},
    "cal_vis50": {"video_fill_share": 0.55},
    "cal_noisy": {"video_error_scale": 1.5},
    "cal_noctx": {"video_fill_share": 0.10},
    "cal_restart60": {"video_fill_share": 0.45, "video_spare_share": 0.05},
}
# feature set per tag: "v4" = features.compute; "b2" = plus features_b2.compute_extra
TAG_FSET = {"gbm5": "b2", "gbm5_nop": "b2", "gbm5_nogap": "b2", "gbm6_w05": "b2", "gbm6_w1": "b2",
            # Phase D: plus the touch-kink features (features_d)
            "gbm7": "d", "gbm7_w05": "d",
            # Phase E: retrained on atd_cal2; gbm8 / gbm8n add the video channel (features_v)
            "gbm5n_c2": "b2", "gbm7_c2": "d", "gbm8": "dv", "gbm8n": "v"}
# Phase D: possession-length weighting of the training rows (spell_weights). spellF1 counts each
# possession once, a per-frame loss counts it once per frame, so short possessions are nearly
# invisible to the fit. gamma 0 = unweighted (every earlier tag).
TAG_WEIGHT = {"gbm6_w05": 0.5, "gbm6_w1": 1.0, "gbm7_w05": 0.5}
WEIGHT_MIN_FRAMES = 25             # possessions shorter than 1 s are weighted as 1 s long


def mix_for(tag):
    if tag in CAL2_TAGS or tag in CAL2V_TAGS:
        from .. import modelbench as mb
        base = mb.ATD_CAL2V if tag in CAL2V_TAGS else mb.ATD_CAL2
        return {k: dict(base, **v, **(VIDEO_MIX[k] if tag in CAL2V_TAGS else {}))
                for k, v in CAL_MIX_OVERRIDES.items()}
    if tag in CAL_TAGS:
        from .. import modelbench as mb
        return {k: dict(mb.ATD_CAL, **v) for k, v in CAL_MIX_OVERRIDES.items()}
    return TAG_MIX.get(tag, TRAIN_MIX)


def fset_for(tag):
    return TAG_FSET.get(tag, "v4")


def _fset_suffix(fset):
    from .. import features_b2, features_d, features_v
    if fset == "v4":
        return ""
    s = "b2{}".format(features_b2.EXTRA_VERSION)
    if fset in ("d", "dv"):
        s += "d{}".format(features_d.EXTRA_VERSION)
    if fset in ("v", "dv"):
        s += "v{}".format(features_v.EXTRA_VERSION)
    return s


def compute_features(table, fset, cached=False):
    f = F.cached(table) if cached else F.compute(table)
    if fset in ("b2", "d", "v", "dv"):
        from .. import features_b2
        f = dict(f)
        f.update(features_b2.compute_extra(table, f))
    if fset in ("d", "dv"):
        from .. import features_d
        f.update(features_d.compute_extra(table))
    if fset in ("v", "dv"):
        from .. import features_v
        f.update(features_v.compute_extra(table, f["home_attack_sign"]))
    return f


def _extra_columns(table, sets):
    """Only the columns of the extra feature sets `sets` ("d", "v") for a table."""
    out = {}
    if "d" in sets:
        from .. import features_d
        out.update(features_d.compute_extra(table))
    if "v" in sets:
        from .. import features_v
        out.update(features_v.compute_extra(table))
    return out


def _base_parts(mix_cfg, fset):
    """[(mix config, base fset, extra sets)] cached parts a part of `fset` can be built from by
    adding columns: the same degraded table (video options never change the ATD part, so they are
    dropped from the base's config) with fewer feature sets. Most complete first."""
    if fset == "d":
        return [(mix_cfg, "b2", ("d",))]
    no_video = {k: v for k, v in mix_cfg.items() if not k.startswith("video_")}
    if fset == "v":
        return [(no_video, "b2", ("v",))]
    if fset == "dv":
        return [(no_video, "d", ("v",)), (no_video, "b2", ("d", "v"))]
    return []


TRAIN_SEEDS = (0, 1)               # x TRAIN_MIX, per game; degrader seed 10000 + 100 * s + 10 * k + game
STRIDE = 10                        # every 10th graded frame (neighbours are near-duplicates)

# Features the model must not see: absolute positions and directions (they encode the half and the
# game), raw player speeds (their level moves with position noise; the per-game standardised _z
# versions are used instead), and bookkeeping.
EXCLUDE = {
    "bx", "by", "last_bx", "last_by", "next_bx", "next_by", "cx_h", "cx_a", "cy_h", "cy_a",
    "left_h", "left_a", "dcx", "home_attack_sign", "period", "n_present",
    "pspd", "pspd_h", "pspd_a", "pspd_p80", "pspd_1s", "pspd_3s", "pspd_8s", "pspd_p80_3s",
    "pspd_past4s", "pspd_next4s", "pspd_past10s", "pspd_next10s", "pfast", "pfast_3s", "pslow",
}

HGB = dict(max_iter=300, learning_rate=0.1, max_leaf_nodes=31, min_samples_leaf=200,
           l2_regularization=1.0, early_stopping=False, random_state=0)


VIS = {"bvis", "bvis_5s", "bvis_20s", "since_vis", "until_vis", "gap_len"}
Z = {k for k in ("pspd_z", "pspd_1s_z", "pspd_3s_z", "pspd_8s_z", "pspd_p80_3s_z",
                 "pspd_past4s_z", "pspd_next4s_z", "pspd_past10s_z", "pspd_next10s_z",
                 "pfast_3s_z")}
R = {k[:-2] + "_r" for k in Z}
# feature set per model tag: gbm1 = the first version (per-game median / IQR motion);
# gbm1_novis = without any ball-visibility feature (an ablation: how much of the skill rides on
# where the degrader hides the ball); gbm2 = range-scaled motion instead of median / IQR;
# gbm3 = gbm2 without ball-visibility features.
TAG_EXCLUDE = {
    "gbm1": EXCLUDE | R,
    "gbm1_novis": EXCLUDE | R | VIS,
    "gbm2": EXCLUDE | Z,
    # gbm3: both robustness choices at once (range-scaled motion, no ball-visibility features);
    # the bench cannot separate gbm1 / gbm1_novis / gbm2, so take the one that assumes least
    "gbm3": EXCLUDE | Z | VIS,
    # gbm4: gbm3's features, trained on the mixture with the dead-ball rollout (TRAIN_MIX_ROLL)
    "gbm4": EXCLUDE | Z | VIS,
    # gbm5a: gbm4's features retrained on the calibrated mixture (isolates the retraining)
    "gbm5a": EXCLUDE | Z | VIS,
    # gbm5: plus the gap / player features (features_b2) and, now that the degrader's gaps look
    # like the real ones, the ball-visibility features again
    "gbm5": EXCLUDE | Z,
}
# gbm5 ablations (same training rows): without the B2 player features, and without anything about
# the gap (ball-visibility features and the B2 gap / gap-end features)
B2_PLAYER = {"b2_{}_{}".format(k, t) for k in ("adv", "cvel", "fwd", "len", "wid", "comp", "opphalf",
                                                "line", "press") for t in ("h", "a", "d")}
B2_PLAYER |= {"b2_{}_{}".format(k, w) for k in ("cvel_d", "fwd_d", "adv_d", "opphalf_d", "press_d",
                                                 "comp_d", "len_d", "line_d") for w in ("2s", "5s")}
B2_PLAYER |= {"b2_cvel_d_past3s", "b2_cvel_d_next3s"}
B2_GAP = {"b2_{}_{}".format(s_, k) for s_ in ("last", "next") for k in ("by_c", "dline", "bspeed", "dmin")}
B2_GAP |= {"b2_gap_disp", "b2_gap_dx_att", "b2_gap_frac", "b2_gap_s"}
TAG_EXCLUDE["gbm5_nop"] = EXCLUDE | Z | B2_PLAYER
TAG_EXCLUDE["gbm5_nogap"] = EXCLUDE | Z | VIS | B2_GAP
# Phase D: gbm5_nogap's features and training rows, weighted by possession length
TAG_EXCLUDE["gbm6_w05"] = TAG_EXCLUDE["gbm5_nogap"]
TAG_EXCLUDE["gbm6_w1"] = TAG_EXCLUDE["gbm5_nogap"]
# gbm7: gbm5_nogap's features plus every features_d column (fset "d")
TAG_EXCLUDE["gbm7"] = TAG_EXCLUDE["gbm5_nogap"]
TAG_EXCLUDE["gbm7_w05"] = TAG_EXCLUDE["gbm5_nogap"]
# Phase E: the same designs retrained on atd_cal2, and with the video channel (all v_ columns)
for _t in ("gbm5n_c2", "gbm7_c2", "gbm8", "gbm8n"):
    TAG_EXCLUDE[_t] = TAG_EXCLUDE["gbm5_nogap"]


def feature_names(f, tag="gbm1"):
    ex = TAG_EXCLUDE[tag]
    return sorted(k for k in f if k not in ex)


def all_feature_names(f):
    return sorted(f)


def swapped(table):
    t = table.copy()
    t.slot_team = (1 - t.slot_team).astype(np.int8)
    t.meta = dict(table.meta)
    t.meta["swapped"] = True
    return t


def swap_labels(labels):
    out = labels.copy()
    out[labels == "H"] = "A"
    out[labels == "A"] = "H"
    return out


def _train_part(args):
    """Design rows for one (game, mix config, seed): as is and swapped. Cached on disk."""
    game, mix, seed, gt_key = args[:4]
    mixes = args[4] if len(args) > 4 else TRAIN_MIX
    fset = args[5] if len(args) > 5 else "v4"
    from .. import modelbench as mb
    k = sorted(mixes).index(mix)
    dseed = 10000 + 100 * seed + 10 * k + game          # the degrader seed, part of the cache key

    def key_of(cfg, fs):
        return "{}_{}_d{}_{}_v{}".format(mix, hashlib.md5(json.dumps(cfg, sort_keys=True)
                                                          .encode()).hexdigest()[:6],
                                         dseed, gt_key, F.FEATURE_VERSION) + _fset_suffix(fs)
    key = key_of(mixes[mix], fset)
    path = os.path.join(TRAIN_CACHE, key + ".npz")
    if os.path.isfile(path):
        z = np.load(path, allow_pickle=False)
        return z["X"], z["y"], list(z["names"])
    base, extra_sets = None, None
    for bcfg, bfs, ex in _base_parts(mixes[mix], fset):
        # the cached base part of this table: compute only the missing columns
        bpath = os.path.join(TRAIN_CACHE, key_of(bcfg, bfs) + ".npz")
        if os.path.isfile(bpath):
            z = np.load(bpath, allow_pickle=False)
            base, extra_sets = (z["X"], z["y"], list(z["names"])), ex
            break
    clean, gts, graded = mb.game_truth(game)
    cfg = degrade.make_config(dict(dict(dead_ball_path="random"), **mixes[mix]), seed=dseed)
    t, _ = degrade.degrade(clean, graded, cfg)
    t.period = np.where(graded, t.period, 0).astype(np.int8)
    gt = gts[gt_key]
    rng = np.random.default_rng(cfg.seed)
    rows = np.flatnonzero(graded)[int(rng.integers(0, STRIDE))::STRIDE]
    Xs, ys = [], []
    names = None
    for tab, lab in ((t, gt), (swapped(t), swap_labels(gt))):
        if base is None:
            f = compute_features(tab, fset)
        else:
            f = _extra_columns(tab, extra_sets)
        names = all_feature_names(f)
        X, _ = F.matrix(f, names)
        Xs.append(X[rows])
        ys.append(np.array([LABEL_CODE[c] for c in lab[rows]], dtype=np.int8))
    X, y = np.concatenate(Xs), np.concatenate(ys)
    if base is not None:
        if not np.array_equal(base[1], y):
            raise ValueError("cached base part for {} does not line up with its rows".format(key))
        combined = base[2] + names
        names = sorted(combined)                      # the column order a full compute gives
        X = np.concatenate([base[0], X], axis=1)[:, [combined.index(k) for k in names]]
    os.makedirs(TRAIN_CACHE, exist_ok=True)
    tmp = "{}.{}.tmp.npz".format(path, os.getpid())
    np.savez_compressed(tmp, X=X, y=y, names=np.array(names))
    os.replace(tmp, path)
    return X, y, names


def possession_lengths(labels):
    """(n,) length in frames of the maximal H or A run each frame sits in (0 on D / X frames).
    Runs are cut wherever the label changes, as the organiser's possession_spells cuts them."""
    lab = np.asarray(labels)
    out = np.zeros(len(lab), dtype=np.int64)
    if not len(lab):
        return out
    b = np.flatnonzero(lab[1:] != lab[:-1]) + 1
    bounds = np.concatenate(([0], b, [len(lab)]))
    for a, e in zip(bounds[:-1], bounds[1:]):
        if lab[a] in ("H", "A"):
            out[a:e] = e - a
    return out


def row_weights(lengths, gamma, min_frames=WEIGHT_MIN_FRAMES):
    """Weights for training rows: (1 / max(L, min_frames)) ** gamma on possession rows (L > 0),
    rescaled so possession rows keep their total weight (their mean is 1); D rows weigh 1.
    The H / A / D balance the model sees is therefore unchanged, only short possessions count more."""
    L = np.asarray(lengths, dtype=np.float64)
    w = np.ones(len(L))
    alive = L > 0
    if gamma and alive.any():
        raw = np.maximum(L[alive], min_frames) ** (-float(gamma))
        w[alive] = raw / raw.mean()
    return w


def _part_rows(game, mix, seed, gt_key, mixes):
    """The training rows of one part and their GT labels, exactly as _train_part draws them."""
    from .. import modelbench as mb
    k = sorted(mixes).index(mix)
    dseed = 10000 + 100 * seed + 10 * k + game
    cfg = degrade.make_config(dict(dict(dead_ball_path="random"), **mixes[mix]), seed=dseed)
    _, gts, graded = mb.game_truth(game)
    rng = np.random.default_rng(cfg.seed)
    rows = np.flatnonzero(graded)[int(rng.integers(0, STRIDE))::STRIDE]
    return rows, gts[gt_key]


def training_weights(games, gt_key, mixes, gamma, y=None):
    """Row weights in build_training's order (per part: the rows as is, then swapped). If `y` is
    given it is checked against the GT labels at the recomputed rows, so a drift between this and
    _train_part's row draw fails loudly instead of weighting the wrong rows."""
    Ls = []
    for g in games:
        for m in sorted(mixes):
            for s in TRAIN_SEEDS:
                rows, gt = _part_rows(g, m, s, gt_key, mixes)
                L = possession_lengths(gt)[rows]
                Ls.extend([L, L])
                if y is not None:
                    want = np.array([LABEL_CODE[c] for c in gt[rows]], dtype=np.int8)
                    n0 = sum(len(x) for x in Ls[:-2])
                    got = y[n0:n0 + len(rows)]
                    sw = y[n0 + len(rows):n0 + 2 * len(rows)]
                    flip = np.where(want == 0, 1, np.where(want == 1, 0, want))
                    if not (np.array_equal(got, want) and np.array_equal(sw, flip)):
                        raise ValueError("training rows drifted for game {} mix {} seed {}".format(g, m, s))
    L = np.concatenate(Ls)
    if y is not None and len(L) != len(y):
        raise ValueError("weights {} != rows {}".format(len(L), len(y)))
    return row_weights(L, gamma)


def build_training(games, gt_key="touch", workers=4, mixes=None, fset="v4"):
    mixes = mixes or TRAIN_MIX
    jobs = [(g, m, s, gt_key, mixes, fset) for g in games for m in sorted(mixes) for s in TRAIN_SEEDS]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            parts = list(pool.map(_train_part, jobs))
    else:
        parts = [_train_part(j) for j in jobs]
    names = parts[0][2]
    for p in parts:
        if p[2] != names:
            raise ValueError("feature names differ between training parts")
    return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]), names


def model_path(tag, games, gt_key):
    return os.path.join(MODEL_DIR, "{}_{}_{}_v{}{}.pkl".format(
        tag, gt_key, "".join(str(g) for g in sorted(games)), F.FEATURE_VERSION,
        _fset_suffix(fset_for(tag))))


def fit(games, gt_key="touch", tag="gbm1", workers=4, hgb=None):
    """Fit (or load) the model trained on `games`; returns dict(model, names, prior)."""
    path = model_path(tag, games, gt_key)
    if os.path.isfile(path):
        with open(path, "rb") as fh:
            return pickle.load(fh)
    from sklearn.ensemble import HistGradientBoostingClassifier
    from threadpoolctl import threadpool_limits
    X, y, all_names = build_training(games, gt_key, workers, mix_for(tag), fset_for(tag))
    names = [k for k in all_names if k not in TAG_EXCLUDE[tag]]
    X = X[:, [all_names.index(k) for k in names]]
    gamma = TAG_WEIGHT.get(tag, 0.0)
    w = training_weights(games, gt_key, mix_for(tag), gamma, y) if gamma else None
    with threadpool_limits(1):
        clf = HistGradientBoostingClassifier(**dict(HGB, **(hgb or {})))
        clf.fit(X, y, sample_weight=w)
    ww = np.ones(len(y)) if w is None else w
    prior = np.bincount(y, weights=ww, minlength=3) / ww.sum()
    obj = {"model": clf, "names": names, "prior": prior, "games": list(games), "gt_key": gt_key,
           "n_rows": int(len(y)), "hgb": dict(HGB, **(hgb or {})), "weight_gamma": gamma}
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(path + ".tmp", "wb") as fh:
        pickle.dump(obj, fh)
    os.replace(path + ".tmp", path)
    return obj


def fold_games(table):
    g = table.meta.get("game")
    if g in (1, 2, 3) and table.meta.get("kind") == "degraded":
        return tuple(x for x in (1, 2, 3) if x != g)
    return (1, 2, 3)


def probabilities(table, tag="gbm1", gt_key="touch", cache=True):
    """(n, 3) symmetrised P(H, A, D) per frame (NaN outside graded frames)."""
    m = table.meta
    cpath = None
    if cache and all(k in m for k in ("variant", "seed", "game")):
        cpath = os.path.join(paths.RUNS_DIR, "cache", "learned_P", "{}_{}_{}_s{}_g{}_v{}{}.npy".format(
            tag, gt_key, m.get("variant_key", m["variant"]), m["seed"], m["game"], F.FEATURE_VERSION,
            _fset_suffix(fset_for(tag))))
        if os.path.isfile(cpath):
            return np.load(cpath)
    obj = fit(fold_games(table), gt_key, tag)
    from threadpoolctl import threadpool_limits
    graded = table.period > 0
    P = np.full((table.n_frames, 3), np.nan)
    fset = fset_for(tag)
    f1 = compute_features(table, fset, cached=bool(cpath))
    f2 = compute_features(swapped(table), fset)
    X1, _ = F.matrix(f1, obj["names"])
    X2, _ = F.matrix(f2, obj["names"])
    with threadpool_limits(1):
        p1 = obj["model"].predict_proba(X1[graded])
        p2 = obj["model"].predict_proba(X2[graded])
    P[graded, 0] = 0.5 * (p1[:, 0] + p2[:, 1])
    P[graded, 1] = 0.5 * (p1[:, 1] + p2[:, 0])
    P[graded, 2] = 0.5 * (p1[:, 2] + p2[:, 2])
    if cpath:
        os.makedirs(os.path.dirname(cpath), exist_ok=True)
        np.save(cpath + ".tmp.npy", P)
        os.replace(cpath + ".tmp.npy", cpath)
    return P


DEFAULTS = dict(alpha=0.5, cost_team=8.0, cost_dead=12.0, tag="gbm1", gt_key="touch",
                abstain_below=0.0, prior_mode="sym")


def emissions_from_P(P, prior, alpha):
    """alpha * log(P / prior); prior symmetrised over H and A."""
    pr = np.array([0.5 * (prior[0] + prior[1]), 0.5 * (prior[0] + prior[1]), prior[2]])
    with np.errstate(divide="ignore", invalid="ignore"):
        e = alpha * (np.log(np.clip(P, 1e-6, 1.0)) - np.log(pr))
    return np.nan_to_num(e, nan=0.0)


def predict(table, intervals=None, **params):
    from .segment import abstain
    p = dict(DEFAULTS, **params)
    P = probabilities(table, p["tag"], p["gt_key"])
    obj = fit(fold_games(table), p["gt_key"], p["tag"])
    emit = emissions_from_P(P, obj["prior"], p["alpha"])
    ct, cd = p["cost_team"], p["cost_dead"]
    cost = np.array([[0.0, ct, cd], [ct, 0.0, cd], [cd, cd, 0.0]])
    want = p["abstain_below"] > 0
    path, post = hmm.decode(emit, cost, table.period, want_post=want)
    labels = np.where(path >= 0, LABELS[np.clip(path, 0, 2)], "X")
    return abstain(labels, path, post, p["abstain_below"])


FOLDS = [(1, 2), (1, 3), (2, 3), (1, 2, 3)]


def train_all(gt_key="touch", tag="gbm1", workers=6, folds=None):
    """Fit the three LOGO folds and the all-games model, in parallel processes."""
    build_training((1, 2, 3), gt_key, workers, mix_for(tag), fset_for(tag))   # fills the cache once
    folds = folds or FOLDS
    with ProcessPoolExecutor(max_workers=min(4, workers)) as pool:
        list(pool.map(_fit_job, [(g, gt_key, tag) for g in folds]))


def _fit_job(args):
    games, gt_key, tag = args
    obj = fit(games, gt_key, tag, workers=1)
    return model_path(tag, games, gt_key), obj["n_rows"]


if __name__ == "__main__":
    import argparse
    import time
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default="touch")
    ap.add_argument("--tag", nargs="+", default=["gbm1"])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--folds", nargs="+", default=None,
                    help="fold game sets, e.g. 12 13 23 (default: the three LOGO folds and 123)")
    a = ap.parse_args()
    t0 = time.time()
    folds = [tuple(int(c) for c in f) for f in a.folds] if a.folds else None
    for tag in a.tag:
        train_all(a.gt, tag, a.workers, folds)
        print("{} done at {:.0f} s".format(tag, time.time() - t0), flush=True)
    print("trained in {:.0f} s".format(time.time() - t0))
