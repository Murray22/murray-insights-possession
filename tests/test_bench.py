"""The fidelity bench's bookkeeping: both GT variants scored, summaries ordered correctly."""

import numpy as np
import pytest

from mtah import baseline, bench
from test_score import synthetic_gt


def test_score_all_covers_both_gt_variants_and_can_tell_baselines_apart():
    gt = synthetic_gt(n_poss=300)
    gts = {1: {k: gt for k in bench.GT_VARIANTS}}
    perfect = {name: gt for name, _ in baseline.VARIANTS}
    perfect["baseline_raw"] = np.where(gt == "X", "X", "D")          # a constant guess
    out = bench._score_all(gts, {1: perfect}, [1], 0)
    assert set(out) == set(bench.GT_VARIANTS)
    for gk in out:
        assert out[gk]["baseline_smooth50"]["pooled"]["spell_f1_mean"] == 1.0
        assert out[gk]["baseline_raw"]["pooled"]["spell_f1_mean"] == 0.0


def test_summarise_means_and_order():
    def fake(seed, values):
        sc = {name: {"1": {"spell_f1_mean": v, "macro_f1": v},
                     "pooled": {"spell_f1_mean": v, "macro_f1": v, "spell_bootstrap_low": v - .01,
                                "spell_bootstrap_high": v + .01, "bootstrap_low": v - .01,
                                "bootstrap_high": v + .01, "n_pred_spells": 10, "n_gt_spells": 9}}
              for (name, _), v in zip(baseline.VARIANTS, values)}
        return {"variant": "default", "seed": seed, "scores": {"controlled": sc}}
    runs = [fake(0, [0.1, 0.2, 0.4, 0.3]), fake(1, [0.1, 0.2, 0.2, 0.3])]
    s = bench.summarise(runs, "default", "controlled")
    assert s["baseline_smooth50"]["spell_mean"] == pytest.approx(0.3)
    assert s["baseline_smooth50"]["spell_min"] == 0.2
    order = bench._order(s, "spell_mean")
    assert order.split(" > ")[0] in ("smooth50", "smooth125")
    assert order.endswith("raw")
