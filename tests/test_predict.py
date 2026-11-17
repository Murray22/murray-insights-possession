"""The predict CLI: one row per frame from 0, X outside graded time, a labels file the scorer reads."""

import os
import sys

import numpy as np
import pandas as pd

from mtah import paths, predict
from test_atd_io import write_epts


def test_predict_on_epts_pair(tmp_path):
    meta, raw = write_epts(str(tmp_path / "in"), n_frames=400, specs=((1, 400),))
    out = str(tmp_path / "out.csv")
    predict.main(["--model", "seg_v4", "--epts", meta, raw, "--out", out])
    df = pd.read_csv(out)
    assert list(df.columns) == ["frame", "label"]
    assert np.array_equal(df["frame"].values, np.arange(400))
    # graded rows 10..99 (metadata first half 11..100), X elsewhere
    assert (df["label"].values[:10] == "X").all() and (df["label"].values[100:] == "X").all()
    assert set(df["label"].values[10:100]) <= {"H", "A", "D"}
    sys.path.insert(0, paths.REF_EVAL_DIR)
    import possession_metrics as pm
    assert pm.validate_submission(pm.read_label_csv(out), 400) is None
