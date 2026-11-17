"""The gap-length cache is written by several training workers at once on a fresh clone. Each writer must use its
own temporary file: with one shared name, a second writer can move the file away before the first one's rename,
and training dies with FileNotFoundError (seen 2026-11-10 on a fresh clone, --workers 6)."""
import os

import numpy as np

from mtah import atd_io, atd_profile, paths


def test_gap_cache_survives_a_second_writer(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "RUNS_DIR", str(tmp_path))
    monkeypatch.setattr(atd_io, "load_competition", lambda: None)
    monkeypatch.setattr(atd_io, "ball_gap_lengths", lambda table: np.array([12, 268, 845]))
    real_replace = os.replace
    raced = {"done": False}

    def racing_replace(src, dst):
        # another worker computes and publishes the same cache between this writer's save and its rename
        if not raced["done"]:
            raced["done"] = True
            atd_profile.real_gap_lengths(cache=True)
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", racing_replace)
    out = atd_profile.real_gap_lengths(cache=True)
    assert raced["done"]
    assert list(out) == [12, 268, 845]
    assert list(np.load(tmp_path / "cache" / "atd_gap_lengths.npy")) == [12, 268, 845]
    assert not [p for p in os.listdir(tmp_path / "cache") if "tmp" in p]
