"""The sample-game tracking cache is built by several training workers at once on a fresh clone. It must be written
under a per-process temporary name and renamed into place: written straight to its final name, a second worker can
find the file half-written and die with BadZipFile (seen 2026-11-13 on a fresh run, --workers 6)."""
import os

import numpy as np

from mtah import sample_io
from mtah.frames import FrameTable


def _table():
    n = 50
    return FrameTable(np.arange(n), np.ones(n, dtype=np.int8), np.zeros((n, 2)), np.zeros((n, 2, 2)),
                      np.array([0, 1], dtype=np.int8), (105.0, 68.0), None, {})


def test_clean_cache_is_never_written_in_place(tmp_path, monkeypatch):
    monkeypatch.setattr(sample_io, "CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(sample_io, "_csv_tracking", lambda game: _table())
    final = os.path.join(str(tmp_path), "sample_game_1_clean.npz")
    targets = []
    real_save = FrameTable.save_npz
    monkeypatch.setattr(FrameTable, "save_npz", lambda self, path: (targets.append(path), real_save(self, path))[1])
    real_replace = os.replace
    raced = {"done": False}

    def racing_replace(src, dst):
        # another worker asks for the same game between this writer's save and its rename
        if not raced["done"]:
            raced["done"] = True
            assert not os.path.exists(final)
            sample_io.clean_tracking(1)
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", racing_replace)
    out = sample_io.clean_tracking(1)
    assert raced["done"]
    assert final not in targets
    assert out.n_frames == 50
    assert FrameTable.load_npz(final).n_frames == 50
    assert not [p for p in os.listdir(tmp_path) if "tmp" in p]
