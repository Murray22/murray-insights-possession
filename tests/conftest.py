import os

import pytest

from mtah import paths


def sample_available(game=1):
    return os.path.isdir(paths.sample_game_dir(game))


needs_sample = pytest.mark.skipif(
    not all(sample_available(g) for g in paths.SAMPLE_GAMES),
    reason="Metrica sample games not in external/metrica_sample/data")
