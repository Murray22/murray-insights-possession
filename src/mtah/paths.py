"""Where things live. Everything is relative to the project root unless an env var says otherwise."""

import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REF_DIR = os.path.join(ROOT, "ref")
REF_EVAL_DIR = os.path.join(REF_DIR, "evaluation")
SAMPLE_DIR = os.environ.get("MTAH_SAMPLE_DIR") or os.path.join(
    ROOT, "external", "metrica_sample", "data")
RUNS_DIR = os.path.join(ROOT, "runs")

FPS = 25
SAMPLE_GAMES = (1, 2, 3)
# Metrica's sample games are documented as 105 x 68 m (external/metrica_sample/README.md).
SAMPLE_PITCH = (105.0, 68.0)


def sample_game_dir(game):
    return os.path.join(SAMPLE_DIR, "Sample_Game_{}".format(game))
