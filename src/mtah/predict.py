"""Labels for one FrameTable from a named model version, written as a frame,label CSV.

    # the competition match (once the bundle is in data/ or $MTAH_DATA_DIR):
    PYTHONPATH=src .venv/bin/python -m mtah.predict --model gbm_v1 --competition \
        --out runs/submission/MurrayInsights.csv
    # any EPTS pair:
    PYTHONPATH=src .venv/bin/python -m mtah.predict --model gbm_v1 --epts META.xml RAW.txt --out x.csv
    # a degraded bench table (sample game, degrader variant, seed):
    PYTHONPATH=src .venv/bin/python -m mtah.predict --model gbm_v1 --sample-game 3 --out x.csv

The output goes under runs/ (gitignored): a labels file is a frame-level derivative of the match.
"""

import argparse
import os
import sys

import numpy as np

from . import atd_io, paths, registry, smart_tagging


def write_labels(path, labels):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="\n") as fh:
        fh.write("frame,label\n")
        fh.writelines("{},{}\n".format(i, lab) for i, lab in enumerate(labels))
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="gbm_v1", choices=sorted(registry.MODELS))
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--competition", action="store_true")
    src.add_argument("--epts", nargs=2, metavar=("META", "RAW"))
    src.add_argument("--sample-game", type=int)
    ap.add_argument("--variant", default="default")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smart-tagging", action="store_true",
                    help="pass the competition's Smart Tagging as intervals")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    intervals = None
    if a.competition:
        table = atd_io.load_competition()
        if a.smart_tagging:
            st = atd_io._games_module().asset_paths(data_dir=atd_io.data_dir())["smart_tagging"]
            names = table.meta["team_names"]
            intervals = smart_tagging.load(st, names[0], names[1])
    elif a.epts:
        table = atd_io.load_epts(*a.epts)
    else:
        from . import modelbench
        table, _ = modelbench.degraded_table(a.variant, a.seed, a.sample_game)
    if a.competition and a.model in registry.VIDEO_MODELS:
        # Phase E: the saved video ball track (runs/video/full/, from mtah.video.fullrun) as a
        # separate channel on the ATD's gaps
        from .video import source
        table = source.with_video_channel(table)
        print("video channel: {} frames".format(int(np.isfinite(table.vball[:, 0]).sum())))
    labels = np.asarray(registry.get(a.model)(table, intervals))
    labels = np.where(table.period > 0, labels, "X")
    write_labels(a.out, labels)
    counts = {k: int((labels == k).sum()) for k in "HADX"}
    print("{} rows -> {}  {}".format(len(labels), a.out, counts))
    if a.competition:
        sys.path.insert(0, paths.REF_EVAL_DIR)
        import possession_metrics as pm
        reason = pm.validate_submission(pm.read_label_csv(a.out), len(labels))
        if reason:
            raise SystemExit("validate_submission: " + reason)
        print("validate_submission: OK")


if __name__ == "__main__":
    main()
