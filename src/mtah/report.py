"""Markdown tables from a modelbench result file (runs/phaseC/<tag>.json), for docs/.

    PYTHONPATH=src .venv/bin/python -m mtah.report runs/phaseC/final.json
    PYTHONPATH=src .venv/bin/python -m mtah.report --main atd_cal runs/phaseC/sweep_*.json

Several files are merged (per model, per variant; a later file wins on a clash).
"""

import json
import sys

import numpy as np


def _iv(x):
    lo, hi = x
    return "[{:.3f}, {:.3f}]".format(lo, hi) if lo is not None else "[-]"


def main_table(summary, variant="default"):
    rows = ["| model | touch spellF1 [boot 5-95] (seeds min-max) | touch macroF1 | controlled spellF1 "
            "[boot] | controlled macroF1 | pred poss (touch GT 1146, controlled 697) | dead F1 | "
            "touch spellF1 game 1 / 2 / 3 |", "|---|---|---|---|---|---|---|---|"]
    for m, by_v in summary.items():
        s = by_v.get(variant)
        if not s:
            continue
        t, c = s["touch"], s["controlled"]
        rows.append("| {} | {:.3f} {} ({:.3f}-{:.3f}) | {:.3f} | {:.3f} {} | {:.3f} | {:.0f} | {:.3f} | {} |"
                    .format(m, t["spell"], _iv(t["spell_boot"]), t["spell_min"], t["spell_max"],
                            t["macro"], c["spell"], _iv(c["spell_boot"]), c["macro"], t["n_pred"],
                            s["dead_f1"], " / ".join("{:.3f}".format(v)
                                                     for v in t["per_game_spell"].values())))
    return "\n".join(rows)


def robust_table(summary, key="touch", what="spell"):
    variants = list(next(iter(summary.values())).keys())
    head = "| model | " + " | ".join(variants) + " | min | max |"
    rows = [head, "|" + "---|" * (len(variants) + 3)]
    for m, by_v in summary.items():
        vals = []
        for v in variants:
            s = by_v.get(v)
            if s is None:
                vals.append(np.nan)
            elif what == "dead":
                vals.append(s["dead_f1"])
            else:
                vals.append(s[key][what])
        rows.append("| {} | ".format(m) + " | ".join("{:.3f}".format(x) for x in vals)
                    + " | {:.3f} | {:.3f} |".format(np.nanmin(vals), np.nanmax(vals)))
    return "\n".join(rows)


def merge(files):
    out = {}
    for path in files:
        for m, by_v in json.load(open(path))["summary"].items():
            for v, x in by_v.items():
                if x is not None:
                    out.setdefault(m, {})[v] = x
    return out


def main(argv=None):
    argv = list(argv or sys.argv[1:])
    variant = "default"
    if argv and argv[0] == "--main":
        variant, argv = argv[1], argv[2:]
    s = merge(argv)
    print("## {} degrader\n".format(variant))
    print(main_table(s, variant))
    for key in ("touch", "controlled"):
        print("\n## pooled spellF1, GT {}, mean over seeds\n".format(key))
        print(robust_table(s, key, "spell"))
    print("\n## pooled macroF1, GT touch\n")
    print(robust_table(s, "touch", "macro"))
    print("\n## dead F1 (alive/dead, touch GT)\n")
    print(robust_table(s, what="dead"))


if __name__ == "__main__":
    main()
