#!/bin/bash
# Full video ball run: detect on every ATD gap frame (plus 25-frame windows), then track.
# Resumable: re-run to continue a killed job. About 6 h of CPU. Output: runs/video/full/ (gitignored).
#   DET=<video venv python> scripts/run_video.sh     (default ~/.cache/wt-mtah-video/venv/bin/python)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DET="${DET:-$HOME/.cache/wt-mtah-video/venv/bin/python}"
cd "$ROOT/src"
export YOLO_OFFLINE=1
"$ROOT/.venv/bin/python" -m mtah.video.cache
"$DET" -m mtah.video.weights
nice -n 19 "$DET" -m mtah.video.fullrun detect --config soccana_ov --scope gaps --window 25
nice -n 19 "$DET" -m mtah.video.fullrun track --config soccana_ov
echo FULL_RUN_DONE
