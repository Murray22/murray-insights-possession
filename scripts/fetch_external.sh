#!/bin/bash
# Fetch the two public inputs the code needs, pinned and hash-checked:
#   ref/                      the organiser repo (scorer, baselines, public 5-minute GT), commit f8f2629
#   external/metrica_sample/  Metrica's public sample games 1-3 (the training and validation bench), commit e706dd5
# The licensed competition bundle is NOT fetched here: each participant downloads it after accepting the
# data terms, into data/ (or $MTAH_DATA_DIR). See README.md.
#
#   scripts/fetch_external.sh            # into the project root
#   ROOT=/some/dir scripts/fetch_external.sh
set -euo pipefail
ROOT="${ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"

REF_URL=https://github.com/metrica-sports/more-than-a-hack-2027.git
REF_COMMIT=f8f2629
SAMPLE_COMMIT=e706dd506b360d69d9d123d5b8026e7294b13996
RAW=https://raw.githubusercontent.com/metrica-sports/sample-data/$SAMPLE_COMMIT

if [ ! -d "$ROOT/ref/.git" ]; then
    git clone --quiet "$REF_URL" "$ROOT/ref"
fi
git -C "$ROOT/ref" checkout --quiet "$REF_COMMIT"
echo "ref/ at $(git -C "$ROOT/ref" rev-parse --short HEAD)"

SAMPLE="$ROOT/external/metrica_sample"
while read -r sha path; do
    out="$SAMPLE/$path"
    if [ -f "$out" ] && echo "$sha  $out" | sha256sum --check --status; then
        continue
    fi
    mkdir -p "$(dirname "$out")"
    curl -fsSL "$RAW/$path" -o "$out.part"
    mv "$out.part" "$out"
    echo "$sha  $out" | sha256sum --check --quiet
    echo "fetched $path"
done <<'LIST'
fa4b178e62c4d3559e13b6d3a4fdd42e45293461c2ab656e05a13360a6adcbcf README.md
097dda6345e17390dca095331458fd00e47960c480c76f4356fd4edf2575ef05 data/Sample_Game_1/Sample_Game_1_RawEventsData.csv
0a3df68a92af0290ede472e62de3b6489ddee29223da09bdf0cf76c8e0ab0618 data/Sample_Game_1/Sample_Game_1_RawTrackingData_Away_Team.csv
0a3d0eff7785950379c587d3597b86a2d91c2492281b4652e1abb97c99b6b364 data/Sample_Game_1/Sample_Game_1_RawTrackingData_Home_Team.csv
edf31a18599265b77a8baf150f2ce6d89456fb0324b62ea7a391229657a619ba data/Sample_Game_2/Sample_Game_2_RawEventsData.csv
aa639f40762eade46c389020fe4458266b5ba14027b1e96e22e75db62803718b data/Sample_Game_2/Sample_Game_2_RawTrackingData_Away_Team.csv
3f9990ac204b07133e4202441d6fcbb386897c7c7f81e1bc3ce4d35bc8644694 data/Sample_Game_2/Sample_Game_2_RawTrackingData_Home_Team.csv
28a53bfe9eaf32535362335553d517cdce0eabb588af6166009081753751f4e0 data/Sample_Game_3/Sample_Game_3_events.json
009c7dcfcb66192014917ca1559c158f612353e390eafb85f8214b06a1c7dec1 data/Sample_Game_3/Sample_Game_3_metadata.xml
570e6ce82c2d93bc92723af331d93c31110bec1d6dc0be07c6d44ca3f981d658 data/Sample_Game_3/Sample_Game_3_tracking.txt
LIST
echo "external/metrica_sample/ verified (sample-data ${SAMPLE_COMMIT:0:7})"
