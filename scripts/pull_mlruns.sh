#!/bin/bash
# Sync the cluster's mlruns/ to a local copy and rewrite artifact paths so the
# local mlflow UI can serve everything off the laptop SSD (fast).
#
# Idempotent: only deltas (new/changed files) ship; the path-rewrite is a
# no-op on already-patched files. Run as often as you like.
#
# Usage:
#   bash scripts/pull_mlruns.sh
# then:
#   mlflow ui --backend-store-uri file://$HOME/Desktop/tfm_code/mlruns_cluster --port 5001

set -euo pipefail

REMOTE_HOST="ac5725@shell2.doc.ic.ac.uk"
REMOTE_PATH="/vol/bitbucket/ac5725/eeg_biomarkers_models/mlruns"
LOCAL_PATH="$HOME/Desktop/tfm_code/mlruns_cluster"

mkdir -p "$LOCAL_PATH"

echo "==> rsync ${REMOTE_HOST}:${REMOTE_PATH}/ -> ${LOCAL_PATH}/"
rsync -avz --info=progress2,stats2 --human-readable \
    "${REMOTE_HOST}:${REMOTE_PATH}/" \
    "${LOCAL_PATH}/"

echo "==> rewriting artifact paths in meta.yaml files"
mapfile -t META_FILES < <(find "$LOCAL_PATH" -name meta.yaml)
TOTAL=${#META_FILES[@]}
for i in "${!META_FILES[@]}"; do
    sed -i "s|${REMOTE_PATH}|${LOCAL_PATH}|g" "${META_FILES[$i]}"
    printf "\r    [%d/%d] (%d%%)" "$((i + 1))" "$TOTAL" "$(( (i + 1) * 100 / TOTAL ))"
done
echo ""

echo ""
echo "Done. View the runs locally with:"
echo "  mlflow ui --backend-store-uri file://${LOCAL_PATH} --port 5001"
echo "Then open http://localhost:5001"
