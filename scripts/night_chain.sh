#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."

ART=data/artifacts
LAB=(env HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 uv run python -m candgen.scripts.selection_lab
  --references EXP-019/e5ft_user_s42 --dense-model data/artifacts/models/e5ft
  --second-encoder deepvk/USER-base --geo-history fallback --transitions full --geo-damping
  --neighbor-centroid --filter-match --microcats neighbors --global-k 400 --local-k 300)

log() { echo "[$(date '+%F %T')] $*" | tee -a "$ART/night_chain.log"; }
scope() { systemd-run --user --scope -q -p MemoryMax="$1" "${@:2}"; }

log "lab features_core: start"
scope 20G "${LAB[@]}" --radius-km 25 --radius-k 50 \
  --variants configs/lab/features_core.json --lab features_core > "$ART/lab_features_core.log" 2>&1
log "lab features_core: exit $?"
log "lab region: start"
scope 20G "${LAB[@]}" --radius-km 25 --radius-k 50 --region-k 200 \
  --variants configs/lab/region.json --lab region > "$ART/lab_region.log" 2>&1
log "lab region: exit $?"
log "lab radius: start"
scope 20G "${LAB[@]}" --radius-km 100 --radius-k 100 \
  --variants configs/lab/radius.json --lab radius > "$ART/lab_radius.log" 2>&1
log "lab radius: exit $?"

scripts/night_022.sh &
cpu=$!

log "finetune e5ft2: start"
scope 8G env HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 uv run python -m candgen.scripts.finetune_dense \
  --per-text 8 --tag e5ft2 > "$ART/finetune_e5ft2.log" 2>&1
log "finetune e5ft2: exit $?"
if [[ -f $ART/models/e5ft2/meta.json ]]; then
  for corpus in split benchmark; do
    log "embed $corpus e5ft2: start"
    scope 8G env HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 uv run python -m candgen.scripts.embed_corpus \
      --corpus "$corpus" --model data/artifacts/models/e5ft2 > "$ART/embed_${corpus}_e5ft2.log" 2>&1
    log "embed $corpus e5ft2: exit $?"
  done
fi
wait "$cpu"
log "chain finished"
