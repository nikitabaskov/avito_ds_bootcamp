#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."

ART=data/artifacts
LAB=(env HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 uv run python -m candgen.scripts.selection_lab
  --references EXP-019/e5ft_user_s42 --second-encoder deepvk/USER-base --geo-history fallback
  --transitions full --geo-damping --neighbor-centroid --filter-match --microcats neighbors
  --global-k 400 --dense-model data/artifacts/models/e5ft)

log() { echo "[$(date '+%F %T')] $*" | tee -a "$ART/day_labs.log"; }
run() {
  local name=$1 mem=$2
  shift 2
  log "lab $name: start"
  systemd-run --user --scope -q -p MemoryMax="$mem" "${LAB[@]}" "$@" \
    --lab "$name" > "$ART/lab_$name.log" 2>&1
  log "lab $name: exit $?"
}

while pgrep -f "candgen.scripts.(selection_lab|experiment)" > /dev/null; do sleep 30; done

run capacity 20G --local-k 300 --radius-km 25 --radius-k 50 --variants configs/lab/capacity.json
run geo 20G --local-k 0 --geo-km 50 --geo-k 350 --geo-weight 0.6 --geo-delta 0.02 \
  --variants configs/lab/geo.json
run data12000 24G --local-k 300 --radius-km 25 --radius-k 50 --train-queries 12000 \
  --variants configs/lab/data12000.json
log "day labs finished"
