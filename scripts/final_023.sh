#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."

EXP_DIR=data/artifacts/experiments
LOG_DIR=$EXP_DIR/night
mkdir -p "$LOG_DIR"
BASE=(--parent EXP-022/core_s42 --dense-model data/artifacts/models/e5ft
  --second-encoder deepvk/USER-base --geo-history fallback --transitions full --geo-damping
  --neighbor-centroid --microcats neighbors --global-k 400 --local-k 300 --radius-km 25
  --radius-k 50 --early-stopping-rounds 0 --loss-function YetiRank
  --depth 8 --iterations 1000)

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG_DIR/queue_023.log"; }

while pgrep -f "candgen.scripts.(selection_lab|experiment)" > /dev/null; do sleep 60; done

for seed in 42 43 44; do
  variant=d8_s$seed
  report=$EXP_DIR/EXP-023/$variant/report.json
  if [[ ! -f $report ]]; then
    log "EXP-023/$variant: start"
    systemd-run --user --scope -q -p MemoryMax=20G env HF_HUB_OFFLINE=1 \
      uv run python -m candgen.scripts.experiment --exp EXP-023 --variant "$variant" \
      "${BASE[@]}" --seed "$seed" > "$LOG_DIR/EXP-023_$variant.log" 2>&1
  fi
  if [[ ! -f $report ]]; then
    log "EXP-023/$variant: FAILED, see $LOG_DIR/EXP-023_$variant.log"
    continue
  fi
  log "EXP-023/$variant: done, $(python3 -c "
import json, sys
r = json.load(open(sys.argv[1])); p = r['vs_parent']
print(f\"R@50 {r['recall@50']:.5f}, vs EXP-022 {p['diff'] * 100:+.2f} [{p['ci95'][0] * 100:+.2f}; {p['ci95'][1] * 100:+.2f}], pool {r['pool_recall']:.4f}\")
" "$report")"
  if [[ $seed == 42 ]]; then
    log "predict EXP-023/$variant: start"
    systemd-run --user --scope -q -p MemoryMax=20G env HF_HUB_OFFLINE=1 \
      uv run python -m candgen.scripts.predict --method catboost \
      --model "$EXP_DIR/EXP-023/$variant/model.cbm" --tag catboost_exp023_d8_s42 \
      > "$LOG_DIR/predict_EXP-023_$variant.log" 2>&1
    log "predict EXP-023/$variant: exit $?"
  fi
done
log "queue finished"
