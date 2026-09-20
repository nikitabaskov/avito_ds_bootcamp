#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."

EXP_DIR=data/artifacts/experiments
LAB=$EXP_DIR/lab
LOG_DIR=$EXP_DIR/night
mkdir -p "$LOG_DIR"
BASE=(--parent EXP-019/e5ft_user_s42 --dense-model data/artifacts/models/e5ft
  --second-encoder deepvk/USER-base --geo-history fallback --transitions full --geo-damping
  --neighbor-centroid --microcats neighbors --global-k 400 --local-k 300 --radius-km 25
  --radius-k 50 --early-stopping-rounds 0 --loss-function YetiRank)

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG_DIR/queue_022.log"; }

while pgrep -f "candgen.scripts.(selection_lab|experiment)" > /dev/null; do sleep 60; done

decide() {
  HF_HUB_OFFLINE=1 uv run python - "$LAB" <<'EOF'
import json, sys
from pathlib import Path
import numpy as np
from candgen.core.evaluation import paired_bootstrap

lab = Path(sys.argv[1])
core = json.loads((lab / "features_core/results.json").read_text())

def wins(name):
    ci = core.get(name, {}).get("vs_baseline", {}).get("seed_mean", {}).get("ci95")
    return bool(ci) and ci[0] > 0

flags = []
if wins("core_filter"):
    flags.append("--filter-match")
if wins("core_i1000"):
    flags += ["--iterations", "1000"]
pools = {
    "region/core_region": ["--region-k", "200"],
    "radius/core_radius": ["--radius-km", "100", "--radius-k", "100"],
}
base = np.load(lab / "features_core/core_seed_mean.npy")
best, best_diff = None, 0.0
for name, pool_flags in pools.items():
    path = lab / f"{name}_seed_mean.npy"
    if path.exists():
        r = paired_bootstrap(np.load(path), base)
        print(f"{name}: {r['diff'] * 100:+.2f} [{r['ci95'][0] * 100:+.2f}; {r['ci95'][1] * 100:+.2f}]", file=sys.stderr)
        if r["ci95"][0] > 0 and r["diff"] > best_diff:
            best, best_diff = pool_flags, r["diff"]
if best:
    flags += best
print(" ".join(flags))
EOF
}

read -r -a EXTRA <<< "$(decide)"
log "EXP-022 extra flags: ${EXTRA[*]:-none}"

for seed in 42 43 44; do
  variant=core_s$seed
  if [[ ! -f $EXP_DIR/EXP-022/$variant/report.json ]]; then
    log "EXP-022/$variant: start"
    systemd-run --user --scope -q -p MemoryMax=20G env HF_HUB_OFFLINE=1 \
      uv run python -m candgen.scripts.experiment --exp EXP-022 --variant "$variant" \
      "${BASE[@]}" "${EXTRA[@]}" --seed "$seed" > "$LOG_DIR/EXP-022_$variant.log" 2>&1
  fi
  report=$EXP_DIR/EXP-022/$variant/report.json
  if [[ ! -f $report ]]; then
    log "EXP-022/$variant: FAILED, see $LOG_DIR/EXP-022_$variant.log"
    continue
  fi
  log "EXP-022/$variant: done, $(python3 -c "
import json, sys
r = json.load(open(sys.argv[1])); p = r['vs_parent']
print(f\"R@50 {r['recall@50']:.5f}, vs EXP-019 {p['diff'] * 100:+.2f} [{p['ci95'][0] * 100:+.2f}; {p['ci95'][1] * 100:+.2f}], pool {r['pool_recall']:.4f}\")
" "$report")"
  if [[ $seed == 42 ]]; then
    log "predict EXP-022/$variant: start"
    systemd-run --user --scope -q -p MemoryMax=20G env HF_HUB_OFFLINE=1 \
      uv run python -m candgen.scripts.predict --method catboost \
      --model "$EXP_DIR/EXP-022/$variant/model.cbm" --tag catboost_exp022_s42 \
      > "$LOG_DIR/predict_EXP-022_$variant.log" 2>&1
    log "predict EXP-022/$variant: exit $?"
  fi
done
log "queue finished"
