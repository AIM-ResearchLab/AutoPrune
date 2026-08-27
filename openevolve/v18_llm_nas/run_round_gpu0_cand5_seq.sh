#!/usr/bin/env bash
set -u

ROUND="${1:-round1}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT" || exit 1

export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export EVO_GPU=0
export CUDA_VISIBLE_DEVICES=0
export EVO_TOKEN=32
export EVO_FORCE_FIXED_TOKENS=32
export EVO_UNIMERGE_DEBUG=0
export EVO_UNIMERGE_OUTPUT_MODE=inplace_full

# Enable per-candidate mask/hash diagnostics.
export EVO_CDPRUNER_MASK_HASH_LOG=1
export EVO_MASK_HASH_LOG_N="${EVO_MASK_HASH_LOG_N:-100000000}"
export EVO_MASK_HASH_LOG_BATCH="${EVO_MASK_HASH_LOG_BATCH:-1}"
export EVO_CDPRUNER_MASK_HASH_DEBUG="${EVO_CDPRUNER_MASK_HASH_DEBUG:-0}"

MANIFEST="openevolve/policies/v18_llm_nas/${ROUND}/manifest.psv"
RUN_ROOT="openevolve/runs/v18_llm_nas_${ROUND}_gpu0_cand5_seq"
mkdir -p "$RUN_ROOT"

if [ ! -f "$MANIFEST" ]; then
  echo "ERROR: manifest not found: $MANIFEST"
  exit 1
fi

run_one() {
  local name="$1"
  local anchor_policy="$2"
  local merge_policy="$3"

  if [ -z "$name" ] || [ -z "$anchor_policy" ] || [ -z "$merge_policy" ]; then
    echo "ERROR: bad row name=[$name] anchor=[$anchor_policy] merge=[$merge_policy]"
    return 1
  fi

  if [ ! -f "$anchor_policy" ]; then
    echo "ERROR: anchor policy not found: $anchor_policy"
    return 1
  fi

  if [ ! -f "$merge_policy" ]; then
    echo "ERROR: merge policy not found: $merge_policy"
    return 1
  fi

  local out_dir="$RUN_ROOT/$name"
  mkdir -p "$out_dir"

  local exp="v18llm_${ROUND}_${name}_g0_$(date +%Y%m%d_%H%M%S)"
  local mask_hash_log_abs
  mask_hash_log_abs="$(pwd)/$out_dir/mask_hash.jsonl"

  rm -f "$mask_hash_log_abs"

  {
    echo "ROUND=$ROUND"
    echo "NAME=$name"
    echo "ANCHOR_POLICY=$anchor_policy"
    echo "MERGE_POLICY=$merge_policy"
    echo "EXP=$exp"
    echo "GPU=0"
    echo "BASELINE=clean_cdpruner_k32_1382"
    echo "MASK_HASH_LOG=$mask_hash_log_abs"
    echo "START_TIME=$(date '+%F %T')"
  } > "$out_dir/status.txt"

  rm -f "playground/data/eval/MME/answers/${exp}.jsonl" 2>/dev/null || true
  rm -f "playground/data/eval/MME/answers/${exp}.txt" 2>/dev/null || true

  echo "===== RUN ${ROUND}/${name} on GPU 0 ====="
  echo "ANCHOR=$anchor_policy"
  echo "MERGE=$merge_policy"
  echo "MASK_HASH_LOG=$mask_hash_log_abs"

  EVO_MASK_HASH_LOG="$mask_hash_log_abs" \
  openevolve/run_autoprune_policy.sh \
    "$anchor_policy" \
    "$merge_policy" \
    bash scripts/v1_5/eval/mme.sh 32 "$exp" \
    > "$out_dir/mme_full.log" 2>&1

  local ret=$?

  echo "EXIT_CODE=$ret" >> "$out_dir/status.txt"
  echo "END_TIME=$(date '+%F %T')" >> "$out_dir/status.txt"

  grep -n "CDPRUNER_POLICY\|V16 anchor policy\|Anchor policy\|Merge policy\|EVO_MASK_HASH\|UniMergeHybridDelta\|UniMergeHybrid\|vtn=\|Traceback\|JSONDecodeError\|===========\|total score:\|score:" "$out_dir/mme_full.log" \
    | tail -320 \
    > "$out_dir/mme_full_score_summary.txt" || true

  if [ -f "$mask_hash_log_abs" ]; then
    echo "MASK_HASH_ROWS=$(wc -l < "$mask_hash_log_abs")" >> "$out_dir/status.txt"
  else
    echo "MASK_HASH_ROWS=0" >> "$out_dir/status.txt"
  fi

  if [ "$ret" -eq 0 ]; then
    echo "===== DONE_OK ${ROUND}/${name} ====="
  else
    echo "===== DONE_FAIL ${ROUND}/${name} ====="
  fi

  return 0
}

echo "===== v18 LLM NAS ${ROUND} GPU0 cand5 sequential runner ====="
echo "MANIFEST=$MANIFEST"
echo "RUN_ROOT=$RUN_ROOT"
echo "EVO_CDPRUNER_MASK_HASH_LOG=$EVO_CDPRUNER_MASK_HASH_LOG"

tail -n +2 "$MANIFEST" | while IFS='|' read -r name anchor_policy merge_policy; do
  run_one "$name" "$anchor_policy" "$merge_policy"
done

echo "All jobs for ${ROUND} finished."
