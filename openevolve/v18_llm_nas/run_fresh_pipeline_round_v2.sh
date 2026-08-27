#!/usr/bin/env bash
set -euo pipefail

ROUND="${1:?Usage: $0 <round> <history> <num>}"
HISTORY="${2:?Usage: $0 <round> <history> <num>}"
NUM="${3:-5}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export EVO_TOKEN=32
export EVO_FORCE_FIXED_TOKENS=32
export EVO_UNIMERGE_DEBUG=0
export EVO_UNIMERGE_OUTPUT_MODE=inplace_full

SPEC_DIR="openevolve/policies/v18_llm_nas/${ROUND}"
SPEC_JSONL="${SPEC_DIR}/candidates_${ROUND}.jsonl"
RUN_ROOT="openevolve/runs/v18_llm_nas_${ROUND}_gpu0_cand5_seq"

mkdir -p "$SPEC_DIR"
mkdir -p "$RUN_ROOT"

echo "===== fresh AutoNAS v2 pipeline ====="
echo "ROUND=$ROUND"
echo "HISTORY=$HISTORY"
echo "NUM=$NUM"
echo "FRESH_GPUS=${FRESH_GPUS:-}"
echo "OPENAI_MODEL=${OPENAI_MODEL:-unset}"
echo

if [ ! -f "$HISTORY" ]; then
  echo "ERROR: history not found: $HISTORY"
  exit 1
fi

echo "===== Step 1: fresh LLM planner ====="
python openevolve/v18_llm_nas/plan_fresh_round.py \
  --round "$ROUND" \
  --history "$HISTORY" \
  --num "$NUM" \
  --out-jsonl "$SPEC_JSONL"

echo
echo "===== Step 2: materialize YAML policies ====="
python openevolve/v18_llm_nas/materialize_round.py \
  --round "$ROUND" \
  --spec-jsonl "$SPEC_JSONL"

echo
echo "===== Step 3: run MME ====="
rm -rf "$RUN_ROOT"
mkdir -p "$RUN_ROOT"

# Force single-GPU0 sequential runner for the isolated cand5 workspace.
# Do not use run_round_gpu_pool_dynamic.py here, because it may still depend on the original CDPruner root.
if [ -n "${FRESH_GPUS:-}" ]; then

  echo "===== dynamic multi-GPU candidate evaluation ====="

  echo "FRESH_GPUS=$FRESH_GPUS"

  python \
    openevolve/v18_llm_nas/run_round_gpu_pool_dynamic.py \
    --round "$ROUND" \
    --gpus "$FRESH_GPUS"

else

  echo "===== fallback: sequential GPU0 evaluation ====="

  bash \
    openevolve/v18_llm_nas/run_round_gpu0_cand5_seq.sh \
    "$ROUND"

fi \
  > "$RUN_ROOT/launcher.log" 2>&1

echo
echo "===== Step 4: summarize ====="
python openevolve/v18_llm_nas/summarize_round.py --round "$ROUND" \
  | tee "$RUN_ROOT/summary.md"

echo
echo "===== Step 5: build augmented history ====="

AUG_HISTORY="openevolve/v18_llm_nas/${ROUND}_augmented_history.md"

{
  cat "$HISTORY"

  echo
  echo
  echo "## Search results from ${ROUND}"
  echo
  echo "- LLM editor: ${OPENAI_MODEL:-unknown}"
  echo "- Visual token budget: ${EVO_TOKEN:-32}"
  echo "- Candidate count: ${NUM:-unknown}"
  echo

  cat "$RUN_ROOT/summary.md"

} > "$AUG_HISTORY"

echo "augmented history: $AUG_HISTORY"

test -s "$AUG_HISTORY"

echo "===== Done $ROUND ====="
echo "summary: $RUN_ROOT/summary.md"
