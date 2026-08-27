#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT" || exit 1

export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0
export GPU=0
export EVO_GPU=0
export FRESH_GPUS=""

export EVO_TOKEN=32
export EVO_FORCE_FIXED_TOKENS=32
export EVO_UNIMERGE_DEBUG=0
export EVO_UNIMERGE_OUTPUT_MODE=inplace_full

export EVO_NUM_CANDIDATES=5
export EVO_N_CANDIDATES=5
export EVO_CANDIDATES_PER_ITER=5
export EVO_SAMPLES_PER_ITERATION=5
export EVO_BATCH_SIZE=5

PREFIX="${1:-fresh5_gpu0_cand5}"
START_ROUND="${2:-1}"
NUM_ROUNDS="${3:-10}"
NUM_CAND="${4:-5}"
BASE_HISTORY="${5:-openevolve/v18_llm_nas/fresh5_small5_history.md}"

CTRL_ROOT="openevolve/runs/v18_llm_nas_${PREFIX}_10rounds_controller_v2"
mkdir -p "$CTRL_ROOT"

{
  echo "===== CONTROLLER V2 START ====="
  date
  echo "ROOT=$ROOT"
  echo "PREFIX=$PREFIX"
  echo "START_ROUND=$START_ROUND"
  echo "NUM_ROUNDS=$NUM_ROUNDS"
  echo "NUM_CAND=$NUM_CAND"
  echo "BASE_HISTORY=$BASE_HISTORY"
  env | sort | grep -E '^(CUDA_VISIBLE_DEVICES|GPU|EVO_|FRESH_GPUS)=' || true
  echo
  nvidia-smi || true
} > "$CTRL_ROOT/controller_env.txt" 2>&1

if [ ! -f "$BASE_HISTORY" ]; then
  echo "ERROR: base history not found: $BASE_HISTORY"
  exit 1
fi

HISTORY="$BASE_HISTORY"
END_ROUND=$((START_ROUND + NUM_ROUNDS - 1))

for i in $(seq "$START_ROUND" "$END_ROUND"); do
  ROUND="${PREFIX}_r$(printf "%02d" "$i")"
  ROUND_LOG_DIR="$CTRL_ROOT/$ROUND"
  mkdir -p "$ROUND_LOG_DIR"

  echo
  echo "============================================================"
  echo "ROUND=$ROUND"
  echo "HISTORY=$HISTORY"
  echo "NUM_CAND=$NUM_CAND"
  echo "============================================================"

  {
    echo "ROUND=$ROUND"
    echo "HISTORY=$HISTORY"
    echo "NUM_CAND=$NUM_CAND"
    echo "START_TIME=$(date '+%F %T')"
  } > "$ROUND_LOG_DIR/status.txt"

  echo "===== RUN FULL PIPELINE $ROUND ====="

  bash openevolve/v18_llm_nas/run_fresh_pipeline_round_v2.sh "$ROUND" "$HISTORY" "$NUM_CAND" \
    > "$ROUND_LOG_DIR/pipeline.log" 2>&1

  PIPE_RET=$?
  echo "PIPELINE_EXIT=$PIPE_RET" >> "$ROUND_LOG_DIR/status.txt"

  if [ "$PIPE_RET" -ne 0 ]; then
    echo "ERROR: pipeline failed for $ROUND"
    echo "See: $ROUND_LOG_DIR/pipeline.log"
    tail -120 "$ROUND_LOG_DIR/pipeline.log"
    exit 1
  fi

  MANIFEST="openevolve/policies/v18_llm_nas/${ROUND}/manifest.psv"
  RUN_ROOT="openevolve/runs/v18_llm_nas_${ROUND}_gpu0_cand5_seq"
  SUMMARY="$RUN_ROOT/summary.md"

  # Force regenerate summary.md because summarize_round.py prints to stdout by default.
  echo "===== FORCE SAVE SUMMARY $ROUND ====="
  python openevolve/v18_llm_nas/summarize_round.py --round "$ROUND" \
    | tee "$SUMMARY" > "$ROUND_LOG_DIR/resummarize.stdout"

  if [ ! -f "$MANIFEST" ]; then
    echo "ERROR: manifest not found: $MANIFEST"
    tail -120 "$ROUND_LOG_DIR/pipeline.log"
    exit 1
  fi

  N_MANIFEST=$(tail -n +2 "$MANIFEST" | grep -v '^[[:space:]]*$' | wc -l)
  echo "MANIFEST=$MANIFEST" >> "$ROUND_LOG_DIR/status.txt"
  echo "N_MANIFEST=$N_MANIFEST" >> "$ROUND_LOG_DIR/status.txt"

  if [ "$N_MANIFEST" -ne "$NUM_CAND" ]; then
    echo "ERROR: expected $NUM_CAND candidates, got $N_MANIFEST"
    cat "$MANIFEST"
    exit 1
  fi

  if [ ! -d "$RUN_ROOT" ]; then
    echo "ERROR: run root not found: $RUN_ROOT"
    tail -120 "$ROUND_LOG_DIR/pipeline.log"
    exit 1
  fi

  # 如果 summary 不存在或为空，补跑一次 summarize
  if [ ! -s "$SUMMARY" ]; then
    echo "WARNING: summary missing or empty. Re-running summarize."
    python openevolve/v18_llm_nas/summarize_round.py --round "$ROUND" \
      > "$ROUND_LOG_DIR/resummarize.log" 2>&1 || true
  fi

  # 检查 summary 是否有候选行
  if [ -f "$SUMMARY" ]; then
    N_ROWS=$(grep -c "^| ${ROUND}_" "$SUMMARY" || true)
  else
    N_ROWS=0
  fi

  echo "SUMMARY=$SUMMARY" >> "$ROUND_LOG_DIR/status.txt"
  echo "SUMMARY_ROWS=$N_ROWS" >> "$ROUND_LOG_DIR/status.txt"

  if [ "$N_ROWS" -lt 1 ]; then
    echo "WARNING: summary has no parsed candidate rows. Scores may still exist in mme_full.log."
    echo "Trying to show score lines from logs:"
    find "$RUN_ROOT" -name "mme_full.log" | head -5 | while read -r log; do
      echo "---- $log"
      grep -nE "total score:|score:" "$log" | tail -20 || true
    done
  fi

  # 选择下一轮 history
  NEXT_HISTORY=""

  for cand in \
    "openevolve/v18_llm_nas/${ROUND}_augmented_history.md" \
    "openevolve/v18_llm_nas/${ROUND}_history.md" \
    "$SUMMARY"
  do
    if [ -s "$cand" ]; then
      NEXT_HISTORY="$cand"
      break
    fi
  done

  if [ -n "$NEXT_HISTORY" ]; then
    HISTORY="$NEXT_HISTORY"
    echo "NEXT_HISTORY=$NEXT_HISTORY" >> "$ROUND_LOG_DIR/status.txt"
    echo "Next history: $NEXT_HISTORY"
  else
    echo "WARNING: no next history found. Reusing previous: $HISTORY"
    echo "NEXT_HISTORY_REUSED=$HISTORY" >> "$ROUND_LOG_DIR/status.txt"
  fi

  echo "END_TIME=$(date '+%F %T')" >> "$ROUND_LOG_DIR/status.txt"
  echo "===== ROUND DONE: $ROUND ====="
done

echo
echo "===== ALL ${NUM_ROUNDS} ROUNDS DONE ====="
echo "Controller logs: $CTRL_ROOT"
