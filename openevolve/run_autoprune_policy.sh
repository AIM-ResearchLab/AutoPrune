#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ "$#" -lt 3 ]; then
  echo "Usage: $0 <anchor_policy_yaml> <merge_policy_yaml> <command...>"
  exit 1
fi

ANCHOR_POLICY="$1"
MERGE_POLICY="$2"
shift 2

case "$ANCHOR_POLICY" in
  /*) ;;
  *) ANCHOR_POLICY="${ROOT}/${ANCHOR_POLICY}" ;;
esac
case "$MERGE_POLICY" in
  /*) ;;
  *) MERGE_POLICY="${ROOT}/${MERGE_POLICY}" ;;
esac

test -f "$ANCHOR_POLICY" || { echo "ERROR: anchor policy not found: $ANCHOR_POLICY"; exit 1; }
test -f "$MERGE_POLICY" || { echo "ERROR: merge policy not found: $MERGE_POLICY"; exit 1; }

export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export EVO_CDPRUNER_ENABLE=1
export EVO_CDPRUNER_V14_ENABLE=1
export CDPRUNER_POLICY="$ANCHOR_POLICY"
export EVO_POLICY_YAML_PATH="$ANCHOR_POLICY"
export EVO_CANDIDATE_POLICY_PATH="$ANCHOR_POLICY"
export EVO_POLICY_YAML="$(cat "$ANCHOR_POLICY")"
export EVO_CANDIDATE_POLICY_YAML="$EVO_POLICY_YAML"

export EVO_UNIMERGE_ENABLE=1
export EVO_UNIMERGE_MODE="hybrid_v16_anchor"
export EVO_UNIMERGE_POLICY_PATH="$MERGE_POLICY"
export EVO_UNIMERGE_POLICY_YAML="$(cat "$MERGE_POLICY")"
export EVO_UNIMERGE_DEBUG="${EVO_UNIMERGE_DEBUG:-0}"
export EVO_UNIMERGE_OUTPUT_MODE="${EVO_UNIMERGE_OUTPUT_MODE:-inplace_full}"

export EVO_TOKEN="${EVO_TOKEN:-32}"
export EVO_FORCE_FIXED_TOKENS="${EVO_FORCE_FIXED_TOKENS:-${EVO_TOKEN}}"

if [ -n "${EVO_GPU:-}" ]; then
  export CUDA_VISIBLE_DEVICES="${EVO_GPU}"
fi

if [ "${AUTOPRUNE_RUNTIME_DEBUG:-0}" = "1" ]; then
  echo "=== AutoPrune runtime ==="
  echo "Anchor policy: $ANCHOR_POLICY"
  echo "Merge policy: $MERGE_POLICY"
  echo "EVO_TOKEN=$EVO_TOKEN"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
fi

exec "$@"
