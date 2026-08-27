#!/usr/bin/env bash

set -Eeuo pipefail


# ============================================================
# 0. Basic configuration
# ============================================================

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$ROOT"


export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"


MULTI_ROOT="${MULTI_ROOT:?MULTI_ROOT is not set}"


SEARCH_GPU="${SEARCH_GPU:-0}"

# Candidate-evaluation GPU pool.
# Five candidates per round can use up to five GPUs concurrently.
SEARCH_GPUS="${SEARCH_GPUS:-0,1,2,3,4}"


PIPELINE="openevolve/v18_llm_nas/run_fresh_pipeline_round_v2.sh"


BASE_HISTORY="openevolve/v18_llm_nas/fresh5_small5_history.md"


NUM_ROUNDS=10


NUM_CANDIDATES=5


TOKEN_BUDGET=32


STAMP="$(date +%Y%m%d_%H%M%S)"


mkdir -p "$MULTI_ROOT"


mkdir -p "$MULTI_ROOT/config_backup"


mkdir -p "$MULTI_ROOT/api_smoke"


mkdir -p "$MULTI_ROOT/results"


# ============================================================
# 1. API environment
# ============================================================

if [ -z "${DASHSCOPE_API_KEY:-}" ] && \
   [ -z "${OPENAI_API_KEY:-}" ]; then

    echo
    echo "ERROR:"
    echo "Neither DASHSCOPE_API_KEY nor OPENAI_API_KEY is set."
    echo
    exit 1

fi


if [ -z "${DASHSCOPE_API_KEY:-}" ]; then

    export DASHSCOPE_API_KEY="$OPENAI_API_KEY"

fi


export OPENAI_API_KEY="$DASHSCOPE_API_KEY"


export OPENAI_BASE_URL="${OPENAI_BASE_URL:-${OPENAI_API_BASE:-https://dashscope.aliyuncs.com/compatible-mode/v1}}"


export OPENAI_API_BASE="$OPENAI_BASE_URL"


# ============================================================
# 2. Fixed experiment settings
# ============================================================

# Do not restrict the launcher to one visible GPU.
# The inner candidate launcher assigns one GPU per candidate.
unset CUDA_VISIBLE_DEVICES


export EVO_GPU="$SEARCH_GPU"


export FRESH_GPUS="$SEARCH_GPUS"


export EVO_TOKEN="$TOKEN_BUDGET"


export EVO_FORCE_FIXED_TOKENS="$TOKEN_BUDGET"


export EVO_UNIMERGE_DEBUG=0


export EVO_UNIMERGE_OUTPUT_MODE=inplace_full


export TOKENIZERS_PARALLELISM=false


# ============================================================
# 3. New LLM editors
#
# qwen-plus is not rerun.
#
# New:
#   deepseek-v4-flash
#   qwen-max
# ============================================================

MODEL_SPECS=(

    "dsv4flash|deepseek-v4-flash"

    "qwenmax|qwen-max"

)


# ============================================================
# 4. Preflight
# ============================================================

echo "============================================================"

echo "MULTI-LLM API SEARCH"

echo "============================================================"

echo "ROOT=$ROOT"

echo "MULTI_ROOT=$MULTI_ROOT"

echo "SEARCH_GPU=$SEARCH_GPU"
echo "SEARCH_GPUS=$SEARCH_GPUS"
echo "FRESH_GPUS=$FRESH_GPUS"
echo "MAX_CONCURRENT_GPU_JOBS=$NUM_CANDIDATES"

echo "BACKBONE=llava-v1.5-7b"

echo "BENCHMARK=MME"

echo "TOKEN_BUDGET=$TOKEN_BUDGET"

echo "NUM_ROUNDS=$NUM_ROUNDS"

echo "NUM_CANDIDATES=$NUM_CANDIDATES"

echo "OPENAI_BASE_URL=$OPENAI_BASE_URL"

echo "MODELS=deepseek-v4-flash,qwen-max"

echo "QWEN_PLUS=REUSE_EXISTING_RESULT"

echo "START_TIME=$(date '+%F %T')"

echo "============================================================"


for required in \
    "$PIPELINE" \
    "$BASE_HISTORY" \
    "openevolve/config.yaml"

do

    if [ ! -f "$required" ]; then

        echo "ERROR: missing required file"

        echo "$required"

        exit 1

    fi

done


echo

echo "===== GPU status ====="


nvidia-smi \
    --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader


# ============================================================
# 5. Lock
#
# This experiment temporarily switches the LLM name in the
# OpenEvolve configuration. Do not run another search pipeline
# in the same repository at the same time.
# ============================================================

LOCK_DIR="/tmp/cdpruner_multi_llm_api_llava15_mme_k32.lock"


if ! mkdir "$LOCK_DIR" 2>/dev/null; then

    echo

    echo "ERROR: lock already exists"

    echo "$LOCK_DIR"

    echo

    echo "Another multi-LLM search may still be running."

    exit 1

fi


echo "$$" > "$LOCK_DIR/pid.txt"


# ============================================================
# 6. Back up shared LLM configuration
# ============================================================

CONFIG_FILES=(

    "openevolve/config.yaml"

    "openevolve/config_yaml_policy.yaml"

)


for config in "${CONFIG_FILES[@]}"; do

    if [ -f "$config" ]; then

        cp -a \
            "$config" \
            "$MULTI_ROOT/config_backup/$(basename "$config")"

    fi

done


restore_configs() {

    echo

    echo "===== restore OpenEvolve configuration ====="


    for config in "${CONFIG_FILES[@]}"; do


        backup="$MULTI_ROOT/config_backup/$(basename "$config")"


        if [ -f "$backup" ]; then


            cp -f \
                "$backup" \
                "$config"


            echo "restored: $config"


        fi


    done


    rm -rf "$LOCK_DIR" 2>/dev/null || true

}


trap restore_configs EXIT


# ============================================================
# 7. Save manual restore script
# ============================================================

cat > "$MULTI_ROOT/restore_original_llm_config.sh" <<RESTORE
#!/usr/bin/env bash

set -e

cd "$ROOT"


if [ -f "$MULTI_ROOT/config_backup/config.yaml" ]; then

    cp -f \
        "$MULTI_ROOT/config_backup/config.yaml" \
        openevolve/config.yaml

fi


if [ -f "$MULTI_ROOT/config_backup/config_yaml_policy.yaml" ]; then

    cp -f \
        "$MULTI_ROOT/config_backup/config_yaml_policy.yaml" \
        openevolve/config_yaml_policy.yaml

fi


rm -rf \
    /tmp/cdpruner_multi_llm_api_llava15_mme_k32.lock


echo "Original LLM configuration restored."

RESTORE


chmod +x \
    "$MULTI_ROOT/restore_original_llm_config.sh"


# ============================================================
# 8. API smoke test before GPU search
# ============================================================

echo

echo "============================================================"

echo "API SMOKE TEST"

echo "============================================================"


python - "$MULTI_ROOT/api_smoke/api_smoke_results.json" <<'PY'

import json

import os

import sys

import time

import traceback


from openai import OpenAI


output = sys.argv[1]


models = [

    "qwen-plus",

    "deepseek-v4-flash",

    "qwen-max",

]


client = OpenAI(

    api_key=os.environ["OPENAI_API_KEY"],

    base_url=os.environ["OPENAI_BASE_URL"],

    timeout=300.0,

)


results = []


for model in models:


    print()

    print(

        "===== API TEST:",

        model,

        "=====",

        flush=True,

    )


    started = time.time()


    try:


        response = client.chat.completions.create(

            model=model,

            messages=[

                {

                    "role":

                    "system",

                    "content":

                    "Return a concise plain-text answer.",

                },

                {

                    "role":

                    "user",

                    "content":

                    "Reply with exactly: API_OK",

                },

            ],

            max_tokens=512,

        )


        message = response.choices[0].message


        content = str(

            getattr(

                message,

                "content",

                "",

            )

            or

            ""

        ).strip()


        elapsed = time.time() - started


        print(

            "content =",

            repr(

                content[:300]

            ),

        )


        print(

            "elapsed_sec =",

            round(

                elapsed,

                3,

            ),

        )


        if not content:


            raise RuntimeError(

                "API returned empty final content"

            )


        results.append(

            {

                "model":

                model,

                "status":

                "OK",

                "elapsed_sec":

                elapsed,

                "content":

                content[:1000],

            }

        )


    except Exception as exc:


        elapsed = time.time() - started


        traceback.print_exc()


        results.append(

            {

                "model":

                model,

                "status":

                "ERROR",

                "elapsed_sec":

                elapsed,

                "error":

                repr(

                    exc

                ),

            }

        )


with open(

    output,

    "w",

    encoding="utf-8",

) as f:


    json.dump(

        results,

        f,

        ensure_ascii=False,

        indent=2,

    )


bad = [

    row

    for row

    in results

    if row[

        "status"

    ]

    !=

    "OK"

]


print()

print(

    "API_RESULT_FILE =",

    output,

)


if bad:


    print()

    print(

        "API smoke test failed:",

        [

            row[

                "model"

            ]

            for row

            in bad

        ],

    )


    raise SystemExit(

        2

    )


print()

print(

    "[PASS] All three API models are available."

)

PY


# ============================================================
# 9. Patch selected LLM into configuration
# ============================================================

patch_model_config() {

    local model="$1"


    echo

    echo "===== patch model: $model ====="


    for config in "${CONFIG_FILES[@]}"; do


        backup="$MULTI_ROOT/config_backup/$(basename "$config")"


        if [ ! -f "$backup" ]; then


            continue


        fi


        python - \
            "$backup" \
            "$config" \
            "$model" <<'PY'

import re

import sys


from pathlib import Path


source = Path(

    sys.argv[1]

)


target = Path(

    sys.argv[2]

)


model = sys.argv[3]


text = source.read_text(

    encoding="utf-8"

)


original = text


# Exact quoted forms

text = text.replace(

    '"qwen-plus"',

    f'"{model}"',

)


text = text.replace(

    "'qwen-plus'",

    f"'{model}'",

)


# Exact unquoted YAML values

text = re.sub(

    r'(?m)^(\s*(?:-\s*)?name:\s*)qwen-plus(\s*(?:#.*)?)$',

    rf'\1"{model}"\2',

    text,

)


text = re.sub(

    r'(?m)^(\s*model:\s*)qwen-plus(\s*(?:#.*)?)$',

    rf'\1"{model}"\2',

    text,

)


if model not in text:


    raise RuntimeError(

        f"Failed to inject model={model} into {target}"

    )


target.write_text(

    text,

    encoding="utf-8",

)


print(

    "patched:",

    target,

)


print(

    "model:",

    model,

)

PY


    done

}


# ============================================================
# 10. Main search
# ============================================================

printf \
    "slug\tmodel\tround\tround_name\tstatus\tstart_time\tend_time\tpipeline_log\thistory_in\thistory_out\n" \
    > "$MULTI_ROOT/status.tsv"


printf \
    "slug\tmodel\tround_name\n" \
    > "$MULTI_ROOT/models.tsv"


run_one_model() {

    local slug="$1"

    local model="$2"


    local model_root="$MULTI_ROOT/$slug"


    mkdir -p "$model_root"


    patch_model_config \
        "$model"


    export OPENAI_MODEL="$model"


    export LLM_MODEL="$model"


    export EVO_LLM_MODEL="$model"


    export EVO_V4_LLM_MODEL="$model"


    export EVO_V32_LLM_MODEL="$model"


    export EVO_V31_LLM_MODEL="$model"


    export OPENEVOLE_LLM_MODEL="$model"


    cp -f \
        "$BASE_HISTORY" \
        "$model_root/history_r00.md"


    local history="$model_root/history_r00.md"


    {

        echo "MODEL=$model"

        echo "SLUG=$slug"

        echo "BACKBONE=llava-v1.5-7b"

        echo "BENCHMARK=MME"

        echo "TOKEN_BUDGET=$TOKEN_BUDGET"

        echo "NUM_ROUNDS=$NUM_ROUNDS"

        echo "NUM_CANDIDATES=$NUM_CANDIDATES"

        echo "SEARCH_GPU=$SEARCH_GPU"

        echo "BASE_HISTORY=$BASE_HISTORY"

        echo "OPENAI_BASE_URL=$OPENAI_BASE_URL"

        echo "START_TIME=$(date '+%F %T')"

    } > "$model_root/model_manifest.txt"


    cp -f \
        openevolve/config.yaml \
        "$model_root/config_model_snapshot.yaml"


    if [ -f openevolve/config_yaml_policy.yaml ]; then


        cp -f \
            openevolve/config_yaml_policy.yaml \
            "$model_root/config_yaml_policy_model_snapshot.yaml"


    fi


    echo

    echo "############################################################"

    echo "MODEL SEARCH START"

    echo "slug=$slug"

    echo "model=$model"

    echo "############################################################"


    for round_index in $(

        seq 1 "$NUM_ROUNDS"

    ); do


        round_id="$(

            printf '%02d' \
                "$round_index"

        )"


        round_name="mapi_${slug}_k32_${STAMP}_r${round_id}"


        round_root="$model_root/r${round_id}"


        mkdir -p \
            "$round_root"


        start_time="$(

            date '+%F %T'

        )"


        touch \
            "$round_root/start.marker"


        echo

        echo "============================================================"

        echo "MODEL=$model"

        echo "ROUND=$round_index/$NUM_ROUNDS"

        echo "ROUND_NAME=$round_name"

        echo "HISTORY=$history"

        echo "CANDIDATES=$NUM_CANDIDATES"

        echo "TOKEN=$TOKEN_BUDGET"

        echo "GPU=$SEARCH_GPU"

        echo "START=$start_time"

        echo "============================================================"


        printf \
            "%s\t%s\t%s\t%s\tRUNNING\t%s\t-\t%s\t%s\t-\n" \
            "$slug" \
            "$model" \
            "$round_id" \
            "$round_name" \
            "$start_time" \
            "$round_root/pipeline.log" \
            "$history" \
            >> "$MULTI_ROOT/status.tsv"


        set +e


        bash \
            "$PIPELINE" \
            "$round_name" \
            "$history" \
            "$NUM_CANDIDATES" \
            > "$round_root/pipeline.log" \
            2>&1


        ret=$?


        set -e


        end_time="$(

            date '+%F %T'

        )"


        if [ "$ret" -ne 0 ]; then


            {

                echo "MODEL=$model"

                echo "ROUND=$round_id"

                echo "ROUND_NAME=$round_name"

                echo "EXIT_CODE=$ret"

                echo "STATUS=FAILED"

                echo "START_TIME=$start_time"

                echo "END_TIME=$end_time"

            } > "$round_root/status.txt"


            printf \
                "%s\t%s\t%s\t%s\tFAILED_%s\t%s\t%s\t%s\t%s\t-\n" \
                "$slug" \
                "$model" \
                "$round_id" \
                "$round_name" \
                "$ret" \
                "$start_time" \
                "$end_time" \
                "$round_root/pipeline.log" \
                "$history" \
                >> "$MULTI_ROOT/status.tsv"


            echo

            echo "ERROR: pipeline failed"

            echo "MODEL=$model"

            echo "ROUND=$round_name"

            echo "LOG=$round_root/pipeline.log"


            tail -200 \
                "$round_root/pipeline.log"


            exit "$ret"


        fi


        expected_history="openevolve/v18_llm_nas/${round_name}_augmented_history.md"


        next_history=""


        if [ -f "$expected_history" ]; then


            next_history="$expected_history"


        else


            next_history="$(

                find \
                    openevolve/v18_llm_nas \
                    -maxdepth 1 \
                    -type f \
                    -name '*augmented_history.md' \
                    -newer "$round_root/start.marker" \
                    -printf '%T@ %p\n' \
                    2>/dev/null \
                | sort -nr \
                | head -1 \
                | cut -d' ' -f2-

            )"


        fi


        if [ -z "$next_history" ] || \
           [ ! -f "$next_history" ]; then


            {

                echo "MODEL=$model"

                echo "ROUND=$round_id"

                echo "ROUND_NAME=$round_name"

                echo "EXIT_CODE=0"

                echo "STATUS=FAILED_NO_AUGMENTED_HISTORY"

                echo "START_TIME=$start_time"

                echo "END_TIME=$end_time"

            } > "$round_root/status.txt"


            echo

            echo "ERROR: no augmented history was generated"

            echo "ROUND=$round_name"

            echo "EXPECTED=$expected_history"

            echo "LOG=$round_root/pipeline.log"


            tail -200 \
                "$round_root/pipeline.log"


            exit 3


        fi


        history_copy="$round_root/history_for_next_round.md"


        cp -f \
            "$next_history" \
            "$history_copy"


        {

            echo "MODEL=$model"

            echo "ROUND=$round_id"

            echo "ROUND_NAME=$round_name"

            echo "EXIT_CODE=0"

            echo "STATUS=SUCCESS"

            echo "START_TIME=$start_time"

            echo "END_TIME=$end_time"

            echo "HISTORY_IN=$history"

            echo "HISTORY_GENERATED=$next_history"

            echo "HISTORY_COPY=$history_copy"

            echo "PIPELINE_LOG=$round_root/pipeline.log"

        } > "$round_root/status.txt"


        printf \
            "%s\t%s\t%s\t%s\tSUCCESS\t%s\t%s\t%s\t%s\t%s\n" \
            "$slug" \
            "$model" \
            "$round_id" \
            "$round_name" \
            "$start_time" \
            "$end_time" \
            "$round_root/pipeline.log" \
            "$history" \
            "$history_copy" \
            >> "$MULTI_ROOT/status.tsv"


        printf \
            "%s\t%s\t%s\n" \
            "$slug" \
            "$model" \
            "$round_name" \
            >> "$MULTI_ROOT/models.tsv"


        history="$history_copy"


        echo

        echo "[SUCCESS]"

        echo "MODEL=$model"

        echo "ROUND=$round_name"

        echo "NEXT_HISTORY=$history"


        echo

        echo "===== pipeline tail ====="


        tail -60 \
            "$round_root/pipeline.log"


    done


    echo \
        "END_TIME=$(date '+%F %T')" \
        >> "$model_root/model_manifest.txt"


    echo

    echo "############################################################"

    echo "MODEL SEARCH FINISHED"

    echo "model=$model"

    echo "############################################################"

}


for item in "${MODEL_SPECS[@]}"; do


    slug="${item%%|*}"


    model="${item#*|}"


    run_one_model \
        "$slug" \
        "$model"


done


# ============================================================
# 11. Collect all candidate MME results
# ============================================================

cat > "$MULTI_ROOT/collect_multi_llm_results.py" <<'PY'

import csv

import json

import re

import sys


from pathlib import Path


repo = Path(

    sys.argv[1]

)


root = Path(

    sys.argv[2]

)


models_file = (

    root

    /

    "models.tsv"

)


rows = []


with models_file.open(

    "r",

    encoding="utf-8",

) as f:


    model_rows = list(

        csv.DictReader(

            f,

            delimiter="\t",

        )

    )


for item in model_rows:


    slug = item[

        "slug"

    ]


    model = item[

        "model"

    ]


    round_name = item[

        "round_name"

    ]


    possible_roots = list(

        (

            repo

            /

            "openevolve"

            /

            "runs"

        ).glob(

            f"v18_llm_nas_{round_name}_gpu*"

        )

    )


    for search_root in possible_roots:


        for log_path in sorted(

            search_root.glob(

                "*/mme_full.log"

            )

        ):


            candidate_dir = (

                log_path.parent

            )


            text = log_path.read_text(

                encoding="utf-8",

                errors="ignore",

            )


            totals = [

                float(

                    value

                )

                for value

                in re.findall(

                    r"total score:\s*([0-9]+(?:\.[0-9]+)?)",

                    text,

                    flags=re.I,

                )

            ]


            if len(

                totals

            ) < 2:


                continue


            perception = totals[0]


            cognition = totals[1]


            raw = (

                perception

                +

                cognition

            )


            selected_tokens = ""


            mask_path = (

                candidate_dir

                /

                "mask_hash.jsonl"

            )


            if mask_path.exists():


                try:


                    with mask_path.open(

                        "r",

                        encoding="utf-8",

                    ) as f:


                        first = json.loads(

                            f.readline()

                        )


                    selected_tokens = first.get(

                        "selected_tokens",

                        "",

                    )


                except Exception:


                    selected_tokens = "PARSE_ERROR"


            anchor = (

                repo

                /

                "configs"

                /

                "v18_llm_nas"

                /

                round_name

                /

                "anchors"

                /

                f"anchor_{candidate_dir.name}.yaml"

            )


            merge = (

                repo

                /

                "openevolve"

                /

                "policies"

                /

                "v18_llm_nas"

                /

                round_name

                /

                "merge"

                /

                f"merge_{candidate_dir.name}.yaml"

            )


            rows.append(

                {

                    "slug":

                    slug,

                    "model":

                    model,

                    "round":

                    round_name,

                    "candidate":

                    candidate_dir.name,

                    "mme_total":

                    raw,

                    "perception":

                    perception,

                    "cognition":

                    cognition,

                    "selected_tokens":

                    selected_tokens,

                    "anchor":

                    str(

                        anchor

                    ),

                    "merge":

                    str(

                        merge

                    ),

                    "run_dir":

                    str(

                        candidate_dir

                    ),

                }

            )


rows.sort(

    key=lambda row: (

        row[

            "model"

        ],

        -

        row[

            "mme_total"

        ],

    )

)


all_tsv = (

    root

    /

    "results"

    /

    "all_candidates.tsv"

)


fields = [

    "slug",

    "model",

    "round",

    "candidate",

    "mme_total",

    "perception",

    "cognition",

    "selected_tokens",

    "anchor",

    "merge",

    "run_dir",

]


with all_tsv.open(

    "w",

    encoding="utf-8",

    newline="",

) as f:


    writer = csv.DictWriter(

        f,

        fieldnames=fields,

        delimiter="\t",

    )


    writer.writeheader()


    writer.writerows(

        rows

    )


best = {}


for row in rows:


    model = row[

        "model"

    ]


    if (

        model

        not in

        best

    ):


        best[

            model

        ] = row


summary_tsv = (

    root

    /

    "results"

    /

    "multi_llm_best_summary.tsv"

)


with summary_tsv.open(

    "w",

    encoding="utf-8",

    newline="",

) as f:


    fields_summary = [

        "model",

        "source",

        "best_candidate",

        "mme_total",

        "perception",

        "cognition",

        "selected_tokens",

        "round",

        "anchor",

        "merge",

    ]


    writer = csv.DictWriter(

        f,

        fieldnames=fields_summary,

        delimiter="\t",

    )


    writer.writeheader()


    # Existing qwen-plus reference

    writer.writerow(

        {

            "model":

            "qwen-plus",

            "source":

            "existing_10round_search",

            "best_candidate":

            "fresh5_gpu0_cand5_r04_q2_ref30_dl019_rkw022_cqp102_rs0002",

            "mme_total":

            "1713.46",

            "perception":

            "1413.46",

            "cognition":

            "300.00",

            "selected_tokens":

            "32",

            "round":

            "fresh5_gpu0_cand5_r04",

            "anchor":

            "configs/v18_llm_nas/fresh5_gpu0_cand5_r04/anchors/anchor_fresh5_gpu0_cand5_r04_q2_ref30_dl019_rkw022_cqp102_rs0002.yaml",

            "merge":

            "openevolve/policies/v18_llm_nas/fresh5_gpu0_cand5_r04/merge/merge_fresh5_gpu0_cand5_r04_q2_ref30_dl019_rkw022_cqp102_rs0002.yaml",

        }

    )


    for model in [

        "deepseek-v4-flash",

        "qwen-max",

    ]:


        row = best.get(

            model

        )


        if row is None:


            writer.writerow(

                {

                    "model":

                    model,

                    "source":

                    "new_search",

                    "best_candidate":

                    "NOT_FOUND",

                }

            )


            continue


        writer.writerow(

            {

                "model":

                model,

                "source":

                "new_search",

                "best_candidate":

                row[

                    "candidate"

                ],

                "mme_total":

                f"{row['mme_total']:.6f}",

                "perception":

                f"{row['perception']:.6f}",

                "cognition":

                f"{row['cognition']:.6f}",

                "selected_tokens":

                row[

                    "selected_tokens"

                ],

                "round":

                row[

                    "round"

                ],

                "anchor":

                row[

                    "anchor"

                ],

                "merge":

                row[

                    "merge"

                ],

            }

        )


summary_md = (

    root

    /

    "results"

    /

    "multi_llm_best_summary.md"

)


lines = [

    "# Multi-LLM API Search on LLaVA-1.5-7B / MME / 32 Tokens",

    "",

    "| LLM editor | Best candidate | MME Total | Perception | Cognition | Selected tokens |",

    "|---|---|---:|---:|---:|---:|",

    "| qwen-plus | fresh5_gpu0_cand5_r04_q2_ref30_dl019_rkw022_cqp102_rs0002 | 1713.46 | 1413.46 | 300.00 | 32 |",

]


for model in [

    "deepseek-v4-flash",

    "qwen-max",

]:


    row = best.get(

        model

    )


    if row is None:


        lines.append(

            f"| {model} | NOT_FOUND | - | - | - | - |"

        )


    else:


        lines.append(

            "| "

            +

            model

            +

            " | "

            +

            row[

                "candidate"

            ]

            +

            " | "

            +

            f"{row['mme_total']:.2f}"

            +

            " | "

            +

            f"{row['perception']:.2f}"

            +

            " | "

            +

            f"{row['cognition']:.2f}"

            +

            " | "

            +

            str(

                row[

                    "selected_tokens"

                ]

            )

            +

            " |"

        )


summary_md.write_text(

    "\n".join(

        lines

    )

    +

    "\n",

    encoding="utf-8",

)


print(

    "ALL_CANDIDATES =",

    all_tsv,

)


print(

    "BEST_SUMMARY_TSV =",

    summary_tsv,

)


print(

    "BEST_SUMMARY_MD =",

    summary_md,

)


print()


print(

    summary_md.read_text(

        encoding="utf-8"

    )

)

PY


python \
    "$MULTI_ROOT/collect_multi_llm_results.py" \
    "$ROOT" \
    "$MULTI_ROOT" \
    | tee \
    "$MULTI_ROOT/results/collect_results.log"


# ============================================================
# 12. Final
# ============================================================

echo

echo "============================================================"

echo "ALL MULTI-LLM SEARCHES FINISHED"

echo "============================================================"

echo "END_TIME=$(date '+%F %T')"

echo "MULTI_ROOT=$MULTI_ROOT"

echo "STATUS=$MULTI_ROOT/status.tsv"

echo "SUMMARY=$MULTI_ROOT/results/multi_llm_best_summary.tsv"

echo "SUMMARY_MD=$MULTI_ROOT/results/multi_llm_best_summary.md"

echo "ALL_CANDIDATES=$MULTI_ROOT/results/all_candidates.tsv"

echo "============================================================"


cat \
    "$MULTI_ROOT/results/multi_llm_best_summary.md"

