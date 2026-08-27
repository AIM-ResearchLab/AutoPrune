#!/usr/bin/env python3
from pathlib import Path
from datetime import datetime
import argparse
import csv
import json
import math
import os
import random
import re
import subprocess
import sys
import yaml

BASE = Path(__file__).resolve().parents[2]

TEMPLATE_POLICY = BASE / "configs/ablation/exchange_quota_k32_20260714_004819/anchors/anchor_cdpruner_q2_ref30_k32.yaml"
MERGE_POLICY = "openevolve/policies/ablation/exchange_quota_k32_20260714_004819/merge/merge_fixed_cdpruner_q2_best_rs0002.yaml"

ALL_SCORES = ["semantic", "attn", "spatial", "redundancy", "contrast"]
SINGLE_SCORE_CHOICES = ["semantic", "attn", "spatial", "contrast"]

SCORE_OUTPUTS = {
    "semantic": "semantic_score",
    "attn": "attn_score",
    "spatial": "spatial_score",
    "redundancy": "redundancy_score",
    "contrast": "contrast_score",
}

BASE_WEIGHTS = {
    "semantic": 0.28,
    "attn": 0.10,
    "spatial": 0.16,
    "redundancy": 0.07,
    "contrast": 0.06,
}

SPACE_DESCRIPTIONS = {
    "full": "Full TPDSL. Use multi-score fusion, diversity-aware selection, and reference-anchored residual exchange. Fixed qe=2 and rmin=30.",
    "wo_diversity": "Full TPDSL without the diversity module. Use multi-score fusion and reference-anchored residual exchange, but set diversity_lambda=0. Fixed qe=2 and rmin=30.",
    "wo_multiscore": "Full TPDSL without multi-score fusion. Use only one scoring atom, diversity-aware selection, and reference-anchored residual exchange. Fixed qe=2 and rmin=30.",
}

def parse_score(log_path):
    text = log_path.read_text(errors="ignore") if log_path.exists() else ""
    vals = [float(x) for x in re.findall(r"total score:\s*([0-9.]+)", text)]
    if len(vals) >= 2:
        return vals[0], vals[1], vals[0] + vals[1]
    return None, None, None

def clamp(x, lo, hi):
    try:
        x = float(x)
    except Exception:
        x = lo
    return max(lo, min(hi, x))

def normalize_weights(weights, scores):
    out = {}
    for s in scores:
        out[s] = clamp(weights.get(s, BASE_WEIGHTS.get(s, 0.1)), 0.01, 1.0)
    total = sum(out.values())
    if total <= 0:
        return {s: 1.0 / len(scores) for s in scores}
    return {s: out[s] / total for s in scores}

def get_llm_client():
    key = os.getenv("OPENAI_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
    if not key:
        return None
    base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE") or "https://dashscope.aliyuncs.com/compatible-mode/v1"
    try:
        from openai import OpenAI
        return OpenAI(api_key=key, base_url=base_url)
    except Exception:
        return None

def default_parent(space):
    if space == "wo_multiscore":
        scores = ["semantic"]
    else:
        scores = list(ALL_SCORES)

    div = 0.0 if space == "wo_diversity" else 0.19

    return {
        "scores": scores,
        "weights": normalize_weights(BASE_WEIGHTS, scores),
        "replace_quota": 2,
        "min_reference_keep": 30,
        "diversity_lambda": div,
        "center_bias": 0.75,
        "reference_keep_weight": 0.22,
        "candidate_quality_power": 1.02,
    }

def repair_candidate(raw, space, parent, rng):
    if not isinstance(raw, dict):
        raw = {}

    parent_scores = parent.get("scores", list(ALL_SCORES))
    raw_scores = raw.get("scores", parent_scores)
    if not isinstance(raw_scores, list):
        raw_scores = list(parent_scores)

    if space == "wo_multiscore":
        candidates = [s for s in raw_scores if s in SINGLE_SCORE_CHOICES]
        if not candidates:
            candidates = [s for s in parent_scores if s in SINGLE_SCORE_CHOICES]
        if not candidates:
            candidates = ["semantic"]
        scores = [candidates[0]]
    else:
        scores = [s for s in raw_scores if s in ALL_SCORES]
        if not scores:
            scores = list(parent_scores)
        scores = [s for s in scores if s in ALL_SCORES]
        if not scores:
            scores = list(ALL_SCORES)
        if "semantic" not in scores and rng.random() < 0.70:
            scores = ["semantic"] + scores
        scores = list(dict.fromkeys(scores))

    weights = raw.get("weights", parent.get("weights", BASE_WEIGHTS))
    if not isinstance(weights, dict):
        weights = dict(BASE_WEIGHTS)
    weights = normalize_weights(weights, scores)

    if space == "wo_diversity":
        diversity_lambda = 0.0
    else:
        diversity_lambda = clamp(raw.get("diversity_lambda", parent.get("diversity_lambda", 0.19)), 0.05, 0.50)

    center_bias = clamp(raw.get("center_bias", parent.get("center_bias", 0.75)), 0.35, 0.95)
    reference_keep_weight = clamp(raw.get("reference_keep_weight", parent.get("reference_keep_weight", 0.22)), 0.0, 0.60)
    candidate_quality_power = clamp(raw.get("candidate_quality_power", parent.get("candidate_quality_power", 1.02)), 0.50, 1.80)

    return {
        "scores": scores,
        "weights": weights,
        "replace_quota": 2,
        "min_reference_keep": 30,
        "diversity_lambda": diversity_lambda,
        "center_bias": center_bias,
        "reference_keep_weight": reference_keep_weight,
        "candidate_quality_power": candidate_quality_power,
    }

def mutate_candidate(parent, space, rng):
    raw = dict(parent)

    if space == "wo_multiscore":
        raw["scores"] = [rng.choice(SINGLE_SCORE_CHOICES)]
    else:
        scores = list(parent.get("scores", ALL_SCORES))
        if rng.random() < 0.45 and len(scores) > 2:
            scores.remove(rng.choice(scores))
        if rng.random() < 0.55:
            add = rng.choice(ALL_SCORES)
            if add not in scores:
                scores.append(add)
        if not scores:
            scores = ["semantic"]
        raw["scores"] = scores

    weights = dict(parent.get("weights", BASE_WEIGHTS))
    for s in ALL_SCORES:
        weights[s] = max(0.01, weights.get(s, BASE_WEIGHTS[s]) * rng.uniform(0.70, 1.35))
    raw["weights"] = weights

    if space == "wo_diversity":
        raw["diversity_lambda"] = 0.0
    else:
        raw["diversity_lambda"] = parent.get("diversity_lambda", 0.19) + rng.uniform(-0.08, 0.08)

    raw["center_bias"] = parent.get("center_bias", 0.75) + rng.uniform(-0.10, 0.10)
    raw["reference_keep_weight"] = parent.get("reference_keep_weight", 0.22) + rng.uniform(-0.06, 0.06)
    raw["candidate_quality_power"] = parent.get("candidate_quality_power", 1.02) + rng.uniform(-0.15, 0.15)

    return repair_candidate(raw, space, parent, rng)

def llm_propose(client, model, space, parent, history, gen, child, rng):
    if client is None:
        return None

    prompt = {
        "task": "Propose one executable TPDSL token-pruning policy candidate as JSON.",
        "space": space,
        "space_description": SPACE_DESCRIPTIONS[space],
        "fixed_constraints": {
            "token_budget": 32,
            "replace_quota": 2,
            "min_reference_keep": 30,
            "budget_control": "fixed",
            "reassembly": "fixed",
            "objective": "maximize MME-Perception",
        },
        "available_scores": ALL_SCORES,
        "single_score_choices_for_wo_multiscore": SINGLE_SCORE_CHOICES,
        "output_json_keys": [
            "scores",
            "weights",
            "diversity_lambda",
            "center_bias",
            "reference_keep_weight",
            "candidate_quality_power"
        ],
        "parent": parent,
        "recent_history": history[-8:],
        "generation": gen,
        "child": child
    }

    messages = [
        {
            "role": "system",
            "content": "You design executable TPDSL token-pruning policies. Return only one compact JSON object. Do not include markdown."
        },
        {
            "role": "user",
            "content": json.dumps(prompt, ensure_ascii=False)
        }
    ]

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,
            max_tokens=512,
        )
        text = resp.choices[0].message.content.strip()
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            return None
        raw = json.loads(m.group(0))
        return repair_candidate(raw, space, parent, rng)
    except Exception:
        return None

def make_policy(params, out_path, name, space, seed):
    doc = yaml.safe_load(TEMPLATE_POLICY.read_text(encoding="utf-8"))

    scores = params["scores"]
    q = 2
    ref = 30

    doc["policy_name"] = name
    doc["name"] = name
    doc["keep_tokens"] = 32
    doc["exchange_quota"] = q
    doc["fixed_reference_tokens"] = ref

    new_pipeline = []
    for node in doc.get("pipeline", []):
        op = node.get("op")
        output = node.get("output")

        if op == "compute_score" and output in SCORE_OUTPUTS.values():
            keep = any(SCORE_OUTPUTS[s] == output for s in scores)
            if not keep:
                continue

        if op == "compute_score" and node.get("name") == "spatial_centrality_prior":
            node = dict(node)
            node["params"] = dict(node.get("params", {}))
            node["params"]["center_bias"] = float(params["center_bias"])

        if op == "product_fuse_score":
            node = dict(node)
            node["inputs"] = dict(node.get("inputs", {}))
            node["inputs"]["scores"] = {s: SCORE_OUTPUTS[s] for s in scores}
            node["params"] = dict(node.get("params", {}))
            node["params"]["weights"] = {s: float(params["weights"][s]) for s in scores}

        if op == "select":
            node = dict(node)
            node["params"] = dict(node.get("params", {}))
            node["params"]["replace_quota"] = q
            node["params"]["min_reference_keep"] = ref
            node["params"]["diversity_lambda"] = float(params["diversity_lambda"])
            node["params"]["reference_keep_weight"] = float(params["reference_keep_weight"])
            node["params"]["candidate_quality_power"] = float(params["candidate_quality_power"])
            node["params"]["sort_selected"] = True

        new_pipeline.append(node)

    doc["pipeline"] = new_pipeline

    meta = doc.get("meta", {})
    if not isinstance(meta, dict):
        meta = {}

    meta["method"] = "tpdsl_module_evolution"
    meta["search_space"] = space
    meta["seed"] = int(seed)
    meta["enabled_scores"] = scores
    meta["replace_quota"] = q
    meta["min_reference_keep"] = ref
    meta["diversity_enabled"] = space != "wo_diversity"
    meta["multi_score_fusion_enabled"] = space != "wo_multiscore"
    meta["diversity_lambda"] = float(params["diversity_lambda"])
    meta["reference_keep_weight"] = float(params["reference_keep_weight"])
    meta["candidate_quality_power"] = float(params["candidate_quality_power"])
    meta["center_bias"] = float(params["center_bias"])
    meta["candidate_name"] = name

    doc["meta"] = meta
    doc["policy_fp"] = name

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(doc, allow_unicode=True, sort_keys=False), encoding="utf-8")

def run_eval_round(round_name, gpus, log_path):
    cmd = [
        sys.executable,
        "openevolve/v18_llm_nas/run_round_gpu_pool_dynamic.py",
        "--round",
        round_name,
        "--gpus",
        gpus,
    ]
    with log_path.open("w", encoding="utf-8") as f:
        return subprocess.run(cmd, cwd=str(BASE), stdout=f, stderr=subprocess.STDOUT).returncode

def parse_round_results(round_name, candidates):
    run_root = BASE / f"openevolve/runs/v18_llm_nas_{round_name}_gpu0_cand5_seq"
    rows = []

    for cand in candidates:
        name = cand["name"]
        d = run_root / name
        p, c, t = parse_score(d / "mme_full.log")

        status_text = (d / "status.txt").read_text(errors="ignore") if (d / "status.txt").exists() else ""
        exit_code = ""
        mask_rows = ""

        for line in status_text.splitlines():
            if line.startswith("EXIT_CODE="):
                exit_code = line.split("=", 1)[1].strip()
            if line.startswith("MASK_HASH_ROWS="):
                mask_rows = line.split("=", 1)[1].strip()

        row = dict(cand)
        row["perception"] = p
        row["cognition"] = c
        row["total"] = t
        row["exit_code"] = exit_code
        row["mask_hash_rows"] = mask_rows
        row["run_dir"] = str(d.relative_to(BASE))
        rows.append(row)

    return rows

def fnum(x):
    try:
        if x is None or str(x).strip() == "":
            return None
        return float(x)
    except Exception:
        return None

def fmt(x):
    return "" if x is None else f"{x:.6f}"

def save_outputs(out_root, args, all_results, best, proposal_source):
    all_path = out_root / "all_candidates.tsv"

    keys = [
        "space",
        "seed",
        "generation",
        "child",
        "name",
        "proposal_source",
        "scores",
        "replace_quota",
        "fixed_reference_tokens",
        "diversity_enabled",
        "multi_score_fusion_enabled",
        "diversity_lambda",
        "center_bias",
        "reference_keep_weight",
        "candidate_quality_power",
        "perception",
        "cognition",
        "total",
        "exit_code",
        "mask_hash_rows",
        "anchor_policy",
        "run_dir"
    ]

    with all_path.open("w", encoding="utf-8") as f:
        f.write("\t".join(keys) + "\n")
        for r in all_results:
            vals = []
            for k in keys:
                v = r.get(k, "")
                if k in {"perception", "cognition", "total"}:
                    vals.append(fmt(v))
                else:
                    vals.append(str(v))
            f.write("\t".join(vals) + "\n")

    summary = out_root / "table_tpdsl_module_evo.tsv"
    with summary.open("w", encoding="utf-8") as f:
        f.write("space\tseed\tsearch_budget\tbest_candidate\tbest_generation\tbest_mme_perception\treplacement_quota\tfixed_reference_tokens\tdiversity_enabled\tmulti_score_fusion_enabled\tscores\tproposal_source\n")
        if best is None:
            f.write(f"{args.space}\t{args.seed}\t{args.rounds}x{args.children}\t\t\t\t2\t30\t\t\t\t{proposal_source}\n")
        else:
            f.write(f"{args.space}\t{args.seed}\t{args.rounds}x{args.children}\t{best['name']}\t{best['generation']}\t{fmt(best['perception'])}\t{best['replace_quota']}\t{best['fixed_reference_tokens']}\t{best['diversity_enabled']}\t{best['multi_score_fusion_enabled']}\t{best['scores']}\t{proposal_source}\n")

    state = {
        "run_id": args.run_id,
        "space": args.space,
        "seed": args.seed,
        "rounds": args.rounds,
        "children": args.children,
        "gpus": args.gpus,
        "proposal_source": proposal_source,
        "best": best,
    }

    (out_root / "state.json").write_text(json.dumps(state, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    print("[SAVE]", summary, flush=True)
    print(summary.read_text(encoding="utf-8"), flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", type=str, required=True, choices=["full", "wo_diversity", "wo_multiscore"])
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--children", type=int, default=5)
    ap.add_argument("--gpus", type=str, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", type=str, default=os.getenv("OPENAI_MODEL", "qwen-plus"))
    ap.add_argument("--run-id", type=str, default="")
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not TEMPLATE_POLICY.exists():
        raise FileNotFoundError(TEMPLATE_POLICY)

    rng = random.Random(args.seed)

    if args.run_id:
        run_id = args.run_id
    else:
        run_id = f"tpdsl_{args.space}_s{args.seed}_" + datetime.now().strftime("%Y%m%d_%H%M%S")

    args.run_id = run_id

    out_root = BASE / "openevolve/runs" / run_id
    policy_root = BASE / "configs/ablation" / run_id / "anchors"
    round_policy_root = BASE / "openevolve/policies/v18_llm_nas"
    log_root = BASE / "openevolve/logs" / run_id

    out_root.mkdir(parents=True, exist_ok=True)
    policy_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    (BASE / "openevolve/runs/LATEST_TPDSL_MODULE_EVO_ROOT.txt").write_text(str(out_root.relative_to(BASE)), encoding="utf-8")
    (BASE / "openevolve/runs/LATEST_TPDSL_MODULE_EVO_ID.txt").write_text(run_id, encoding="utf-8")

    client = None if args.no_llm else get_llm_client()
    proposal_source = "llm" if client is not None else "mutation_fallback"

    print("[INFO] RUN_ID=", run_id, flush=True)
    print("[INFO] OUT_ROOT=", out_root, flush=True)
    print("[INFO] SPACE=", args.space, flush=True)
    print("[INFO] SPACE_DESC=", SPACE_DESCRIPTIONS[args.space], flush=True)
    print("[INFO] SEED=", args.seed, flush=True)
    print("[INFO] ROUNDS=", args.rounds, "CHILDREN=", args.children, "GPUS=", args.gpus, flush=True)
    print("[INFO] PROPOSER=", proposal_source, "MODEL=", args.model, flush=True)

    parent = default_parent(args.space)
    history = []
    all_results = []
    best = None

    for gen in range(1, args.rounds + 1):
        round_name = f"{run_id}_g{gen:02d}"
        round_dir = round_policy_root / round_name
        round_dir.mkdir(parents=True, exist_ok=True)

        manifest_rows = ["name|anchor_policy|merge_policy"]
        candidates = []

        for child in range(args.children):
            prop = llm_propose(client, args.model, args.space, parent, history, gen, child, rng)
            if prop is None:
                prop = mutate_candidate(parent, args.space, rng)

            name = f"{args.space}_s{args.seed}_g{gen:02d}_c{child:02d}"
            policy_name = f"{name}_q2_ref30_k32"
            policy_path = policy_root / f"{policy_name}.yaml"

            make_policy(prop, policy_path, policy_name, args.space, args.seed)

            rel_policy = policy_path.relative_to(BASE)
            manifest_rows.append(f"{name}|{rel_policy}|{MERGE_POLICY}")

            candidates.append({
                "space": args.space,
                "seed": args.seed,
                "generation": gen,
                "child": child,
                "name": name,
                "policy_name": policy_name,
                "anchor_policy": str(rel_policy),
                "merge_policy": MERGE_POLICY,
                "proposal_source": proposal_source,
                "scores": ",".join(prop["scores"]),
                "weights_json": json.dumps(prop["weights"], ensure_ascii=False, sort_keys=True),
                "replace_quota": prop["replace_quota"],
                "fixed_reference_tokens": prop["min_reference_keep"],
                "diversity_enabled": args.space != "wo_diversity",
                "multi_score_fusion_enabled": args.space != "wo_multiscore",
                "diversity_lambda": prop["diversity_lambda"],
                "center_bias": prop["center_bias"],
                "reference_keep_weight": prop["reference_keep_weight"],
                "candidate_quality_power": prop["candidate_quality_power"],
            })

        manifest = round_dir / "manifest.psv"
        manifest.write_text("\n".join(manifest_rows) + "\n", encoding="utf-8")
        print(f"[GEN {gen:02d}] manifest={manifest} candidates={len(candidates)}", flush=True)

        if args.dry_run:
            continue

        round_log = log_root / f"{round_name}.log"
        ret = run_eval_round(round_name, args.gpus, round_log)
        print(f"[GEN {gen:02d}] eval_exit={ret} log={round_log}", flush=True)

        results = parse_round_results(round_name, candidates)
        all_results.extend(results)

        for row in results:
            p = fnum(row.get("perception"))

            history.append({
                "generation": row["generation"],
                "name": row["name"],
                "perception": p,
                "scores": row["scores"],
                "diversity_lambda": row["diversity_lambda"],
                "reference_keep_weight": row["reference_keep_weight"],
            })

            if p is not None:
                if best is None or p > best["perception"]:
                    best = row
                    parent = {
                        "scores": row["scores"].split(",") if row["scores"] else default_parent(args.space)["scores"],
                        "weights": json.loads(row["weights_json"]),
                        "replace_quota": 2,
                        "min_reference_keep": 30,
                        "diversity_lambda": float(row["diversity_lambda"]),
                        "center_bias": float(row["center_bias"]),
                        "reference_keep_weight": float(row["reference_keep_weight"]),
                        "candidate_quality_power": float(row["candidate_quality_power"]),
                    }

        if best is None:
            print(f"[BEST after G{gen:02d}] NONE", flush=True)
        else:
            print(f"[BEST after G{gen:02d}] {best['name']} P={best['perception']:.6f} scores={best['scores']} div={best['diversity_lambda']}", flush=True)

        save_outputs(out_root, args, all_results, best, proposal_source)

    save_outputs(out_root, args, all_results, best, proposal_source)
    print("[DONE] OUT_ROOT=", out_root, flush=True)

if __name__ == "__main__":
    main()
