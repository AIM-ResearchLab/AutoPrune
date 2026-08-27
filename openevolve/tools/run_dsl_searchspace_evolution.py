#!/usr/bin/env python3
from pathlib import Path
from datetime import datetime
import argparse
import json
import os
import random
import re
import subprocess
import sys
import time
import yaml

BASE = Path(__file__).resolve().parents[2]
TEMPLATE_POLICY = BASE / "configs/ablation/exchange_quota_k32_20260714_004819/anchors/anchor_cdpruner_q2_ref30_k32.yaml"
MERGE_POLICY = "openevolve/policies/ablation/exchange_quota_k32_20260714_004819/merge/merge_fixed_cdpruner_q2_best_rs0002.yaml"
OLD_Q0_LOG = BASE / "openevolve/runs/exchange_quota_k32_20260714_004819_fullmme/q0_ref32/score.log"

ALL_SCORES = ["semantic", "attn", "spatial", "redundancy", "contrast"]

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

SPACE_DESC = {
    "parameter_only": "fixed full DSL structure, tune only weights and numeric parameters",
    "score_only": "choose scoring atoms and weights, disable diversity",
    "score_diversity": "choose scoring atoms, weights, and diversity strength",
    "full_typed": "full typed DSL with scoring atoms, fusion, diversity, and residual exchange",
}

def parse_score(log_path: Path):
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
        out = {s: 1.0 / len(scores) for s in scores}
    else:
        out = {s: out[s] / total for s in scores}
    return out

def repair_candidate(raw, space, parent, rng):
    if not isinstance(raw, dict):
        raw = {}

    q_choices = [0, 1, 2, 4, 8]

    if space == "parameter_only":
        scores = list(ALL_SCORES)
    else:
        scores = raw.get("scores", parent.get("scores", ALL_SCORES))
        if not isinstance(scores, list):
            scores = list(parent.get("scores", ALL_SCORES))
        scores = [s for s in scores if s in ALL_SCORES]
        if not scores:
            scores = ["semantic"]

    if space in {"score_only", "score_diversity"}:
        if "semantic" not in scores and rng.random() < 0.6:
            scores = ["semantic"] + scores
        scores = list(dict.fromkeys(scores))

    q = raw.get("replace_quota", parent.get("replace_quota", 2))
    try:
        q = int(q)
    except Exception:
        q = 2
    q = min(q_choices, key=lambda z: abs(z - q))

    if space == "score_only":
        diversity_lambda = 0.0
    else:
        diversity_lambda = clamp(raw.get("diversity_lambda", parent.get("diversity_lambda", 0.19)), 0.0, 0.50)

    weights = raw.get("weights", parent.get("weights", BASE_WEIGHTS))
    if not isinstance(weights, dict):
        weights = dict(BASE_WEIGHTS)
    weights = normalize_weights(weights, scores)

    center_bias = clamp(raw.get("center_bias", parent.get("center_bias", 0.75)), 0.35, 0.95)
    reference_keep_weight = clamp(raw.get("reference_keep_weight", parent.get("reference_keep_weight", 0.22)), 0.0, 0.60)
    candidate_quality_power = clamp(raw.get("candidate_quality_power", parent.get("candidate_quality_power", 1.02)), 0.50, 1.80)

    return {
        "scores": scores,
        "weights": weights,
        "replace_quota": q,
        "min_reference_keep": 32 - q,
        "diversity_lambda": diversity_lambda,
        "center_bias": center_bias,
        "reference_keep_weight": reference_keep_weight,
        "candidate_quality_power": candidate_quality_power,
    }

def mutate_candidate(parent, space, rng):
    raw = dict(parent)

    if space != "parameter_only":
        scores = list(parent.get("scores", ALL_SCORES))
        if rng.random() < 0.55 and len(scores) > 1:
            scores.remove(rng.choice(scores))
        if rng.random() < 0.55:
            add = rng.choice(ALL_SCORES)
            if add not in scores:
                scores.append(add)
        if not scores:
            scores = ["semantic"]
        raw["scores"] = scores
    else:
        raw["scores"] = list(ALL_SCORES)

    weights = dict(parent.get("weights", BASE_WEIGHTS))
    for s in ALL_SCORES:
        weights[s] = max(0.01, weights.get(s, BASE_WEIGHTS[s]) * rng.uniform(0.65, 1.45))
    raw["weights"] = weights

    raw["replace_quota"] = rng.choice([0, 1, 2, 2, 4, 8])
    raw["diversity_lambda"] = parent.get("diversity_lambda", 0.19) + rng.uniform(-0.10, 0.10)
    raw["center_bias"] = parent.get("center_bias", 0.75) + rng.uniform(-0.12, 0.12)
    raw["reference_keep_weight"] = parent.get("reference_keep_weight", 0.22) + rng.uniform(-0.08, 0.08)
    raw["candidate_quality_power"] = parent.get("candidate_quality_power", 1.02) + rng.uniform(-0.18, 0.18)

    return repair_candidate(raw, space, parent, rng)

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

def llm_propose(client, model, space, parent, history, gen, child, rng):
    if client is None:
        return None

    recent = history[-8:]
    prompt = {
        "task": "Propose one executable token-pruning DSL policy candidate as JSON.",
        "space": space,
        "space_description": SPACE_DESC[space],
        "constraints": {
            "token_budget": 32,
            "replace_quota_choices": [0, 1, 2, 4, 8],
            "available_scores": ALL_SCORES,
            "output_json_keys": [
                "scores",
                "weights",
                "replace_quota",
                "diversity_lambda",
                "center_bias",
                "reference_keep_weight",
                "candidate_quality_power"
            ],
            "parameter_only_rule": "For parameter_only, scores must be all five available scores.",
            "score_only_rule": "For score_only, diversity_lambda must be 0.",
            "objective": "maximize MME-Perception under v18 reference-anchored UniMerge runtime"
        },
        "parent": parent,
        "recent_history": recent,
        "generation": gen,
        "child": child
    }

    messages = [
        {
            "role": "system",
            "content": "You design executable DSL policies for visual-token pruning. Return only one compact JSON object. Do not include markdown."
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
    except Exception as e:
        return None

def make_policy(params, out_path: Path, name: str, space: str):
    doc = yaml.safe_load(TEMPLATE_POLICY.read_text(encoding="utf-8"))

    scores = params["scores"]
    q = int(params["replace_quota"])
    ref = 32 - q

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
            keep = False
            for s in scores:
                if SCORE_OUTPUTS[s] == output:
                    keep = True
                    break
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
    meta["method"] = "dsl_search_space_evolution"
    meta["search_space"] = space
    meta["variant"] = name
    meta["enabled_scores"] = scores
    meta["replace_quota"] = q
    meta["min_reference_keep"] = ref
    meta["diversity_lambda"] = float(params["diversity_lambda"])
    meta["center_bias"] = float(params["center_bias"])
    meta["reference_keep_weight"] = float(params["reference_keep_weight"])
    meta["candidate_quality_power"] = float(params["candidate_quality_power"])
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
        ret = subprocess.run(cmd, cwd=str(BASE), stdout=f, stderr=subprocess.STDOUT).returncode

    return ret

def parse_round_results(round_name, candidates):
    run_root = BASE / f"openevolve/runs/v18_llm_nas_{round_name}_gpu0_cand5_seq"
    results = []

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
        results.append(row)

    return results

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--children", type=int, default=5)
    ap.add_argument("--gpus", type=str, default="2,3,4,5,6,7")
    ap.add_argument("--seed", type=int, default=20260719)
    ap.add_argument("--model", type=str, default=os.getenv("OPENAI_MODEL", "qwen-plus"))
    ap.add_argument("--spaces", type=str, default="parameter_only,score_only,score_diversity,full_typed")
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not TEMPLATE_POLICY.exists():
        raise FileNotFoundError(TEMPLATE_POLICY)

    rng = random.Random(args.seed)
    spaces = [x.strip() for x in args.spaces.split(",") if x.strip()]

    run_id = "dsl_searchspace_evo_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = BASE / "openevolve/runs" / run_id
    policy_root = BASE / "configs/ablation" / run_id / "anchors"
    round_policy_root = BASE / "openevolve/policies/v18_llm_nas"
    log_root = BASE / "openevolve/logs" / run_id
    out_root.mkdir(parents=True, exist_ok=True)
    policy_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    (BASE / "openevolve/runs/LATEST_DSL_SEARCHSPACE_EVO_ROOT.txt").write_text(str(out_root.relative_to(BASE)), encoding="utf-8")
    (BASE / "openevolve/runs/LATEST_DSL_SEARCHSPACE_EVO_ID.txt").write_text(run_id, encoding="utf-8")

    client = None if args.no_llm else get_llm_client()
    proposer_name = "llm" if client is not None else "mutation_fallback"

    init_parent = {
        "scores": list(ALL_SCORES),
        "weights": dict(BASE_WEIGHTS),
        "replace_quota": 0,
        "min_reference_keep": 32,
        "diversity_lambda": 0.19,
        "center_bias": 0.75,
        "reference_keep_weight": 0.22,
        "candidate_quality_power": 1.02,
    }

    parents = {space: dict(init_parent) for space in spaces}
    best = {space: None for space in spaces}
    history = {space: [] for space in spaces}
    all_results = []

    print("[INFO] RUN_ID=", run_id, flush=True)
    print("[INFO] OUT_ROOT=", out_root, flush=True)
    print("[INFO] SPACES=", spaces, flush=True)
    print("[INFO] ROUNDS=", args.rounds, "CHILDREN=", args.children, "GPUS=", args.gpus, flush=True)
    print("[INFO] PROPOSER=", proposer_name, "MODEL=", args.model, flush=True)

    for gen in range(1, args.rounds + 1):
        round_name = f"{run_id}_g{gen:02d}"
        round_dir = round_policy_root / round_name
        round_dir.mkdir(parents=True, exist_ok=True)

        manifest_rows = ["name|anchor_policy|merge_policy"]
        candidates = []

        for space in spaces:
            parent = parents[space]
            for child in range(args.children):
                prop = llm_propose(client, args.model, space, parent, history[space], gen, child, rng)
                if prop is None:
                    prop = mutate_candidate(parent, space, rng)

                name = f"{space}_g{gen:02d}_c{child:02d}"
                q = int(prop["replace_quota"])
                ref = 32 - q
                policy_name = f"{name}_q{q}_ref{ref}_k32"
                policy_path = policy_root / f"{policy_name}.yaml"
                make_policy(prop, policy_path, policy_name, space)

                rel_policy = policy_path.relative_to(BASE)
                manifest_rows.append(f"{name}|{rel_policy}|{MERGE_POLICY}")

                cand = {
                    "space": space,
                    "generation": gen,
                    "child": child,
                    "name": name,
                    "policy_name": policy_name,
                    "anchor_policy": str(rel_policy),
                    "merge_policy": MERGE_POLICY,
                    "proposal_source": proposer_name,
                    "scores": ",".join(prop["scores"]),
                    "weights_json": json.dumps(prop["weights"], ensure_ascii=False, sort_keys=True),
                    "replace_quota": prop["replace_quota"],
                    "fixed_reference_tokens": prop["min_reference_keep"],
                    "diversity_lambda": prop["diversity_lambda"],
                    "center_bias": prop["center_bias"],
                    "reference_keep_weight": prop["reference_keep_weight"],
                    "candidate_quality_power": prop["candidate_quality_power"],
                }
                candidates.append(cand)

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
            space = row["space"]
            p = row["perception"]
            history[space].append({
                "generation": row["generation"],
                "name": row["name"],
                "perception": p,
                "scores": row["scores"],
                "replace_quota": row["replace_quota"],
                "diversity_lambda": row["diversity_lambda"],
            })
            if p is not None:
                cur = best[space]
                if cur is None or p > cur["perception"]:
                    best[space] = row
                    parents[space] = {
                        "scores": row["scores"].split(",") if row["scores"] else list(ALL_SCORES),
                        "weights": json.loads(row["weights_json"]),
                        "replace_quota": int(row["replace_quota"]),
                        "min_reference_keep": int(row["fixed_reference_tokens"]),
                        "diversity_lambda": float(row["diversity_lambda"]),
                        "center_bias": float(row["center_bias"]),
                        "reference_keep_weight": float(row["reference_keep_weight"]),
                        "candidate_quality_power": float(row["candidate_quality_power"]),
                    }

        for space in spaces:
            b = best[space]
            if b is None:
                print(f"[BEST after G{gen:02d}] {space}: NONE", flush=True)
            else:
                print(f"[BEST after G{gen:02d}] {space}: {b['name']} P={b['perception']:.6f} q={b['replace_quota']} scores={b['scores']}", flush=True)

        save_outputs(out_root, spaces, all_results, best, proposer_name)

    save_outputs(out_root, spaces, all_results, best, proposer_name)
    print("[DONE] OUT_ROOT=", out_root, flush=True)

def save_outputs(out_root, spaces, all_results, best, proposer_name):
    def fmt(x):
        return "" if x is None else f"{x:.6f}"

    all_path = out_root / "all_candidates.tsv"
    keys = [
        "space", "generation", "child", "name", "proposal_source",
        "scores", "replace_quota", "fixed_reference_tokens", "diversity_lambda",
        "center_bias", "reference_keep_weight", "candidate_quality_power",
        "perception", "cognition", "total", "exit_code", "mask_hash_rows",
        "anchor_policy", "run_dir"
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

    summary = out_root / "table_dsl_searchspace_evolution.tsv"
    anchor_p, _, _ = parse_score(OLD_Q0_LOG)
    if anchor_p is None:
        anchor_p = 1378.574830
    with summary.open("w", encoding="utf-8") as f:
        f.write("search_space\tsearch_budget\tbest_candidate\tbest_generation\tbest_mme_perception\treplacement_quota\tscores\tproposal_source\n")
        f.write(f"anchor_only\t0\tanchor_only\t0\t{fmt(anchor_p)}\t0\treference_only\tfixed\n")
        for space in spaces:
            b = best.get(space)
            if b is None:
                f.write(f"{space}\t10x5\t\t\t\t\t\t{proposer_name}\n")
            else:
                f.write(f"{space}\t10x5\t{b['name']}\t{b['generation']}\t{fmt(b['perception'])}\t{b['replace_quota']}\t{b['scores']}\t{proposer_name}\n")

    state = {
        "spaces": spaces,
        "proposal_source": proposer_name,
        "best": best,
    }
    (out_root / "state.json").write_text(json.dumps(state, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    print("[SAVE]", summary, flush=True)
    print(summary.read_text(encoding="utf-8"), flush=True)

if __name__ == "__main__":
    main()
