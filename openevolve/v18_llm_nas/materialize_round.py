#!/usr/bin/env python3
import argparse
import copy
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def load_yaml(path: Path):
    return yaml.safe_load(path.read_text())


def ensure_base_files():
    q0 = ROOT / "configs/base/cdpruner_k32_reference_q0.yaml"
    merge_initial = ROOT / "openevolve/policies/base/merge_anchor_only.yaml"
    merge_res = ROOT / "openevolve/policies/base/merge_residual_template.yaml"

    missing = [p for p in (q0, merge_initial, merge_res) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "missing AutoPrune base template(s): " + ", ".join(str(x) for x in missing)
        )

    return q0, merge_initial, merge_res

def patch_anchor(base_anchor, spec, out_dir: Path):
    d = copy.deepcopy(base_anchor)
    name = "anchor_" + spec["name"]

    d["policy_name"] = name
    d["name"] = name
    d["keep_tokens"] = 32

    for step in d.get("pipeline", []):
        if step.get("op") == "select" and step.get("name") == "native_reference_residual_exchange_selector":
            p = step.setdefault("params", {})
            p["replace_quota"] = int(spec["replace_quota"])
            p["min_reference_keep"] = int(spec["min_reference_keep"])
            p["diversity_lambda"] = float(spec["diversity_lambda"])
            p["reference_keep_weight"] = float(spec["reference_keep_weight"])
            p["candidate_quality_power"] = float(spec["candidate_quality_power"])
            p["sort_selected"] = True

    d.setdefault("meta", {})
    d["meta"].update({
        "method": "v18_llm_nas_anchor",
        "candidate_name": spec["name"],
        "replace_quota": int(spec["replace_quota"]),
        "min_reference_keep": int(spec["min_reference_keep"]),
        "diversity_lambda": float(spec["diversity_lambda"]),
        "reference_keep_weight": float(spec["reference_keep_weight"]),
        "candidate_quality_power": float(spec["candidate_quality_power"]),
        "baseline": "clean_cdpruner_k32_1382",
    })

    path = out_dir / f"{name}.yaml"
    path.write_text(yaml.safe_dump(d, sort_keys=False, allow_unicode=True))
    return path


def replace_scale(obj, scale: float):
    changed = 0

    if isinstance(obj, dict):
        for k, v in list(obj.items()):
            if k in {"scale", "residual_scale", "alpha", "merge_scale"} and isinstance(v, (int, float)):
                obj[k] = float(scale)
                changed += 1
            elif isinstance(v, float) and abs(v - 0.005) < 1e-12:
                obj[k] = float(scale)
                changed += 1
            else:
                changed += replace_scale(v, scale)

    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            if isinstance(v, float) and abs(v - 0.005) < 1e-12:
                obj[i] = float(scale)
                changed += 1
            else:
                changed += replace_scale(v, scale)

    return changed


def patch_merge(base_initial, base_residual, spec, out_dir: Path):
    scale = float(spec["residual_scale"])

    if scale <= 0:
        d = copy.deepcopy(base_initial)
        name = "merge_" + spec["name"] + "_anchor_only"
    else:
        d = copy.deepcopy(base_residual)
        name = "merge_" + spec["name"]
        changed = replace_scale(d, scale)
        d.setdefault("meta", {})
        d["meta"]["changed_scale_fields"] = changed

    d["policy_name"] = name
    d["name"] = name
    d.setdefault("meta", {})
    d["meta"].update({
        "method": "v18_llm_nas_merge",
        "candidate_name": spec["name"],
        "residual_scale": scale,
        "baseline": "clean_cdpruner_k32_1382",
    })

    path = out_dir / f"{name}.yaml"
    path.write_text(yaml.safe_dump(d, sort_keys=False, allow_unicode=True))
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", required=True)
    parser.add_argument("--spec-jsonl", required=True)
    args = parser.parse_args()

    q0_path, merge_initial_path, merge_res_path = ensure_base_files()

    spec_path = Path(args.spec_jsonl)
    if not spec_path.exists():
        raise FileNotFoundError(spec_path)

    anchor_out = ROOT / f"configs/v18_llm_nas/{args.round}/anchors"
    merge_out = ROOT / f"openevolve/policies/v18_llm_nas/{args.round}/merge"
    manifest = ROOT / f"openevolve/policies/v18_llm_nas/{args.round}/manifest.psv"

    anchor_out.mkdir(parents=True, exist_ok=True)
    merge_out.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)

    base_anchor = load_yaml(q0_path)
    base_initial = load_yaml(merge_initial_path)
    base_residual = load_yaml(merge_res_path)

    rows = []
    seen = set()

    for line in spec_path.read_text().splitlines():
        if not line.strip():
            continue

        spec = json.loads(line)
        name = spec["name"]

        if name in seen:
            raise ValueError(f"duplicate name: {name}")
        seen.add(name)

        anchor = patch_anchor(base_anchor, spec, anchor_out)
        merge = patch_merge(base_initial, base_residual, spec, merge_out)

        rows.append((name, anchor.relative_to(ROOT), merge.relative_to(ROOT)))

    with manifest.open("w") as f:
        f.write("name|anchor_policy|merge_policy\n")
        for name, anchor, merge in rows:
            f.write(f"{name}|{anchor}|{merge}\n")

    print(f"wrote {manifest}")
    print(f"num_candidates={len(rows)}")


if __name__ == "__main__":
    main()
