#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys
import urllib.request
from pathlib import Path


CLEAN_P = 1382.8248299319728
CLEAN_C = 304.2857142857143


def read_text(path: Path) -> str:
    return path.read_text(errors="ignore") if path.exists() else ""


def call_openai_compatible(messages, temperature=None, max_tokens=5000) -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL", "").rstrip("/")
    model = os.environ.get("OPENAI_MODEL")

    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    if not base_url:
        raise RuntimeError("OPENAI_BASE_URL is not set")
    if not model:
        raise RuntimeError("OPENAI_MODEL is not set")

    if temperature is None:
        temperature = float(os.environ.get("FRESH_LLM_TEMPERATURE", "0.75"))

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    req = urllib.request.Request(
        base_url + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=180) as resp:
        obj = json.loads(resp.read().decode("utf-8"))

    return obj["choices"][0]["message"]["content"]


def extract_jsonl(text: str):
    rows = []

    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("```"):
            continue
        if s.startswith("{") and s.endswith("}"):
            try:
                rows.append(json.loads(s))
            except Exception:
                pass

    if not rows:
        for m in re.finditer(r"\{[^{}]*\}", text, flags=re.S):
            try:
                rows.append(json.loads(m.group(0)))
            except Exception:
                pass

    return rows


def validate_candidate(x):
    required = [
        "name",
        "replace_quota",
        "min_reference_keep",
        "diversity_lambda",
        "reference_keep_weight",
        "candidate_quality_power",
        "residual_scale",
    ]
    for k in required:
        if k not in x:
            raise ValueError(f"missing key {k}: {x}")

    name = str(x["name"])
    if not re.match(r"^[A-Za-z0-9_]+$", name):
        raise ValueError(f"bad name: {name}")

    rq = int(x["replace_quota"])
    mk = int(x["min_reference_keep"])
    div = float(x["diversity_lambda"])
    refw = float(x["reference_keep_weight"])
    power = float(x["candidate_quality_power"])
    rs = float(x["residual_scale"])

    if not (0 <= rq <= 3):
        raise ValueError(f"replace_quota out of range: {x}")
    if not (29 <= mk <= 32):
        raise ValueError(f"min_reference_keep out of range: {x}")
    if mk < 32 - rq:
        raise ValueError(f"min_reference_keep too small for replace_quota: {x}")
    if not (0.04 <= div <= 0.22):
        raise ValueError(f"diversity_lambda out of range: {x}")
    if not (0.10 <= refw <= 0.80):
        raise ValueError(f"reference_keep_weight out of range: {x}")
    if not (0.70 <= power <= 1.50):
        raise ValueError(f"candidate_quality_power out of range: {x}")
    if not (0.0 <= rs <= 0.006):
        raise ValueError(f"residual_scale out of range: {x}")

    if rq >= 2 and rs > 0.003:
        raise ValueError(f"q2/q3 residual should be weak <=0.003: {x}")

    return {
        "name": name,
        "replace_quota": rq,
        "min_reference_keep": mk,
        "diversity_lambda": div,
        "reference_keep_weight": refw,
        "candidate_quality_power": power,
        "residual_scale": rs,
    }


def parse_best_from_history(history: str):
    best_p = CLEAN_P
    best_name = "clean_cdpruner_k32"

    for line in history.splitlines():
        if not line.startswith("|"):
            continue
        if "Perception" in line or "---" in line:
            continue
        parts = [x.strip() for x in line.strip().strip("|").split("|")]
        if len(parts) < 5:
            continue
        try:
            name = parts[0]
            p = float(parts[3])
            if p > best_p:
                best_p = p
                best_name = name
        except Exception:
            pass

    return best_name, best_p


def build_prompt(history: str, round_name: str, num_candidates: int) -> str:
    best_name, best_p = parse_best_from_history(history)

    return f"""
You are an LLM NAS planner for a constrained DSL-based visual token pruning system.

This is a FRESH AutoNAS run. Do not assume access to any previous manual final policy.

TRUE BASELINE:
- Clean CDPruner K=32.
- Baseline Perception = {CLEAN_P:.6f}.
- Baseline Cognition = {CLEAN_C:.6f}.
- VTN must remain 32.
- A candidate is successful if Perception > {CLEAN_P:.6f}.

Current history:
{history}

Current best parsed from history:
- name = {best_name}
- perception = {best_p:.4f}

DSL SEARCH SPACE:
Each candidate must be a JSON object with:
- name: unique identifier using only letters, numbers, underscores.
- replace_quota: integer 0, 1, 2, or 3.
- min_reference_keep: integer 29 to 32, satisfying min_reference_keep >= 32 - replace_quota.
- diversity_lambda: float in [0.04, 0.22].
- reference_keep_weight: float in [0.10, 0.80].
- candidate_quality_power: float in [0.70, 1.50].
- residual_scale: float in [0.0, 0.006]. Use 0.0 for no residual.

INTERPRETATION:
- replace_quota=0 means no token-set edit, only optional residual.
- replace_quota=1 or 2 means trust-region token-set editing over CDPruner-selected tokens.
- residual_scale > 0 means weak embedding residual refinement.
- This is token pruning / token-set editing, not token merge unless residual_scale > 0 is used.

SEARCH GUIDANCE:
- Start broad if history only contains the clean baseline.
- Prefer q1/q2 trust-region token-set editing.
- Include several q2_ref30 candidates.
- Include several q0/q1 diagnostic candidates early.
- Try both anchor-only policies and weak residual policies.
- For q2/q3 with residual, residual_scale should be weak, usually 0.0005 to 0.003.
- Avoid aggressive q4/q8 exchange.
- Avoid too many near-duplicates.
- If previous rounds already found strong q2 candidates, exploit around them while keeping some exploration.

ROUND:
- round name = {round_name}
- generate {num_candidates} candidates.

OUTPUT RULES:
- Output exactly JSONL.
- One JSON object per line.
- No markdown.
- No explanations.
""".strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", required=True)
    parser.add_argument("--history", required=True)
    parser.add_argument("--num", type=int, default=5)
    parser.add_argument("--out-jsonl", required=True)
    args = parser.parse_args()

    history = read_text(Path(args.history))
    if not history:
        raise RuntimeError(f"history file is empty or missing: {args.history}")

    prompt = build_prompt(history, args.round, args.num)

    messages = [
        {"role": "system", "content": "You are a careful NAS planner. Output valid JSONL only."},
        {"role": "user", "content": prompt},
    ]

    raw = call_openai_compatible(messages)
    rows = extract_jsonl(raw)

    valid = []
    seen = set()

    for x in rows:
        try:
            y = validate_candidate(x)
        except Exception as e:
            print(f"[skip invalid] {e}", file=sys.stderr)
            continue

        if y["name"] in seen:
            print(f"[skip duplicate] {y['name']}", file=sys.stderr)
            continue

        seen.add(y["name"])
        valid.append(y)

    if len(valid) < max(3, args.num // 2):
        print("Raw LLM output:", raw, file=sys.stderr)
        raise RuntimeError(f"Too few valid candidates: {len(valid)}")

    out = Path(args.out_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w") as f:
        for y in valid[:args.num]:
            f.write(json.dumps(y, ensure_ascii=False) + "\n")

    out.with_suffix(".raw.txt").write_text(raw)

    print(f"wrote {out}")
    print(f"num_valid={len(valid[:args.num])}")


if __name__ == "__main__":
    main()
