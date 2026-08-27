from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import json
import os
import re
import shutil
import subprocess
import sys

BASE = Path(__file__).resolve().parents[2]
ABL_DIR = BASE / "configs/ablation/exchange_quota_k32_20260714_004819"
MERGE_POLICY = BASE / "openevolve/policies/ablation/exchange_quota_k32_20260714_004819/merge/merge_fixed_cdpruner_q2_best_rs0002.yaml"

RUN_ROOT = BASE / "openevolve/runs" / f"exchange_quota_k32_q16_q32_{datetime.now().strftime('%Y%m%d_%H%M%S')}_fullmme"
RUN_ROOT.mkdir(parents=True, exist_ok=True)
(BASE / "openevolve/runs/LATEST_EXCHANGE_QUOTA_Q16_Q32_FULLMME.txt").write_text(str(RUN_ROOT), encoding="utf-8")

MODEL_PATH = os.environ.get("AUTOPRUNE_MODEL_PATH", str(Path(os.environ.get("CKPT_DIR", "/path/to/hf_models")) / "llava-v1.5-7b"))
QUESTION_FILE = str(BASE / "playground/data/eval/MME/llava_mme.jsonl")
IMAGE_FOLDER = str(BASE / "playground/data/eval/MME/MME_Benchmark_release_version")
ANSWER_DIR = BASE / "playground/data/eval/MME/answers/llava_mme/llava-v1.5-7b"
EVAL_ROOT = BASE / "playground/data/eval/MME"
EVAL_TOOL = EVAL_ROOT / "eval_tool"

print("[INFO] RUN_ROOT=", RUN_ROOT, flush=True)
print("[INFO] ABL_DIR=", ABL_DIR, flush=True)
print("[INFO] MERGE_POLICY=", MERGE_POLICY, flush=True)

def run_cmd(cmd, cwd, env, log_path, timeout=None):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        try:
            p = subprocess.run(cmd, cwd=str(cwd), env=env, stdout=f, stderr=subprocess.STDOUT, timeout=timeout)
            return p.returncode
        except subprocess.TimeoutExpired:
            f.write("\n[TIMEOUT]\n")
            return 124
        except Exception as e:
            f.write("\n[EXCEPTION] " + repr(e) + "\n")
            return 125

def check_jsonl(path, out_log):
    try:
        n = 0
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                json.loads(line)
                n += 1
        out_log.write_text(f"JSONL_OK {path} lines {n}\n", encoding="utf-8")
        return n == 2374
    except Exception as e:
        out_log.write_text(repr(e) + "\n", encoding="utf-8")
        return False

def parse_score(path):
    text = path.read_text(errors="ignore") if path.exists() else ""
    vals = [float(x) for x in re.findall(r"total score:\s*([0-9.]+)", text)]
    perception = vals[0] if len(vals) >= 1 else None
    cognition = vals[1] if len(vals) >= 2 else None
    total = perception + cognition if perception is not None and cognition is not None else None
    return perception, cognition, total

def run_one(q, gpu):
    ref = 32 - q
    tag = f"q{q}_ref{ref}"
    out_dir = RUN_ROOT / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    anchor = ABL_DIR / "anchors" / f"anchor_cdpruner_q{q}_ref{ref}_k32.yaml"
    exp = f"exchange_quota_k32_{tag}_fullmme"
    answer_file = ANSWER_DIR / f"{exp}.jsonl"
    converted_dir = EVAL_TOOL / "answers/llava_mme/llava-v1.5-7b" / exp

    if not anchor.exists():
        (out_dir / "status.txt").write_text("STATUS=FAILED_MISSING_ANCHOR\n", encoding="utf-8")
        (out_dir / "mme_full.log").write_text(f"[FAIL] missing anchor: {anchor}\n", encoding="utf-8")
        return

    if not MERGE_POLICY.exists():
        (out_dir / "status.txt").write_text("STATUS=FAILED_MISSING_MERGE\n", encoding="utf-8")
        (out_dir / "mme_full.log").write_text(f"[FAIL] missing merge policy: {MERGE_POLICY}\n", encoding="utf-8")
        return

    if answer_file.exists():
        answer_file.unlink()
    if converted_dir.exists():
        shutil.rmtree(converted_dir)

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{BASE}:{env.get('PYTHONPATH', '')}"
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["EVO_GPU"] = str(gpu)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["EVO_GPU"] = str(gpu)
    env["EVO_TOKEN"] = "32"
    env["EVO_FORCE_FIXED_TOKENS"] = "32"

    env["CDPRUNER_POLICY"] = str(anchor)
    env["EVO_UNIMERGE_ENABLE"] = "1"
    env["EVO_UNIMERGE_MODE"] = "hybrid_v16_anchor"
    env["EVO_UNIMERGE_OUTPUT_MODE"] = "inplace_full"
    env["EVO_UNIMERGE_POLICY_PATH"] = str(MERGE_POLICY)
    env["EVO_UNIMERGE_DEBUG"] = "1"
    env["EVO_UNIMERGE_DEBUG_N"] = "2"
    env["EVO_UNIMERGE_MERGE_POLICY"] = str(MERGE_POLICY)
    env["EVO_MERGE_POLICY"] = str(MERGE_POLICY)
    env["MERGE_POLICY"] = str(MERGE_POLICY)

    env["EVO_CDPRUNER_MASK_HASH_LOG"] = "1"
    env["EVO_CDPRUNER_MASK_HASH_DEBUG"] = "0"
    env["EVO_MASK_HASH_LOG"] = str(out_dir / "mask_hash.jsonl")
    env["EVO_MASK_HASH_LOG_N"] = "50"
    env["EVO_MASK_HASH_LOG_BATCH"] = "1"

    env["HF_ENABLE_PARALLEL_LOADING"] = "true"
    env["HF_PARALLEL_LOADING_WORKERS"] = "8"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["HF_HUB_OFFLINE"] = "1"

    run_info = []
    run_info.append(f"Anchor policy: {anchor}")
    run_info.append(f"Merge policy: {MERGE_POLICY}")
    run_info.append(f"Q={q}")
    run_info.append(f"REF={ref}")
    run_info.append(f"EVO_GPU={gpu}")
    run_info.append("EVO_TOKEN=32")
    run_info.append(f"EXP={exp}")
    run_info.append(f"CDPRUNER_POLICY={anchor}")
    run_info.append("EVO_UNIMERGE_MODE=hybrid_v16_anchor")
    run_info.append("EVO_UNIMERGE_ENABLE=1")
    run_info.append("EVO_UNIMERGE_OUTPUT_MODE=inplace_full")
    run_info.append(f"EVO_UNIMERGE_POLICY_PATH={MERGE_POLICY}")
    (out_dir / "run_info.txt").write_text("\n".join(run_info) + "\n", encoding="utf-8")

    print(f"[START] {tag} gpu={gpu}", flush=True)

    infer_cmd = [
        sys.executable,
        "-m",
        "llava.eval.model_vqa_loader",
        "--model-path",
        MODEL_PATH,
        "--question-file",
        QUESTION_FILE,
        "--image-folder",
        IMAGE_FOLDER,
        "--answers-file",
        str(answer_file),
        "--visual_token_num",
        "32",
        "--temperature",
        "0",
        "--conv-mode",
        "vicuna_v1",
    ]

    rc = run_cmd(infer_cmd, BASE, env, out_dir / "mme_full.log", timeout=21600)
    (out_dir / "exit_code.txt").write_text(str(rc), encoding="utf-8")

    if rc != 0:
        (out_dir / "status.txt").write_text("STATUS=FAILED_INFER\n", encoding="utf-8")
        print(f"[END] {tag} FAILED_INFER rc={rc}", flush=True)
        return

    ok_json = check_jsonl(answer_file, out_dir / "json_check.log")
    if not ok_json:
        (out_dir / "status.txt").write_text("STATUS=FAILED_JSON\n", encoding="utf-8")
        print(f"[END] {tag} FAILED_JSON", flush=True)
        return

    convert_cmd = [
        sys.executable,
        "convert_answer_to_mme.py",
        "--experiment",
        f"llava_mme/llava-v1.5-7b/{exp}",
    ]

    rc = run_cmd(convert_cmd, EVAL_ROOT, env, out_dir / "convert.log", timeout=3600)
    (out_dir / "convert_exit_code.txt").write_text(str(rc), encoding="utf-8")

    if rc != 0:
        (out_dir / "status.txt").write_text("STATUS=FAILED_CONVERT\n", encoding="utf-8")
        print(f"[END] {tag} FAILED_CONVERT rc={rc}", flush=True)
        return

    score_cmd = [
        sys.executable,
        "calculation.py",
        "--results_dir",
        f"answers/llava_mme/llava-v1.5-7b/{exp}",
    ]

    rc = run_cmd(score_cmd, EVAL_TOOL, env, out_dir / "score.log", timeout=3600)
    (out_dir / "score_exit_code.txt").write_text(str(rc), encoding="utf-8")

    if rc != 0:
        (out_dir / "status.txt").write_text("STATUS=FAILED_SCORE\n", encoding="utf-8")
        print(f"[END] {tag} FAILED_SCORE rc={rc}", flush=True)
        return

    p, c, t = parse_score(out_dir / "score.log")
    (out_dir / "status.txt").write_text("STATUS=SUCCESS\n", encoding="utf-8")
    print(f"[END] {tag} SUCCESS perception={p} cognition={c} total={t}", flush=True)

def summarize():
    result_dir = RUN_ROOT / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    out = result_dir / "exchange_quota_q16_q32_mme.tsv"

    with out.open("w", encoding="utf-8") as f:
        f.write("quota\tfixed_reference_tokens\tmodified_tokens\tstatus\tperception\tcognition\ttotal\n")
        for q in [16, 32]:
            ref = 32 - q
            d = RUN_ROOT / f"q{q}_ref{ref}"
            status = (d / "status.txt").read_text(errors="ignore").strip() if (d / "status.txt").exists() else "NO_STATUS"
            p, c, t = parse_score(d / "score.log")
            f.write(f"{q}\t{ref}\t{q}\t{status}\t")
            f.write("" if p is None else f"{p:.6f}")
            f.write("\t")
            f.write("" if c is None else f"{c:.6f}")
            f.write("\t")
            f.write("" if t is None else f"{t:.6f}")
            f.write("\n")

    print("[PASS] wrote", out, flush=True)
    print(out.read_text(encoding="utf-8"), flush=True)
    (RUN_ROOT / "status.txt").write_text("STATUS=DONE\n", encoding="utf-8")

def main():
    jobs = [(16, 2), (32, 3)]
    with ThreadPoolExecutor(max_workers=2) as ex:
        futs = [ex.submit(run_one, q, gpu) for q, gpu in jobs]
        for fut in as_completed(futs):
            fut.result()
    summarize()
    print("[DONE] RUN_ROOT=", RUN_ROOT, flush=True)

if __name__ == "__main__":
    main()
