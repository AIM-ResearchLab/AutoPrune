#!/usr/bin/env python3
import argparse
import os
import queue
import re
import subprocess
import threading
import time
from pathlib import Path


SCORE_PAT = re.compile(
    r"CDPRUNER_POLICY|V16 anchor policy|Anchor policy|Merge policy|"
    r"EVO_MASK_HASH|UniMergeHybridDelta|UniMergeHybrid|vtn=|Traceback|"
    r"JSONDecodeError|===========|total score:|score:"
)


def read_manifest(path: Path):
    rows = []
    lines = path.read_text(errors="ignore").splitlines()
    for line in lines[1:]:
        if not line.strip():
            continue
        parts = line.split("|")
        if len(parts) < 3:
            continue
        name = parts[0].strip()
        anchor = parts[1].strip()
        merge = parts[2].strip()
        rows.append((name, anchor, merge))
    return rows


def write_score_summary(log_path: Path, out_path: Path, max_lines=320):
    keep = []
    if log_path.exists():
        for i, line in enumerate(log_path.read_text(errors="ignore").splitlines(), 1):
            if SCORE_PAT.search(line):
                keep.append(f"{i}:{line}")
    out_path.write_text("\n".join(keep[-max_lines:]) + ("\n" if keep else ""))


def safe_count_lines(path: Path):
    if not path.exists():
        return 0
    n = 0
    with path.open("r", errors="ignore") as f:
        for _ in f:
            n += 1
    return n


def clean_jsonl_file_for_mme(path: Path):
    import json

    if not path.exists():
        return False, 0, 0

    raw = path.read_text(errors="ignore").splitlines()
    good = []
    bad = 0

    for line in raw:
        s = line.strip()
        if not s:
            bad += 1
            continue
        try:
            json.loads(s)
            good.append(s)
        except Exception:
            bad += 1

    if bad == 0:
        return False, len(good), 0

    backup = path.with_suffix(path.suffix + ".badbak")
    try:
        backup.write_text("\n".join(raw) + ("\n" if raw else ""))
    except Exception:
        pass

    path.write_text("\n".join(good) + ("\n" if good else ""))
    return True, len(good), bad


def try_rescore_mme(exp: str, log_path: Path):
    mme_dir = Path("playground/data/eval/MME")
    ans = mme_dir / "answers" / f"{exp}.jsonl"

    changed, good, bad = clean_jsonl_file_for_mme(ans)

    cmd = (
        f"cd {mme_dir} && "
        f"python convert_answer_to_mme.py --experiment {exp} && "
        f"cd eval_tool && "
        f"python calculation.py --results_dir answers/{exp}"
    )

    with log_path.open("w") as f:
        ret = subprocess.run(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT).returncode

    return ret, changed, good, bad


def run_one(round_name, run_root: Path, name, anchor_policy, merge_policy, gpu):
    out_dir = run_root / name
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    exp = f"v18llm_{round_name}_{name}_gpu{gpu}_{timestamp}"
    mask_hash_log = (out_dir / "mask_hash.jsonl").resolve()

    try:
        mask_hash_log.unlink(missing_ok=True)
    except Exception:
        pass

    status_path = out_dir / "status.txt"
    log_path = out_dir / "mme_full.log"
    summary_path = out_dir / "mme_full_score_summary.txt"

    status_lines = [
        f"ROUND={round_name}",
        f"NAME={name}",
        f"ANCHOR_POLICY={anchor_policy}",
        f"MERGE_POLICY={merge_policy}",
        f"EXP={exp}",
        f"GPU={gpu}",
        f"MASK_HASH_LOG={mask_hash_log}",
        f"START_TIME={time.strftime('%F %T')}",
    ]
    status_path.write_text("\n".join(status_lines) + "\n")

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{Path(__file__).resolve().parents[2]}:{env.get('PYTHONPATH', '')}"
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["EVO_GPU"] = str(gpu)
    env["EVO_TOKEN"] = "32"
    env["EVO_FORCE_FIXED_TOKENS"] = "32"
    env["EVO_UNIMERGE_DEBUG"] = env.get("EVO_UNIMERGE_DEBUG", "0")
    env["EVO_UNIMERGE_OUTPUT_MODE"] = env.get("EVO_UNIMERGE_OUTPUT_MODE", "inplace_full")
    env["EVO_CDPRUNER_MASK_HASH_LOG"] = "1"
    env["EVO_MASK_HASH_LOG"] = str(mask_hash_log)
    env["EVO_MASK_HASH_LOG_N"] = env.get("EVO_MASK_HASH_LOG_N", "100000000")
    env["EVO_MASK_HASH_LOG_BATCH"] = env.get("EVO_MASK_HASH_LOG_BATCH", "1")
    env["EVO_CDPRUNER_MASK_HASH_DEBUG"] = env.get("EVO_CDPRUNER_MASK_HASH_DEBUG", "0")

    cmd = [
        "bash",
        "openevolve/run_autoprune_policy.sh",
        anchor_policy,
        merge_policy,
        "bash",
        "scripts/v1_5/eval/mme.sh",
        "32",
        exp,
    ]

    print(f"===== RUN {round_name}/{name} on GPU {gpu} =====", flush=True)
    print(f"ANCHOR={anchor_policy}", flush=True)
    print(f"MERGE={merge_policy}", flush=True)
    print(f"MASK_HASH_LOG={mask_hash_log}", flush=True)

    with log_path.open("w") as f:
        ret = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env).returncode

    with status_path.open("a") as f:
        f.write(f"EXIT_CODE={ret}\n")
        f.write(f"END_TIME={time.strftime('%F %T')}\n")
        f.write(f"MASK_HASH_ROWS={safe_count_lines(mask_hash_log)}\n")

    write_score_summary(log_path, summary_path)

    if ret == 0:
        print(f"===== DONE_OK {round_name}/{name} on GPU {gpu} =====", flush=True)
    else:
        print(f"===== DONE_FAIL {round_name}/{name} on GPU {gpu}, ret={ret} =====", flush=True)

    return ret


def worker(gpu, q: queue.Queue, round_name, run_root):
    while True:
        try:
            item = q.get_nowait()
        except queue.Empty:
            return

        name, anchor, merge = item
        try:
            run_one(round_name, run_root, name, anchor, merge, gpu)
        except Exception as e:
            print(f"===== WORKER_ERROR {round_name}/{name} on GPU {gpu}: {e} =====", flush=True)
        finally:
            q.task_done()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", required=True)
    parser.add_argument("--gpus", required=True, help="comma-separated GPU ids, e.g. 0,1,7")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]

    # Always run inside the current isolated repository.
    os.chdir(repo_root)
    manifest = repo_root / "openevolve" / "policies" / "v18_llm_nas" / args.round / "manifest.psv"
    run_root = Path(f"openevolve/runs/v18_llm_nas_{args.round}_gpu0_cand5_seq")
    run_root.mkdir(parents=True, exist_ok=True)

    if not manifest.exists():
        raise FileNotFoundError(manifest)

    gpus = [x.strip() for x in args.gpus.split(",") if x.strip()]
    rows = read_manifest(manifest)

    print("===== dynamic GPU pool runner =====", flush=True)
    print(f"ROUND={args.round}", flush=True)
    print(f"MANIFEST={manifest}", flush=True)
    print(f"RUN_ROOT={run_root}", flush=True)
    print(f"GPUS={gpus}", flush=True)
    print(f"NUM_TASKS={len(rows)}", flush=True)

    if not gpus:
        raise RuntimeError("No GPUs provided")

    if args.dry_run:
        for i, row in enumerate(rows):
            print(f"task[{i}] name={row[0]} anchor={row[1]} merge={row[2]}")
        return

    q = queue.Queue()
    for row in rows:
        q.put(row)

    threads = []
    for gpu in gpus:
        t = threading.Thread(
            target=worker,
            args=(gpu, q, args.round, run_root),
            daemon=False,
        )
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    print(f"All jobs for {args.round} finished.", flush=True)


if __name__ == "__main__":
    main()
