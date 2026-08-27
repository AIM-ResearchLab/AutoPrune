from pathlib import Path
import re
import csv

ROOT = Path(".")
summary_files = sorted(ROOT.glob("openevolve/runs/v18_llm_nas_fresh5_gpu0_cand5_r*_gpu0_cand5_seq/summary.md"))

rows = []

def clean_cell(x):
    x = x.strip()
    if x.startswith("`") and x.endswith("`"):
        x = x[1:-1]
    return x.strip()

for sf in summary_files:
    round_name = re.search(r"v18_llm_nas_(fresh5_gpu0_cand5_r\d+)_gpu0_cand5_seq", str(sf))
    round_name = round_name.group(1) if round_name else sf.parent.name

    for line in sf.read_text(errors="ignore").splitlines():
        if not line.startswith("| fresh5_gpu0_cand5_r"):
            continue

        cells = [clean_cell(c) for c in line.strip().strip("|").split("|")]
        if len(cells) < 18:
            continue

        # columns:
        # 0 Name, 1 Exit, 2 VTN, 3 Perception, 4 Cognition, 5 Raw,
        # 6 ΔP clean, 7 ΔC clean, 8 ΔP known-best, 9 NAS score,
        # 10 MaskRows, 11 SelUnique, 12 SelHash, ...
        name = cells[0]

        def to_float(x):
            x = x.replace("+", "").strip()
            if x in {"", "NA", "nan", "None"}:
                return None
            try:
                return float(x)
            except Exception:
                return None

        def to_int(x):
            if x in {"", "NA"}:
                return None
            try:
                return int(float(x))
            except Exception:
                return None

        item = {
            "round": round_name,
            "name": name,
            "exit": cells[1],
            "vtn": cells[2],
            "perception": to_float(cells[3]),
            "cognition": to_float(cells[4]),
            "raw": to_float(cells[5]),
            "delta_p_clean": to_float(cells[6]),
            "delta_c_clean": to_float(cells[7]),
            "delta_p_known_best": to_float(cells[8]),
            "nas_score": to_float(cells[9]),
            "mask_rows": to_int(cells[10]),
            "sel_unique": to_int(cells[11]),
            "sel_hash": cells[12],
            "note": cells[15],
            "anchor": cells[16],
            "merge": cells[17],
            "summary_file": str(sf),
        }
        rows.append(item)

rows_sorted = sorted(
    rows,
    key=lambda x: (
        x["raw"] if x["raw"] is not None else -1e9,
        x["nas_score"] if x["nas_score"] is not None else -1e9,
        x["perception"] if x["perception"] is not None else -1e9,
    ),
    reverse=True,
)

out_dir = ROOT / "openevolve/runs/fresh5_gpu0_cand5_10rounds_analysis"
out_dir.mkdir(parents=True, exist_ok=True)

tsv_path = out_dir / "all_candidates.tsv"
with tsv_path.open("w", newline="") as f:
    fieldnames = [
        "rank", "round", "name", "exit", "vtn", "perception", "cognition", "raw",
        "delta_p_clean", "delta_c_clean", "delta_p_known_best", "nas_score",
        "mask_rows", "sel_unique", "sel_hash", "note", "anchor", "merge", "summary_file"
    ]
    w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
    w.writeheader()
    for i, r in enumerate(rows_sorted, 1):
        rr = dict(r)
        rr["rank"] = i
        w.writerow(rr)

top_md = out_dir / "top20.md"
with top_md.open("w") as f:
    f.write("# fresh5 gpu0 cand5 10-round Top20\n\n")
    f.write(f"Total parsed candidates: **{len(rows)}**\n\n")
    f.write("| Rank | Round | Name | Raw | Perception | Cognition | NAS score | SelHash | Note |\n")
    f.write("|---:|---|---|---:|---:|---:|---:|---|---|\n")
    for i, r in enumerate(rows_sorted[:20], 1):
        f.write(
            f"| {i} | {r['round']} | `{r['name']}` | "
            f"{r['raw'] if r['raw'] is not None else ''} | "
            f"{r['perception'] if r['perception'] is not None else ''} | "
            f"{r['cognition'] if r['cognition'] is not None else ''} | "
            f"{r['nas_score'] if r['nas_score'] is not None else ''} | "
            f"`{r['sel_hash']}` | {r['note']} |\n"
        )

best_path = out_dir / "best_policy.sh"
if rows_sorted:
    b = rows_sorted[0]
    with best_path.open("w") as f:
        f.write("# Best policy from fresh5_gpu0_cand5 10-round search\n")
        f.write(f"BEST_NAME='{b['name']}'\n")
        f.write(f"BEST_ROUND='{b['round']}'\n")
        f.write(f"BEST_RAW='{b['raw']}'\n")
        f.write(f"BEST_PERCEPTION='{b['perception']}'\n")
        f.write(f"BEST_COGNITION='{b['cognition']}'\n")
        f.write(f"BEST_NAS_SCORE='{b['nas_score']}'\n")
        f.write(f"BEST_SEL_HASH='{b['sel_hash']}'\n")
        f.write(f"BEST_ANCHOR='{b['anchor']}'\n")
        f.write(f"BEST_MERGE='{b['merge']}'\n")

hash_counts = {}
for r in rows:
    h = r["sel_hash"]
    hash_counts[h] = hash_counts.get(h, 0) + 1

hash_path = out_dir / "selhash_counts.tsv"
with hash_path.open("w") as f:
    f.write("sel_hash\tcount\n")
    for h, c in sorted(hash_counts.items(), key=lambda x: x[1], reverse=True):
        f.write(f"{h}\t{c}\n")

print("Parsed candidates:", len(rows))
print("Summary files:", len(summary_files))
print("Output dir:", out_dir)
print("All candidates:", tsv_path)
print("Top20:", top_md)
print("Best policy:", best_path)
print("SelHash counts:", hash_path)

if rows_sorted:
    b = rows_sorted[0]
    print()
    print("===== BEST =====")
    print("rank:", 1)
    print("round:", b["round"])
    print("name:", b["name"])
    print("raw:", b["raw"])
    print("perception:", b["perception"])
    print("cognition:", b["cognition"])
    print("nas_score:", b["nas_score"])
    print("sel_hash:", b["sel_hash"])
    print("anchor:", b["anchor"])
    print("merge:", b["merge"])

    fresh4_best = 1713.46
    if b["raw"] is not None:
        print()
        print("Compared with fresh4_targeted best Raw=1713.46:")
        print("delta_raw:", round(b["raw"] - fresh4_best, 4))
