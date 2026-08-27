#!/usr/bin/env python3
import argparse
import json
import re
from collections import Counter
from pathlib import Path


CLEAN_P = 1382.8248299319728
CLEAN_C = 304.2857142857143
KNOWN_BEST_P = 1406.8248299319728


def read_mask_stats(path: Path):
    if not path.exists():
        return {
            "mask_rows": 0,
            "sel_unique": "",
            "sel_top_hash": "",
            "ref_top_hash": "",
            "overlap_avg": "",
            "changed_avg": "",
        }

    rows = []
    for line in path.read_text(errors="ignore").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            pass

    if not rows:
        return {
            "mask_rows": 0,
            "sel_unique": "",
            "sel_top_hash": "",
            "ref_top_hash": "",
            "overlap_avg": "",
            "changed_avg": "",
        }

    sel_hashes = [r.get("selected_mask_hash", "") for r in rows if r.get("selected_mask_hash")]
    ref_hashes = [r.get("reference_mask_hash", "") for r in rows if r.get("reference_mask_hash")]

    sel_counter = Counter(sel_hashes)
    ref_counter = Counter(ref_hashes)

    overlaps = [float(r["overlap_with_reference"]) for r in rows if "overlap_with_reference" in r]
    changed = [float(r["changed_tokens"]) for r in rows if "changed_tokens" in r]

    sel_top_hash = sel_counter.most_common(1)[0][0] if sel_counter else ""
    ref_top_hash = ref_counter.most_common(1)[0][0] if ref_counter else ""

    overlap_avg = sum(overlaps) / len(overlaps) if overlaps else ""
    changed_avg = sum(changed) / len(changed) if changed else ""

    return {
        "mask_rows": len(rows),
        "sel_unique": len(sel_counter) if sel_counter else 0,
        "sel_top_hash": sel_top_hash,
        "ref_top_hash": ref_top_hash,
        "overlap_avg": overlap_avg,
        "changed_avg": changed_avg,
    }


def fmt_float(x, nd=2):
    if x == "" or x is None:
        return ""
    return f"{float(x):.{nd}f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", required=True)
    args = parser.parse_args()

    root = Path(f"openevolve/runs/v18_llm_nas_{args.round}_gpu0_cand5_seq")
    rows = []

    for d in sorted(root.iterdir()) if root.exists() else []:
        if not d.is_dir():
            continue

        status = d / "status.txt"
        summary = d / "mme_full_score_summary.txt"
        mask_stats = read_mask_stats(d / "mask_hash.jsonl")

        info = {
            "round": args.round,
            "name": d.name,
            "anchor": "",
            "merge": "",
            "exit": "NA",
        }

        if status.exists():
            st = status.read_text(errors="ignore")
            for key, out_key in [
                ("NAME", "name"),
                ("ANCHOR_POLICY", "anchor"),
                ("MERGE_POLICY", "merge"),
                ("EXIT_CODE", "exit"),
            ]:
                m = re.findall(rf"{key}=(.*)", st)
                if m:
                    info[out_key] = m[-1]

        if not summary.exists():
            rows.append((info, None, None, None, "", "no_summary", None, mask_stats))
            continue

        text = summary.read_text(errors="ignore")
        totals = [float(x) for x in re.findall(r"total score:\s*([0-9.]+)", text)]
        vtns = re.findall(r"vtn=(\d+)", text)
        vtn = vtns[-1] if vtns else ""

        if len(totals) >= 2:
            p, c = totals[0], totals[1]
            raw = p + c

            score = (p - CLEAN_P) + 0.20 * (c - CLEAN_C)
            if vtn != "32":
                score -= 1000.0

            rows.append((info, p, c, raw, vtn, f"score_vs_clean={score:+.2f}", score, mask_stats))
        else:
            note = "json_error" if "JSONDecodeError" in text else "no_score"
            rows.append((info, None, None, None, vtn, note, None, mask_stats))

    print("| Name | Exit | VTN | Perception | Cognition | Raw | ΔP clean | ΔC clean | ΔP known-best | NAS score | MaskRows | SelUnique | SelHash | RefHash | OverlapAvg | ChangedAvg | Note | Anchor | Merge |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---|---|---|")

    def sort_key(row):
        info, p, c, raw, vtn, note, score, mask_stats = row
        if p is None:
            return (999999, info["name"])
        return (-score, info["name"])

    for info, p, c, raw, vtn, note, score, ms in sorted(rows, key=sort_key):
        mask_cols = (
            f"{ms['mask_rows']} | {ms['sel_unique']} | {ms['sel_top_hash']} | {ms['ref_top_hash']} | "
            f"{fmt_float(ms['overlap_avg'], 4)} | {fmt_float(ms['changed_avg'], 2)}"
        )

        if p is None:
            print(
                f"| {info['name']} | {info['exit']} | {vtn} |  |  |  |  |  |  |  | "
                f"{mask_cols} | {note} | `{info['anchor']}` | `{info['merge']}` |"
            )
        else:
            print(
                f"| {info['name']} | {info['exit']} | {vtn} | "
                f"{p:.2f} | {c:.2f} | {raw:.2f} | "
                f"{p-CLEAN_P:+.2f} | {c-CLEAN_C:+.2f} | {p-KNOWN_BEST_P:+.2f} | "
                f"{score:+.2f} | {mask_cols} | {note} | `{info['anchor']}` | `{info['merge']}` |"
            )


if __name__ == "__main__":
    main()
