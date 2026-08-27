from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

RUN_ROOT = Path(
    (
        ROOT
        / "openevolve"
        / "runs"
        / "LATEST_EXCHANGE_QUOTA_K32_RUN.txt"
    )
    .read_text(
        encoding="utf-8"
    )
    .strip()
)

MANIFEST = (
    RUN_ROOT
    / "manifest.tsv"
)

RESULTS = (
    RUN_ROOT
    / "results"
)

RESULTS.mkdir(
    parents=True,
    exist_ok=True,
)


def recursive_numeric_values(
    value: Any,
    target_keys: set[str],
):

    found = []

    if isinstance(
        value,
        dict,
    ):

        for key, child in value.items():

            if (
                key in target_keys
                and
                isinstance(
                    child,
                    (
                        int,
                        float,
                    ),
                )
            ):

                found.append(
                    float(
                        child
                    )
                )

            found.extend(
                recursive_numeric_values(
                    child,
                    target_keys,
                )
            )

    elif isinstance(
        value,
        list,
    ):

        for child in value:

            found.extend(
                recursive_numeric_values(
                    child,
                    target_keys,
                )
            )

    return found


with MANIFEST.open(
    encoding="utf-8"
) as handle:

    specs = list(
        csv.DictReader(
            handle,
            delimiter="\t",
        )
    )


rows = []


for spec in specs:

    quota = int(
        spec[
            "exchange_quota"
        ]
    )

    min_ref = int(
        spec[
            "min_reference_keep"
        ]
    )

    out = (
        RUN_ROOT
        / f"q{quota}_ref{min_ref}"
    )

    log_path = (
        out
        / "mme_full.log"
    )

    exit_path = (
        out
        / "exit_code.txt"
    )

    mask_path = (
        out
        / "mask_hash.jsonl"
    )


    exit_code = (
        int(
            exit_path
            .read_text()
            .strip()
        )
        if exit_path.exists()
        else -999
    )


    text = (
        log_path.read_text(
            encoding="utf-8",
            errors="ignore",
        )
        if log_path.exists()
        else ""
    )


    score_values = [
        float(
            value
        )
        for value in re.findall(
            r"total score:\s*"
            r"([0-9.]+)",
            text,
        )
    ]


    perception = (
        score_values[-2]
        if len(
            score_values
        ) >= 2
        else None
    )

    cognition = (
        score_values[-1]
        if len(
            score_values
        ) >= 2
        else None
    )

    mme_total = (
        perception
        + cognition
        if (
            perception
            is not None
            and
            cognition
            is not None
        )
        else None
    )


    mask_rows = 0

    bad_vtn = 0

    selected_hashes = set()

    changed_values = []


    if mask_path.exists():

        with mask_path.open(
            encoding="utf-8",
            errors="ignore",
        ) as handle:

            for line in handle:

                line = line.strip()

                if not line:

                    continue

                try:

                    record = json.loads(
                        line
                    )

                except Exception:

                    continue


                mask_rows += 1


                selected_tokens = (
                    record.get(
                        "selected_tokens"
                    )
                )

                if (
                    selected_tokens
                    is not None
                    and
                    int(
                        selected_tokens
                    )
                    != 32
                ):

                    bad_vtn += 1


                selected_hash = (
                    record.get(
                        "selected_mask_hash"
                    )
                    or
                    record.get(
                        "selected_hash"
                    )
                )

                if selected_hash:

                    selected_hashes.add(
                        str(
                            selected_hash
                        )
                    )


                changed_values.extend(
                    recursive_numeric_values(
                        record,
                        {
                            "changed_count",
                            "num_changed",
                            "exchange_count",
                            "changed_tokens_avg",
                        },
                    )
                )


    log_changed = [
        float(
            value
        )
        for value in re.findall(
            r"changed_tokens_avg"
            r"\s*[=:]\s*"
            r"([0-9.]+)",
            text,
        )
    ]


    actual_changed_avg = None


    if changed_values:

        actual_changed_avg = mean(
            changed_values
        )

    elif log_changed:

        actual_changed_avg = (
            log_changed[-1]
        )


    rows.append(
        {
            "exchange_quota": quota,
            "min_reference_keep": min_ref,
            "perception": perception,
            "cognition": cognition,
            "mme_total": mme_total,
            "mme_total_div20": (
                mme_total / 20.0
                if mme_total
                is not None
                else None
            ),
            "actual_changed_avg": (
                actual_changed_avg
            ),
            "exit": exit_code,
            "mask_rows": mask_rows,
            "bad_vtn_rows": bad_vtn,
            (
                "unique_selected_hashes"
            ): len(
                selected_hashes
            ),
            "anchor": spec[
                "anchor"
            ],
            "merge": spec[
                "merge"
            ],
        }
    )


rows.sort(
    key=lambda row: (
        row[
            "exchange_quota"
        ]
    )
)


q0 = next(
    (
        row
        for row in rows
        if row[
            "exchange_quota"
        ]
        == 0
    ),
    None,
)


for row in rows:

    row[
        "delta_perception_vs_q0"
    ] = (
        row[
            "perception"
        ]
        - q0[
            "perception"
        ]
        if (
            q0
            and
            q0[
                "perception"
            ]
            is not None
            and
            row[
                "perception"
            ]
            is not None
        )
        else None
    )

    row[
        "delta_total_vs_q0"
    ] = (
        row[
            "mme_total"
        ]
        - q0[
            "mme_total"
        ]
        if (
            q0
            and
            q0[
                "mme_total"
            ]
            is not None
            and
            row[
                "mme_total"
            ]
            is not None
        )
        else None
    )


fields = [
    "exchange_quota",
    "min_reference_keep",
    "perception",
    "cognition",
    "mme_total",
    "mme_total_div20",
    "delta_perception_vs_q0",
    "delta_total_vs_q0",
    "actual_changed_avg",
    "exit",
    "mask_rows",
    "bad_vtn_rows",
    "unique_selected_hashes",
    "anchor",
    "merge",
]


tsv_path = (
    RESULTS
    / "exchange_quota_mme.tsv"
)


with tsv_path.open(
    "w",
    encoding="utf-8",
    newline="",
) as handle:

    writer = csv.DictWriter(
        handle,
        fieldnames=fields,
        delimiter="\t",
    )

    writer.writeheader()

    writer.writerows(
        rows
    )


md = []

md.append(
    "# Exchange-Quota Ablation at K=32"
)

md.append("")

md.append(
    "| q | Min reference keep | "
    "MME Perception | MME Cognition | "
    "MME Total | Δ Total vs q=0 |"
)

md.append(
    "|---:|---:|---:|---:|---:|---:|"
)


for row in rows:

    def fmt(
        value,
        signed=False,
    ):

        if value is None:

            return "N/A"

        if signed:

            return (
                f"{value:+.2f}"
            )

        return (
            f"{value:.2f}"
        )


    md.append(
        "| "
        f"{row['exchange_quota']} | "
        f"{row['min_reference_keep']} | "
        f"{fmt(row['perception'])} | "
        f"{fmt(row['cognition'])} | "
        f"{fmt(row['mme_total'])} | "
        f"{fmt(row['delta_total_vs_q0'], True)} |"
    )


md_path = (
    RESULTS
    / "exchange_quota_mme.md"
)

md_path.write_text(
    "\n".join(
        md
    )
    + "\n",
    encoding="utf-8",
)


print(
    "[PASS]",
    tsv_path,
)

print(
    "[PASS]",
    md_path,
)

print("")


print(
    f"{'q':>3}"
    f"{'Ref':>6}"
    f"{'P':>12}"
    f"{'C':>11}"
    f"{'Total':>12}"
    f"{'Δ vs q0':>12}"
)


print(
    "-" * 56
)


for row in rows:

    print(
        f"{row['exchange_quota']:>3}"
        f"{row['min_reference_keep']:>6}"
        f"{row['perception']:>12.2f}"
        f"{row['cognition']:>11.2f}"
        f"{row['mme_total']:>12.2f}"
        f"{row['delta_total_vs_q0']:>+12.2f}"
    )


q2 = next(
    row
    for row in rows
    if row[
        "exchange_quota"
    ]
    == 2
)


print("")

print(
    "Q2 reproducibility check:"
)

print(
    "Perception delta from "
    "1413.46 =",
    (
        q2[
            "perception"
        ]
        - 1413.46
    ),
)

print(
    "Cognition delta from "
    "300.00 =",
    (
        q2[
            "cognition"
        ]
        - 300.00
    ),
)

print(
    "Total delta from "
    "1713.46 =",
    (
        q2[
            "mme_total"
        ]
        - 1713.46
    ),
)
