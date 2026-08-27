from __future__ import annotations

import copy
import csv
import hashlib
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]

SOURCE_ANCHOR = (
    ROOT
    / "configs"
    / "v18_llm_nas"
    / "fresh5_gpu0_cand5_r04"
    / "anchors"
    / (
        "anchor_fresh5_gpu0_cand5_r04_"
        "q2_ref30_dl019_rkw022_"
        "cqp102_rs0002.yaml"
    )
)

SOURCE_MERGE = (
    ROOT
    / "openevolve"
    / "policies"
    / "v18_llm_nas"
    / "fresh5_gpu0_cand5_r04"
    / "merge"
    / (
        "merge_fresh5_gpu0_cand5_r04_"
        "q2_ref30_dl019_rkw022_"
        "cqp102_rs0002.yaml"
    )
)

SETTINGS = [
    (0, 32),
    (1, 31),
    (2, 30),
    (4, 28),
    (8, 24),
]

QUOTA_KEYS = {
    "replace_quota",
    "exchange_quota",
    "max_replacements",
    "max_exchange",
}

MIN_REFERENCE_KEYS = {
    "min_reference_keep",
    "min_ref_keep",
}

EXPERIMENT_ID = (
    "exchange_quota_k32_"
    + datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )
)

SPEC_ROOT = (
    ROOT
    / "openevolve"
    / "policies"
    / "ablation"
    / EXPERIMENT_ID
)

ANCHOR_ROOT = (
    ROOT
    / "configs"
    / "ablation"
    / EXPERIMENT_ID
    / "anchors"
)

MERGE_ROOT = (
    SPEC_ROOT
    / "merge"
)

MANIFEST = (
    SPEC_ROOT
    / "manifest.tsv"
)

LATEST = (
    ROOT
    / "openevolve"
    / "runs"
    / "LATEST_EXCHANGE_QUOTA_K32_MANIFEST.txt"
)


def sha256(
    path: Path,
) -> str:

    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as handle:

        while True:

            block = handle.read(
                1024 * 1024
            )

            if not block:

                break

            digest.update(
                block
            )

    return digest.hexdigest()


def collect_values(
    value: Any,
    keys: set[str],
    prefix: str = "root",
):

    found = []

    if isinstance(
        value,
        dict,
    ):

        for key, child in value.items():

            path = (
                f"{prefix}.{key}"
            )

            if (
                key in keys
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
                    (
                        path,
                        child,
                    )
                )

            found.extend(
                collect_values(
                    child,
                    keys,
                    path,
                )
            )

    elif isinstance(
        value,
        list,
    ):

        for index, child in enumerate(
            value
        ):

            found.extend(
                collect_values(
                    child,
                    keys,
                    (
                        f"{prefix}"
                        f"[{index}]"
                    ),
                )
            )

    return found


def patch_existing_fields(
    value: Any,
    quota: int,
    min_reference_keep: int,
    prefix: str = "root",
):

    quota_hits = []

    min_reference_hits = []

    if isinstance(
        value,
        dict,
    ):

        for key, child in list(
            value.items()
        ):

            path = (
                f"{prefix}.{key}"
            )

            if (
                key in QUOTA_KEYS
                and
                isinstance(
                    child,
                    (
                        int,
                        float,
                    ),
                )
            ):

                value[
                    key
                ] = int(
                    quota
                )

                quota_hits.append(
                    (
                        path,
                        child,
                        quota,
                    )
                )

            elif (
                key
                in MIN_REFERENCE_KEYS
                and
                isinstance(
                    child,
                    (
                        int,
                        float,
                    ),
                )
            ):

                value[
                    key
                ] = int(
                    min_reference_keep
                )

                min_reference_hits.append(
                    (
                        path,
                        child,
                        min_reference_keep,
                    )
                )

            else:

                child_quota, child_ref = (
                    patch_existing_fields(
                        child,
                        quota,
                        min_reference_keep,
                        path,
                    )
                )

                quota_hits.extend(
                    child_quota
                )

                min_reference_hits.extend(
                    child_ref
                )

    elif isinstance(
        value,
        list,
    ):

        for index, child in enumerate(
            value
        ):

            child_quota, child_ref = (
                patch_existing_fields(
                    child,
                    quota,
                    min_reference_keep,
                    (
                        f"{prefix}"
                        f"[{index}]"
                    ),
                )
            )

            quota_hits.extend(
                child_quota
            )

            min_reference_hits.extend(
                child_ref
            )

    return (
        quota_hits,
        min_reference_hits,
    )


if not SOURCE_ANCHOR.exists():

    raise FileNotFoundError(
        SOURCE_ANCHOR
    )

if not SOURCE_MERGE.exists():

    raise FileNotFoundError(
        SOURCE_MERGE
    )


source_policy = yaml.safe_load(
    SOURCE_ANCHOR.read_text(
        encoding="utf-8"
    )
)

source_quota = collect_values(
    source_policy,
    QUOTA_KEYS,
)

source_min_reference = collect_values(
    source_policy,
    MIN_REFERENCE_KEYS,
)


if not source_quota:

    raise RuntimeError(
        "No exchange-quota field "
        "was found."
    )

if not source_min_reference:

    raise RuntimeError(
        "No min-reference field "
        "was found."
    )


print(
    "SOURCE QUOTA FIELDS"
)

for path, value in source_quota:

    print(
        path,
        "=",
        value,
    )


print(
    "SOURCE MIN-REFERENCE FIELDS"
)

for path, value in (
    source_min_reference
):

    print(
        path,
        "=",
        value,
    )


ANCHOR_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

MERGE_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

LATEST.parent.mkdir(
    parents=True,
    exist_ok=True,
)


fixed_merge = (
    MERGE_ROOT
    / "merge_fixed_cdpruner_"
    "q2_best_rs0002.yaml"
)

shutil.copy2(
    SOURCE_MERGE,
    fixed_merge,
)


rows = []


for quota, min_reference_keep in (
    SETTINGS
):

    policy = copy.deepcopy(
        source_policy
    )

    quota_hits, min_ref_hits = (
        patch_existing_fields(
            policy,
            quota,
            min_reference_keep,
        )
    )

    if not quota_hits:

        raise RuntimeError(
            f"q={quota}: quota "
            "was not patched."
        )

    if not min_ref_hits:

        raise RuntimeError(
            f"q={quota}: "
            "min_reference_keep "
            "was not patched."
        )

    anchor_path = (
        ANCHOR_ROOT
        / (
            "anchor_cdpruner_"
            f"q{quota}_"
            f"ref{min_reference_keep}_"
            "k32.yaml"
        )
    )

    anchor_path.write_text(
        yaml.safe_dump(
            policy,
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )

    reloaded = yaml.safe_load(
        anchor_path.read_text(
            encoding="utf-8"
        )
    )

    final_quota = collect_values(
        reloaded,
        QUOTA_KEYS,
    )

    final_min_ref = collect_values(
        reloaded,
        MIN_REFERENCE_KEYS,
    )

    quota_values = {
        int(
            value
        )
        for _, value in final_quota
    }

    min_ref_values = {
        int(
            value
        )
        for _, value in final_min_ref
    }

    if quota_values != {
        quota
    }:

        raise RuntimeError(
            (
                f"q={quota}: "
                f"unexpected values "
                f"{quota_values}"
            )
        )

    if min_ref_values != {
        min_reference_keep
    }:

        raise RuntimeError(
            (
                f"q={quota}: "
                f"unexpected min-ref "
                f"{min_ref_values}"
            )
        )

    rows.append(
        {
            "exchange_quota": quota,
            (
                "min_reference_keep"
            ): min_reference_keep,
            "anchor": str(
                anchor_path.relative_to(
                    ROOT
                )
            ),
            "merge": str(
                fixed_merge.relative_to(
                    ROOT
                )
            ),
            "anchor_sha256": sha256(
                anchor_path
            ),
            "merge_sha256": sha256(
                fixed_merge
            ),
        }
    )

    print(
        "[PASS]",
        f"q={quota}",
        (
            "min_reference_keep="
            f"{min_reference_keep}"
        ),
    )


with MANIFEST.open(
    "w",
    encoding="utf-8",
    newline="",
) as handle:

    writer = csv.DictWriter(
        handle,
        fieldnames=[
            "exchange_quota",
            "min_reference_keep",
            "anchor",
            "merge",
            "anchor_sha256",
            "merge_sha256",
        ],
        delimiter="\t",
    )

    writer.writeheader()

    writer.writerows(
        rows
    )


LATEST.write_text(
    str(
        MANIFEST
    )
    + "\n",
    encoding="utf-8",
)


print(
    "[PASS] EXPERIMENT_ID=",
    EXPERIMENT_ID,
)

print(
    "[PASS] MANIFEST=",
    MANIFEST,
)

print(
    "[PASS] FIXED MERGE SHA256=",
    sha256(
        fixed_merge
    ),
)
