from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PipelineNode:
    op: str
    name: str
    inputs: Dict[str, Any] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)
    output: Optional[str] = None


@dataclass
class PipelineProgram:
    name: str
    keep_tokens: int
    pipeline: List[PipelineNode]
    metadata: Dict[str, Any] = field(default_factory=dict)


def is_pipeline_policy(policy: Dict[str, Any]) -> bool:
    return isinstance(policy.get("pipeline"), list) and len(policy.get("pipeline")) > 0


def parse_pipeline_policy(policy: Dict[str, Any]) -> PipelineProgram:
    nodes = []
    for raw in policy.get("pipeline", []):
        nodes.append(
            PipelineNode(
                op=str(raw.get("op", "")),
                name=str(raw.get("name", "")),
                inputs=dict(raw.get("inputs") or {}),
                params=dict(raw.get("params") or {}),
                output=raw.get("output"),
            )
        )

    return PipelineProgram(
        name=str(policy.get("name", "v9_pipeline_policy")),
        keep_tokens=int(policy.get("keep_tokens", policy.get("token_budget", 32))),
        pipeline=nodes,
        metadata=dict(policy.get("metadata") or {}),
    )
