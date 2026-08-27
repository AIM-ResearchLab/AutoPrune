from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import yaml


@dataclass
class TokenBudget:
    mode: str = "fixed"
    keep: int = 32


@dataclass
class TokenSource:
    stream: str = "vision_tokens"
    layer: str = "pre_llm"
    include_cls: bool = False
    include_register: str = "optional"


@dataclass
class AnchorProviderConfig:
    """
    Universal anchor provider interface.

    This is intentionally adapter-agnostic:
      - cdpruner_v16: use CDPruner/v16 selected index_masks
      - openvla: future VLA action-relevant anchors
      - heuristic: pure DSL anchor selection
      - external_mask: externally provided mask
    """
    type: str = "external_policy"
    adapter: str = "cdpruner_v16"
    policy_path: str = ""
    output: str = "index_mask"
    freeze: bool = True


@dataclass
class ScoringConfig:
    features: List[str] = field(default_factory=lambda: [
        "attention_importance",
        "spatial_centrality",
        "semantic_similarity",
        "redundancy",
    ])
    weights: Dict[str, float] = field(default_factory=dict)


@dataclass
class AnchorSelectionConfig:
    operator: str = "topk_diverse"
    diversity: str = "spatial_semantic"
    min_spatial_coverage: float = 0.0
    protect: List[str] = field(default_factory=list)


@dataclass
class MergeAssignmentConfig:
    operator: str = "nearest_anchor"
    metric: str = "weighted_cosine_spatial"
    max_group_size: int = 16


@dataclass
class MergeOperatorConfig:
    type: str = "weighted_average"
    weights: Dict[str, float] = field(default_factory=dict)
    residual_scale: float = 0.0


@dataclass
class ResidualConfig:
    enabled: bool = False
    type: str = "mean_residual"
    inject_to: str = "anchors"
    scale: float = 0.0


@dataclass
class ConstraintConfig:
    preserve_budget: bool = True
    no_shape_change: bool = True
    deterministic: bool = True
    fallback_to_selection: bool = True


@dataclass
class UniversalMergePolicy:
    name: str
    token_budget: TokenBudget = field(default_factory=TokenBudget)
    token_source: TokenSource = field(default_factory=TokenSource)
    anchor_provider: AnchorProviderConfig = field(default_factory=AnchorProviderConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    anchor_selection: AnchorSelectionConfig = field(default_factory=AnchorSelectionConfig)
    merge_assignment: MergeAssignmentConfig = field(default_factory=MergeAssignmentConfig)
    merge_operator: MergeOperatorConfig = field(default_factory=MergeOperatorConfig)
    residual: ResidualConfig = field(default_factory=ResidualConfig)
    constraints: ConstraintConfig = field(default_factory=ConstraintConfig)
    raw: Dict[str, Any] = field(default_factory=dict)


def _get(d: Dict[str, Any], key: str, default: Any) -> Any:
    v = d.get(key, default)
    return default if v is None else v


def load_policy_yaml(text: str) -> UniversalMergePolicy:
    data = yaml.safe_load(text) or {}
    if "policy" in data:
        data = data["policy"]

    name = str(data.get("name", "unnamed_universal_merge_policy"))

    token_budget = TokenBudget(**_get(data, "token_budget", {}))
    token_source = TokenSource(**_get(data, "token_source", {}))
    anchor_provider = AnchorProviderConfig(**_get(data, "anchor_provider", {}))
    scoring = ScoringConfig(**_get(data, "scoring", {}))
    anchor_selection = AnchorSelectionConfig(**_get(data, "anchor_selection", {}))
    merge_assignment = MergeAssignmentConfig(**_get(data, "merge_assignment", {}))
    merge_operator = MergeOperatorConfig(**_get(data, "merge_operator", {}))
    residual = ResidualConfig(**_get(data, "residual", {}))
    constraints = ConstraintConfig(**_get(data, "constraints", {}))

    return UniversalMergePolicy(
        name=name,
        token_budget=token_budget,
        token_source=token_source,
        anchor_provider=anchor_provider,
        scoring=scoring,
        anchor_selection=anchor_selection,
        merge_assignment=merge_assignment,
        merge_operator=merge_operator,
        residual=residual,
        constraints=constraints,
        raw=data,
    )


def dump_policy(policy: UniversalMergePolicy) -> str:
    data = {
        "policy": {
            "name": policy.name,
            "token_budget": policy.token_budget.__dict__,
            "token_source": policy.token_source.__dict__,
            "anchor_provider": policy.anchor_provider.__dict__,
            "scoring": policy.scoring.__dict__,
            "anchor_selection": policy.anchor_selection.__dict__,
            "merge_assignment": policy.merge_assignment.__dict__,
            "merge_operator": policy.merge_operator.__dict__,
            "residual": policy.residual.__dict__,
            "constraints": policy.constraints.__dict__,
        }
    }
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
