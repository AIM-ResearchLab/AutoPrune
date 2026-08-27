from __future__ import annotations
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional
import torch, yaml
from . import atoms

def _is_v9_pipeline_policy(policy: Mapping[str, Any]) -> bool:
    pipe = policy.get("pipeline")
    if not isinstance(pipe, list) or not pipe:
        return False
    first = pipe[0]
    if not isinstance(first, Mapping):
        return False
    # Old YAML pipeline[0] has scorer/fusion/selector/budget.
    # v9 typed DSL pipeline nodes have op/name/output/inputs/params.
    return "op" in first and "name" in first


def load_policy(path_or_dict):
    if isinstance(path_or_dict, Mapping):
        return dict(path_or_dict)
    p = Path(path_or_dict)
    data = yaml.safe_load(p.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Invalid policy YAML: {p}")
    return data

class CDPrunerPolicyExecutor:
    def __init__(self, policy):
        self.policy = load_policy(policy)
        self.policy_name = str(self.policy.get("policy_name", "unnamed_policy"))
        self.last_processed_features = None
        self.last_v14_diagnostics = {}

    def select(self, ctx: Any) -> torch.Tensor:
        self.last_processed_features = None  # v14 reset
        self.last_v14_diagnostics = {}
        x = atoms.get_image_features(ctx)
        b, n, _ = x.shape

        if _is_v9_pipeline_policy(self.policy):
            return self._select_v9_pipeline(ctx, x)

        step = (self.policy.get("pipeline") or [None])[0]
        if not isinstance(step, Mapping):
            raise ValueError("policy.pipeline is empty")
        k = self._resolve_budget(step.get("budget", {}), ctx, n)
        scores: Dict[str, Any] = {}
        for name, spec in (step.get("scorer") or {}).items():
            if isinstance(spec, Mapping):
                scores[name] = self._run_scorer(spec, ctx, k)
        fused = self._run_fusion(step.get("fusion") or {"atom": "weighted_sum_score_fusion", "params": {}}, scores)
        mask = self._run_selector(step.get("selector") or {"atom": "topk_selector", "params": {}}, fused, k)
        if mask.dtype != torch.bool:
            mask = mask.bool()
        score, _, _ = atoms._get_score_and_sim(fused)
        if mask.shape != (b, n) or not torch.all(mask.sum(dim=1) == k):
            mask = self._force_exact_k(mask if mask.shape == (b,n) else torch.zeros((b,n),device=x.device,dtype=torch.bool), score, k)
        self._maybe_write_diagnostics(ctx, mask, fused, k)
        return mask

    def _select_v9_pipeline(self, ctx: Any, x: torch.Tensor) -> torch.Tensor:
        from yaml_policy.v9_pipeline.executor import execute_pipeline_policy

        b, n, _ = x.shape
        out = execute_pipeline_policy(
            self.policy,
            image_features=x,
            text_features=atoms.get_text_features(ctx),
            runtime_ctx=ctx,
        )
        selected = out["selected_indices"]

        # v14: optional post-selection processed features [B,K,D].
        self.last_processed_features = out.get("processed_features", None)
        self.last_v14_diagnostics = out.get("v14_diagnostics", {}) or {}


        k = int(self.policy.get("keep_tokens", atoms.get_keep_tokens(ctx, default=n)))
        k = max(1, min(n, k))

        if selected.dim() != 2:
            raise ValueError(f"v9 selected_indices must be [B,K], got {tuple(selected.shape)}")

        if selected.size(0) != b:
            if selected.size(0) == 1 and b > 1:
                selected = selected.expand(b, -1)
            else:
                raise ValueError(f"v9 selected batch mismatch: selected={tuple(selected.shape)}, B={b}")

        selected = selected[:, :k].long().to(x.device)

        mask = torch.zeros((b, n), device=x.device, dtype=torch.bool)
        mask.scatter_(1, selected, True)

        if mask.shape != (b, n) or not torch.all(mask.sum(dim=1) == k):
            score = torch.norm(x.float(), dim=-1)
            mask = self._force_exact_k(mask if mask.shape == (b, n) else torch.zeros((b, n), device=x.device, dtype=torch.bool), score, k)

        fused_for_diag = torch.norm(x.float(), dim=-1)
        self._maybe_write_diagnostics(ctx, mask, fused_for_diag, k)
        return mask

    def _resolve_budget(self, spec, ctx, n):
        atom = spec.get("atom", "fixed_keep_tokens") if isinstance(spec, Mapping) else "fixed_keep_tokens"
        params = spec.get("params", {}) if isinstance(spec, Mapping) else {}
        if atom == "fixed_keep_ratio":
            ratio = float(params.get("keep_ratio", 1.0))
            return max(1, min(n, max(int(params.get("min_keep_tokens", 1)), int(round(ratio*n)))))
        keep = params.get("keep_tokens", "host_argument")
        if keep == "host_argument":
            return atoms.get_keep_tokens(ctx, default=n)
        return max(1, min(n, int(keep)))

    def _run_scorer(self, spec, ctx, k=None):
        atom = spec.get("atom")
        params = dict(spec.get("params", {}) or {})

        # v6.2: reference prior must know the current token budget K.
        if atom == "cdpruner_reference_mask_prior" and k is not None:
            params.setdefault("keep_tokens", int(k))

        table = {
            "instruction_guided_visual_relevance": atoms.instruction_guided_visual_relevance,
            "visual_token_norm_saliency": atoms.visual_token_norm_saliency,
            "spatial_centrality_prior": atoms.spatial_centrality_prior,
            "pairwise_token_similarity": atoms.pairwise_token_similarity,
            "conditional_similarity_score": atoms.conditional_similarity_score,
            "redundancy_density_penalty": atoms.redundancy_density_penalty,
            "attention_proxy_saliency": atoms.attention_proxy_saliency,
            "cdpruner_default_soft_score": atoms.cdpruner_default_soft_score,
            "cdpruner_reference_mask_prior": atoms.cdpruner_reference_mask_prior,
            "native_teacher_mask_prior": atoms.native_teacher_mask_prior,
        }

        if atom not in table:
            raise ValueError(f"Unsupported scorer atom: {atom}")

        return table[atom](ctx, params)

    def _run_fusion(self, spec, scores):
        atom = spec.get("atom")
        params = dict(spec.get("params", {}) or {})

        if atom in {"weighted_sum_score_fusion", "full_token_score_fusion"}:
            return atoms.full_token_score_fusion(scores, params)

        if atom == "quality_diversity_kernel":
            return atoms.quality_diversity_kernel(scores, params)

        if atom == "v7_rank_fusion":
            return atoms.v7_rank_fusion(scores, params)

        if atom == "v7_score_product_fusion":
            return atoms.v7_score_product_fusion(scores, params)

        if atom == "v7_teacher_residual_fusion":
            return atoms.v7_teacher_residual_fusion(scores, params)

        raise ValueError(f"Unsupported fusion atom: {atom}")

    def _run_selector(self, spec, fused, k):
        atom = spec.get("atom")
        params = dict(spec.get("params", {}) or {})

        if atom == "topk_selector":
            return atoms.topk_selector(fused, k, params)

        if atom == "dpp_selector":
            return atoms.dpp_selector(fused, k, params)

        if atom in {"full_token_pool_dpp_selector", "topk_pool_dpp_selector"}:
            return atoms.full_token_pool_dpp_selector(fused, k, params)

        if atom in {"full_token_pool_mmr_selector", "topk_pool_mmr_selector", "mmr_selector"}:
            return atoms.full_token_pool_mmr_selector(fused, k, params)

        if atom == "stratified_full_token_selector":
            return atoms.stratified_full_token_selector(fused, k, params)

        if atom == "native_bounded_replace_selector":
            return atoms.native_bounded_replace_selector(fused, k, params)

        if atom == "v7_conditional_dpp_selector":
            return atoms.v7_conditional_dpp_selector(fused, k, params)

        if atom == "v7_two_stage_pool_selector":
            return atoms.v7_two_stage_pool_selector(fused, k, params)

        raise ValueError(f"Unsupported selector atom: {atom}")

    def _force_exact_k(self, mask, score, k):
        adjusted = score + mask.float() * 1e-4
        idx = torch.topk(adjusted, k=k, dim=-1).indices
        out = torch.zeros_like(mask, dtype=torch.bool)
        out.scatter_(1, idx, True)
        return out

    def _default_reference_mask(self, ctx, k) -> Optional[torch.Tensor]:
        # v6.3: diagnostics must compare candidate mask against native CDPruner teacher mask
        # when llava_arch.py provides it. Otherwise overlap/changed will be computed against
        # the weaker reconstructed YAML reference.
        try:
            native = atoms._native_teacher_mask_from_ctx(ctx)
            if native is not None:
                return native
        except Exception:
            pass

        try:
            return atoms.build_cdpruner_reference_mask(ctx, k)
        except Exception:
            return None

    def _maybe_write_diagnostics(self, ctx, mask, fused, k):
        path = os.environ.get("EVO_MASK_DIAG_PATH", "")
        if not path:
            return
        try:
            score, _, _ = atoms._get_score_and_sim(fused)
            ref = self._default_reference_mask(ctx, k)
            step = (self.policy.get("pipeline") or [{}])[0]
            row = {
                "policy_name": self.policy_name,
                "token_budget": int(k),
                "selector": (step.get("selector") or {}).get("atom"),
                "fusion": (step.get("fusion") or {}).get("atom"),
            }
            row.update(atoms.compute_mask_stats(mask, ref, score))
            try:
                row.update(self.last_v14_diagnostics)
            except Exception:
                pass
            atoms.append_jsonl(path, row)
        except Exception as exc:
            atoms.append_jsonl(path, {"policy_name": self.policy_name, "diag_error": repr(exc)})
