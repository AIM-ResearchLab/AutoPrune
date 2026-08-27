from __future__ import annotations

from typing import Any, Dict, Mapping
import torch
from yaml_policy.v14_pipeline import processing as V14P

from .schema import parse_pipeline_policy
from . import primitives as P


def infer_grid_hw(n: int):
    side = int(round(n ** 0.5))
    if side * side == n:
        return (side, side)
    return None


def _as_score_tensor(v: Any) -> torch.Tensor:
    if torch.is_tensor(v):
        if v.dim() == 1:
            return v.unsqueeze(0)
        if v.dim() == 2:
            return v
        if v.dim() == 3:
            return torch.norm(v.float(), dim=-1)
        raise ValueError(f"cannot convert tensor with shape {tuple(v.shape)} to score")

    if isinstance(v, Mapping):
        for key in ["score", "quality", "relevance", "saliency", "scores"]:
            if key in v:
                return _as_score_tensor(v[key])

    raise TypeError(f"cannot convert {type(v)!r} to score tensor")



def _runtime_conditional_kernel(
    runtime_ctx: Any,
    quality: torch.Tensor,
    quality_power: float = 1.0,
    shift_similarity: bool = True,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Build a CDPruner-like conditional similarity kernel from runtime image_embeds.

    quality: [B, N]
    output: [B, N, N]
    """
    if runtime_ctx is None:
        raise ValueError("runtime_conditional_similarity_kernel requires runtime_ctx")

    from cdpruner_policy import atoms as legacy_atoms

    image_embeds = legacy_atoms._get_attr(runtime_ctx, ["image_embeds", "image_embed", "clip_embeds"], None)
    if image_embeds is None:
        image_embeds = legacy_atoms.get_image_features(runtime_ctx)

    if image_embeds.dim() == 2:
        image_embeds = image_embeds.unsqueeze(0)
    if image_embeds.dim() != 3:
        raise ValueError(f"image_embeds must be [B,N,D], got {tuple(image_embeds.shape)}")

    image_embeds = image_embeds.to(device=quality.device).float()
    x = torch.nn.functional.normalize(image_embeds, dim=-1)

    sim = torch.matmul(x, x.transpose(-1, -2))
    if shift_similarity:
        sim = (sim + 1.0) * 0.5
    sim = sim.clamp(min=0.0)

    q = P.normalize_score(quality).clamp(min=eps, max=1.0)
    q = q.pow(float(quality_power))

    kernel = sim * q.unsqueeze(-1) * q.unsqueeze(-2)

    # Symmetrize for numerical stability.
    kernel = 0.5 * (kernel + kernel.transpose(-1, -2))
    return kernel.clamp(min=0.0)



def _legacy_score(name: str, runtime_ctx: Any, params: Dict[str, Any], keep_tokens: int | None = None) -> torch.Tensor:
    if runtime_ctx is None:
        raise ValueError(f"legacy scorer {name} requires runtime_ctx")

    params = dict(params or {})
    if keep_tokens is not None and name in {"cdpruner_reference_mask_prior", "native_teacher_mask_prior"}:
        params.setdefault("keep_tokens", int(keep_tokens))

    from cdpruner_policy import atoms as legacy_atoms

    table = {
        "instruction_guided_visual_relevance": legacy_atoms.instruction_guided_visual_relevance,
        "attention_proxy_saliency": legacy_atoms.attention_proxy_saliency,
        "spatial_centrality_prior": legacy_atoms.spatial_centrality_prior,
        "visual_token_norm_saliency": legacy_atoms.visual_token_norm_saliency,
        "redundancy_density_penalty": legacy_atoms.redundancy_density_penalty,
        "cdpruner_default_soft_score": legacy_atoms.cdpruner_default_soft_score,
        "cdpruner_reference_mask_prior": legacy_atoms.cdpruner_reference_mask_prior,
        "native_teacher_mask_prior": legacy_atoms.native_teacher_mask_prior,
    }

    if name not in table:
        raise ValueError(f"unsupported legacy scorer in v9: {name}")

    return P.normalize_score(_as_score_tensor(table[name](runtime_ctx, params)))





def _v11_step_inputs(local_vars):
    """Return current YAML node inputs from executor local variables.

    Supports:
      - dict step: step["inputs"]
      - PipelineNode-like object: node.inputs
      - pre-existing local variable: inputs
    """
    direct = local_vars.get("inputs")
    if isinstance(direct, dict):
        return direct

    for key in ("node", "step", "item", "entry", "spec", "op_spec", "stage", "block"):
        obj = local_vars.get(key)
        if obj is None:
            continue

        if isinstance(obj, dict):
            inp = obj.get("inputs", {}) or {}
            return inp if isinstance(inp, dict) else {}

        inp = getattr(obj, "inputs", None)
        if isinstance(inp, dict):
            return inp

    return {}



def _v11_get_k_keep(local_vars, params=None, quality=None, default=32):
    """Robustly infer keep-token count inside v9 executor.

    The original executor may use different variable names, so v11 atoms should
    not assume a local variable named k_keep exists.
    """
    params = params or {}

    for key in ("k_keep", "keep_tokens", "keep", "k", "num_keep", "token_budget"):
        v = local_vars.get(key)
        if isinstance(v, int):
            return int(v)
        if isinstance(v, float):
            return int(v)

    for key in ("node", "step", "item", "entry", "spec", "op_spec", "stage", "block"):
        obj = local_vars.get(key)
        if obj is None:
            continue

        if isinstance(obj, dict):
            for kk in ("keep_tokens", "k_keep", "k", "keep"):
                if kk in obj:
                    try:
                        return int(obj[kk])
                    except Exception:
                        pass

        for kk in ("keep_tokens", "k_keep", "k", "keep"):
            if hasattr(obj, kk):
                try:
                    return int(getattr(obj, kk))
                except Exception:
                    pass

    for kk in ("keep_tokens", "k_keep", "k", "keep"):
        if kk in params:
            try:
                return int(params[kk])
            except Exception:
                pass

    # Policy-level fallback if available.
    policy = local_vars.get("policy")
    if isinstance(policy, dict):
        for kk in ("keep_tokens", "k_keep", "k", "keep"):
            if kk in policy:
                try:
                    return int(policy[kk])
                except Exception:
                    pass

    # Runtime-context fallback.
    runtime_ctx = local_vars.get("runtime_ctx")
    if isinstance(runtime_ctx, dict):
        for kk in ("keep_tokens", "k_keep", "k", "keep", "token_budget"):
            if kk in runtime_ctx:
                try:
                    return int(runtime_ctx[kk])
                except Exception:
                    pass

    try:
        if quality is not None:
            return min(int(default), int(quality.shape[1]))
    except Exception:
        pass

    return int(default)



def execute_pipeline_policy(
    policy: Dict[str, Any],
    image_features: torch.Tensor,
    text_features: torch.Tensor | None = None,
    runtime_ctx: Any | None = None,
) -> Dict[str, Any]:
    program = parse_pipeline_policy(policy)
    x = image_features
    b, n, d = x.shape
    grid_hw = infer_grid_hw(n)

    env: Dict[str, Any] = {"image_features": x}
    trace: Dict[str, Any] = {
        "program_name": program.name,
        "keep_tokens": program.keep_tokens,
        "nodes": [],
    }

    for node in program.pipeline:
        op = node.op
        name = node.name
        out = node.output or name
        params = node.params or {}

        if op == "compute_score":
            if name == "local_contrast_saliency":
                env[out] = P.local_contrast_saliency(x, grid_hw=grid_hw)
            elif name == "visual_token_norm_saliency":
                env[out] = P.normalize_score(torch.norm(x.float(), dim=-1))
            elif name == "cluster_centroid_representativeness":
                clusters = env[_v11_step_inputs(locals()).get("clusters", "clusters")]
                env[out] = P.cluster_centroid_representativeness(
                    x, clusters["cluster_id"], clusters["centers"]
                )
            elif name in {
                "instruction_guided_visual_relevance",
                "attention_proxy_saliency",
                "spatial_centrality_prior",
                "redundancy_density_penalty",
                "cdpruner_default_soft_score",
                "cdpruner_reference_mask_prior",
                "native_teacher_mask_prior",
            }:
                if runtime_ctx is not None:
                    env[out] = _legacy_score(name, runtime_ctx, params, keep_tokens=program.keep_tokens)
                else:
                    env[out] = P.normalize_score(torch.norm(x.float(), dim=-1))
                    trace["nodes"].append({"warning": f"{name} uses norm placeholder without runtime_ctx"})
            else:
                raise ValueError(f"unknown compute_score primitive: {name}")

        elif op in {"fuse_score", "fuse_scores"}:
            if name not in {"weighted_sum_fusion", "weighted_sum_score_fusion", "v9_weighted_sum"}:
                raise ValueError(f"unknown fuse_score primitive: {name}")

            score_inputs = _v11_step_inputs(locals()).get("scores", {})
            weights = dict(params.get("weights", {}) or {})

            if not score_inputs:
                raise ValueError("fuse_score requires inputs.scores")

            acc = None
            total_w = 0.0
            for alias, key in score_inputs.items():
                sc = P.normalize_score(env[key])
                w = float(weights.get(alias, 1.0))
                acc = sc * w if acc is None else acc + sc * w
                total_w += abs(w)

            if total_w <= 0:
                total_w = 1.0
            env[out] = P.normalize_score(acc / total_w)


        elif op in {"product_fuse_score", "product_fuse_scores"}:
            if name not in {"weighted_product_fusion", "v9_weighted_product", "v7_score_product_fusion", "reference_confidence_weighted_product_fusion", "question_aware_weighted_product_fusion"}:
                raise ValueError(f"unknown product_fuse_score primitive: {name}")

            score_inputs = _v11_step_inputs(locals()).get("scores", {})
            weights = dict(params.get("weights", {}) or {})

            if not score_inputs:
                raise ValueError("product_fuse_score requires inputs.scores")

            scores = {}
            for alias, key in score_inputs.items():
                scores[alias] = env[key]

            if name in {"reference_confidence_weighted_product_fusion", "question_aware_weighted_product_fusion"}:
                ref_key = _v11_step_inputs(locals()).get("reference", "reference_score")
                ref_score = env.get(ref_key) if isinstance(ref_key, str) else None
                env[out] = P.reference_confidence_weighted_product_fusion(scores, ref_score, params)
            else:
                env[out] = P.weighted_product_fusion(scores, weights)


        elif op == "compute_kernel":
            if name in {"pairwise_cosine_kernel", "pairwise_token_similarity_kernel"}:
                env[out] = P.pairwise_cosine_kernel(x)

            elif name in {"runtime_conditional_similarity_kernel", "cdpruner_conditional_similarity_kernel"}:
                quality_key = _v11_step_inputs(locals()).get("quality")
                if not quality_key:
                    raise ValueError("runtime_conditional_similarity_kernel requires inputs.quality")
                env[out] = _runtime_conditional_kernel(
                    runtime_ctx=runtime_ctx,
                    quality=env[quality_key],
                    quality_power=float(params.get("quality_power", 1.0)),
                    shift_similarity=bool(params.get("shift_similarity", True)),
                )

            else:
                raise ValueError(f"unknown compute_kernel primitive: {name}")

        elif op == "partition":
            if name == "feature_cluster_partition":
                c = int(params.get("num_clusters", 8))
                cluster_id, centers = P.feature_cluster_partition(x, num_clusters=c)
                env[out] = {"cluster_id": cluster_id, "centers": centers}
            elif name == "spatial_grid_partition":
                grid = tuple(params.get("grid", [4, 4]))
                env[out] = P.spatial_grid_ids(n, x.device, grid_hw=grid)
            else:
                raise ValueError(f"unknown partition primitive: {name}")

        elif op == "build_pool":
            pool_factor = float(params.get("pool_factor", 3.0))
            pool_size = int(min(n, max(program.keep_tokens, round(program.keep_tokens * pool_factor))))

            if name == "multi_source_pool_builder":
                score_inputs = _v11_step_inputs(locals()).get("scores", {})
                scores = {alias: env[key] for alias, key in score_inputs.items()}
                env[out] = P.build_multi_source_pool(scores, pool_size=pool_size)

            elif name == "spatial_coverage_pool_builder":
                q_key = _v11_step_inputs(locals()).get("quality")
                if q_key is None:
                    raise ValueError("spatial_coverage_pool_builder requires inputs.quality")
                quality = env[q_key]
                spatial_key = _v11_step_inputs(locals()).get("spatial", "spatial_score")
                spatial = env.get(spatial_key) if isinstance(spatial_key, str) else None
                pool_factor = float(params.get("pool_factor", 3.0))
                _v11_k_keep = _v11_get_k_keep(locals(), params=params, quality=quality, default=32)
                pool_size = int(max(_v11_k_keep, min(quality.shape[1], round(_v11_k_keep * pool_factor))))
                env[out] = P.spatial_coverage_pool_builder(
                    quality,
                    spatial=spatial,
                    pool_size=pool_size,
                    grid_size=int(params.get("grid_size", 4)),
                    min_tokens_per_bin=int(params.get("min_tokens_per_bin", 1)),
                    high_quality_ratio=float(params.get("high_quality_ratio", 0.75)),
                )

            elif name in {"quality_pool_builder", "top_quality_pool_builder"}:
                quality_key = _v11_step_inputs(locals()).get("quality")
                if not quality_key:
                    raise ValueError("quality_pool_builder requires inputs.quality")
                env[out] = P.build_quality_pool(env[quality_key], pool_size=pool_size)

            else:
                raise ValueError(f"unknown build_pool primitive: {name}")

        elif op == "process_tokens":
            step_dict = {
                "op": op,
                "name": name,
                "inputs": getattr(node, "inputs", {}) or {},
                "params": params,
                "output": out,
            }
            processed, diag = V14P.run_process_tokens_step(step_dict, env, runtime_ctx)
            env[out] = processed
            env["processed_features"] = processed
            env["v14_diagnostics"] = dict(diag)
            trace["v14_diagnostics"] = dict(diag)
            trace["nodes"].append({
                "op": op,
                "name": name,
                "output": out,
                "v14_processor": diag.get("v14_processor", name),
                "v14_feature_delta_norm_avg": diag.get("v14_feature_delta_norm_avg", None),
                "v14_feature_norm_ratio_avg": diag.get("v14_feature_norm_ratio_avg", None),
            })

        elif op == "select":
            if name in {"constrained_pool_dpp_selector", "coverage_aware_pool_dpp_selector", "cluster_aware_pool_dpp_selector"}:
                quality = env[node.inputs["quality"]]
                kernel = env[node.inputs["kernel"]]
                pool = env[node.inputs["pool"]]
                grid_ids = env.get(_v11_step_inputs(locals()).get("grid")) if _v11_step_inputs(locals()).get("grid") else None
                clusters = env.get(_v11_step_inputs(locals()).get("clusters")) if _v11_step_inputs(locals()).get("clusters") else None
                cluster_id = clusters["cluster_id"] if isinstance(clusters, dict) else None

                env[out] = P.constrained_pool_dpp_select(
                    quality=quality,
                    kernel=kernel,
                    pool=pool,
                    keep_tokens=program.keep_tokens,
                    grid_ids=grid_ids,
                    cluster_id=cluster_id,
                    min_per_grid=int(params.get("min_per_grid", 0)),
                    min_per_cluster=int(params.get("min_per_cluster", 0)),
                    diversity_lambda=float(params.get("diversity_lambda", 0.35)),
                )

            elif name == "anchor_controlled_pool_dpp_selector":
                quality = env[node.inputs["quality"]]
                kernel = env[node.inputs["kernel"]]
                pool = env[node.inputs["pool"]]
                reference = env[node.inputs["reference"]]

                env[out] = P.anchor_controlled_pool_dpp_select(
                    quality=quality,
                    kernel=kernel,
                    pool=pool,
                    reference_score=reference,
                    keep_tokens=program.keep_tokens,
                    anchor_quota=int(params.get("anchor_quota", 8)),
                    diversity_lambda=float(params.get("diversity_lambda", 0.25)),
                    allow_anchor_reorder=bool(params.get("allow_anchor_reorder", True)),
                )

            elif name == "native_reference_strict_margin_exchange_selector":
                inp = _v11_step_inputs(locals())

                quality_key = inp.get("quality", "quality_score")
                reference_key = inp.get("reference", "reference_score")
                kernel_key = inp.get("kernel", "sim_kernel")

                if quality_key not in env:
                    raise ValueError(f"native_reference_strict_margin_exchange_selector missing quality: {quality_key}")
                if reference_key not in env:
                    raise ValueError(f"native_reference_strict_margin_exchange_selector missing reference: {reference_key}")

                quality = env[quality_key]
                reference = env[reference_key]
                kernel = env.get(kernel_key) if isinstance(kernel_key, str) else None

                env[out] = P.native_reference_strict_margin_exchange_select(
                    quality=quality,
                    reference_score=reference,
                    kernel=kernel,
                    k_keep=_v11_get_k_keep(locals(), params=params, quality=quality, default=32),
                    max_replace_quota=int(params.get("max_replace_quota", 3)),
                    margin_threshold=float(params.get("margin_threshold", 0.12)),
                    min_gain_ratio=float(params.get("min_gain_ratio", 1.10)),
                    min_candidate_quality=float(params.get("min_candidate_quality", 0.75)),
                    max_drop_quality=float(params.get("max_drop_quality", 0.80)),
                    protect_top_ref=int(params.get("protect_top_ref", 28)),
                    diversity_lambda=float(params.get("diversity_lambda", 0.08)),
                    reference_keep_weight=float(params.get("reference_keep_weight", 0.20)),
                    candidate_quality_power=float(params.get("candidate_quality_power", 1.0)),
                    sort_selected=bool(params.get("sort_selected", True)),
                )

            elif name == "native_reference_margin_gated_exchange_selector":
                inp = _v11_step_inputs(locals())

                quality_key = inp.get("quality", "quality_score")
                reference_key = inp.get("reference", "reference_score")
                kernel_key = inp.get("kernel", "sim_kernel")

                if quality_key not in env:
                    raise ValueError(f"native_reference_margin_gated_exchange_selector missing quality: {quality_key}")
                if reference_key not in env:
                    raise ValueError(f"native_reference_margin_gated_exchange_selector missing reference: {reference_key}")

                quality = env[quality_key]
                reference = env[reference_key]
                kernel = env.get(kernel_key) if isinstance(kernel_key, str) else None

                env[out] = P.native_reference_margin_gated_exchange_select(
                    quality=quality,
                    reference_score=reference,
                    kernel=kernel,
                    k_keep=_v11_get_k_keep(locals(), params=params, quality=quality, default=32),
                    max_replace_quota=int(params.get("max_replace_quota", 3)),
                    margin_threshold=float(params.get("margin_threshold", 0.04)),
                    protect_top_ref=int(params.get("protect_top_ref", 26)),
                    diversity_lambda=float(params.get("diversity_lambda", 0.08)),
                    reference_keep_weight=float(params.get("reference_keep_weight", 0.20)),
                    candidate_quality_power=float(params.get("candidate_quality_power", 1.0)),
                    sort_selected=bool(params.get("sort_selected", True)),
                )

            elif name == "native_reference_residual_exchange_selector":
                inp = _v11_step_inputs(locals())

                quality_key = inp.get("quality", "quality_score")
                reference_key = inp.get("reference", "reference_score")
                kernel_key = inp.get("kernel", "sim_kernel")

                if quality_key not in env:
                    raise ValueError(f"native_reference_residual_exchange_selector missing quality: {quality_key}")
                if reference_key not in env:
                    raise ValueError(f"native_reference_residual_exchange_selector missing reference: {reference_key}")

                quality = env[quality_key]
                reference = env[reference_key]
                kernel = env.get(kernel_key) if isinstance(kernel_key, str) else None

                env[out] = P.native_reference_residual_exchange_select(
                    quality=quality,
                    reference_score=reference,
                    kernel=kernel,
                    k_keep=_v11_get_k_keep(locals(), params=params, quality=quality, default=32),
                    replace_quota=int(params.get("replace_quota", 4)),
                    min_reference_keep=int(params.get("min_reference_keep", 28)),
                    diversity_lambda=float(params.get("diversity_lambda", 0.08)),
                    reference_keep_weight=float(params.get("reference_keep_weight", 0.20)),
                    candidate_quality_power=float(params.get("candidate_quality_power", 1.0)),
                    sort_selected=bool(params.get("sort_selected", True)),
                )

            elif name == "adaptive_reference_anchor_pool_dpp_selector":
                if runtime_ctx is None:
                    raise ValueError("adaptive_reference_anchor_pool_dpp_selector requires runtime_ctx")

                pool_key = _v11_step_inputs(locals()).get("pool", "pool")
                quality_key = _v11_step_inputs(locals()).get("quality", "quality_score")
                kernel_key = _v11_step_inputs(locals()).get("kernel", "sim_kernel")

                pool = env[pool_key]
                quality = env[quality_key]
                kernel = env[kernel_key]

                ref_mask = None
                if isinstance(runtime_ctx, dict):
                    for _k in ["reference_mask", "ref_mask", "cdpruner_reference_mask", "cdpruner_mask", "base_mask", "mask"]:
                        if _k in runtime_ctx:
                            ref_mask = runtime_ctx[_k]
                            break
                if ref_mask is None:
                    # Fallback to YAML-provided reference score.
                    # In v11 policies, the selector may pass:
                    #   inputs:
                    #     reference: reference_score
                    # If runtime_ctx does not expose a native bool mask, we build a
                    # pseudo-reference mask by taking the top-K reference_score tokens.
                    ref_key = _v11_step_inputs(locals()).get("reference", "reference_score")
                    ref_signal = env.get(ref_key) if isinstance(ref_key, str) else None
                    if ref_signal is None:
                        ref_signal = env.get("reference_score")

                    if ref_signal is None:
                        raise ValueError("adaptive_reference_anchor_pool_dpp_selector cannot find reference mask or reference_score")

                    rr = ref_signal.to(device=quality.device)

                    if rr.dim() == 2 and rr.shape[1] == quality.shape[1] and rr.dtype == torch.bool:
                        ref_mask = rr.bool()
                    elif rr.dim() == 2 and rr.shape[1] == quality.shape[1]:
                        rr = rr.float()
                        kk = int(min(_v11_get_k_keep(locals(), params=params, quality=quality, default=32), rr.shape[1]))
                        idx = rr.topk(k=kk, dim=1).indices
                        ref_mask = torch.zeros_like(rr, dtype=torch.bool, device=quality.device)
                        ref_mask.scatter_(1, idx, True)
                    elif rr.dim() == 2:
                        ref_mask = torch.zeros_like(quality, dtype=torch.bool, device=quality.device)
                        idx = rr.long().clamp(min=0, max=quality.shape[1] - 1)
                        ref_mask.scatter_(1, idx, True)
                    else:
                        raise ValueError(f"bad reference_score shape for adaptive selector fallback: {tuple(rr.shape)}")

                env[out] = P.adaptive_reference_anchor_pool_dpp_select(
                    quality=quality,
                    kernel=kernel,
                    pool=pool,
                    reference_mask=ref_mask,
                    k_keep=_v11_get_k_keep(locals(), params=params, quality=quality, default=32),
                    min_anchor_quota=int(params.get("min_anchor_quota", 6)),
                    default_anchor_quota=int(params.get("default_anchor_quota", params.get("anchor_quota", 8))),
                    max_anchor_quota=int(params.get("max_anchor_quota", 10)),
                    diversity_lambda=float(params.get("diversity_lambda", 0.18)),
                    confidence_threshold_low=float(params.get("confidence_threshold_low", 0.45)),
                    confidence_threshold_high=float(params.get("confidence_threshold_high", 0.62)),
                    exclude_unanchored_reference=bool(params.get("exclude_unanchored_reference", True)),
                    sort_selected=bool(params.get("sort_selected", True)),
                )

            elif name == "runtime_reference_anchor_pool_dpp_selector":
                if runtime_ctx is None:
                    raise ValueError("runtime_reference_anchor_pool_dpp_selector requires runtime_ctx")

                from cdpruner_policy import atoms as legacy_atoms

                quality = env[node.inputs["quality"]]
                kernel = env[node.inputs["kernel"]]
                pool = env[node.inputs["pool"]]

                ref_mask = None
                try:
                    ref_mask = legacy_atoms._native_teacher_mask_from_ctx(runtime_ctx)
                except Exception:
                    ref_mask = None

                if ref_mask is None:
                    try:
                        ref_mask = legacy_atoms.build_cdpruner_reference_mask(runtime_ctx, program.keep_tokens)
                    except Exception:
                        ref_mask = None

                if ref_mask is None:
                    # Last fallback: use provided reference score top-k.
                    reference = env[node.inputs["reference"]]
                    idx = torch.topk(reference, k=program.keep_tokens, dim=-1).indices
                    ref_mask = torch.zeros_like(reference, dtype=torch.bool)
                    ref_mask.scatter_(1, idx, True)

                if ref_mask.dim() == 1:
                    ref_mask = ref_mask.unsqueeze(0)
                ref_mask = ref_mask.to(device=quality.device).bool()

                if ref_mask.size(0) != quality.size(0):
                    if ref_mask.size(0) == 1:
                        ref_mask = ref_mask.expand(quality.size(0), -1)
                    else:
                        raise ValueError(f"reference mask batch mismatch: {tuple(ref_mask.shape)} vs {tuple(quality.shape)}")

                env[out] = P.reference_mask_anchor_pool_dpp_select(
                    quality=quality,
                    kernel=kernel,
                    pool=pool,
                    reference_mask=ref_mask,
                    keep_tokens=program.keep_tokens,
                    anchor_quota=int(params.get("anchor_quota", 8)),
                    diversity_lambda=float(params.get("diversity_lambda", 0.25)),
                    exclude_unanchored_reference=bool(params.get("exclude_unanchored_reference", True)),
                    sort_selected=bool(params.get("sort_selected", True)),
                )

            else:
                raise ValueError(f"unknown select primitive: {name}")
        else:
            raise ValueError(f"unknown pipeline op: {op}")

        trace["nodes"].append({"op": op, "name": name, "output": out})

    selected = None
    for node in reversed(program.pipeline):
        key = node.output or node.name
        if key in env and torch.is_tensor(env[key]) and env[key].dtype == torch.long:
            selected = env[key]
            break

    if selected is None:
        raise ValueError("pipeline did not produce selected indices")

    return {
        "selected_indices": selected,
        "processed_features": env.get("processed_features", None),
        "v14_diagnostics": env.get("v14_diagnostics", {}),
        "trace": trace,
    }
