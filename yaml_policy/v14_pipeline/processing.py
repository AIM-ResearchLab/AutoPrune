from __future__ import annotations

from typing import Any, Dict, Tuple, Optional
import math
import hashlib
import torch


def _safe_float(x: Any, default: float) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _safe_int(x: Any, default: int) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


def _minmax_score(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x = x.float()
    xmin = x.amin(dim=1, keepdim=True)
    xmax = x.amax(dim=1, keepdim=True)
    return (x - xmin) / (xmax - xmin + eps)


def _gather_features(features: torch.Tensor, selected: torch.Tensor) -> torch.Tensor:
    """features [B,N,D], selected [B,K] -> [B,K,D]."""
    bsz, _, dim = features.shape
    idx = selected.long().to(features.device)
    return torch.gather(features, 1, idx.unsqueeze(-1).expand(bsz, idx.size(1), dim))


def _gather_score(score: Optional[torch.Tensor], selected: torch.Tensor, default: float = 0.5) -> torch.Tensor:
    if score is None:
        return torch.full(selected.shape, float(default), device=selected.device)
    return torch.gather(score.float().to(selected.device), 1, selected.long())


def _selected_mask(selected: torch.Tensor, n_tokens: int) -> torch.Tensor:
    mask = torch.zeros((selected.size(0), n_tokens), dtype=torch.bool, device=selected.device)
    mask.scatter_(1, selected.long(), True)
    return mask


def _restore_token_norm(anchor: torch.Tensor, out: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    anchor_norm = anchor.norm(dim=-1, keepdim=True).clamp_min(eps)
    out_norm = out.norm(dim=-1, keepdim=True).clamp_min(eps)
    return out * (anchor_norm / out_norm)


def _tensor_hash(x: torch.Tensor, max_elems: int = 4096) -> str:
    """Small diagnostic hash. Avoid dumping huge tensors."""
    try:
        y = x.detach().float().reshape(-1)
        if y.numel() > max_elems:
            step = max(1, y.numel() // max_elems)
            y = y[::step][:max_elems]
        arr = y.cpu().numpy().tobytes()
        return hashlib.md5(arr).hexdigest()[:16]
    except Exception:
        return "NA"


def _feature_diag(anchor: torch.Tensor, out: torch.Tensor, processor: str) -> Dict[str, float | str]:
    with torch.no_grad():
        delta = out - anchor
        delta_norm = delta.norm(dim=-1).mean().item()
        anchor_norm = anchor.norm(dim=-1).mean().item()
        out_norm = out.norm(dim=-1).mean().item()
        ratio = out_norm / max(anchor_norm, 1e-6)
    return {
        "v14_processor": processor,
        "v14_feature_delta_norm_avg": float(delta_norm),
        "v14_feature_anchor_norm_avg": float(anchor_norm),
        "v14_feature_output_norm_avg": float(out_norm),
        "v14_feature_norm_ratio_avg": float(ratio),
        "v14_processed_feature_hash": _tensor_hash(out),
    }


def identity_token_processing(
    features: torch.Tensor,
    selected: torch.Tensor,
    **kwargs: Any,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    anchor = _gather_features(features, selected)
    diag = _feature_diag(anchor, anchor, "identity_token_processing")
    diag["v14_effective_alpha"] = 0.0
    return anchor, diag


def score_rescale_selected_features(
    features: torch.Tensor,
    selected: torch.Tensor,
    score: Optional[torch.Tensor] = None,
    beta: float = 0.04,
    center_score: bool = True,
    norm_preserve: bool = True,
    **kwargs: Any,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    anchor = _gather_features(features, selected)
    s = _gather_score(score, selected, default=0.5)

    if center_score:
        s = s - s.mean(dim=1, keepdim=True)

    scale = 1.0 + float(beta) * s.unsqueeze(-1)
    out = anchor * scale

    if norm_preserve:
        out = _restore_token_norm(anchor, out)

    diag = _feature_diag(anchor, out, "score_rescale_selected_features")
    diag["v14_effective_beta"] = float(beta)
    return out, diag


def _residual_merge_core(
    features: torch.Tensor,
    selected: torch.Tensor,
    kernel: Optional[torch.Tensor],
    gate_score: Optional[torch.Tensor],
    merge_alpha: float = 0.04,
    top_merge_neighbors: int = 2,
    temperature: float = 0.07,
    gate_threshold: float = 0.0,
    norm_preserve: bool = True,
    spatial_positions: Optional[torch.Tensor] = None,
    spatial_radius: Optional[float] = None,
    processor_name: str = "similarity_weighted_residual_merge",
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    device = features.device
    dtype = features.dtype
    bsz, n_tokens, dim = features.shape
    selected = selected.long().to(device)
    k = selected.size(1)
    anchor = _gather_features(features, selected)
    mask = _selected_mask(selected, n_tokens)

    if kernel is None:
        x = torch.nn.functional.normalize(features.float(), dim=-1)
        kernel = torch.bmm(x, x.transpose(1, 2)).to(device)
    else:
        kernel = kernel.float().to(device)

    if gate_score is None:
        gate_score = torch.ones((bsz, n_tokens), device=device)
    else:
        gate_score = _minmax_score(gate_score.float().to(device))

    out_all = []
    merged_counts = []

    for b in range(bsz):
        sel = selected[b]
        drop = torch.where(~mask[b])[0]

        if drop.numel() == 0:
            out_all.append(anchor[b])
            merged_counts.append(0.0)
            continue

        sim = kernel[b][sel][:, drop]  # [K, M]
        gate = gate_score[b][drop].float()  # [M]

        # Gate low-quality dropped tokens.
        if gate_threshold > 0:
            valid_gate = gate >= float(gate_threshold)
            sim = sim.masked_fill(~valid_gate.unsqueeze(0), -1e4)

        # Optional spatial local constraint.
        if spatial_positions is not None and spatial_radius is not None:
            try:
                pos = spatial_positions.to(device).float()
                sel_pos = pos[sel]       # [K,2]
                drop_pos = pos[drop]     # [M,2]
                dist = torch.cdist(sel_pos, drop_pos, p=2)
                sim = sim.masked_fill(dist > float(spatial_radius), -1e4)
            except Exception:
                pass

        kk = max(1, min(int(top_merge_neighbors), drop.numel()))
        top_val, top_pos = torch.topk(sim, k=kk, dim=1)
        weights = torch.softmax(top_val / max(float(temperature), 1e-6), dim=1)

        drop_idx = drop[top_pos]  # [K, kk]
        drop_feat = features[b][drop_idx]  # [K, kk, D]
        anchor_b = anchor[b]  # [K,D]

        gate_w = gate_score[b][drop_idx].unsqueeze(-1).to(dtype)
        residual = drop_feat - anchor_b.unsqueeze(1)
        merged = (weights.unsqueeze(-1).to(dtype) * gate_w * residual).sum(dim=1)

        out_b = anchor_b + float(merge_alpha) * merged
        out_all.append(out_b)
        merged_counts.append(float((top_val > -1e3).float().sum().item()) / float(k))

    out = torch.stack(out_all, dim=0)

    if norm_preserve:
        out = _restore_token_norm(anchor, out)

    diag = _feature_diag(anchor, out, processor_name)
    diag["v14_effective_alpha"] = float(merge_alpha)
    diag["v14_top_merge_neighbors"] = float(top_merge_neighbors)
    diag["v14_merged_dropped_tokens_avg"] = float(sum(merged_counts) / max(len(merged_counts), 1))
    return out.to(dtype), diag


def similarity_weighted_residual_merge(
    features: torch.Tensor,
    selected: torch.Tensor,
    kernel: Optional[torch.Tensor] = None,
    quality: Optional[torch.Tensor] = None,
    merge_alpha: float = 0.04,
    top_merge_neighbors: int = 2,
    temperature: float = 0.07,
    quality_gate_threshold: float = 0.0,
    norm_preserve: bool = True,
    **kwargs: Any,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    return _residual_merge_core(
        features=features,
        selected=selected,
        kernel=kernel,
        gate_score=quality,
        merge_alpha=merge_alpha,
        top_merge_neighbors=top_merge_neighbors,
        temperature=temperature,
        gate_threshold=quality_gate_threshold,
        norm_preserve=norm_preserve,
        processor_name="similarity_weighted_residual_merge",
    )


def semantic_gated_residual_merge(
    features: torch.Tensor,
    selected: torch.Tensor,
    kernel: Optional[torch.Tensor] = None,
    semantic: Optional[torch.Tensor] = None,
    quality: Optional[torch.Tensor] = None,
    merge_alpha: float = 0.04,
    top_merge_neighbors: int = 2,
    temperature: float = 0.07,
    semantic_gate_threshold: float = 0.65,
    norm_preserve: bool = True,
    **kwargs: Any,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    gate = semantic if semantic is not None else quality
    return _residual_merge_core(
        features=features,
        selected=selected,
        kernel=kernel,
        gate_score=gate,
        merge_alpha=merge_alpha,
        top_merge_neighbors=top_merge_neighbors,
        temperature=temperature,
        gate_threshold=semantic_gate_threshold,
        norm_preserve=norm_preserve,
        processor_name="semantic_gated_residual_merge",
    )


def spatial_local_residual_merge(
    features: torch.Tensor,
    selected: torch.Tensor,
    kernel: Optional[torch.Tensor] = None,
    quality: Optional[torch.Tensor] = None,
    token_positions: Optional[torch.Tensor] = None,
    merge_alpha: float = 0.04,
    top_merge_neighbors: int = 2,
    temperature: float = 0.07,
    spatial_radius: float = 2.0,
    norm_preserve: bool = True,
    **kwargs: Any,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    # If positions are unavailable, infer square grid when possible.
    if token_positions is None:
        n = features.size(1)
        side = int(round(math.sqrt(n)))
        if side * side == n:
            yy, xx = torch.meshgrid(
                torch.arange(side, device=features.device),
                torch.arange(side, device=features.device),
                indexing="ij",
            )
            token_positions = torch.stack([yy.reshape(-1), xx.reshape(-1)], dim=-1).float()

    return _residual_merge_core(
        features=features,
        selected=selected,
        kernel=kernel,
        gate_score=quality,
        merge_alpha=merge_alpha,
        top_merge_neighbors=top_merge_neighbors,
        temperature=temperature,
        gate_threshold=0.0,
        norm_preserve=norm_preserve,
        spatial_positions=token_positions,
        spatial_radius=spatial_radius,
        processor_name="spatial_local_residual_merge",
    )


def run_process_tokens_step(step: Dict[str, Any], env: Dict[str, torch.Tensor], ctx: Any) -> Tuple[torch.Tensor, Dict[str, Any]]:
    name = step.get("name")
    params = step.get("params", {}) or {}
    inputs = step.get("inputs", {}) or {}

    features_key = inputs.get("features", "image_features")
    selected_key = inputs.get("selected", "selected")
    quality_key = inputs.get("quality", "quality_score")
    semantic_key = inputs.get("semantic", "semantic_score")
    score_key = inputs.get("score", quality_key)
    kernel_key = inputs.get("kernel", "sim_kernel")

    features = env.get(features_key, None)
    if features is None:
        features = getattr(ctx, "image_features", None)
    if features is None:
        raise ValueError("v14 process_tokens cannot find image_features")

    selected = env.get(selected_key, None)
    if selected is None:
        raise ValueError(f"v14 process_tokens cannot find selected indices: {selected_key}")

    kernel = env.get(kernel_key, None)
    quality = env.get(quality_key, None)
    semantic = env.get(semantic_key, None)
    score = env.get(score_key, quality)
    token_positions = env.get("token_positions", getattr(ctx, "token_positions", None))

    common = dict(
        features=features,
        selected=selected,
        kernel=kernel,
        quality=quality,
        semantic=semantic,
        score=score,
        token_positions=token_positions,
    )

    if name == "identity_token_processing":
        return identity_token_processing(features=features, selected=selected)

    if name == "score_rescale_selected_features":
        return score_rescale_selected_features(
            features=features,
            selected=selected,
            score=score,
            beta=_safe_float(params.get("rescale_beta", params.get("beta", 0.04)), 0.04),
            center_score=bool(params.get("center_score", True)),
            norm_preserve=bool(params.get("norm_preserve", True)),
        )

    if name == "similarity_weighted_residual_merge":
        return similarity_weighted_residual_merge(
            **common,
            merge_alpha=_safe_float(params.get("merge_alpha", 0.04), 0.04),
            top_merge_neighbors=_safe_int(params.get("top_merge_neighbors", 2), 2),
            temperature=_safe_float(params.get("temperature", 0.07), 0.07),
            quality_gate_threshold=_safe_float(params.get("quality_gate_threshold", 0.0), 0.0),
            norm_preserve=bool(params.get("norm_preserve", True)),
        )

    if name == "semantic_gated_residual_merge":
        return semantic_gated_residual_merge(
            **common,
            merge_alpha=_safe_float(params.get("merge_alpha", 0.04), 0.04),
            top_merge_neighbors=_safe_int(params.get("top_merge_neighbors", 2), 2),
            temperature=_safe_float(params.get("temperature", 0.07), 0.07),
            semantic_gate_threshold=_safe_float(params.get("semantic_gate_threshold", 0.65), 0.65),
            norm_preserve=bool(params.get("norm_preserve", True)),
        )

    if name == "spatial_local_residual_merge":
        return spatial_local_residual_merge(
            **common,
            merge_alpha=_safe_float(params.get("merge_alpha", 0.04), 0.04),
            top_merge_neighbors=_safe_int(params.get("top_merge_neighbors", 2), 2),
            temperature=_safe_float(params.get("temperature", 0.07), 0.07),
            spatial_radius=_safe_float(params.get("spatial_radius", 2.0), 2.0),
            norm_preserve=bool(params.get("norm_preserve", True)),
        )

    raise ValueError(f"Unsupported v14 process_tokens processor: {name}")
