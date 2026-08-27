from __future__ import annotations
import json, math, os
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union
import torch
import torch.nn.functional as F

Tensor = torch.Tensor

def _get_attr(ctx: Any, names: Sequence[str], default: Any = None) -> Any:
    if isinstance(ctx, Mapping):
        for n in names:
            if n in ctx:
                return ctx[n]
    for n in names:
        if hasattr(ctx, n):
            return getattr(ctx, n)
    return default

def get_image_features(ctx: Any) -> Tensor:
    if torch.is_tensor(ctx):
        x = ctx
    else:
        x = _get_attr(ctx, ["image_features", "visual_tokens", "image_tokens", "features", "tokens", "x"])
    if x is None:
        raise ValueError("PolicyContext does not contain image features")
    if not torch.is_tensor(x):
        raise TypeError(f"image features must be a torch.Tensor, got {type(x)!r}")
    if x.dim() == 2:
        x = x.unsqueeze(0)
    if x.dim() != 3:
        raise ValueError(f"image features must be [B,N,D], got {tuple(x.shape)}")
    return x

def get_text_features(ctx: Any) -> Optional[Tensor]:
    x = _get_attr(ctx, ["text_features", "instruction_features", "query_features", "txt_features"], None)
    if x is None or not torch.is_tensor(x):
        return None
    if x.dim() == 1:
        x = x.unsqueeze(0)
    return x

def get_keep_tokens(ctx: Any, default: Optional[int] = None) -> int:
    feats = get_image_features(ctx)
    n = feats.size(1)
    for name in ["keep_tokens", "visual_token_num", "token_budget", "k"]:
        v = _get_attr(ctx, [name], None)
        if v is not None:
            try: return max(1, min(int(v), n))
            except Exception: pass
    for env_name in ["EVO_FORCE_FIXED_TOKENS", "EVO_TOKEN", "VISUAL_TOKEN_NUMBER"]:
        v = os.environ.get(env_name)
        if v:
            try: return max(1, min(int(v), n))
            except Exception: pass
    return max(1, min(int(default if default is not None else n), n))

def _as_float(x: Any, default: float) -> float:
    try: return float(x)
    except Exception: return float(default)

def _as_int(x: Any, default: int) -> int:
    try: return int(x)
    except Exception: return int(default)

def _rank_norm(x: Tensor) -> Tensor:
    order = torch.argsort(x, dim=-1)
    ranks = torch.zeros_like(x, dtype=torch.float32)
    base = torch.arange(x.size(-1), device=x.device, dtype=torch.float32).view(1, -1).expand_as(order)
    ranks.scatter_(dim=-1, index=order, src=base)
    return ranks / float(max(x.size(-1) - 1, 1))

def _minmax(x: Tensor, dim: int = -1, eps: float = 1e-6) -> Tensor:
    lo = x.amin(dim=dim, keepdim=True)
    hi = x.amax(dim=dim, keepdim=True)
    return (x - lo) / (hi - lo + eps)

def normalize_score(x: Tensor, transform: str = "minmax", power: float = 1.0) -> Tensor:
    t = (transform or "identity").lower()
    if t in {"identity", "none"}:
        y = x.float()
    elif t in {"minmax", "global_minmax"}:
        y = _minmax(x.float())
    elif t in {"rank", "rank_norm"}:
        y = _rank_norm(x.float())
    elif t == "softmax":
        y = _minmax(torch.softmax(x.float(), dim=-1))
    elif t == "sigmoid":
        y = _minmax(torch.sigmoid(x.float()))
    elif t == "zscore":
        z = (x.float() - x.float().mean(dim=-1, keepdim=True)) / (x.float().std(dim=-1, keepdim=True) + 1e-6)
        y = _minmax(z)
    else:
        y = _minmax(x.float())
    if power != 1.0:
        y = torch.clamp(y, 0.0, 1.0).pow(float(power))
    return y

def instruction_guided_visual_relevance(ctx: Any, params: Mapping[str, Any] | None = None) -> Tensor:
    params = params or {}
    feats = get_image_features(ctx).float()
    text = get_text_features(ctx)
    if text is not None and text.dim() == 2 and text.size(0) == feats.size(0) and text.size(1) == feats.size(2):
        score = F.cosine_similarity(F.normalize(feats, dim=-1), F.normalize(text[:, None, :], dim=-1), dim=-1)
    else:
        sign = str(params.get("sign", "negative_mean"))
        if sign == "max":
            score = feats.max(dim=-1).values
        elif sign == "mean":
            score = feats.mean(dim=-1)
        elif sign == "norm":
            score = feats.norm(dim=-1)
        elif sign == "abs_mean":
            score = feats.mean(dim=-1).abs()
        else:
            score = -feats.mean(dim=-1)
    return normalize_score(score, params.get("transform", params.get("score_normalization", "minmax")), _as_float(params.get("power", 1.0), 1.0))

def visual_token_norm_saliency(ctx: Any, params: Mapping[str, Any] | None = None) -> Tensor:
    params = params or {}
    return normalize_score(get_image_features(ctx).float().norm(dim=-1), params.get("transform", "minmax"), _as_float(params.get("power", 1.0), 1.0))

def spatial_centrality_prior(ctx: Any, params: Mapping[str, Any] | None = None) -> Tensor:
    params = params or {}
    feats = get_image_features(ctx)
    b, n, _ = feats.shape
    side = int(round(math.sqrt(n)))
    if side * side != n:
        pos = torch.linspace(-1, 1, steps=n, device=feats.device).abs()
        return (1.0 - pos).view(1, n).expand(b, n)
    yy, xx = torch.meshgrid(torch.linspace(-1,1,side,device=feats.device), torch.linspace(-1,1,side,device=feats.device), indexing="ij")
    dist = torch.sqrt(xx*xx + yy*yy)
    score = torch.exp(-_as_float(params.get("center_bias", 1.0), 1.0) * dist).reshape(1, n).expand(b, n)
    return normalize_score(score, params.get("transform", "minmax"))

def pairwise_token_similarity(ctx: Any, params: Mapping[str, Any] | None = None) -> Tensor:
    params = params or {}
    x = F.normalize(get_image_features(ctx).float(), dim=-1)
    sim = torch.bmm(x, x.transpose(1, 2))
    tr = params.get("similarity_transform", params.get("transform", "identity"))
    if tr in {"minmax", "rank_norm"}:
        sim = normalize_score(sim.reshape(sim.size(0), -1), tr).reshape_as(sim)
    return sim

def conditional_similarity_score(ctx: Any, params: Mapping[str, Any] | None = None) -> Tensor:
    return pairwise_token_similarity(ctx, params)

def redundancy_density_penalty(ctx: Any, params: Mapping[str, Any] | None = None) -> Tensor:
    params = params or {}
    sim = pairwise_token_similarity(ctx, params)
    density = (sim.sum(dim=-1) - 1.0) / max(sim.size(-1) - 1, 1)
    return normalize_score(-density, params.get("transform", "minmax"), _as_float(params.get("power", 1.0), 1.0))

def attention_proxy_saliency(ctx: Any, params: Mapping[str, Any] | None = None) -> Tensor:
    params = params or {}
    norm = visual_token_norm_saliency(ctx, {"transform": "minmax"})
    center = spatial_centrality_prior(ctx, {"transform": "minmax", "center_bias": params.get("center_bias", 1.0)})
    score = _as_float(params.get("norm_weight", 0.7), 0.7) * norm + _as_float(params.get("center_weight", 0.3), 0.3) * center
    return normalize_score(score, params.get("transform", "minmax"), _as_float(params.get("power", 1.0), 1.0))

def cdpruner_default_soft_score(ctx: Any, params: Mapping[str, Any] | None = None) -> Tensor:
    params = params or {}
    q = instruction_guided_visual_relevance(ctx, {"sign": params.get("sign", "negative_mean"), "transform": params.get("quality_transform", "minmax"), "power": params.get("quality_power", 1.0)})
    nonred = redundancy_density_penalty(ctx, {"transform": params.get("redundancy_transform", "minmax")})
    score = _as_float(params.get("quality_weight", 0.75), 0.75) * q + _as_float(params.get("diversity_weight", 0.25), 0.25) * nonred
    return normalize_score(score, params.get("transform", "minmax"))

def _extract_tensor_scores(scores: Mapping[str, Any]) -> Dict[str, Tensor]:
    return {k: v.float() for k, v in scores.items() if torch.is_tensor(v) and v.dim() == 2}

def full_token_score_fusion(scores: Mapping[str, Any], params: Mapping[str, Any] | None = None) -> Dict[str, Tensor]:
    params = params or {}
    ts = _extract_tensor_scores(scores)
    if not ts:
        raise ValueError("full_token_score_fusion requires score tensors")
    keys = list(ts.keys())
    wc = params.get("weights", None)
    if isinstance(wc, Mapping):
        weights = [float(wc.get(k, 0.0)) for k in keys]
    elif isinstance(wc, Sequence) and not isinstance(wc, (str, bytes)):
        weights = [float(x) for x in wc]
        weights = (weights + [1.0] * len(keys))[:len(keys)]
    else:
        weights = [1.0 / len(keys)] * len(keys)
    total = sum(abs(w) for w in weights) or 1.0
    weights = [w / total for w in weights]
    norm = params.get("score_normalization", "minmax")
    fused = None
    for k, w in zip(keys, weights):
        x = normalize_score(ts[k], norm)
        fused = w*x if fused is None else fused + w*x
    sim = next((v.float() for v in scores.values() if torch.is_tensor(v) and v.dim() == 3), None)
    if sim is None:
        b, n = fused.shape
        sim = torch.eye(n, device=fused.device).unsqueeze(0).expand(b, n, n)
    return {"score": normalize_score(fused, params.get("output_transform", "minmax")), "similarity": sim}

def weighted_sum_score_fusion(scores: Mapping[str, Any], params: Mapping[str, Any] | None = None) -> Tensor:
    return full_token_score_fusion(scores, params)["score"]

def quality_diversity_kernel(scores: Mapping[str, Any], params: Mapping[str, Any] | None = None) -> Dict[str, Tensor]:
    params = params or {}
    quality = None
    for key in ["quality", "relevance", "cdpruner_prior", "baseline"]:
        v = scores.get(key)
        if torch.is_tensor(v) and v.dim() == 2:
            quality = v.float(); break
    if quality is None:
        ts = _extract_tensor_scores(scores)
        if not ts: raise ValueError("quality_diversity_kernel requires quality")
        quality = next(iter(ts.values()))
    sim = None
    for key in ["similarity", "pairwise_similarity", "conditional_similarity"]:
        v = scores.get(key)
        if torch.is_tensor(v) and v.dim() == 3:
            sim = v.float(); break
    if sim is None:
        b, n = quality.shape
        sim = torch.eye(n, device=quality.device).unsqueeze(0).expand(b, n, n)
    q = torch.clamp(normalize_score(quality, "minmax"), min=1e-6).pow(_as_float(params.get("quality_power", 1.0), 1.0))
    s = torch.clamp((sim + 1.0) / 2.0, min=1e-6).pow(_as_float(params.get("similarity_power", 1.0), 1.0))
    kernel = q[:, :, None] * s * q[:, None, :]
    if params.get("symmetrize", True):
        kernel = 0.5 * (kernel + kernel.transpose(1, 2))
    return {"score": q, "similarity": sim, "kernel": kernel}

def _make_bool_mask(indices: Tensor, n: int) -> Tensor:
    mask = torch.zeros((indices.size(0), n), device=indices.device, dtype=torch.bool)
    mask.scatter_(1, indices.clamp(0, n-1), True)
    return mask

def _get_score_and_sim(fused):
    """Extract score / similarity / kernel from fused output.

    Do not use `tensor_a or tensor_b`, because PyTorch tensors with more
    than one element cannot be converted to bool.
    """
    if torch.is_tensor(fused):
        return fused.float(), None, None

    if not isinstance(fused, Mapping):
        raise TypeError(f"fused must be Tensor or Mapping, got {type(fused)!r}")

    score = fused.get("score", None)
    if score is None:
        score = fused.get("quality", None)
    if score is None:
        raise ValueError("selector needs score")

    sim = fused.get("similarity", None)
    kernel = fused.get("kernel", None)

    return (
        score.float(),
        sim.float() if torch.is_tensor(sim) else None,
        kernel.float() if torch.is_tensor(kernel) else None,
    )

def topk_selector(fused: Union[Tensor, Mapping[str, Tensor]], k: int, params: Mapping[str, Any] | None = None) -> Tensor:
    score, _, _ = _get_score_and_sim(fused)
    k = max(1, min(int(k), score.size(1)))
    idx = torch.sort(torch.topk(score, k=k, dim=-1).indices, dim=-1).values
    return _make_bool_mask(idx, score.size(1))

def _greedy_dpp_single(score: Tensor, sim: Tensor, k: int, pool_size: int) -> Tensor:
    n = score.numel()
    pool = torch.topk(score, k=max(k, min(pool_size, n))).indices.tolist()
    selected = []
    if pool: selected.append(pool.pop(0))
    while len(selected) < k and pool:
        rem = torch.tensor(pool, device=score.device)
        sel = torch.tensor(selected, device=score.device)
        q = score[rem]
        ms = sim[rem][:, sel].max(dim=1).values if sel.numel() else torch.zeros_like(q)
        marginal = q * (1.0 - torch.clamp(ms, 0.0, 1.0))
        j = int(torch.argmax(marginal).item())
        selected.append(pool.pop(j))
    if len(selected) < k:
        for i in torch.argsort(score, descending=True).tolist():
            if i not in selected:
                selected.append(i)
                if len(selected) == k: break
    return torch.tensor(selected[:k], device=score.device, dtype=torch.long)

def full_token_pool_dpp_selector(fused: Union[Tensor, Mapping[str, Tensor]], k: int, params: Mapping[str, Any] | None = None) -> Tensor:
    params = params or {}
    score, sim, _ = _get_score_and_sim(fused)
    b, n = score.shape
    if sim is None:
        sim = torch.eye(n, device=score.device).unsqueeze(0).expand(b, n, n)
    else:
        sim = (sim + 1.0) / 2.0
    k = max(1, min(int(k), n))
    pool_size = _as_int(params.get("pool_size", round(_as_float(params.get("pool_multiplier", 4.0), 4.0) * k)), round(4.0 * k))
    idx = torch.stack([_greedy_dpp_single(score[i], sim[i], k, pool_size) for i in range(b)], dim=0)
    if params.get("postprocess", "sort_by_position") == "sort_by_position":
        idx = torch.sort(idx, dim=-1).values
    return _make_bool_mask(idx, n)

def dpp_selector(fused: Union[Tensor, Mapping[str, Tensor]], k: int, params: Mapping[str, Any] | None = None) -> Tensor:
    return full_token_pool_dpp_selector(fused, k, params)

def full_token_pool_mmr_selector(fused: Union[Tensor, Mapping[str, Tensor]], k: int, params: Mapping[str, Any] | None = None) -> Tensor:
    params = params or {}
    score, sim, _ = _get_score_and_sim(fused)
    b, n = score.shape
    if sim is None:
        sim = torch.eye(n, device=score.device).unsqueeze(0).expand(b, n, n)
    else:
        sim = (sim + 1.0) / 2.0
    k = max(1, min(int(k), n))
    lamb = _as_float(params.get("lambda", params.get("mmr_lambda", 0.8)), 0.8)
    pool_size = max(k, min(int(round(_as_float(params.get("pool_multiplier", 4.0), 4.0) * k)), n))
    outs = []
    for bi in range(b):
        pool = torch.topk(score[bi], k=pool_size).indices.tolist()
        selected = []
        if pool: selected.append(pool.pop(0))
        while len(selected) < k and pool:
            rem = torch.tensor(pool, device=score.device)
            sel = torch.tensor(selected, device=score.device)
            rel = score[bi, rem]
            red = sim[bi][rem][:, sel].max(dim=1).values if sel.numel() else torch.zeros_like(rel)
            mmr = lamb * rel - (1.0 - lamb) * red
            j = int(torch.argmax(mmr).item())
            selected.append(pool.pop(j))
        if len(selected) < k:
            for i in torch.argsort(score[bi], descending=True).tolist():
                if i not in selected:
                    selected.append(i)
                    if len(selected) == k: break
        outs.append(torch.tensor(selected[:k], device=score.device))
    idx = torch.stack(outs, dim=0).long()
    if params.get("postprocess", "sort_by_position") == "sort_by_position":
        idx = torch.sort(idx, dim=-1).values
    return _make_bool_mask(idx, n)

def stratified_full_token_selector(fused: Union[Tensor, Mapping[str, Tensor]], k: int, params: Mapping[str, Any] | None = None) -> Tensor:
    params = params or {}
    score, _, _ = _get_score_and_sim(fused)
    b, n = score.shape
    k = max(1, min(int(k), n))
    core_k = max(1, min(k, int(round(_as_float(params.get("core_ratio", 0.875), 0.875) * k))))
    core_idx = torch.topk(score, k=core_k, dim=-1).indices
    if core_k == k:
        return _make_bool_mask(torch.sort(core_idx, dim=-1).values, n)
    masks = []
    side = int(round(math.sqrt(n)))
    grid_size = _as_int(params.get("grid_size", 4), 4)
    for bi in range(b):
        selected = set(int(x) for x in core_idx[bi].tolist())
        candidates = torch.argsort(score[bi], descending=True).tolist()
        if side * side == n:
            taken = set()
            for idx in selected:
                y, x = divmod(idx, side)
                taken.add((min(grid_size-1, int(y*grid_size/side)), min(grid_size-1, int(x*grid_size/side))))
            for idx in candidates:
                if len(selected) >= k: break
                y, x = divmod(idx, side)
                cell = (min(grid_size-1, int(y*grid_size/side)), min(grid_size-1, int(x*grid_size/side)))
                if idx not in selected and cell not in taken:
                    selected.add(idx); taken.add(cell)
        for idx in candidates:
            if len(selected) >= k: break
            selected.add(idx)
        masks.append(_make_bool_mask(torch.tensor(sorted(selected)[:k], device=score.device).view(1, -1), n)[0])
    return torch.stack(masks, dim=0)

def compute_mask_stats(mask: Tensor, reference_mask: Optional[Tensor] = None, score: Optional[Tensor] = None) -> Dict[str, float]:
    out: Dict[str, float] = {"selected_tokens_avg": float(mask.sum(dim=1).float().mean().detach().cpu().item())}
    if reference_mask is not None and reference_mask.shape == mask.shape:
        inter = (mask & reference_mask).sum(dim=1).float()
        denom = reference_mask.sum(dim=1).clamp_min(1).float()
        out["overlap_with_reference_avg"] = float((inter/denom).mean().detach().cpu().item())
        out["changed_tokens_avg"] = float(((mask ^ reference_mask).sum(dim=1).float()/2.0).mean().detach().cpu().item())
    if score is not None and score.shape == mask.shape:
        vals = (score * mask.float()).sum(dim=1) / mask.sum(dim=1).clamp_min(1).float()
        out["selected_score_avg"] = float(vals.mean().detach().cpu().item())
    return out

def append_jsonl(path: str, row: Mapping[str, Any]) -> None:
    if not path: return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")

def reference_quota_full_token_mmr_selector(fused, k: int, params=None):
    """Full-token MMR with a CDPruner-reference quota.

    It first keeps a quota of high-score tokens from the CDPruner-like
    reference mask, then fills the remaining slots from all tokens by MMR.
    This is v5.1: full-token direct search with overlap-aware constraint.
    """
    params = params or {}
    score, sim, _ = _get_score_and_sim(fused)
    b, n = score.shape
    k = max(1, min(int(k), n))

    # Approximate reference mask from score itself if no explicit reference is available.
    # A stronger reference is injected by executor diagnostics, but selector only sees fused.
    # To keep engineering simple, use top-k of cdpruner-prior-dominated fused score as reference.
    ref_ratio = float(params.get("reference_keep_ratio", 0.625))
    ref_keep = int(round(ref_ratio * k))
    ref_keep = max(0, min(k, ref_keep))

    pool_mult = float(params.get("pool_multiplier", 4.0))
    pool_size = max(k, min(int(round(pool_mult * k)), n))
    lamb = float(params.get("mmr_lambda", 0.90))

    if sim is None:
        sim = torch.eye(n, device=score.device).unsqueeze(0).expand(b, n, n)
    else:
        sim = (sim + 1.0) / 2.0

    outs = []
    for bi in range(b):
        # Reference core: high-score stable tokens.
        ref_idx = torch.topk(score[bi], k=k).indices.tolist()
        selected = ref_idx[:ref_keep]

        # Full-token MMR fill.
        pool = torch.topk(score[bi], k=pool_size).indices.tolist()
        pool = [x for x in pool if x not in selected]

        while len(selected) < k and pool:
            rem = torch.tensor(pool, device=score.device, dtype=torch.long)
            sel = torch.tensor(selected, device=score.device, dtype=torch.long)
            rel = score[bi, rem]
            red = sim[bi][rem][:, sel].max(dim=1).values if sel.numel() else torch.zeros_like(rel)
            mmr = lamb * rel - (1.0 - lamb) * red
            j = int(torch.argmax(mmr).item())
            selected.append(pool.pop(j))

        if len(selected) < k:
            for idx in torch.argsort(score[bi], descending=True).tolist():
                if idx not in selected:
                    selected.append(idx)
                    if len(selected) == k:
                        break

        outs.append(torch.tensor(selected[:k], device=score.device, dtype=torch.long))

    idx = torch.stack(outs, dim=0)
    if params.get("postprocess", "sort_by_position") == "sort_by_position":
        idx = torch.sort(idx, dim=-1).values
    return _make_bool_mask(idx, n)


# ---------------------------------------------------------------------
# v6.2 true reference-aware prior
# ---------------------------------------------------------------------

def build_cdpruner_reference_mask(ctx: Any, k: int) -> Tensor:
    # CDPruner-like reference teacher mask used by v6.2.
    # Reference path: instruction relevance + similarity + quality_diversity_kernel + DPP.
    q = instruction_guided_visual_relevance(
        ctx,
        {"sign": "negative_mean", "transform": "minmax", "power": 1.0},
    )
    sim = conditional_similarity_score(
        ctx,
        {"similarity_metric": "cosine", "similarity_transform": "identity"},
    )
    fused = quality_diversity_kernel(
        {"quality": q, "similarity": sim},
        {"quality_power": 1.0, "similarity_power": 1.0, "symmetrize": True},
    )
    return dpp_selector(
        fused,
        k,
        {"pool_multiplier": 4.0, "postprocess": "sort_by_position"},
    )


def cdpruner_reference_mask_prior(ctx: Any, params: Mapping[str, Any] | None = None) -> Tensor:
    # Reference-mask prior as scalar [B,N] score.
    # It is not a hard anchor; it is just a scorer for full-token selectors.
    params = params or {}
    feats = get_image_features(ctx)
    n = feats.size(1)

    k = _as_int(
        params.get("keep_tokens", get_keep_tokens(ctx, default=n)),
        get_keep_tokens(ctx, default=n),
    )
    k = max(1, min(k, n))

    ref = build_cdpruner_reference_mask(ctx, k).float()

    selected_score = _as_float(params.get("selected_score", 1.0), 1.0)
    unselected_score = _as_float(params.get("unselected_score", 0.0), 0.0)
    prior = ref * selected_score + (1.0 - ref) * unselected_score

    blend_soft = _as_float(params.get("blend_soft_prior", 0.0), 0.0)
    if blend_soft > 0:
        soft = cdpruner_default_soft_score(ctx, {"transform": "minmax"})
        prior = (1.0 - blend_soft) * prior + blend_soft * soft

    transform = params.get("transform", "identity")
    if transform not in {"identity", "none"}:
        prior = normalize_score(prior, transform)

    return prior


# ---------------------------------------------------------------------
# v6.3 native teacher prior + bounded replacement selector
# ---------------------------------------------------------------------

def _native_teacher_mask_from_ctx(ctx: Any) -> Optional[Tensor]:
    v = _get_attr(
        ctx,
        ["native_reference_mask", "native_teacher_mask", "teacher_mask", "reference_mask"],
        None,
    )
    if v is None:
        meta = _get_attr(ctx, ["metadata"], None)
        if isinstance(meta, Mapping):
            for key in ["native_reference_mask", "native_teacher_mask", "teacher_mask", "reference_mask"]:
                if key in meta:
                    v = meta[key]
                    break
    if torch.is_tensor(v):
        if v.dim() == 3:
            v = v.squeeze(1)
        return v.bool()
    return None


def native_teacher_mask_prior(ctx: Any, params: Mapping[str, Any] | None = None) -> Tensor:
    params = params or {}
    feats = get_image_features(ctx)
    n = feats.size(1)

    k = _as_int(
        params.get("keep_tokens", get_keep_tokens(ctx, default=n)),
        get_keep_tokens(ctx, default=n),
    )
    k = max(1, min(k, n))

    ref = _native_teacher_mask_from_ctx(ctx)
    if ref is None:
        # selftest fallback; real eval should provide native_reference_mask from llava_arch.py
        if "build_cdpruner_reference_mask" in globals():
            ref = build_cdpruner_reference_mask(ctx, k)
        else:
            ref = topk_selector(cdpruner_default_soft_score(ctx, {"transform": "minmax"}), k, {})

    ref = ref.to(feats.device).bool()

    selected_score = _as_float(params.get("selected_score", 1.0), 1.0)
    unselected_score = _as_float(params.get("unselected_score", 0.0), 0.0)
    prior = ref.float() * selected_score + (~ref).float() * unselected_score

    blend_soft = _as_float(params.get("blend_soft_prior", 0.0), 0.0)
    if blend_soft > 0:
        soft = cdpruner_default_soft_score(ctx, {"transform": "minmax"})
        prior = (1.0 - blend_soft) * prior + blend_soft * soft

    transform = params.get("transform", "identity")
    if transform not in {"identity", "none"}:
        prior = normalize_score(prior, transform)

    return prior


# Override full_token_score_fusion so selector can access the native teacher mask.
def full_token_score_fusion(scores: Mapping[str, Any], params: Mapping[str, Any] | None = None) -> Dict[str, Tensor]:
    params = params or {}
    ts = _extract_tensor_scores(scores)
    if not ts:
        raise ValueError("full_token_score_fusion requires score tensors")

    keys = list(ts.keys())
    wc = params.get("weights", None)

    if isinstance(wc, Mapping):
        weights = [float(wc.get(k, 0.0)) for k in keys]
    elif isinstance(wc, Sequence) and not isinstance(wc, (str, bytes)):
        weights = [float(x) for x in wc]
        weights = (weights + [1.0] * len(keys))[:len(keys)]
    else:
        weights = [1.0 / len(keys)] * len(keys)

    total = sum(abs(w) for w in weights) or 1.0
    weights = [w / total for w in weights]

    norm = params.get("score_normalization", "minmax")
    fused = None
    for k, w in zip(keys, weights):
        x = normalize_score(ts[k], norm)
        fused = w * x if fused is None else fused + w * x

    sim = next((v.float() for v in scores.values() if torch.is_tensor(v) and v.dim() == 3), None)
    if sim is None:
        b, n = fused.shape
        sim = torch.eye(n, device=fused.device).unsqueeze(0).expand(b, n, n)

    out = {
        "score": normalize_score(fused, params.get("output_transform", "minmax")),
        "similarity": sim,
    }

    # Pass teacher mask to bounded selector.
    for key in ["native_teacher", "native_teacher_prior", "teacher_prior", "reference_prior"]:
        if key in ts:
            out["native_reference_mask"] = ts[key] > 0.5
            out["reference_mask"] = ts[key] > 0.5
            break

    return out


def native_bounded_replace_selector(
    fused: Union[Tensor, Mapping[str, Tensor]],
    k: int,
    params: Mapping[str, Any] | None = None,
) -> Tensor:
    params = params or {}
    score, sim, _ = _get_score_and_sim(fused)
    b, n = score.shape
    k = max(1, min(int(k), n))

    ref = None
    if isinstance(fused, Mapping):
        ref = fused.get("native_reference_mask", None)
        if ref is None:
            ref = fused.get("reference_mask", None)

    if ref is None:
        # fallback only for selftest
        ref = topk_selector(score, k, {})

    ref = ref.to(score.device).bool()

    replace_tokens = params.get("replace_tokens", None)
    if replace_tokens is not None:
        replace_tokens = max(0, min(int(replace_tokens), k))
        keep_teacher_tokens = k - replace_tokens
    else:
        keep_teacher_tokens = int(params.get("keep_teacher_tokens", max(1, k - 4)))
        keep_teacher_tokens = max(0, min(keep_teacher_tokens, k))
        replace_tokens = k - keep_teacher_tokens

    fill_from = str(params.get("fill_from", "non_teacher_first"))

    outs = []
    for bi in range(b):
        selected = set()

        ref_idx = torch.where(ref[bi])[0]
        if ref_idx.numel() > 0 and keep_teacher_tokens > 0:
            vals = score[bi, ref_idx]
            take = min(keep_teacher_tokens, ref_idx.numel())
            keep_idx = ref_idx[torch.topk(vals, k=take).indices].tolist()
            selected.update(int(x) for x in keep_idx)

        need = k - len(selected)
        if need > 0:
            cand = score[bi].clone()
            if selected:
                cand[list(selected)] = -float("inf")

            # Prefer non-native tokens for replacement so changed_tokens reflects replace_tokens.
            if fill_from == "non_teacher_first":
                non_ref = ~ref[bi]
                if int(non_ref.sum().item()) >= need:
                    cand[ref[bi]] = -float("inf")

            fill_idx = torch.topk(cand, k=need).indices.tolist()
            selected.update(int(x) for x in fill_idx)

        if len(selected) < k:
            for idx in torch.argsort(score[bi], descending=True).tolist():
                selected.add(int(idx))
                if len(selected) >= k:
                    break

        idx = torch.tensor(sorted(list(selected))[:k], device=score.device, dtype=torch.long)
        outs.append(idx)

    idx = torch.stack(outs, dim=0)
    return _make_bool_mask(idx, n)


# ---------------------------------------------------------------------
# v7 YAML atom evolution atoms
# ---------------------------------------------------------------------

def _v7_extract_tensor_scores(scores):
    if isinstance(scores, Mapping):
        return {
            k: v.float()
            for k, v in scores.items()
            if torch.is_tensor(v) and v.dim() == 2
        }
    if torch.is_tensor(scores) and scores.dim() == 2:
        return {"score": scores.float()}
    return {}


def _v7_rank_norm(x):
    idx = torch.argsort(torch.argsort(x, dim=-1), dim=-1).float()
    denom = max(1, x.size(-1) - 1)
    return idx / denom


def v7_rank_fusion(scores, params=None):
    params = params or {}
    ts = _v7_extract_tensor_scores(scores)
    if not ts:
        raise ValueError("v7_rank_fusion requires score tensors")

    weights_cfg = params.get("weights", {})
    keys = list(ts.keys())
    weights = []
    for k in keys:
        try:
            weights.append(float(weights_cfg.get(k, 1.0)))
        except Exception:
            weights.append(1.0)

    total = sum(abs(w) for w in weights) or 1.0
    weights = [w / total for w in weights]

    fused = None
    for k, w in zip(keys, weights):
        x = _v7_rank_norm(ts[k])
        fused = w * x if fused is None else fused + w * x

    sim = next(
        (v.float() for v in scores.values() if torch.is_tensor(v) and v.dim() == 3),
        None,
    ) if isinstance(scores, Mapping) else None

    if sim is None:
        b, n = fused.shape
        sim = torch.eye(n, device=fused.device).unsqueeze(0).expand(b, n, n)

    return {
        "score": normalize_score(fused, params.get("output_transform", "minmax")),
        "similarity": sim,
    }


def v7_score_product_fusion(scores, params=None):
    params = params or {}
    ts = _v7_extract_tensor_scores(scores)
    if not ts:
        raise ValueError("v7_score_product_fusion requires score tensors")

    product_keys = params.get("product_keys", None)
    if not isinstance(product_keys, (list, tuple)) or not product_keys:
        product_keys = [
            k for k in ["relevance", "native_teacher", "norm", "redundancy", "spatial", "attn"]
            if k in ts
        ]
        if not product_keys:
            product_keys = list(ts.keys())

    eps = _as_float(params.get("eps", 1e-4), 1e-4)
    powers = params.get("powers", {})
    fused = None

    for k in product_keys:
        if k not in ts:
            continue
        x = torch.clamp(
            normalize_score(ts[k], params.get("score_normalization", "minmax")),
            eps,
            1.0,
        )
        power = _as_float(
            powers.get(k, 1.0) if isinstance(powers, Mapping) else 1.0,
            1.0,
        )
        x = x.pow(power)
        fused = x if fused is None else fused * x

    if fused is None:
        fused = next(iter(ts.values())).float()

    add_weights = params.get("add_weights", {})
    if isinstance(add_weights, Mapping):
        add = 0
        add_total = 0.0
        for k, w in add_weights.items():
            if k in ts:
                wf = _as_float(w, 0.0)
                add = add + wf * normalize_score(ts[k], params.get("score_normalization", "minmax"))
                add_total += abs(wf)
        if add_total > 0:
            mix = _as_float(params.get("add_mix", 0.2), 0.2)
            fused = (1.0 - mix) * fused + mix * (add / add_total)

    sim = next(
        (v.float() for v in scores.values() if torch.is_tensor(v) and v.dim() == 3),
        None,
    ) if isinstance(scores, Mapping) else None

    if sim is None:
        b, n = fused.shape
        sim = torch.eye(n, device=fused.device).unsqueeze(0).expand(b, n, n)

    return {
        "score": normalize_score(fused, params.get("output_transform", "minmax")),
        "similarity": sim,
    }


def v7_teacher_residual_fusion(scores, params=None):
    params = params or {}
    ts = _v7_extract_tensor_scores(scores)
    if not ts:
        raise ValueError("v7_teacher_residual_fusion requires score tensors")

    teacher_key = params.get("teacher_key", "native_teacher")
    if teacher_key not in ts:
        teacher_key = "reference_prior" if "reference_prior" in ts else None

    base_tensor = next(iter(ts.values()))
    if teacher_key in ts:
        teacher = normalize_score(ts[teacher_key], "identity")
    else:
        teacher = torch.zeros_like(base_tensor)

    residual_keys = params.get("residual_keys", [k for k in ts.keys() if k != teacher_key])
    residual = None
    valid_residual = 0

    for k in residual_keys:
        if k in ts:
            x = normalize_score(ts[k], params.get("score_normalization", "minmax"))
            residual = x if residual is None else residual + x
            valid_residual += 1

    if residual is None:
        residual = torch.zeros_like(teacher)
    else:
        residual = residual / max(1, valid_residual)

    tw = _as_float(params.get("teacher_weight", 0.25), 0.25)
    rw = _as_float(params.get("residual_weight", 0.75), 0.75)
    fused = tw * teacher + rw * residual

    sim = next(
        (v.float() for v in scores.values() if torch.is_tensor(v) and v.dim() == 3),
        None,
    ) if isinstance(scores, Mapping) else None

    if sim is None:
        b, n = fused.shape
        sim = torch.eye(n, device=fused.device).unsqueeze(0).expand(b, n, n)

    return {
        "score": normalize_score(fused, params.get("output_transform", "minmax")),
        "similarity": sim,
    }


def v7_conditional_dpp_selector(fused, k, params=None):
    params = params or {}
    score, sim, _ = _get_score_and_sim(fused)
    b, n = score.shape
    k = max(1, min(int(k), n))

    if sim is None:
        sim = torch.eye(n, device=score.device).unsqueeze(0).expand(b, n, n)

    q = torch.clamp(
        normalize_score(score, params.get("quality_transform", "minmax")),
        min=1e-6,
    )
    q = q.pow(_as_float(params.get("quality_power", 1.0), 1.0))

    s = sim.float()
    if bool(params.get("shift_similarity", False)):
        s = (s + 1.0) / 2.0
    s = torch.clamp(s, min=1e-6)

    kernel = q[:, :, None] * s * q[:, None, :]
    if bool(params.get("symmetrize", True)):
        kernel = 0.5 * (kernel + kernel.transpose(1, 2))

    cis = torch.zeros((k, b, n), device=score.device)
    di2s = torch.diagonal(kernel, dim1=1, dim2=2).clone()
    select_idx = torch.empty((k, b), dtype=torch.long, device=score.device)
    batch_idx = torch.arange(b, device=score.device)

    for i in range(k):
        j = torch.argmax(di2s, dim=-1)
        select_idx[i] = j

        denom = torch.sqrt(torch.clamp(di2s[batch_idx, j], min=1e-8)).unsqueeze(-1)
        eis = (
            kernel[batch_idx, j]
            - torch.einsum("tb,tbn->bn", cis[:i, batch_idx, j], cis[:i])
        ) / denom

        cis[i, :, :] = eis
        di2s -= torch.square(eis)
        di2s[batch_idx, j] = -float("inf")

    idx = torch.sort(select_idx.t()).values
    return _make_bool_mask(idx, n)


def v7_two_stage_pool_selector(fused, k, params=None):
    params = params or {}
    score, sim, _ = _get_score_and_sim(fused)
    b, n = score.shape
    k = max(1, min(int(k), n))

    pool_multiplier = _as_float(params.get("pool_multiplier", 3.0), 3.0)
    pool_size = max(k, min(int(round(pool_multiplier * k)), n))
    pool_idx = torch.topk(score, k=pool_size, dim=-1).indices

    masks = []
    method = str(params.get("stage2", "dpp"))

    for bi in range(b):
        sub_score = score[bi:bi+1, pool_idx[bi]]

        if sim is not None:
            sub_sim = sim[bi:bi+1][:, pool_idx[bi]][:, :, pool_idx[bi]]
        else:
            sub_sim = torch.eye(pool_size, device=score.device).unsqueeze(0)

        sub_fused = {"score": sub_score, "similarity": sub_sim}

        if method == "mmr":
            sub_mask = full_token_pool_mmr_selector(
                sub_fused,
                k,
                {
                    "pool_multiplier": 1.0,
                    "mmr_lambda": params.get("mmr_lambda", 0.85),
                },
            )
        elif method == "topk":
            sub_mask = topk_selector(sub_fused, k, {})
        else:
            sub_mask = v7_conditional_dpp_selector(sub_fused, k, params)

        chosen_local = torch.where(sub_mask[0])[0]
        chosen_global = pool_idx[bi, chosen_local]

        m = torch.zeros(n, device=score.device, dtype=torch.bool)
        m[chosen_global] = True
        masks.append(m)

    return torch.stack(masks, dim=0)
