from __future__ import annotations

from typing import Dict, Optional, Tuple
import torch
import torch.nn.functional as F


def normalize_score(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x = x.float()
    mn = x.amin(dim=-1, keepdim=True)
    mx = x.amax(dim=-1, keepdim=True)
    return (x - mn) / (mx - mn + eps)


def local_contrast_saliency(x: torch.Tensor, grid_hw: Optional[Tuple[int, int]] = None) -> torch.Tensor:
    b, n, d = x.shape
    x_norm = F.normalize(x.float(), dim=-1)

    if grid_hw is None:
        side = int(round(n ** 0.5))
        if side * side == n:
            grid_hw = (side, side)

    if grid_hw is not None:
        h, w = grid_hw
        if h * w == n:
            feat = x_norm.reshape(b, h, w, d)
            neigh = []
            for dh, dw in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                neigh.append(torch.roll(feat, shifts=(dh, dw), dims=(1, 2)))
            mean_neigh = torch.stack(neigh, dim=0).mean(dim=0)
            score = torch.norm(feat - mean_neigh, dim=-1).reshape(b, n)
            return normalize_score(score)

    sim = torch.matmul(x_norm, x_norm.transpose(-1, -2))
    k = min(8, n)
    _, idx = sim.topk(k=k, dim=-1)
    gather_idx = idx.unsqueeze(-1).expand(-1, -1, -1, d)
    neigh = torch.gather(x_norm.unsqueeze(1).expand(-1, n, -1, -1), 2, gather_idx)
    mean_neigh = neigh.mean(dim=2)
    score = torch.norm(x_norm - mean_neigh, dim=-1)
    return normalize_score(score)


def pairwise_cosine_kernel(x: torch.Tensor) -> torch.Tensor:
    x_norm = F.normalize(x.float(), dim=-1)
    k = torch.matmul(x_norm, x_norm.transpose(-1, -2))
    k = (k + 1.0) * 0.5
    return k.clamp(0, 1)


def feature_cluster_partition(x: torch.Tensor, num_clusters: int = 8):
    b, n, d = x.shape
    c = max(1, min(int(num_clusters), n))
    x_norm = F.normalize(x.float(), dim=-1)

    centers = torch.zeros(b, c, dtype=torch.long, device=x.device)
    first = torch.norm(x.float(), dim=-1).argmax(dim=-1)
    centers[:, 0] = first

    dist = torch.ones(b, n, device=x.device) * 1e9
    for j in range(1, c):
        prev = centers[:, j - 1]
        prev_feat = x_norm[torch.arange(b, device=x.device), prev].unsqueeze(1)
        sim = (x_norm * prev_feat).sum(dim=-1)
        dcur = 1.0 - sim
        dist = torch.minimum(dist, dcur)
        centers[:, j] = dist.argmax(dim=-1)

    center_feat = x_norm[torch.arange(b, device=x.device).unsqueeze(-1), centers]
    sim_to_centers = torch.matmul(x_norm, center_feat.transpose(-1, -2))
    cluster_id = sim_to_centers.argmax(dim=-1)
    return cluster_id, centers


def cluster_centroid_representativeness(x: torch.Tensor, cluster_id: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    b, n, d = x.shape
    x_norm = F.normalize(x.float(), dim=-1)
    center_feat = x_norm[torch.arange(b, device=x.device).unsqueeze(-1), centers]
    assigned = torch.gather(
        center_feat,
        1,
        cluster_id.unsqueeze(-1).expand(-1, -1, d),
    )
    score = (x_norm * assigned).sum(dim=-1)
    return normalize_score(score)


def spatial_grid_ids(n: int, device, grid_hw=(4, 4)) -> torch.Tensor:
    gh, gw = grid_hw
    side = int(round(n ** 0.5))
    if side * side != n:
        return torch.arange(n, device=device) % (gh * gw)

    yy, xx = torch.meshgrid(
        torch.arange(side, device=device),
        torch.arange(side, device=device),
        indexing="ij",
    )
    gy = torch.clamp((yy.float() / side * gh).long(), 0, gh - 1)
    gx = torch.clamp((xx.float() / side * gw).long(), 0, gw - 1)
    return (gy * gw + gx).reshape(-1)


def build_multi_source_pool(scores: Dict[str, torch.Tensor], pool_size: int) -> torch.Tensor:
    names = list(scores.keys())
    if not names:
        raise ValueError("empty scores for pool")

    b, n = next(iter(scores.values())).shape
    p = max(1, min(int(pool_size), n))
    per = max(1, p // len(names))

    all_idx = []
    for name in names:
        s = scores[name]
        _, idx = s.topk(k=min(per, n), dim=-1)
        all_idx.append(idx)

    avg = torch.stack([scores[k] for k in names], dim=0).mean(dim=0)
    _, extra = avg.topk(k=p, dim=-1)
    cat = torch.cat(all_idx + [extra], dim=-1)

    out = []
    for bi in range(b):
        seen = []
        used = set()
        for v in cat[bi].tolist():
            if v not in used:
                used.add(v)
                seen.append(v)
            if len(seen) >= p:
                break
        out.append(torch.tensor(seen[:p], dtype=torch.long, device=cat.device))
    return torch.stack(out, dim=0)


def constrained_pool_dpp_select(
    quality: torch.Tensor,
    kernel: torch.Tensor,
    pool: torch.Tensor,
    keep_tokens: int,
    grid_ids: Optional[torch.Tensor] = None,
    cluster_id: Optional[torch.Tensor] = None,
    min_per_grid: int = 0,
    min_per_cluster: int = 0,
    diversity_lambda: float = 0.35,
) -> torch.Tensor:
    b, n = quality.shape
    k_keep = max(1, min(int(keep_tokens), n))
    selected_all = []

    for bi in range(b):
        q = quality[bi]
        Kmat = kernel[bi]
        cand_set = list(dict.fromkeys(pool[bi].tolist()))
        selected = []
        used = set()

        def add_best_from_group(group_indices, limit):
            group = [i for i in group_indices if i in cand_set and i not in used]
            group = sorted(group, key=lambda i: float(q[i]), reverse=True)
            for i in group[:limit]:
                if len(selected) < k_keep and i not in used:
                    selected.append(i)
                    used.add(i)

        if grid_ids is not None and min_per_grid > 0:
            gids = grid_ids.tolist()
            for g in sorted(set(gids)):
                group = [i for i, gg in enumerate(gids) if gg == g]
                add_best_from_group(group, min_per_grid)

        if cluster_id is not None and min_per_cluster > 0:
            cids = cluster_id[bi].tolist()
            for c in sorted(set(cids)):
                group = [i for i, cc in enumerate(cids) if cc == c]
                add_best_from_group(group, min_per_cluster)

        while len(selected) < k_keep:
            best_i = None
            best_val = -1e9
            for i in cand_set:
                if i in used:
                    continue
                if selected:
                    sim_pen = Kmat[i, torch.tensor(selected, device=quality.device)].max()
                else:
                    sim_pen = torch.tensor(0.0, device=quality.device)
                val = float(q[i] - diversity_lambda * sim_pen)
                if val > best_val:
                    best_val = val
                    best_i = i
            if best_i is None:
                break
            selected.append(best_i)
            used.add(best_i)

        if len(selected) < k_keep:
            for i in torch.argsort(q, descending=True).tolist():
                if i not in used:
                    selected.append(i)
                    used.add(i)
                if len(selected) >= k_keep:
                    break

        selected_all.append(torch.tensor(selected[:k_keep], dtype=torch.long, device=quality.device))

    return torch.stack(selected_all, dim=0)


def anchor_controlled_pool_dpp_select(
    quality: torch.Tensor,
    kernel: torch.Tensor,
    pool: torch.Tensor,
    reference_score: torch.Tensor,
    keep_tokens: int,
    anchor_quota: int = 8,
    diversity_lambda: float = 0.25,
    allow_anchor_reorder: bool = True,
) -> torch.Tensor:
    """
    Anchor-controlled greedy selector.

    quality: [B, N]
    kernel: [B, N, N]
    pool: [B, P]
    reference_score: [B, N]
    output: selected indices [B, K]

    This selector first preserves top reference tokens, then fills the rest
    with quality-diversity greedy selection from the v9 candidate pool.
    """
    b, n = quality.shape
    k_keep = max(1, min(int(keep_tokens), n))
    anchor_quota = max(0, min(int(anchor_quota), k_keep))

    selected_all = []

    for bi in range(b):
        q = quality[bi]
        Kmat = kernel[bi]
        ref = reference_score[bi]
        cand_set = list(dict.fromkeys(pool[bi].tolist()))

        selected = []
        used = set()

        if anchor_quota > 0:
            anchor_idx = torch.argsort(ref, descending=True).tolist()
            for i in anchor_idx:
                if i not in used:
                    selected.append(i)
                    used.add(i)
                if len(selected) >= anchor_quota:
                    break

        while len(selected) < k_keep:
            best_i = None
            best_val = -1e9

            for i in cand_set:
                if i in used:
                    continue

                if selected:
                    sim_pen = Kmat[i, torch.tensor(selected, device=quality.device)].max()
                else:
                    sim_pen = torch.tensor(0.0, device=quality.device)

                val = float(q[i] - diversity_lambda * sim_pen)

                if val > best_val:
                    best_val = val
                    best_i = i

            if best_i is None:
                break

            selected.append(best_i)
            used.add(best_i)

        if len(selected) < k_keep:
            for i in torch.argsort(q, descending=True).tolist():
                if i not in used:
                    selected.append(i)
                    used.add(i)
                if len(selected) >= k_keep:
                    break

        selected = selected[:k_keep]

        if allow_anchor_reorder:
            selected = sorted(selected)

        selected_all.append(torch.tensor(selected, dtype=torch.long, device=quality.device))

    return torch.stack(selected_all, dim=0)


def reference_mask_anchor_pool_dpp_select(
    quality: torch.Tensor,
    kernel: torch.Tensor,
    pool: torch.Tensor,
    reference_mask: torch.Tensor,
    keep_tokens: int,
    anchor_quota: int = 8,
    diversity_lambda: float = 0.25,
    exclude_unanchored_reference: bool = True,
    sort_selected: bool = True,
) -> torch.Tensor:
    """
    Runtime-reference anchored selector.

    Unlike anchor_controlled_pool_dpp_select, this uses an actual boolean
    reference mask, not a reference score. This makes overlap/changed
    explicitly controllable.

    quality: [B, N]
    kernel: [B, N, N]
    pool: [B, P]
    reference_mask: [B, N] bool
    output: selected indices [B, K]
    """
    b, n = quality.shape
    k_keep = max(1, min(int(keep_tokens), n))
    anchor_quota = max(0, min(int(anchor_quota), k_keep))

    if reference_mask.dtype != torch.bool:
        reference_mask = reference_mask.bool()

    selected_all = []

    for bi in range(b):
        q = quality[bi]
        Kmat = kernel[bi]
        ref = reference_mask[bi]
        cand_set = list(dict.fromkeys(pool[bi].tolist()))

        selected = []
        used = set()

        # 1. Preserve exactly anchor_quota tokens from the actual reference mask.
        ref_idx = torch.where(ref)[0].tolist()
        ref_idx = sorted(ref_idx, key=lambda i: float(q[i]), reverse=True)

        for i in ref_idx:
            if i not in used:
                selected.append(i)
                used.add(i)
            if len(selected) >= anchor_quota:
                break

        # 2. Fill remaining slots from v9 pool.
        # Optionally avoid selecting extra reference tokens, so overlap is controlled.
        while len(selected) < k_keep:
            best_i = None
            best_val = -1e9

            for i in cand_set:
                if i in used:
                    continue

                if exclude_unanchored_reference and bool(ref[i]):
                    continue

                if selected:
                    sim_pen = Kmat[i, torch.tensor(selected, device=quality.device)].max()
                else:
                    sim_pen = torch.tensor(0.0, device=quality.device)

                val = float(q[i] - diversity_lambda * sim_pen)

                if val > best_val:
                    best_val = val
                    best_i = i

            if best_i is None:
                break

            selected.append(best_i)
            used.add(best_i)

        # 3. Fallback fill if pool is insufficient.
        if len(selected) < k_keep:
            for i in torch.argsort(q, descending=True).tolist():
                if i in used:
                    continue
                if exclude_unanchored_reference and bool(ref[i]) and len(selected) >= anchor_quota:
                    continue
                selected.append(i)
                used.add(i)
                if len(selected) >= k_keep:
                    break

        # 4. Final emergency fill.
        if len(selected) < k_keep:
            for i in torch.argsort(q, descending=True).tolist():
                if i not in used:
                    selected.append(i)
                    used.add(i)
                if len(selected) >= k_keep:
                    break

        selected = selected[:k_keep]

        if sort_selected:
            selected = sorted(selected)

        selected_all.append(torch.tensor(selected, dtype=torch.long, device=quality.device))

    return torch.stack(selected_all, dim=0)


def weighted_product_fusion(
    scores: Dict[str, torch.Tensor],
    weights: Dict[str, float],
    eps: float = 1e-4,
) -> torch.Tensor:
    """
    Weighted geometric/product fusion.
    Each score is normalized to [0,1], then fused in log space.

    output: [B, N]
    """
    if not scores:
        raise ValueError("weighted_product_fusion requires non-empty scores")

    acc = None
    total_w = 0.0

    for name, score in scores.items():
        s = normalize_score(score).clamp(min=eps, max=1.0)
        w = float(weights.get(name, 1.0))
        if abs(w) <= 0:
            continue
        term = w * torch.log(s)
        acc = term if acc is None else acc + term
        total_w += abs(w)

    if acc is None or total_w <= 0:
        first = next(iter(scores.values()))
        return normalize_score(first)

    out = torch.exp(acc / total_w)
    return normalize_score(out)


def build_quality_pool(
    quality: torch.Tensor,
    pool_size: int,
) -> torch.Tensor:
    """
    Build pool only from top quality tokens.

    quality: [B, N]
    output: [B, P]
    """
    b, n = quality.shape
    p = max(1, min(int(pool_size), n))
    _, idx = quality.topk(k=p, dim=-1)
    return idx


# ============================================================
# v11 Route-B primitives: conditional fusion + coverage + guard
# ============================================================

def _v11_minmax_score(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x = x.float()
    mn = x.min(dim=1, keepdim=True).values
    mx = x.max(dim=1, keepdim=True).values
    return (x - mn) / (mx - mn + eps)


def _v11_weighted_product(scores: Dict[str, torch.Tensor], weights: Dict[str, float], eps: float = 1e-6) -> torch.Tensor:
    if not scores:
        raise ValueError("_v11_weighted_product requires non-empty scores")

    first = next(iter(scores.values()))
    out = torch.ones_like(first, dtype=torch.float32)

    used = False
    for k, s in scores.items():
        w = float(weights.get(k, 0.0))
        if w <= 0:
            continue
        ss = _v11_minmax_score(s).clamp(min=eps, max=1.0)
        out = out * (ss ** w)
        used = True

    if not used:
        # Fallback to uniform average when all weights are zero/missing.
        vals = [_v11_minmax_score(v) for v in scores.values()]
        out = torch.stack(vals, dim=0).mean(dim=0)

    return _v11_minmax_score(out)


def reference_confidence_weighted_product_fusion(
    scores: Dict[str, torch.Tensor],
    reference_score: torch.Tensor = None,
    params: Dict[str, object] = None,
) -> torch.Tensor:
    """Reference-confidence-aware product fusion.

    Motivation:
      - When the reference mask/prior is reliable, preserve a conservative
        cognition-friendly weighting.
      - When reference confidence is weaker, use a slightly more perception-oriented
        weighting, but without aggressive semantic/contrast boosts.

    This is still fully deterministic and input-conditional.
    """
    params = params or {}

    conservative_weights = params.get("conservative_weights", None) or {
        "semantic": 0.34,
        "attn": 0.10,
        "spatial": 0.15,
        "redundancy": 0.34,
        "contrast": 0.07,
    }
    perception_weights = params.get("perception_weights", None) or {
        "semantic": 0.37,
        "attn": 0.10,
        "spatial": 0.13,
        "redundancy": 0.30,
        "contrast": 0.10,
    }

    if reference_score is None:
        return _v11_weighted_product(scores, conservative_weights)

    ref = _v11_minmax_score(reference_score.float())
    bsz, n = ref.shape

    topk = max(1, min(n, int(params.get("confidence_topk", 32))))
    ref_conf = ref.topk(k=topk, dim=1).values.mean(dim=1, keepdim=True)

    threshold = float(params.get("confidence_threshold", 0.55))
    temperature = float(params.get("confidence_temperature", 0.08))

    # gate close to 1.0 means use perception weights.
    # lower reference confidence => more perception-oriented.
    gate = torch.sigmoid((threshold - ref_conf) / max(temperature, 1e-6))

    conservative = _v11_weighted_product(scores, conservative_weights)
    perception = _v11_weighted_product(scores, perception_weights)

    return _v11_minmax_score((1.0 - gate) * conservative + gate * perception)


def spatial_coverage_pool_builder(
    quality: torch.Tensor,
    spatial: torch.Tensor = None,
    pool_size: int = 96,
    grid_size: int = 4,
    min_tokens_per_bin: int = 1,
    high_quality_ratio: float = 0.75,
) -> torch.Tensor:
    """Build a candidate pool with both quality and spatial coverage.

    Returns:
      LongTensor [B, pool_size]
    """
    quality = quality.float()
    bsz, n = quality.shape
    pool_size = int(max(1, min(pool_size, n)))
    grid_size = int(max(1, grid_size))
    min_tokens_per_bin = int(max(0, min_tokens_per_bin))
    high_quality_ratio = float(max(0.0, min(1.0, high_quality_ratio)))

    # If token layout is not square, fallback to simple top-k.
    side = int(round(n ** 0.5))
    if side * side != n:
        return quality.topk(pool_size, dim=1).indices

    coords = torch.arange(n, device=quality.device)
    row = coords // side
    col = coords % side
    bin_r = torch.clamp((row * grid_size) // side, 0, grid_size - 1)
    bin_c = torch.clamp((col * grid_size) // side, 0, grid_size - 1)
    bins = bin_r * grid_size + bin_c
    num_bins = grid_size * grid_size

    out_all = []
    high_k = max(1, min(pool_size, int(round(pool_size * high_quality_ratio))))

    for b in range(bsz):
        used = set()
        chosen = []

        # 1) high-quality main pool
        for idx in quality[b].topk(high_k).indices.tolist():
            if idx not in used:
                chosen.append(idx)
                used.add(idx)
            if len(chosen) >= pool_size:
                break

        # 2) coverage quota by spatial bin
        for bb in range(num_bins):
            if len(chosen) >= pool_size:
                break
            mask = (bins == bb)
            idxs = torch.where(mask)[0]
            if idxs.numel() == 0:
                continue
            vals = quality[b, idxs]
            order = idxs[vals.argsort(descending=True)]
            take = 0
            for idx in order.tolist():
                if idx not in used:
                    chosen.append(idx)
                    used.add(idx)
                    take += 1
                if take >= min_tokens_per_bin or len(chosen) >= pool_size:
                    break

        # 3) fill remaining by quality
        if len(chosen) < pool_size:
            for idx in quality[b].argsort(descending=True).tolist():
                if idx not in used:
                    chosen.append(idx)
                    used.add(idx)
                if len(chosen) >= pool_size:
                    break

        out_all.append(torch.tensor(chosen[:pool_size], dtype=torch.long, device=quality.device))

    return torch.stack(out_all, dim=0)


def adaptive_reference_anchor_pool_dpp_select(
    quality: torch.Tensor,
    kernel: torch.Tensor,
    pool: torch.Tensor,
    reference_mask: torch.Tensor,
    k_keep: int,
    min_anchor_quota: int = 6,
    default_anchor_quota: int = 8,
    max_anchor_quota: int = 10,
    diversity_lambda: float = 0.18,
    confidence_threshold_low: float = 0.45,
    confidence_threshold_high: float = 0.62,
    exclude_unanchored_reference: bool = True,
    sort_selected: bool = True,
) -> torch.Tensor:
    """Adaptive reference-anchored DPP selector.

    The anchor quota is selected per sample:
      weak reference confidence  -> min_anchor_quota
      medium confidence          -> default_anchor_quota
      strong reference confidence -> max_anchor_quota
    """
    quality = quality.float()
    bsz, n = quality.shape
    k_keep = int(k_keep)
    min_anchor_quota = int(min_anchor_quota)
    default_anchor_quota = int(default_anchor_quota)
    max_anchor_quota = int(max_anchor_quota)

    ref_mask = reference_mask.bool()
    selected_all = []

    qn = _v11_minmax_score(quality)

    for b in range(bsz):
        q = quality[b]
        q01 = qn[b]
        Kmat = kernel[b]
        cand_pool = pool[b].long().tolist()
        ref = ref_mask[b]

        ref_idxs = torch.where(ref)[0]
        if ref_idxs.numel() > 0:
            ref_conf = q01[ref_idxs].topk(min(int(k_keep), ref_idxs.numel())).values.mean().item()
        else:
            ref_conf = 0.0

        if ref_conf >= confidence_threshold_high:
            anchor_quota = max_anchor_quota
        elif ref_conf <= confidence_threshold_low:
            anchor_quota = min_anchor_quota
        else:
            anchor_quota = default_anchor_quota

        anchor_quota = max(0, min(anchor_quota, k_keep))

        selected = []
        used = set()

        # 1) preserve strongest reference tokens first
        if ref_idxs.numel() > 0 and anchor_quota > 0:
            ref_scores = q[ref_idxs]
            order = ref_idxs[ref_scores.argsort(descending=True)]
            for idx in order.tolist():
                if idx not in used:
                    selected.append(idx)
                    used.add(idx)
                if len(selected) >= anchor_quota:
                    break

        # 2) quality-diversity fill from pool
        while len(selected) < k_keep:
            best_i = None
            best_score = None

            for i in cand_pool:
                if i in used:
                    continue
                if exclude_unanchored_reference and bool(ref[i]) and len(selected) >= anchor_quota:
                    continue

                score = q[i]
                if selected:
                    sel_t = torch.tensor(selected, dtype=torch.long, device=quality.device)
                    sim_pen = Kmat[i, sel_t].max()
                    score = score - float(diversity_lambda) * sim_pen

                if best_score is None or score > best_score:
                    best_score = score
                    best_i = i

            if best_i is None:
                break

            selected.append(best_i)
            used.add(best_i)

        # 3) robust fallback fill
        if len(selected) < k_keep:
            for i in q.argsort(descending=True).tolist():
                if i not in used:
                    selected.append(i)
                    used.add(i)
                if len(selected) >= k_keep:
                    break

        selected = selected[:k_keep]
        if sort_selected:
            selected = sorted(selected)

        selected_all.append(torch.tensor(selected, dtype=torch.long, device=quality.device))

    return torch.stack(selected_all, dim=0)


# ============================================================
# v12 Base-preserving residual exchange selector
# ============================================================

def _v12_to_reference_mask(
    reference_score: torch.Tensor,
    k_keep: int,
    n_tokens: int = None,
    threshold: float = 0.5,
) -> torch.Tensor:
    """Convert reference score / bool mask / index tensor to bool mask [B, N]."""
    ref = reference_score
    if ref is None:
        raise ValueError("_v12_to_reference_mask requires reference_score")

    if ref.dtype == torch.bool and ref.dim() == 2:
        return ref.bool()

    if ref.dim() != 2:
        raise ValueError(f"bad reference_score shape: {tuple(ref.shape)}")

    bsz, n = ref.shape
    if n_tokens is None:
        n_tokens = n

    # If shape is [B, N], interpret as score and take top-K.
    if n == n_tokens:
        rr = ref.float()
        kk = max(1, min(int(k_keep), n_tokens))
        idx = rr.topk(k=kk, dim=1).indices
        mask = torch.zeros((bsz, n_tokens), dtype=torch.bool, device=ref.device)
        mask.scatter_(1, idx.long(), True)
        return mask

    # Otherwise interpret as index tensor [B, K].
    idx = ref.long().clamp(min=0, max=n_tokens - 1)
    mask = torch.zeros((bsz, n_tokens), dtype=torch.bool, device=ref.device)
    mask.scatter_(1, idx, True)
    return mask


def native_reference_residual_exchange_select(
    quality: torch.Tensor,
    reference_score: torch.Tensor,
    kernel: torch.Tensor = None,
    k_keep: int = 32,
    replace_quota: int = 4,
    min_reference_keep: int = None,
    diversity_lambda: float = 0.08,
    reference_keep_weight: float = 0.20,
    candidate_quality_power: float = 1.0,
    sort_selected: bool = True,
) -> torch.Tensor:
    """Base-preserving residual exchange selector.

    It first constructs the CDPruner/native reference mask, keeps most reference
    tokens, and only replaces a small number of low-quality reference tokens with
    high-quality non-reference tokens.

    Args:
      quality: [B, N] candidate quality score.
      reference_score: [B, N] reference prior or bool mask.
      kernel: optional [B, N, N] similarity kernel for diversity penalty.
      k_keep: number of selected tokens.
      replace_quota: number of non-reference tokens to add, usually 0/2/4/6/8.
      min_reference_keep: lower bound on reference tokens kept.
      diversity_lambda: diversity penalty among added tokens and kept tokens.
      reference_keep_weight: how strongly to keep high-reference-score tokens.
      candidate_quality_power: power transform for candidate quality.
      sort_selected: whether to sort final indices.
    """
    quality = quality.float()
    bsz, n = quality.shape
    k_keep = int(max(1, min(k_keep, n)))
    replace_quota = int(max(0, min(replace_quota, k_keep)))

    if min_reference_keep is None:
        min_reference_keep = k_keep - replace_quota
    min_reference_keep = int(max(0, min(min_reference_keep, k_keep)))

    # Actual number of reference tokens to keep.
    ref_keep = max(min_reference_keep, k_keep - replace_quota)
    ref_keep = int(max(0, min(ref_keep, k_keep)))

    ref_score = reference_score.to(device=quality.device)
    ref_mask = _v12_to_reference_mask(ref_score, k_keep=k_keep, n_tokens=n).to(device=quality.device)

    q = _v11_minmax_score(quality).clamp(min=1e-6, max=1.0)
    if candidate_quality_power != 1.0:
        q = q ** float(candidate_quality_power)

    try:
        ref_soft = _v11_minmax_score(ref_score.float().to(device=quality.device))
        if ref_soft.shape != q.shape:
            ref_soft = ref_mask.float()
    except Exception:
        ref_soft = ref_mask.float()

    selected_all = []

    for b in range(bsz):
        q_b = q[b]
        ref_b = ref_mask[b]
        ref_soft_b = ref_soft[b] if ref_soft.shape == q.shape else ref_b.float()

        ref_idxs = torch.where(ref_b)[0]
        nonref_idxs = torch.where(~ref_b)[0]

        selected = []
        used = set()

        # 1) Keep most reference tokens.
        if ref_idxs.numel() > 0 and ref_keep > 0:
            keep_score = q_b[ref_idxs] + float(reference_keep_weight) * ref_soft_b[ref_idxs]
            order = ref_idxs[keep_score.argsort(descending=True)]
            for idx in order.tolist():
                if idx not in used:
                    selected.append(idx)
                    used.add(idx)
                if len(selected) >= ref_keep:
                    break

        # 2) Add a few non-reference residual tokens.
        target = k_keep
        while len(selected) < target:
            best_i = None
            best_score = None

            for idx in nonref_idxs.tolist():
                if idx in used:
                    continue

                score = q_b[idx]

                if kernel is not None and selected:
                    try:
                        sel_t = torch.tensor(selected, dtype=torch.long, device=quality.device)
                        sim_pen = kernel[b, idx, sel_t].max()
                        score = score - float(diversity_lambda) * sim_pen
                    except Exception:
                        pass

                if best_score is None or score > best_score:
                    best_score = score
                    best_i = idx

            if best_i is None:
                break

            selected.append(best_i)
            used.add(best_i)

        # 3) Fallback: fill from reference first, then global quality.
        if len(selected) < k_keep:
            for idx in ref_idxs.tolist():
                if idx not in used:
                    selected.append(idx)
                    used.add(idx)
                if len(selected) >= k_keep:
                    break

        if len(selected) < k_keep:
            for idx in q_b.argsort(descending=True).tolist():
                if idx not in used:
                    selected.append(idx)
                    used.add(idx)
                if len(selected) >= k_keep:
                    break

        selected = selected[:k_keep]
        if sort_selected:
            selected = sorted(selected)

        selected_all.append(torch.tensor(selected, dtype=torch.long, device=quality.device))

    return torch.stack(selected_all, dim=0)


# ============================================================
# v13 Native-reference margin-gated residual exchange selector
# ============================================================

def native_reference_margin_gated_exchange_select(
    quality: torch.Tensor,
    reference_score: torch.Tensor,
    kernel: torch.Tensor = None,
    k_keep: int = 32,
    max_replace_quota: int = 3,
    margin_threshold: float = 0.04,
    protect_top_ref: int = 26,
    diversity_lambda: float = 0.08,
    reference_keep_weight: float = 0.20,
    candidate_quality_power: float = 1.0,
    sort_selected: bool = True,
) -> torch.Tensor:
    """Native-reference margin-gated residual exchange.

    Start from native/reference selected tokens. Then replace a weak reference
    token only if the best non-reference candidate is better by a margin.

    Compared with v12 fixed residual exchange, this selector can replace
    0/1/2/3 tokens per sample dynamically.
    """
    quality = quality.float()
    bsz, n = quality.shape
    k_keep = int(max(1, min(k_keep, n)))
    max_replace_quota = int(max(0, min(max_replace_quota, k_keep)))
    protect_top_ref = int(max(0, min(protect_top_ref, k_keep)))

    ref_score = reference_score.to(device=quality.device)
    ref_mask = _v12_to_reference_mask(ref_score, k_keep=k_keep, n_tokens=n).to(device=quality.device)

    q = _v11_minmax_score(quality).clamp(min=1e-6, max=1.0)
    if candidate_quality_power != 1.0:
        q = q ** float(candidate_quality_power)

    try:
        ref_soft = _v11_minmax_score(ref_score.float().to(device=quality.device))
        if ref_soft.shape != q.shape:
            ref_soft = ref_mask.float()
    except Exception:
        ref_soft = ref_mask.float()

    selected_all = []

    for b in range(bsz):
        q_b = q[b]
        ref_b = ref_mask[b]
        ref_soft_b = ref_soft[b] if ref_soft.shape == q.shape else ref_b.float()

        ref_idxs = torch.where(ref_b)[0]
        nonref_idxs = torch.where(~ref_b)[0]

        # Fallback if reference is malformed.
        if ref_idxs.numel() == 0:
            idx = q_b.topk(k=k_keep).indices
            if sort_selected:
                idx = idx.sort().values
            selected_all.append(idx)
            continue

        # Start from reference/native mask.
        # If reference has more than k tokens, keep top-k by ref+quality.
        ref_keep_score = q_b[ref_idxs] + float(reference_keep_weight) * ref_soft_b[ref_idxs]
        ref_order = ref_idxs[ref_keep_score.argsort(descending=True)]
        selected = ref_order[:k_keep].tolist()
        selected_set = set(int(x) for x in selected)

        # Protected reference tokens are never dropped.
        protected = set(int(x) for x in ref_order[:protect_top_ref].tolist())

        for _ in range(max_replace_quota):
            # Droppable selected reference tokens.
            drop_candidates = [idx for idx in selected if idx not in protected]
            if not drop_candidates:
                break

            drop_t = torch.tensor(drop_candidates, dtype=torch.long, device=quality.device)
            drop_score = q_b[drop_t] + float(reference_keep_weight) * ref_soft_b[drop_t]
            weakest_pos = int(drop_score.argmin().item())
            drop_idx = int(drop_candidates[weakest_pos])
            weakest_score = drop_score[weakest_pos]

            best_idx = None
            best_score = None

            for cand in nonref_idxs.tolist():
                cand = int(cand)
                if cand in selected_set:
                    continue

                score = q_b[cand]

                # Penalize adding a token too similar to already selected tokens.
                if kernel is not None and len(selected) > 0:
                    try:
                        sel_t = torch.tensor(selected, dtype=torch.long, device=quality.device)
                        sim_pen = kernel[b, cand, sel_t].max()
                        score = score - float(diversity_lambda) * sim_pen
                    except Exception:
                        pass

                if best_score is None or score > best_score:
                    best_score = score
                    best_idx = cand

            if best_idx is None or best_score is None:
                break

            # Margin gate: replace only if candidate is sufficiently better.
            gain = best_score - weakest_score
            if float(gain.item() if hasattr(gain, "item") else gain) < float(margin_threshold):
                break

            # Apply exchange.
            selected_set.remove(drop_idx)
            selected_set.add(best_idx)
            selected = [x for x in selected if x != drop_idx] + [best_idx]

        # Robustly enforce exactly k tokens.
        if len(selected) < k_keep:
            for idx in ref_order.tolist():
                idx = int(idx)
                if idx not in selected_set:
                    selected.append(idx)
                    selected_set.add(idx)
                if len(selected) >= k_keep:
                    break

        if len(selected) < k_keep:
            for idx in q_b.argsort(descending=True).tolist():
                idx = int(idx)
                if idx not in selected_set:
                    selected.append(idx)
                    selected_set.add(idx)
                if len(selected) >= k_keep:
                    break

        selected = selected[:k_keep]
        if sort_selected:
            selected = sorted(selected)

        selected_all.append(torch.tensor(selected, dtype=torch.long, device=quality.device))

    return torch.stack(selected_all, dim=0)


# ============================================================
# v13.1 Native-reference strict margin-gated exchange selector
# ============================================================

def native_reference_strict_margin_exchange_select(
    quality: torch.Tensor,
    reference_score: torch.Tensor,
    kernel: torch.Tensor = None,
    k_keep: int = 32,
    max_replace_quota: int = 3,
    margin_threshold: float = 0.12,
    min_gain_ratio: float = 1.10,
    min_candidate_quality: float = 0.75,
    max_drop_quality: float = 0.80,
    protect_top_ref: int = 28,
    diversity_lambda: float = 0.08,
    reference_keep_weight: float = 0.20,
    candidate_quality_power: float = 1.0,
    sort_selected: bool = True,
) -> torch.Tensor:
    """Strict native-reference residual exchange.

    This is stricter than v13 margin-gated exchange. It starts from the
    native/reference mask and only replaces a reference token when all gates pass:

    1) candidate_score - drop_score >= margin_threshold
    2) candidate_score / drop_score >= min_gain_ratio
    3) candidate_score >= min_candidate_quality
    4) drop_score <= max_drop_quality

    Therefore each sample can replace 0/1/2/3 tokens dynamically.
    """
    quality = quality.float()
    bsz, n = quality.shape
    k_keep = int(max(1, min(k_keep, n)))
    max_replace_quota = int(max(0, min(max_replace_quota, k_keep)))
    protect_top_ref = int(max(0, min(protect_top_ref, k_keep)))

    ref_score = reference_score.to(device=quality.device)
    ref_mask = _v12_to_reference_mask(ref_score, k_keep=k_keep, n_tokens=n).to(device=quality.device)

    q = _v11_minmax_score(quality).clamp(min=1e-6, max=1.0)
    if candidate_quality_power != 1.0:
        q = q ** float(candidate_quality_power)

    try:
        ref_soft = _v11_minmax_score(ref_score.float().to(device=quality.device))
        if ref_soft.shape != q.shape:
            ref_soft = ref_mask.float()
    except Exception:
        ref_soft = ref_mask.float()

    selected_all = []

    for b in range(bsz):
        q_b = q[b]
        ref_b = ref_mask[b]
        ref_soft_b = ref_soft[b] if ref_soft.shape == q.shape else ref_b.float()

        ref_idxs = torch.where(ref_b)[0]
        nonref_idxs = torch.where(~ref_b)[0]

        if ref_idxs.numel() == 0:
            idx = q_b.topk(k=k_keep).indices
            if sort_selected:
                idx = idx.sort().values
            selected_all.append(idx)
            continue

        # Start from native/reference top-k.
        ref_keep_score = q_b[ref_idxs] + float(reference_keep_weight) * ref_soft_b[ref_idxs]
        ref_order = ref_idxs[ref_keep_score.argsort(descending=True)]
        selected = [int(x) for x in ref_order[:k_keep].tolist()]
        selected_set = set(selected)

        protected = set(int(x) for x in ref_order[:protect_top_ref].tolist())

        for _ in range(max_replace_quota):
            drop_candidates = [idx for idx in selected if idx not in protected]
            if not drop_candidates:
                break

            # Drop weakest selected reference by pure quality, not by ref-soft score.
            drop_t = torch.tensor(drop_candidates, dtype=torch.long, device=quality.device)
            drop_quality = q_b[drop_t]
            weakest_pos = int(drop_quality.argmin().item())
            drop_idx = int(drop_candidates[weakest_pos])
            drop_score = q_b[drop_idx]

            best_idx = None
            best_score = None

            for cand in nonref_idxs.tolist():
                cand = int(cand)
                if cand in selected_set:
                    continue

                score = q_b[cand]

                if kernel is not None and len(selected) > 0:
                    try:
                        sel_t = torch.tensor(selected, dtype=torch.long, device=quality.device)
                        sim_pen = kernel[b, cand, sel_t].max()
                        score = score - float(diversity_lambda) * sim_pen
                    except Exception:
                        pass

                if best_score is None or score > best_score:
                    best_score = score
                    best_idx = cand

            if best_idx is None or best_score is None:
                break

            cand_score = best_score
            gain = cand_score - drop_score
            ratio = cand_score / (drop_score + 1e-6)

            gain_v = float(gain.item() if hasattr(gain, "item") else gain)
            ratio_v = float(ratio.item() if hasattr(ratio, "item") else ratio)
            cand_v = float(cand_score.item() if hasattr(cand_score, "item") else cand_score)
            drop_v = float(drop_score.item() if hasattr(drop_score, "item") else drop_score)

            if gain_v < float(margin_threshold):
                break
            if ratio_v < float(min_gain_ratio):
                break
            if cand_v < float(min_candidate_quality):
                break
            if drop_v > float(max_drop_quality):
                break

            selected_set.remove(drop_idx)
            selected_set.add(best_idx)
            selected = [x for x in selected if x != drop_idx] + [best_idx]

        # Enforce exactly K.
        if len(selected) < k_keep:
            for idx in ref_order.tolist():
                idx = int(idx)
                if idx not in selected_set:
                    selected.append(idx)
                    selected_set.add(idx)
                if len(selected) >= k_keep:
                    break

        if len(selected) < k_keep:
            for idx in q_b.argsort(descending=True).tolist():
                idx = int(idx)
                if idx not in selected_set:
                    selected.append(idx)
                    selected_set.add(idx)
                if len(selected) >= k_keep:
                    break

        selected = selected[:k_keep]

        if sort_selected:
            selected = sorted(selected)

        selected_all.append(torch.tensor(selected, dtype=torch.long, device=quality.device))

    return torch.stack(selected_all, dim=0)
