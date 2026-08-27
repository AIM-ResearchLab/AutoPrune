from __future__ import annotations

from typing import Dict, Optional, Tuple
import torch
import torch.nn.functional as F

from .schema import UniversalMergePolicy


def _infer_grid(n: int) -> Optional[Tuple[int, int]]:
    r = int(round(n ** 0.5))
    if r * r == n:
        return r, r
    return None


def spatial_centrality_score(n: int, device, dtype):
    grid = _infer_grid(n)
    if grid is None:
        return torch.zeros(n, device=device, dtype=dtype)

    h, w = grid
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, h, device=device, dtype=dtype),
        torch.linspace(-1, 1, w, device=device, dtype=dtype),
        indexing="ij",
    )
    dist = torch.sqrt(xx ** 2 + yy ** 2)
    score = 1.0 - dist / (dist.max() + 1e-6)
    return score.reshape(-1)


def redundancy_score(x: torch.Tensor):
    # x: [N, C]
    x_norm = F.normalize(x, dim=-1)
    sim = x_norm @ x_norm.t()
    sim.fill_diagonal_(0.0)
    return sim.max(dim=-1).values


def score_tokens(
    x: torch.Tensor,
    policy: UniversalMergePolicy,
    aux: Optional[Dict[str, torch.Tensor]] = None,
):
    """
    x: [N, C]
    return score: [N]
    """
    aux = aux or {}
    n = x.shape[0]
    device, dtype = x.device, x.dtype

    weights = policy.scoring.weights or {}
    score = torch.zeros(n, device=device, dtype=dtype)

    if "spatial_centrality" in policy.scoring.features:
        w = float(weights.get("spatial_centrality", weights.get("spatial", 0.2)))
        score = score + w * spatial_centrality_score(n, device, dtype)

    if "attention_importance" in policy.scoring.features and "attention" in aux:
        attn = aux["attention"].to(device=device, dtype=dtype).reshape(-1)
        if attn.numel() == n:
            attn = (attn - attn.min()) / (attn.max() - attn.min() + 1e-6)
            w = float(weights.get("attention_importance", weights.get("attention", 0.3)))
            score = score + w * attn

    if "semantic_similarity" in policy.scoring.features and "query" in aux:
        q = aux["query"].to(device=device, dtype=dtype)
        q = q.reshape(1, -1)
        sim = F.cosine_similarity(x, q.expand_as(x), dim=-1)
        sim = (sim - sim.min()) / (sim.max() - sim.min() + 1e-6)
        w = float(weights.get("semantic_similarity", weights.get("semantic", 0.3)))
        score = score + w * sim

    if "redundancy" in policy.scoring.features:
        red = redundancy_score(x)
        red = (red - red.min()) / (red.max() - red.min() + 1e-6)
        # redundancy 越高越应该被 merge，所以 selection score 里减掉
        w = float(weights.get("redundancy", 0.1))
        score = score - w * red

    return score


def select_anchors(
    x: torch.Tensor,
    score: torch.Tensor,
    keep: int,
    policy: UniversalMergePolicy,
):
    """
    return anchor_idx: [K]
    """
    n = x.shape[0]
    keep = min(max(1, int(keep)), n)

    op = policy.anchor_selection.operator

    if op in {"topk", "score_topk"}:
        return torch.topk(score, k=keep, largest=True).indices.sort().values

    if op in {"topk_diverse", "score_topk_diverse"}:
        # 简单多样性版本：先按 score 排序，再用 cosine 去重
        x_norm = F.normalize(x, dim=-1)
        order = torch.argsort(score, descending=True)

        selected = []
        threshold = float(policy.raw.get("diversity_threshold", 0.92))

        for idx in order.tolist():
            if not selected:
                selected.append(idx)
            else:
                cur = x_norm[idx:idx+1]
                prev = x_norm[torch.tensor(selected, device=x.device)]
                max_sim = (cur @ prev.t()).max().item()
                if max_sim < threshold:
                    selected.append(idx)
            if len(selected) >= keep:
                break

        # 如果 diversity 太强导致不够，补 top score
        if len(selected) < keep:
            chosen = set(selected)
            for idx in order.tolist():
                if idx not in chosen:
                    selected.append(idx)
                    chosen.add(idx)
                if len(selected) >= keep:
                    break

        return torch.tensor(sorted(selected), device=x.device, dtype=torch.long)

    # fallback
    return torch.topk(score, k=keep, largest=True).indices.sort().values


def assign_to_anchors(
    x: torch.Tensor,
    anchor_idx: torch.Tensor,
    policy: UniversalMergePolicy,
):
    """
    return assign: [N] with value in [0, K-1]
    """
    anchors = x[anchor_idx]
    x_norm = F.normalize(x, dim=-1)
    a_norm = F.normalize(anchors, dim=-1)
    sim = x_norm @ a_norm.t()
    assign = torch.argmax(sim, dim=-1)

    # 强制 anchor 自己归到自己
    for j, idx in enumerate(anchor_idx.tolist()):
        assign[idx] = j
    return assign


def merge_tokens(
    x: torch.Tensor,
    anchor_idx: torch.Tensor,
    assign: torch.Tensor,
    score: torch.Tensor,
    policy: UniversalMergePolicy,
):
    """
    x: [N, C]
    return y: [K, C]
    """
    anchors = x[anchor_idx]
    k, c = anchors.shape
    y = torch.zeros_like(anchors)

    merge_type = policy.merge_operator.type
    residual_scale = float(policy.residual.scale if policy.residual.enabled else 0.0)

    for j in range(k):
        group_mask = assign == j
        group = x[group_mask]
        group_score = score[group_mask]

        if group.numel() == 0:
            y[j] = anchors[j]
            continue

        if merge_type in {"anchor_only", "none"}:
            merged = anchors[j]

        elif merge_type in {"mean", "mean_merge"}:
            merged = group.mean(dim=0)

        elif merge_type in {"non_anchor_weighted_residual", "external_anchor_residual"}:
            # Stronger but still anchor-preserving merge:
            # use only non-anchor tokens assigned to this anchor to build residual.
            group_indices = torch.nonzero(group_mask, as_tuple=False).reshape(-1)
            anchor_global_idx = int(anchor_idx[j].item())
            non_anchor_indices = group_indices[group_indices != anchor_global_idx]

            if non_anchor_indices.numel() == 0:
                merged = anchors[j]
            else:
                non_anchor = x[non_anchor_indices]
                non_anchor_score = score[non_anchor_indices]

                # Novelty: tokens less similar to the anchor may contain complementary information.
                anchor_vec = anchors[j:j+1]
                sim_to_anchor = F.cosine_similarity(non_anchor, anchor_vec.expand_as(non_anchor), dim=-1)
                novelty = 1.0 - sim_to_anchor

                logits = non_anchor_score.float() + novelty.float()

                topk = int(policy.raw.get("residual_topk_per_anchor", 0) or 0)
                if topk > 0 and non_anchor_indices.numel() > topk:
                    top_vals, top_pos = torch.topk(logits, k=topk, largest=True)
                    non_anchor = non_anchor[top_pos]
                    logits = top_vals

                w = torch.softmax(logits, dim=0).to(dtype=x.dtype)
                group_mean = (non_anchor * w[:, None]).sum(dim=0)

                residual = group_mean - anchors[j]

                # Optional residual norm clipping.
                clip = float(policy.raw.get("residual_norm_clip", 0.0) or 0.0)
                if clip > 0:
                    r_norm = residual.float().norm() + 1e-6
                    a_norm = anchors[j].float().norm() + 1e-6
                    max_norm = clip * a_norm
                    if r_norm > max_norm:
                        residual = residual * (max_norm / r_norm).to(dtype=x.dtype)

                scale = float(policy.merge_operator.residual_scale)
                if scale == 0.0 and policy.residual.enabled:
                    scale = float(policy.residual.scale)

                merged = anchors[j] + scale * residual

        elif merge_type in {"anchor_exchange", "residual_exchange", "anchor_token_exchange"}:
            # Baseline-anchored but stronger than residual injection.
            # For each anchor group, find the most novel non-anchor token and
            # blend it into the anchor. Only a limited number of anchors are
            # actually exchanged, controlled by replace_quota.
            group_indices = torch.nonzero(group_mask, as_tuple=False).reshape(-1)
            anchor_global_idx = int(anchor_idx[j].item())
            non_anchor_indices = group_indices[group_indices != anchor_global_idx]

            if non_anchor_indices.numel() == 0:
                merged = anchors[j]
            else:
                non_anchor = x[non_anchor_indices]
                anchor_vec = anchors[j:j+1]

                sim_to_anchor = F.cosine_similarity(
                    non_anchor,
                    anchor_vec.expand_as(non_anchor),
                    dim=-1,
                )
                novelty = 1.0 - sim_to_anchor

                # Add optional token score. This keeps the operator compatible
                # with general DSL scoring.
                local_score = score[non_anchor_indices].float()
                logits = novelty.float() + local_score

                best_pos = torch.argmax(logits)
                candidate = non_anchor[best_pos]

                alpha = float(policy.raw.get("exchange_alpha", policy.merge_operator.residual_scale))
                alpha = max(0.0, min(1.0, alpha))

                merged = (1.0 - alpha) * anchors[j] + alpha * candidate

        elif merge_type in {"non_anchor_mean_residual"}:
            group_indices = torch.nonzero(group_mask, as_tuple=False).reshape(-1)
            anchor_global_idx = int(anchor_idx[j].item())
            non_anchor_indices = group_indices[group_indices != anchor_global_idx]

            if non_anchor_indices.numel() == 0:
                merged = anchors[j]
            else:
                group_mean = x[non_anchor_indices].mean(dim=0)
                scale = float(policy.merge_operator.residual_scale)
                if scale == 0.0 and policy.residual.enabled:
                    scale = float(policy.residual.scale)
                merged = anchors[j] + scale * (group_mean - anchors[j])

        elif merge_type in {"anchor_weighted_residual", "anchor_residual_weighted"}:
            # Conservative merge: preserve the selected anchor and only inject
            # a small residual from its assigned token group.
            w = torch.softmax(group_score.float(), dim=0).to(dtype=x.dtype)
            group_mean = (group * w[:, None]).sum(dim=0)
            scale = float(policy.merge_operator.residual_scale)
            if scale == 0.0 and policy.residual.enabled:
                scale = float(policy.residual.scale)
            merged = anchors[j] + scale * (group_mean - anchors[j])

        elif merge_type in {"anchor_blend", "protected_weighted_average"}:
            # Equivalent to a convex blend between the original anchor and
            # the weighted group mean. alpha should be small, e.g. 0.05-0.20.
            w = torch.softmax(group_score.float(), dim=0).to(dtype=x.dtype)
            group_mean = (group * w[:, None]).sum(dim=0)
            alpha = float(policy.merge_operator.residual_scale)
            if alpha == 0.0 and policy.residual.enabled:
                alpha = float(policy.residual.scale)
            alpha = max(0.0, min(1.0, alpha))
            merged = (1.0 - alpha) * anchors[j] + alpha * group_mean

        elif merge_type in {"weighted_average", "weighted_merge"}:
            w = torch.softmax(group_score.float(), dim=0).to(dtype=x.dtype)
            merged = (group * w[:, None]).sum(dim=0)

        elif merge_type in {"anchor_residual", "residual_anchor_merge"}:
            residual = group.mean(dim=0) - anchors[j]
            merged = anchors[j] + residual_scale * residual

        else:
            # fallback
            merged = anchors[j]

        y[j] = merged

    # Optional global exchange quota. This is useful for baseline-anchored
    # search: most anchors remain exactly the CDPruner/v16 selected tokens,
    # and only a small number of anchors are modified.
    replace_quota = int(policy.raw.get("replace_quota", 0) or 0)
    if replace_quota > 0 and merge_type in {"anchor_exchange", "residual_exchange", "anchor_token_exchange"}:
        replace_quota = min(replace_quota, k)
        exchange_delta = (y - anchors).float().norm(dim=-1)
        keep_pos = torch.topk(exchange_delta, k=replace_quota, largest=True).indices
        mask = torch.zeros(k, dtype=torch.bool, device=x.device)
        mask[keep_pos] = True
        y = torch.where(mask[:, None], y, anchors)

    return y


def apply_universal_merge(
    x: torch.Tensor,
    policy: UniversalMergePolicy,
    aux: Optional[Dict[str, torch.Tensor]] = None,
):
    """
    Universal token merge API.

    x can be:
      [N, C] or [B, N, C]

    return same rank, with N compressed to K.
    """
    if x.dim() == 2:
        return _apply_one(x, policy, aux)

    if x.dim() == 3:
        outs = []
        for b in range(x.shape[0]):
            aux_b = {}
            if aux:
                for k, v in aux.items():
                    if torch.is_tensor(v) and v.dim() > 0 and v.shape[0] == x.shape[0]:
                        aux_b[k] = v[b]
                    else:
                        aux_b[k] = v
            outs.append(_apply_one(x[b], policy, aux_b))
        return torch.stack(outs, dim=0)

    raise ValueError(f"Unsupported token tensor shape: {tuple(x.shape)}")


def _apply_one(
    x: torch.Tensor,
    policy: UniversalMergePolicy,
    aux: Optional[Dict[str, torch.Tensor]] = None,
):
    n = x.shape[0]
    k = int(policy.token_budget.keep)

    if k >= n:
        return x

    score = score_tokens(x, policy, aux=aux)
    anchor_idx = select_anchors(x, score, k, policy)
    assign = assign_to_anchors(x, anchor_idx, policy)
    y = merge_tokens(x, anchor_idx, assign, score, policy)

    if policy.constraints.preserve_budget and y.shape[0] != k:
        raise RuntimeError(f"Budget violation: expected {k}, got {y.shape[0]}")

    return y

def apply_universal_merge_with_anchor_indices(
    x: torch.Tensor,
    anchor_idx: torch.Tensor,
    policy: UniversalMergePolicy,
    aux: Optional[Dict[str, torch.Tensor]] = None,
):
    """
    Apply merge using externally provided anchor indices.

    x: [N, C]
    anchor_idx: [K]
    return: [K, C]

    This is used for v17-hybrid: reuse CDPruner/v16 selected tokens
    as anchors, then merge non-anchor token residuals into these anchors.
    """
    if x.dim() != 2:
        raise ValueError(f"Expected [N, C], got {tuple(x.shape)}")

    anchor_idx = anchor_idx.to(device=x.device, dtype=torch.long).reshape(-1)
    if anchor_idx.numel() == 0:
        return _apply_one(x, policy, aux=aux)

    # Keep exactly the external anchor budget.
    score = score_tokens(x, policy, aux=aux)
    assign = assign_to_anchors(x, anchor_idx, policy)
    y = merge_tokens(x, anchor_idx, assign, score, policy)

    return y


def apply_universal_merge_with_anchor_mask(
    x: torch.Tensor,
    anchor_mask: torch.Tensor,
    policy: UniversalMergePolicy,
    aux: Optional[Dict[str, torch.Tensor]] = None,
):
    """
    x: [B, N, C] or [N, C]
    anchor_mask: [B, N] or [N]
    return merged tokens and all-true new mask.
    """
    if x.dim() == 2:
        idx = torch.nonzero(anchor_mask.reshape(-1), as_tuple=False).reshape(-1)
        y = apply_universal_merge_with_anchor_indices(x, idx, policy, aux=aux)
        new_mask = torch.ones(y.shape[0], dtype=torch.bool, device=x.device)
        return y, new_mask

    if x.dim() == 3:
        outs = []
        masks = []

        for b in range(x.shape[0]):
            idx = torch.nonzero(anchor_mask[b].reshape(-1), as_tuple=False).reshape(-1)

            aux_b = {}
            if aux:
                for k, v in aux.items():
                    if torch.is_tensor(v) and v.dim() > 0 and v.shape[0] == x.shape[0]:
                        aux_b[k] = v[b]
                    else:
                        aux_b[k] = v

            y = apply_universal_merge_with_anchor_indices(x[b], idx, policy, aux=aux_b)
            outs.append(y)
            masks.append(torch.ones(y.shape[0], dtype=torch.bool, device=x.device))

        # all samples should have same K under fixed token budget
        y = torch.stack(outs, dim=0)
        new_mask = torch.stack(masks, dim=0)
        return y, new_mask

    raise ValueError(f"Unsupported x shape: {tuple(x.shape)}")

def apply_universal_merge_with_anchor_indices(
    x: torch.Tensor,
    anchor_idx: torch.Tensor,
    policy: UniversalMergePolicy,
    aux: Optional[Dict[str, torch.Tensor]] = None,
):
    """
    Apply merge using externally provided anchor indices.

    x: [N, C]
    anchor_idx: [K]
    return: [K, C]

    This is used for v17-hybrid: reuse CDPruner/v16 selected tokens
    as anchors, then merge non-anchor token residuals into these anchors.
    """
    if x.dim() != 2:
        raise ValueError(f"Expected [N, C], got {tuple(x.shape)}")

    anchor_idx = anchor_idx.to(device=x.device, dtype=torch.long).reshape(-1)
    if anchor_idx.numel() == 0:
        return _apply_one(x, policy, aux=aux)

    # Keep exactly the external anchor budget.
    score = score_tokens(x, policy, aux=aux)
    assign = assign_to_anchors(x, anchor_idx, policy)
    y = merge_tokens(x, anchor_idx, assign, score, policy)

    return y


def apply_universal_merge_with_anchor_mask(
    x: torch.Tensor,
    anchor_mask: torch.Tensor,
    policy: UniversalMergePolicy,
    aux: Optional[Dict[str, torch.Tensor]] = None,
):
    """
    x: [B, N, C] or [N, C]
    anchor_mask: [B, N] or [N]
    return merged tokens and all-true new mask.
    """
    if x.dim() == 2:
        idx = torch.nonzero(anchor_mask.reshape(-1), as_tuple=False).reshape(-1)
        y = apply_universal_merge_with_anchor_indices(x, idx, policy, aux=aux)
        new_mask = torch.ones(y.shape[0], dtype=torch.bool, device=x.device)
        return y, new_mask

    if x.dim() == 3:
        outs = []
        masks = []

        for b in range(x.shape[0]):
            idx = torch.nonzero(anchor_mask[b].reshape(-1), as_tuple=False).reshape(-1)

            aux_b = {}
            if aux:
                for k, v in aux.items():
                    if torch.is_tensor(v) and v.dim() > 0 and v.shape[0] == x.shape[0]:
                        aux_b[k] = v[b]
                    else:
                        aux_b[k] = v

            y = apply_universal_merge_with_anchor_indices(x[b], idx, policy, aux=aux_b)
            outs.append(y)
            masks.append(torch.ones(y.shape[0], dtype=torch.bool, device=x.device))

        # all samples should have same K under fixed token budget
        y = torch.stack(outs, dim=0)
        new_mask = torch.stack(masks, dim=0)
        return y, new_mask

    raise ValueError(f"Unsupported x shape: {tuple(x.shape)}")
