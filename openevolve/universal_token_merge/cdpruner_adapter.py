from __future__ import annotations

import os
from typing import Optional, Dict
import torch

from .schema import load_policy_yaml
from .operators import apply_universal_merge

_HYBRID_DEBUG_COUNT = 0


def get_policy_from_env():
    text = None

    if os.environ.get("EVO_UNIMERGE_POLICY_YAML"):
        text = os.environ["EVO_UNIMERGE_POLICY_YAML"]

    path = os.environ.get("EVO_UNIMERGE_POLICY_PATH")
    if text is None and path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

    if not text:
        return None

    return load_policy_yaml(text)



# ---------------------------------------------------------------------
# v17 formal override:
#   CDPruner-anchored UniMerge should NOT materialize [B, K, C] with
#   all-true masks. That changes the downstream LLaVA/CDPruner behavior.
#
#   Correct behavior:
#     input : vision_tokens [B, 576, C], index_masks [B, 576]
#     merge : compute updated selected-anchor embeddings [B, K, C]
#     output: updated full vision_tokens [B, 576, C], original index_masks
#
#   The downstream CDPruner/LLaVA path will still use index_masks to select
#   the 32 tokens, so anchor_only exactly recovers the CDPruner/v16 baseline.
# ---------------------------------------------------------------------
def maybe_apply_universal_merge_with_mask(
    vision_tokens: torch.Tensor,
    index_masks: torch.Tensor,
    aux: Optional[Dict[str, torch.Tensor]] = None,
):
    if os.environ.get("EVO_UNIMERGE_ENABLE", "0") != "1":
        return vision_tokens, index_masks

    if os.environ.get("EVO_UNIMERGE_MODE", "hybrid_v16_anchor") not in {"post_cdpruner", "hybrid_v16_anchor"}:
        return vision_tokens, index_masks

    policy = get_policy_from_env()
    if policy is None:
        return vision_tokens, index_masks

    try:
        from .operators import apply_universal_merge_with_anchor_mask

        # Compute merged/updated anchor embeddings [B, K, C] or [K, C].
        merged, new_masks = apply_universal_merge_with_anchor_mask(
            vision_tokens,
            index_masks,
            policy,
            aux=aux,
        )

        output_mode = os.environ.get("EVO_UNIMERGE_OUTPUT_MODE", "inplace_full")

        # Legacy mode: materialize to [B, K, C]. Kept only for debugging.
        if output_mode in {"materialize", "compact"}:
            return merged, new_masks

        # Formal mode: update selected positions in full token tensor and
        # preserve the original index_masks.
        updated = vision_tokens.clone()

        if vision_tokens.dim() == 3:
            selected_before = []
            for b in range(vision_tokens.shape[0]):
                mask_b = index_masks[b].bool()
                selected_before.append(vision_tokens[b][mask_b])
                updated[b][mask_b] = merged[b].to(dtype=updated.dtype)

            selected_before = torch.stack(selected_before, dim=0)

        elif vision_tokens.dim() == 2:
            mask = index_masks.bool()
            selected_before = vision_tokens[mask]
            updated[mask] = merged.to(dtype=updated.dtype)

        else:
            raise ValueError(f"Unsupported vision_tokens shape: {tuple(vision_tokens.shape)}")

        global _HYBRID_DEBUG_COUNT
        if os.environ.get("EVO_UNIMERGE_DEBUG", "0") == "1":
            max_n = int(os.environ.get("EVO_UNIMERGE_DEBUG_N", "5"))
            if _HYBRID_DEBUG_COUNT < max_n:
                try:
                    delta = (merged - selected_before).float()
                    print(
                        "[UniMergeHybridDelta] "
                        f"policy={policy.name} "
                        f"merge_type={policy.merge_operator.type} "
                        f"scale={policy.merge_operator.residual_scale} "
                        f"output_mode={output_mode} "
                        f"mean_abs_delta={delta.abs().mean().item():.8f} "
                        f"max_abs_delta={delta.abs().max().item():.8f}",
                        flush=True,
                    )
                except Exception as _e:
                    print(f"[UniMergeHybridDelta][debug_failed] {repr(_e)}", flush=True)

                _HYBRID_DEBUG_COUNT += 1

        return updated, index_masks

    except Exception as e:
        if policy.constraints.fallback_to_selection:
            print(f"[UniMerge][hybrid fallback] merge failed: {repr(e)}", flush=True)
            return vision_tokens, index_masks
        raise
