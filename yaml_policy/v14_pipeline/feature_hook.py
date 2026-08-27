from __future__ import annotations

from typing import Any, Tuple
import os
import torch


def v14_enabled() -> bool:
    return os.environ.get("EVO_CDPRUNER_V14_ENABLE", "0").strip() == "1"


def apply_v14_processed_features_if_available(
    executor: Any,
    image_features: torch.Tensor,
    index_mask: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor | None, bool]:
    """Replace gathered/pruned image_features by executor.last_processed_features.

    This must be called in the CDPruner/LLaVA forward hook after executor.select(ctx).
    """
    if not v14_enabled():
        return image_features, index_mask, False

    processed = getattr(executor, "last_processed_features", None)
    if processed is None:
        return image_features, index_mask, False

    processed = processed.to(device=image_features.device, dtype=image_features.dtype)

    # processed shape is [B,K,D]. The downstream path should not prune it again.
    new_mask = torch.ones(
        processed.shape[:2],
        dtype=torch.bool,
        device=processed.device,
    )

    return processed, new_mask, True
