from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any
import torch


@dataclass
class CDPrunerContext:
    """Runtime tensors available at CDPruner encode_images stage."""
    image_features: torch.Tensor  # projected features: [B, N, D]
    image_embeds: torch.Tensor    # CLIP/vision-tower embeds: [B, N, C]
    text_embeds: torch.Tensor     # text embeds: [M, C] or [B, M, C]
    visual_token_budget: int
    token_positions: Optional[torch.Tensor] = None  # [N,2] or [B,N,2]
    metadata: Optional[Dict[str, Any]] = None

    def validate(self) -> None:
        if self.image_features.ndim != 3:
            raise ValueError(f"image_features must be [B,N,D], got {tuple(self.image_features.shape)}")
        if self.image_embeds.ndim != 3:
            raise ValueError(f"image_embeds must be [B,N,C], got {tuple(self.image_embeds.shape)}")
        if self.text_embeds.ndim not in (2, 3):
            raise ValueError(f"text_embeds must be [M,C] or [B,M,C], got {tuple(self.text_embeds.shape)}")
        b1, n1, _ = self.image_features.shape
        b2, n2, _ = self.image_embeds.shape
        if b1 != b2 or n1 != n2:
            raise ValueError("image_features and image_embeds must share [B,N].")
        if not (1 <= int(self.visual_token_budget) <= n1):
            raise ValueError(f"visual_token_budget must be in [1,{n1}], got {self.visual_token_budget}")
