from __future__ import annotations

import argparse
import torch

from cdpruner_policy import load_policy, CDPrunerPolicyExecutor
from cdpruner_policy.types import CDPrunerContext


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=str, default="configs/base/cdpruner_k32_reference_q0.yaml")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--tokens", type=int, default=576)
    parser.add_argument("--dim", type=int, default=4096)
    parser.add_argument("--clip-dim", type=int, default=1024)
    parser.add_argument("--text-tokens", type=int, default=16)
    parser.add_argument("--keep", type=int, default=128)
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    policy = load_policy(args.policy)
    executor = CDPrunerPolicyExecutor(policy)

    ctx = CDPrunerContext(
        image_features=torch.randn(args.batch, args.tokens, args.dim, device=device),
        image_embeds=torch.randn(args.batch, args.tokens, args.clip_dim, device=device),
        text_embeds=torch.randn(args.text_tokens, args.clip_dim, device=device),
        visual_token_budget=args.keep,
    )

    mask = executor.select(ctx)
    print("policy:", policy.get("policy_name"))
    print("mask shape:", tuple(mask.shape))
    print("selected per sample:", mask.sum(dim=1).tolist())
    assert mask.shape == (args.batch, args.tokens)
    assert mask.dtype == torch.bool
    assert all(int(x) == args.keep for x in mask.sum(dim=1).tolist())
    print("SELFTEST PASS")


if __name__ == "__main__":
    main()
