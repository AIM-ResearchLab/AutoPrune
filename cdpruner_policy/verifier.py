ALLOWED_SCORERS = {
    "instruction_guided_visual_relevance", "visual_token_norm_saliency", "spatial_centrality_prior",
    "pairwise_token_similarity", "conditional_similarity_score", "redundancy_density_penalty",
    "attention_proxy_saliency", "cdpruner_default_soft_score",
    "native_teacher_mask_prior",
    "cdpruner_reference_mask_prior",
    "native_teacher_mask_prior",
}
ALLOWED_FUSIONS = {"weighted_sum_score_fusion", "quality_diversity_kernel",
    "v7_teacher_residual_fusion",
    "v7_score_product_fusion",
    "v7_rank_fusion", "baseline_residual_score_fusion", "full_token_score_fusion"}
ALLOWED_SELECTORS = {
    "topk_selector", "dpp_selector", "topk_pool_dpp_selector", "topk_pool_mmr_selector",
    "full_token_pool_dpp_selector", "full_token_pool_mmr_selector", "stratified_full_token_selector",
    "v7_two_stage_pool_selector",
    "v7_conditional_dpp_selector",
    "native_bounded_replace_selector",
    "baseline_anchored_residual_selector", "cdpruner_mask_anchored_residual_selector",
}
def verify_policy(policy, strict=True):
    step = (policy.get("pipeline") or [None])[0]
    if not step:
        if strict: raise ValueError("empty pipeline")
        return False
    for name, spec in (step.get("scorer") or {}).items():
        atom = spec.get("atom") if isinstance(spec, dict) else None
        if atom not in ALLOWED_SCORERS:
            if strict: raise ValueError(f"unsupported scorer {name}: {atom}")
            return False
    if (step.get("fusion") or {}).get("atom") not in ALLOWED_FUSIONS:
        if strict: raise ValueError(f"unsupported fusion: {(step.get('fusion') or {}).get('atom')}")
        return False
    if (step.get("selector") or {}).get("atom") not in ALLOWED_SELECTORS:
        if strict: raise ValueError(f"unsupported selector: {(step.get('selector') or {}).get('atom')}")
        return False
    return True
