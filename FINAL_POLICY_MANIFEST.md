# AutoPrune Canonical Final Policy

The canonical main-experiment policy in this release is the Qwen-Plus Full-TPDSL state selected on LLaVA-1.5-7B / MME at 32 visual tokens.

- Anchor policy: `configs/final/autoprune_llava15_mme_k32_anchor.yaml`
- Residual merge policy: `openevolve/policies/final/autoprune_llava15_mme_k32_merge.yaml`
- Reported MME Perception score: **1413.46**
- Residual exchange quota: **2**
- Minimum preserved base-policy tokens: **30 / 32**
- Diversity coefficient: **0.19**
- Reference-keep weight: **0.22**
- Candidate-quality power: **1.02**

The files are byte-for-byte copies of the selected formal-search state from `fresh5_gpu0_cand5_r04`. The original formal search artifacts are retained for provenance; obsolete pilot searches are removed.
