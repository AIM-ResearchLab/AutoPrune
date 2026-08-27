# Fresh5 Small-Batch Adaptive Search History

Goal:
- Try to reach the 1411-level policy again.
- Token budget is fixed at 32.
- Method remains token pruning / trust-region token-set editing.
- The search should not waste candidates on q0/q1 diagnostics.
- The search should focus on q2_ref30.

Known clean baseline:
- Clean CDPruner K=32
- Perception = 1382.8248
- Cognition = 304.2857

Fresh AutoNAS evidence:
- fresh0 best reached 1406.82.
- fresh3 best reached 1408.46.
- fresh3 repeatedly converged to the plateau around:
  - replace_quota = 2
  - min_reference_keep = 30
  - diversity_lambda = 0.22
  - reference_keep_weight = 0.24
  - candidate_quality_power = 1.08~1.12
  - residual_scale = 0.0004
  - Perception = 1408.46

Important targeted region:
- A previous best-found 1411-level region used:
  - replace_quota = 2
  - min_reference_keep = 30
  - diversity_lambda ≈ 0.19
  - reference_keep_weight ≈ 0.20
  - candidate_quality_power ≈ 1.02
  - residual_scale = 0.0

Small-batch candidate rule:
Each round should generate exactly 5 candidates:
1. One candidate around the current fresh3 plateau: d≈0.22, rw≈0.24, cqp≈1.10, rs≈0.0004.
2. One candidate around the 1411-level anchor-only region: d≈0.19, rw≈0.20, cqp≈1.02, rs=0.0.
3. One nearby anchor-only perturbation: d in [0.17, 0.21], rw in [0.18, 0.22], cqp in [0.98, 1.06], rs=0.0.
4. One weak-residual perturbation: residual_scale in [0.0002, 0.0012].
5. One exploration candidate, but still q2_ref30.

Avoid:
- q0 unless absolutely necessary.
- q1 unless used as a diagnostic.
- q3/q4/q8 aggressive exchange.
- Repeating the exact same parameter combination.
- Generating many variants only around d=0.22, rw=0.24, rs=0.0004.

# Fresh3 Round4 Summary
| Name | Exit | VTN | Perception | Cognition | Raw | ΔP clean | ΔC clean | ΔP known-best | NAS score | MaskRows | SelUnique | SelHash | RefHash | OverlapAvg | ChangedAvg | Note | Anchor | Merge |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---|---|---|
| fresh3_round4_q2_ref30_dl022_rkw024_cqp108_rs0004 | 0 | 32 | 1408.46 | 300.00 | 1708.46 | +25.63 | -4.29 | +1.63 | +24.77 | 2374 | 2372 | 1adc5248df74e163 |  |  |  | score_vs_clean=+24.77 | `configs/v18_llm_nas/fresh3_round4/anchors/anchor_fresh3_round4_q2_ref30_dl022_rkw024_cqp108_rs0004.yaml` | `openevolve/policies/v18_llm_nas/fresh3_round4/merge/merge_fresh3_round4_q2_ref30_dl022_rkw024_cqp108_rs0004.yaml` |
| fresh3_round4_q2_ref30_dl022_rkw024_cqp110_rs0004 | 0 | 32 | 1408.46 | 300.00 | 1708.46 | +25.63 | -4.29 | +1.63 | +24.77 | 2374 | 2372 | 1adc5248df74e163 |  |  |  | score_vs_clean=+24.77 | `configs/v18_llm_nas/fresh3_round4/anchors/anchor_fresh3_round4_q2_ref30_dl022_rkw024_cqp110_rs0004.yaml` | `openevolve/policies/v18_llm_nas/fresh3_round4/merge/merge_fresh3_round4_q2_ref30_dl022_rkw024_cqp110_rs0004.yaml` |
| fresh3_round4_q2_ref30_dl022_rkw024_cqp112_rs0004 | 0 | 32 | 1408.46 | 300.00 | 1708.46 | +25.63 | -4.29 | +1.63 | +24.77 | 2374 | 2372 | 1adc5248df74e163 |  |  |  | score_vs_clean=+24.77 | `configs/v18_llm_nas/fresh3_round4/anchors/anchor_fresh3_round4_q2_ref30_dl022_rkw024_cqp112_rs0004.yaml` | `openevolve/policies/v18_llm_nas/fresh3_round4/merge/merge_fresh3_round4_q2_ref30_dl022_rkw024_cqp112_rs0004.yaml` |
| fresh3_round4_q2_ref30_dl016_rkw018_cqp095_rs0000 | 0 | 32 | 1406.04 | 300.00 | 1706.04 | +23.21 | -4.29 | -0.79 | +22.36 | 2374 | 2372 | 1adc5248df74e163 |  |  |  | score_vs_clean=+22.36 | `configs/v18_llm_nas/fresh3_round4/anchors/anchor_fresh3_round4_q2_ref30_dl016_rkw018_cqp095_rs0000.yaml` | `openevolve/policies/v18_llm_nas/fresh3_round4/merge/merge_fresh3_round4_q2_ref30_dl016_rkw018_cqp095_rs0000_anchor_only.yaml` |
| fresh3_round4_q2_ref30_dl018_rkw020_cqp100_rs0000 | 0 | 32 | 1406.04 | 300.00 | 1706.04 | +23.21 | -4.29 | -0.79 | +22.36 | 2374 | 2372 | 1adc5248df74e163 |  |  |  | score_vs_clean=+22.36 | `configs/v18_llm_nas/fresh3_round4/anchors/anchor_fresh3_round4_q2_ref30_dl018_rkw020_cqp100_rs0000.yaml` | `openevolve/policies/v18_llm_nas/fresh3_round4/merge/merge_fresh3_round4_q2_ref30_dl018_rkw020_cqp100_rs0000_anchor_only.yaml` |
| fresh3_round4_q2_ref30_dl018_rkw020_cqp100_rs0008 | 0 | 32 | 1406.04 | 300.00 | 1706.04 | +23.21 | -4.29 | -0.79 | +22.36 | 2374 | 2372 | 1adc5248df74e163 |  |  |  | score_vs_clean=+22.36 | `configs/v18_llm_nas/fresh3_round4/anchors/anchor_fresh3_round4_q2_ref30_dl018_rkw020_cqp100_rs0008.yaml` | `openevolve/policies/v18_llm_nas/fresh3_round4/merge/merge_fresh3_round4_q2_ref30_dl018_rkw020_cqp100_rs0008.yaml` |
| fresh3_round4_q2_ref30_dl020_rkw022_cqp105_rs0000 | 0 | 32 | 1406.04 | 300.00 | 1706.04 | +23.21 | -4.29 | -0.79 | +22.36 | 2374 | 2372 | 1adc5248df74e163 |  |  |  | score_vs_clean=+22.36 | `configs/v18_llm_nas/fresh3_round4/anchors/anchor_fresh3_round4_q2_ref30_dl020_rkw022_cqp105_rs0000.yaml` | `openevolve/policies/v18_llm_nas/fresh3_round4/merge/merge_fresh3_round4_q2_ref30_dl020_rkw022_cqp105_rs0000_anchor_only.yaml` |
| fresh3_round4_q2_ref30_dl020_rkw022_cqp105_rs0008 | 0 | 32 | 1406.04 | 300.00 | 1706.04 | +23.21 | -4.29 | -0.79 | +22.36 | 2374 | 2372 | 1adc5248df74e163 |  |  |  | score_vs_clean=+22.36 | `configs/v18_llm_nas/fresh3_round4/anchors/anchor_fresh3_round4_q2_ref30_dl020_rkw022_cqp105_rs0008.yaml` | `openevolve/policies/v18_llm_nas/fresh3_round4/merge/merge_fresh3_round4_q2_ref30_dl020_rkw022_cqp105_rs0008.yaml` |
| fresh3_round4_q2_ref30_dl022_rkw024_cqp110_rs0000 | 0 | 32 | 1406.04 | 300.00 | 1706.04 | +23.21 | -4.29 | -0.79 | +22.36 | 2374 | 2372 | 1adc5248df74e163 |  |  |  | score_vs_clean=+22.36 | `configs/v18_llm_nas/fresh3_round4/anchors/anchor_fresh3_round4_q2_ref30_dl022_rkw024_cqp110_rs0000.yaml` | `openevolve/policies/v18_llm_nas/fresh3_round4/merge/merge_fresh3_round4_q2_ref30_dl022_rkw024_cqp110_rs0000_anchor_only.yaml` |
| fresh3_round4_q2_ref30_dl016_rkw016_cqp105_rs0000 | 0 | 32 | 1405.91 | 300.00 | 1705.91 | +23.08 | -4.29 | -0.92 | +22.22 | 2374 | 2372 | 1adc5248df74e163 |  |  |  | score_vs_clean=+22.22 | `configs/v18_llm_nas/fresh3_round4/anchors/anchor_fresh3_round4_q2_ref30_dl016_rkw016_cqp105_rs0000.yaml` | `openevolve/policies/v18_llm_nas/fresh3_round4/merge/merge_fresh3_round4_q2_ref30_dl016_rkw016_cqp105_rs0000_anchor_only.yaml` |
| fresh3_round4_q2_ref30_dl016_rkw016_cqp105_rs0008 | 0 | 32 | 1405.91 | 300.00 | 1705.91 | +23.08 | -4.29 | -0.92 | +22.22 | 2374 | 2372 | 1adc5248df74e163 |  |  |  | score_vs_clean=+22.22 | `configs/v18_llm_nas/fresh3_round4/anchors/anchor_fresh3_round4_q2_ref30_dl016_rkw016_cqp105_rs0008.yaml` | `openevolve/policies/v18_llm_nas/fresh3_round4/merge/merge_fresh3_round4_q2_ref30_dl016_rkw016_cqp105_rs0008.yaml` |
| fresh3_round4_q3_ref29_dl016_rkw020_cqp100_rs0000 | 0 | 32 | 1397.85 | 301.79 | 1699.64 | +15.02 | -2.50 | -8.98 | +14.52 | 2374 | 2372 | a4d78bce1b7d7931 |  |  |  | score_vs_clean=+14.52 | `configs/v18_llm_nas/fresh3_round4/anchors/anchor_fresh3_round4_q3_ref29_dl016_rkw020_cqp100_rs0000.yaml` | `openevolve/policies/v18_llm_nas/fresh3_round4/merge/merge_fresh3_round4_q3_ref29_dl016_rkw020_cqp100_rs0000_anchor_only.yaml` |
