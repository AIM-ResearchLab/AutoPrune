# AutoPrune

AutoPrune is a training-free framework for LLM-driven visual-token pruning design. It represents each candidate as a TPDSL-structured residual modification around a strong base pruning policy and applies bounded residual refinement to preserve reliable selections while correcting uncertain ones.

## Canonical final policy

The main Qwen-Plus policy reported at 32 visual tokens is provided as:

- `configs/final/autoprune_llava15_mme_k32_anchor.yaml`
- `openevolve/policies/final/autoprune_llava15_mme_k32_merge.yaml`

See `FINAL_POLICY_MANIFEST.md` for the exact provenance and selected hyperparameters.

## Repository structure

- `cdpruner_policy/`: TPDSL policy loader, executor, verifier, and pruning atoms.
- `yaml_policy/v9_pipeline/`: typed TPDSL execution pipeline.
- `yaml_policy/v14_pipeline/`: processing/feature hooks still required by the final executor.
- `openevolve/v18_llm_nas/`: formal LLM-driven TPDSL search pipeline.
- `openevolve/universal_token_merge/`: residual embedding-refinement implementation used by the final policy.
- `configs/final/`: canonical final anchor policy.
- `configs/base/`: base CDPruner reference policy used to materialize new search candidates.
- `configs/v18_llm_nas/`: retained formal search artifacts (main Qwen-Plus and proposer-robustness runs).
- `configs/ablation/`: retained paper ablations and formal multi-seed/search-space experiments.
- `openevolve/final_tables/`: retained aggregate search-stability tables.
- `scripts/`: benchmark evaluation utilities inherited from the LLaVA/CDPruner evaluation stack.

## Run the final policy

Set model/data paths used by the benchmark scripts, then invoke the semantic AutoPrune wrapper:

```bash
export CKPT_DIR=/path/to/hf_models
export DATA_DIR=/path/to/eval_data

bash openevolve/run_autoprune_policy.sh \
  configs/final/autoprune_llava15_mme_k32_anchor.yaml \
  openevolve/policies/final/autoprune_llava15_mme_k32_merge.yaml \
  bash scripts/v1_5/eval/mme.sh 32 autoprune_k32
```

## Re-run the 10 x 5 Qwen-Plus search

```bash
export OPENAI_API_KEY=your_api_key
export OPENAI_BASE_URL=https://your-openai-compatible-endpoint/v1
export OPENAI_MODEL=your_model_name

bash run_autoprune_search_10x5.sh
```

For the multi-provider robustness experiment, use `openevolve/run_multi_llm_api_llava15_mme_k32.sh` and provide its required API/model environment variables.

## Data and checkpoints

Model checkpoints, benchmark datasets, generated predictions, logs, and run directories are intentionally not included in this source release.

## Citation

If you use AutoPrune, please cite the corresponding paper.

## License

This repository is released for academic and non-commercial research use only. Commercial use is prohibited without prior permission from the authors. See `LICENSE`.
