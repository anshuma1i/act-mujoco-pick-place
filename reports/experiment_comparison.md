# ACT Pick-and-Place Experiment Comparison

Generated after integrating the RunPod RTX 4090 continuation outputs.

## Integrated Artifacts

RunPod outputs were copied into the existing ACT project layout:

- `checkpoints/act_300_mixed_cuda4090/`
  - `policy_best.ckpt`
  - `policy_last.ckpt`
  - `dataset_stats.pkl`
  - `epoch_timing.csv`
  - `train_val_loss_seed_42.png`
  - `train_val_l1_seed_42.png`
  - `train_val_kl_seed_42.png`
- `eval_results/act_300_mixed_cuda4090/`
  - `results.json`
  - `per_episode_results.csv`
  - `eval_best_50ep.mp4`
- `eval_results/act_300_mixed_cuda4090_hard/`
  - `results.json`
  - `per_episode_results.csv`
  - `eval_hard_50ep.mp4`
- `reports/runpod/RUNPOD_REPRODUCE_NOTES.md`
- `reports/runpod/RUNPOD_RUN_SUMMARY.txt`

## RunPod 300 Mixed Dataset and Training

Source: `reports/runpod/RUNPOD_RUN_SUMMARY.txt` and RunPod result JSON files.

| Field | Value |
| --- | --- |
| Dataset | `data/panda_pick_place_300_mixed_success` on RunPod |
| Dataset split | 240 clean, 40 recovery, 20 hard-case |
| Successful demos | 300 |
| Expert attempts | 352 |
| Failed attempts | 52 |
| Expert collection success rate | 85.23% |
| Robot | Franka Panda |
| Image key | `front` |
| Image resolution | 480x640x3 |
| qpos / qvel / action dim | 8 / 8 / 8 |
| Device | RTX 4090 CUDA |
| Torch | 2.6.0+cu124 |
| ACT num_queries / chunk_size | 100 |
| Hidden dim | 512 |
| Feedforward dim | 3200 |
| Encoder / decoder layers | 4 / 7 |
| Heads | 8 |
| KL weight | 10 |
| Batch size | 8 |
| Epochs | 2000 |
| Approx training time | 92 minutes |
| Best epoch | 1999 |
| Best validation loss | 0.03754 |
| Final train loss | 0.04917 |

## Evaluation Results

Structured file-backed results currently available in this repo:

| Run | Dataset type | Device | Checkpoint | Eval mode | Episodes | Seed | Temporal agg | Eval chunk | Success | Success rate | Avg final distance | Median final distance | Avg episode length | Notes |
| --- | --- | --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 300 clean | Clean successful demos | MPS | `checkpoints/act_300_clean_mps/policy_best.ckpt` | hard | 50 | 0 | yes | 50 | 44/50 | 88% | 0.04504 m | 0.01972 m | 145.56 | Same hard-case definition as mixed hard eval, but eval chunk differs from mixed |
| 300 mixed | Mixed successful demos | CUDA | `checkpoints/act_300_mixed_cuda4090/policy_best.ckpt` | normal | 50 | 5000 | yes | 100 | 49/50 | 98% | 0.01409 m | 0.01114 m | 101.26 | RunPod RTX 4090 result |
| 300 mixed | Mixed successful demos | CUDA | `checkpoints/act_300_mixed_cuda4090/policy_best.ckpt` | hard | 50 | 0 | yes | 100 | 40/50 | 80% | 0.03853 m | 0.01543 m | 149.60 | Same hard-case seed and definition as 300-clean hard eval |

Previously recorded local comparison results:

| Run | Dataset type | Checkpoint | Eval mode | Episodes | Temporal agg | Eval chunk | Success | Success rate | Source / caveat |
| --- | --- | --- | --- | ---: | --- | ---: | ---: | ---: | --- |
| 100 clean | Clean demos | `checkpoints/panda_pick_place_100_demos/policy_best.ckpt` | normal | 50 | yes | 50 | 30/50 | 60% | Prior controlled audit/result. Current folder retains video but no structured JSON/log after cleanup |
| 150 clean | Clean demos | `checkpoints/panda_pick_place_150_demos_chunk50_tagg/policy_best.ckpt` | normal | 50 | yes | 50 | 46/50 | 92% | Historical/intermediate reference. Not fully controlled because checkpoint architecture used `num_queries=50` |
| 300 clean | Clean demos | `checkpoints/act_300_clean_mps/policy_best.ckpt` | normal | 50 | yes | 50 | 50/50 | 100% | Prior controlled audit/result. Current folder retains video but no structured JSON/log after cleanup |

Note: the older `README.md` still contains earlier assignment-era numbers for 50/100/150 demos. The latest project state used for this report follows the later controlled audit values above.

## Comparable Conclusions

### 100 Clean vs 300 Clean

This is the strongest clean-data comparison in the local experiment lineage because the 100-clean and 300-clean runs were confirmed to use the same ACT architecture:

- ACT policy
- `num_queries=100`
- action dim 8
- qpos dim 8
- image key `front`
- 480x640x3 images
- hidden dim 512
- feedforward dim 3200
- encoder / decoder layers 4 / 7
- heads 8
- KL weight 10
- batch size 8

Result:

- 100 clean: 30/50 = 60%
- 300 clean: 50/50 = 100%

Interpretation: increasing clean demonstrations from 100 to 300 produced a large normal-evaluation improvement in this setup.

### 150 Clean Historical Reference

The 150-clean run is useful context but should not be treated as a fully controlled comparison against the 100-clean and 300-clean runs because its checkpoint was identified as using `num_queries=50`, while the 100-clean and 300-clean runs use `num_queries=100`.

Result:

- 150 clean historical: 46/50 = 92%

Interpretation: the 150-demo result supports the broader trend that more clean demos helped, but it should be labeled historical/intermediate rather than strictly controlled.

### 300 Clean vs 300 Mixed

The central question was whether replacing part of the 300 clean dataset with recovery and hard-case successful demos improves robustness.

Observed results:

- 300 clean normal: 50/50 = 100%
- 300 clean hard: 44/50 = 88%
- 300 mixed normal: 49/50 = 98%
- 300 mixed hard: 40/50 = 80%

Important caveat:

- 300-clean hard eval used `eval_chunk_size=50`.
- 300-mixed normal/hard RunPod eval used `eval_chunk_size=100`, matching the checkpoint `num_queries=100`.

Because the hard evaluations do not use the same eval chunk size, the robustness comparison is suggestive but not final. Under the currently saved results, the mixed dataset did not improve hard-eval success rate over the clean 300 model. It did have slightly lower average and median final cube-goal distance on hard eval, but success count is the primary metric and is lower.

Recommended strict follow-up:

- Re-evaluate `checkpoints/act_300_clean_mps/policy_best.ckpt` with `--chunk_size 100`, or re-evaluate both 300-clean and 300-mixed with the same device-independent command settings.
- Use the same hard eval seed `0`, same 50 episodes, same temporal aggregation setting, same checkpoint-selection rule, and same hard reset definition.

## Speed and Practical Training Notes

The RunPod RTX 4090 run trained the 300-mixed ACT model in about 92 minutes. The previous 300-clean MPS run took about 848 minutes. These are not identical datasets, but they use the same ACT architecture and training scale, so the practical conclusion is clear:

- CUDA on RTX 4090 is dramatically faster for this ACT setup.
- MPS remained useful for correctness checks and smaller experiments, but remote CUDA is the practical path for full 2000-epoch reruns.

The RunPod notes also record one important setup issue:

- Do not blindly run `uv sync` on RunPod.
- It installed a newer CUDA/PyTorch combination that broke CUDA on that pod.
- The working setup used Python 3.11 with `torch==2.6.0`, `torchvision==0.21.0`, CUDA 12.4 wheels, Shaka-Labs DETR, and `uv run --no-sync`.

## Current Best Artifacts

Primary clean 300 model:

- `checkpoints/act_300_clean_mps/policy_best.ckpt`
- `checkpoints/act_300_clean_mps/policy_last.ckpt`
- `eval_results/act_300_clean_mps/eval_best_50ep.mp4`
- `eval_results/act_300_clean_mps_hard/results.json`
- `eval_results/act_300_clean_mps_hard/per_episode_results.csv`
- `eval_results/act_300_clean_mps_hard/eval_hard_50ep.mp4`

Primary mixed 300 model:

- `checkpoints/act_300_mixed_cuda4090/policy_best.ckpt`
- `checkpoints/act_300_mixed_cuda4090/policy_last.ckpt`
- `checkpoints/act_300_mixed_cuda4090/dataset_stats.pkl`
- `checkpoints/act_300_mixed_cuda4090/epoch_timing.csv`
- `eval_results/act_300_mixed_cuda4090/results.json`
- `eval_results/act_300_mixed_cuda4090/per_episode_results.csv`
- `eval_results/act_300_mixed_cuda4090/eval_best_50ep.mp4`
- `eval_results/act_300_mixed_cuda4090_hard/results.json`
- `eval_results/act_300_mixed_cuda4090_hard/per_episode_results.csv`
- `eval_results/act_300_mixed_cuda4090_hard/eval_hard_50ep.mp4`

