# RunPod Reproduction Notes — ACT 300 Mixed Pick-and-Place

## Main result

Trained ACT on RunPod using RTX 4090.

Dataset:
- 300 mixed successful demos
- 240 clean
- 40 recovery
- 20 hard-case
- 352 total expert attempts
- 300 successful demos
- Expert collection success rate: 85.23%
- Dataset size: about 78 GB

Training:
- Model: ACT
- chunk_size / num_queries: 100
- hidden_dim: 512
- dim_feedforward: 3200
- encoder layers: 4
- decoder layers: 7
- heads: 8
- KL weight: 10
- batch size: 8
- epochs: 2000
- GPU: RTX 4090 CUDA
- torch: 2.6.0+cu124
- training time: about 92 minutes
- best epoch: 1999
- best val loss: 0.03754
- train loss: 0.04917

Evaluation:
- Normal eval: 49/50 success = 98%
- Normal avg final distance: 0.0141 m
- Hard eval: 40/50 success = 80%
- Hard avg final distance: 0.0385 m

Important files:
- checkpoints/act_300_mixed_cuda4090/policy_best.ckpt
- checkpoints/act_300_mixed_cuda4090/policy_last.ckpt
- checkpoints/act_300_mixed_cuda4090/dataset_stats.pkl
- eval_results/act_300_mixed_cuda4090/results.json
- eval_results/act_300_mixed_cuda4090_hard/results.json
- runpod_act_minimal_outputs.tar.gz
- RUNPOD_RUN_SUMMARY.txt

Important RunPod setup lesson:
Do not blindly run uv sync. It installed torch 2.11.0+cu130, which broke CUDA on this pod.

Correct CUDA setup was:
1. Remove .venv
2. Create uv Python 3.11 env
3. Install torch==2.6.0 and torchvision==0.21.0 with CUDA 12.4 wheels
4. Install remaining requirements without torch packages
5. Install Shaka-Labs DETR fork
6. Patch DETR
7. Always use uv run --no-sync

Correct environment variables:
- MUJOCO_GL=egl
- MUJOCO_MENAGERIE_PATH=/workspace/mujoco_menagerie

Future direction:
Use policy_best.ckpt and dataset_stats.pkl as the baseline/warm-start for broader manipulation:
- grasp arbitrary objects
- place them anywhere on the table
- randomize object shape, color, size, mass, friction
- randomize object and goal positions
- compare warm-start from this mixed model vs training from scratch
