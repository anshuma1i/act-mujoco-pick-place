# ACT Pick-and-Place in MuJoCo

This project trains an ACT (Action Chunking with Transformers) policy for a simulated Franka Panda pick-and-place task. The robot observes a front camera image plus joint state, then predicts joint-position and gripper actions to move a cube to a target location.

The pipeline is fully simulation-based:

1. Run a scripted MuJoCo expert and save only successful demonstration episodes.
2. Train an ACT policy on HDF5 demonstrations.
3. Evaluate the learned policy in MuJoCo, optionally with temporal aggregation and MP4 output.

## Current Results

The project was extended beyond the original assignment with 300-demo clean and mixed-success ACT runs. The best current normal-eval result is the 300-clean MPS checkpoint; the newest RunPod continuation trained the 300-mixed model on an RTX 4090.

| Run | Dataset | Checkpoint | Eval | Result | Notes |
| --- | --- | --- | --- | --- | --- |
| 100 clean | 100 clean demos | `checkpoints/panda_pick_place_100_demos/policy_best.ckpt` | normal, 50 eps | `30/50 = 60%` | controlled local reference |
| 150 clean | 150 clean demos | `checkpoints/panda_pick_place_150_demos_chunk50_tagg/policy_best.ckpt` | normal, 50 eps | `46/50 = 92%` | historical reference; `num_queries=50` |
| 300 clean | 300 clean demos | `checkpoints/act_300_clean_mps/policy_best.ckpt` | normal, 50 eps | `50/50 = 100%` | strict MPS training |
| 300 clean hard | 300 clean demos | `checkpoints/act_300_clean_mps/policy_best.ckpt` | hard, 50 eps | `44/50 = 88%` | hard reset seed `0` |
| 300 mixed | 240 clean + 40 recovery + 20 hard-case | `checkpoints/act_300_mixed_cuda4090/policy_best.ckpt` | normal, 50 eps | `49/50 = 98%` | RunPod RTX 4090 |
| 300 mixed hard | 240 clean + 40 recovery + 20 hard-case | `checkpoints/act_300_mixed_cuda4090/policy_best.ckpt` | hard, 50 eps | `40/50 = 80%` | RunPod RTX 4090, hard reset seed `0` |

Detailed results and caveats are in `reports/experiment_comparison.md`. Large checkpoints, datasets, and videos are intentionally kept out of Git.

## Deliverables / Artifact Locations

These are the submission-relevant deliverables requested by the assignment: code, demonstration data, trained checkpoint, training curves, evaluation video, evaluation metrics, and report.

For Git submission, keep the code and documentation in the repository and provide the large generated artifacts separately. In this workspace, the large artifacts live under `data/` and `checkpoints/` and are intended to be uploaded to Google Drive rather than pushed to GitHub.

- Code:
  - `https://github.com/anshuma1i/act-mujoco-pick-place`

- Demonstration data:
  - `data/panda_pick_place_100_demos/`

- Trained model checkpoints:
  - 100-demo final run: `checkpoints/panda_pick_place_100_demos/policy_best.ckpt`
  - 50-demo comparison: `checkpoints/panda_pick_place_50_demos_chunk50_tagg/policy_best.ckpt`
  - 150-demo comparison: `checkpoints/panda_pick_place_150_demos_chunk50_tagg/policy_best.ckpt`
  - 300-clean extension: `checkpoints/act_300_clean_mps/policy_best.ckpt`
  - 300-mixed RunPod extension: `checkpoints/act_300_mixed_cuda4090/policy_best.ckpt`

- Training curves:
  - 100-demo: `checkpoints/panda_pick_place_100_demos/train_val_loss_seed_42.png`
  - 100-demo: `checkpoints/panda_pick_place_100_demos/train_val_l1_seed_42.png`
  - 100-demo: `checkpoints/panda_pick_place_100_demos/train_val_kl_seed_42.png`
  - 50-demo: `checkpoints/panda_pick_place_50_demos_chunk50_tagg/train_val_loss_seed_42.png`
  - 150-demo: `checkpoints/panda_pick_place_150_demos_chunk50_tagg/train_val_loss_seed_42.png`

- Evaluation videos:
  - 100-demo final comparison video: `checkpoints/panda_pick_place_100_demos/eval_best_50ep_chunk50_tagg.mp4`
  - 100-demo earlier temporal-aggregation video: `checkpoints/panda_pick_place_100_demos/eval_50ep_policy_best_100demo_temporal_agg.mp4`
  - 50-demo comparison video: `checkpoints/panda_pick_place_50_demos_chunk50_tagg/eval_best_50ep.mp4`
  - 150-demo comparison video: `checkpoints/panda_pick_place_150_demos_chunk50_tagg/eval_best_50ep.mp4`

- Evaluation metrics:
  - structured JSON/CSV metrics under `eval_results/`
  - RunPod notes under `reports/runpod/`

- Report:
  - `reports/experiment_comparison.md`

The exact command that produced the final reported result was:

```bash
env DEVICE=mps UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/mpl \
  uv run python -u sim/evaluate_sim.py \
  --ckpt checkpoints/panda_pick_place_100_demos/policy_best.ckpt \
  --chunk_size 50 \
  --temporal_agg \
  --num_episodes 50 \
  --video \
  --video_path checkpoints/panda_pick_place_100_demos/eval_best_50ep_chunk50_tagg.mp4 \
  2>&1 | tee logs/eval_pick_place_best_50ep_chunk50_tagg.log
```

## Repository Structure

```text
ACT/
├── train.py                    # ACT training script and optional checkpoint evaluation
├── config/
│   └── config.py               # Task, model, device, and training config
├── sim/
│   ├── env.py                  # MuJoCo Panda pick-place environment
│   ├── scripted_expert.py      # IK-based demonstration policy
│   ├── collect_data.py         # HDF5 demonstration collection
│   ├── evaluate_sim.py         # Simulation evaluation and video export
│   └── visualize.py            # Expert demo visualization
├── training/
│   ├── policy.py               # ACT policy wrapper
│   └── utils.py                # Dataset loading and preprocessing helpers
├── scripts/
│   ├── patch_detr.py           # Required DETR compatibility patch
│   ├── smoke_check.py          # Fast dependency/model checks
│   ├── validate_setup.py       # Small end-to-end setup validation
│   └── verify_dataset.py       # HDF5 dataset validation
├── data/panda_pick_place_100_demos/            # Final 100-demo dataset
├── checkpoints/panda_pick_place_100_demos/     # Final checkpoints, plots, and evaluation videos
└── logs/                       # Collection, training, and evaluation logs
```

There is also a `checkpoints/panda_pick_place_50_demos_backup_20260423_030830/` directory kept as provenance from an earlier 50-demo run. It is not the final reported run.

## Artifact Folder Map

- `data/panda_pick_place_100_demos/`  
  Demonstration dataset used for the main ACT runs in this repo.

- `data/pick_place`  
  Compatibility symlink to `data/panda_pick_place_100_demos/` for older commands that use `--task pick_place`.

- `data/panda_pick_place_300_demos/`  
  Self-motivated extension dataset with 300 successful scripted-expert demonstrations. Trained checkpoint artifacts are under `checkpoints/act_300_clean_mps/`.

- `data/panda_pick_place_300_mixed_success/`  
  Success-only mixed extension dataset with 240 clean, 40 recovery, and 20 hard-case demos. RunPod checkpoint artifacts are under `checkpoints/act_300_mixed_cuda4090/`.

- `checkpoints/panda_pick_place_100_demos/`  
  Main 100-demo run artifacts. This is the primary final-results folder.

- `checkpoints/panda_pick_place_50_demos_chunk50_tagg/`  
  Comparative 50-demo experiment artifacts for `chunk_size=50` and temporal aggregation.

- `checkpoints/panda_pick_place_150_demos_chunk50_tagg/`  
  Comparative 150-demo experiment artifacts for `chunk_size=50` and temporal aggregation.

- `checkpoints/act_300_clean_mps/`  
  300-clean ACT extension trained on Apple MPS.

- `checkpoints/act_300_mixed_cuda4090/`  
  300-mixed ACT extension trained on RunPod RTX 4090.

- `checkpoints/panda_pick_place_50_demos_backup_20260423_030830/`  
  Older backup/provenance folder from an earlier 50-demo run; kept for traceability, not as the main comparison folder.

Do not overwrite `checkpoints/panda_pick_place_50_demos_chunk50_tagg/` or `checkpoints/panda_pick_place_150_demos_chunk50_tagg/`. They are preserved comparative experiment folders.

## Setup

The project was developed with Python 3.11 and `uv`.

From the parent project directory, make sure the Franka Panda MuJoCo model repository is available next to `ACT/`:

```bash
cd /path/to/picknmove
git clone https://github.com/google-deepmind/mujoco_menagerie.git
```

Then install the ACT project dependencies:

```bash
cd /path/to/picknmove/ACT
uv venv --python 3.11
source .venv/bin/activate
uv sync
```

If `uv sync` is unavailable in your environment, `requirements.txt` is provided as a pip-style fallback:

```bash
uv pip install -r requirements.txt
```

### DETR Patch

The ACT policy uses the Shaka-Labs DETR fork. This project needs a small local compatibility patch because the upstream DETR fork assumes a hardcoded 5D robot state and parses CLI args too broadly.

After installing dependencies, run:

```bash
uv pip install git+https://github.com/Shaka-Labs/detr.git
uv run python scripts/patch_detr.py
```

Re-run `scripts/patch_detr.py` after recreating `.venv` or reinstalling DETR.

## Apple Silicon / MPS Notes

`config/config.py` automatically selects `mps` when available, then CUDA, then CPU. On Apple Silicon, the commands below use:

```bash
env DEVICE=mps UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/mpl
```

MuJoCo rendering on macOS should be run from a normal Terminal, iTerm, or VS Code terminal session with access to the GUI display. Headless or restricted runners may fail with a CoreGraphics renderer error.

## Smoke Checks

Fast import/model check:

```bash
uv run python scripts/smoke_check.py --check-policy
```

Renderer check, if running in a GUI-capable terminal:

```bash
uv run python scripts/smoke_check.py --renderer
```

Small end-to-end validation that patches DETR, collects one tiny validation demo, trains for one epoch, and checks artifacts:

```bash
uv run python scripts/validate_setup.py
```

This writes temporary validation artifacts under `data/setup_validation/` and `checkpoints/setup_validation/`.

## Data Collection

Collect successful scripted-expert demonstrations:

```bash
uv run python sim/collect_data.py \
  --num_demos 100 \
  --dataset_dir data/panda_pick_place_100_demos \
  --verbose
```

For the self-motivated 300-demo extension, collect into a separate dataset folder so the existing 50/100/150-demo artifacts are left untouched:

```bash
env DEVICE=mps PYTORCH_ENABLE_MPS_FALLBACK=1 UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/mpl \
  uv run python -u sim/collect_data.py \
  --num_demos 300 \
  --dataset_dir data/panda_pick_place_300_demos \
  --verbose
```

The collector resumes safely: if `episode_N.hdf5` files already exist in the target folder, it fills missing indices without overwriting existing episodes and continues until the folder contains 300 successful demos. It also writes `metadata.json` and logs discarded failed expert attempts to `failed_attempts.jsonl`.

For GUI preview during collection:

```bash
uv run python sim/collect_data.py \
  --num_demos 100 \
  --dataset_dir data/panda_pick_place_100_demos \
  --render \
  --verbose
```

The scripted expert is IK-based and can fail on some randomized episodes. In this repo, `sim/collect_data.py` keeps retrying until it has saved the requested number of successful demonstrations, so the final HDF5 dataset is success-filtered even though the expert policy itself is not guaranteed to succeed every time.

A renderless diagnostic run of the expert in this workspace produced:

```text
Success rate: 87/100 = 87%
```

The final dataset lives at:

```text
data/panda_pick_place_100_demos/episode_0.hdf5
...
data/panda_pick_place_100_demos/episode_99.hdf5
```

Verify a dataset:

```bash
uv run python scripts/verify_dataset.py \
  --dataset_dir data/panda_pick_place_100_demos \
  --expected 100 \
  --exact \
  --episode-len 300
```

Verify the 300-demo extension dataset:

```bash
uv run python scripts/verify_dataset.py \
  --dataset_dir data/panda_pick_place_300_demos \
  --expected 300 \
  --exact \
  --episode-len 300
```

### Collecting 300 Mixed Expert Demonstrations

For a more robust self-motivated extension dataset, collect a success-only mix of clean expert demos, recovery demos, and hard-case demos:

```bash
env DEVICE=mps PYTORCH_ENABLE_MPS_FALLBACK=1 UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/mpl \
  uv run python -u sim/collect_data.py \
  --num_clean 240 \
  --num_recovery 40 \
  --num_hard 20 \
  --dataset_dir data/panda_pick_place_300_mixed_success \
  --failed_dir data/panda_pick_place_failed_analysis_only \
  --verbose
```

The ACT training dataset remains success-only. Failed attempts are not saved as training HDF5 files; when `--failed_dir` is provided, their metadata is written separately for analysis.

Sanity-check the mixed dataset:

```bash
uv run python sim/check_dataset.py \
  --dataset_dir data/panda_pick_place_300_mixed_success
```

## Training

Train the ACT policy:

```bash
env DEVICE=mps UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/mpl \
  uv run python -u train.py --task panda_pick_place_100_demos --eval_every 0 \
  2>&1 | tee logs/train_panda_pick_place_100_demos.log
```

`--eval_every 0` disables checkpoint evaluation videos during training, which is faster and was used for the long final training run. Periodic model checkpoints and training curves are still saved through `--save_every`.

If checkpoint evaluation videos are desired during training:

```bash
uv run python -u train.py --task panda_pick_place_100_demos --eval_every 200 --eval_episodes 3
```

Checkpoint evaluation is implemented in `train.py::run_checkpoint_eval(...)`. It evaluates the in-memory policy at the save epoch and writes videos named:

```text
checkpoints/panda_pick_place_100_demos/eval_epoch_{epoch}.mp4
```

## Evaluation

Evaluate the best checkpoint without video:

```bash
uv run python -u sim/evaluate_sim.py \
  --ckpt checkpoints/panda_pick_place_100_demos/policy_best.ckpt \
  --num_episodes 50
```

Evaluate with the final reported settings and save a video:

```bash
env DEVICE=mps UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/mpl \
  uv run python -u sim/evaluate_sim.py \
  --ckpt checkpoints/panda_pick_place_100_demos/policy_best.ckpt \
  --chunk_size 50 \
  --temporal_agg \
  --num_episodes 50 \
  --video \
  --video_path checkpoints/panda_pick_place_100_demos/eval_best_50ep_chunk50_tagg.mp4 \
  2>&1 | tee logs/eval_pick_place_best_50ep_chunk50_tagg.log
```

The evaluator prints:

- checkpoint used
- checkpoint `num_queries`
- evaluation chunk size
- whether temporal aggregation is enabled
- stats file path
- video path
- final success rate
- average episode length
- episode length standard deviation

## Final Artifacts

Key files for grading/submission:

```text
data/panda_pick_place_100_demos/                                      # 100 HDF5 demonstrations
checkpoints/panda_pick_place_100_demos/dataset_stats.pkl             # normalization stats
checkpoints/panda_pick_place_100_demos/policy_best.ckpt              # final best checkpoint
checkpoints/panda_pick_place_100_demos/policy_last.ckpt              # final training checkpoint
checkpoints/panda_pick_place_100_demos/train_val_loss_seed_42.png    # loss curve
checkpoints/panda_pick_place_100_demos/train_val_l1_seed_42.png      # L1 curve
checkpoints/panda_pick_place_100_demos/train_val_kl_seed_42.png      # KL curve
checkpoints/panda_pick_place_100_demos/eval_best_50ep_chunk50_tagg.mp4
logs/eval_pick_place_best_50ep_chunk50_tagg.log
logs/train_pick_place_200ep.log                      # includes best val loss summary
```

Best validation loss recovered from logs:

```text
Best checkpoint: epoch 143, val_loss=0.17053
```

## Configuration

Main settings are in `config/config.py`:

| Parameter | Value | Description |
| --- | --- | --- |
| `state_dim` | 8 | 7 arm joints + 1 gripper state |
| `action_dim` | 8 | 7 joint targets + 1 gripper target |
| `episode_len` | 300 | Timesteps per episode |
| `camera_names` | `['front']` | Single camera input |
| `cam_width`, `cam_height` | 640, 480 | Rendered image size |
| `backbone` | ResNet18 | Vision backbone |
| `hidden_dim` | 512 | Transformer hidden size |
| `num_queries` | 100 | ACT model action chunk size |
| `kl_weight` | 10 | CVAE KL weight |
| `num_epochs` | 2000 | Configured full training length |
| `batch_size_train`, `batch_size_val` | 8 | Training/validation batch sizes |

At evaluation time, `--chunk_size` can use fewer predicted actions than the checkpoint's trained `num_queries`. It cannot exceed the checkpoint `num_queries`.

## Known Caveats

- `requirements.txt` is a fallback dependency list. The preferred setup path is `uv sync` plus DETR install/patch.
- Real-robot files (`robot.py`, `teleoperation.py`, `dynamixel.py`, `evaluate.py`) are inherited from the ACT base project and are not used for the MuJoCo submission pipeline.
- Large generated assets under `data/` and `checkpoints/` are intentionally kept because they are part of the assignment evidence.

## References

- Zhao et al. (2023). *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware.* arXiv:2304.13705
- [Shaka-Labs/ACT](https://github.com/Shaka-Labs/ACT)
- [Shaka-Labs/detr](https://github.com/Shaka-Labs/detr)
- [google-deepmind/mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie)
