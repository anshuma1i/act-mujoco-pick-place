# ACT Pick-and-Place in MuJoCo

This project trains an ACT (Action Chunking with Transformers) policy for a simulated Franka Panda pick-and-place task. The robot observes a front camera image plus joint state, then predicts joint-position and gripper actions to move a cube to a target location.

The pipeline is fully simulation-based:

1. Run a scripted MuJoCo expert and save only successful demonstration episodes.
2. Train an ACT policy on HDF5 demonstrations.
3. Evaluate the learned policy in MuJoCo, optionally with temporal aggregation and MP4 output.

## Final Result

The final reported evaluation used the best checkpoint from the 100-demo `pick_place` run:

```text
Checkpoint: checkpoints/pick_place/policy_best.ckpt
Dataset: data/pick_place/ (100 demonstrations)
Evaluation: 50 episodes
Temporal aggregation: enabled
Eval chunk size: 50
Success: 32/50 = 64%
Average episode length: 164.5
Episode length std: 94.0
Video: checkpoints/pick_place/eval_best_50ep_chunk50_tagg.mp4
Log: logs/eval_pick_place_best_50ep_chunk50_tagg.log
```

## Experiment Comparison

All three comparison runs below use the same evaluation setting:

- evaluation episodes: `50`
- evaluation chunk size: `50`
- temporal aggregation: `on`

| Demonstrations | Artifact Folder | Result |
| --- | --- | --- |
| 50 demos | `checkpoints/pick_place_chunk50_tagg/` | `17/50 = 34%` |
| 100 demos | `checkpoints/pick_place/` | `32/50 = 64%` |
| 150 demos | `checkpoints/pick_place_150ep_chunk50_tagg/` | `39/50 = 78%` |

The 100-demo run is the primary final run for this workspace. The 50-demo and 150-demo folders are preserved as comparative experiment artifacts and should not be overwritten.

## Deliverables / Artifact Locations

These are the submission-relevant deliverables requested by the assignment: code, demonstration data, trained checkpoint, training curves, evaluation video, evaluation metrics, and report.

For Git submission, keep the code and documentation in the repository and provide the large generated artifacts separately. In this workspace, the large artifacts live under `data/` and `checkpoints/` and are intended to be uploaded to Google Drive rather than pushed to GitHub.

- Code:
  - `https://github.com/anshuma1i/act-mujoco-pick-place`

- Demonstration data:
  - `data/pick_place/`

- Trained model checkpoints:
  - 100-demo final run: `checkpoints/pick_place/policy_best.ckpt`
  - 50-demo comparison: `checkpoints/pick_place_chunk50_tagg/policy_best.ckpt`
  - 150-demo comparison: `checkpoints/pick_place_150ep_chunk50_tagg/policy_best.ckpt`

- Training curves:
  - 100-demo: `checkpoints/pick_place/train_val_loss_seed_42.png`
  - 100-demo: `checkpoints/pick_place/train_val_l1_seed_42.png`
  - 100-demo: `checkpoints/pick_place/train_val_kl_seed_42.png`
  - 50-demo: `checkpoints/pick_place_chunk50_tagg/train_val_loss_seed_42.png`
  - 150-demo: `checkpoints/pick_place_150ep_chunk50_tagg/train_val_loss_seed_42.png`

- Evaluation videos:
  - 100-demo final comparison video: `checkpoints/pick_place/eval_best_50ep_chunk50_tagg.mp4`
  - 100-demo earlier temporal-aggregation video: `checkpoints/pick_place/eval_50ep_policy_best_100demo_temporal_agg.mp4`
  - 50-demo comparison video: `checkpoints/pick_place_chunk50_tagg/eval_best_50ep.mp4`
  - 150-demo comparison video: `checkpoints/pick_place_150ep_chunk50_tagg/eval_best_50ep.mp4`

- Evaluation metrics:
  - 100-demo final comparison log: `logs/eval_pick_place_best_50ep_chunk50_tagg.log`
  - 50-demo / 150-demo comparison numbers are reported in the final report

- Report:
  - attached in email/submission

The exact command that produced the final reported result was:

```bash
env DEVICE=mps UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/mpl \
  uv run python -u sim/evaluate_sim.py \
  --ckpt checkpoints/pick_place/policy_best.ckpt \
  --chunk_size 50 \
  --temporal_agg \
  --num_episodes 50 \
  --video \
  --video_path checkpoints/pick_place/eval_best_50ep_chunk50_tagg.mp4 \
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
├── data/pick_place/            # Final 100-demo dataset
├── checkpoints/pick_place/     # Final checkpoints, plots, and evaluation videos
└── logs/                       # Collection, training, and evaluation logs
```

There is also a `checkpoints/pick_place_50demo_backup_20260423_030830/` directory kept as provenance from an earlier 50-demo run. It is not the final reported run.

## Artifact Folder Map

- `data/pick_place/`  
  Demonstration dataset used for the main ACT runs in this repo.

- `checkpoints/pick_place/`  
  Main 100-demo run artifacts. This is the primary final-results folder.

- `checkpoints/pick_place_chunk50_tagg/`  
  Comparative 50-demo experiment artifacts for `chunk_size=50` and temporal aggregation.

- `checkpoints/pick_place_150ep_chunk50_tagg/`  
  Comparative 150-demo experiment artifacts for `chunk_size=50` and temporal aggregation.

- `checkpoints/pick_place_50demo_backup_20260423_030830/`  
  Older backup/provenance folder from an earlier 50-demo run; kept for traceability, not as the main comparison folder.

Do not overwrite `checkpoints/pick_place_chunk50_tagg/` or `checkpoints/pick_place_150ep_chunk50_tagg/`. They are preserved comparative experiment folders.

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
uv run python sim/collect_data.py --task pick_place --num_episodes 100 --verbose
```

For GUI preview during collection:

```bash
uv run python sim/collect_data.py --task pick_place --num_episodes 100 --render --verbose
```

The scripted expert is IK-based and can fail on some randomized episodes. In this repo, `sim/collect_data.py` keeps retrying until it has saved the requested number of successful demonstrations, so the final HDF5 dataset is success-filtered even though the expert policy itself is not guaranteed to succeed every time.

A renderless diagnostic run of the expert in this workspace produced:

```text
Success rate: 87/100 = 87%
```

The final dataset lives at:

```text
data/pick_place/episode_0.hdf5
...
data/pick_place/episode_99.hdf5
```

Verify a dataset:

```bash
uv run python scripts/verify_dataset.py --task pick_place
```

## Training

Train the ACT policy:

```bash
env DEVICE=mps UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/mpl \
  uv run python -u train.py --task pick_place --eval_every 0 \
  2>&1 | tee logs/train_pick_place.log
```

`--eval_every 0` disables checkpoint evaluation videos during training, which is faster and was used for the long final training run. Periodic model checkpoints and training curves are still saved through `--save_every`.

If checkpoint evaluation videos are desired during training:

```bash
uv run python -u train.py --task pick_place --eval_every 200 --eval_episodes 3
```

Checkpoint evaluation is implemented in `train.py::run_checkpoint_eval(...)`. It evaluates the in-memory policy at the save epoch and writes videos named:

```text
checkpoints/pick_place/eval_epoch_{epoch}.mp4
```

## Evaluation

Evaluate the best checkpoint without video:

```bash
uv run python -u sim/evaluate_sim.py \
  --ckpt checkpoints/pick_place/policy_best.ckpt \
  --num_episodes 50
```

Evaluate with the final reported settings and save a video:

```bash
env DEVICE=mps UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/mpl \
  uv run python -u sim/evaluate_sim.py \
  --ckpt checkpoints/pick_place/policy_best.ckpt \
  --chunk_size 50 \
  --temporal_agg \
  --num_episodes 50 \
  --video \
  --video_path checkpoints/pick_place/eval_best_50ep_chunk50_tagg.mp4 \
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
data/pick_place/                                      # 100 HDF5 demonstrations
checkpoints/pick_place/dataset_stats.pkl             # normalization stats
checkpoints/pick_place/policy_best.ckpt              # final best checkpoint
checkpoints/pick_place/policy_last.ckpt              # final training checkpoint
checkpoints/pick_place/train_val_loss_seed_42.png    # loss curve
checkpoints/pick_place/train_val_l1_seed_42.png      # L1 curve
checkpoints/pick_place/train_val_kl_seed_42.png      # KL curve
checkpoints/pick_place/eval_best_50ep_chunk50_tagg.mp4
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
