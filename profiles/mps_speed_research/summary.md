# MPS Speed Research Summary

## Diagnosis

Short fixed-step benchmarks confirm that this ACT training slowdown is mainly model compute on Apple MPS, not HDF5 loading, CPU preprocessing, CPU-to-MPS transfer, or thermal drift. With the original 480x640 image configuration, synthetic tensors already on MPS run at nearly the same speed as real HDF5 data, and dummy zero images are also nearly identical. Backward pass dominates the step time. The only large speedups measured came from experiment-changing interventions: resizing images or freezing the visual backbone.

## Benchmark Setup

- Dataset: `data/panda_pick_place_300_demos`
- Model: ACT, ResNet18-style visual encoder, transformer policy
- Original image shape: `480x640`
- qpos/action dim: `8/8`
- ACT queries: `100`
- Device: strict `mps`
- Fallback: `PYTORCH_ENABLE_MPS_FALLBACK=0`
- Torch: `2.11.0`
- Platform: `macOS-26.4.1-arm64-arm-64bit`
- Checkpoints: not modified
- Full training: not started
- 300-mixed training: not started

Code added for diagnostics:

- `train.py`: fixed-step `--speed_benchmark`, benchmark variants, AMP pilot flag, channels-last pilot flag, torch.compile pilot flag, timed forward/backward/optimizer sections.
- `training/utils.py`: diagnostic DataLoader knobs and diagnostic-only image resizing.
- Output folder: `profiles/mps_speed_research/`

## Results

| Benchmark | Batch | Workers | Image | Variant / Flag | Finite | Sec / Step | Samples / Sec | Main Note |
| --- | ---: | ---: | --- | --- | --- | ---: | ---: | --- |
| baseline_fp32_b8_w0 | 8 | 0 | 480x640 | original FP32 | yes | 0.831 | 9.63 | Control |
| fp32_b8_w2 | 8 | 2 | 480x640 | original FP32 | yes | 0.586 | 13.65 | Best worker setting |
| fp32_b8_w4 | 8 | 4 | 480x640 | original FP32 | yes | 0.587 | 13.64 | No gain over 2 workers |
| synthetic_mps | 8 | 2 | 480x640 | tensors already on MPS | yes | 0.572 | 13.98 | Data/transfer not bottleneck |
| dummy_images | 8 | 2 | 480x640 | zero images | yes | 0.575 | 13.91 | Image content/loading not bottleneck |
| batch16_fp32 | 16 | 2 | 480x640 | original FP32 | yes | 1.142 | 14.00 | Slight throughput gain |
| batch32_fp32 | 32 | 2 | 480x640 | original FP32 | yes | 2.173 | 14.73 | Best unchanged-resolution throughput, high memory |
| batch64_failed | 64 | 2 | 480x640 | original FP32 | no | n/a | n/a | Impractically slow / interrupted |
| frozen_vision | 8 | 2 | 480x640 | frozen visual backbone | yes | 0.393 | 20.36 | Experiment-changing |
| resize_240x320 | 8 | 2 | 240x320 | resized images | yes | 0.264 | 30.28 | Experiment-changing |
| resize_320x240 | 8 | 2 | 320x240 | resized images | yes | 0.262 | 30.52 | Experiment-changing |
| resize_224x224 | 8 | 2 | 224x224 | resized images | yes | 0.226 | 35.44 | Experiment-changing |
| amp_float16 | 8 | 2 | 480x640 | autocast fp16 | no | 0.420 | 19.07 | NaN loss |
| amp_bfloat16 | 8 | 2 | 480x640 | autocast bf16 | no | 0.413 | 19.35 | NaN loss |
| prefer_metal_failed | 8 | 2 | 480x640 | `PYTORCH_MPS_PREFER_METAL=1` | no | n/a | n/a | MPS unavailable in strict mode |
| fast_math_failed | 8 | 2 | 480x640 | `PYTORCH_MPS_FAST_MATH=1` | no | n/a | n/a | MPS unavailable in strict mode |
| channels_last_failed | 8 | 2 | 480x640 | channels_last | no | n/a | n/a | Whole-model conversion unsafe |
| torch_compile_failed | 8 | 2 | 480x640 | torch.compile | no | n/a | n/a | MPS Inductor shader compile failure |

Representative worker-tuned original-config breakdown (`fp32_b8_w2`):

- dataloader: `0.16%`
- transfer to MPS: `0.30%`
- forward: `34.13%`
- backward: `57.45%`
- optimizer: `7.60%`

The synthetic-MPS and dummy-image runs have almost the same throughput as real data, so the bottleneck is not HDF5, image decoding, or CPU-to-GPU movement.

## Top Speedups

Measured by samples/sec, excluding unstable failed runs:

| Rank | Change | Samples / Sec | Speedup vs worker-tuned baseline | Classification |
| ---: | --- | ---: | ---: | --- |
| 1 | Resize to 224x224 | 35.44 | 2.60x | Experiment-changing |
| 2 | Resize to 320x240 | 30.52 | 2.24x | Experiment-changing |
| 3 | Resize to 240x320 | 30.28 | 2.22x | Experiment-changing |
| 4 | Freeze visual backbone | 20.36 | 1.49x | Experiment-changing |
| 5 | Batch size 32 | 14.73 | 1.08x | Experiment-changing / memory-sensitive |

Using `num_workers=2` improved the raw no-worker baseline from `9.63` to `13.65` samples/sec, but it does not explain the RTX 3060 gap and may subtly alter stochastic training order unless worker seeding is locked down.

## Attention Path

The ACT transformer is imported from the installed `detr` package in `.venv/lib/python3.11/site-packages/detr/models/transformer.py`. It uses classic `torch.nn.MultiheadAttention` in DETR internals. I did not rewrite this to explicit Q/K/V plus `scaled_dot_product_attention` because that would mean patching external DETR transformer internals and validating mask/dropout behavior. For this diagnostic pass, that is too risky relative to the user's instruction not to change the scientific experiment.

## Classification

### A. Safe to Use for Short Diagnostics Only

- `--speed_benchmark`
- synthetic-MPS benchmark
- dummy-image benchmark
- AMP pilots
- torch.compile pilot
- channels-last pilot
- diagnostic image resize
- frozen-backbone diagnostic

These are useful for identifying bottlenecks, not for producing final comparison checkpoints.

### B. Safe Engineering Changes for Future Reruns

- Keep epoch and batch timing instrumentation.
- Keep strict MPS diagnostics.
- Keep MuJoCo disabled during offline training.
- Use DataLoader workers only with explicit worker seeding if reproducibility matters.
- Avoid per-batch `.item()`, `.cpu()`, `.numpy()`, plotting, or checkpointing.

### C. Experiment-Changing Changes Requiring Rerun of All Baselines

- Image resize to `224x224`, `240x320`, or `320x240`.
- Batch size changes to `16` or `32`.
- Freezing or partially freezing the visual encoder.
- Changing `num_queries`, model dimensions, backbone, or learning rate.
- Any SDPA transformer rewrite.

These may be excellent future experiments, but they require rerunning 100-clean, 300-clean, and 300-mixed under the same new setup.

### D. Risky / Unstable Changes to Avoid Here

- AMP fp16: produced NaN losses.
- AMP bf16: produced NaN losses.
- `PYTORCH_MPS_PREFER_METAL=1`: made MPS unavailable under strict startup on this build.
- `PYTORCH_MPS_FAST_MATH=1`: made MPS unavailable under strict startup on this build.
- `torch.compile`: failed inside MPS Inductor Metal shader compilation.
- Whole-model `channels_last`: failed on non-rank-4 tensors.
- Batch size 64 at 480x640: impractical / memory-pressure risky.

### E. Remote CUDA Fallback

If wall-clock time is the priority, remote CUDA is the practical answer. To keep the science clean, rerun the compared models on the same CUDA setup rather than comparing a new CUDA 300-mixed model against the already-completed MPS 300-clean model. A controlled CUDA rerun of 300-clean and 300-mixed would answer the dataset question without the Apple MPS speed penalty.

## Final Recommendation

For the current 300-mixed comparison against the existing 300-clean checkpoint, keep the original scientific configuration: 480x640 images, full visual encoder training, batch size 8, FP32, no AMP, no compile, no resize, no SDPA rewrite. If exact reproducibility against the 300-clean MPS run matters, keep the loader behavior unchanged too; otherwise `num_workers=2` is the only measured safe engineering speed tweak, but it should be paired with explicit worker seeding.

MPS is unlikely to approach RTX 3060 training time for this ACT setup without changing the experiment. The measured safe knobs do not close the gap. The big levers are image resolution, freezing the visual encoder, and larger batches, all of which change training conditions and require rerunning every baseline. For final training at the original config, use remote CUDA if the goal is to finish in hours rather than overnight.
