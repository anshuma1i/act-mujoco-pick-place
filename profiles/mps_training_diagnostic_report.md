# ACT MPS Training Speed Diagnostic

## Problem Statement

ACT training for Panda pick-and-place on Apple Silicon MPS took about 848 minutes
for the 300-clean run, versus an approximately comparable RTX 3060 CUDA run
reported at about 3 hours. The goal is to determine whether the slowdown is
caused by MPS backend/model compute, high-resolution image processing,
HDF5/DataLoader throughput, CPU-to-MPS transfer, validation/checkpoint overhead,
batch-size inefficiency, thermal throttling, memory pressure, or hidden config
differences.

## Config Comparability

Local MPS 300-clean configuration was verified from the repo and checkpoint:

- ACT policy
- ResNet18-style visual encoder
- image key: `front`
- image shape: `480x640x3`
- qpos/action dim: `8/8`
- ACT `num_queries`: `100`
- hidden dim: `512`
- feedforward dim: `3200`
- encoder/decoder layers: `4/7`
- heads: `8`
- KL weight: `10`
- batch size: `8`
- epochs: `2000`
- eval during training disabled with `--eval_every 0`
- strict MPS with `PYTORCH_ENABLE_MPS_FALLBACK=0`

No local RTX 3060 training log/config was found in this repository, so exact
CUDA comparability cannot be fully proven from local artifacts. The MPS-vs-CUDA
comparison should still be treated as broadly comparable, but the RTX run should
be checked for image resolution, batch size, epochs, validation frequency,
checkpoint frequency, PyTorch version, and DataLoader settings.

## Long-Run Timing / Thermal Signal

The 300-clean run did not save per-epoch timestamps, but periodic checkpoint file
modification times give a coarse 200-epoch timing signal:

| Epoch Interval | Seconds / Epoch |
| --- | ---: |
| 0-200 | 25.69 |
| 200-400 | 25.23 |
| 400-600 | 25.51 |
| 600-800 | 25.41 |
| 800-1000 | 25.42 |
| 1000-1200 | 25.40 |
| 1200-1400 | 25.38 |
| 1400-1600 | 25.40 |
| 1600-1800 | 25.43 |

Interpretation: epoch time is very stable. There is no strong evidence of
thermal throttling, memory leak, or progressive memory pressure over the run.
The slowdown is likely a stable backend/model-compute or data-pipeline cost.

## Batch-Level Profiling Summary

All profiles used the 300-clean dataset and strict MPS unless noted.

| Profile | Batch | Workers | Resize | Sec / Batch | Samples / Sec |
| --- | ---: | ---: | --- | ---: | ---: |
| real | 8 | 0 | none | 1.089 | 7.34 |
| synthetic MPS tensors | 8 | 0 | none | 1.017 | 7.87 |
| real | 8 | 2 | none | 0.909 | 8.80 |
| real | 8 | 4 | none | 0.940 | 8.51 |
| dummy zero images | 8 | 0 | none | 0.848 | 9.44 |
| frozen visual/backbone | 8 | 0 | none | 0.599 | 13.36 |
| real | 8 | 2 | 240x320 | 0.533 | 15.00 |
| real | 8 | 2 | 224x224 | 0.479 | 16.71 |
| real | 1 | 2 | none | 0.320 | 3.13 |
| real | 2 | 2 | none | 0.458 | 4.37 |
| real | 4 | 2 | none | 0.617 | 6.48 |
| real | 16 | 2 | none | 1.802 | 8.88 |

Representative real-data breakdown at batch size 8, workers 2:

- dataloader wait: 0.2%
- CPU-to-MPS transfer: 0.3%
- forward + loss: 26.5%
- backward: 44.6%
- optimizer step: 6.1%
- validation forward: 17.3%
- checkpoint save: 4.8%

Synthetic tensors already on MPS were only slightly faster than real HDF5 data,
which argues strongly against HDF5 loading or transfer as the main bottleneck.

## DataLoader Audit

Current default MPS DataLoader behavior:

- `num_workers=0`
- `pin_memory=False`
- no `persistent_workers`
- no `prefetch_factor`
- `shuffle=True`
- HDF5 file opened per sample in `__getitem__`
- image dataset is uncompressed HDF5, chunked as `(1, 480, 640, 3)`
- images are converted to float32 and normalized in the Dataset
- tensors are moved to MPS in the training loop

Worker sweep:

- workers 0: 7.34 samples/sec
- workers 2: 8.80 samples/sec
- workers 4: 8.51 samples/sec

Interpretation: `num_workers=2` is a small safe engineering improvement, but
DataLoader is not the core 5x issue.

## Synchronization Audit

Search results:

- `.item()` is used for epoch-level printing and plotting only, not per batch.
- `torch.save` occurs on best checkpoint and periodic checkpoints.
- plotting occurs only at `save_every`.
- no repeated `.cpu()` / `.numpy()` in the training inner loop.
- explicit `torch.mps.synchronize()` is now used only for timing correctness.

Interpretation: synchronization from logging is not the main slowdown. Validation
each epoch is nontrivial, but not enough to explain 5x by itself.

## Bottleneck Ranking

1. MPS backend/model compute for ACT ResNet + Transformer training.
2. Backward pass through the visual/backbone stack at `480x640`.
3. High-resolution image size.
4. Validation each epoch, moderate overhead.
5. Batch size and DataLoader worker tuning, modest impact.
6. HDF5 loading and CPU-to-MPS transfer, low impact in current profiles.
7. Thermal throttling or progressive memory pressure, not supported by current evidence.

## Recommended Fixes

Safe engineering changes:

- Add/keep epoch timing CSV and profile mode.
- Use `num_workers=2`, `persistent_workers=True`, `prefetch_factor=2` if stable.
- Avoid unnecessary per-batch `.item()`, `.cpu()`, `.numpy()`.
- Consider reducing validation frequency for speed-only runs, but keep evaluation protocol documented.
- Keep MuJoCo disabled during offline training.

Experiment-changing changes:

- Resize training images to `240x320` or `224x224`; this roughly doubled diagnostic throughput.
- Increase batch size to 16 if memory and learning dynamics are acceptable.
- Freeze or partially freeze the visual backbone; this substantially reduces backward time.
- Try mixed precision if MPS support is stable for this model.
- Change architecture, `num_queries`, or image encoder only if all comparison runs are rerun.

For the scientific comparison, experiment-changing changes should not be applied
unless 100-clean, 300-clean, and 300-mixed are all rerun under the same new setup.
