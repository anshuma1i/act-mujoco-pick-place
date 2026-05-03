import argparse
import json
import os
import pickle
import platform
import sys
import tempfile
import time
from copy import deepcopy


def parse_args():
    parser = argparse.ArgumentParser(description='Train ACT/CNNMLP policies on HDF5 demonstrations')
    parser.add_argument('--task', '--task_name', dest='task', type=str, default='task1')
    parser.add_argument('--dataset_dir', type=str, default=None,
                        help='Dataset directory containing episode_*.hdf5 files')
    parser.add_argument('--ckpt_dir', type=str, default=None,
                        help='Checkpoint output directory')
    parser.add_argument('--policy_class', type=str, default=None,
                        choices=['ACT', 'CNNMLP'],
                        help='Override configured policy class')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Override train and validation batch size')
    parser.add_argument('--device', type=str, default=None,
                        choices=['mps', 'cpu', 'cuda'],
                        help='Requested torch device')
    parser.add_argument('--mps_only', action='store_true',
                        help='Require Apple MPS and fail instead of using CUDA/CPU/fallback')
    parser.add_argument('--eval_every', type=int, default=200,
                        help='Run sim evaluation every N epochs (0 to disable)')
    parser.add_argument('--eval_episodes', type=int, default=3,
                        help='Number of episodes per checkpoint evaluation video')
    parser.add_argument('--num_epochs', type=int, default=None,
                        help='Override configured epoch count for smoke tests')
    parser.add_argument('--save_every', type=int, default=200,
                        help='Save periodic epoch checkpoint every N epochs')
    parser.add_argument('--resume_ckpt', type=str, default=None,
                        help='Load model weights from a checkpoint before training')
    parser.add_argument('--profile_speed', action='store_true',
                        help='Run a short timing benchmark and exit without full training')
    parser.add_argument('--profile_batches', type=int, default=20,
                        help='Number of train batches to time in --profile_speed mode')
    parser.add_argument('--profile_variant', type=str, default='real',
                        choices=['real', 'synthetic_mps', 'dummy_images', 'frozen_vision', 'train_only'],
                        help='Speed benchmark variant')
    parser.add_argument('--profile_output', type=str, default=None,
                        help='Optional JSON path for speed profiling results')
    parser.add_argument('--loader_num_workers', type=int, default=None,
                        help='Override DataLoader num_workers for diagnostics')
    parser.add_argument('--loader_persistent_workers', action='store_true',
                        help='Enable DataLoader persistent_workers when num_workers > 0')
    parser.add_argument('--loader_prefetch_factor', type=int, default=None,
                        help='Override DataLoader prefetch_factor when num_workers > 0')
    parser.add_argument('--loader_pin_memory', action='store_true',
                        help='Force DataLoader pin_memory on for diagnostics')
    parser.add_argument('--image_resize', type=str, default=None,
                        help='Diagnostic-only image resize as HEIGHTxWIDTH, e.g. 240x320')
    parser.add_argument('--speed_benchmark', action='store_true',
                        help='Run fixed-step benchmark and exit without validation/checkpoint/full training')
    parser.add_argument('--benchmark_steps', type=int, default=100,
                        help='Timed optimizer steps for --speed_benchmark')
    parser.add_argument('--benchmark_warmup_steps', type=int, default=10,
                        help='Untimed warmup optimizer steps for --speed_benchmark')
    parser.add_argument('--benchmark_output', type=str, default=None,
                        help='JSON output path for --speed_benchmark')
    parser.add_argument('--benchmark_name', type=str, default=None,
                        help='Human-readable benchmark name')
    parser.add_argument('--benchmark_variant', type=str, default='real',
                        choices=['real', 'synthetic_mps', 'dummy_images', 'frozen_vision'],
                        help='Benchmark input/model variant')
    parser.add_argument('--amp_dtype', type=str, default='none',
                        choices=['none', 'float16', 'bfloat16'],
                        help='Diagnostic autocast dtype')
    parser.add_argument('--channels_last', action='store_true',
                        help='Diagnostic attempt to use channels_last for model conv weights and image tensors')
    parser.add_argument('--torch_compile', action='store_true',
                        help='Diagnostic torch.compile pilot')
    return parser.parse_args()


def configure_environment(args):
    if args.mps_only:
        if args.device and args.device != 'mps':
            raise RuntimeError('--mps_only requires --device mps')
        os.environ['DEVICE'] = 'mps'
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
    elif args.device:
        os.environ['DEVICE'] = args.device


args = parse_args()
configure_environment(args)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from config.config import POLICY_CONFIG, TASK_CONFIG, TRAIN_CONFIG
from training.utils import (
    compute_dict_mean,
    detach_dict,
    load_data,
    make_optimizer,
    make_policy,
    set_seed,
)


def resolve_device():
    requested = args.device or os.environ.get('DEVICE') or POLICY_CONFIG.get('device', 'cpu')
    if args.mps_only:
        fallback = os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')
        if fallback != '0':
            raise RuntimeError(
                f'--mps_only requires PYTORCH_ENABLE_MPS_FALLBACK=0, got {fallback!r}'
            )
        if requested != 'mps':
            raise RuntimeError(f'--mps_only requires selected device mps, got {requested!r}')
        if not torch.backends.mps.is_built():
            raise RuntimeError('PyTorch was not built with MPS support')
        if not torch.backends.mps.is_available():
            raise RuntimeError('Apple MPS is not available; refusing CPU/CUDA fallback')
        if torch.cuda.is_available():
            print('CUDA is visible but will not be used because --mps_only is set.')
        return torch.device('mps')

    return torch.device(requested)


device = resolve_device()
task = args.task
task_cfg = dict(TASK_CONFIG)
train_cfg = dict(TRAIN_CONFIG)
policy_config = dict(POLICY_CONFIG)

if args.num_epochs is not None:
    train_cfg['num_epochs'] = args.num_epochs
if args.batch_size is not None:
    train_cfg['batch_size_train'] = args.batch_size
    train_cfg['batch_size_val'] = args.batch_size
if args.policy_class is not None:
    policy_config['policy_class'] = args.policy_class

policy_config['device'] = device.type
checkpoint_dir = args.ckpt_dir or os.path.join(train_cfg['checkpoint_dir'], task)
data_dir = args.dataset_dir or os.path.join(task_cfg['dataset_dir'], task)


def parse_image_resize(value):
    if not value:
        return None
    try:
        height, width = value.lower().split('x')
        return int(height), int(width)
    except ValueError as exc:
        raise ValueError('--image_resize must be HEIGHTxWIDTH, for example 240x320') from exc


image_resize = parse_image_resize(args.image_resize)


def assert_tensor_ready(name, tensor, require_float32=False):
    if args.mps_only and tensor.device.type != 'mps':
        raise RuntimeError(f'{name} is on {tensor.device}, expected mps')
    if tensor.dtype == torch.float64:
        raise RuntimeError(f'{name} is float64; expected float32 or an intentional mask/index dtype')
    if require_float32 and tensor.dtype != torch.float32:
        raise RuntimeError(f'{name} has dtype {tensor.dtype}, expected float32')


def assert_module_on_device(module):
    if not args.mps_only:
        return
    bad = []
    for name, param in module.named_parameters():
        if param.device.type != 'mps':
            bad.append((f'parameter:{name}', str(param.device)))
    for name, buf in module.named_buffers():
        if buf.device.type != 'mps':
            bad.append((f'buffer:{name}', str(buf.device)))
    if bad:
        details = ', '.join(f'{name}={dev}' for name, dev in bad[:8])
        raise RuntimeError(f'Model contains tensors off MPS: {details}')


def move_batch_to_device(data):
    image_data, qpos_data, action_data, is_pad = data
    image_data = image_data.to(device=device, dtype=torch.float32)
    qpos_data = qpos_data.to(device=device, dtype=torch.float32)
    action_data = action_data.to(device=device, dtype=torch.float32)
    is_pad = is_pad.to(device=device)

    assert_tensor_ready('image batch', image_data, require_float32=True)
    assert_tensor_ready('qpos batch', qpos_data, require_float32=True)
    assert_tensor_ready('action batch', action_data, require_float32=True)
    assert_tensor_ready('is_pad batch', is_pad, require_float32=False)
    return image_data, qpos_data, action_data, is_pad


def maybe_channels_last_image(image_data):
    if not args.channels_last:
        return image_data
    if image_data.ndim == 5:
        b, k, c, h, w = image_data.shape
        flat = image_data.reshape(b * k, c, h, w).contiguous(memory_format=torch.channels_last)
        return flat.reshape(b, k, c, h, w)
    if image_data.ndim == 4:
        return image_data.contiguous(memory_format=torch.channels_last)
    return image_data


def sync_device():
    if device.type == 'mps':
        torch.mps.synchronize()
    elif device.type == 'cuda':
        torch.cuda.synchronize()


def mps_memory_stats():
    if device.type != 'mps':
        return {}
    stats = {}
    for name in ['current_allocated_memory', 'driver_allocated_memory', 'recommended_max_memory']:
        fn = getattr(torch.mps, name, None)
        if fn is not None:
            try:
                stats[f'mps_{name}'] = int(fn())
            except RuntimeError:
                pass
    return stats


def timestamp():
    return time.strftime('%Y-%m-%dT%H:%M:%S%z')


def forward_pass(data, policy):
    image_data, qpos_data, action_data, is_pad = move_batch_to_device(data)
    return policy(qpos_data, image_data, action_data, is_pad)


def tensor_summary(tensor):
    return f'device={tensor.device}, dtype={tensor.dtype}, shape={tuple(tensor.shape)}'


def print_device_diagnostics(policy, sample_batch):
    sample_image, sample_qpos, sample_action, sample_is_pad = move_batch_to_device(sample_batch)
    first_param = next(policy.parameters(), None)
    first_buffer = next(policy.buffers(), None)

    print('\n=== MPS / Training Diagnostics ===')
    print(f'Python version: {sys.version.split()[0]}')
    print(f'torch version: {torch.__version__}')
    print(f'platform: {platform.platform()}')
    print(f'machine: {platform.machine()}')
    print(f'MPS built: {torch.backends.mps.is_built()}')
    print(f'MPS available: {torch.backends.mps.is_available()}')
    print(f'CUDA available: {torch.cuda.is_available()}')
    print(f'selected device: {device}')
    print(f'PYTORCH_ENABLE_MPS_FALLBACK: {os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK")}')
    print(f'model parameter device: {first_param.device if first_param is not None else "none"}')
    print(f'model buffer device: {first_buffer.device if first_buffer is not None else "none"}')
    print(f'sample image tensor: {tensor_summary(sample_image)}')
    print(f'sample qpos tensor: {tensor_summary(sample_qpos)}')
    print(f'sample action tensor: {tensor_summary(sample_action)}')
    print(f'sample is_pad tensor: {tensor_summary(sample_is_pad)}')
    print(f'batch size train/val: {train_cfg["batch_size_train"]}/{train_cfg["batch_size_val"]}')
    print(f'dataset path: {data_dir}')
    print(f'checkpoint directory: {checkpoint_dir}')
    print('==================================\n')


def plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed):
    if not train_history or not validation_history:
        return
    for key in train_history[0]:
        plot_path = os.path.join(ckpt_dir, f'train_val_{key}_seed_{seed}.png')
        plt.figure()
        train_values = [summary[key].item() for summary in train_history]
        val_values = [summary[key].item() for summary in validation_history]
        plt.plot(np.linspace(0, num_epochs - 1, len(train_history)), train_values, label='train')
        plt.plot(np.linspace(0, num_epochs - 1, len(validation_history)), val_values, label='validation')
        plt.tight_layout()
        plt.legend()
        plt.title(key)
        plt.savefig(plot_path)
        plt.close()
    print(f'Saved plots to {ckpt_dir}')


def save_epoch_timing(ckpt_dir, row):
    timing_path = os.path.join(ckpt_dir, 'epoch_timing.csv')
    write_header = not os.path.exists(timing_path)
    fieldnames = [
        'timestamp',
        'epoch',
        'train_time_sec',
        'val_time_sec',
        'checkpoint_time_sec',
        'plot_time_sec',
        'total_epoch_time_sec',
        'samples_per_sec',
        'batches_per_sec',
        'mps_current_allocated_memory',
        'mps_driver_allocated_memory',
        'mps_recommended_max_memory',
    ]
    import csv
    with open(timing_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def run_checkpoint_eval(policy, stats, epoch):
    """Run simulation evaluation and save video at a checkpoint."""
    from sim.env import PandaPickPlaceEnv
    from sim.evaluate_sim import render_eval_video

    print(f"\n--- Checkpoint evaluation at epoch {epoch} ---")
    eval_env = PandaPickPlaceEnv(
        cam_width=task_cfg['cam_width'],
        cam_height=task_cfg['cam_height'],
    )

    video_path = os.path.join(checkpoint_dir, f'eval_epoch_{epoch}.mp4')

    policy.eval()
    successes, *_ = render_eval_video(
        eval_env, policy, stats, policy_config, device,
        output_path=video_path,
        num_episodes=args.eval_episodes,
        max_steps=task_cfg['episode_len'],
        seed_offset=epoch,
    )

    eval_env.close()
    n_success = sum(successes)
    print(f"Epoch {epoch} eval: {n_success}/{len(successes)} success")
    return successes


def train_bc(train_dataloader, val_dataloader, stats, sample_batch):
    policy = make_policy(policy_config['policy_class'], policy_config)
    if args.resume_ckpt:
        loading_status = policy.load_state_dict(
            torch.load(args.resume_ckpt, map_location=device, weights_only=True)
        )
        print(f"Loaded resume checkpoint: {args.resume_ckpt} ({loading_status})")
    policy.to(device)
    assert_module_on_device(policy)
    print_device_diagnostics(policy, sample_batch)

    optimizer = make_optimizer(policy_config['policy_class'], policy)
    os.makedirs(checkpoint_dir, exist_ok=True)

    if args.eval_every <= 0:
        print('Checkpoint simulation evaluation disabled; MuJoCo is not imported during training.')

    train_history = []
    validation_history = []
    min_val_loss = np.inf
    best_ckpt_info = None
    train_start = time.time()

    for epoch in range(train_cfg['num_epochs']):
        epoch_start = time.perf_counter()
        checkpoint_time = 0.0
        plot_time = 0.0
        print(f'\nEpoch {epoch}')
        with torch.inference_mode():
            policy.eval()
            epoch_dicts = []
            val_start = time.perf_counter()
            for data in val_dataloader:
                forward_dict = forward_pass(data, policy)
                epoch_dicts.append(forward_dict)
            sync_device()
            val_time = time.perf_counter() - val_start
            epoch_summary = compute_dict_mean(epoch_dicts)
            validation_history.append(epoch_summary)

            epoch_val_loss = epoch_summary['loss']
            if epoch_val_loss < min_val_loss:
                min_val_loss = epoch_val_loss
                best_ckpt_info = (epoch, min_val_loss, deepcopy(policy.state_dict()))
                best_ckpt_path = os.path.join(checkpoint_dir, 'policy_best.ckpt')
                ckpt_start = time.perf_counter()
                torch.save(best_ckpt_info[2], best_ckpt_path)
                checkpoint_time += time.perf_counter() - ckpt_start
                print(f"Saved best checkpoint so far: epoch {epoch}, val_loss={min_val_loss:.5f}")
        print(f'Val loss:   {epoch_val_loss:.5f}')
        print(''.join(f'{k}: {v.item():.3f} ' for k, v in epoch_summary.items()))

        policy.train()
        optimizer.zero_grad()
        batches_this_epoch = 0
        train_start_epoch = time.perf_counter()
        for data in train_dataloader:
            forward_dict = forward_pass(data, policy)
            loss = forward_dict['loss']
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_history.append(detach_dict(forward_dict))
            batches_this_epoch += 1
        sync_device()
        train_time = time.perf_counter() - train_start_epoch

        epoch_summary = compute_dict_mean(train_history[-batches_this_epoch:])
        epoch_train_loss = epoch_summary['loss']
        print(f'Train loss: {epoch_train_loss:.5f}')
        print(''.join(f'{k}: {v.item():.3f} ' for k, v in epoch_summary.items()))

        if args.save_every > 0 and epoch % args.save_every == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"policy_epoch_{epoch}_seed_{train_cfg['seed']}.ckpt")
            ckpt_start = time.perf_counter()
            torch.save(policy.state_dict(), ckpt_path)
            checkpoint_time += time.perf_counter() - ckpt_start
            plot_start = time.perf_counter()
            plot_history(train_history, validation_history, epoch + 1, checkpoint_dir, train_cfg['seed'])
            plot_time += time.perf_counter() - plot_start

            if args.eval_every > 0 and epoch > 0 and epoch % args.eval_every == 0:
                run_checkpoint_eval(policy, stats, epoch)
                policy.train()
        total_epoch_time = time.perf_counter() - epoch_start
        mem = mps_memory_stats()
        timing_row = {
            'timestamp': timestamp(),
            'epoch': epoch,
            'train_time_sec': f'{train_time:.4f}',
            'val_time_sec': f'{val_time:.4f}',
            'checkpoint_time_sec': f'{checkpoint_time:.4f}',
            'plot_time_sec': f'{plot_time:.4f}',
            'total_epoch_time_sec': f'{total_epoch_time:.4f}',
            'samples_per_sec': f'{(batches_this_epoch * train_cfg["batch_size_train"]) / train_time:.4f}' if train_time else '0',
            'batches_per_sec': f'{batches_this_epoch / train_time:.4f}' if train_time else '0',
            'mps_current_allocated_memory': mem.get('mps_current_allocated_memory', ''),
            'mps_driver_allocated_memory': mem.get('mps_driver_allocated_memory', ''),
            'mps_recommended_max_memory': mem.get('mps_recommended_max_memory', ''),
        }
        save_epoch_timing(checkpoint_dir, timing_row)
        print(
            'Epoch timing: '
            f'train={train_time:.2f}s val={val_time:.2f}s '
            f'ckpt={checkpoint_time:.2f}s plot={plot_time:.2f}s '
            f'total={total_epoch_time:.2f}s '
            f'samples/sec={timing_row["samples_per_sec"]}'
        )

    ckpt_path = os.path.join(checkpoint_dir, 'policy_last.ckpt')
    torch.save(policy.state_dict(), ckpt_path)
    print(f'Saved final checkpoint: {ckpt_path}')

    if best_ckpt_info is not None:
        best_epoch, best_val_loss, best_state_dict = best_ckpt_info
        best_ckpt_path = os.path.join(checkpoint_dir, 'policy_best.ckpt')
        torch.save(best_state_dict, best_ckpt_path)
        print(f"Best checkpoint: epoch {best_epoch}, val_loss={best_val_loss:.5f}")

    plot_history(train_history, validation_history, train_cfg['num_epochs'], checkpoint_dir, train_cfg['seed'])
    print(f'Training time: {(time.time() - train_start) / 60.0:.2f} minutes')

    if args.eval_every > 0:
        run_checkpoint_eval(policy, stats, train_cfg['num_epochs'])


def freeze_vision_parameters(policy):
    frozen = 0
    total = 0
    for name, param in policy.named_parameters():
        total += param.numel()
        if 'backbone' in name or 'input_proj' in name:
            param.requires_grad_(False)
            frozen += param.numel()
    print(f'Frozen visual/backbone parameters for diagnostic: {frozen}/{total}')


def make_synthetic_batch_like(sample_batch):
    image, qpos, action, is_pad = move_batch_to_device(sample_batch)
    image = torch.rand_like(image)
    qpos = torch.rand_like(qpos)
    action = torch.rand_like(action)
    is_pad = torch.zeros_like(is_pad, dtype=torch.bool, device=device)
    return image, qpos, action, is_pad


def autocast_context():
    if args.amp_dtype == 'none':
        return torch.autocast(device_type=device.type, enabled=False)
    dtype = {'float16': torch.float16, 'bfloat16': torch.bfloat16}[args.amp_dtype]
    return torch.autocast(device_type=device.type, dtype=dtype)


def prepare_benchmark_batch(cpu_batch, synthetic_batch=None):
    if args.benchmark_variant == 'synthetic_mps':
        return synthetic_batch
    batch = move_batch_to_device(cpu_batch)
    image_data, qpos_data, action_data, is_pad = batch
    image_data = maybe_channels_last_image(image_data)
    if args.benchmark_variant == 'dummy_images':
        image_data = image_data.new_zeros(image_data.shape)
    return image_data, qpos_data, action_data, is_pad


def benchmark_step(policy, optimizer, batch, do_backward=True):
    image_data, qpos_data, action_data, is_pad = batch
    with autocast_context():
        forward_dict = policy(qpos_data, image_data, action_data, is_pad)
        loss = forward_dict['loss']
    if do_backward:
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    return forward_dict


def finite_status(forward_dict):
    status = {}
    bad_count = 0
    for key, value in forward_dict.items():
        value_detached = value.detach()
        is_finite = torch.isfinite(value_detached)
        bad_count += int((~is_finite).sum().item())
        status[key] = float(value_detached.float().mean().item())
    status['nan_inf_count'] = bad_count
    status['loss_is_finite'] = bool(torch.isfinite(forward_dict['loss']).all().item())
    return status


def run_speed_benchmark(train_dataloader, sample_batch):
    if args.benchmark_variant == 'synthetic_mps' and device.type == 'cpu':
        raise RuntimeError('synthetic_mps benchmark requires a GPU-like device')

    policy = make_policy(policy_config['policy_class'], policy_config)
    policy.to(device)
    if args.channels_last:
        try:
            policy.to(memory_format=torch.channels_last)
        except TypeError as exc:
            print(f'channels_last model conversion skipped: {exc}')
    assert_module_on_device(policy)
    if args.benchmark_variant == 'frozen_vision':
        freeze_vision_parameters(policy)
    if args.torch_compile:
        try:
            policy = torch.compile(
                policy,
                backend='inductor',
                mode='default',
                fullgraph=False,
                dynamic=False,
            )
            print('torch.compile enabled for diagnostic benchmark')
        except Exception as exc:
            print(f'torch.compile setup failed: {exc}')
            raise
    print_device_diagnostics(policy, sample_batch)
    optimizer = make_optimizer(policy_config['policy_class'], policy)
    policy.train()
    optimizer.zero_grad()

    synthetic_batch = None
    if args.benchmark_variant == 'synthetic_mps':
        synthetic_batch = make_synthetic_batch_like(sample_batch)
        synthetic_batch = (
            maybe_channels_last_image(synthetic_batch[0]),
            synthetic_batch[1],
            synthetic_batch[2],
            synthetic_batch[3],
        )

    iterator = iter(train_dataloader)
    timings = {
        'dataloader_wait_sec': 0.0,
        'transfer_to_device_sec': 0.0,
        'forward_sec': 0.0,
        'backward_sec': 0.0,
        'optimizer_sec': 0.0,
    }
    first_status = None
    final_status = None

    total_steps = args.benchmark_warmup_steps + args.benchmark_steps
    timed_start = None
    for step in range(total_steps):
        timed = step >= args.benchmark_warmup_steps
        if timed and timed_start is None:
            sync_device()
            timed_start = time.perf_counter()

        if args.benchmark_variant == 'synthetic_mps':
            batch = synthetic_batch
        else:
            load_start = time.perf_counter()
            try:
                cpu_batch = next(iterator)
            except StopIteration:
                iterator = iter(train_dataloader)
                cpu_batch = next(iterator)
            if timed:
                timings['dataloader_wait_sec'] += time.perf_counter() - load_start

            transfer_start = time.perf_counter()
            batch = prepare_benchmark_batch(cpu_batch)
            sync_device()
            if timed:
                timings['transfer_to_device_sec'] += time.perf_counter() - transfer_start

        image_data, qpos_data, action_data, is_pad = batch
        forward_start = time.perf_counter()
        with autocast_context():
            forward_dict = policy(qpos_data, image_data, action_data, is_pad)
            loss = forward_dict['loss']
        sync_device()
        if timed:
            timings['forward_sec'] += time.perf_counter() - forward_start

        if timed and first_status is None:
            first_status = finite_status(forward_dict)

        backward_start = time.perf_counter()
        loss.backward()
        sync_device()
        if timed:
            timings['backward_sec'] += time.perf_counter() - backward_start

        optim_start = time.perf_counter()
        optimizer.step()
        optimizer.zero_grad()
        sync_device()
        if timed:
            timings['optimizer_sec'] += time.perf_counter() - optim_start
            final_status = finite_status(forward_dict)

        if final_status and not final_status['loss_is_finite']:
            print('Non-finite loss detected; stopping benchmark early.')
            break

    sync_device()
    timed_total = time.perf_counter() - timed_start if timed_start is not None else 0.0
    steps_completed = step - args.benchmark_warmup_steps + 1
    steps_completed = max(0, min(steps_completed, args.benchmark_steps))
    memory_stats = mps_memory_stats()
    output = {
        'benchmark_name': args.benchmark_name or args.benchmark_variant,
        'dataset_dir': data_dir,
        'device': device.type,
        'torch_version': torch.__version__,
        'platform': platform.platform(),
        'machine': platform.machine(),
        'mps_built': bool(torch.backends.mps.is_built()),
        'mps_available': bool(torch.backends.mps.is_available()),
        'PYTORCH_ENABLE_MPS_FALLBACK': os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK'),
        'PYTORCH_MPS_PREFER_METAL': os.environ.get('PYTORCH_MPS_PREFER_METAL'),
        'PYTORCH_MPS_FAST_MATH': os.environ.get('PYTORCH_MPS_FAST_MATH'),
        'batch_size': int(train_cfg['batch_size_train']),
        'image_resolution': list(image_resize) if image_resize else [task_cfg['cam_height'], task_cfg['cam_width']],
        'amp_dtype': args.amp_dtype,
        'sdpa_path': 'not_attempted_detr_site_package_attention',
        'channels_last': bool(args.channels_last),
        'torch_compile': bool(args.torch_compile),
        'benchmark_variant': args.benchmark_variant,
        'benchmark_steps_requested': int(args.benchmark_steps),
        'benchmark_warmup_steps': int(args.benchmark_warmup_steps),
        'benchmark_steps_completed': int(steps_completed),
        'seconds_total_timed': float(timed_total),
        'seconds_per_step': float(timed_total / max(steps_completed, 1)),
        'samples_per_sec': float((steps_completed * train_cfg['batch_size_train']) / timed_total) if timed_total else 0.0,
        'timings_sec': timings,
        'timings_sec_per_step': {
            key: float(value / max(steps_completed, 1))
            for key, value in timings.items()
        },
        'breakdown_percent': {
            key: float(value / sum(timings.values()) * 100.0) if sum(timings.values()) else 0.0
            for key, value in timings.items()
        },
        'mps_memory': memory_stats,
        'first_loss': first_status.get('loss') if first_status else None,
        'final_loss': final_status.get('loss') if final_status else None,
        'first_l1': first_status.get('l1') if first_status else None,
        'final_l1': final_status.get('l1') if final_status else None,
        'first_kl': first_status.get('kl') if first_status else None,
        'final_kl': final_status.get('kl') if final_status else None,
        'loss_is_finite': final_status.get('loss_is_finite') if final_status else None,
        'nan_inf_count': final_status.get('nan_inf_count') if final_status else None,
        'loader_num_workers': args.loader_num_workers if args.loader_num_workers is not None else (1 if device.type == 'cuda' else 0),
        'loader_persistent_workers': bool(args.loader_persistent_workers),
        'loader_prefetch_factor': args.loader_prefetch_factor,
        'loader_pin_memory': bool(args.loader_pin_memory),
    }
    print('\n=== Fixed-Step Speed Benchmark ===')
    print(json.dumps(output, indent=2))
    if args.benchmark_output:
        os.makedirs(os.path.dirname(args.benchmark_output) or '.', exist_ok=True)
        with open(args.benchmark_output, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2)
        print(f'Wrote benchmark JSON: {args.benchmark_output}')
    print('Speed benchmark complete; no full training was run.')


def add_time(bucket, key, value):
    bucket[key] = bucket.get(key, 0.0) + value


def profile_speed(train_dataloader, val_dataloader, sample_batch):
    if args.profile_variant == 'synthetic_mps' and device.type == 'cpu':
        raise RuntimeError('synthetic_mps profile requires a GPU-like device')

    policy = make_policy(policy_config['policy_class'], policy_config)
    policy.to(device)
    assert_module_on_device(policy)
    if args.profile_variant == 'frozen_vision':
        freeze_vision_parameters(policy)
    print_device_diagnostics(policy, sample_batch)
    optimizer = make_optimizer(policy_config['policy_class'], policy)

    timings = {
        'dataloader_wait_sec': 0.0,
        'transfer_to_device_sec': 0.0,
        'forward_loss_sec': 0.0,
        'backward_sec': 0.0,
        'optimizer_step_sec': 0.0,
        'validation_forward_sec': 0.0,
        'checkpoint_save_sec': 0.0,
    }
    train_batches = 0
    profile_start = time.perf_counter()

    policy.train()
    optimizer.zero_grad()
    synthetic_batch = None
    if args.profile_variant == 'synthetic_mps':
        synthetic_batch = make_synthetic_batch_like(sample_batch)

    iterator = iter(train_dataloader)
    for _ in range(args.profile_batches):
        if args.profile_variant == 'synthetic_mps':
            batch = synthetic_batch
        else:
            load_start = time.perf_counter()
            try:
                cpu_batch = next(iterator)
            except StopIteration:
                iterator = iter(train_dataloader)
                cpu_batch = next(iterator)
            add_time(timings, 'dataloader_wait_sec', time.perf_counter() - load_start)

            transfer_start = time.perf_counter()
            batch = move_batch_to_device(cpu_batch)
            if args.profile_variant == 'dummy_images':
                batch = (batch[0].new_zeros(batch[0].shape), batch[1], batch[2], batch[3])
            sync_device()
            add_time(timings, 'transfer_to_device_sec', time.perf_counter() - transfer_start)

        forward_start = time.perf_counter()
        image_data, qpos_data, action_data, is_pad = batch
        forward_dict = policy(qpos_data, image_data, action_data, is_pad)
        sync_device()
        add_time(timings, 'forward_loss_sec', time.perf_counter() - forward_start)

        backward_start = time.perf_counter()
        forward_dict['loss'].backward()
        sync_device()
        add_time(timings, 'backward_sec', time.perf_counter() - backward_start)

        optim_start = time.perf_counter()
        optimizer.step()
        optimizer.zero_grad()
        sync_device()
        add_time(timings, 'optimizer_step_sec', time.perf_counter() - optim_start)
        train_batches += 1

    if args.profile_variant != 'train_only':
        policy.eval()
        with torch.inference_mode():
            val_iterator = iter(val_dataloader)
            for _ in range(min(args.profile_batches, len(val_dataloader))):
                val_cpu_batch = next(val_iterator)
                val_batch = move_batch_to_device(val_cpu_batch)
                val_start = time.perf_counter()
                image_data, qpos_data, action_data, is_pad = val_batch
                _ = policy(qpos_data, image_data, action_data, is_pad)
                sync_device()
                add_time(timings, 'validation_forward_sec', time.perf_counter() - val_start)

    with tempfile.NamedTemporaryFile(prefix='act_profile_', suffix='.ckpt', delete=True) as tmp:
        ckpt_start = time.perf_counter()
        torch.save(policy.state_dict(), tmp.name)
        timings['checkpoint_save_sec'] = time.perf_counter() - ckpt_start

    total_profile_time = time.perf_counter() - profile_start
    timed_total = sum(timings.values())
    results = {
        'variant': args.profile_variant,
        'dataset_dir': data_dir,
        'device': device.type,
        'mps_only': bool(args.mps_only),
        'profile_batches': int(args.profile_batches),
        'train_batches_timed': int(train_batches),
        'batch_size': int(train_cfg['batch_size_train']),
        'image_resize': list(image_resize) if image_resize else None,
        'loader_num_workers': args.loader_num_workers if args.loader_num_workers is not None else (1 if device.type == 'cuda' else 0),
        'loader_persistent_workers': bool(args.loader_persistent_workers),
        'loader_prefetch_factor': args.loader_prefetch_factor,
        'loader_pin_memory': bool(args.loader_pin_memory),
        'total_wall_time_sec': total_profile_time,
        'timings_sec': timings,
        'avg_sec_per_train_batch': total_profile_time / max(train_batches, 1),
        'samples_per_sec': (train_batches * train_cfg['batch_size_train']) / total_profile_time,
        'breakdown_percent': {
            key: (value / timed_total * 100.0 if timed_total else 0.0)
            for key, value in timings.items()
        },
        'mps_memory': mps_memory_stats(),
    }
    print('\n=== Speed Profile Summary ===')
    print(json.dumps(results, indent=2))
    if args.profile_output:
        os.makedirs(os.path.dirname(args.profile_output) or '.', exist_ok=True)
        with open(args.profile_output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        print(f'Wrote profile JSON: {args.profile_output}')
    print('Profile mode complete; no full training was run.')


def main():
    if args.mps_only and data_dir == 'data/panda_pick_place_failed_analysis_only':
        raise RuntimeError('Refusing to train on failed analysis-only data')
    if 'failed_analysis_only' in os.path.normpath(data_dir).split(os.sep):
        raise RuntimeError(f'Refusing to train on failed analysis-only dataset: {data_dir}')
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f'Dataset directory not found: {data_dir}')

    set_seed(train_cfg['seed'])
    os.makedirs(checkpoint_dir, exist_ok=True)

    num_episodes = len([f for f in os.listdir(data_dir) if f.endswith('.hdf5')])
    if num_episodes <= 0:
        raise RuntimeError(f'No HDF5 episodes found in {data_dir}')

    train_dataloader, val_dataloader, stats, _ = load_data(
        data_dir,
        num_episodes,
        task_cfg['camera_names'],
        train_cfg['batch_size_train'],
        train_cfg['batch_size_val'],
        device_type=device.type,
        mps_only=args.mps_only,
        num_workers=args.loader_num_workers,
        persistent_workers=args.loader_persistent_workers,
        prefetch_factor=args.loader_prefetch_factor,
        pin_memory=args.loader_pin_memory if args.loader_pin_memory else None,
        image_resize=image_resize,
    )
    sample_batch = next(iter(train_dataloader))

    if args.speed_benchmark:
        run_speed_benchmark(train_dataloader, sample_batch)
        return

    if args.profile_speed:
        profile_speed(train_dataloader, val_dataloader, sample_batch)
        return

    stats_path = os.path.join(checkpoint_dir, 'dataset_stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)
    print(f'Saved dataset stats: {stats_path}')

    train_bc(train_dataloader, val_dataloader, stats, sample_batch)


if __name__ == '__main__':
    main()
