"""Evaluate a trained ACT policy in the MuJoCo simulation."""

import argparse
import csv
import json
import os
import platform
import sys
import cv2
import pickle
import numpy as np


def _configure_environment_from_argv(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--device', type=str, choices=['mps', 'cpu', 'cuda'])
    parser.add_argument('--mps_only', action='store_true')
    known_args, _ = parser.parse_known_args(argv[1:])
    if known_args.mps_only:
        if known_args.device and known_args.device != 'mps':
            raise RuntimeError('--mps_only requires --device mps')
        os.environ['DEVICE'] = 'mps'
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
    elif known_args.device:
        os.environ['DEVICE'] = known_args.device


_configure_environment_from_argv(sys.argv)

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import POLICY_CONFIG, TASK_CONFIG, TRAIN_CONFIG
from sim.env import (
    CUBE_X_RANGE,
    CUBE_Y_RANGE,
    MIN_CUBE_TARGET_DIST,
    PandaPickPlaceEnv,
    TARGET_X_RANGE,
    TARGET_Y_RANGE,
)
from training.utils import make_policy, get_image


EVAL_SEED_OFFSET = 5000


def resolve_device(device_arg=None, mps_only=False, policy_config=None):
    policy_config = policy_config or {}
    requested = device_arg or os.environ.get('DEVICE') or policy_config.get('device', 'cpu')
    if mps_only:
        fallback = os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')
        if requested != 'mps':
            raise RuntimeError(f'--mps_only requires selected device mps, got {requested!r}')
        if fallback != '0':
            raise RuntimeError(
                f'--mps_only requires PYTORCH_ENABLE_MPS_FALLBACK=0, got {fallback!r}'
            )
        if not torch.backends.mps.is_built():
            raise RuntimeError('PyTorch was not built with MPS support')
        if not torch.backends.mps.is_available():
            raise RuntimeError('Apple MPS is not available; refusing CPU/CUDA fallback')
        return torch.device('mps')
    return torch.device(requested)


def assert_tensor_ready(name, tensor, device, mps_only=False, require_float32=False):
    if mps_only and tensor.device.type != 'mps':
        raise RuntimeError(f'{name} is on {tensor.device}, expected mps')
    if tensor.dtype == torch.float64:
        raise RuntimeError(f'{name} is float64; expected float32 or an intentional mask/index dtype')
    if require_float32 and tensor.dtype != torch.float32:
        raise RuntimeError(f'{name} has dtype {tensor.dtype}, expected float32')
    if tensor.device.type != device.type:
        raise RuntimeError(f'{name} is on {tensor.device}, expected {device}')


def assert_module_on_device(module, mps_only=False):
    if not mps_only:
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


def obs_to_tensors(obs, stats, policy_config, device, mps_only=False):
    qpos_numpy = obs['qpos'].astype(np.float32)
    qpos_numpy = ((qpos_numpy - stats['qpos_mean']) / stats['qpos_std']).astype(np.float32)
    qpos = torch.from_numpy(qpos_numpy).to(device=device, dtype=torch.float32).unsqueeze(0)
    curr_image = get_image(obs['images'], policy_config['camera_names'], device)
    curr_image = curr_image.to(device=device, dtype=torch.float32)
    assert_tensor_ready('eval qpos', qpos, device, mps_only=mps_only, require_float32=True)
    assert_tensor_ready('eval image', curr_image, device, mps_only=mps_only, require_float32=True)
    return qpos, curr_image


def print_eval_diagnostics(policy, ckpt_path, device, sample_qpos, sample_image):
    first_param = next(policy.parameters(), None)
    first_buffer = next(policy.buffers(), None)
    print('\n=== MPS / Evaluation Diagnostics ===')
    print(f'Python version: {sys.version.split()[0]}')
    print(f'torch version: {torch.__version__}')
    print(f'platform: {platform.platform()}')
    print(f'machine: {platform.machine()}')
    print(f'MPS built: {torch.backends.mps.is_built()}')
    print(f'MPS available: {torch.backends.mps.is_available()}')
    print(f'CUDA available: {torch.cuda.is_available()}')
    print(f'selected device: {device}')
    print(f'PYTORCH_ENABLE_MPS_FALLBACK: {os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK")}')
    print(f'checkpoint path: {ckpt_path}')
    print(f'model parameter device: {first_param.device if first_param is not None else "none"}')
    print(f'model buffer device: {first_buffer.device if first_buffer is not None else "none"}')
    print(f'sample qpos tensor: device={sample_qpos.device}, dtype={sample_qpos.dtype}, shape={tuple(sample_qpos.shape)}')
    print(f'sample image tensor: device={sample_image.device}, dtype={sample_image.dtype}, shape={tuple(sample_image.shape)}')
    print('====================================\n')


def infer_num_queries_from_state_dict(state_dict):
    query_embed = state_dict.get('model.query_embed.weight')
    if query_embed is None:
        return None
    return int(query_embed.shape[0])


def _sample_hard_scene(seed):
    """Sample reachable but harder cube/target placements near workspace edges."""
    rng = np.random.default_rng(seed)
    for _ in range(200):
        cube_x = rng.choice([
            rng.uniform(CUBE_X_RANGE[0] + 0.005, CUBE_X_RANGE[0] + 0.055),
            rng.uniform(CUBE_X_RANGE[1] - 0.055, CUBE_X_RANGE[1] - 0.005),
        ])
        cube_y = rng.choice([
            rng.uniform(CUBE_Y_RANGE[0] + 0.005, CUBE_Y_RANGE[0] + 0.065),
            rng.uniform(CUBE_Y_RANGE[1] - 0.065, CUBE_Y_RANGE[1] - 0.005),
            rng.uniform(CUBE_Y_RANGE[0] + 0.04, CUBE_Y_RANGE[1] - 0.04),
        ])
        target_x = rng.choice([
            rng.uniform(TARGET_X_RANGE[0] + 0.005, TARGET_X_RANGE[0] + 0.06),
            rng.uniform(TARGET_X_RANGE[1] - 0.06, TARGET_X_RANGE[1] - 0.005),
        ])
        target_y = rng.choice([
            rng.uniform(TARGET_Y_RANGE[0] + 0.005, TARGET_Y_RANGE[0] + 0.065),
            rng.uniform(TARGET_Y_RANGE[1] - 0.065, TARGET_Y_RANGE[1] - 0.005),
        ])

        cube_xy = np.array([cube_x, cube_y], dtype=np.float64)
        target_xy = np.array([target_x, target_y], dtype=np.float64)
        object_goal_distance = float(np.linalg.norm(cube_xy - target_xy))
        if object_goal_distance >= max(0.22, MIN_CUBE_TARGET_DIST):
            return cube_xy, target_xy, object_goal_distance

    cube_xy = np.array([0.37, -0.17], dtype=np.float64)
    target_xy = np.array([0.63, 0.17], dtype=np.float64)
    return cube_xy, target_xy, float(np.linalg.norm(cube_xy - target_xy))


def reset_eval_episode(env, eval_mode, episode_seed):
    """Reset a normal or hard evaluation episode and return reset metadata."""
    obs = env.reset(seed=episode_seed)
    metadata = {
        'eval_mode': eval_mode,
        'seed': int(episode_seed),
    }
    if eval_mode == 'hard':
        cube_xy, target_xy, object_goal_distance = _sample_hard_scene(episode_seed)
        env.set_cube_and_target(cube_xy, target_xy, settle_steps=50)
        obs = env.get_obs()
        metadata.update({
            'hard_case_type': 'workspace_boundary_long_displacement',
            'cube_xy': cube_xy.tolist(),
            'target_xy': target_xy.tolist(),
            'object_goal_distance': object_goal_distance,
        })
    return obs, metadata


def load_policy(ckpt_path, stats_path, policy_config, device, mps_only=False, state_dict=None):
    """Load a trained policy and normalization stats."""
    if state_dict is None:
        state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    policy = make_policy(policy_config['policy_class'], policy_config)
    loading_status = policy.load_state_dict(state_dict)
    print(f"Loaded checkpoint: {ckpt_path} ({loading_status})")
    policy.to(device)
    policy.eval()
    assert_module_on_device(policy, mps_only=mps_only)

    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    return policy, stats


def run_policy_episode(env, policy, stats, policy_config, device,
                       max_steps=300, render_frames=False, seed=None,
                       mps_only=False, eval_mode='normal'):
    """Run one episode with the trained policy.

    Args:
        env: PandaPickPlaceEnv
        policy: trained ACTPolicy
        stats: normalization stats dict
        policy_config: policy configuration dict
        device: torch device string
        max_steps: max timesteps
        render_frames: if True, collect rendered frames for video

    Returns:
        success: whether cube reached target
        frames: list of BGR frames (empty if render_frames=False)
        final_dist: final cube-target distance
        episode_length: first success timestep, or max_steps if unsuccessful
    """
    obs, episode_metadata = reset_eval_episode(env, eval_mode, seed)

    post_process = lambda a: a * stats['action_std'] + stats['action_mean']

    model_num_queries = policy_config['num_queries']
    eval_chunk_size = policy_config.get('eval_chunk_size', model_num_queries)
    if eval_chunk_size > model_num_queries:
        raise ValueError(
            f"chunk_size ({eval_chunk_size}) cannot exceed checkpoint num_queries ({model_num_queries})"
        )

    query_frequency = eval_chunk_size
    if policy_config.get('temporal_agg', False):
        query_frequency = 1
        num_queries = eval_chunk_size
        all_time_actions = torch.zeros(
            [max_steps, max_steps + num_queries, policy_config['action_dim']],
            device=device,
            dtype=torch.float32,
        )
        all_time_actions_populated = torch.zeros(
            [max_steps, max_steps + num_queries], dtype=torch.bool, device=device
        )
        assert_tensor_ready(
            'temporal action buffer',
            all_time_actions,
            device,
            mps_only=mps_only,
            require_float32=True,
        )
        assert_tensor_ready(
            'temporal action mask',
            all_time_actions_populated,
            device,
            mps_only=mps_only,
        )

    frames = []
    all_actions = None
    first_success_step = None

    with torch.inference_mode():
        for t in range(max_steps):
            # Pre-process observation
            qpos, curr_image = obs_to_tensors(obs, stats, policy_config, device, mps_only=mps_only)

            # Query policy
            if t % query_frequency == 0:
                all_actions = policy(qpos, curr_image)
                assert_tensor_ready(
                    'predicted action chunk',
                    all_actions,
                    device,
                    mps_only=mps_only,
                    require_float32=True,
                )

            if policy_config.get('temporal_agg', False):
                all_time_actions[t, t:t + num_queries] = all_actions.squeeze(0)[:num_queries]
                all_time_actions_populated[t, t:t + num_queries] = True
                actions_for_curr_step = all_time_actions[:, t]
                actions_populated = all_time_actions_populated[:, t]
                actions_for_curr_step = actions_for_curr_step[actions_populated]
                k = 0.01
                exp_weights = torch.arange(
                    len(actions_for_curr_step),
                    device=device,
                    dtype=torch.float32,
                )
                exp_weights = torch.exp(-k * exp_weights)
                exp_weights = (exp_weights / exp_weights.sum()).unsqueeze(dim=1)
                raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
            else:
                raw_action = all_actions[:, t % query_frequency]
            assert_tensor_ready(
                'raw action',
                raw_action,
                device,
                mps_only=mps_only,
                require_float32=True,
            )

            # Post-process action
            raw_action = raw_action.squeeze(0).cpu().numpy()
            action = post_process(raw_action)

            # Render frame before stepping
            if render_frames:
                frame = env.render_camera('front')
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                # Add overlay text
                cube_pos = env.get_cube_pos()
                target_pos = env.get_target_pos()
                dist = np.linalg.norm(cube_pos[:2] - target_pos[:2])
                cv2.putText(frame_bgr, f't={t}',
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame_bgr, f'dist={dist:.3f}',
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                frames.append(frame_bgr)

            # Step environment
            obs, success = env.step(action)
            if success and first_success_step is None:
                first_success_step = t + 1

    # Final success check
    success = env.check_success()
    cube_pos = env.get_cube_pos()
    target_pos = env.get_target_pos()
    final_dist = np.linalg.norm(cube_pos[:2] - target_pos[:2])
    episode_length = first_success_step if first_success_step is not None else max_steps

    episode_metadata.update({
        'success': bool(success),
        'final_cube_goal_distance': float(final_dist),
        'episode_length': int(episode_length),
    })
    return success, frames, final_dist, episode_length, episode_metadata


def render_eval_video(env, policy, stats, policy_config, device,
                      output_path, num_episodes=3, max_steps=300, fps=25,
                      seed_offset=EVAL_SEED_OFFSET, mps_only=False,
                      eval_mode='normal'):
    """Run evaluation episodes and save a compiled video.

    Args:
        env: PandaPickPlaceEnv
        policy: trained ACTPolicy
        stats: normalization stats
        policy_config: policy configuration
        device: torch device string
        output_path: path to save MP4 video
        num_episodes: number of episodes to render
        max_steps: max steps per episode
        fps: video frame rate
        seed_offset: base seed for reproducibility

    Returns:
        successes: list of bool per episode
        episode_lengths: list of first-success timesteps or max_steps
        final_distances: list of final cube-target XY distances
        episode_results: list of per-episode metadata/results
    """
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    writer = None
    successes = []
    episode_lengths = []
    final_distances = []
    episode_results = []

    for ep in range(num_episodes):
        seed = seed_offset + ep

        success, frames, final_dist, episode_length, episode_metadata = run_policy_episode(
            env, policy, stats, policy_config, device,
            max_steps=max_steps, render_frames=True, seed=seed,
            mps_only=mps_only, eval_mode=eval_mode,
        )
        successes.append(success)
        episode_lengths.append(episode_length)
        final_distances.append(final_dist)
        episode_metadata['episode_index'] = ep
        episode_results.append(episode_metadata)

        if not frames:
            continue

        if writer is None and frames:
            h, w = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
            if not writer.isOpened():
                raise RuntimeError(f"Failed to open video writer for {output_path}")

        for f in frames:
            writer.write(f)

    if writer is not None:
        writer.release()

    n_success = sum(successes)
    avg_len = np.mean(episode_lengths)
    std_len = np.std(episode_lengths)
    avg_dist = np.mean(final_distances)
    median_dist = np.median(final_distances)
    print(
        f"Eval video: {output_path} | {n_success}/{num_episodes} success | "
        f"episode length {avg_len:.1f} +/- {std_len:.1f} | "
        f"final dist mean/median {avg_dist:.4f}/{median_dist:.4f}"
    )
    return successes, episode_lengths, final_distances, episode_results


def _write_eval_results(output_dir, summary, episode_results):
    if not output_dir:
        return
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, 'results.json')
    csv_path = os.path.join(output_dir, 'per_episode_results.csv')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    fieldnames = [
        'episode_index',
        'eval_mode',
        'seed',
        'success',
        'final_cube_goal_distance',
        'episode_length',
        'hard_case_type',
        'object_goal_distance',
        'cube_xy',
        'target_xy',
    ]
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in episode_results:
            csv_row = {key: row.get(key, '') for key in fieldnames}
            if isinstance(csv_row.get('cube_xy'), list):
                csv_row['cube_xy'] = json.dumps(csv_row['cube_xy'])
            if isinstance(csv_row.get('target_xy'), list):
                csv_row['target_xy'] = json.dumps(csv_row['target_xy'])
            writer.writerow(csv_row)
    print(f"Results JSON: {json_path}")
    print(f"Per-episode CSV: {csv_path}")


def evaluate_policy(ckpt_path, stats_path, num_episodes=50, max_steps=300,
                    render_video=False, video_path=None, temporal_agg=None,
                    chunk_size=None, device_arg=None, mps_only=False,
                    eval_mode='normal', seed=None):
    """Full evaluation: run N episodes, report success rate, optionally save video."""
    cfg = TASK_CONFIG
    policy_config = dict(POLICY_CONFIG)
    if chunk_size is not None:
        policy_config['eval_chunk_size'] = chunk_size
    if temporal_agg is not None:
        policy_config['temporal_agg'] = temporal_agg
    device = resolve_device(device_arg=device_arg, mps_only=mps_only, policy_config=policy_config)
    policy_config['device'] = device.type
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    checkpoint_num_queries = infer_num_queries_from_state_dict(state_dict)
    if checkpoint_num_queries is not None:
        policy_config['num_queries'] = checkpoint_num_queries
    print(f"Temporal aggregation: {policy_config.get('temporal_agg', False)}")
    print(f"Eval mode: {eval_mode}")
    print(f"Eval seed: {seed if seed is not None else EVAL_SEED_OFFSET}")
    print(f"Checkpoint num_queries: {policy_config['num_queries']}")
    print(f"Eval chunk size: {policy_config.get('eval_chunk_size', policy_config['num_queries'])}")
    print(f"Stats path: {stats_path}")
    if render_video and video_path:
        print(f"Video path: {video_path}")

    policy, stats = load_policy(
        ckpt_path,
        stats_path,
        policy_config,
        device,
        mps_only=mps_only,
        state_dict=state_dict,
    )

    env = PandaPickPlaceEnv(
        cam_width=cfg['cam_width'],
        cam_height=cfg['cam_height'],
    )
    sample_obs = env.reset(seed=4999)
    sample_qpos, sample_image = obs_to_tensors(
        sample_obs,
        stats,
        policy_config,
        device,
        mps_only=mps_only,
    )
    print_eval_diagnostics(policy, ckpt_path, device, sample_qpos, sample_image)

    if render_video and video_path:
        successes, episode_lengths, final_distances, episode_results = render_eval_video(
            env, policy, stats, policy_config, device,
            output_path=video_path,
            num_episodes=num_episodes,
            max_steps=max_steps,
            mps_only=mps_only,
            eval_mode=eval_mode,
            seed_offset=seed if seed is not None else EVAL_SEED_OFFSET,
        )
    else:
        successes = []
        episode_lengths = []
        final_distances = []
        episode_results = []
        for ep in range(num_episodes):
            episode_seed = (seed if seed is not None else EVAL_SEED_OFFSET) + ep
            success, _, final_dist, episode_length, episode_metadata = run_policy_episode(
                env, policy, stats, policy_config, device,
                max_steps=max_steps, render_frames=False, seed=episode_seed,
                mps_only=mps_only, eval_mode=eval_mode,
            )
            successes.append(success)
            episode_lengths.append(episode_length)
            final_distances.append(final_dist)
            episode_metadata['episode_index'] = ep
            episode_results.append(episode_metadata)
            print(
                f"  Episode {ep}: {'SUCCESS' if success else 'FAIL'} "
                f"(dist={final_dist:.3f}, len={episode_length})"
            )

    env.close()

    n_success = sum(successes)
    avg_len = np.mean(episode_lengths) if episode_lengths else 0.0
    std_len = np.std(episode_lengths) if episode_lengths else 0.0
    avg_dist = np.mean(final_distances) if final_distances else 0.0
    median_dist = np.median(final_distances) if final_distances else 0.0
    std_dist = np.std(final_distances) if final_distances else 0.0
    output_dir = os.path.dirname(video_path) if video_path else None
    summary = {
        'checkpoint_path': ckpt_path,
        'stats_path': stats_path,
        'num_episodes': int(num_episodes),
        'eval_mode': eval_mode,
        'seed': int(seed if seed is not None else EVAL_SEED_OFFSET),
        'episode_seeds': [int((seed if seed is not None else EVAL_SEED_OFFSET) + ep)
                          for ep in range(num_episodes)],
        'temporal_aggregation': bool(policy_config.get('temporal_agg', False)),
        'checkpoint_num_queries': int(policy_config['num_queries']),
        'eval_chunk_size': int(policy_config.get('eval_chunk_size', policy_config['num_queries'])),
        'success_count': int(n_success),
        'success_rate': float(n_success / num_episodes),
        'average_final_cube_goal_distance': float(avg_dist),
        'median_final_cube_goal_distance': float(median_dist),
        'std_final_cube_goal_distance': float(std_dist),
        'average_episode_length': float(avg_len),
        'std_episode_length': float(std_len),
        'video_path': video_path,
        'hard_eval_definition': (
            'normal random reset' if eval_mode == 'normal'
            else 'cube and target sampled near valid workspace boundaries with object-goal distance >= 0.22m'
        ),
        'device': device.type,
        'mps_only': bool(mps_only),
        'pytorch_mps_fallback': os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK'),
    }
    _write_eval_results(output_dir, summary, episode_results)
    print(f"\nEvaluation: {n_success}/{num_episodes} = {n_success/num_episodes:.0%} success")
    print(f"Average final cube-goal distance: {avg_dist:.4f}")
    print(f"Median final cube-goal distance: {median_dist:.4f}")
    print(f"Std final cube-goal distance: {std_dist:.4f}")
    print(f"Average episode length: {avg_len:.1f}")
    print(f"Episode length std: {std_len:.1f}")
    return successes


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate trained policy in simulation')
    parser.add_argument('--task', type=str, default='pick_place')
    parser.add_argument('--ckpt', type=str, default=None,
                       help='Checkpoint path (default: policy_last.ckpt)')
    parser.add_argument('--num_episodes', type=int, default=50)
    parser.add_argument('--video', action='store_true', help='Save evaluation video')
    parser.add_argument('--video_path', type=str, default=None)
    parser.add_argument('--chunk_size', type=int, default=None,
                       help='Override ACT chunk size / num_queries for evaluation')
    parser.add_argument('--eval_mode', type=str, default='normal',
                       choices=['normal', 'hard'],
                       help='Evaluation reset mode. normal preserves the standard randomized reset.')
    parser.add_argument('--seed', type=int, default=None,
                       help='Base seed for deterministic evaluation episodes')
    parser.add_argument('--device', type=str, default=None, choices=['mps', 'cpu', 'cuda'],
                       help='Requested torch device')
    parser.add_argument('--mps_only', action='store_true',
                       help='Require Apple MPS and fail instead of using CUDA/CPU/fallback')
    parser.add_argument('--temporal_agg', action='store_true',
                       help='Query ACT every step and aggregate overlapping action chunks')
    parser.add_argument('--no_temporal_agg', action='store_true',
                       help='Disable temporal aggregation even if enabled in config')
    args = parser.parse_args()

    train_cfg = TRAIN_CONFIG
    ckpt_dir = os.path.join(train_cfg['checkpoint_dir'], args.task)

    if args.ckpt:
        ckpt_path = args.ckpt
    else:
        ckpt_path = os.path.join(ckpt_dir, train_cfg['eval_ckpt_name'])

    stats_path = os.path.join(ckpt_dir, 'dataset_stats.pkl')
    if args.ckpt:
        ckpt_stats_path = os.path.join(os.path.dirname(ckpt_path), 'dataset_stats.pkl')
        if os.path.exists(ckpt_stats_path):
            stats_path = ckpt_stats_path

    video_path = args.video_path
    if args.video and not video_path:
        video_path = os.path.join(ckpt_dir, 'eval_video.mp4')

    temporal_agg = None
    if args.temporal_agg:
        temporal_agg = True
    if args.no_temporal_agg:
        temporal_agg = False

    evaluate_policy(
        ckpt_path=ckpt_path,
        stats_path=stats_path,
        num_episodes=args.num_episodes,
        render_video=args.video,
        video_path=video_path,
        temporal_agg=temporal_agg,
        chunk_size=args.chunk_size,
        device_arg=args.device,
        mps_only=args.mps_only,
        eval_mode=args.eval_mode,
        seed=args.seed,
    )
