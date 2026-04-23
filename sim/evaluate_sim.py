"""Evaluate a trained ACT policy in the MuJoCo simulation."""

import os
import sys
import cv2
import torch
import pickle
import numpy as np
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import POLICY_CONFIG, TASK_CONFIG, TRAIN_CONFIG
from sim.env import PandaPickPlaceEnv
from training.utils import make_policy, get_image


def load_policy(ckpt_path, stats_path, policy_config, device):
    """Load a trained policy and normalization stats."""
    policy = make_policy(policy_config['policy_class'], policy_config)
    loading_status = policy.load_state_dict(
        torch.load(ckpt_path, map_location=torch.device(device), weights_only=True)
    )
    print(f"Loaded checkpoint: {ckpt_path} ({loading_status})")
    policy.to(device)
    policy.eval()

    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    return policy, stats


def run_policy_episode(env, policy, stats, policy_config, device,
                       max_steps=300, render_frames=False, seed=None):
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
    obs = env.reset(seed=seed)

    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
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
            [max_steps, max_steps + num_queries, policy_config['action_dim']]
        ).to(device)
        all_time_actions_populated = torch.zeros(
            [max_steps, max_steps + num_queries], dtype=torch.bool, device=device
        )

    frames = []
    all_actions = None
    first_success_step = None

    with torch.inference_mode():
        for t in range(max_steps):
            # Pre-process observation
            qpos_numpy = obs['qpos'].astype(np.float32)
            qpos = pre_process(qpos_numpy)
            qpos = torch.from_numpy(qpos).float().to(device).unsqueeze(0)
            curr_image = get_image(obs['images'], policy_config['camera_names'], device)

            # Query policy
            if t % query_frequency == 0:
                all_actions = policy(qpos, curr_image)

            if policy_config.get('temporal_agg', False):
                all_time_actions[t, t:t + num_queries] = all_actions.squeeze(0)[:num_queries]
                all_time_actions_populated[t, t:t + num_queries] = True
                actions_for_curr_step = all_time_actions[:, t]
                actions_populated = all_time_actions_populated[:, t]
                actions_for_curr_step = actions_for_curr_step[actions_populated]
                k = 0.01
                exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                exp_weights = exp_weights / exp_weights.sum()
                exp_weights = torch.from_numpy(exp_weights.astype(np.float32)).to(device).unsqueeze(dim=1)
                raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
            else:
                raw_action = all_actions[:, t % query_frequency]

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

    return success, frames, final_dist, episode_length


def render_eval_video(env, policy, stats, policy_config, device,
                      output_path, num_episodes=3, max_steps=300, fps=25,
                      seed_offset=0):
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
    """
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    writer = None
    successes = []
    episode_lengths = []

    for ep in range(num_episodes):
        seed = seed_offset + ep + 5000

        success, frames, final_dist, episode_length = run_policy_episode(
            env, policy, stats, policy_config, device,
            max_steps=max_steps, render_frames=True, seed=seed
        )
        successes.append(success)
        episode_lengths.append(episode_length)

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
    print(
        f"Eval video: {output_path} | {n_success}/{num_episodes} success | "
        f"episode length {avg_len:.1f} +/- {std_len:.1f}"
    )
    return successes, episode_lengths


def evaluate_policy(ckpt_path, stats_path, num_episodes=50, max_steps=300,
                    render_video=False, video_path=None, temporal_agg=None,
                    chunk_size=None):
    """Full evaluation: run N episodes, report success rate, optionally save video."""
    cfg = TASK_CONFIG
    policy_config = dict(POLICY_CONFIG)
    if chunk_size is not None:
        policy_config['eval_chunk_size'] = chunk_size
    if temporal_agg is not None:
        policy_config['temporal_agg'] = temporal_agg
    device = os.environ.get('DEVICE', policy_config.get('device', 'cpu'))
    print(f"Temporal aggregation: {policy_config.get('temporal_agg', False)}")
    print(f"Checkpoint num_queries: {policy_config['num_queries']}")
    print(f"Eval chunk size: {policy_config.get('eval_chunk_size', policy_config['num_queries'])}")
    print(f"Stats path: {stats_path}")
    if render_video and video_path:
        print(f"Video path: {video_path}")

    policy, stats = load_policy(ckpt_path, stats_path, policy_config, device)

    env = PandaPickPlaceEnv(
        cam_width=cfg['cam_width'],
        cam_height=cfg['cam_height'],
    )

    if render_video and video_path:
        successes, episode_lengths = render_eval_video(
            env, policy, stats, policy_config, device,
            output_path=video_path,
            num_episodes=num_episodes,
            max_steps=max_steps,
        )
    else:
        successes = []
        episode_lengths = []
        for ep in range(num_episodes):
            success, _, final_dist, episode_length = run_policy_episode(
                env, policy, stats, policy_config, device,
                max_steps=max_steps, render_frames=False, seed=ep + 5000
            )
            successes.append(success)
            episode_lengths.append(episode_length)
            print(
                f"  Episode {ep}: {'SUCCESS' if success else 'FAIL'} "
                f"(dist={final_dist:.3f}, len={episode_length})"
            )

    env.close()

    n_success = sum(successes)
    avg_len = np.mean(episode_lengths) if episode_lengths else 0.0
    std_len = np.std(episode_lengths) if episode_lengths else 0.0
    print(f"\nEvaluation: {n_success}/{num_episodes} = {n_success/num_episodes:.0%} success")
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
    )
