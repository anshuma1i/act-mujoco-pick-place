import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REQUESTED_DEVICE = os.environ.get("DEVICE")
REQUESTED_MUJOCO_GL = os.environ.get("MUJOCO_GL")

from config.config import TASK_CONFIG
from sim.env import (
    CUBE_X_RANGE,
    CUBE_Y_RANGE,
    MIN_CUBE_TARGET_DIST,
    TARGET_X_RANGE,
    TARGET_Y_RANGE,
    PandaPickPlaceEnv,
)
from sim.scripted_expert import PickPlaceExpert, run_expert_episode


EPISODE_PREFIX = "episode_"
EPISODE_SUFFIX = ".hdf5"
DEMO_TYPES = ("clean", "recovery", "hard_case")
RECOVERY_PERTURBATIONS = (
    "cube_shift",
    "gripper_offset",
    "approach_offset",
    "partial_grasp",
)


def _episode_index(path):
    """Return the integer index for an episode_N.hdf5 path, or None."""
    name = path.name
    if not name.startswith(EPISODE_PREFIX) or not name.endswith(EPISODE_SUFFIX):
        return None
    try:
        return int(name[len(EPISODE_PREFIX):-len(EPISODE_SUFFIX)])
    except ValueError:
        return None


def _find_episode_indices(data_dir):
    indices = []
    for path in Path(data_dir).glob(f"{EPISODE_PREFIX}*{EPISODE_SUFFIX}"):
        idx = _episode_index(path)
        if idx is not None:
            indices.append(idx)
    return sorted(indices)


def _next_missing_index(existing_indices, target_count):
    existing = set(existing_indices)
    for idx in range(target_count):
        if idx not in existing:
            return idx
    return None


def _load_previous_attempts(metadata_path, existing_target_count):
    if not metadata_path.exists():
        return existing_target_count
    try:
        with metadata_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
        return int(metadata.get("total_attempts", existing_target_count))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return existing_target_count


def _read_episode_metadata(path):
    try:
        with h5py.File(path, "r") as root:
            metadata_json = root.attrs.get("metadata_json")
            if metadata_json:
                return json.loads(metadata_json)
            return {
                "demo_type": root.attrs.get("demo_type"),
                "success": bool(root.attrs.get("success", False)),
                "perturbation_type": root.attrs.get("perturbation_type"),
                "episode_index": int(root.attrs.get("episode_index", -1)),
                "episode_length": int(root.attrs.get("episode_length", 0)),
            }
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def _scan_successful_episode_metadata(data_dir, target_count):
    episodes = []
    counts = {demo_type: 0 for demo_type in DEMO_TYPES}
    for idx in _find_episode_indices(data_dir):
        if not 0 <= idx < target_count:
            continue
        path = Path(data_dir) / f"{EPISODE_PREFIX}{idx}{EPISODE_SUFFIX}"
        metadata = _read_episode_metadata(path)
        if metadata is None:
            continue
        if metadata.get("success") is True and metadata.get("demo_type") in counts:
            counts[metadata["demo_type"]] += 1
            episodes.append(metadata)
    return counts, episodes


def _count_failed_records(failed_dir):
    if not failed_dir:
        return 0
    path = Path(failed_dir) / "failed_attempts.jsonl"
    if not path.exists():
        return 0
    try:
        with path.open("r", encoding="utf-8") as f:
            return sum(1 for _ in f)
    except OSError:
        return 0


def _build_metadata(data_dir, num_demos, attempts, cfg, task_name, command):
    episode_indices = _find_episode_indices(data_dir)
    target_indices = [idx for idx in episode_indices if 0 <= idx < num_demos]
    total_attempts = max(attempts, len(target_indices))
    success_rate = len(target_indices) / total_attempts if total_attempts else 0.0

    return {
        "num_successful_demos": len(target_indices),
        "target_num_demos": num_demos,
        "total_attempts": total_attempts,
        "success_rate_of_expert_collection": success_rate,
        "environment_name": "PandaPickPlaceEnv",
        "task_name": task_name,
        "action_dimension": cfg["action_dim"],
        "observation_fields": {
            "qpos": f"{cfg['state_dim']}",
            "qvel": f"{cfg['state_dim']}",
            "images": cfg["camera_names"],
        },
        "image_resolution": {
            "height": cfg["cam_height"],
            "width": cfg["cam_width"],
            "channels": 3,
        },
        "episode_length": cfg["episode_len"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "collection_command_used": command,
        "collection_environment": {
            "requested_DEVICE": REQUESTED_DEVICE,
            "torch_DEVICE_after_config_import": os.environ.get("DEVICE"),
            "PYTORCH_ENABLE_MPS_FALLBACK": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"),
            "requested_MUJOCO_GL": REQUESTED_MUJOCO_GL,
            "mujoco_renderer": "macOS CoreGraphics/CGL offscreen renderer when run from a GUI-capable session",
        },
    }


def _write_metadata(data_dir, num_demos, attempts, cfg, task_name, command):
    metadata_path = Path(data_dir) / "metadata.json"
    metadata = _build_metadata(data_dir, num_demos, attempts, cfg, task_name, command)
    tmp_path = metadata_path.with_suffix(".json.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
        f.write("\n")
    os.replace(tmp_path, metadata_path)


def _clip_xy(xy, x_range, y_range):
    return np.array([
        np.clip(xy[0], *x_range),
        np.clip(xy[1], *y_range),
    ])


def _sample_hard_scene(rng):
    """Sample reachable but more difficult cube/target positions."""
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

        cube_xy = np.array([cube_x, cube_y])
        target_xy = np.array([target_x, target_y])
        dist = np.linalg.norm(cube_xy - target_xy)
        if dist >= max(0.22, MIN_CUBE_TARGET_DIST):
            return cube_xy, target_xy

    return np.array([0.37, -0.17]), np.array([0.63, 0.17])


def _apply_recovery_perturbation(env, rng):
    perturbation_type = rng.choice(RECOVERY_PERTURBATIONS)
    details = {}

    if perturbation_type == "cube_shift":
        shift = rng.uniform(-0.025, 0.025, size=2)
        cube_xy = _clip_xy(env.get_cube_pos()[:2] + shift, CUBE_X_RANGE, CUBE_Y_RANGE)
        env.set_cube_pose(cube_xy, settle_steps=20)
        details["cube_shift_xy"] = shift.tolist()
    elif perturbation_type == "gripper_offset":
        delta = rng.normal(0.0, 0.035, size=7)
        env.perturb_arm_qpos(delta)
        details["arm_qpos_delta"] = delta.tolist()
    elif perturbation_type == "approach_offset":
        delta = np.zeros(7)
        delta[[0, 1, 3, 5]] = rng.normal(0.0, 0.045, size=4)
        env.perturb_arm_qpos(delta)
        details["arm_qpos_delta"] = delta.tolist()
    elif perturbation_type == "partial_grasp":
        shift = rng.uniform(-0.018, 0.018, size=2)
        cube_xy = _clip_xy(env.get_cube_pos()[:2] + shift, CUBE_X_RANGE, CUBE_Y_RANGE)
        env.set_cube_pose(cube_xy, settle_steps=10)
        delta = rng.normal(0.0, 0.025, size=7)
        env.perturb_arm_qpos(delta)
        details["cube_shift_xy"] = shift.tolist()
        details["arm_qpos_delta"] = delta.tolist()

    return perturbation_type, details


def _run_configured_expert_episode(env, demo_type, max_steps, seed, verbose=False):
    rng = np.random.default_rng(seed)
    env.reset(seed=seed)

    perturbation_type = "none"
    perturbation_details = {}

    if demo_type == "recovery":
        perturbation_type, perturbation_details = _apply_recovery_perturbation(env, rng)
    elif demo_type == "hard_case":
        cube_xy, target_xy = _sample_hard_scene(rng)
        env.set_cube_and_target(cube_xy, target_xy, settle_steps=50)
        perturbation_type = "hard_object_goal_position"
        perturbation_details = {
            "cube_xy": cube_xy.tolist(),
            "target_xy": target_xy.tolist(),
            "object_goal_distance": float(np.linalg.norm(cube_xy - target_xy)),
        }

    obs = env.get_obs()
    expert = PickPlaceExpert(env)
    obs_list = []
    action_list = []

    for t in range(max_steps):
        action = expert.get_action()
        obs_list.append(obs)
        action_list.append(action)
        obs, success = env.step(action)

        if verbose and t % 30 == 0:
            cube_pos = env.get_cube_pos()
            target_pos = env.get_target_pos()
            dist = np.linalg.norm(cube_pos[:2] - target_pos[:2])
            print(f"  t={t:3d} type={demo_type:9s} phase={expert.phase.name:12s} dist={dist:.3f}")

    success = env.check_success()
    cube_pos = env.get_cube_pos()
    target_pos = env.get_target_pos()
    final_dist = float(np.linalg.norm(cube_pos[:2] - target_pos[:2]))

    metadata = {
        "demo_type": demo_type,
        "success": bool(success),
        "perturbation_type": perturbation_type,
        "episode_length": len(obs_list),
        "final_distance_to_target_xy": final_dist,
        "perturbation_details": perturbation_details,
    }
    return obs_list, action_list, success, metadata


def _write_failure_record(failed_dir, record):
    if not failed_dir:
        return
    failed_dir = Path(failed_dir)
    failed_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = failed_dir / "failed_attempts.jsonl"
    with jsonl_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    json_path = failed_dir / f"failed_attempt_{record['attempt']:06d}.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
        f.write("\n")


def _build_mixed_metadata(data_dir, targets, attempts, failed_attempts, cfg,
                          command, failed_dir=None):
    total_target = sum(targets.values())
    counts, episode_metadata = _scan_successful_episode_metadata(data_dir, total_target)
    total_success = sum(counts.values())
    success_rate = total_success / attempts if attempts else 0.0
    return {
        "total_successful_demos": total_success,
        "num_clean": counts["clean"],
        "num_recovery": counts["recovery"],
        "num_hard_case": counts["hard_case"],
        "target_num_clean": targets["clean"],
        "target_num_recovery": targets["recovery"],
        "target_num_hard_case": targets["hard_case"],
        "total_attempts": attempts,
        "failed_attempts": failed_attempts,
        "expert_collection_success_rate": success_rate,
        "environment_name": "PandaPickPlaceEnv",
        "robot": "Franka Panda",
        "action_dim": cfg["action_dim"],
        "qpos_dim": cfg["state_dim"],
        "qvel_dim": cfg["state_dim"],
        "image_keys": cfg["camera_names"],
        "image_resolution": f"{cfg['cam_height']}x{cfg['cam_width']}x3",
        "collection_timestamp": datetime.now(timezone.utc).isoformat(),
        "collection_command": command,
        "failed_analysis_dir": str(failed_dir) if failed_dir else None,
        "episode_metadata_count": len(episode_metadata),
        "collection_environment": {
            "requested_DEVICE": REQUESTED_DEVICE,
            "torch_DEVICE_after_config_import": os.environ.get("DEVICE"),
            "PYTORCH_ENABLE_MPS_FALLBACK": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"),
            "requested_MUJOCO_GL": REQUESTED_MUJOCO_GL,
            "mujoco_renderer": "macOS CoreGraphics/CGL offscreen renderer when run from a GUI-capable session",
        },
    }


def _load_mixed_attempts(metadata_path, existing_successes, failed_dir):
    failed_records = _count_failed_records(failed_dir)
    if not metadata_path.exists():
        return existing_successes + failed_records, failed_records
    try:
        with metadata_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
        attempts = int(metadata.get("total_attempts", existing_successes + failed_records))
        failed_attempts = int(metadata.get("failed_attempts", failed_records))
        return max(attempts, existing_successes + failed_attempts), max(failed_attempts, failed_records)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return existing_successes + failed_records, failed_records


def _write_mixed_metadata(data_dir, targets, attempts, failed_attempts, cfg,
                          command, failed_dir=None):
    metadata_path = Path(data_dir) / "metadata.json"
    metadata = _build_mixed_metadata(
        data_dir, targets, attempts, failed_attempts, cfg, command, failed_dir
    )
    tmp_path = metadata_path.with_suffix(".json.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
        f.write("\n")
    os.replace(tmp_path, metadata_path)


def _next_demo_type(counts, targets):
    for demo_type in DEMO_TYPES:
        if counts[demo_type] < targets[demo_type]:
            return demo_type
    return None


def collect_mixed_episodes(num_clean, num_recovery, num_hard, dataset_dir,
                           failed_dir=None, task_name="pick_place",
                           max_steps=None, verbose=False, render=False,
                           command=None):
    """Collect clean, recovery, and hard-case successful demonstrations."""
    cfg = TASK_CONFIG
    if max_steps is None:
        max_steps = cfg["episode_len"]

    targets = {
        "clean": int(num_clean),
        "recovery": int(num_recovery),
        "hard_case": int(num_hard),
    }
    total_target = sum(targets.values())
    data_dir = Path(dataset_dir) if dataset_dir else Path(cfg["dataset_dir"]) / task_name
    data_dir.mkdir(parents=True, exist_ok=True)
    if failed_dir:
        Path(failed_dir).mkdir(parents=True, exist_ok=True)

    existing_indices = _find_episode_indices(data_dir)
    counts, _ = _scan_successful_episode_metadata(data_dir, total_target)
    existing_successes = sum(counts.values())
    attempts, failed_attempts = _load_mixed_attempts(
        data_dir / "metadata.json", existing_successes, failed_dir
    )

    for demo_type in DEMO_TYPES:
        if counts[demo_type] > targets[demo_type]:
            raise ValueError(
                f"{data_dir} already has {counts[demo_type]} {demo_type} demos, "
                f"which exceeds requested {targets[demo_type]}"
            )

    extra_indices = [idx for idx in existing_indices if idx >= total_target]
    print(f"Data directory: {data_dir}")
    print(f"Failed analysis directory: {failed_dir or '(metadata disabled)'}")
    print(f"Targets: clean={num_clean}, recovery={num_recovery}, hard_case={num_hard}")
    print(
        f"Existing: clean={counts['clean']} / {targets['clean']}, "
        f"recovery={counts['recovery']} / {targets['recovery']}, "
        f"hard-case={counts['hard_case']} / {targets['hard_case']}"
    )
    if extra_indices:
        print(f"Warning: found {len(extra_indices)} episode files with index >= {total_target}; leaving them untouched.")

    if existing_successes == total_target:
        _write_mixed_metadata(data_dir, targets, attempts, failed_attempts, cfg, command, failed_dir)
        print("Mixed dataset already has the requested successful demos; nothing to collect.")
        return

    env = PandaPickPlaceEnv(
        cam_width=cfg["cam_width"],
        cam_height=cfg["cam_height"],
        render_gui=render,
    )

    pbar = tqdm(total=total_target, initial=existing_successes, desc="Collecting mixed demos")

    try:
        while sum(counts.values()) < total_target:
            demo_type = _next_demo_type(counts, targets)
            if demo_type is None:
                break

            episode_idx = _next_missing_index(_find_episode_indices(data_dir), total_target)
            if episode_idx is None:
                raise RuntimeError(f"No free episode index remains below {total_target}")

            attempts += 1
            seed = 2000 + attempts
            obs_list, action_list, success, episode_metadata = _run_configured_expert_episode(
                env, demo_type, max_steps=max_steps, seed=seed, verbose=False
            )
            episode_metadata.update({
                "episode_index": episode_idx,
                "episode_length": len(obs_list),
            })

            if not success:
                failed_attempts += 1
                failure_record = {
                    "demo_type": "failed_analysis_only",
                    "requested_demo_type": demo_type,
                    "success": False,
                    "failure_reason": "expert_did_not_reach_target",
                    "perturbation_type": episode_metadata.get("perturbation_type", "none"),
                    "attempt": attempts,
                    "seed": seed,
                    "episode_length": len(obs_list),
                    "final_distance_to_target_xy": episode_metadata.get("final_distance_to_target_xy"),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                _write_failure_record(failed_dir, failure_record)
                _write_mixed_metadata(data_dir, targets, attempts, failed_attempts, cfg, command, failed_dir)
                print(
                    f"Failed attempt {failed_attempts}: requested={demo_type}, "
                    f"perturbation={failure_record['perturbation_type']}"
                )
            else:
                save_episode(data_dir, episode_idx, obs_list, action_list, cfg, episode_metadata)
                counts[demo_type] += 1
                pbar.update(1)
                _write_mixed_metadata(data_dir, targets, attempts, failed_attempts, cfg, command, failed_dir)

            print(
                f"Clean: {counts['clean']} / {targets['clean']} | "
                f"Recovery: {counts['recovery']} / {targets['recovery']} | "
                f"Hard-case: {counts['hard_case']} / {targets['hard_case']} | "
                f"Total successful: {sum(counts.values())} / {total_target} | "
                f"Failed attempts: {failed_attempts}"
            )
    finally:
        pbar.close()
        env.close()

    _write_mixed_metadata(data_dir, targets, attempts, failed_attempts, cfg, command, failed_dir)
    print("\nMixed collection complete!")
    print(f"  Clean: {counts['clean']} / {targets['clean']}")
    print(f"  Recovery: {counts['recovery']} / {targets['recovery']}")
    print(f"  Hard-case: {counts['hard_case']} / {targets['hard_case']}")
    print(f"  Total successful: {sum(counts.values())} / {total_target}")
    print(f"  Total attempts: {attempts}")
    print(f"  Failed attempts: {failed_attempts}")
    print(f"  Saved to: {data_dir}")


def collect_episodes(num_demos, task_name, dataset_dir=None, max_steps=None,
                     verbose=False, render=False, command=None):
    """Collect demonstration episodes and save to HDF5 files.

    Args:
        num_demos: target number of successful episodes in the dataset
        task_name: task name (used for data directory if dataset_dir is unset)
        dataset_dir: output directory for HDF5 episodes
        max_steps: max timesteps per episode (default from config)
        verbose: print per-episode info
        render: open MuJoCo GUI viewer to watch collection live
        command: command string to record in metadata
    """
    cfg = TASK_CONFIG
    if max_steps is None:
        max_steps = cfg["episode_len"]

    data_dir = Path(dataset_dir) if dataset_dir else Path(cfg["dataset_dir"]) / task_name
    data_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = data_dir / "metadata.json"
    failures_path = data_dir / "failed_attempts.jsonl"

    existing_indices = _find_episode_indices(data_dir)
    target_existing = [idx for idx in existing_indices if 0 <= idx < num_demos]
    extra_indices = [idx for idx in existing_indices if idx >= num_demos]
    missing = [idx for idx in range(num_demos) if idx not in set(target_existing)]
    previous_attempts = _load_previous_attempts(metadata_path, len(target_existing))

    print(f"Data directory: {data_dir}")
    print(f"Target successful demos: {num_demos}")
    print(f"Existing target episodes: {len(target_existing)} / {num_demos}")
    if extra_indices:
        print(f"Warning: found {len(extra_indices)} episode files with index >= {num_demos}; leaving them untouched.")
    if not missing:
        _write_metadata(data_dir, num_demos, previous_attempts, cfg, task_name, command)
        print("Dataset already has the requested successful demos; nothing to collect.")
        return

    env = PandaPickPlaceEnv(
        cam_width=cfg["cam_width"],
        cam_height=cfg["cam_height"],
        render_gui=render,
    )

    run_attempts = 0
    cumulative_attempts = previous_attempts
    n_success_start = len(target_existing)
    n_success = n_success_start

    pbar = tqdm(total=num_demos, initial=n_success, desc="Collecting demos")

    try:
        while n_success < num_demos:
            run_attempts += 1
            cumulative_attempts += 1
            episode_idx = _next_missing_index(_find_episode_indices(data_dir), num_demos)
            if episode_idx is None:
                break
            seed = 1000 + cumulative_attempts

            obs_list, action_list, success = run_expert_episode(
                env, max_steps=max_steps, seed=seed, verbose=False
            )

            cube_pos = env.get_cube_pos()
            target_pos = env.get_target_pos()
            dist = float(np.linalg.norm(cube_pos[:2] - target_pos[:2]))

            if not success:
                failure = {
                    "attempt": cumulative_attempts,
                    "seed": seed,
                    "distance_to_target_xy": dist,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                with failures_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(failure) + "\n")
                if verbose:
                    print(f"  Attempt {cumulative_attempts}: FAILED (discarding, dist={dist:.3f})")
                continue

            save_episode(data_dir, episode_idx, obs_list, action_list, cfg)

            n_success += 1
            pbar.update(1)
            print(f"Collected {n_success} / {num_demos} successful demos")
            _write_metadata(data_dir, num_demos, cumulative_attempts, cfg, task_name, command)

            if verbose:
                print(f"  Episode {episode_idx}: SUCCESS (dist={dist:.3f})")
    finally:
        pbar.close()
        env.close()

    newly_collected = n_success - n_success_start
    success_rate = n_success / cumulative_attempts if cumulative_attempts else 0.0
    _write_metadata(data_dir, num_demos, cumulative_attempts, cfg, task_name, command)

    print("\nCollection complete!")
    print(f"  New successful demos: {newly_collected}")
    print(f"  Successful demos in dataset: {n_success} / {num_demos}")
    print(f"  Total attempts recorded: {cumulative_attempts}")
    print(f"  Success rate: {success_rate:.0%}")
    print(f"  Saved to: {data_dir}")


def save_episode(data_dir, episode_idx, obs_list, action_list, cfg, episode_metadata=None):
    """Save a single episode to HDF5 in ACT format.

    HDF5 structure (matching training/utils.py EpisodicDataset):
        attrs['sim'] = True
        /observations/qpos          (T, state_dim)      float64
        /observations/qvel          (T, state_dim)      float64
        /observations/images/front  (T, H, W, 3)        uint8
        /action                     (T, action_dim)      float64
    """
    T = len(obs_list)
    state_dim = cfg['state_dim']
    action_dim = cfg['action_dim']
    cam_h = cfg['cam_height']
    cam_w = cfg['cam_width']

    dataset_path = Path(data_dir) / f"{EPISODE_PREFIX}{episode_idx}{EPISODE_SUFFIX}"
    tmp_path = dataset_path.with_suffix(".hdf5.tmp")
    if dataset_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing episode: {dataset_path}")

    with h5py.File(tmp_path, "w", rdcc_nbytes=1024**2 * 2) as root:
        root.attrs["sim"] = True
        if episode_metadata is not None:
            metadata = dict(episode_metadata)
            metadata.setdefault("episode_index", episode_idx)
            metadata.setdefault("episode_length", T)
            root.attrs["metadata_json"] = json.dumps(metadata)
            for key, value in metadata.items():
                if key == "perturbation_details":
                    continue
                if isinstance(value, (str, np.str_)):
                    root.attrs[key] = str(value)
                elif isinstance(value, (int, float, bool, np.integer, np.floating, np.bool_)):
                    root.attrs[key] = value

        obs_grp = root.create_group("observations")
        img_grp = obs_grp.create_group("images")

        # Create datasets
        qpos_ds = obs_grp.create_dataset(
            "qpos", (T, state_dim), dtype="float64"
        )
        qvel_ds = obs_grp.create_dataset(
            "qvel", (T, state_dim), dtype="float64"
        )
        action_ds = root.create_dataset(
            "action", (T, action_dim), dtype="float64"
        )

        # Image datasets with chunking for compression
        for cam_name in cfg["camera_names"]:
            img_grp.create_dataset(
                cam_name, (T, cam_h, cam_w, 3), dtype="uint8",
                chunks=(1, cam_h, cam_w, 3),
            )

        # Write data
        for t, (obs, action) in enumerate(zip(obs_list, action_list)):
            qpos_ds[t] = obs["qpos"]
            qvel_ds[t] = obs["qvel"]
            action_ds[t] = action
            for cam_name in cfg["camera_names"]:
                img_grp[cam_name][t] = obs["images"][cam_name]

    os.replace(tmp_path, dataset_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect demonstration episodes")
    parser.add_argument("--task", type=str, default="pick_place",
                        help="Task name for data directory when --dataset_dir is not set")
    parser.add_argument("--num_demos", "--num_episodes", dest="num_demos",
                        type=int, default=50,
                        help="Target number of successful demos in the dataset")
    parser.add_argument("--num_clean", type=int, default=None,
                        help="Number of clean successful demos for mixed collection")
    parser.add_argument("--num_recovery", type=int, default=None,
                        help="Number of recovery successful demos for mixed collection")
    parser.add_argument("--num_hard", type=int, default=None,
                        help="Number of hard-case successful demos for mixed collection")
    parser.add_argument("--dataset_dir", type=str, default=None,
                        help="Output directory for HDF5 episodes")
    parser.add_argument("--failed_dir", type=str, default=None,
                        help="Directory for failed-attempt analysis metadata")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-episode info")
    parser.add_argument("--render", action="store_true",
                        help="Open MuJoCo GUI viewer to watch collection live")
    args = parser.parse_args()

    mixed_args = (args.num_clean, args.num_recovery, args.num_hard)
    if any(value is not None for value in mixed_args):
        collect_mixed_episodes(
            num_clean=args.num_clean or 0,
            num_recovery=args.num_recovery or 0,
            num_hard=args.num_hard or 0,
            dataset_dir=args.dataset_dir,
            failed_dir=args.failed_dir,
            task_name=args.task,
            verbose=args.verbose,
            render=args.render,
            command=" ".join(sys.argv),
        )
    else:
        collect_episodes(
            num_demos=args.num_demos,
            task_name=args.task,
            dataset_dir=args.dataset_dir,
            verbose=args.verbose,
            render=args.render,
            command=" ".join(sys.argv),
        )
