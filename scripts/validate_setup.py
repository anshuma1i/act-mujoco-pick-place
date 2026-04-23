"""One-command setup validation for the ACT MuJoCo pick-and-place project."""

from __future__ import annotations

import argparse
import os
import pickle
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


class StageError(RuntimeError):
    pass


def status(message: str) -> None:
    print(f"[setup-check] {message}", flush=True)


def clean_task_dir(path: Path, task: str) -> None:
    if path.exists():
        if task != "setup_validation":
            raise StageError(f"Refusing to clean non-default task directory: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def run_stage(name: str, func):
    status(f"START {name}")
    try:
        result = func()
    except Exception as exc:
        status(f"FAILED {name}: {exc}")
        print("\nSETUP VALIDATION FAILED", flush=True)
        print(f"Failed stage: {name}", flush=True)
        raise SystemExit(1) from exc
    status(f"OK {name}")
    return result


def patch_detr() -> None:
    from scripts.patch_detr import main as patch_main

    patch_main()


def dependency_and_device_check() -> None:
    import cv2
    import detr
    import h5py as h5py_module
    import mujoco
    import torch
    from config.config import device

    expected_device = "cpu"
    if torch.backends.mps.is_available():
        expected_device = "mps"
    elif torch.cuda.is_available():
        expected_device = "cuda"

    print(f"  torch={torch.__version__}")
    print(f"  mujoco={mujoco.__version__}")
    print(f"  h5py={h5py_module.__version__}")
    print(f"  cv2={cv2.__version__}")
    print(f"  detr={detr.__file__}")
    print(f"  mps_built={torch.backends.mps.is_built()}")
    print(f"  mps_available={torch.backends.mps.is_available()}")
    print(f"  cuda_available={torch.cuda.is_available()}")
    print(f"  project_device={device}")

    if device != expected_device:
        raise StageError(f"project device is {device}, expected {expected_device}")


def mujoco_model_check() -> None:
    import mujoco
    from sim.env import _build_scene_xml

    model = mujoco.MjModel.from_xml_string(_build_scene_xml())
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    print(f"  model nq={model.nq}, nv={model.nv}, nu={model.nu}")
    if model.nu != 8:
        raise StageError(f"expected 8 actuators, got {model.nu}")


def renderer_check_direct(width: int, height: int) -> None:
    import numpy as np
    from sim.env import PandaPickPlaceEnv

    env = PandaPickPlaceEnv(cam_width=width, cam_height=height)
    try:
        obs = env.reset(seed=0)
        image = obs["images"]["front"]
        print(f"  qpos={obs['qpos'].shape}, image={image.shape}, dtype={image.dtype}, min={image.min()}, max={image.max()}")
        if image.shape != (height, width, 3):
            raise StageError(f"unexpected rendered image shape: {image.shape}")
        if image.dtype != np.uint8:
            raise StageError(f"unexpected rendered image dtype: {image.dtype}")
        if image.max() == image.min():
            raise StageError("rendered image is constant; renderer may not be producing a real frame")
    finally:
        env.close()


def renderer_check(width: int, height: int) -> None:
    env = os.environ.copy()
    env.setdefault("UV_CACHE_DIR", "/tmp/uv-cache")
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--renderer-probe",
        "--width",
        str(width),
        "--height",
        str(height),
    ]
    result = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )
    print(result.stdout)
    if result.returncode != 0:
        stderr = result.stderr.strip()
        if stderr and "Exception ignored in:" not in stderr:
            print(stderr)
        raise StageError(f"renderer probe failed with exit code {result.returncode}")


def collect_one_demo(task: str, width: int, height: int, max_attempts: int) -> Path:
    from config.config import TASK_CONFIG
    from sim.collect_data import save_episode
    from sim.env import PandaPickPlaceEnv
    from sim.scripted_expert import run_expert_episode

    cfg = dict(TASK_CONFIG)
    cfg["cam_width"] = width
    cfg["cam_height"] = height

    data_dir = PROJECT_ROOT / cfg["dataset_dir"] / task
    clean_task_dir(data_dir, task)

    env = PandaPickPlaceEnv(cam_width=width, cam_height=height)
    try:
        for attempt in range(max_attempts):
            seed = 123 + attempt
            obs_list, action_list, success = run_expert_episode(
                env, max_steps=cfg["episode_len"], seed=seed, verbose=False
            )
            print(f"  attempt={attempt + 1}, seed={seed}, success={success}")
            if success:
                save_episode(str(data_dir), 0, obs_list, action_list, cfg)
                return data_dir / "episode_0.hdf5"
    finally:
        env.close()

    raise StageError(f"scripted expert did not produce a successful demo in {max_attempts} attempts")


def inspect_hdf5(path: Path, width: int, height: int) -> None:
    import h5py
    import numpy as np

    with h5py.File(path, "r") as root:
        qpos = root["/observations/qpos"][:]
        qvel = root["/observations/qvel"][:]
        image = root["/observations/images/front"][:]
        action = root["/action"][:]

    print(f"  file={path}")
    print(f"  qpos={qpos.shape}, qvel={qvel.shape}, image={image.shape}, action={action.shape}")

    expected_steps = 300
    if qpos.shape != (expected_steps, 8):
        raise StageError(f"unexpected qpos shape: {qpos.shape}")
    if qvel.shape != (expected_steps, 8):
        raise StageError(f"unexpected qvel shape: {qvel.shape}")
    if image.shape != (expected_steps, height, width, 3):
        raise StageError(f"unexpected image shape: {image.shape}")
    if action.shape != (expected_steps, 8):
        raise StageError(f"unexpected action shape: {action.shape}")
    if not np.isfinite(qpos).all() or not np.isfinite(qvel).all() or not np.isfinite(action).all():
        raise StageError("qpos/qvel/action contain non-finite values")
    if image.dtype != np.uint8:
        raise StageError(f"unexpected image dtype: {image.dtype}")
    if image.max() == image.min():
        raise StageError("saved images are constant; this is not a valid rendered demo")


def run_one_epoch_training(task: str) -> None:
    checkpoints_dir = PROJECT_ROOT / "checkpoints" / task
    clean_task_dir(checkpoints_dir, task)

    env = os.environ.copy()
    env.setdefault("UV_CACHE_DIR", "/tmp/uv-cache")
    env.setdefault("MPLCONFIGDIR", "/tmp/mpl")
    env.setdefault("PYTHONFAULTHANDLER", "1")

    cmd = [
        sys.executable,
        "-u",
        "train.py",
        "--task",
        task,
        "--num_epochs",
        "1",
        "--eval_every",
        "0",
    ]
    result = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise StageError(f"training command failed with exit code {result.returncode}")


def artifact_check(task: str) -> None:
    import torch

    ckpt_dir = PROJECT_ROOT / "checkpoints" / task
    required = [
        "dataset_stats.pkl",
        "policy_epoch_0_seed_42.ckpt",
        "policy_last.ckpt",
        "policy_best.ckpt",
        "train_val_loss_seed_42.png",
        "train_val_l1_seed_42.png",
        "train_val_kl_seed_42.png",
    ]
    for name in required:
        path = ckpt_dir / name
        if not path.exists() or path.stat().st_size == 0:
            raise StageError(f"missing or empty artifact: {path}")
        print(f"  {path.relative_to(PROJECT_ROOT)} ({path.stat().st_size / 1024 / 1024:.1f} MB)")

    with open(ckpt_dir / "dataset_stats.pkl", "rb") as f:
        stats = pickle.load(f)
    expected_keys = {"action_mean", "action_std", "qpos_mean", "qpos_std", "example_qpos"}
    if set(stats) != expected_keys:
        raise StageError(f"unexpected dataset_stats keys: {sorted(stats)}")

    state_dict = torch.load(ckpt_dir / "policy_last.ckpt", map_location="cpu", weights_only=True)
    if not state_dict:
        raise StageError("policy_last.ckpt loaded as an empty state_dict")
    print(f"  loaded checkpoint tensors={len(state_dict)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate ACT + MuJoCo setup with one tiny end-to-end run")
    parser.add_argument("--task", default="setup_validation")
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--height", type=int, default=96)
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--renderer-probe", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.renderer_probe:
        try:
            renderer_check_direct(args.width, args.height)
        except Exception as exc:
            print(f"Renderer probe failed: {exc}", flush=True)
            raise SystemExit(2) from exc
        return

    if args.task != "setup_validation":
        raise SystemExit("This validator only cleans the default task 'setup_validation'.")

    run_stage("patch DETR", patch_detr)
    run_stage("dependencies and device", dependency_and_device_check)
    run_stage("MuJoCo model", mujoco_model_check)
    run_stage("MuJoCo renderer", lambda: renderer_check(args.width, args.height))
    hdf5_path = run_stage(
        "one successful demo collection",
        lambda: collect_one_demo(args.task, args.width, args.height, args.max_attempts),
    )
    run_stage("HDF5 inspection", lambda: inspect_hdf5(hdf5_path, args.width, args.height))
    run_stage("one epoch training", lambda: run_one_epoch_training(args.task))
    run_stage("artifact check", lambda: artifact_check(args.task))

    print("\nSETUP VALIDATION PASSED", flush=True)
    print(f"Artifacts: data/{args.task}/ and checkpoints/{args.task}/", flush=True)


if __name__ == "__main__":
    main()
