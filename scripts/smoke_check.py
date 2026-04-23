"""Fast setup smoke checks for the ACT MuJoCo pick-and-place project."""

from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def check_imports() -> None:
    import cv2
    import detr
    import h5py
    import mujoco

    print(f"torch: {torch.__version__}")
    print(f"mujoco: {mujoco.__version__}")
    print(f"h5py: {h5py.__version__}")
    print(f"cv2: {cv2.__version__}")
    print(f"detr: {detr.__file__}")


def check_device() -> None:
    from config.config import device

    print(f"platform: {platform.platform()} ({platform.machine()})")
    print(f"torch mps built: {torch.backends.mps.is_built()}")
    print(f"torch mps available: {torch.backends.mps.is_available()}")
    print(f"torch cuda available: {torch.cuda.is_available()}")
    print(f"project device: {device}")


def check_mujoco_model() -> None:
    import mujoco
    from sim.env import _build_scene_xml

    model = mujoco.MjModel.from_xml_string(_build_scene_xml())
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    print(f"mujoco model: nq={model.nq}, nv={model.nv}, nu={model.nu}")


def check_policy() -> None:
    from config.config import POLICY_CONFIG, device
    from training.utils import make_policy

    policy = make_policy(POLICY_CONFIG["policy_class"], POLICY_CONFIG)
    policy.to(device)
    print(f"policy: {type(policy).__name__}")


def check_renderer() -> None:
    from sim.env import PandaPickPlaceEnv

    env = PandaPickPlaceEnv()
    obs = env.reset(seed=0)
    print(f"env qpos: {obs['qpos'].shape}")
    print(f"front image: {obs['images']['front'].shape}")
    env.close()


def check_expert(steps: int) -> None:
    from sim.env import PandaPickPlaceEnv
    from sim.scripted_expert import run_expert_episode

    env = PandaPickPlaceEnv(render_images=False)
    obs_list, action_list, success = run_expert_episode(env, max_steps=steps, seed=0)
    env.close()
    print(f"expert smoke: steps={len(action_list)}, obs={len(obs_list)}, success={success}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-policy", action="store_true", help="Instantiate ACT policy")
    parser.add_argument("--expert-steps", type=int, default=0, help="Run a short renderless scripted expert rollout")
    parser.add_argument("--renderer", action="store_true", help="Create the MuJoCo renderer and first image")
    args = parser.parse_args()

    check_imports()
    check_device()
    check_mujoco_model()
    if args.check_policy:
        check_policy()
    if args.expert_steps:
        check_expert(args.expert_steps)
    if args.renderer:
        check_renderer()
    print("smoke checks complete")


if __name__ == "__main__":
    main()
