"""Verify ACT demonstration dataset count and HDF5 schema."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def verify_episode(path: Path, episode_len: int, height: int, width: int) -> None:
    with h5py.File(path, "r") as root:
        if bool(root.attrs.get("sim")) is not True:
            raise ValueError(f"{path}: expected attrs['sim'] == True")

        qpos = root["/observations/qpos"][:]
        qvel = root["/observations/qvel"][:]
        image = root["/observations/images/front"]
        action = root["/action"][:]

        expected_state_shape = (episode_len, 8)
        expected_image_shape = (episode_len, height, width, 3)
        if qpos.shape != expected_state_shape:
            raise ValueError(f"{path}: qpos shape {qpos.shape}, expected {expected_state_shape}")
        if qvel.shape != expected_state_shape:
            raise ValueError(f"{path}: qvel shape {qvel.shape}, expected {expected_state_shape}")
        if action.shape != expected_state_shape:
            raise ValueError(f"{path}: action shape {action.shape}, expected {expected_state_shape}")
        if image.shape != expected_image_shape:
            raise ValueError(f"{path}: image shape {image.shape}, expected {expected_image_shape}")
        if image.dtype != np.uint8:
            raise ValueError(f"{path}: image dtype {image.dtype}, expected uint8")
        if not np.isfinite(qpos).all() or not np.isfinite(qvel).all() or not np.isfinite(action).all():
            raise ValueError(f"{path}: qpos/qvel/action contain non-finite values")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify ACT pick-and-place demo dataset")
    parser.add_argument("--task", default="pick_place")
    parser.add_argument("--expected", type=int, default=50)
    parser.add_argument("--episode-len", type=int, default=300)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=640)
    args = parser.parse_args()

    data_dir = Path("data") / args.task
    episodes = sorted(data_dir.glob("episode_*.hdf5"))
    print(f"Dataset: {data_dir}")
    print(f"Episodes found: {len(episodes)}")
    if len(episodes) < args.expected:
        raise SystemExit(f"Expected at least {args.expected} episodes, found {len(episodes)}")

    for path in episodes[:args.expected]:
        verify_episode(path, args.episode_len, args.height, args.width)

    print(f"Verified {args.expected} episodes")
    print("DATASET VERIFICATION PASSED")


if __name__ == "__main__":
    main()
