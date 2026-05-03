"""Verify ACT demonstration dataset count and HDF5 schema."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


REQUIRED_KEYS = (
    "/observations/qpos",
    "/observations/qvel",
    "/action",
)


def _episode_index(path: Path) -> int:
    stem = path.stem
    if not stem.startswith("episode_"):
        raise ValueError(f"{path}: expected filename episode_N.hdf5")
    return int(stem[len("episode_"):])


def verify_episode(path: Path, camera_names: list[str]) -> dict:
    with h5py.File(path, "r") as root:
        if bool(root.attrs.get("sim")) is not True:
            raise ValueError(f"{path}: expected attrs['sim'] == True")

        for key in REQUIRED_KEYS:
            if key not in root:
                raise ValueError(f"{path}: missing required key {key}")
        for cam_name in camera_names:
            key = f"/observations/images/{cam_name}"
            if key not in root:
                raise ValueError(f"{path}: missing required key {key}")

        qpos = root["/observations/qpos"][:]
        qvel = root["/observations/qvel"][:]
        action = root["/action"][:]
        images = {
            cam_name: root[f"/observations/images/{cam_name}"][:]
            for cam_name in camera_names
        }

        if qpos.ndim != 2 or qvel.ndim != 2 or action.ndim != 2:
            raise ValueError(f"{path}: qpos/qvel/action must be rank-2 arrays")
        if qpos.shape[0] == 0 or qvel.shape[0] == 0 or action.shape[0] == 0:
            raise ValueError(f"{path}: empty qpos/qvel/action dataset")
        if qpos.shape[0] != qvel.shape[0] or qpos.shape[0] != action.shape[0]:
            raise ValueError(f"{path}: qpos/qvel/action lengths differ")
        if qpos.shape != qvel.shape:
            raise ValueError(f"{path}: qpos shape {qpos.shape} differs from qvel shape {qvel.shape}")
        if not np.isfinite(qpos).all() or not np.isfinite(qvel).all() or not np.isfinite(action).all():
            raise ValueError(f"{path}: qpos/qvel/action contain non-finite values")

        image_shapes = {}
        image_dtypes = {}
        for cam_name, image in images.items():
            if image.ndim != 4 or image.shape[0] != qpos.shape[0] or image.shape[-1] != 3:
                raise ValueError(f"{path}: image {cam_name} shape {image.shape} is invalid")
            if image.shape[0] == 0:
                raise ValueError(f"{path}: image {cam_name} is empty")
            if image.dtype != np.uint8:
                raise ValueError(f"{path}: image {cam_name} dtype {image.dtype}, expected uint8")
            image_shapes[cam_name] = image.shape
            image_dtypes[cam_name] = str(image.dtype)

        return {
            "path": path,
            "length": qpos.shape[0],
            "qpos_shape": qpos.shape,
            "qvel_shape": qvel.shape,
            "action_shape": action.shape,
            "image_shapes": image_shapes,
            "image_dtypes": image_dtypes,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify ACT pick-and-place demo dataset")
    parser.add_argument("--task", default="pick_place")
    parser.add_argument("--dataset_dir", default=None,
                        help="Dataset directory. Defaults to data/<task>.")
    parser.add_argument("--expected", type=int, default=50)
    parser.add_argument("--exact", action="store_true",
                        help="Require exactly --expected episode files")
    parser.add_argument("--episode-len", type=int, default=None,
                        help="Expected episode length, if fixed")
    parser.add_argument("--min-len", type=int, default=1)
    parser.add_argument("--max-len", type=int, default=None)
    parser.add_argument("--camera_names", nargs="+", default=["front"])
    args = parser.parse_args()

    data_dir = Path(args.dataset_dir) if args.dataset_dir else Path("data") / args.task
    episodes = sorted(data_dir.glob("episode_*.hdf5"), key=_episode_index)
    print(f"Dataset: {data_dir}")
    print(f"Episodes found: {len(episodes)}")
    if args.exact and len(episodes) != args.expected:
        raise SystemExit(f"Expected exactly {args.expected} episodes, found {len(episodes)}")
    if not args.exact and len(episodes) < args.expected:
        raise SystemExit(f"Expected at least {args.expected} episodes, found {len(episodes)}")

    expected_indices = list(range(args.expected))
    actual_indices = sorted(_episode_index(path) for path in episodes[:args.expected])
    if actual_indices != expected_indices:
        raise SystemExit(
            f"Expected contiguous episode indices 0..{args.expected - 1}, got "
            f"{actual_indices[:5]}...{actual_indices[-5:] if actual_indices else []}"
        )

    summaries = []
    for path in episodes[:args.expected]:
        summaries.append(verify_episode(path, args.camera_names))

    lengths = [summary["length"] for summary in summaries]
    if args.episode_len is not None and any(length != args.episode_len for length in lengths):
        raise SystemExit(f"Expected episode length {args.episode_len}, got range {min(lengths)}..{max(lengths)}")
    if any(length < args.min_len for length in lengths):
        raise SystemExit(f"Found episode shorter than --min-len={args.min_len}")
    if args.max_len is not None and any(length > args.max_len for length in lengths):
        raise SystemExit(f"Found episode longer than --max-len={args.max_len}")

    first = summaries[0]
    for summary in summaries[1:]:
        if summary["qpos_shape"][1:] != first["qpos_shape"][1:]:
            raise SystemExit(f"Inconsistent qpos dims in {summary['path']}")
        if summary["qvel_shape"][1:] != first["qvel_shape"][1:]:
            raise SystemExit(f"Inconsistent qvel dims in {summary['path']}")
        if summary["action_shape"][1:] != first["action_shape"][1:]:
            raise SystemExit(f"Inconsistent action dims in {summary['path']}")
        for cam_name in args.camera_names:
            if summary["image_shapes"][cam_name][1:] != first["image_shapes"][cam_name][1:]:
                raise SystemExit(f"Inconsistent image shape for camera {cam_name} in {summary['path']}")

    print(f"Verified episodes: {len(summaries)}")
    print(f"Episode length range: {min(lengths)}..{max(lengths)}")
    print(f"qpos shape: {first['qpos_shape']}")
    print(f"qvel shape: {first['qvel_shape']}")
    print(f"action shape: {first['action_shape']}")
    for cam_name in args.camera_names:
        print(f"image {cam_name} shape: {first['image_shapes'][cam_name]}")
    print("DATASET VERIFICATION PASSED")


if __name__ == "__main__":
    main()
