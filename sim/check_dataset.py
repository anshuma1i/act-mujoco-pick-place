"""Sanity-check mixed ACT pick-and-place demonstration datasets."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import h5py
import numpy as np


REQUIRED_KEYS = (
    "/observations/qpos",
    "/observations/qvel",
    "/action",
)


def episode_index(path: Path) -> int:
    if not path.stem.startswith("episode_"):
        raise ValueError(f"{path}: expected episode_N.hdf5")
    return int(path.stem[len("episode_"):])


def read_episode_metadata(root: h5py.File) -> dict:
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

        metadata = read_episode_metadata(root)
        if metadata.get("success") is not True:
            raise ValueError(f"{path}: expected successful training episode metadata")
        if metadata.get("demo_type") not in {"clean", "recovery", "hard_case"}:
            raise ValueError(f"{path}: invalid or missing demo_type metadata")
        if not metadata.get("perturbation_type"):
            raise ValueError(f"{path}: missing perturbation_type metadata")

        qpos = root["/observations/qpos"][:]
        qvel = root["/observations/qvel"][:]
        action = root["/action"][:]

        if qpos.ndim != 2 or qvel.ndim != 2 or action.ndim != 2:
            raise ValueError(f"{path}: qpos/qvel/action must be rank-2")
        if qpos.shape[0] == 0 or qvel.shape[0] == 0 or action.shape[0] == 0:
            raise ValueError(f"{path}: empty qpos/qvel/action")
        if qpos.shape != qvel.shape:
            raise ValueError(f"{path}: qpos shape {qpos.shape} differs from qvel shape {qvel.shape}")
        if qpos.shape[0] != action.shape[0]:
            raise ValueError(f"{path}: action length differs from qpos length")
        if not np.isfinite(qpos).all() or not np.isfinite(qvel).all() or not np.isfinite(action).all():
            raise ValueError(f"{path}: non-finite qpos/qvel/action values")

        image_shapes = {}
        for cam_name in camera_names:
            image = root[f"/observations/images/{cam_name}"][:]
            if image.ndim != 4 or image.shape[0] != qpos.shape[0] or image.shape[-1] != 3:
                raise ValueError(f"{path}: invalid image shape for {cam_name}: {image.shape}")
            if image.dtype != np.uint8:
                raise ValueError(f"{path}: image {cam_name} dtype {image.dtype}, expected uint8")
            image_shapes[cam_name] = image.shape

    metadata["path"] = str(path)
    metadata["length"] = qpos.shape[0]
    metadata["qpos_shape"] = qpos.shape
    metadata["qvel_shape"] = qvel.shape
    metadata["action_shape"] = action.shape
    metadata["image_shapes"] = image_shapes
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Check mixed ACT demo dataset")
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--expected_total", type=int, default=300)
    parser.add_argument("--expected_clean", type=int, default=240)
    parser.add_argument("--expected_recovery", type=int, default=40)
    parser.add_argument("--expected_hard", type=int, default=20)
    parser.add_argument("--episode-len", type=int, default=300)
    parser.add_argument("--min-len", type=int, default=1)
    parser.add_argument("--camera_names", nargs="+", default=["front"])
    args = parser.parse_args()

    data_dir = Path(args.dataset_dir)
    episodes = sorted(data_dir.glob("episode_*.hdf5"), key=episode_index)
    print(f"Dataset: {data_dir}")
    print(f"Episodes found: {len(episodes)}")
    if len(episodes) != args.expected_total:
        raise SystemExit(f"Expected exactly {args.expected_total} episodes, found {len(episodes)}")

    expected_indices = list(range(args.expected_total))
    actual_indices = [episode_index(path) for path in episodes]
    if actual_indices != expected_indices:
        raise SystemExit("Episode files must be contiguous episode_0.hdf5 .. episode_N.hdf5")

    summaries = [verify_episode(path, args.camera_names) for path in episodes]
    counts = Counter(summary["demo_type"] for summary in summaries)
    expected_counts = {
        "clean": args.expected_clean,
        "recovery": args.expected_recovery,
        "hard_case": args.expected_hard,
    }
    for demo_type, expected in expected_counts.items():
        if counts[demo_type] != expected:
            raise SystemExit(f"Expected {expected} {demo_type} demos, found {counts[demo_type]}")

    lengths = np.array([summary["length"] for summary in summaries])
    if np.any(lengths < args.min_len):
        raise SystemExit(f"Found episode shorter than --min-len={args.min_len}")
    if args.episode_len is not None and np.any(lengths != args.episode_len):
        raise SystemExit(
            f"Expected episode length {args.episode_len}, got range {lengths.min()}..{lengths.max()}"
        )

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
                raise SystemExit(f"Inconsistent image dims for {cam_name} in {summary['path']}")

    print("Demo counts:")
    print(f"  clean: {counts['clean']}")
    print(f"  recovery: {counts['recovery']}")
    print(f"  hard_case: {counts['hard_case']}")
    print(f"Episode length min/max/mean: {lengths.min()} / {lengths.max()} / {lengths.mean():.1f}")
    print(f"qpos shape: {first['qpos_shape']}")
    print(f"qvel shape: {first['qvel_shape']}")
    print(f"action shape: {first['action_shape']}")
    for cam_name in args.camera_names:
        print(f"image {cam_name} shape: {first['image_shapes'][cam_name]}")
    print("MIXED DATASET CHECK PASSED")


if __name__ == "__main__":
    main()
