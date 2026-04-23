"""Inspect one ACT demonstration HDF5 file."""

from __future__ import annotations

import argparse
import h5py
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect ACT demo HDF5 schema")
    parser.add_argument("path")
    args = parser.parse_args()

    with h5py.File(args.path, "r") as root:
        qpos = root["/observations/qpos"]
        qvel = root["/observations/qvel"]
        image = root["/observations/images/front"]
        action = root["/action"]

        print(f"sim attr: {root.attrs.get('sim')}")
        print(f"qpos: shape={qpos.shape}, dtype={qpos.dtype}")
        print(f"qvel: shape={qvel.shape}, dtype={qvel.dtype}")
        print(f"front image: shape={image.shape}, dtype={image.dtype}, min={image[:].min()}, max={image[:].max()}")
        print(f"action: shape={action.shape}, dtype={action.dtype}")
        print(f"qpos finite: {np.isfinite(qpos[:]).all()}")
        print(f"qvel finite: {np.isfinite(qvel[:]).all()}")
        print(f"action finite: {np.isfinite(action[:]).all()}")
        print(f"first qpos: {np.array2string(qpos[0], precision=4)}")
        print(f"first action: {np.array2string(action[0], precision=4)}")


if __name__ == "__main__":
    main()
