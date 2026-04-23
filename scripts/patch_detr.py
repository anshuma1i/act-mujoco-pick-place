"""Patch the Shaka-Labs DETR package for this ACT pick-and-place project.

Run after installing the DETR fork:
    uv run python scripts/patch_detr.py
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


REPLACEMENTS = {
    "models/detr_vae.py": {
        "nn.Linear(5, hidden_dim)": "nn.Linear(state_dim, hidden_dim)",
        "state_dim = 5 # TODO hardcode": "state_dim = args.state_dim",
        "mlp_in_dim = 768 * len(backbones) + 5": "mlp_in_dim = 768 * len(backbones) + state_dim",
        "output_dim=5": "output_dim=state_dim",
    },
    "main.py": {
        "argparse.ArgumentParser('Set transformer detector', add_help=False)": "argparse.ArgumentParser('Set transformer detector', add_help=False, allow_abbrev=False)",
        "argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])": "argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()], allow_abbrev=False)",
        "args = parser.parse_args()": "args, _ = parser.parse_known_args()",
    },
}


def patch_file(path: Path, replacements: dict[str, str]) -> bool:
    text = path.read_text()
    patched = text
    for old, new in replacements.items():
        patched = patched.replace(old, new)

    if patched == text:
        return False

    path.write_text(patched)
    return True


def main() -> None:
    spec = importlib.util.find_spec("detr")
    if spec is None or spec.origin is None:
        raise SystemExit("Could not import detr. Install it first with: uv pip install git+https://github.com/Shaka-Labs/detr.git")

    detr_dir = Path(spec.origin).parent
    changed = []
    for relative_path, replacements in REPLACEMENTS.items():
        path = detr_dir / relative_path
        if not path.exists():
            raise SystemExit(f"Expected DETR file not found: {path}")
        if patch_file(path, replacements):
            changed.append(str(path))

    if changed:
        print("Patched DETR files:")
        for path in changed:
            print(f"  {path}")
    else:
        print("DETR already patched.")


if __name__ == "__main__":
    main()
