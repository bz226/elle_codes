from __future__ import annotations

import argparse
from pathlib import Path

from elle_jax_model.legacy_reference import build_legacy_reference_bundle, write_legacy_reference_bundle


def _parse_checkpoint(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("checkpoint must be NAME=PATH")
    name, raw_path = value.split("=", 1)
    if not name.strip():
        raise argparse.ArgumentTypeError("checkpoint name cannot be empty")
    return name.strip(), Path(raw_path).expanduser()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a compact legacy-reference bundle from old ELLE checkpoint files"
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        type=_parse_checkpoint,
        required=True,
        help="Checkpoint in the form NAME=PATH, e.g. gbm_stage=/path/to/file.elle",
    )
    parser.add_argument("--source-name", required=True, help="Short name for the legacy workflow source")
    parser.add_argument("--label-attribute", default="auto", help="Label attribute or auto-detect/derive mode")
    parser.add_argument(
        "--json-out",
        type=Path,
        required=True,
        help="Path to write the reference bundle JSON",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoints = {name: path for name, path in args.checkpoint}
    bundle = build_legacy_reference_bundle(
        checkpoints,
        source_name=str(args.source_name),
        label_attribute=str(args.label_attribute),
    )
    outpath = write_legacy_reference_bundle(args.json_out, bundle)
    print(f"wrote legacy reference bundle: {outpath}")
    print(f"checkpoints: {', '.join(bundle['checkpoint_order'])}")


if __name__ == "__main__":
    main()
