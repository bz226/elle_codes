from __future__ import annotations

import argparse
import json
from pathlib import Path

from elle_jax_model.legacy_reference import (
    compare_legacy_reference_bundle,
    load_legacy_reference_bundle,
)


def _parse_checkpoint(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("checkpoint must be NAME=PATH")
    name, raw_path = value.split("=", 1)
    if not name.strip():
        raise argparse.ArgumentTypeError("checkpoint name cannot be empty")
    return name.strip(), Path(raw_path).expanduser()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare produced ELLE checkpoints against a committed legacy-reference bundle"
    )
    parser.add_argument(
        "--reference-json",
        type=Path,
        required=True,
        help="Path to a committed legacy-reference bundle JSON file",
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        type=_parse_checkpoint,
        required=True,
        help="Checkpoint in the form NAME=PATH, e.g. step0=/path/to/file.elle",
    )
    parser.add_argument("--label-attribute", default="auto", help="Label attribute or auto-detect/derive mode")
    parser.add_argument("--json-out", type=Path, help="Optional path to write the comparison report JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = load_legacy_reference_bundle(args.reference_json)
    checkpoints = {name: path for name, path in args.checkpoint}
    report = compare_legacy_reference_bundle(
        bundle,
        checkpoints,
        label_attribute=str(args.label_attribute),
    )

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print(f"wrote legacy reference comparison: {args.json_out}")

    print(f"legacy reference match: {report['matches']}")
    print(f"checkpoints compared: {', '.join(report['checkpoint_order'])}")
    if report["missing_checkpoints"]:
        print(f"missing checkpoints: {', '.join(report['missing_checkpoints'])}")
    if report["unexpected_checkpoints"]:
        print(f"unexpected checkpoints: {', '.join(report['unexpected_checkpoints'])}")
    for name in report["checkpoint_order"]:
        checkpoint_report = report["checkpoints"].get(name)
        if checkpoint_report is None:
            continue
        print(f"{name}: match={checkpoint_report['matches']}")
        if checkpoint_report["mismatched_field_summaries"]:
            print(
                f"  mismatched fields: {', '.join(checkpoint_report['mismatched_field_summaries'])}"
            )


if __name__ == "__main__":
    main()
