from __future__ import annotations

import argparse
import json
from pathlib import Path

from elle_jax_model.legacy_reference import extract_legacy_reference_transition


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a compact legacy-reference transition from old ELLE before/after checkpoint files"
    )
    parser.add_argument("--before", type=Path, required=True, help="Path to the legacy before-state ELLE file")
    parser.add_argument("--after", type=Path, required=True, help="Path to the legacy after-state ELLE file")
    parser.add_argument(
        "--checkpoint-name",
        help="Optional short name for the transition, e.g. recovery_stage0",
    )
    parser.add_argument("--label-attribute", default="auto", help="Label attribute or auto-detect/derive mode")
    parser.add_argument(
        "--field",
        action="append",
        dest="field_names",
        help="Optional field name to include. Can be provided multiple times.",
    )
    parser.add_argument("--json-out", type=Path, required=True, help="Path to write the transition JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    transition = extract_legacy_reference_transition(
        args.before,
        args.after,
        checkpoint_name=args.checkpoint_name,
        label_attribute=str(args.label_attribute),
        field_names=args.field_names,
    )
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(transition, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote legacy reference transition: {args.json_out}")
    print(f"transition: {transition['checkpoint_name']}")
    print(f"fields: {', '.join(transition['field_names'])}")


if __name__ == "__main__":
    main()
