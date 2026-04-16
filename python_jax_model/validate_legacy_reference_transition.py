from __future__ import annotations

import argparse
import json
from pathlib import Path

from elle_jax_model.legacy_reference import compare_legacy_reference_transition


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare produced ELLE before/after checkpoints against a committed legacy transition JSON"
    )
    parser.add_argument(
        "--reference-json",
        type=Path,
        required=True,
        help="Path to a committed legacy transition JSON file",
    )
    parser.add_argument("--before", type=Path, required=True, help="Path to the candidate before-state ELLE file")
    parser.add_argument("--after", type=Path, required=True, help="Path to the candidate after-state ELLE file")
    parser.add_argument("--label-attribute", default="auto", help="Label attribute or auto-detect/derive mode")
    parser.add_argument("--json-out", type=Path, help="Optional path to write the comparison report JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reference_transition = json.loads(args.reference_json.read_text(encoding="utf-8"))
    report = compare_legacy_reference_transition(
        args.before,
        args.after,
        reference_transition,
        label_attribute=str(args.label_attribute),
    )

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print(f"wrote legacy transition comparison: {args.json_out}")

    print(f"legacy transition match: {report['matches']}")
    print(f"transition: {report['reference_checkpoint_name']}")
    if report["mismatched_field_transitions"]:
        print(f"mismatched fields: {', '.join(report['mismatched_field_transitions'])}")
    if report["missing_field_transitions"]:
        print(f"missing fields: {', '.join(report['missing_field_transitions'])}")
    if report["unexpected_field_transitions"]:
        print(f"unexpected fields: {', '.join(report['unexpected_field_transitions'])}")


if __name__ == "__main__":
    main()
