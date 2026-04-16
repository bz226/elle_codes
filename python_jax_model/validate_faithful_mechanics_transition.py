from __future__ import annotations

import argparse
import json
from pathlib import Path

from elle_jax_model.mechanics_replay import validate_faithful_mechanics_transition


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a mechanics-only faithful replay and compare the resulting transition against a legacy before/after ELLE pair"
    )
    parser.add_argument("--init-elle", type=Path, required=True, help="Initial ELLE file used to seed the faithful replay")
    parser.add_argument(
        "--mechanics-snapshot-dir",
        type=Path,
        required=True,
        help="Directory containing one legacy FFT bridge snapshot or a sequence of snapshot subdirectories",
    )
    parser.add_argument("--reference-before", type=Path, required=True, help="Legacy before-state ELLE file")
    parser.add_argument("--reference-after", type=Path, required=True, help="Legacy after-state ELLE file")
    parser.add_argument("--init-elle-attribute", default="auto")
    parser.add_argument("--label-attribute", default="auto")
    parser.add_argument(
        "--checkpoint-name",
        help="Optional short name for the transition, e.g. mechanics_stage0",
    )
    parser.add_argument(
        "--field",
        action="append",
        dest="field_names",
        help="Optional field name to include. Can be provided multiple times.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("python_jax_model/validation/faithful_mechanics_transition"),
    )
    parser.add_argument("--json-out", type=Path, help="Optional path to write the comparison report JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = validate_faithful_mechanics_transition(
        args.init_elle,
        args.mechanics_snapshot_dir,
        args.reference_before,
        args.reference_after,
        outdir=args.outdir,
        init_elle_attribute=str(args.init_elle_attribute),
        label_attribute=str(args.label_attribute),
        checkpoint_name=args.checkpoint_name,
        field_names=args.field_names,
    )
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print(f"wrote mechanics transition comparison: {args.json_out}")

    print(f"faithful mechanics transition match: {report['matches']}")
    print(f"candidate before: {report['candidate_before_path']}")
    print(f"candidate after: {report['candidate_after_path']}")
    if report["mismatched_field_transitions"]:
        print(f"mismatched fields: {', '.join(report['mismatched_field_transitions'])}")
    if report["missing_field_transitions"]:
        print(f"missing fields: {', '.join(report['missing_field_transitions'])}")
    if report["unexpected_field_transitions"]:
        print(f"unexpected fields: {', '.join(report['unexpected_field_transitions'])}")


if __name__ == "__main__":
    main()
