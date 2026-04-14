from __future__ import annotations

import argparse
from pathlib import Path

from elle_jax_model.microstructure_validation import (
    collect_elle_microstructure_snapshots,
    compare_elle_microstructure_sequences,
    summarize_liu_suckale_datasets,
    write_microstructure_validation_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize ELLE microstructure sequences and Liu/Suckale training datasets"
    )
    parser.add_argument("--reference-dir", type=Path, help="Directory containing reference .elle snapshots")
    parser.add_argument("--candidate-dir", type=Path, help="Directory containing candidate .elle snapshots")
    parser.add_argument("--data-dir", type=Path, help="Directory containing Liu/Suckale .npz datasets")
    parser.add_argument("--pattern", default="*.elle", help="Glob pattern for ELLE snapshots")
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("python_jax_model/validation/microstructure_report.json"),
        help="Path to write the JSON report",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report: dict[str, object] = {}

    if args.reference_dir is not None and args.candidate_dir is not None:
        report["sequence_comparison"] = compare_elle_microstructure_sequences(
            args.reference_dir,
            args.candidate_dir,
            pattern=args.pattern,
        )
    elif args.reference_dir is not None:
        report["reference_sequence"] = collect_elle_microstructure_snapshots(
            args.reference_dir,
            pattern=args.pattern,
        )

    if args.data_dir is not None:
        report["liu_suckale_datasets"] = summarize_liu_suckale_datasets(args.data_dir)

    write_microstructure_validation_report(args.json_out, report)
    print(f"wrote microstructure validation report: {args.json_out}")
    if "sequence_comparison" in report:
        summary = report["sequence_comparison"]["summary"]
        print(
            "sequence comparison: "
            f"matched_steps={summary['num_matched_steps']} "
            f"grain_count_abs_diff_mean={summary.get('grain_count_abs_diff_mean', float('nan')):.6f} "
            f"mean_grain_area_abs_diff_mean={summary.get('mean_grain_area_abs_diff_mean', float('nan')):.6e}"
        )
    elif "reference_sequence" in report:
        print(f"reference snapshots: {len(report['reference_sequence'])}")
    if "liu_suckale_datasets" in report:
        print(f"liu_suckale datasets: {report['liu_suckale_datasets']['num_datasets']}")


if __name__ == "__main__":
    main()
