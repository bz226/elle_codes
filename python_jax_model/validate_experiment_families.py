from __future__ import annotations

import argparse
from pathlib import Path

from elle_jax_model.benchmark_validation import (
    evaluate_experiment_family_suite,
    write_benchmark_validation_report,
)
from validate_benchmarks import (
    _resolve_experiment_family_dirs,
    _resolve_experiment_family_report_paths,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a local experiment-family suite for the paper-style 0/1/10/25 benchmark runs"
    )
    parser.add_argument(
        "--experiment-family",
        action="append",
        default=[],
        metavar="FAMILY=DIR",
        help="Optional experiment-family run directory, repeatable, e.g. 0=path/to/run",
    )
    parser.add_argument(
        "--experiment-family-manifest",
        type=Path,
        help="Optional JSON manifest mapping family IDs to run directories",
    )
    parser.add_argument(
        "--experiment-family-root",
        type=Path,
        help="Optional root directory to auto-discover family run subdirectories",
    )
    parser.add_argument(
        "--experiment-family-report",
        action="append",
        default=[],
        metavar="FAMILY=JSON",
        help="Optional experiment-family benchmark report JSON, repeatable, e.g. 0=path/to/report.json",
    )
    parser.add_argument(
        "--experiment-family-report-manifest",
        type=Path,
        help="Optional JSON manifest mapping family IDs to benchmark report JSON files",
    )
    parser.add_argument(
        "--experiment-family-report-root",
        type=Path,
        help="Optional root directory to auto-discover experiment-family benchmark report JSON files",
    )
    parser.add_argument("--pattern", default="*.elle", help="Glob pattern for ELLE snapshots")
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("python_jax_model/validation/experiment_family_benchmarks.json"),
        help="Path to write the JSON report",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment_family_dirs = _resolve_experiment_family_dirs(
        explicit_values=list(args.experiment_family),
        manifest_path=args.experiment_family_manifest,
        root=args.experiment_family_root,
    )
    experiment_family_reports = _resolve_experiment_family_report_paths(
        explicit_values=list(args.experiment_family_report),
        manifest_path=args.experiment_family_report_manifest,
        root=args.experiment_family_report_root,
    )
    if not experiment_family_dirs and not experiment_family_reports:
        raise SystemExit(
            "no experiment-family inputs were provided; use directory-based or report-based "
            "family arguments"
        )

    report = evaluate_experiment_family_suite(
        experiment_family_dirs=experiment_family_dirs or None,
        experiment_family_reports=experiment_family_reports or None,
        pattern=args.pattern,
    )
    outpath = write_benchmark_validation_report(args.json_out, report)
    assessment = report.get("paper_signature_assessment", {})
    print(f"wrote experiment family benchmark report: {outpath}")
    print(
        "experiment family acceptance: "
        f"families={','.join(report.get('family_order', []))} "
        f"applicable={int(assessment.get('applicable_checks', 0))} "
        f"passed={int(assessment.get('passed_checks', 0))}"
    )


if __name__ == "__main__":
    main()
