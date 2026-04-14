from __future__ import annotations

import argparse
from pathlib import Path

from elle_jax_model.benchmark_validation import (
    build_benchmark_validation_report,
    write_benchmark_validation_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate local ELLE sequences and release datasets against benchmark trends"
    )
    parser.add_argument("--reference-dir", type=Path, required=True, help="Reference ELLE sequence directory")
    parser.add_argument("--data-dir", type=Path, required=True, help="Liu/Suckale release dataset directory")
    parser.add_argument("--candidate-dir", type=Path, help="Optional candidate ELLE sequence directory")
    parser.add_argument("--pattern", default="*.elle", help="Glob pattern for ELLE snapshots")
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("python_jax_model/validation/benchmark_validation_report.json"),
        help="Path to write the JSON report",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_benchmark_validation_report(
        reference_dir=args.reference_dir,
        candidate_dir=args.candidate_dir,
        data_dir=args.data_dir,
        pattern=args.pattern,
    )
    outpath = write_benchmark_validation_report(args.json_out, report)

    static_flags = report["static_grain_growth"]["reference_trends"]["flags"]
    raster_flags = report["rasterized_grain_growth"]["reference_trends"]["flags"]
    release_flags = report["release_dataset_benchmarks"]["flags"]
    print(f"wrote benchmark validation report: {outpath}")
    print(
        "static reference: "
        f"coarsening_present={int(bool(static_flags['coarsening_present']))} "
        f"grain_area_mostly_increasing={int(bool(static_flags['grain_area_mostly_increasing']))} "
        f"grain_count_mostly_decreasing={int(bool(static_flags['grain_count_mostly_decreasing']))}"
    )
    print(
        "rasterized reference: "
        f"coarsening_present={int(bool(raster_flags['coarsening_present']))} "
        f"grain_area_mostly_increasing={int(bool(raster_flags['grain_area_mostly_increasing']))} "
        f"grain_count_mostly_decreasing={int(bool(raster_flags['grain_count_mostly_decreasing']))}"
    )
    print(
        "release dataset expectations: "
        f"grain_hotter_case_more_active={int(bool(release_flags['grain_hotter_case_more_active']))} "
        f"all_euler_hotter_cases_more_active={int(bool(release_flags['all_euler_hotter_cases_more_active']))}"
    )
    if args.candidate_dir is not None:
        comparison = report["static_grain_growth"].get("comparison", {})
        grain_area = comparison.get("mean_grain_area")
        raster = report["rasterized_grain_growth"].get("comparison", {}).get("mean_grain_area")
        if grain_area is not None:
            print(
                "candidate comparison: "
                f"matched_steps={len(grain_area['matched_steps'])} "
                f"mean_grain_area_nrmse={grain_area['normalized_rmse']:.6f}"
            )
        if raster is not None:
            print(
                "candidate raster comparison: "
                f"matched_steps={len(raster['matched_steps'])} "
                f"mean_grain_area_nrmse={raster['normalized_rmse']:.6f}"
            )


if __name__ == "__main__":
    main()
