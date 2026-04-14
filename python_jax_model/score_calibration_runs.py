from __future__ import annotations

import argparse
from pathlib import Path

from elle_jax_model.calibration import score_existing_calibration_runs, write_calibration_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score existing fine_foam calibration run directories against the reference sequence"
    )
    parser.add_argument("--reference-dir", type=Path, required=True, help="Reference ELLE sequence directory")
    parser.add_argument("--candidate-root", type=Path, required=True, help="Directory containing candidate run folders")
    parser.add_argument("--pattern", default="fine_foam_step*.elle", help="Glob pattern for ELLE snapshots")
    parser.add_argument("--init-elle-attribute", default="auto")
    parser.add_argument("--complete-only", action="store_true")
    parser.add_argument("--coverage-penalty-weight", type=float, default=0.25)
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("python_jax_model/validation/scored_calibration_runs.json"),
        help="Path to write the scoring JSON report",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = score_existing_calibration_runs(
        reference_dir=args.reference_dir,
        candidate_root=args.candidate_root,
        pattern=args.pattern,
        init_elle_attribute=args.init_elle_attribute,
        require_complete=args.complete_only,
        coverage_penalty_weight=args.coverage_penalty_weight,
    )
    outpath = write_calibration_report(args.json_out, report)
    print(f"wrote calibration score report: {outpath}")
    print(f"scored {report['num_runs']} candidate directories")
    best = report["best_run"]
    if best is not None:
        components = best["score"]["components"]
        print(
            "best run: "
            f"{best['run_name']} "
            f"coverage={best['coverage']['matched_steps']}/{best['coverage']['expected_steps']} "
            f"score={best['score']['coverage_adjusted_score']:.6f} "
            f"raster_count_nrmse={components['rasterized_grain_count_nrmse']:.6f} "
            f"raster_area_nrmse={components['rasterized_mean_grain_area_nrmse']:.6f}"
        )


if __name__ == "__main__":
    main()
