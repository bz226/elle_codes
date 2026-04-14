from __future__ import annotations

import argparse
from pathlib import Path

from elle_jax_model.calibration import calibrate_fine_foam, write_calibration_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate the ELLE-seeded grain-growth rewrite against the fine_foam reference sequence"
    )
    parser.add_argument("--reference-dir", type=Path, required=True, help="Reference ELLE sequence directory")
    parser.add_argument("--pattern", default="fine_foam_step*.elle", help="Glob pattern for reference ELLE snapshots")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("python_jax_model/validation/fine_foam_calibration"),
        help="Directory for generated candidate ELLE sequences",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("python_jax_model/validation/fine_foam_calibration_report.json"),
        help="Path to write the calibration JSON report",
    )
    parser.add_argument("--dt-grid", type=float, nargs="+", default=[0.01, 0.02, 0.03])
    parser.add_argument("--mobility-grid", type=float, nargs="+", default=[0.5, 0.75, 1.0])
    parser.add_argument("--gradient-penalty-grid", type=float, nargs="+", default=[1.0])
    parser.add_argument("--interaction-strength-grid", type=float, nargs="+", default=[2.0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--init-elle-attribute", default="auto")
    parser.add_argument("--init-smoothing-steps", type=int, default=0)
    parser.add_argument("--init-noise", type=float, default=0.0)
    parser.add_argument("--no-reuse-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = calibrate_fine_foam(
        reference_dir=args.reference_dir,
        output_dir=args.output_dir,
        pattern=args.pattern,
        dt_grid=args.dt_grid,
        mobility_grid=args.mobility_grid,
        gradient_penalty_grid=args.gradient_penalty_grid,
        interaction_strength_grid=args.interaction_strength_grid,
        seed=args.seed,
        init_elle_attribute=args.init_elle_attribute,
        init_smoothing_steps=args.init_smoothing_steps,
        init_noise=args.init_noise,
        reuse_existing=not args.no_reuse_existing,
    )
    outpath = write_calibration_report(args.json_out, report)
    best = report["best_run"]
    print(f"wrote calibration report: {outpath}")
    print(f"evaluated {report['num_runs']} parameter sets from seed {Path(report['reference_seed_path']).name}")
    if best is not None:
        components = best["score"]["components"]
        print(
            "best run: "
            f"{best['run_name']} "
            f"score={best['score']['score']:.6f} "
            f"raster_count_nrmse={components['rasterized_grain_count_nrmse']:.6f} "
            f"raster_area_nrmse={components['rasterized_mean_grain_area_nrmse']:.6f}"
        )
        print(f"candidate dir: {best['candidate_dir']}")


if __name__ == "__main__":
    main()
