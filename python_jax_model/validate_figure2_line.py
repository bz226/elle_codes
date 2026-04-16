from __future__ import annotations

import argparse
from pathlib import Path

from elle_jax_model.figure2_validation import (
    build_figure2_line_validation_report,
    write_figure2_line_validation_html,
    write_figure2_line_validation_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a Figure-2-style grain-area validation line for two ELLE snapshot sequences"
    )
    parser.add_argument("--reference-dir", type=Path, required=True, help="Reference ELLE sequence directory")
    parser.add_argument("--candidate-dir", type=Path, required=True, help="Candidate ELLE sequence directory")
    parser.add_argument("--pattern", default="*.elle", help="Glob pattern for ELLE snapshots")
    parser.add_argument("--attribute", default="auto", help="Label attribute to rasterize from ELLE snapshots")
    parser.add_argument("--kde-points", type=int, default=128, help="Number of grain-area points in each KDE grid")
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("python_jax_model/validation/figure2_line_validation.json"),
        help="Path to write the JSON report",
    )
    parser.add_argument(
        "--html-out",
        type=Path,
        default=Path("python_jax_model/validation/figure2_line_validation.html"),
        help="Path to write the standalone HTML line chart",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_figure2_line_validation_report(
        reference_dir=args.reference_dir,
        candidate_dir=args.candidate_dir,
        pattern=args.pattern,
        attribute=args.attribute,
        kde_points=int(args.kde_points),
    )
    json_path = write_figure2_line_validation_report(args.json_out, report)
    html_path = write_figure2_line_validation_html(args.html_out, report)

    line = report["figure2_like_validation"]["mean_grain_area_line"]
    print(f"wrote Figure-2-style JSON report: {json_path}")
    print(f"wrote Figure-2-style HTML report: {html_path}")
    print(
        "mean-grain-area line: "
        f"matched_steps={len(line['steps'])} "
        f"rmse={line['rmse']:.6f} "
        f"normalized_rmse={line['normalized_rmse']:.6f}"
    )


if __name__ == "__main__":
    main()
