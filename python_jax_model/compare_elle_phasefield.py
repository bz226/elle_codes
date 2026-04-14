from __future__ import annotations

import argparse
from pathlib import Path

from elle_jax_model.phasefield_compare import (
    compare_elle_phasefield_files,
    inspect_elle_phasefield_binary,
    write_phasefield_comparison_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two ELLE phasefield .elle states numerically")
    parser.add_argument("reference", type=Path, help="reference ELLE phasefield state")
    parser.add_argument("candidate", type=Path, help="candidate ELLE phasefield state")
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--binary", type=Path, help="optional elle_phasefield binary to inspect")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = compare_elle_phasefield_files(args.reference, args.candidate)
    if args.binary is not None:
        report["binary_status"] = inspect_elle_phasefield_binary(args.binary)

    if args.json_out is not None:
        write_phasefield_comparison_report(args.json_out, report)
        print(f"wrote comparison report: {args.json_out}")

    print(f"reference: {report['reference_path']}")
    print(f"candidate: {report['candidate_path']}")
    print(
        "theta: "
        f"rmse={report['theta_rmse']:.6e} "
        f"mae={report['theta_mae']:.6e} "
        f"max_abs={report['theta_max_abs']:.6e} "
        f"solid_iou={report['theta_solid_iou']:.6f}"
    )
    print(
        "temperature: "
        f"rmse={report['temperature_rmse']:.6e} "
        f"mae={report['temperature_mae']:.6e} "
        f"max_abs={report['temperature_max_abs']:.6e}"
    )
    print(
        "summary: "
        f"solid_fraction_delta={report['solid_fraction_delta']:.6e} "
        f"interface_fraction_delta={report['interface_fraction_delta']:.6e} "
        f"mean_temperature_delta={report['mean_temperature_delta']:.6e}"
    )
    if "binary_status" in report:
        binary_status = report["binary_status"]
        missing = ", ".join(binary_status["missing_libraries"]) if binary_status["missing_libraries"] else "none"
        print(
            "binary: "
            f"ready={int(bool(binary_status['ready']))} "
            f"missing_libraries={missing}"
        )


if __name__ == "__main__":
    main()
