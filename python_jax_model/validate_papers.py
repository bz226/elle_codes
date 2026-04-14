from __future__ import annotations

import argparse
from pathlib import Path

from elle_jax_model.paper_validation import (
    build_paper_validation_report,
    write_paper_validation_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract validation targets from the Llorens and Liu/Suckale papers"
    )
    parser.add_argument("--llorens-pdf", type=Path, required=True, help="Path to the Llorens PDF")
    parser.add_argument("--liu-pdf", type=Path, required=True, help="Path to the Liu/Suckale PDF")
    parser.add_argument("--reference-dir", type=Path, help="Optional ELLE snapshot directory to summarize")
    parser.add_argument("--data-dir", type=Path, help="Optional Liu/Suckale release dataset directory")
    parser.add_argument("--pattern", default="*.elle", help="Glob pattern for ELLE snapshots")
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("python_jax_model/validation/paper_validation_report.json"),
        help="Path to write the JSON report",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_paper_validation_report(
        llorens_pdf=args.llorens_pdf,
        liu_pdf=args.liu_pdf,
        reference_dir=args.reference_dir,
        data_dir=args.data_dir,
        pattern=args.pattern,
    )
    outpath = write_paper_validation_report(args.json_out, report)

    dual_layer = report["llorens_structure"]["dual_layer_model"]["present"]
    static_status = report["liu_suckale_paper"]["microscale_benchmarks"][0]["status"]
    dynamic_status = report["liu_suckale_paper"]["microscale_benchmarks"][1]["status"]
    site_count = len(report["liu_suckale_paper"]["macro_benchmarks"][1].get("sites", []))

    print(f"wrote paper validation report: {outpath}")
    print(
        "paper targets: "
        f"llorens_dual_layer={int(bool(dual_layer))} "
        f"liu_static={static_status} "
        f"liu_dynamic={dynamic_status} "
        f"liu_ice_core_sites={site_count}"
    )
    if "reference_sequence" in report:
        print(f"reference snapshots: {len(report['reference_sequence'])}")
    if "liu_suckale_datasets" in report:
        print(f"liu_suckale datasets: {report['liu_suckale_datasets']['num_datasets']}")


if __name__ == "__main__":
    main()
