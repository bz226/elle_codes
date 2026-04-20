from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

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
    parser.add_argument(
        "--candidate-mesh-json",
        type=Path,
        help="Optional saved candidate mesh JSON for legacy old.stats bookkeeping comparison",
    )
    parser.add_argument(
        "--legacy-old-stats",
        type=Path,
        help="Optional legacy old.stats file for bookkeeping comparison against candidate mesh JSON",
    )
    parser.add_argument(
        "--candidate-legacy-statistics",
        type=Path,
        help="Optional legacy tmpstats.dat / last.stats / old.stats summary file for final candidate snapshot comparison",
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
        default=Path("python_jax_model/validation/benchmark_validation_report.json"),
        help="Path to write the JSON report",
    )
    return parser.parse_args()


def _parse_experiment_family_args(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        family_id, separator, directory_text = str(value).partition("=")
        if not separator or not family_id or not directory_text:
            raise ValueError(f"invalid --experiment-family value {value!r}, expected FAMILY=DIR")
        result[str(family_id)] = Path(directory_text)
    return result


def _parse_experiment_family_report_args(values: list[str]) -> dict[str, Path]:
    return _parse_experiment_family_args(values)


def _load_experiment_family_manifest(path: Path) -> dict[str, Path]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"invalid experiment-family manifest {path}, expected JSON object")
    result: dict[str, Path] = {}
    for family_id, directory_text in payload.items():
        if not isinstance(family_id, str) or not isinstance(directory_text, str):
            raise ValueError(f"invalid experiment-family manifest entry {family_id!r}: {directory_text!r}")
        result[str(family_id)] = Path(directory_text)
    return result


def _load_experiment_family_report_manifest(path: Path) -> dict[str, Path]:
    return _load_experiment_family_manifest(path)


def _discover_experiment_family_dirs(root: Path) -> dict[str, Path]:
    family_ids = ("0", "1", "10", "25")
    result: dict[str, Path] = {}
    for family_id in family_ids:
        direct = root / family_id
        if direct.is_dir():
            result[family_id] = direct

    pattern = re.compile(r"(?:^|[_-])(0|1|10|25)(?:$|[_-])")
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        match = pattern.search(child.name)
        if match is None:
            continue
        family_id = str(match.group(1))
        result.setdefault(family_id, child)
    return result


def _discover_experiment_family_report_paths(root: Path) -> dict[str, Path]:
    family_ids = ("0", "1", "10", "25")
    result: dict[str, Path] = {}
    for family_id in family_ids:
        direct = root / f"{family_id}.json"
        if direct.is_file():
            result[family_id] = direct

    pattern = re.compile(r"(?:^|[_-])(0|1|10|25)(?:$|[_-])")
    for child in sorted(root.iterdir()):
        if not child.is_file() or child.suffix.lower() != ".json":
            continue
        match = pattern.search(child.stem)
        if match is None:
            continue
        family_id = str(match.group(1))
        result.setdefault(family_id, child)
    return result


def _resolve_experiment_family_dirs(
    *,
    explicit_values: list[str],
    manifest_path: Path | None,
    root: Path | None,
) -> dict[str, Path]:
    result: dict[str, Path] = {}
    if root is not None:
        result.update(_discover_experiment_family_dirs(root))
    if manifest_path is not None:
        result.update(_load_experiment_family_manifest(manifest_path))
    result.update(_parse_experiment_family_args(explicit_values))
    return result


def _resolve_experiment_family_report_paths(
    *,
    explicit_values: list[str],
    manifest_path: Path | None,
    root: Path | None,
) -> dict[str, Path]:
    result: dict[str, Path] = {}
    if root is not None:
        result.update(_discover_experiment_family_report_paths(root))
    if manifest_path is not None:
        result.update(_load_experiment_family_report_manifest(manifest_path))
    result.update(_parse_experiment_family_report_args(explicit_values))
    return result


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
    report = build_benchmark_validation_report(
        reference_dir=args.reference_dir,
        candidate_dir=args.candidate_dir,
        candidate_mesh_json_path=args.candidate_mesh_json,
        candidate_legacy_statistics_path=args.candidate_legacy_statistics,
        data_dir=args.data_dir,
        legacy_old_stats_path=args.legacy_old_stats,
        experiment_family_dirs=experiment_family_dirs or None,
        experiment_family_reports=experiment_family_reports or None,
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
    family_report = report.get("experiment_family_benchmarks")
    if isinstance(family_report, dict):
        family_assessment = family_report.get("paper_signature_assessment", {})
        print(
            "experiment family acceptance: "
            f"families={','.join(family_report.get('family_order', []))} "
            f"applicable={int(family_assessment.get('applicable_checks', 0))} "
            f"passed={int(family_assessment.get('passed_checks', 0))}"
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
    bookkeeping = report.get("legacy_old_stats_bookkeeping")
    if isinstance(bookkeeping, dict):
        print(
            "legacy old.stats bookkeeping: "
            f"legacy_total_flynns={int(bookkeeping['legacy_total_flynn_count'])} "
            f"candidate_total_flynns={int(bookkeeping['current_total_flynn_count'])} "
            f"legacy_mapped={int(bookkeeping['legacy_mapped_flynn_count'])} "
            f"candidate_mapped={int(bookkeeping['current_source_mapped_flynn_count'])}"
        )
    legacy_final = report.get("legacy_final_statistics")
    if isinstance(legacy_final, dict):
        comparison = legacy_final.get("comparison", {})
        print(
            "legacy final statistics: "
            f"kind={comparison.get('legacy_statistics_kind')} "
            f"grain_count_delta={int(comparison.get('grain_count_delta', 0))} "
            f"mean_grain_area_delta={float(comparison.get('mean_grain_area_delta', float('nan'))):.6f}"
        )


if __name__ == "__main__":
    main()
