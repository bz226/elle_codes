from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from .microstructure_validation import (
    collect_elle_microstructure_snapshots,
    summarize_liu_suckale_datasets,
)
from .simulation import load_elle_label_seed


def _series_summary(steps: np.ndarray, values: np.ndarray) -> dict[str, Any]:
    if values.size == 0:
        return {
            "start": float("nan"),
            "end": float("nan"),
            "delta": float("nan"),
            "relative_delta": float("nan"),
            "slope_per_step": float("nan"),
            "increase_fraction": float("nan"),
            "decrease_fraction": float("nan"),
        }

    delta = float(values[-1] - values[0])
    relative_delta = float(delta / values[0]) if abs(float(values[0])) > 1.0e-12 else float("nan")
    slope = 0.0
    if values.size >= 2 and steps.size >= 2:
        slope = float(np.polyfit(steps.astype(np.float64), values.astype(np.float64), deg=1)[0])

    diffs = np.diff(values.astype(np.float64))
    increase_fraction = float(np.mean(diffs > 0.0)) if diffs.size else 0.0
    decrease_fraction = float(np.mean(diffs < 0.0)) if diffs.size else 0.0
    return {
        "start": float(values[0]),
        "end": float(values[-1]),
        "delta": delta,
        "relative_delta": relative_delta,
        "slope_per_step": slope,
        "increase_fraction": increase_fraction,
        "decrease_fraction": decrease_fraction,
    }


def summarize_sequence_trends(sequence: list[dict[str, Any]]) -> dict[str, Any]:
    if not sequence:
        return {"num_snapshots": 0, "steps": [], "metrics": {}, "flags": {}}

    ordered = sorted(
        sequence,
        key=lambda snapshot: (
            snapshot.get("step") is None,
            snapshot.get("step") if snapshot.get("step") is not None else snapshot.get("path", ""),
        ),
    )
    raw_steps = [
        float(snapshot["step"]) if snapshot.get("step") is not None else float(index)
        for index, snapshot in enumerate(ordered)
    ]
    steps = np.asarray(raw_steps, dtype=np.float64)

    metric_names = (
        "grain_count",
        "mean_grain_area",
        "mean_equivalent_radius",
        "mean_shape_factor",
    )
    metrics: dict[str, Any] = {}
    for name in metric_names:
        values = np.asarray([float(snapshot.get(name, np.nan)) for snapshot in ordered], dtype=np.float64)
        metrics[name] = _series_summary(steps, values)

    orientation_available = all("orientation" in snapshot for snapshot in ordered)
    if orientation_available:
        orientation_mean = np.asarray(
            [float(snapshot["orientation"]["axial_mean_deg"]) for snapshot in ordered],
            dtype=np.float64,
        )
        orientation_resultant = np.asarray(
            [float(snapshot["orientation"]["resultant_length"]) for snapshot in ordered],
            dtype=np.float64,
        )
        metrics["orientation_axial_mean_deg"] = _series_summary(steps, orientation_mean)
        metrics["orientation_resultant_length"] = _series_summary(steps, orientation_resultant)

    flags = {
        "coarsening_present": bool(
            metrics["mean_grain_area"]["delta"] > 0.0 and metrics["grain_count"]["delta"] <= 0.0
        ),
        "grain_area_mostly_increasing": bool(metrics["mean_grain_area"]["increase_fraction"] >= 0.5),
        "grain_count_mostly_decreasing": bool(metrics["grain_count"]["decrease_fraction"] >= 0.5),
    }

    return {
        "num_snapshots": int(len(ordered)),
        "steps": [int(step) if float(step).is_integer() else float(step) for step in steps.tolist()],
        "metrics": metrics,
        "flags": flags,
    }


def _matched_metric_trajectory(
    reference: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    metric: str,
) -> dict[str, Any] | None:
    ref_by_step = {snapshot["step"]: snapshot for snapshot in reference if snapshot.get("step") is not None}
    cand_by_step = {snapshot["step"]: snapshot for snapshot in candidate if snapshot.get("step") is not None}
    matched_steps = sorted(set(ref_by_step) & set(cand_by_step))
    if not matched_steps:
        return None

    ref_values = np.asarray([float(ref_by_step[step][metric]) for step in matched_steps], dtype=np.float64)
    cand_values = np.asarray([float(cand_by_step[step][metric]) for step in matched_steps], dtype=np.float64)
    diffs = cand_values - ref_values
    denom = max(float(np.max(np.abs(ref_values))), 1.0e-12)
    return {
        "matched_steps": [int(step) for step in matched_steps],
        "rmse": float(np.sqrt(np.mean(diffs * diffs))),
        "mae": float(np.mean(np.abs(diffs))),
        "normalized_rmse": float(np.sqrt(np.mean(diffs * diffs)) / denom),
    }


def _periodic_component_pixel_counts(labels: np.ndarray) -> list[int]:
    labels_np = np.asarray(labels, dtype=np.int32)
    nx, ny = labels_np.shape
    visited = np.zeros((nx, ny), dtype=bool)
    counts: list[int] = []

    for ix in range(nx):
        for iy in range(ny):
            if visited[ix, iy]:
                continue
            label = int(labels_np[ix, iy])
            stack = [(ix, iy)]
            visited[ix, iy] = True
            pixel_count = 0

            while stack:
                cx, cy = stack.pop()
                pixel_count += 1
                for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nx_i = (cx + dx) % nx
                    ny_i = (cy + dy) % ny
                    if visited[nx_i, ny_i] or int(labels_np[nx_i, ny_i]) != label:
                        continue
                    visited[nx_i, ny_i] = True
                    stack.append((nx_i, ny_i))

            counts.append(pixel_count)

    return counts


def summarize_elle_label_components(elle_path: str | Path, *, attribute: str = "auto") -> dict[str, Any]:
    seed = load_elle_label_seed(elle_path, attribute=attribute)
    labels = np.asarray(seed["label_field"], dtype=np.int32)
    counts = np.asarray(_periodic_component_pixel_counts(labels), dtype=np.float64)
    nx, ny = labels.shape
    total_pixels = float(nx * ny)
    areas = counts / total_pixels if counts.size else np.zeros(0, dtype=np.float64)
    radii = np.sqrt(areas / np.pi) if areas.size else np.zeros(0, dtype=np.float64)
    match = re.search(r"(\d+)(?=\.elle$)", str(Path(elle_path).name))
    step = int(match.group(1)) if match is not None else None

    return {
        "path": str(Path(elle_path)),
        "step": step,
        "attribute": str(seed["attribute"]),
        "grid_shape": [int(nx), int(ny)],
        "grain_count": int(len(counts)),
        "mean_grain_area": float(areas.mean()) if areas.size else 0.0,
        "mean_equivalent_radius": float(radii.mean()) if radii.size else 0.0,
        "mean_shape_factor": float("nan"),
        "source_label_count": int(seed["num_labels"]),
    }


def collect_elle_label_component_snapshots(
    directory: str | Path,
    *,
    pattern: str = "*.elle",
    attribute: str = "auto",
) -> list[dict[str, Any]]:
    directory_path = Path(directory)
    snapshots = [summarize_elle_label_components(path, attribute=attribute) for path in sorted(directory_path.glob(pattern))]
    return sorted(
        snapshots,
        key=lambda snapshot: (
            snapshot["step"] is None,
            snapshot["step"] if snapshot["step"] is not None else snapshot["path"],
        ),
    )


def evaluate_static_grain_growth_benchmark(
    reference_dir: str | Path,
    *,
    candidate_dir: str | Path | None = None,
    pattern: str = "*.elle",
) -> dict[str, Any]:
    reference = collect_elle_microstructure_snapshots(reference_dir, pattern=pattern)
    report: dict[str, Any] = {
        "reference_dir": str(Path(reference_dir)),
        "reference_sequence": reference,
        "reference_trends": summarize_sequence_trends(reference),
    }

    if candidate_dir is not None:
        candidate = collect_elle_microstructure_snapshots(candidate_dir, pattern=pattern)
        report["candidate_dir"] = str(Path(candidate_dir))
        report["candidate_sequence"] = candidate
        report["candidate_trends"] = summarize_sequence_trends(candidate)
        report["comparison"] = {
            "grain_count": _matched_metric_trajectory(reference, candidate, "grain_count"),
            "mean_grain_area": _matched_metric_trajectory(reference, candidate, "mean_grain_area"),
            "mean_equivalent_radius": _matched_metric_trajectory(reference, candidate, "mean_equivalent_radius"),
        }

    return report


def evaluate_rasterized_grain_growth_benchmark(
    reference_dir: str | Path,
    *,
    candidate_dir: str | Path | None = None,
    pattern: str = "*.elle",
    attribute: str = "auto",
) -> dict[str, Any]:
    reference = collect_elle_label_component_snapshots(reference_dir, pattern=pattern, attribute=attribute)
    report: dict[str, Any] = {
        "reference_dir": str(Path(reference_dir)),
        "reference_sequence": reference,
        "reference_trends": summarize_sequence_trends(reference),
    }

    if candidate_dir is not None:
        candidate = collect_elle_label_component_snapshots(candidate_dir, pattern=pattern, attribute=attribute)
        report["candidate_dir"] = str(Path(candidate_dir))
        report["candidate_sequence"] = candidate
        report["candidate_trends"] = summarize_sequence_trends(candidate)
        report["comparison"] = {
            "grain_count": _matched_metric_trajectory(reference, candidate, "grain_count"),
            "mean_grain_area": _matched_metric_trajectory(reference, candidate, "mean_grain_area"),
            "mean_equivalent_radius": _matched_metric_trajectory(reference, candidate, "mean_equivalent_radius"),
        }

    return report


def evaluate_release_dataset_benchmarks(data_dir: str | Path) -> dict[str, Any]:
    summary = summarize_liu_suckale_datasets(data_dir)
    datasets = summary["datasets"]

    def _find_entry(variable: str, strain: str, pressure: str, temperature: str) -> dict[str, Any]:
        for entry in datasets:
            if (
                entry.get("variable") == variable
                and entry.get("strain") == strain
                and entry.get("pressure") == pressure
                and entry.get("temperature") == temperature
            ):
                return entry
        raise KeyError(f"missing dataset {variable} S{strain} H{pressure} T{temperature}")

    hot_grain = _find_entry("grain_kde", "1800000e-14", "1000", "-1.0")
    cold_grain = _find_entry("grain_kde", "21e-14", "1", "-26.0")

    grain_hot_shift = float(hot_grain["summary"]["mean_index_end"] - hot_grain["summary"]["mean_index_start"])
    grain_cold_shift = float(cold_grain["summary"]["mean_index_end"] - cold_grain["summary"]["mean_index_start"])

    euler_expectations: list[dict[str, Any]] = []
    for variable in ("euler_1", "euler_2", "euler_3"):
        hot = _find_entry(variable, "1800000e-14", "1000", "-1.0")
        cold = _find_entry(variable, "21e-14", "1", "-26.0")
        hot_change = abs(float(hot["summary"]["axial_mean_end_deg"]) - float(hot["summary"]["axial_mean_start_deg"]))
        cold_change = abs(
            float(cold["summary"]["axial_mean_end_deg"]) - float(cold["summary"]["axial_mean_start_deg"])
        )
        euler_expectations.append(
            {
                "variable": variable,
                "hot_change_deg": hot_change,
                "cold_change_deg": cold_change,
                "hot_exceeds_cold": bool(hot_change > cold_change + 1.0e-9),
            }
        )

    benchmark_flags = {
        "grain_hotter_case_more_active": bool(grain_hot_shift > grain_cold_shift + 1.0e-9),
        "all_euler_hotter_cases_more_active": bool(all(item["hot_exceeds_cold"] for item in euler_expectations)),
    }

    return {
        "dataset_summary": summary,
        "benchmarks": {
            "grain_size_activity": {
                "hot_scenario": hot_grain["name"],
                "cold_scenario": cold_grain["name"],
                "hot_mean_index_shift": grain_hot_shift,
                "cold_mean_index_shift": grain_cold_shift,
                "hotter_case_more_active": benchmark_flags["grain_hotter_case_more_active"],
            },
            "euler_activity": euler_expectations,
        },
        "flags": benchmark_flags,
    }


def build_benchmark_validation_report(
    *,
    reference_dir: str | Path,
    data_dir: str | Path,
    candidate_dir: str | Path | None = None,
    pattern: str = "*.elle",
) -> dict[str, Any]:
    return {
        "static_grain_growth": evaluate_static_grain_growth_benchmark(
            reference_dir,
            candidate_dir=candidate_dir,
            pattern=pattern,
        ),
        "rasterized_grain_growth": evaluate_rasterized_grain_growth_benchmark(
            reference_dir,
            candidate_dir=candidate_dir,
            pattern=pattern,
        ),
        "release_dataset_benchmarks": evaluate_release_dataset_benchmarks(data_dir),
    }


def write_benchmark_validation_report(path: str | Path, report: dict[str, Any]) -> Path:
    outpath = Path(path)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    outpath.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return outpath
