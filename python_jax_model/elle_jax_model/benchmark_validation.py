from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from .legacy_statistics import (
    compare_mesh_bookkeeping_to_legacy_old_stats,
    compare_snapshot_summary_to_legacy_statistics,
)
from .microstructure_validation import (
    collect_elle_microstructure_snapshots,
    summarize_grain_survival_diagnostics,
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


def _nested_snapshot_value(snapshot: dict[str, Any], path: tuple[str, ...]) -> float | None:
    current: Any = snapshot
    for key in path:
        if isinstance(current, dict):
            if key not in current:
                return None
            current = current[key]
            continue
        if isinstance(current, (list, tuple)):
            try:
                current = current[int(key)]
            except (TypeError, ValueError, IndexError):
                return None
            continue
        return None
    if current is None:
        return None
    return float(current)


def _cumulative_trapezoid_series(steps: np.ndarray, values: np.ndarray) -> np.ndarray:
    values_np = np.asarray(values, dtype=np.float64).reshape(-1)
    steps_np = np.asarray(steps, dtype=np.float64).reshape(-1)
    if values_np.size == 0:
        return np.zeros(0, dtype=np.float64)
    cumulative = np.zeros_like(values_np, dtype=np.float64)
    if values_np.size >= 2 and steps_np.size >= 2:
        increments = 0.5 * (values_np[:-1] + values_np[1:]) * np.diff(steps_np)
        cumulative[1:] = np.cumsum(increments, dtype=np.float64)
    return cumulative


def summarize_sequence_trends(sequence: list[dict[str, Any]]) -> dict[str, Any]:
    if not sequence:
        return {"num_snapshots": 0, "steps": [], "metrics": {}, "flags": {}, "curves": {}}

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
    normalized_steps = [
        int(step) if float(step).is_integer() else float(step)
        for step in steps.tolist()
    ]

    metric_names = (
        "grain_count",
        "mean_grain_area",
        "second_moment_grain_size",
        "mean_equivalent_radius",
        "mean_shape_factor",
        "mean_aspect_ratio",
    )
    metrics: dict[str, Any] = {}
    curves: dict[str, Any] = {}
    for name in metric_names:
        if not all(name in snapshot for snapshot in ordered):
            continue
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

    fabric_metric_paths = {
        "fabric_c_axis_largest_eigenvalue": ("fabric", "c_axis", "eigenvalues", "0"),
        "fabric_c_axis_p_index": ("fabric", "c_axis", "P_index"),
        "fabric_c_axis_g_index": ("fabric", "c_axis", "G_index"),
        "fabric_c_axis_r_index": ("fabric", "c_axis", "R_index"),
        "fabric_c_axis_vertical_fraction_15deg": ("fabric", "c_axis", "pole_figure", "fraction_within_15deg"),
        "fabric_c_axis_mean_colatitude_deg": ("fabric", "c_axis", "pole_figure", "mean_colatitude_deg"),
        "fabric_a_axis_p_index": ("fabric", "a_axis", "P_index"),
        "fabric_prism_normal_p_index": ("fabric", "prism_normal", "P_index"),
    }
    fabric_available = all("fabric" in snapshot for snapshot in ordered)
    if fabric_available:
        for metric_name, path in fabric_metric_paths.items():
            values: list[float] = []
            valid = True
            for snapshot in ordered:
                current: Any = snapshot
                for key in path[:-1]:
                    if not isinstance(current, dict) or key not in current:
                        valid = False
                        break
                    current = current[key]
                if not valid:
                    break
                terminal = path[-1]
                if terminal == "0":
                    if not isinstance(current, (list, tuple)) or not current:
                        valid = False
                        break
                    values.append(float(current[0]))
                else:
                    if not isinstance(current, dict) or terminal not in current:
                        valid = False
                        break
                    values.append(float(current[terminal]))
            if valid:
                metrics[metric_name] = _series_summary(steps, np.asarray(values, dtype=np.float64))

    mechanics_metric_paths = {
        "mechanics_mean_normalized_strain_rate": ("mechanics", "mean_normalized_strain_rate"),
        "mechanics_mean_normalized_stress": ("mechanics", "mean_normalized_stress"),
        "mechanics_mean_von_mises_strain_rate": ("mechanics", "mean_von_mises_strain_rate"),
        "mechanics_mean_von_mises_stress": ("mechanics", "mean_von_mises_stress"),
        "mechanics_mean_differential_stress": ("mechanics", "mean_differential_stress"),
        "mechanics_stress_field_error": ("mechanics", "stress_field_error"),
        "mechanics_strain_rate_field_error": ("mechanics", "strain_rate_field_error"),
        "mechanics_mean_basal_activity": ("mechanics", "mean_basal_activity"),
        "mechanics_mean_prismatic_activity": ("mechanics", "mean_prismatic_activity"),
        "mechanics_mean_pyramidal_activity": ("mechanics", "mean_pyramidal_activity"),
        "mechanics_mean_total_activity": ("mechanics", "mean_total_activity"),
        "mechanics_mean_prismatic_fraction": ("mechanics", "mean_prismatic_fraction"),
        "mechanics_prismatic_to_basal_ratio": ("mechanics", "prismatic_to_basal_ratio"),
    }
    mechanics_available = all("mechanics" in snapshot for snapshot in ordered)
    mechanics_series: dict[str, np.ndarray] = {}
    if mechanics_available:
        for metric_name, path in mechanics_metric_paths.items():
            values: list[float] = []
            valid = True
            for snapshot in ordered:
                value = _nested_snapshot_value(snapshot, path)
                if value is None:
                    valid = False
                    break
                values.append(float(value))
            if valid:
                value_array = np.asarray(values, dtype=np.float64)
                metrics[metric_name] = _series_summary(steps, value_array)
                mechanics_series[metric_name] = value_array
        direct_strain_values: list[float] = []
        direct_strain_available = True
        direct_strain_source = "integrated_mean_normalized_strain_rate"
        for snapshot in ordered:
            value = _nested_snapshot_value(snapshot, ("mechanics", "direct_strain_axis"))
            if value is None:
                direct_strain_available = False
                break
            direct_strain_values.append(float(value))
            mechanics_summary = snapshot.get("mechanics", {})
            if isinstance(mechanics_summary, dict) and "strain_axis_source" in mechanics_summary:
                direct_strain_source = str(mechanics_summary["strain_axis_source"])
        required_curve_series = {
            "mechanics_mean_basal_activity",
            "mechanics_mean_prismatic_activity",
            "mechanics_mean_prismatic_fraction",
        }
        if required_curve_series.issubset(mechanics_series):
            if direct_strain_available:
                cumulative_strain = np.asarray(direct_strain_values, dtype=np.float64)
            elif "mechanics_mean_normalized_strain_rate" in mechanics_series:
                cumulative_strain = _cumulative_trapezoid_series(
                    steps,
                    mechanics_series["mechanics_mean_normalized_strain_rate"],
                )
            elif "mechanics_mean_von_mises_strain_rate" in mechanics_series:
                cumulative_strain = _cumulative_trapezoid_series(
                    steps,
                    mechanics_series["mechanics_mean_von_mises_strain_rate"],
                )
                direct_strain_source = "integrated_von_mises_strain_rate"
            else:
                cumulative_strain = None
            stress_metric_name = (
                "mechanics_mean_differential_stress"
                if "mechanics_mean_differential_stress" in mechanics_series
                else (
                    "mechanics_mean_normalized_stress"
                    if "mechanics_mean_normalized_stress" in mechanics_series
                    else (
                        "mechanics_mean_von_mises_stress"
                        if "mechanics_mean_von_mises_stress" in mechanics_series
                        else None
                    )
                )
            )
            if cumulative_strain is not None and stress_metric_name in mechanics_series:
                metrics["mechanics_cumulative_normalized_strain"] = _series_summary(steps, cumulative_strain)
                curves["mechanics_stress_strain"] = {
                    "steps": normalized_steps,
                    "strain": cumulative_strain.tolist(),
                    "stress": mechanics_series[stress_metric_name].tolist(),
                    "strain_rate": (
                        mechanics_series["mechanics_mean_normalized_strain_rate"].tolist()
                        if "mechanics_mean_normalized_strain_rate" in mechanics_series
                        else mechanics_series["mechanics_mean_von_mises_strain_rate"].tolist()
                    ),
                    "strain_source": direct_strain_source,
                    "stress_source": (
                        "mean_differential_stress"
                        if stress_metric_name == "mechanics_mean_differential_stress"
                        else (
                            "mean_normalized_stress"
                            if stress_metric_name == "mechanics_mean_normalized_stress"
                            else "mean_von_mises_stress"
                        )
                    ),
                }
            if cumulative_strain is not None:
                activity_curve: dict[str, Any] = {
                    "steps": normalized_steps,
                    "strain": cumulative_strain.tolist(),
                    "basal_activity": mechanics_series["mechanics_mean_basal_activity"].tolist(),
                    "prismatic_activity": mechanics_series["mechanics_mean_prismatic_activity"].tolist(),
                    "prismatic_fraction": mechanics_series["mechanics_mean_prismatic_fraction"].tolist(),
                    "strain_source": direct_strain_source,
                }
                if "mechanics_mean_pyramidal_activity" in mechanics_series:
                    activity_curve["pyramidal_activity"] = mechanics_series["mechanics_mean_pyramidal_activity"].tolist()
                if "mechanics_mean_total_activity" in mechanics_series:
                    activity_curve["total_activity"] = mechanics_series["mechanics_mean_total_activity"].tolist()
                if "mechanics_prismatic_to_basal_ratio" in mechanics_series:
                    activity_curve["prismatic_to_basal_ratio"] = mechanics_series["mechanics_prismatic_to_basal_ratio"].tolist()
                curves["mechanics_activity_strain"] = activity_curve

    flags = {
        "coarsening_present": bool(
            metrics["mean_grain_area"]["delta"] > 0.0 and metrics["grain_count"]["delta"] <= 0.0
        ),
        "grain_area_mostly_increasing": bool(metrics["mean_grain_area"]["increase_fraction"] >= 0.5),
        "grain_count_mostly_decreasing": bool(metrics["grain_count"]["decrease_fraction"] >= 0.5),
    }

    return {
        "num_snapshots": int(len(ordered)),
        "steps": normalized_steps,
        "metrics": metrics,
        "flags": flags,
        "curves": curves,
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


def _matched_nested_metric_trajectory(
    reference: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    path: tuple[str, ...],
) -> dict[str, Any] | None:
    ref_by_step = {snapshot["step"]: snapshot for snapshot in reference if snapshot.get("step") is not None}
    cand_by_step = {snapshot["step"]: snapshot for snapshot in candidate if snapshot.get("step") is not None}
    matched_steps = sorted(set(ref_by_step) & set(cand_by_step))
    if not matched_steps:
        return None

    ref_values: list[float] = []
    cand_values: list[float] = []
    kept_steps: list[int] = []
    for step in matched_steps:
        ref_value = _nested_snapshot_value(ref_by_step[step], path)
        cand_value = _nested_snapshot_value(cand_by_step[step], path)
        if ref_value is None or cand_value is None:
            continue
        kept_steps.append(int(step))
        ref_values.append(float(ref_value))
        cand_values.append(float(cand_value))
    if not kept_steps:
        return None

    ref_array = np.asarray(ref_values, dtype=np.float64)
    cand_array = np.asarray(cand_values, dtype=np.float64)
    diffs = cand_array - ref_array
    denom = max(float(np.max(np.abs(ref_array))), 1.0e-12)
    return {
        "matched_steps": kept_steps,
        "rmse": float(np.sqrt(np.mean(diffs * diffs))),
        "mae": float(np.mean(np.abs(diffs))),
        "normalized_rmse": float(np.sqrt(np.mean(diffs * diffs)) / denom),
    }


def _matched_trend_curve(
    reference_trends: dict[str, Any],
    candidate_trends: dict[str, Any],
    curve_name: str,
    components: tuple[str, ...],
) -> dict[str, Any] | None:
    reference_curve = dict(reference_trends.get("curves", {})).get(curve_name)
    candidate_curve = dict(candidate_trends.get("curves", {})).get(curve_name)
    if not isinstance(reference_curve, dict) or not isinstance(candidate_curve, dict):
        return None

    reference_steps = tuple(reference_curve.get("steps", ()))
    candidate_steps = tuple(candidate_curve.get("steps", ()))
    if not reference_steps or not candidate_steps:
        return None

    reference_step_index = {step: index for index, step in enumerate(reference_steps)}
    candidate_step_index = {step: index for index, step in enumerate(candidate_steps)}
    matched_steps = [step for step in reference_steps if step in candidate_step_index]
    if not matched_steps:
        return None

    report: dict[str, Any] = {"matched_steps": list(matched_steps)}
    for component in components:
        if component not in reference_curve or component not in candidate_curve:
            return None
        reference_values_all = tuple(reference_curve.get(component, ()))
        candidate_values_all = tuple(candidate_curve.get(component, ()))
        if len(reference_values_all) != len(reference_steps) or len(candidate_values_all) != len(candidate_steps):
            return None
        reference_values = np.asarray(
            [float(reference_values_all[reference_step_index[step]]) for step in matched_steps],
            dtype=np.float64,
        )
        candidate_values = np.asarray(
            [float(candidate_values_all[candidate_step_index[step]]) for step in matched_steps],
            dtype=np.float64,
        )
        diffs = candidate_values - reference_values
        denom = max(float(np.max(np.abs(reference_values))), 1.0e-12)
        report[component] = {
            "reference_values": reference_values.tolist(),
            "candidate_values": candidate_values.tolist(),
            "rmse": float(np.sqrt(np.mean(diffs * diffs))),
            "mae": float(np.mean(np.abs(diffs))),
            "normalized_rmse": float(np.sqrt(np.mean(diffs * diffs)) / denom),
        }
    return report


def _metric_delta(trends: dict[str, Any], metric_name: str) -> float | None:
    metrics = dict(trends.get("metrics", {}))
    metric = metrics.get(metric_name)
    if not isinstance(metric, dict) or "delta" not in metric:
        return None
    return float(metric["delta"])


def _comparison_nrmse(comparison: dict[str, Any], metric_name: str) -> float | None:
    metric = comparison.get(metric_name)
    if not isinstance(metric, dict) or "normalized_rmse" not in metric:
        return None
    return float(metric["normalized_rmse"])


def _survival_summary_difference(
    reference: dict[str, Any] | None,
    candidate: dict[str, Any] | None,
    key: str,
) -> float | None:
    if not isinstance(reference, dict) or not isinstance(candidate, dict):
        return None
    ref_summary = dict(reference.get("summary", {}))
    cand_summary = dict(candidate.get("summary", {}))
    if key not in ref_summary or key not in cand_summary:
        return None
    return float(cand_summary[key]) - float(ref_summary[key])


def assess_paper_signature_alignment(
    *,
    reference_trends: dict[str, Any],
    candidate_trends: dict[str, Any],
    comparison: dict[str, Any],
    reference_survival: dict[str, Any] | None = None,
    candidate_survival: dict[str, Any] | None = None,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def _append_direction_check(
        name: str,
        metric_name: str,
        expected_direction: str,
        *,
        nrmse_threshold: float | None = None,
    ) -> None:
        reference_delta = _metric_delta(reference_trends, metric_name)
        candidate_delta = _metric_delta(candidate_trends, metric_name)
        normalized_rmse = _comparison_nrmse(comparison, metric_name)
        if reference_delta is None or candidate_delta is None:
            checks.append(
                {
                    "name": name,
                    "status": "missing",
                    "metric": metric_name,
                }
            )
            return

        if expected_direction == "increase":
            direction_match = bool(reference_delta >= 0.0 and candidate_delta >= 0.0)
        elif expected_direction == "decrease":
            direction_match = bool(reference_delta <= 0.0 and candidate_delta <= 0.0)
        else:
            raise ValueError(f"unsupported expected direction {expected_direction}")

        within_error = True
        if nrmse_threshold is not None and normalized_rmse is not None:
            within_error = bool(normalized_rmse <= float(nrmse_threshold))

        checks.append(
            {
                "name": name,
                "status": "pass" if direction_match and within_error else "fail",
                "metric": metric_name,
                "expected_direction": expected_direction,
                "reference_delta": reference_delta,
                "candidate_delta": candidate_delta,
                "normalized_rmse": normalized_rmse,
                "normalized_rmse_threshold": nrmse_threshold,
            }
        )

    _append_direction_check(
        "coarsening_mean_area",
        "mean_grain_area",
        "increase",
        nrmse_threshold=0.25,
    )
    _append_direction_check(
        "grain_count_decline",
        "grain_count",
        "decrease",
        nrmse_threshold=0.25,
    )
    _append_direction_check(
        "equidimensionalization",
        "mean_aspect_ratio",
        "decrease",
        nrmse_threshold=0.35,
    )
    _append_direction_check(
        "c_axis_single_max_strengthening",
        "fabric_c_axis_p_index",
        "increase",
        nrmse_threshold=0.35,
    )
    _append_direction_check(
        "c_axis_vertical_clustering",
        "fabric_c_axis_vertical_fraction_15deg",
        "increase",
        nrmse_threshold=0.35,
    )
    _append_direction_check(
        "c_axis_colatitude_drop",
        "fabric_c_axis_mean_colatitude_deg",
        "decrease",
        nrmse_threshold=0.10,
    )
    stress_metric_name = (
        "mechanics_mean_differential_stress"
        if _metric_delta(reference_trends, "mechanics_mean_differential_stress") is not None
        and _metric_delta(candidate_trends, "mechanics_mean_differential_stress") is not None
        and _comparison_nrmse(comparison, "mechanics_mean_differential_stress") is not None
        else (
            "mechanics_mean_normalized_stress"
            if _metric_delta(reference_trends, "mechanics_mean_normalized_stress") is not None
            and _metric_delta(candidate_trends, "mechanics_mean_normalized_stress") is not None
            and _comparison_nrmse(comparison, "mechanics_mean_normalized_stress") is not None
            else "mechanics_mean_von_mises_stress"
        )
    )
    _append_direction_check(
        "mean_stress_trend",
        stress_metric_name,
        "decrease",
        nrmse_threshold=0.35,
    )
    _append_direction_check(
        "prismatic_activity_trend",
        "mechanics_mean_prismatic_activity",
        "increase",
        nrmse_threshold=0.50,
    )
    _append_direction_check(
        "pyramidal_activity_trend",
        "mechanics_mean_pyramidal_activity",
        "increase",
        nrmse_threshold=0.50,
    )
    _append_direction_check(
        "prismatic_to_basal_ratio_trend",
        "mechanics_prismatic_to_basal_ratio",
        "increase",
        nrmse_threshold=0.50,
    )

    stress_strain_curve = comparison.get("mechanics_stress_strain_curve")
    if isinstance(stress_strain_curve, dict):
        strain_curve = stress_strain_curve.get("strain")
        stress_curve = stress_strain_curve.get("stress")
        if isinstance(strain_curve, dict) and isinstance(stress_curve, dict):
            strain_nrmse = strain_curve.get("normalized_rmse")
            stress_nrmse = stress_curve.get("normalized_rmse")
            if strain_nrmse is not None and stress_nrmse is not None:
                checks.append(
                    {
                        "name": "stress_strain_curve_alignment",
                        "status": (
                            "pass"
                            if float(strain_nrmse) <= 0.35 and float(stress_nrmse) <= 0.35
                            else "fail"
                        ),
                        "strain_normalized_rmse": float(strain_nrmse),
                        "stress_normalized_rmse": float(stress_nrmse),
                        "normalized_rmse_threshold": 0.35,
                    }
                )

    activity_curve = comparison.get("mechanics_activity_strain_curve")
    if isinstance(activity_curve, dict):
        basal_curve = activity_curve.get("basal_activity")
        prismatic_curve = activity_curve.get("prismatic_activity")
        fraction_curve = activity_curve.get("prismatic_fraction")
        if (
            isinstance(basal_curve, dict)
            and isinstance(prismatic_curve, dict)
            and isinstance(fraction_curve, dict)
        ):
            basal_nrmse = basal_curve.get("normalized_rmse")
            prismatic_nrmse = prismatic_curve.get("normalized_rmse")
            fraction_nrmse = fraction_curve.get("normalized_rmse")
            pyramidal_curve = activity_curve.get("pyramidal_activity")
            pyramidal_nrmse = (
                pyramidal_curve.get("normalized_rmse")
                if isinstance(pyramidal_curve, dict)
                else None
            )
            if basal_nrmse is not None and prismatic_nrmse is not None and fraction_nrmse is not None:
                max_curve_nrmse = max(
                    float(basal_nrmse),
                    float(prismatic_nrmse),
                    float(fraction_nrmse),
                    *([float(pyramidal_nrmse)] if pyramidal_nrmse is not None else []),
                )
                checks.append(
                    {
                        "name": "activity_curve_alignment",
                        "status": (
                            "pass"
                            if max_curve_nrmse <= 0.50
                            else "fail"
                        ),
                        "basal_normalized_rmse": float(basal_nrmse),
                        "prismatic_normalized_rmse": float(prismatic_nrmse),
                        "prismatic_fraction_normalized_rmse": float(fraction_nrmse),
                        "pyramidal_normalized_rmse": (
                            None if pyramidal_nrmse is None else float(pyramidal_nrmse)
                        ),
                        "normalized_rmse_threshold": 0.50,
                    }
                )

    ref_survival_summary = dict(reference_survival.get("summary", {})) if isinstance(reference_survival, dict) else {}
    cand_survival_summary = dict(candidate_survival.get("summary", {})) if isinstance(candidate_survival, dict) else {}
    ref_survival_flag = ref_survival_summary.get("initial_size_correlation_stronger_than_schmid")
    cand_survival_flag = cand_survival_summary.get("initial_size_correlation_stronger_than_schmid")
    if isinstance(ref_survival_flag, bool) and isinstance(cand_survival_flag, bool):
        checks.append(
            {
                "name": "size_stronger_than_schmid_survival",
                "status": (
                    "pass"
                    if ref_survival_flag and cand_survival_flag
                    else "fail"
                ),
                "reference_value": ref_survival_flag,
                "candidate_value": cand_survival_flag,
            }
        )

    applicable = [entry for entry in checks if entry["status"] != "missing"]
    passed = [entry for entry in applicable if entry["status"] == "pass"]
    return {
        "checks": checks,
        "applicable_checks": int(len(applicable)),
        "passed_checks": int(len(passed)),
        "pass_fraction": float(len(passed) / len(applicable)) if applicable else float("nan"),
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
    reference_survival = summarize_grain_survival_diagnostics(reference_dir, pattern=pattern)
    reference_trends = summarize_sequence_trends(reference)
    report: dict[str, Any] = {
        "reference_dir": str(Path(reference_dir)),
        "reference_sequence": reference,
        "reference_trends": reference_trends,
        "reference_survival_diagnostics": reference_survival,
    }

    if candidate_dir is not None:
        candidate = collect_elle_microstructure_snapshots(candidate_dir, pattern=pattern)
        candidate_survival = summarize_grain_survival_diagnostics(candidate_dir, pattern=pattern)
        candidate_trends = summarize_sequence_trends(candidate)
        report["candidate_dir"] = str(Path(candidate_dir))
        report["candidate_sequence"] = candidate
        report["candidate_trends"] = candidate_trends
        report["candidate_survival_diagnostics"] = candidate_survival
        report["comparison"] = {
            "grain_count": _matched_metric_trajectory(reference, candidate, "grain_count"),
            "mean_grain_area": _matched_metric_trajectory(reference, candidate, "mean_grain_area"),
            "second_moment_grain_size": _matched_metric_trajectory(reference, candidate, "second_moment_grain_size"),
            "mean_equivalent_radius": _matched_metric_trajectory(reference, candidate, "mean_equivalent_radius"),
            "mean_aspect_ratio": _matched_metric_trajectory(reference, candidate, "mean_aspect_ratio"),
            "fabric_c_axis_largest_eigenvalue": _matched_nested_metric_trajectory(
                reference,
                candidate,
                ("fabric", "c_axis", "eigenvalues", "0"),
            ),
            "fabric_c_axis_p_index": _matched_nested_metric_trajectory(
                reference,
                candidate,
                ("fabric", "c_axis", "P_index"),
            ),
            "fabric_c_axis_vertical_fraction_15deg": _matched_nested_metric_trajectory(
                reference,
                candidate,
                ("fabric", "c_axis", "pole_figure", "fraction_within_15deg"),
            ),
            "mechanics_mean_normalized_stress": _matched_nested_metric_trajectory(
                reference,
                candidate,
                ("mechanics", "mean_normalized_stress"),
            ),
            "mechanics_mean_von_mises_stress": _matched_nested_metric_trajectory(
                reference,
                candidate,
                ("mechanics", "mean_von_mises_stress"),
            ),
            "mechanics_mean_von_mises_strain_rate": _matched_nested_metric_trajectory(
                reference,
                candidate,
                ("mechanics", "mean_von_mises_strain_rate"),
            ),
            "mechanics_mean_differential_stress": _matched_nested_metric_trajectory(
                reference,
                candidate,
                ("mechanics", "mean_differential_stress"),
            ),
            "mechanics_stress_field_error": _matched_nested_metric_trajectory(
                reference,
                candidate,
                ("mechanics", "stress_field_error"),
            ),
            "mechanics_strain_rate_field_error": _matched_nested_metric_trajectory(
                reference,
                candidate,
                ("mechanics", "strain_rate_field_error"),
            ),
            "mechanics_mean_prismatic_activity": _matched_nested_metric_trajectory(
                reference,
                candidate,
                ("mechanics", "mean_prismatic_activity"),
            ),
            "mechanics_mean_pyramidal_activity": _matched_nested_metric_trajectory(
                reference,
                candidate,
                ("mechanics", "mean_pyramidal_activity"),
            ),
            "mechanics_prismatic_to_basal_ratio": _matched_nested_metric_trajectory(
                reference,
                candidate,
                ("mechanics", "prismatic_to_basal_ratio"),
            ),
            "mechanics_stress_strain_curve": _matched_trend_curve(
                reference_trends,
                candidate_trends,
                "mechanics_stress_strain",
                ("strain", "stress", "strain_rate"),
            ),
            "mechanics_activity_strain_curve": _matched_trend_curve(
                reference_trends,
                candidate_trends,
                "mechanics_activity_strain",
                tuple(
                    component
                    for component in (
                        "strain",
                        "basal_activity",
                        "prismatic_activity",
                        "pyramidal_activity",
                        "prismatic_fraction",
                    )
                    if (
                        component != "pyramidal_activity"
                        or (
                            isinstance(reference_trends.get("curves", {}).get("mechanics_activity_strain"), dict)
                            and "pyramidal_activity" in reference_trends.get("curves", {}).get("mechanics_activity_strain", {})
                            and isinstance(candidate_trends.get("curves", {}).get("mechanics_activity_strain"), dict)
                            and "pyramidal_activity" in candidate_trends.get("curves", {}).get("mechanics_activity_strain", {})
                        )
                    )
                ),
            ),
            "fabric_a_axis_p_index": _matched_nested_metric_trajectory(
                reference,
                candidate,
                ("fabric", "a_axis", "P_index"),
            ),
        }
        report["survival_comparison"] = {
            "survival_fraction_diff": _survival_summary_difference(
                reference_survival,
                candidate_survival,
                "survival_fraction",
            ),
            "initial_size_survival_correlation_diff": _survival_summary_difference(
                reference_survival,
                candidate_survival,
                "initial_size_survival_correlation",
            ),
            "initial_basal_schmid_survival_correlation_diff": _survival_summary_difference(
                reference_survival,
                candidate_survival,
                "initial_basal_schmid_survival_correlation",
            ),
        }
        report["paper_signature_assessment"] = assess_paper_signature_alignment(
            reference_trends=report["reference_trends"],
            candidate_trends=report["candidate_trends"],
            comparison=report["comparison"],
            reference_survival=reference_survival,
            candidate_survival=candidate_survival,
        )

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


def assess_release_dataset_signature_alignment(report: dict[str, Any]) -> dict[str, Any]:
    benchmarks = dict(report.get("benchmarks", {}))
    flags = dict(report.get("flags", {}))
    checks: list[dict[str, Any]] = []

    grain_activity = dict(benchmarks.get("grain_size_activity", {}))
    if grain_activity:
        hot_shift = grain_activity.get("hot_mean_index_shift")
        cold_shift = grain_activity.get("cold_mean_index_shift")
        hotter_case_more_active = grain_activity.get("hotter_case_more_active")
        if hot_shift is not None and cold_shift is not None and hotter_case_more_active is not None:
            checks.append(
                {
                    "name": "grain_size_hotter_case_more_active",
                    "status": "pass" if bool(hotter_case_more_active) else "fail",
                    "hot_mean_index_shift": float(hot_shift),
                    "cold_mean_index_shift": float(cold_shift),
                }
            )

    for entry in benchmarks.get("euler_activity", ()):
        if not isinstance(entry, dict):
            continue
        variable = str(entry.get("variable", "unknown"))
        hot_change = entry.get("hot_change_deg")
        cold_change = entry.get("cold_change_deg")
        hot_exceeds_cold = entry.get("hot_exceeds_cold")
        if hot_change is None or cold_change is None or hot_exceeds_cold is None:
            continue
        checks.append(
            {
                "name": f"{variable}_hotter_case_more_active",
                "status": "pass" if bool(hot_exceeds_cold) else "fail",
                "hot_change_deg": float(hot_change),
                "cold_change_deg": float(cold_change),
            }
        )

    if "all_euler_hotter_cases_more_active" in flags:
        checks.append(
            {
                "name": "all_euler_hotter_cases_more_active",
                "status": "pass" if bool(flags["all_euler_hotter_cases_more_active"]) else "fail",
            }
        )

    applicable = [entry for entry in checks if entry["status"] != "missing"]
    passed = [entry for entry in applicable if entry["status"] == "pass"]
    return {
        "checks": checks,
        "applicable_checks": int(len(applicable)),
        "passed_checks": int(len(passed)),
        "pass_fraction": float(len(passed) / len(applicable)) if applicable else float("nan"),
    }


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

    report = {
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
    report["paper_signature_assessment"] = assess_release_dataset_signature_alignment(report)
    return report


def _family_sort_key(family_id: str) -> tuple[int, int | str]:
    text = str(family_id)
    try:
        return (0, int(text))
    except ValueError:
        return (1, text)


def _family_metric_value(
    family_report: dict[str, Any],
    metric_name: str,
    field_name: str,
) -> float | None:
    trends = dict(family_report.get("reference_trends", {}))
    metrics = dict(trends.get("metrics", {}))
    metric = metrics.get(metric_name)
    if not isinstance(metric, dict) or field_name not in metric:
        return None
    return float(metric[field_name])


def _family_peak_stress_strain(family_report: dict[str, Any]) -> float | None:
    trends = dict(family_report.get("reference_trends", {}))
    curves = dict(trends.get("curves", {}))
    curve = curves.get("mechanics_stress_strain")
    if not isinstance(curve, dict):
        return None
    stress = np.asarray(curve.get("stress", ()), dtype=np.float64)
    strain = np.asarray(curve.get("strain", ()), dtype=np.float64)
    if stress.size == 0 or strain.size == 0 or stress.size != strain.size:
        return None
    return float(strain[int(np.argmax(stress))])


def _normalize_experiment_family_static_report(source: dict[str, Any]) -> dict[str, Any]:
    static_report = source
    if "static_grain_growth" in source and isinstance(source["static_grain_growth"], dict):
        static_report = dict(source["static_grain_growth"])
    return {
        "reference_dir": static_report.get("reference_dir"),
        "reference_trends": static_report.get("reference_trends", {}),
        "reference_survival_diagnostics": static_report.get("reference_survival_diagnostics", {}),
    }


def load_experiment_family_benchmark_report(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"invalid experiment-family benchmark report {path}, expected JSON object")
    normalized = _normalize_experiment_family_static_report(payload)
    normalized["source_report_path"] = str(Path(path))
    return normalized


def _build_experiment_family_benchmark_report(
    families: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    ordered_family_ids = sorted((str(key) for key in families), key=_family_sort_key)
    benchmarks = {
        "end_mean_grain_area": {
            family_id: _family_metric_value(families[family_id], "mean_grain_area", "end")
            for family_id in ordered_family_ids
        },
        "end_mean_aspect_ratio": {
            family_id: _family_metric_value(families[family_id], "mean_aspect_ratio", "end")
            for family_id in ordered_family_ids
        },
        "end_mean_differential_stress": {
            family_id: _family_metric_value(families[family_id], "mechanics_mean_differential_stress", "end")
            for family_id in ordered_family_ids
        },
        "peak_stress_strain": {
            family_id: _family_peak_stress_strain(families[family_id])
            for family_id in ordered_family_ids
        },
    }

    report = {
        "family_order": ordered_family_ids,
        "families": families,
        "benchmarks": benchmarks,
    }
    report["paper_signature_assessment"] = assess_experiment_family_signature_alignment(families)
    return report


def assess_experiment_family_signature_alignment(
    family_reports: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    ordered_family_ids = sorted((str(key) for key in family_reports), key=_family_sort_key)
    checks: list[dict[str, Any]] = []

    def _append_family0_nearly_constant_check() -> None:
        family0 = family_reports.get("0")
        if not isinstance(family0, dict):
            checks.append(
                {
                    "name": "family0_mean_area_nearly_constant",
                    "status": "missing",
                    "family_id": "0",
                }
            )
            return
        relative_delta = _family_metric_value(family0, "mean_grain_area", "relative_delta")
        if relative_delta is None:
            checks.append(
                {
                    "name": "family0_mean_area_nearly_constant",
                    "status": "missing",
                    "family_id": "0",
                    "metric": "mean_grain_area",
                }
            )
            return
        checks.append(
            {
                "name": "family0_mean_area_nearly_constant",
                "status": "pass" if abs(float(relative_delta)) <= 0.25 else "fail",
                "family_id": "0",
                "relative_delta": float(relative_delta),
                "relative_delta_threshold": 0.25,
            }
        )

    def _append_monotonic_metric_check(
        name: str,
        metric_name: str,
        field_name: str,
        direction: str,
    ) -> None:
        values: list[float] = []
        for family_id in ordered_family_ids:
            family_report = family_reports.get(family_id, {})
            value = _family_metric_value(family_report, metric_name, field_name)
            if value is None:
                checks.append(
                    {
                        "name": name,
                        "status": "missing",
                        "family_ids": ordered_family_ids,
                        "metric": metric_name,
                        "field": field_name,
                    }
                )
                return
            values.append(float(value))
        if len(values) < 2:
            checks.append(
                {
                    "name": name,
                    "status": "missing",
                    "family_ids": ordered_family_ids,
                    "metric": metric_name,
                    "field": field_name,
                }
            )
            return
        diffs = np.diff(np.asarray(values, dtype=np.float64))
        if direction == "increase":
            passed = bool(np.all(diffs >= -1.0e-9))
        elif direction == "decrease":
            passed = bool(np.all(diffs <= 1.0e-9))
        else:
            raise ValueError(f"unsupported family direction {direction}")
        checks.append(
            {
                "name": name,
                "status": "pass" if passed else "fail",
                "family_ids": ordered_family_ids,
                "metric": metric_name,
                "field": field_name,
                "direction": direction,
                "values": values,
            }
        )

    def _append_all_family_delta_check(name: str, metric_name: str, direction: str) -> None:
        values: list[float] = []
        for family_id in ordered_family_ids:
            family_report = family_reports.get(family_id, {})
            value = _family_metric_value(family_report, metric_name, "delta")
            if value is None:
                checks.append(
                    {
                        "name": name,
                        "status": "missing",
                        "family_ids": ordered_family_ids,
                        "metric": metric_name,
                    }
                )
                return
            values.append(float(value))
        if direction == "increase":
            passed = bool(all(value >= 0.0 for value in values))
        elif direction == "decrease":
            passed = bool(all(value <= 0.0 for value in values))
        else:
            raise ValueError(f"unsupported family direction {direction}")
        checks.append(
            {
                "name": name,
                "status": "pass" if passed else "fail",
                "family_ids": ordered_family_ids,
                "metric": metric_name,
                "direction": direction,
                "deltas": values,
            }
        )

    def _append_peak_strain_check() -> None:
        values: list[float] = []
        for family_id in ordered_family_ids:
            family_report = family_reports.get(family_id, {})
            value = _family_peak_stress_strain(family_report)
            if value is None:
                checks.append(
                    {
                        "name": "higher_drx_peak_stress_strain_decreasing",
                        "status": "missing",
                        "family_ids": ordered_family_ids,
                        "curve": "mechanics_stress_strain",
                    }
                )
                return
            values.append(float(value))
        diffs = np.diff(np.asarray(values, dtype=np.float64))
        checks.append(
            {
                "name": "higher_drx_peak_stress_strain_decreasing",
                "status": "pass" if bool(np.all(diffs <= 1.0e-9)) else "fail",
                "family_ids": ordered_family_ids,
                "peak_strain_values": values,
            }
        )

    def _append_survival_check() -> None:
        values: list[bool] = []
        for family_id in ordered_family_ids:
            family_report = family_reports.get(family_id, {})
            survival = dict(family_report.get("reference_survival_diagnostics", {}))
            summary = dict(survival.get("summary", {}))
            value = summary.get("initial_size_correlation_stronger_than_schmid")
            if not isinstance(value, bool):
                checks.append(
                    {
                        "name": "all_families_size_stronger_than_schmid_survival",
                        "status": "missing",
                        "family_ids": ordered_family_ids,
                    }
                )
                return
            values.append(bool(value))
        checks.append(
            {
                "name": "all_families_size_stronger_than_schmid_survival",
                "status": "pass" if all(values) else "fail",
                "family_ids": ordered_family_ids,
                "values": values,
            }
        )

    _append_family0_nearly_constant_check()
    _append_monotonic_metric_check(
        "higher_drx_mean_area_end_increasing",
        "mean_grain_area",
        "end",
        "increase",
    )
    _append_monotonic_metric_check(
        "higher_drx_aspect_ratio_end_decreasing",
        "mean_aspect_ratio",
        "end",
        "decrease",
    )
    _append_all_family_delta_check(
        "all_families_c_axis_strengthening",
        "fabric_c_axis_p_index",
        "increase",
    )
    _append_all_family_delta_check(
        "all_families_c_axis_vertical_clustering",
        "fabric_c_axis_vertical_fraction_15deg",
        "increase",
    )
    stress_metric_name = None
    if all(
        _family_metric_value(family_reports.get(family_id, {}), "mechanics_mean_differential_stress", "end") is not None
        for family_id in ordered_family_ids
    ):
        stress_metric_name = "mechanics_mean_differential_stress"
    elif all(
        _family_metric_value(family_reports.get(family_id, {}), "mechanics_mean_normalized_stress", "end") is not None
        for family_id in ordered_family_ids
    ):
        stress_metric_name = "mechanics_mean_normalized_stress"
    elif all(
        _family_metric_value(family_reports.get(family_id, {}), "mechanics_mean_von_mises_stress", "end") is not None
        for family_id in ordered_family_ids
    ):
        stress_metric_name = "mechanics_mean_von_mises_stress"
    if stress_metric_name is None:
        checks.append(
            {
                "name": "higher_drx_end_stress_decreasing",
                "status": "missing",
                "family_ids": ordered_family_ids,
            }
        )
    else:
        _append_monotonic_metric_check(
            "higher_drx_end_stress_decreasing",
            stress_metric_name,
            "end",
            "decrease",
        )
    _append_peak_strain_check()
    _append_survival_check()

    applicable = [entry for entry in checks if entry["status"] != "missing"]
    passed = [entry for entry in applicable if entry["status"] == "pass"]
    return {
        "family_order": ordered_family_ids,
        "checks": checks,
        "applicable_checks": int(len(applicable)),
        "passed_checks": int(len(passed)),
        "pass_fraction": float(len(passed) / len(applicable)) if applicable else float("nan"),
    }


def evaluate_experiment_family_benchmark_reports(
    experiment_family_reports: dict[str, str | Path | dict[str, Any]],
) -> dict[str, Any]:
    return evaluate_experiment_family_suite(
        experiment_family_reports=experiment_family_reports,
    )


def evaluate_experiment_family_suite(
    *,
    experiment_family_dirs: dict[str, str | Path] | None = None,
    experiment_family_reports: dict[str, str | Path | dict[str, Any]] | None = None,
    pattern: str = "*.elle",
) -> dict[str, Any]:
    ordered_family_ids = sorted(
        {
            *(str(key) for key in dict(experiment_family_dirs or {})),
            *(str(key) for key in dict(experiment_family_reports or {})),
        },
        key=_family_sort_key,
    )
    families: dict[str, dict[str, Any]] = {}
    for family_id in ordered_family_ids:
        if experiment_family_reports and family_id in experiment_family_reports:
            report_source = experiment_family_reports[family_id]
            if isinstance(report_source, dict):
                family_report = _normalize_experiment_family_static_report(report_source)
            else:
                family_report = load_experiment_family_benchmark_report(report_source)
            families[family_id] = family_report
            continue
        if experiment_family_dirs and family_id in experiment_family_dirs:
            family_dir = experiment_family_dirs[family_id]
            family_report = evaluate_static_grain_growth_benchmark(family_dir, pattern=pattern)
            families[family_id] = _normalize_experiment_family_static_report(family_report)
            continue
        raise ValueError(f"missing experiment-family input for family {family_id!r}")

    return _build_experiment_family_benchmark_report(families)


def evaluate_experiment_family_benchmarks(
    experiment_family_dirs: dict[str, str | Path],
    *,
    pattern: str = "*.elle",
) -> dict[str, Any]:
    ordered_family_ids = sorted((str(key) for key in experiment_family_dirs), key=_family_sort_key)
    return evaluate_experiment_family_suite(
        experiment_family_dirs={family_id: experiment_family_dirs[family_id] for family_id in ordered_family_ids},
        pattern=pattern,
    )


def assess_benchmark_report_acceptance(report: dict[str, Any]) -> dict[str, Any]:
    sections: list[dict[str, Any]] = []
    for section_name in ("static_grain_growth", "release_dataset_benchmarks", "experiment_family_benchmarks"):
        section = report.get(section_name)
        if not isinstance(section, dict):
            continue
        assessment = section.get("paper_signature_assessment")
        if not isinstance(assessment, dict):
            continue
        applicable = int(assessment.get("applicable_checks", 0))
        passed = int(assessment.get("passed_checks", 0))
        sections.append(
            {
                "section": section_name,
                "applicable_checks": applicable,
                "passed_checks": passed,
                "pass_fraction": (
                    float(passed / applicable)
                    if applicable > 0
                    else float("nan")
                ),
            }
        )

    total_applicable = int(sum(section["applicable_checks"] for section in sections))
    total_passed = int(sum(section["passed_checks"] for section in sections))
    return {
        "sections": sections,
        "applicable_checks": total_applicable,
        "passed_checks": total_passed,
        "pass_fraction": (
            float(total_passed / total_applicable)
            if total_applicable > 0
            else float("nan")
        ),
    }


def evaluate_final_snapshot_against_legacy_statistics(
    directory: str | Path,
    legacy_statistics_path: str | Path,
    *,
    pattern: str = "*.elle",
) -> dict[str, Any]:
    snapshots = collect_elle_microstructure_snapshots(directory, pattern=pattern)
    if not snapshots:
        raise ValueError(f"no ELLE snapshots found in {directory!r} for pattern {pattern!r}")
    final_snapshot = snapshots[-1]
    comparison = compare_snapshot_summary_to_legacy_statistics(
        final_snapshot,
        legacy_statistics_path,
    ).to_dict()
    return {
        "final_snapshot_path": str(final_snapshot["path"]),
        "final_snapshot_step": final_snapshot.get("step"),
        "comparison": comparison,
    }


def build_benchmark_validation_report(
    *,
    reference_dir: str | Path,
    data_dir: str | Path,
    candidate_dir: str | Path | None = None,
    candidate_mesh_json_path: str | Path | None = None,
    candidate_legacy_statistics_path: str | Path | None = None,
    legacy_old_stats_path: str | Path | None = None,
    experiment_family_dirs: dict[str, str | Path] | None = None,
    experiment_family_reports: dict[str, str | Path | dict[str, Any]] | None = None,
    pattern: str = "*.elle",
) -> dict[str, Any]:
    report = {
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
    if (candidate_mesh_json_path is None) ^ (legacy_old_stats_path is None):
        raise ValueError(
            "candidate_mesh_json_path and legacy_old_stats_path must be provided together"
        )
    if candidate_legacy_statistics_path is not None and candidate_dir is None:
        raise ValueError(
            "candidate_dir is required when candidate_legacy_statistics_path is provided"
        )
    if candidate_mesh_json_path is not None and legacy_old_stats_path is not None:
        report["legacy_old_stats_bookkeeping"] = compare_mesh_bookkeeping_to_legacy_old_stats(
            candidate_mesh_json_path,
            legacy_old_stats_path,
        ).to_dict()
    if candidate_legacy_statistics_path is not None and candidate_dir is not None:
        report["legacy_final_statistics"] = evaluate_final_snapshot_against_legacy_statistics(
            candidate_dir,
            candidate_legacy_statistics_path,
            pattern=pattern,
        )
    if experiment_family_dirs or experiment_family_reports:
        report["experiment_family_benchmarks"] = evaluate_experiment_family_suite(
            experiment_family_dirs=experiment_family_dirs,
            experiment_family_reports=experiment_family_reports,
            pattern=pattern,
        )
    report["benchmark_acceptance"] = assess_benchmark_report_acceptance(report)
    return report


def write_benchmark_validation_report(path: str | Path, report: dict[str, Any]) -> Path:
    outpath = Path(path)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    outpath.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return outpath
