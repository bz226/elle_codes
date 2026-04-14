from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np

from .elle_visualize import (
    _parse_elle_sections,
    _parse_flynns,
    _parse_location,
    _parse_sparse_values,
)


def _parse_step_from_name(path: str | Path) -> int | None:
    match = re.search(r"(\d+)(?=\.elle$)", Path(path).name)
    if match is None:
        return None
    return int(match.group(1))


def _parse_cell_bounding_box(lines: tuple[str, ...] | list[str]) -> list[tuple[float, float]] | None:
    for index, raw_line in enumerate(lines):
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if not parts or parts[0] != "CellBoundingBox" or len(parts) < 3:
            continue
        corners: list[tuple[float, float]] = [(float(parts[1]), float(parts[2]))]
        for extra_line in lines[index + 1 : index + 4]:
            extra_parts = extra_line.strip().split()
            if len(extra_parts) >= 2:
                corners.append((float(extra_parts[0]), float(extra_parts[1])))
        if len(corners) >= 4:
            return corners[:4]
    return None


def _make_cell_transform(
    cell_bbox: list[tuple[float, float]] | None,
) -> dict[str, Any]:
    if cell_bbox is not None and len(cell_bbox) >= 4:
        origin = cell_bbox[0]
        axis_u = (cell_bbox[1][0] - origin[0], cell_bbox[1][1] - origin[1])
        axis_v = (cell_bbox[3][0] - origin[0], cell_bbox[3][1] - origin[1])
        determinant = axis_u[0] * axis_v[1] - axis_u[1] * axis_v[0]
        if abs(determinant) > 1.0e-12:
            return {
                "origin": origin,
                "axis_u": axis_u,
                "axis_v": axis_v,
                "determinant": determinant,
                "periodic": True,
                "cell_area": abs(determinant),
            }

    return {
        "origin": (0.0, 0.0),
        "axis_u": (1.0, 0.0),
        "axis_v": (0.0, 1.0),
        "determinant": 1.0,
        "periodic": False,
        "cell_area": 1.0,
    }


def _point_to_cell(point: tuple[float, float], transform: dict[str, Any]) -> tuple[float, float]:
    dx = float(point[0]) - float(transform["origin"][0])
    dy = float(point[1]) - float(transform["origin"][1])
    determinant = float(transform["determinant"])
    axis_u = transform["axis_u"]
    axis_v = transform["axis_v"]
    u_coord = (dx * float(axis_v[1]) - dy * float(axis_v[0])) / determinant
    v_coord = (-dx * float(axis_u[1]) + dy * float(axis_u[0])) / determinant
    return (u_coord, v_coord)


def _cell_to_point(coords: tuple[float, float], transform: dict[str, Any]) -> tuple[float, float]:
    u_coord, v_coord = coords
    axis_u = transform["axis_u"]
    axis_v = transform["axis_v"]
    origin = transform["origin"]
    return (
        float(origin[0]) + float(u_coord) * float(axis_u[0]) + float(v_coord) * float(axis_v[0]),
        float(origin[1]) + float(u_coord) * float(axis_u[1]) + float(v_coord) * float(axis_v[1]),
    )


def _unwrap_cell_points(
    points: list[tuple[float, float]],
    *,
    periodic: bool,
) -> list[tuple[float, float]]:
    if not points:
        return []
    unwrapped = [points[0]]
    for point in points[1:]:
        prev_u, prev_v = unwrapped[-1]
        delta_u = float(point[0]) - prev_u
        delta_v = float(point[1]) - prev_v
        if periodic:
            delta_u -= round(delta_u)
            delta_v -= round(delta_v)
        unwrapped.append((prev_u + delta_u, prev_v + delta_v))
    return unwrapped


def _polygon_area(points: list[tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    area = 0.0
    for index, (x0, y0) in enumerate(points):
        x1, y1 = points[(index + 1) % len(points)]
        area += x0 * y1 - x1 * y0
    return 0.5 * area


def _polygon_perimeter(points: list[tuple[float, float]]) -> float:
    if len(points) < 2:
        return 0.0
    perimeter = 0.0
    for index, point_a in enumerate(points):
        point_b = points[(index + 1) % len(points)]
        perimeter += float(np.hypot(point_b[0] - point_a[0], point_b[1] - point_a[1]))
    return perimeter


def _flynn_polygon_metrics(
    node_points: list[tuple[float, float]],
    transform: dict[str, Any],
) -> tuple[float, float]:
    point_uv = [_point_to_cell(point, transform) for point in node_points]
    unwrapped_uv = _unwrap_cell_points(point_uv, periodic=bool(transform["periodic"]))
    unwrapped_xy = [_cell_to_point(point, transform) for point in unwrapped_uv]
    area = abs(_polygon_area(unwrapped_xy))
    perimeter = _polygon_perimeter(unwrapped_xy)
    return area, perimeter


def _weighted_axial_statistics(angles_deg: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    if angles_deg.size == 0 or weights.size == 0 or float(weights.sum()) <= 0.0:
        return {"mean_deg": float("nan"), "resultant_length": float("nan"), "dispersion": float("nan")}
    radians = np.deg2rad(angles_deg.astype(np.float64) * 2.0)
    weight_sum = float(weights.sum())
    mean_cos = float(np.sum(weights * np.cos(radians)) / weight_sum)
    mean_sin = float(np.sum(weights * np.sin(radians)) / weight_sum)
    resultant = float(np.hypot(mean_cos, mean_sin))
    mean_angle = 0.5 * math.degrees(math.atan2(mean_sin, mean_cos))
    if mean_angle < 0.0:
        mean_angle += 180.0
    return {
        "mean_deg": float(mean_angle),
        "resultant_length": resultant,
        "dispersion": float(1.0 - resultant),
    }


def summarize_elle_microstructure(
    elle_path: str | Path,
    *,
    orientation_section: str = "EULER_3",
    orientation_bins: int = 18,
) -> dict[str, Any]:
    path = Path(elle_path)
    sections = _parse_elle_sections(path)
    flynns = _parse_flynns(sections.get("FLYNNS", ()))
    nodes = _parse_location(sections.get("LOCATION", ()))
    cell_bbox = _parse_cell_bounding_box(sections.get("OPTIONS", ()))
    transform = _make_cell_transform(cell_bbox)

    areas: list[float] = []
    perimeters: list[float] = []
    eq_radii: list[float] = []
    shape_factors: list[float] = []
    kept_flynn_ids: list[int] = []

    for flynn in flynns:
        node_points = [nodes[node_id] for node_id in flynn["node_ids"] if node_id in nodes]
        if len(node_points) < 3:
            continue
        area, perimeter = _flynn_polygon_metrics(node_points, transform)
        if area <= 0.0:
            continue
        areas.append(float(area))
        perimeters.append(float(perimeter))
        eq_radii.append(float(math.sqrt(area / math.pi)))
        if perimeter > 0.0:
            shape_factors.append(float(4.0 * math.pi * area / (perimeter * perimeter)))
        kept_flynn_ids.append(int(flynn["flynn_id"]))

    area_array = np.asarray(areas, dtype=np.float64)
    perimeter_array = np.asarray(perimeters, dtype=np.float64)
    radius_array = np.asarray(eq_radii, dtype=np.float64)
    shape_array = np.asarray(shape_factors, dtype=np.float64)

    summary: dict[str, Any] = {
        "path": str(path),
        "step": _parse_step_from_name(path),
        "grain_count": int(len(area_array)),
        "cell_area": float(transform["cell_area"]),
        "area_fraction_sum": float(area_array.sum() / max(float(transform["cell_area"]), 1.0e-12)) if area_array.size else 0.0,
        "mean_grain_area": float(area_array.mean()) if area_array.size else 0.0,
        "median_grain_area": float(np.median(area_array)) if area_array.size else 0.0,
        "std_grain_area": float(area_array.std()) if area_array.size else 0.0,
        "mean_equivalent_radius": float(radius_array.mean()) if radius_array.size else 0.0,
        "median_equivalent_radius": float(np.median(radius_array)) if radius_array.size else 0.0,
        "mean_perimeter": float(perimeter_array.mean()) if perimeter_array.size else 0.0,
        "mean_shape_factor": float(shape_array.mean()) if shape_array.size else 0.0,
    }

    if orientation_section in sections and flynns and area_array.size:
        _, orientation_map = _parse_sparse_values(sections[orientation_section])
        flynn_ids = np.array(kept_flynn_ids, dtype=np.int32)
        valid_mask = np.array([flynn_id in orientation_map for flynn_id in flynn_ids], dtype=bool)
        if np.any(valid_mask):
            orientation_values = np.array(
                [float(orientation_map[int(flynn_id)]) for flynn_id in flynn_ids[valid_mask]],
                dtype=np.float64,
            )
            orientation_weights = area_array[valid_mask]
            stats = _weighted_axial_statistics(orientation_values, orientation_weights)
            hist, bin_edges = np.histogram(
                orientation_values % 180.0,
                bins=int(orientation_bins),
                range=(0.0, 180.0),
                weights=orientation_weights,
                density=False,
            )
            hist = hist.astype(np.float64)
            if hist.sum() > 0.0:
                hist /= hist.sum()
            summary["orientation"] = {
                "section": orientation_section,
                "axial_mean_deg": stats["mean_deg"],
                "resultant_length": stats["resultant_length"],
                "dispersion": stats["dispersion"],
                "histogram": hist.tolist(),
                "bin_edges_deg": bin_edges.tolist(),
            }

    return summary


def collect_elle_microstructure_snapshots(
    directory: str | Path,
    *,
    pattern: str = "*.elle",
) -> list[dict[str, Any]]:
    directory_path = Path(directory)
    snapshots = [summarize_elle_microstructure(path) for path in sorted(directory_path.glob(pattern))]
    return sorted(
        snapshots,
        key=lambda snapshot: (
            snapshot["step"] is None,
            snapshot["step"] if snapshot["step"] is not None else snapshot["path"],
        ),
    )


def compare_elle_microstructure_sequences(
    reference_dir: str | Path,
    candidate_dir: str | Path,
    *,
    pattern: str = "*.elle",
) -> dict[str, Any]:
    reference = collect_elle_microstructure_snapshots(reference_dir, pattern=pattern)
    candidate = collect_elle_microstructure_snapshots(candidate_dir, pattern=pattern)
    reference_by_step = {snapshot["step"]: snapshot for snapshot in reference if snapshot["step"] is not None}
    candidate_by_step = {snapshot["step"]: snapshot for snapshot in candidate if snapshot["step"] is not None}

    matched_steps = sorted(set(reference_by_step) & set(candidate_by_step))
    per_step: list[dict[str, Any]] = []

    for step in matched_steps:
        ref = reference_by_step[step]
        cand = candidate_by_step[step]
        step_report = {
            "step": int(step),
            "grain_count_abs_diff": abs(int(cand["grain_count"]) - int(ref["grain_count"])),
            "mean_grain_area_abs_diff": abs(float(cand["mean_grain_area"]) - float(ref["mean_grain_area"])),
            "mean_equivalent_radius_abs_diff": abs(
                float(cand["mean_equivalent_radius"]) - float(ref["mean_equivalent_radius"])
            ),
            "mean_shape_factor_abs_diff": abs(float(cand["mean_shape_factor"]) - float(ref["mean_shape_factor"])),
        }
        ref_orientation = ref.get("orientation")
        cand_orientation = cand.get("orientation")
        if ref_orientation is not None and cand_orientation is not None:
            ref_hist = np.asarray(ref_orientation["histogram"], dtype=np.float64)
            cand_hist = np.asarray(cand_orientation["histogram"], dtype=np.float64)
            if ref_hist.shape == cand_hist.shape:
                step_report["orientation_hist_l1"] = float(np.abs(ref_hist - cand_hist).sum())
            step_report["orientation_mean_abs_diff_deg"] = abs(
                float(cand_orientation["axial_mean_deg"]) - float(ref_orientation["axial_mean_deg"])
            )
        per_step.append(step_report)

    summary: dict[str, Any] = {
        "num_reference": int(len(reference)),
        "num_candidate": int(len(candidate)),
        "num_matched_steps": int(len(matched_steps)),
        "matched_steps": [int(step) for step in matched_steps],
    }

    if per_step:
        summary["grain_count_abs_diff_mean"] = float(
            np.mean([entry["grain_count_abs_diff"] for entry in per_step], dtype=np.float64)
        )
        summary["mean_grain_area_abs_diff_mean"] = float(
            np.mean([entry["mean_grain_area_abs_diff"] for entry in per_step], dtype=np.float64)
        )
        summary["mean_equivalent_radius_abs_diff_mean"] = float(
            np.mean([entry["mean_equivalent_radius_abs_diff"] for entry in per_step], dtype=np.float64)
        )
        summary["mean_shape_factor_abs_diff_mean"] = float(
            np.mean([entry["mean_shape_factor_abs_diff"] for entry in per_step], dtype=np.float64)
        )
        orientation_l1 = [entry["orientation_hist_l1"] for entry in per_step if "orientation_hist_l1" in entry]
        if orientation_l1:
            summary["orientation_hist_l1_mean"] = float(np.mean(orientation_l1, dtype=np.float64))
        orientation_mean_diff = [
            entry["orientation_mean_abs_diff_deg"]
            for entry in per_step
            if "orientation_mean_abs_diff_deg" in entry
        ]
        if orientation_mean_diff:
            summary["orientation_mean_abs_diff_deg_mean"] = float(np.mean(orientation_mean_diff, dtype=np.float64))

    return {
        "reference_dir": str(Path(reference_dir)),
        "candidate_dir": str(Path(candidate_dir)),
        "reference": reference,
        "candidate": candidate,
        "per_step": per_step,
        "summary": summary,
    }


def _parse_lsu_filename(path: str | Path) -> dict[str, Any]:
    name = Path(path).name
    match = re.match(
        r"(?P<variable>grain_kde|euler_[123])_S(?P<strain>[^_]+)_H(?P<pressure>[^_]+)_T(?P<temperature>[^_]+)_data_train\.npy\.npz$",
        name,
    )
    if match is None:
        return {"name": name}
    result = match.groupdict()
    result["name"] = name
    return result


def _normalize_columns(values: np.ndarray) -> np.ndarray:
    totals = np.sum(values, axis=0, keepdims=True)
    totals = np.where(np.abs(totals) < 1.0e-12, 1.0, totals)
    return values / totals


def summarize_liu_suckale_datasets(data_dir: str | Path) -> dict[str, Any]:
    data_path = Path(data_dir)
    datasets: list[dict[str, Any]] = []

    for path in sorted(data_path.glob("*.npz")):
        meta = _parse_lsu_filename(path)
        data = np.load(path)
        key = list(data.keys())[0]
        array = np.asarray(data[key])
        record: dict[str, Any] = {
            "path": str(path),
            "key": key,
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            **meta,
        }

        if meta.get("variable") == "grain_kde":
            kde = np.asarray(array[0], dtype=np.float64)
            kde_norm = _normalize_columns(kde)
            index = np.arange(kde.shape[0], dtype=np.float64)
            mean_index = np.sum(kde_norm * index[:, None], axis=0)
            peak_index = np.argmax(kde, axis=0)
            entropy_terms = np.zeros_like(kde_norm)
            positive_mask = kde_norm > 0.0
            entropy_terms[positive_mask] = kde_norm[positive_mask] * np.log(kde_norm[positive_mask])
            entropy = -np.sum(entropy_terms, axis=0)
            record["summary"] = {
                "time_steps": int(kde.shape[1]),
                "mean_index_start": float(mean_index[0]),
                "mean_index_end": float(mean_index[-1]),
                "peak_index_start": int(peak_index[0]),
                "peak_index_end": int(peak_index[-1]),
                "entropy_start": float(entropy[0]),
                "entropy_end": float(entropy[-1]),
            }
        elif meta.get("variable", "").startswith("euler_"):
            angles = np.asarray(array[0], dtype=np.float64)
            flat = angles.reshape(-1, angles.shape[-1])
            weights = np.ones(flat.shape[0], dtype=np.float64)
            axial_means = []
            resultant_lengths = []
            for index in range(flat.shape[1]):
                stats = _weighted_axial_statistics(flat[:, index], weights)
                axial_means.append(stats["mean_deg"])
                resultant_lengths.append(stats["resultant_length"])
            record["summary"] = {
                "time_steps": int(angles.shape[-1]),
                "axial_mean_start_deg": float(axial_means[0]),
                "axial_mean_end_deg": float(axial_means[-1]),
                "resultant_length_start": float(resultant_lengths[0]),
                "resultant_length_end": float(resultant_lengths[-1]),
                "value_min": float(angles.min()),
                "value_max": float(angles.max()),
            }

        datasets.append(record)

    return {
        "data_dir": str(data_path),
        "num_datasets": int(len(datasets)),
        "datasets": datasets,
    }


def write_microstructure_validation_report(path: str | Path, report: dict[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return output
