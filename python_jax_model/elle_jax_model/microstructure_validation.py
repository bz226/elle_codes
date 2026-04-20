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
    _parse_unodes,
)
from .legacy_statistics import load_legacy_allout_statistics
from .mesh import _collect_numeric_section_specs
from .simulation import load_elle_label_seed


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


def _unwrap_flynn_polygon_points(
    node_points: list[tuple[float, float]],
    transform: dict[str, Any],
) -> list[tuple[float, float]]:
    point_uv = [_point_to_cell(point, transform) for point in node_points]
    unwrapped_uv = _unwrap_cell_points(point_uv, periodic=bool(transform["periodic"]))
    return [_cell_to_point(point, transform) for point in unwrapped_uv]


def _flynn_polygon_metrics(
    node_points: list[tuple[float, float]],
    transform: dict[str, Any],
) -> tuple[float, float]:
    unwrapped_xy = _unwrap_flynn_polygon_points(node_points, transform)
    area = abs(_polygon_area(unwrapped_xy))
    perimeter = _polygon_perimeter(unwrapped_xy)
    return area, perimeter


def _polygon_vertex_aspect_ratio(points: list[tuple[float, float]]) -> float:
    if len(points) < 3:
        return 1.0
    coords = np.asarray(points, dtype=np.float64)
    centered = coords - np.mean(coords, axis=0, keepdims=True)
    covariance = np.cov(centered.T, bias=True)
    eigenvalues = np.sort(np.asarray(np.linalg.eigvalsh(covariance), dtype=np.float64))
    minor = max(float(eigenvalues[0]), 1.0e-12)
    major = max(float(eigenvalues[-1]), minor)
    return float(math.sqrt(major / minor))


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


def _bunge_rotation_matrix_deg(phi1_deg: float, Phi_deg: float, phi2_deg: float) -> np.ndarray:
    phi1 = math.radians(float(phi1_deg))
    Phi = math.radians(float(Phi_deg))
    phi2 = math.radians(float(phi2_deg))
    c1, s1 = math.cos(phi1), math.sin(phi1)
    c2, s2 = math.cos(Phi), math.sin(Phi)
    c3, s3 = math.cos(phi2), math.sin(phi2)
    return np.asarray(
        [
            [c1 * c3 - s1 * c2 * s3, -c1 * s3 - s1 * c2 * c3, s2 * c1],
            [s1 * c3 + c1 * c2 * s3, -s1 * s3 + c1 * c2 * c3, s1 * s2],
            [s2 * s3, s2 * c3, c2],
        ],
        dtype=np.float64,
    )


def _normalize_axis_sign(vector: np.ndarray) -> np.ndarray:
    normalized = np.asarray(vector, dtype=np.float64).copy()
    for component in normalized[::-1]:
        if abs(float(component)) <= 1.0e-12:
            continue
        if float(component) < 0.0:
            normalized *= -1.0
        break
    return normalized


def _upper_hemisphere_vectors(vectors: np.ndarray) -> np.ndarray:
    mapped = np.asarray(vectors, dtype=np.float64).copy()
    for index in range(mapped.shape[0]):
        vector = mapped[index]
        if float(vector[2]) < -1.0e-12:
            mapped[index] = -vector
            continue
        if abs(float(vector[2])) > 1.0e-12:
            continue
        if float(vector[1]) < -1.0e-12 or (
            abs(float(vector[1])) <= 1.0e-12 and float(vector[0]) < -1.0e-12
        ):
            mapped[index] = -vector
    return mapped


def _pole_figure_summary(vectors: np.ndarray, weights: np.ndarray) -> dict[str, Any] | None:
    if vectors.ndim != 2 or vectors.shape[1] != 3 or weights.shape != (vectors.shape[0],):
        return None
    if vectors.shape[0] == 0:
        return None

    safe_vectors = np.asarray(vectors, dtype=np.float64)
    safe_weights = np.asarray(weights, dtype=np.float64)
    norms = np.linalg.norm(safe_vectors, axis=1)
    valid_mask = np.asarray(norms > 1.0e-12, dtype=bool)
    if not np.any(valid_mask):
        return None
    safe_vectors = safe_vectors[valid_mask] / norms[valid_mask][:, None]
    safe_weights = safe_weights[valid_mask]
    weight_sum = float(np.sum(safe_weights, dtype=np.float64))
    if weight_sum <= 0.0:
        return None

    pole_vectors = _upper_hemisphere_vectors(safe_vectors)
    colatitude_deg = np.degrees(np.arccos(np.clip(pole_vectors[:, 2], -1.0, 1.0)))
    azimuth_deg = np.degrees(np.arctan2(pole_vectors[:, 1], pole_vectors[:, 0])) % 360.0

    colatitude_edges = np.linspace(0.0, 90.0, 10, dtype=np.float64)
    azimuth_edges = np.linspace(0.0, 360.0, 19, dtype=np.float64)
    colatitude_clipped = np.clip(colatitude_deg, 0.0, np.nextafter(90.0, 0.0))
    azimuth_clipped = np.clip(azimuth_deg, 0.0, np.nextafter(360.0, 0.0))
    histogram, _, _ = np.histogram2d(
        colatitude_clipped,
        azimuth_clipped,
        bins=(colatitude_edges, azimuth_edges),
        weights=safe_weights,
        density=False,
    )
    histogram = np.asarray(histogram, dtype=np.float64)
    if float(histogram.sum()) > 0.0:
        histogram /= float(histogram.sum())

    peak_index = np.unravel_index(int(np.argmax(histogram)), histogram.shape)
    peak_colatitude_deg = float(0.5 * (colatitude_edges[peak_index[0]] + colatitude_edges[peak_index[0] + 1]))
    peak_azimuth_deg = float(0.5 * (azimuth_edges[peak_index[1]] + azimuth_edges[peak_index[1] + 1]))

    return {
        "histogram": histogram.tolist(),
        "colatitude_bin_edges_deg": colatitude_edges.tolist(),
        "azimuth_bin_edges_deg": azimuth_edges.tolist(),
        "mean_colatitude_deg": float(np.sum(safe_weights * colatitude_deg, dtype=np.float64) / weight_sum),
        "fraction_within_15deg": float(
            np.sum(safe_weights[colatitude_deg <= 15.0], dtype=np.float64) / weight_sum
        ),
        "peak_bin_fraction": float(histogram[peak_index]),
        "peak_colatitude_deg": peak_colatitude_deg,
        "peak_azimuth_deg": peak_azimuth_deg,
    }


def _axis_tensor_summary(vectors: np.ndarray, weights: np.ndarray) -> dict[str, Any] | None:
    if vectors.ndim != 2 or vectors.shape[1] != 3 or weights.shape != (vectors.shape[0],):
        return None
    if vectors.shape[0] == 0:
        return None

    safe_vectors = np.asarray(vectors, dtype=np.float64)
    safe_weights = np.asarray(weights, dtype=np.float64)
    norms = np.linalg.norm(safe_vectors, axis=1)
    valid_mask = np.asarray(norms > 1.0e-12, dtype=bool)
    if not np.any(valid_mask):
        return None
    safe_vectors = safe_vectors[valid_mask] / norms[valid_mask][:, None]
    safe_weights = safe_weights[valid_mask]
    weight_sum = float(np.sum(safe_weights, dtype=np.float64))
    if weight_sum <= 0.0:
        return None

    tensor = np.einsum("n,ni,nj->ij", safe_weights, safe_vectors, safe_vectors, dtype=np.float64)
    tensor /= weight_sum
    eigenvalues, eigenvectors = np.linalg.eigh(tensor)
    descending_order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.asarray(eigenvalues[descending_order], dtype=np.float64)
    eigenvectors = np.asarray(eigenvectors[:, descending_order], dtype=np.float64)
    principal_direction = _normalize_axis_sign(eigenvectors[:, 0])
    principal_z = float(np.clip(principal_direction[2], -1.0, 1.0))
    lambda1, lambda2, lambda3 = (float(value) for value in eigenvalues)

    woodcock_k = float("nan")
    ascending = np.sort(np.asarray(eigenvalues, dtype=np.float64))
    if (
        float(ascending[0]) > 1.0e-12
        and float(ascending[1]) > float(ascending[0]) + 1.0e-12
        and float(ascending[2]) > float(ascending[1]) + 1.0e-12
    ):
        woodcock_k = float(
            math.sqrt(
                math.log(float(ascending[2]) / float(ascending[1]))
                / math.log(float(ascending[1]) / float(ascending[0]))
            )
        )

    pole_figure = _pole_figure_summary(safe_vectors, safe_weights)
    summary = {
        "eigenvalues": [lambda1, lambda2, lambda3],
        "tensor": np.asarray(tensor, dtype=np.float64).tolist(),
        "principal_direction": [float(value) for value in principal_direction.tolist()],
        "principal_colatitude_deg": float(math.degrees(math.acos(principal_z))),
        "principal_azimuth_deg": float(
            math.degrees(math.atan2(principal_direction[1], principal_direction[0])) % 360.0
        ),
        "P_index": float(lambda1 - lambda2),
        "G_index": float(2.0 * (lambda2 - lambda3)),
        "R_index": float(3.0 * lambda3),
        "woodcock_k": woodcock_k,
    }
    if pole_figure is not None:
        summary["pole_figure"] = pole_figure
    return summary


def _orientation_vectors_from_euler_rows(euler_rows: np.ndarray) -> dict[str, np.ndarray] | None:
    rows = np.asarray(euler_rows, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[0] == 0 or rows.shape[1] < 3:
        return None

    c_axes: list[np.ndarray] = []
    a_axes: list[np.ndarray] = []
    prism_normals: list[np.ndarray] = []
    for phi1_deg, Phi_deg, phi2_deg, *_ in rows:
        rotation = _bunge_rotation_matrix_deg(phi1_deg, Phi_deg, phi2_deg)
        a_axes.append(np.asarray(rotation[:, 0], dtype=np.float64))
        prism_normals.append(np.asarray(rotation[:, 1], dtype=np.float64))
        c_axes.append(np.asarray(rotation[:, 2], dtype=np.float64))
    return {
        "c_axis": np.asarray(c_axes, dtype=np.float64),
        "a_axis": np.asarray(a_axes, dtype=np.float64),
        "prism_normal": np.asarray(prism_normals, dtype=np.float64),
    }


def _source_label_area_fractions(seed: dict[str, object]) -> dict[int, float]:
    label_field = np.asarray(seed["label_field"], dtype=np.int32)
    source_labels = [int(value) for value in seed["source_labels"]]
    counts = np.bincount(label_field.ravel(), minlength=len(source_labels)).astype(np.float64)
    total = float(label_field.size)
    return {
        int(source_labels[index]): float(counts[index] / total)
        for index in range(len(source_labels))
        if float(counts[index]) > 0.0
    }


def _basal_schmid_factor_from_euler_row(
    euler_row: np.ndarray,
    *,
    load_axis: tuple[float, float, float] = (0.0, 1.0, 0.0),
) -> float:
    phi1_deg, Phi_deg, phi2_deg = (float(value) for value in np.asarray(euler_row, dtype=np.float64)[:3])
    rotation = _bunge_rotation_matrix_deg(phi1_deg, Phi_deg, phi2_deg)
    load = np.asarray(load_axis, dtype=np.float64)
    load_norm = float(np.linalg.norm(load))
    if load_norm <= 1.0e-12:
        return float("nan")
    load /= load_norm
    plane_normal = np.asarray(rotation @ np.asarray([0.0, 0.0, 1.0], dtype=np.float64), dtype=np.float64)
    basal_slip_directions = (
        np.asarray([1.0, 0.0, 0.0], dtype=np.float64),
        np.asarray([-0.5, math.sqrt(3.0) * 0.5, 0.0], dtype=np.float64),
        np.asarray([-0.5, -math.sqrt(3.0) * 0.5, 0.0], dtype=np.float64),
    )
    cos_phi = float(np.dot(plane_normal, load))
    schmid_values = []
    for slip_direction_crystal in basal_slip_directions:
        slip_direction = np.asarray(rotation @ slip_direction_crystal, dtype=np.float64)
        direction_norm = float(np.linalg.norm(slip_direction))
        if direction_norm <= 1.0e-12:
            continue
        slip_direction /= direction_norm
        cos_lambda = float(np.dot(slip_direction, load))
        schmid_values.append(abs(cos_phi * cos_lambda))
    if not schmid_values:
        return float("nan")
    return float(max(schmid_values))


def _initial_source_label_schmid_factors(
    elle_path: str | Path,
    *,
    attribute: str = "auto",
) -> dict[int, float]:
    seed = load_elle_label_seed(elle_path, attribute=attribute)
    sections = _parse_elle_sections(elle_path)
    unodes = _parse_unodes(sections.get("UNODES", ()))
    if not unodes:
        return {}
    unode_ids = [int(unode_id) for unode_id, _, _ in unodes]
    unode_sections = _collect_numeric_section_specs(sections, ["U_EULER_3"], unode_ids)
    component_counts = dict(unode_sections.get("component_counts", {}))
    if int(component_counts.get("U_EULER_3", 0)) < 3:
        return {}
    euler_rows = np.asarray(dict(unode_sections.get("values", {}))["U_EULER_3"], dtype=np.float64)
    label_field = np.asarray(seed["label_field"], dtype=np.int32)
    source_labels = [int(value) for value in seed["source_labels"]]
    sample_source_labels = [
        int(source_labels[int(label_field[int(ix), int(iy)])])
        for ix, iy in seed["unode_grid_indices"]
    ]
    schmid_by_label: dict[int, list[float]] = {}
    for source_label, euler_row in zip(sample_source_labels, euler_rows, strict=False):
        schmid_by_label.setdefault(int(source_label), []).append(_basal_schmid_factor_from_euler_row(euler_row))
    return {
        int(source_label): float(np.mean(values, dtype=np.float64))
        for source_label, values in schmid_by_label.items()
        if values
    }


def _safe_pearson(x_values: np.ndarray, y_values: np.ndarray) -> float:
    x_array = np.asarray(x_values, dtype=np.float64)
    y_array = np.asarray(y_values, dtype=np.float64)
    if x_array.size < 2 or y_array.size != x_array.size:
        return float("nan")
    if float(np.std(x_array, dtype=np.float64)) <= 1.0e-12:
        return float("nan")
    if float(np.std(y_array, dtype=np.float64)) <= 1.0e-12:
        return float("nan")
    return float(np.corrcoef(x_array, y_array)[0, 1])


def _fabric_summary_from_orientations(
    *,
    source_name: str,
    euler_rows: np.ndarray,
    weights: np.ndarray,
) -> dict[str, Any] | None:
    axis_vectors = _orientation_vectors_from_euler_rows(euler_rows)
    if axis_vectors is None:
        return None

    c_axis_summary = _axis_tensor_summary(axis_vectors["c_axis"], np.asarray(weights, dtype=np.float64))
    a_axis_summary = _axis_tensor_summary(axis_vectors["a_axis"], np.asarray(weights, dtype=np.float64))
    prism_normal_summary = _axis_tensor_summary(
        axis_vectors["prism_normal"],
        np.asarray(weights, dtype=np.float64),
    )
    if c_axis_summary is None or a_axis_summary is None or prism_normal_summary is None:
        return None

    return {
        "source": str(source_name),
        "num_orientations": int(np.asarray(euler_rows).shape[0]),
        "c_axis": c_axis_summary,
        "a_axis": a_axis_summary,
        "prism_normal": prism_normal_summary,
    }


def _mechanics_summary_from_unode_fields(
    sections: dict[str, tuple[str, ...]],
    unode_ids: list[int],
) -> dict[str, Any] | None:
    mechanics_sections = ["U_ATTRIB_A", "U_ATTRIB_B", "U_ATTRIB_D", "U_ATTRIB_E"]
    unode_specs = _collect_numeric_section_specs(sections, mechanics_sections, unode_ids)
    available_fields = tuple(str(name) for name in unode_specs.get("field_order", ()))
    if tuple(mechanics_sections) != available_fields:
        return None

    component_counts = dict(unode_specs.get("component_counts", {}))
    if any(int(component_counts.get(section_name, 0)) != 1 for section_name in mechanics_sections):
        return None

    values = dict(unode_specs.get("values", {}))
    strain_rate = np.asarray(values["U_ATTRIB_A"], dtype=np.float64).reshape(-1)
    stress = np.asarray(values["U_ATTRIB_B"], dtype=np.float64).reshape(-1)
    basal = np.asarray(values["U_ATTRIB_D"], dtype=np.float64).reshape(-1)
    prismatic = np.asarray(values["U_ATTRIB_E"], dtype=np.float64).reshape(-1)
    total_activity = basal + prismatic
    prismatic_fraction = np.divide(
        prismatic,
        total_activity,
        out=np.zeros_like(prismatic, dtype=np.float64),
        where=np.abs(total_activity) > 1.0e-12,
    )
    mean_basal = float(np.mean(basal, dtype=np.float64))
    mean_prismatic = float(np.mean(prismatic, dtype=np.float64))
    ratio = float("nan")
    if abs(mean_basal) > 1.0e-12:
        ratio = float(mean_prismatic / mean_basal)

    return {
        "source_sections": mechanics_sections,
        "num_unodes": int(len(unode_ids)),
        "mean_normalized_strain_rate": float(np.mean(strain_rate, dtype=np.float64)),
        "mean_normalized_stress": float(np.mean(stress, dtype=np.float64)),
        "mean_basal_activity": mean_basal,
        "mean_prismatic_activity": mean_prismatic,
        "mean_total_activity": float(np.mean(total_activity, dtype=np.float64)),
        "mean_prismatic_fraction": float(np.mean(prismatic_fraction, dtype=np.float64)),
        "prismatic_to_basal_ratio": ratio,
    }


def _adjacent_mesh_artifact_path(elle_path: str | Path) -> Path | None:
    path = Path(elle_path)
    match = re.search(r"(\d+)(?=\.elle$)", path.name)
    if match is None:
        return None
    return path.with_name(f"mesh_{match.group(1)}.json")


def _mechanics_payload_summary_from_mesh_sidecar(elle_path: str | Path) -> dict[str, Any] | None:
    mesh_path = _adjacent_mesh_artifact_path(elle_path)
    if mesh_path is None or not mesh_path.exists():
        return None
    try:
        mesh_payload = json.loads(mesh_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    payload_summary = mesh_payload.get("mechanics_payload_summary")
    if isinstance(payload_summary, dict):
        return dict(payload_summary)

    stats = mesh_payload.get("stats")
    if not isinstance(stats, dict):
        return None
    if "mechanics_snapshot_cumulative_simple_shear" not in stats:
        return None
    direct_strain_axis = stats.get("mechanics_snapshot_direct_strain_axis")
    strain_axis_source = stats.get("mechanics_snapshot_strain_axis_source")
    if direct_strain_axis is None:
        direct_strain_axis = float(stats["mechanics_snapshot_cumulative_simple_shear"])
    if strain_axis_source is None:
        strain_axis_source = "cumulative_simple_shear"
    return {
        "source": "mesh_stats",
        "cumulative_simple_shear": float(stats["mechanics_snapshot_cumulative_simple_shear"]),
        "simple_shear_increment": float(stats.get("mechanics_snapshot_simple_shear_increment", 0.0)),
        "simple_shear_offset": float(stats.get("mechanics_snapshot_simple_shear_offset", 0.0)),
        "direct_strain_axis": float(direct_strain_axis),
        "strain_axis_source": str(strain_axis_source),
    }


def _load_legacy_allout_rows(directory: str | Path) -> list[dict[str, Any]] | None:
    allout_path = Path(directory) / "AllOutData.txt"
    if not allout_path.exists():
        return None
    rows = [row.to_dict() for row in load_legacy_allout_statistics(allout_path)]
    return rows or None


def _overlay_legacy_allout_rows(
    snapshots: list[dict[str, Any]],
    *,
    directory: str | Path,
) -> list[dict[str, Any]]:
    rows = _load_legacy_allout_rows(directory)
    if rows is None or not snapshots:
        return snapshots

    offset = 0
    if len(rows) == len(snapshots):
        offset = 0
    elif len(rows) == len(snapshots) - 1 and snapshots[0].get("step") == 0:
        offset = 1
    else:
        return snapshots

    updated: list[dict[str, Any]] = []
    for index, snapshot in enumerate(snapshots):
        row_index = index - offset
        if row_index < 0 or row_index >= len(rows):
            updated.append(snapshot)
            continue

        row = rows[row_index]
        updated_snapshot = dict(snapshot)
        mechanics_summary = dict(updated_snapshot.get("mechanics", {}))
        mechanics_summary.update(
            {
                "statistics_source": "AllOutData.txt",
                "statistics_row_index": int(row_index),
                "mean_von_mises_stress": float(row["mean_von_mises_stress"]),
                "mean_von_mises_strain_rate": float(row["mean_von_mises_strain_rate"]),
                "mean_differential_stress": float(row["mean_differential_stress"]),
                "stress_field_error": float(row["stress_field_error"]),
                "strain_rate_field_error": float(row["strain_rate_field_error"]),
                "mean_basal_activity": float(row["mean_basal_activity"]),
                "mean_prismatic_activity": float(row["mean_prismatic_activity"]),
                "mean_pyramidal_activity": float(row["mean_pyramidal_activity"]),
                "mean_total_activity": float(row["mean_total_activity"]),
                "mean_prismatic_fraction": float(row["mean_prismatic_fraction"]),
                "prismatic_to_basal_ratio": float(row["prismatic_to_basal_ratio"]),
                "stress_tensor": list(row["stress_tensor"]),
                "strain_rate_tensor": list(row["strain_rate_tensor"]),
            }
        )
        updated_snapshot["mechanics"] = mechanics_summary
        updated_snapshot["mechanics_statistics"] = dict(row)
        updated.append(updated_snapshot)

    return updated


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
    aspect_ratios: list[float] = []
    kept_flynn_ids: list[int] = []

    for flynn in flynns:
        node_points = [nodes[node_id] for node_id in flynn["node_ids"] if node_id in nodes]
        if len(node_points) < 3:
            continue
        area, perimeter = _flynn_polygon_metrics(node_points, transform)
        if area <= 0.0:
            continue
        unwrapped_xy = _unwrap_flynn_polygon_points(node_points, transform)
        areas.append(float(area))
        perimeters.append(float(perimeter))
        eq_radii.append(float(math.sqrt(area / math.pi)))
        aspect_ratios.append(_polygon_vertex_aspect_ratio(unwrapped_xy))
        if perimeter > 0.0:
            shape_factors.append(float(4.0 * math.pi * area / (perimeter * perimeter)))
        kept_flynn_ids.append(int(flynn["flynn_id"]))

    area_array = np.asarray(areas, dtype=np.float64)
    perimeter_array = np.asarray(perimeters, dtype=np.float64)
    radius_array = np.asarray(eq_radii, dtype=np.float64)
    shape_array = np.asarray(shape_factors, dtype=np.float64)
    aspect_ratio_array = np.asarray(aspect_ratios, dtype=np.float64)
    aspect_ratio_hist_edges = np.asarray([1.0, 1.1, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 5.0, 7.5, 10.0])
    aspect_ratio_histogram: list[float] | None = None
    if aspect_ratio_array.size:
        clipped_aspect_ratios = np.clip(
            aspect_ratio_array,
            float(aspect_ratio_hist_edges[0]),
            float(np.nextafter(aspect_ratio_hist_edges[-1], 0.0)),
        )
        aspect_hist, _ = np.histogram(clipped_aspect_ratios, bins=aspect_ratio_hist_edges, density=False)
        aspect_hist = np.asarray(aspect_hist, dtype=np.float64)
        if float(aspect_hist.sum()) > 0.0:
            aspect_hist /= float(aspect_hist.sum())
        aspect_ratio_histogram = aspect_hist.tolist()

    summary: dict[str, Any] = {
        "path": str(path),
        "step": _parse_step_from_name(path),
        "grain_count": int(len(area_array)),
        "cell_area": float(transform["cell_area"]),
        "area_fraction_sum": float(area_array.sum() / max(float(transform["cell_area"]), 1.0e-12)) if area_array.size else 0.0,
        "mean_grain_area": float(area_array.mean()) if area_array.size else 0.0,
        "second_moment_grain_size": (
            float(np.mean(np.square((area_array / np.mean(area_array)) - 1.0), dtype=np.float64))
            if area_array.size
            else 0.0
        ),
        "median_grain_area": float(np.median(area_array)) if area_array.size else 0.0,
        "std_grain_area": float(area_array.std()) if area_array.size else 0.0,
        "mean_equivalent_radius": float(radius_array.mean()) if radius_array.size else 0.0,
        "median_equivalent_radius": float(np.median(radius_array)) if radius_array.size else 0.0,
        "mean_perimeter": float(perimeter_array.mean()) if perimeter_array.size else 0.0,
        "mean_shape_factor": float(shape_array.mean()) if shape_array.size else 0.0,
        "mean_aspect_ratio": float(aspect_ratio_array.mean()) if aspect_ratio_array.size else 0.0,
        "median_aspect_ratio": float(np.median(aspect_ratio_array)) if aspect_ratio_array.size else 0.0,
    }
    if aspect_ratio_histogram is not None:
        summary["aspect_ratio_histogram"] = aspect_ratio_histogram
        summary["aspect_ratio_bin_edges"] = aspect_ratio_hist_edges.tolist()

    flynn_fabric = None
    if kept_flynn_ids:
        flynn_sections = _collect_numeric_section_specs(sections, ["EULER_3"], kept_flynn_ids)
        if "EULER_3" in dict(flynn_sections.get("values", {})) and int(
            dict(flynn_sections.get("component_counts", {})).get("EULER_3", 0)
        ) >= 3:
            flynn_euler = np.asarray(flynn_sections["values"]["EULER_3"], dtype=np.float64)
            flynn_fabric = _fabric_summary_from_orientations(
                source_name="EULER_3",
                euler_rows=flynn_euler,
                weights=area_array,
            )

    unode_fabric = None
    unodes = _parse_unodes(sections.get("UNODES", ()))
    if unodes:
        unode_ids = [int(unode_id) for unode_id, _, _ in unodes]
        unode_sections = _collect_numeric_section_specs(sections, ["U_EULER_3"], unode_ids)
        if "U_EULER_3" in dict(unode_sections.get("values", {})) and int(
            dict(unode_sections.get("component_counts", {})).get("U_EULER_3", 0)
        ) >= 3:
            unode_euler = np.asarray(unode_sections["values"]["U_EULER_3"], dtype=np.float64)
            unode_fabric = _fabric_summary_from_orientations(
                source_name="U_EULER_3",
                euler_rows=unode_euler,
                weights=np.ones((unode_euler.shape[0],), dtype=np.float64),
            )

    if unode_fabric is not None:
        summary["fabric"] = unode_fabric
    elif flynn_fabric is not None:
        summary["fabric"] = flynn_fabric

    if unodes:
        unode_ids = [int(unode_id) for unode_id, _, _ in unodes]
        mechanics_summary = _mechanics_summary_from_unode_fields(sections, unode_ids)
        if mechanics_summary is not None:
            summary["mechanics"] = mechanics_summary
    payload_summary = _mechanics_payload_summary_from_mesh_sidecar(path)
    if payload_summary is not None:
        mechanics_summary = dict(summary.get("mechanics", {}))
        mechanics_summary.update(
            {
                key: value
                for key, value in payload_summary.items()
                if key
                not in {
                    "mean_normalized_strain_rate",
                    "mean_normalized_stress",
                    "mean_differential_stress",
                    "mean_basal_activity",
                    "mean_prismatic_activity",
                    "mean_pyramidal_activity",
                    "mean_total_activity",
                    "mean_prismatic_fraction",
                    "prismatic_to_basal_ratio",
                }
                or key not in mechanics_summary
            }
        )
        for key in (
            "mean_normalized_strain_rate",
            "mean_normalized_stress",
            "mean_differential_stress",
            "mean_basal_activity",
            "mean_prismatic_activity",
            "mean_pyramidal_activity",
            "mean_total_activity",
            "mean_prismatic_fraction",
            "prismatic_to_basal_ratio",
        ):
            if key in payload_summary and payload_summary[key] is not None:
                mechanics_summary[key] = payload_summary[key]
        if "direct_strain_axis" in payload_summary and payload_summary["direct_strain_axis"] is not None:
            mechanics_summary["direct_strain_axis"] = float(payload_summary["direct_strain_axis"])
        elif "cumulative_simple_shear" in payload_summary and payload_summary["cumulative_simple_shear"] is not None:
            mechanics_summary["direct_strain_axis"] = float(payload_summary["cumulative_simple_shear"])
        if "strain_axis_source" in payload_summary:
            mechanics_summary["strain_axis_source"] = str(payload_summary["strain_axis_source"])
        summary["mechanics"] = mechanics_summary
        summary["mechanics_payload"] = payload_summary

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
    ordered = sorted(
        snapshots,
        key=lambda snapshot: (
            snapshot["step"] is None,
            snapshot["step"] if snapshot["step"] is not None else snapshot["path"],
        ),
    )
    return _overlay_legacy_allout_rows(ordered, directory=directory_path)


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
            "mean_aspect_ratio_abs_diff": abs(float(cand["mean_aspect_ratio"]) - float(ref["mean_aspect_ratio"])),
        }
        ref_aspect_hist = ref.get("aspect_ratio_histogram")
        cand_aspect_hist = cand.get("aspect_ratio_histogram")
        if ref_aspect_hist is not None and cand_aspect_hist is not None:
            ref_hist = np.asarray(ref_aspect_hist, dtype=np.float64)
            cand_hist = np.asarray(cand_aspect_hist, dtype=np.float64)
            if ref_hist.shape == cand_hist.shape:
                step_report["aspect_ratio_hist_l1"] = float(np.abs(ref_hist - cand_hist).sum())
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
        ref_fabric = ref.get("fabric")
        cand_fabric = cand.get("fabric")
        if ref_fabric is not None and cand_fabric is not None:
            ref_pole = (
                ref_fabric.get("c_axis", {}).get("pole_figure")
                if isinstance(ref_fabric.get("c_axis"), dict)
                else None
            )
            cand_pole = (
                cand_fabric.get("c_axis", {}).get("pole_figure")
                if isinstance(cand_fabric.get("c_axis"), dict)
                else None
            )
            if ref_pole is not None and cand_pole is not None:
                ref_hist = np.asarray(ref_pole.get("histogram", ()), dtype=np.float64)
                cand_hist = np.asarray(cand_pole.get("histogram", ()), dtype=np.float64)
                if ref_hist.shape == cand_hist.shape and ref_hist.size:
                    step_report["fabric_c_axis_pole_figure_l1"] = float(np.abs(ref_hist - cand_hist).sum())
                step_report["fabric_c_axis_vertical_fraction_abs_diff"] = abs(
                    float(cand_pole["fraction_within_15deg"]) - float(ref_pole["fraction_within_15deg"])
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
        summary["mean_aspect_ratio_abs_diff_mean"] = float(
            np.mean([entry["mean_aspect_ratio_abs_diff"] for entry in per_step], dtype=np.float64)
        )
        aspect_l1 = [entry["aspect_ratio_hist_l1"] for entry in per_step if "aspect_ratio_hist_l1" in entry]
        if aspect_l1:
            summary["aspect_ratio_hist_l1_mean"] = float(np.mean(aspect_l1, dtype=np.float64))
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
        pole_l1 = [entry["fabric_c_axis_pole_figure_l1"] for entry in per_step if "fabric_c_axis_pole_figure_l1" in entry]
        if pole_l1:
            summary["fabric_c_axis_pole_figure_l1_mean"] = float(np.mean(pole_l1, dtype=np.float64))
        pole_vertical_diff = [
            entry["fabric_c_axis_vertical_fraction_abs_diff"]
            for entry in per_step
            if "fabric_c_axis_vertical_fraction_abs_diff" in entry
        ]
        if pole_vertical_diff:
            summary["fabric_c_axis_vertical_fraction_abs_diff_mean"] = float(
                np.mean(pole_vertical_diff, dtype=np.float64)
            )

    return {
        "reference_dir": str(Path(reference_dir)),
        "candidate_dir": str(Path(candidate_dir)),
        "reference": reference,
        "candidate": candidate,
        "per_step": per_step,
        "summary": summary,
    }


def summarize_grain_survival_diagnostics(
    directory: str | Path,
    *,
    pattern: str = "*.elle",
    attribute: str = "auto",
) -> dict[str, Any]:
    directory_path = Path(directory)
    paths = sorted(directory_path.glob(pattern))
    if not paths:
        return {
            "directory": str(directory_path),
            "num_snapshots": 0,
            "path_sequence": [],
            "per_grain": [],
            "summary": {},
        }

    ordered_paths = sorted(paths, key=lambda path: (_parse_step_from_name(path) is None, _parse_step_from_name(path) or 0, str(path)))
    initial_path = ordered_paths[0]
    final_path = ordered_paths[-1]
    try:
        initial_seed = load_elle_label_seed(initial_path, attribute=attribute)
        final_seed = load_elle_label_seed(final_path, attribute=attribute)
    except ValueError:
        return {
            "directory": str(directory_path),
            "num_snapshots": int(len(ordered_paths)),
            "path_sequence": [str(path) for path in ordered_paths],
            "per_grain": [],
            "summary": {},
        }
    initial_areas = _source_label_area_fractions(initial_seed)
    final_areas = _source_label_area_fractions(final_seed)
    schmid_by_label = _initial_source_label_schmid_factors(initial_path, attribute=attribute)

    first_zero_step_by_label: dict[int, int | None] = {}
    for source_label in sorted(int(value) for value in initial_areas):
        first_zero_step_by_label[source_label] = None
    for path in ordered_paths[1:]:
        seed = load_elle_label_seed(path, attribute=attribute)
        source_areas = _source_label_area_fractions(seed)
        step = _parse_step_from_name(path)
        for source_label in list(first_zero_step_by_label):
            if first_zero_step_by_label[source_label] is not None:
                continue
            if float(source_areas.get(source_label, 0.0)) <= 0.0:
                first_zero_step_by_label[source_label] = int(step) if step is not None else None

    per_grain: list[dict[str, Any]] = []
    initial_sizes: list[float] = []
    survival_flags: list[float] = []
    basal_schmid_factors: list[float] = []
    for source_label in sorted(int(value) for value in initial_areas):
        initial_area = float(initial_areas.get(source_label, 0.0))
        final_area = float(final_areas.get(source_label, 0.0))
        survived = bool(final_area > 0.0)
        schmid = schmid_by_label.get(source_label)
        per_grain.append(
            {
                "source_label": int(source_label),
                "initial_area_fraction": initial_area,
                "final_area_fraction": final_area,
                "retention_ratio": float(final_area / initial_area) if initial_area > 1.0e-12 else float("nan"),
                "survived_to_final": survived,
                "first_zero_step": first_zero_step_by_label.get(source_label),
                "initial_basal_schmid_factor": None if schmid is None else float(schmid),
            }
        )
        initial_sizes.append(initial_area)
        survival_flags.append(1.0 if survived else 0.0)
        if schmid is not None and np.isfinite(float(schmid)):
            basal_schmid_factors.append(float(schmid))

    size_correlation = _safe_pearson(np.asarray(initial_sizes, dtype=np.float64), np.asarray(survival_flags, dtype=np.float64))
    schmid_correlation = float("nan")
    if len(basal_schmid_factors) == len(per_grain):
        schmid_correlation = _safe_pearson(
            np.asarray(basal_schmid_factors, dtype=np.float64),
            np.asarray(survival_flags, dtype=np.float64),
        )

    size_stronger: bool | None = None
    if np.isfinite(size_correlation) and np.isfinite(schmid_correlation):
        size_stronger = bool(abs(float(size_correlation)) > abs(float(schmid_correlation)) + 1.0e-12)
    elif np.isfinite(size_correlation) and not np.isfinite(schmid_correlation):
        size_stronger = True

    return {
        "directory": str(directory_path),
        "num_snapshots": int(len(ordered_paths)),
        "path_sequence": [str(path) for path in ordered_paths],
        "per_grain": per_grain,
        "summary": {
            "initial_step": _parse_step_from_name(initial_path),
            "final_step": _parse_step_from_name(final_path),
            "num_initial_grains": int(len(per_grain)),
            "num_survived_to_final": int(sum(1 for entry in per_grain if entry["survived_to_final"])),
            "survival_fraction": float(np.mean(survival_flags, dtype=np.float64)) if survival_flags else float("nan"),
            "initial_size_survival_correlation": size_correlation,
            "initial_basal_schmid_survival_correlation": schmid_correlation,
            "initial_size_correlation_stronger_than_schmid": size_stronger,
        },
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
