from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import boundary_mask, dominant_grain_map
from .elle_export import extract_flynn_topology
from .elle_visualize import (
    _parse_elle_sections,
    _parse_flynns,
    _parse_location,
    _parse_sparse_values,
)


@dataclass(frozen=True)
class MeshRelaxationConfig:
    steps: int = 1
    speed_up: float = 1.0
    switch_distance: float | None = None
    min_angle_degrees: float = 20.0
    movement_model: str = "legacy"
    use_diagonal_trials: bool = False
    use_elle_physical_units: bool = False
    boundary_energy: float = 1.0
    random_seed: int = 0
    topology_steps: int = 0
    min_node_separation_factor: float = 1.0
    max_node_separation_factor: float = 2.2


@dataclass(frozen=True)
class MeshFeedbackConfig:
    every: int = 0
    strength: float = 0.2
    transport_strength: float = 1.0
    update_mode: str = "blend"
    kernel_advection_every: int = 0
    kernel_advection_strength: float = 0.0
    kernel_predictor_corrector: bool = False
    boundary_width: int = 1
    initial_mesh_state: dict[str, Any] | None = None
    relax_config: MeshRelaxationConfig = field(default_factory=MeshRelaxationConfig)


_GBM_TRANSFER_INCREMENTS = 20
_GBM_BELL_VOLUME = (0.5 * np.pi) * (np.pi * np.pi - 4.0)
_GBM_PARTITION_COEFF = 1.0
_GBM_PARTITION_TOL = 1.0e-12


def _as_labels(data) -> np.ndarray:
    array = np.asarray(data)
    if array.ndim == 3:
        return dominant_grain_map(array)
    if array.ndim == 2:
        return array.astype(np.int32)
    raise ValueError("expected phi with shape (num_grains, nx, ny) or labels with shape (nx, ny)")


def _periodic_relative(point: np.ndarray, origin: np.ndarray) -> np.ndarray:
    delta = point - origin
    return (delta + 0.5) % 1.0 - 0.5


def _plot_xy(point: np.ndarray, origin: np.ndarray) -> np.ndarray:
    return origin + _periodic_relative(point, origin)


def _edge_length(nodes: np.ndarray, node_a: int, node_b: int) -> float:
    delta = _periodic_relative(nodes[int(node_b)], nodes[int(node_a)])
    return float(np.hypot(delta[0], delta[1]))


def _periodic_midpoint(point_a: np.ndarray, point_b: np.ndarray) -> np.ndarray:
    plotted_b = _plot_xy(point_b, point_a)
    return ((point_a + plotted_b) * 0.5) % 1.0


def _dilate_mask(mask: np.ndarray, width: int) -> np.ndarray:
    dilated = np.asarray(mask, dtype=bool).copy()
    for _ in range(max(int(width), 0)):
        dilated = (
            dilated
            | np.roll(dilated, +1, axis=0)
            | np.roll(dilated, -1, axis=0)
            | np.roll(dilated, +1, axis=1)
            | np.roll(dilated, -1, axis=1)
        )
    return dilated


def _grid_sample_coordinates(grid_shape: tuple[int, int] | list[int]) -> tuple[np.ndarray, np.ndarray]:
    nx, ny = int(grid_shape[0]), int(grid_shape[1])
    x_centers = (np.arange(nx, dtype=np.float64) + 0.5) / float(nx)
    y_centers = 1.0 - (np.arange(ny, dtype=np.float64) + 0.5) / float(ny)
    return (
        np.broadcast_to(x_centers[:, None], (nx, ny)),
        np.broadcast_to(y_centers[None, :], (nx, ny)),
    )


def _node_neighbors(flynns: list[dict[str, Any]]) -> dict[int, set[int]]:
    neighbors: dict[int, set[int]] = {}
    for flynn in flynns:
        node_ids = flynn["node_ids"]
        count = len(node_ids)
        for index, node_id in enumerate(node_ids):
            left = node_ids[index - 1]
            right = node_ids[(index + 1) % count]
            neighbors.setdefault(int(node_id), set()).update((int(left), int(right)))
    return neighbors


def _parse_options(lines: tuple[str, ...]) -> dict[str, float]:
    options: dict[str, float] = {}
    for raw_line in lines:
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) >= 2:
            try:
                options[parts[0]] = float(parts[1])
            except ValueError:
                continue
    return options


def _node_flynn_ids(flynns: list[dict[str, Any]]) -> dict[int, set[int]]:
    membership: dict[int, set[int]] = {}
    for flynn in flynns:
        for node_id in flynn["node_ids"]:
            membership.setdefault(int(node_id), set()).add(int(flynn["flynn_id"]))
    return membership


def _copy_flynns(flynns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    copied: list[dict[str, Any]] = []
    for flynn in flynns:
        copied.append(
            {
                key: ([int(entry) for entry in value] if key in {"node_ids", "neighbors", "parents"} else value)
                for key, value in flynn.items()
            }
        )
    return copied


def _serializable_mesh(
    nodes: np.ndarray,
    flynns: list[dict[str, Any]],
    node_neighbors: dict[int, set[int]],
    node_flynn_membership: dict[int, set[int]],
    stats: dict[str, Any],
    *,
    events: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    node_records = []
    for node_id, xy in enumerate(nodes):
        neighbors = sorted(int(neighbor) for neighbor in node_neighbors.get(node_id, set()))
        flynn_ids = sorted(int(flynn_id) for flynn_id in node_flynn_membership.get(node_id, set()))
        degree = len(neighbors)
        junction_type = "other"
        if degree == 2:
            junction_type = "double"
        elif degree == 3:
            junction_type = "triple"
        node_records.append(
            {
                "node_id": int(node_id),
                "x": float(xy[0]),
                "y": float(xy[1]),
                "degree": int(degree),
                "junction_type": junction_type,
                "neighbors": neighbors,
                "flynns": flynn_ids,
            }
        )

    flynn_records = []
    for flynn in flynns:
        record = {key: value for key, value in flynn.items() if key != "node_ids"}
        record["node_ids"] = [int(node_id) for node_id in flynn["node_ids"]]
        flynn_records.append(record)

    return {
        "nodes": node_records,
        "flynns": flynn_records,
        "stats": stats,
        "events": [dict(event) for event in (events or [])],
    }


def load_elle_mesh_seed(
    elle_path: str | Path,
    label_seed: dict[str, object],
) -> tuple[dict[str, Any], dict[str, float]]:
    path = Path(elle_path)
    sections = _parse_elle_sections(path)
    if "LOCATION" not in sections or "FLYNNS" not in sections:
        raise ValueError(f"ELLE file must contain LOCATION and FLYNNS sections: {path}")

    source_labels = [int(value) for value in label_seed.get("source_labels", [])]
    if not source_labels:
        raise ValueError(f"label_seed for {path} does not contain source_labels")
    source_to_compact = {int(label): index for index, label in enumerate(source_labels)}
    grid_shape = tuple(int(value) for value in label_seed["grid_shape"])

    location_map = _parse_location(sections["LOCATION"])
    ordered_node_ids = sorted(int(node_id) for node_id in location_map)
    node_id_map = {int(node_id): index for index, node_id in enumerate(ordered_node_ids)}
    nodes = np.asarray([location_map[node_id] for node_id in ordered_node_ids], dtype=np.float64)

    flynn_records = _parse_flynns(sections["FLYNNS"])
    flynn_label_values: dict[int, int] = {}
    if "F_ATTRIB_C" in sections:
        _, flynn_attr_values = _parse_sparse_values(sections["F_ATTRIB_C"])
        flynn_label_values = {
            int(flynn_id): int(round(value)) for flynn_id, value in flynn_attr_values.items()
        }

    flynns: list[dict[str, Any]] = []
    missing_labels: list[int] = []
    for component_index, flynn in enumerate(flynn_records):
        original_flynn_id = int(flynn["flynn_id"])
        source_label = int(flynn_label_values.get(original_flynn_id, original_flynn_id))
        compact_label = source_to_compact.get(source_label)
        if compact_label is None:
            missing_labels.append(original_flynn_id)
            continue
        remapped_nodes = [int(node_id_map[int(node_id)]) for node_id in flynn["node_ids"] if int(node_id) in node_id_map]
        if len(remapped_nodes) < 3:
            continue
        flynns.append(
            {
                "flynn_id": int(compact_label),
                "label": int(compact_label),
                "component_index": int(component_index),
                "node_ids": remapped_nodes,
                "source_flynn_id": int(original_flynn_id),
                "source_label": int(source_label),
                "retained_identity": True,
                "neighbors": [],
                "parents": [],
            }
        )

    if not flynns:
        raise ValueError(f"could not map any ELLE flynns onto compact seed labels in {path}")

    node_neighbors = _node_neighbors(flynns)
    node_flynn_membership = _node_flynn_ids(flynns)
    stats = {
        "grid_shape": [int(grid_shape[0]), int(grid_shape[1])],
        "num_nodes": int(len(nodes)),
        "num_flynns": int(len(flynns)),
        "double_junctions": int(sum(len(neighbors) == 2 for neighbors in node_neighbors.values())),
        "triple_junctions": int(sum(len(neighbors) == 3 for neighbors in node_neighbors.values())),
        "topology_components": int(len(flynns)),
        "holes_skipped": 0,
        "mesh_seed_source": "elle",
        "mesh_seed_path": str(path),
        "mesh_seed_missing_flynns": int(len(missing_labels)),
    }
    mesh_state = _serializable_mesh(nodes, flynns, node_neighbors, node_flynn_membership, stats)

    option_values = _parse_options(sections.get("OPTIONS", ()))
    relax_overrides: dict[str, float] = {}
    switch_distance = float(option_values["SwitchDistance"]) if "SwitchDistance" in option_values else 0.0
    if switch_distance > 0.0:
        relax_overrides["switch_distance"] = switch_distance
        if "MinNodeSeparation" in option_values:
            relax_overrides["min_node_separation_factor"] = float(
                option_values["MinNodeSeparation"] / switch_distance
            )
        if "MaxNodeSeparation" in option_values:
            relax_overrides["max_node_separation_factor"] = float(
                option_values["MaxNodeSeparation"] / switch_distance
            )
    if "SpeedUp" in option_values:
        relax_overrides["speed_up"] = float(option_values["SpeedUp"])

    for key in ("SwitchDistance", "MinNodeSeparation", "MaxNodeSeparation", "SpeedUp"):
        if key in option_values:
            mesh_state["stats"][f"elle_option_{key.lower()}"] = float(option_values[key])
    for key in ("BoundaryWidth", "UnitLength", "MassIncrement"):
        if key in option_values:
            mesh_state["stats"][f"elle_option_{key.lower()}"] = float(option_values[key])

    unode_ids = tuple(int(unode_id) for unode_id in label_seed.get("unode_ids", ()))
    unode_positions = tuple(tuple(float(value) for value in xy) for xy in label_seed.get("unode_positions", ()))
    unode_grid_indices = tuple(
        tuple(int(value) for value in ij) for ij in label_seed.get("unode_grid_indices", ())
    )
    if unode_ids and unode_positions and unode_grid_indices:
        mesh_state["_runtime_seed_unodes"] = {
            "ids": unode_ids,
            "positions": unode_positions,
            "grid_indices": unode_grid_indices,
            "grid_shape": tuple(int(value) for value in grid_shape),
        }
    if label_seed.get("unode_field_values"):
        mesh_state["_runtime_seed_unode_fields"] = {
            "label_attribute": str(label_seed.get("attribute", "U_ATTRIB_A")),
            "field_order": tuple(str(name) for name in label_seed.get("unode_field_order", ())),
            "source_labels": tuple(int(value) for value in source_labels),
            "roi": float(_estimate_seed_unode_roi(mesh_state["_runtime_seed_unodes"]))
            if "_runtime_seed_unodes" in mesh_state
            else 0.0,
            "unode_area": float(_estimate_seed_unode_area(mesh_state["_runtime_seed_unodes"]))
            if "_runtime_seed_unodes" in mesh_state
            else 0.0,
            "values": {
                str(name): tuple(float(value) for value in values)
                for name, values in dict(label_seed.get("unode_field_values", {})).items()
            },
        }
    node_field_names = [
        str(name)
        for name in sections
        if str(name).startswith("N_ATTRIB_") or str(name).startswith("N_CONC_")
    ]
    if node_field_names:
        node_field_values: dict[str, tuple[float, ...]] = {}
        for field_name in node_field_names:
            default_value, sparse_values = _parse_sparse_values(sections[field_name])
            ordered_values = []
            for node_id in ordered_node_ids:
                ordered_values.append(float(sparse_values.get(int(node_id), default_value)))
            node_field_values[str(field_name)] = tuple(ordered_values)
        mesh_state["_runtime_seed_node_fields"] = {
            "field_order": tuple(node_field_names),
            "positions": tuple((float(xy[0]), float(xy[1])) for xy in nodes),
            "values": node_field_values,
        }

    return mesh_state, relax_overrides


def build_mesh_state(data, tracked_topology: dict[str, Any] | None = None) -> dict[str, Any]:
    labels = _as_labels(data)
    nx, ny = labels.shape
    node_locations, flynns, topology_stats = extract_flynn_topology(labels)
    nodes = np.asarray(node_locations, dtype=np.float64)

    if tracked_topology is not None:
        tracked_by_local_index = {
            int(flynn["local_index"]): flynn for flynn in tracked_topology.get("flynns", [])
        }
        for flynn in flynns:
            tracked_entry = tracked_by_local_index.get(int(flynn["component_index"]))
            if tracked_entry is None:
                continue
            flynn["flynn_id"] = int(tracked_entry["flynn_id"])
            flynn["neighbors"] = [int(neighbor) for neighbor in tracked_entry.get("neighbors", [])]
            flynn["parents"] = [int(parent) for parent in tracked_entry.get("parents", [])]
            flynn["retained_identity"] = bool(tracked_entry.get("retained_identity", False))
        flynns.sort(key=lambda flynn: int(flynn["flynn_id"]))

    node_neighbors = _node_neighbors(flynns)
    node_flynn_membership = _node_flynn_ids(flynns)
    stats = {
        "grid_shape": [int(nx), int(ny)],
        "num_nodes": int(len(nodes)),
        "num_flynns": int(len(flynns)),
        "double_junctions": int(sum(len(neighbors) == 2 for neighbors in node_neighbors.values())),
        "triple_junctions": int(sum(len(neighbors) == 3 for neighbors in node_neighbors.values())),
        "topology_components": int(topology_stats["components"]),
        "holes_skipped": int(topology_stats["holes_skipped"]),
    }
    return _serializable_mesh(nodes, flynns, node_neighbors, node_flynn_membership, stats)


def rasterize_mesh_labels(
    mesh_state: dict[str, Any],
    grid_shape: tuple[int, int] | list[int],
    *,
    fallback_labels=None,
) -> np.ndarray:
    nx, ny = int(grid_shape[0]), int(grid_shape[1])
    nodes = np.array([[node["x"], node["y"]] for node in mesh_state["nodes"]], dtype=np.float64)
    if fallback_labels is None:
        mesh_labels = np.full((nx, ny), -1, dtype=np.int32)
    else:
        mesh_labels = np.asarray(fallback_labels, dtype=np.int32).copy()

    sample_x, sample_y = _grid_sample_coordinates((nx, ny))
    snap_eps = 0.5 / float(max(nx, ny))

    flynn_entries = []
    for flynn in mesh_state["flynns"]:
        points = np.array([nodes[int(node_id)] for node_id in flynn["node_ids"]], dtype=np.float64)
        points[np.abs(points) <= snap_eps] = 0.0
        points[np.abs(points - 1.0) <= snap_eps] = 1.0
        if len(points) < 3:
            continue
        point_list = [points[index] for index in range(len(points))]
        flynn_entries.append((abs(_polygon_signed_area_from_points(point_list)), flynn, point_list))

    for _, flynn, points in sorted(flynn_entries, key=lambda item: item[0], reverse=True):
        polygon_x = np.array([point[0] for point in points], dtype=np.float64)
        polygon_y = np.array([point[1] for point in points], dtype=np.float64)
        candidate_mask = (
            (sample_x >= polygon_x.min() - 1e-9)
            & (sample_x <= polygon_x.max() + 1e-9)
            & (sample_y >= polygon_y.min() - 1e-9)
            & (sample_y <= polygon_y.max() + 1e-9)
        )
        inside_mask = candidate_mask & _point_in_polygon_mask(sample_x, sample_y, points)
        mesh_labels[inside_mask] = int(flynn["label"])

    if np.any(mesh_labels < 0):
        if fallback_labels is not None:
            fallback_np = np.asarray(fallback_labels, dtype=np.int32)
            mesh_labels[mesh_labels < 0] = fallback_np[mesh_labels < 0]
        else:
            mesh_labels[mesh_labels < 0] = 0
    return mesh_labels.astype(np.int32)


def mesh_labels_to_order_parameters(labels: np.ndarray, num_grains: int) -> np.ndarray:
    label_array = np.asarray(labels, dtype=np.int32)
    if label_array.ndim != 2:
        raise ValueError("expected labels with shape (nx, ny)")
    if int(num_grains) < 1:
        raise ValueError("num_grains must be >= 1")
    clipped = np.clip(label_array, 0, int(num_grains) - 1)
    phi = np.moveaxis(np.eye(int(num_grains), dtype=np.float32)[clipped], -1, 0)
    total = phi.sum(axis=0, keepdims=True) + 1e-12
    return phi / total


def _plot_xy_polygon(point: np.ndarray, origin: np.ndarray) -> np.ndarray:
    delta = np.asarray(point, dtype=np.float64) - np.asarray(origin, dtype=np.float64)
    adjusted = (delta + 0.5) % 1.0 - 0.5
    adjusted = np.where(np.abs(np.abs(delta) - 1.0) <= 1e-9, delta, adjusted)
    adjusted = np.where((np.abs(adjusted + 0.5) <= 1e-9) & (delta > 0.0), 0.5, adjusted)
    adjusted = np.where((np.abs(adjusted - 0.5) <= 1e-9) & (delta < 0.0), -0.5, adjusted)
    return np.asarray(origin, dtype=np.float64) + adjusted


def _plot_cycle_points_polygon(node_ids: list[int], nodes: np.ndarray) -> list[np.ndarray]:
    if not node_ids:
        return []
    points = [np.asarray(nodes[int(node_ids[0])], dtype=np.float64)]
    for node_id in node_ids[1:]:
        points.append(_plot_xy_polygon(nodes[int(node_id)], points[-1]))
    return points


def assign_seed_unodes_from_mesh(
    mesh_state: dict[str, Any],
    seed_unodes: dict[str, Any],
    *,
    fallback_labels=None,
) -> tuple[np.ndarray, dict[str, int]]:
    grid_shape = tuple(int(value) for value in seed_unodes["grid_shape"])
    nx, ny = grid_shape
    if fallback_labels is None:
        mesh_labels = np.full((nx, ny), -1, dtype=np.int32)
    else:
        mesh_labels = np.asarray(fallback_labels, dtype=np.int32).copy()

    node_positions = np.array([[node["x"], node["y"]] for node in mesh_state["nodes"]], dtype=np.float64)
    sample_points = np.asarray(seed_unodes["positions"], dtype=np.float64)
    sample_grid_indices = np.asarray(seed_unodes["grid_indices"], dtype=np.int32)
    if sample_points.shape[0] != sample_grid_indices.shape[0]:
        raise ValueError("seed unode positions and grid indices must have the same length")

    unassigned_mask = np.ones(sample_points.shape[0], dtype=bool)
    assigned = 0

    flynn_entries = []
    for flynn in mesh_state["flynns"]:
        node_ids = [int(node_id) for node_id in flynn["node_ids"]]
        if len(node_ids) < 3:
            continue
        polygon_points = _plot_cycle_points_polygon(node_ids, node_positions)
        if len(polygon_points) < 3:
            continue
        area = abs(_polygon_signed_area_from_points(polygon_points))
        flynn_entries.append((float(area), int(flynn["label"]), polygon_points))

    for _, label, polygon_points in sorted(flynn_entries, key=lambda item: item[0], reverse=True):
        if not np.any(unassigned_mask):
            continue
        inside_mask = _mark_seed_points_in_polygon(
            sample_points,
            polygon_points,
            active_mask=unassigned_mask,
        )
        if not np.any(inside_mask):
            continue
        assigned_indices = np.flatnonzero(inside_mask)
        mesh_labels[
            sample_grid_indices[assigned_indices, 0],
            sample_grid_indices[assigned_indices, 1],
        ] = int(label)
        unassigned_mask[assigned_indices] = False
        assigned += int(assigned_indices.size)

    if np.any(mesh_labels < 0):
        if fallback_labels is not None:
            fallback_np = np.asarray(fallback_labels, dtype=np.int32)
            mesh_labels[mesh_labels < 0] = fallback_np[mesh_labels < 0]
        else:
            mesh_labels[mesh_labels < 0] = 0

    return mesh_labels.astype(np.int32), {
        "assigned_unodes": int(assigned),
        "unassigned_unodes": int(unassigned_mask.sum()),
    }


def _mark_seed_points_in_polygon(
    sample_points: np.ndarray,
    polygon_points: list[np.ndarray],
    *,
    active_mask: np.ndarray | None = None,
) -> np.ndarray:
    if not polygon_points:
        return np.zeros(sample_points.shape[0], dtype=bool)
    if active_mask is None:
        candidate_indices = np.arange(sample_points.shape[0], dtype=np.int32)
    else:
        candidate_indices = np.flatnonzero(np.asarray(active_mask, dtype=bool))
    if candidate_indices.size == 0:
        return np.zeros(sample_points.shape[0], dtype=bool)

    anchor = np.asarray(polygon_points[0], dtype=np.float64)
    plotted_points = _plot_xy_polygon(sample_points[candidate_indices], anchor)
    polygon_x = np.array([float(point[0]) for point in polygon_points], dtype=np.float64)
    polygon_y = np.array([float(point[1]) for point in polygon_points], dtype=np.float64)
    inside_mask = np.zeros(candidate_indices.shape[0], dtype=bool)

    if len(polygon_points) == 3:
        candidate_bbox = (
            (plotted_points[:, 0] >= polygon_x.min() - 1e-9)
            & (plotted_points[:, 0] <= polygon_x.max() + 1e-9)
            & (plotted_points[:, 1] >= polygon_y.min() - 1e-9)
            & (plotted_points[:, 1] <= polygon_y.max() + 1e-9)
        )
        if np.any(candidate_bbox):
            local_indices = np.flatnonzero(candidate_bbox)
            local_inside = _point_in_polygon_mask(
                plotted_points[local_indices, 0][:, None],
                plotted_points[local_indices, 1][:, None],
                polygon_points,
            )[:, 0]
            if np.any(local_inside):
                inside_mask[local_indices] = local_inside
        result = np.zeros(sample_points.shape[0], dtype=bool)
        result[candidate_indices] = inside_mask
        return result

    for shift_x in (-1.0, 0.0, 1.0):
        for shift_y in (-1.0, 0.0, 1.0):
            shifted = plotted_points + np.array([shift_x, shift_y], dtype=np.float64)
            candidate_bbox = (
                (shifted[:, 0] >= polygon_x.min() - 1e-9)
                & (shifted[:, 0] <= polygon_x.max() + 1e-9)
                & (shifted[:, 1] >= polygon_y.min() - 1e-9)
                & (shifted[:, 1] <= polygon_y.max() + 1e-9)
            )
            if not np.any(candidate_bbox):
                continue
            local_indices = np.flatnonzero(candidate_bbox)
            local_inside = _point_in_polygon_mask(
                shifted[local_indices, 0][:, None],
                shifted[local_indices, 1][:, None],
                polygon_points,
            )[:, 0]
            if np.any(local_inside):
                inside_mask[local_indices] = inside_mask[local_indices] | local_inside

    result = np.zeros(sample_points.shape[0], dtype=bool)
    result[candidate_indices] = inside_mask
    return result


def _compute_swept_seed_unode_mask(
    base_mesh_state: dict[str, Any],
    moved_mesh_state: dict[str, Any],
    seed_unodes: dict[str, Any],
) -> np.ndarray:
    base_nodes = np.array([[node["x"], node["y"]] for node in base_mesh_state["nodes"]], dtype=np.float64)
    moved_nodes = np.array([[node["x"], node["y"]] for node in moved_mesh_state["nodes"]], dtype=np.float64)
    sample_points = np.asarray(seed_unodes["positions"], dtype=np.float64)
    active_seed_mask = np.zeros(sample_points.shape[0], dtype=bool)

    if base_nodes.shape != moved_nodes.shape:
        active_seed_mask[:] = True
        return active_seed_mask

    node_neighbors = _node_neighbors(base_mesh_state["flynns"])
    for node_id, neighbors in node_neighbors.items():
        old_node = base_nodes[int(node_id)]
        new_node = _plot_xy_polygon(moved_nodes[int(node_id)], old_node)
        if float(np.hypot(*(new_node - old_node))) <= 1e-9:
            continue
        for neighbor_id in sorted(int(value) for value in neighbors):
            pivot = _plot_xy_polygon(base_nodes[int(neighbor_id)], old_node)
            polygon_points = [
                np.asarray(old_node, dtype=np.float64),
                np.asarray(new_node, dtype=np.float64),
                np.asarray(pivot, dtype=np.float64),
            ]
            active_seed_mask |= _mark_seed_points_in_polygon(
                sample_points,
                polygon_points,
                active_mask=~active_seed_mask,
            )
    return active_seed_mask


def _estimate_seed_field_value(
    sample_points: np.ndarray,
    donor_values: np.ndarray,
    donor_mask: np.ndarray,
    point_index: int,
    *,
    roi: float,
) -> float | None:
    return _estimate_field_value_from_samples(
        sample_points,
        donor_values,
        donor_mask,
        np.asarray(sample_points[int(point_index)], dtype=np.float64),
        roi=roi,
    )


def _estimate_field_value_from_samples(
    sample_points: np.ndarray,
    donor_values: np.ndarray,
    donor_mask: np.ndarray,
    point: np.ndarray,
    *,
    roi: float,
) -> float | None:
    donor_indices = np.flatnonzero(np.asarray(donor_mask, dtype=bool))
    if donor_indices.size == 0:
        return None
    deltas = _periodic_relative(sample_points[donor_indices], np.asarray(point, dtype=np.float64))
    radial = np.sqrt(np.sum(deltas * deltas, axis=1))

    if roi > 0.0:
        local_mask = radial <= roi
        if np.any(local_mask):
            local_indices = donor_indices[local_mask]
            local_radial = radial[local_mask]
            weights = (1.0 + np.cos(np.pi * local_radial / roi)) * 0.5
            total_weight = float(np.sum(weights))
            if total_weight > 0.0:
                return float(np.sum(donor_values[local_indices] * weights) / total_weight)

    distances_sq = radial * radial
    order = np.argsort(distances_sq)[: min(8, donor_indices.size)]
    nearest_indices = donor_indices[order]
    nearest_dist_sq = distances_sq[order]
    weights = 1.0 / (nearest_dist_sq + 1.0e-10)
    return float(np.sum(donor_values[nearest_indices] * weights) / np.sum(weights))


def _estimate_seed_unode_roi(seed_unodes: dict[str, Any]) -> float:
    sample_points = np.asarray(seed_unodes["positions"], dtype=np.float64)
    if sample_points.shape[0] < 2:
        return 0.0

    x_values = np.unique(np.round(sample_points[:, 0], 12))
    y_values = np.unique(np.round(sample_points[:, 1], 12))
    spacings: list[float] = []
    if x_values.size > 1:
        diffs = np.diff(np.sort(x_values))
        positive = diffs[diffs > 1.0e-12]
        if positive.size:
            spacings.append(float(np.min(positive)))
    if y_values.size > 1:
        diffs = np.diff(np.sort(y_values))
        positive = diffs[diffs > 1.0e-12]
        if positive.size:
            spacings.append(float(np.min(positive)))

    if not spacings:
        return 0.0
    return 2.5 * float(max(spacings))


def _estimate_seed_unode_area(seed_unodes: dict[str, Any]) -> float:
    grid_shape = tuple(int(value) for value in seed_unodes.get("grid_shape", (0, 0)))
    if len(grid_shape) == 2 and grid_shape[0] > 0 and grid_shape[1] > 0:
        return 1.0 / float(grid_shape[0] * grid_shape[1])

    sample_points = np.asarray(seed_unodes.get("positions", ()), dtype=np.float64)
    if sample_points.shape[0] == 0:
        return 1.0
    return 1.0 / float(sample_points.shape[0])


def _is_mass_like_unode_field(field_name: str) -> bool:
    return str(field_name).startswith("U_CONC_")


def _apply_mass_conservation(
    current_values: np.ndarray,
    updated_values: np.ndarray,
    adjustment_mask: np.ndarray,
    *,
    unode_area: float,
) -> tuple[np.ndarray, float]:
    total_mass_before = float(np.sum(np.asarray(current_values, dtype=np.float64)) * unode_area)
    return _apply_mass_target(
        updated_values,
        total_mass_before,
        adjustment_mask,
        unode_area=unode_area,
    )


def _apply_mass_target(
    updated_values: np.ndarray,
    target_total_mass: float,
    adjustment_mask: np.ndarray,
    *,
    unode_area: float,
) -> tuple[np.ndarray, float]:
    adjusted = np.asarray(updated_values, dtype=np.float64).copy()
    active_mask = np.asarray(adjustment_mask, dtype=bool)
    if not np.any(active_mask):
        return adjusted, 0.0

    total_mass_after = float(np.sum(adjusted) * unode_area)
    residual_mass = float(target_total_mass) - total_mass_after
    if abs(residual_mass) <= 1.0e-12:
        return adjusted, 0.0

    if residual_mass > 0.0:
        weights = adjusted[active_mask]
        if float(np.sum(weights)) <= 1.0e-12:
            weights = np.ones(active_mask.sum(), dtype=np.float64)
        weights = weights / float(np.sum(weights))
        adjusted[active_mask] += residual_mass * weights / unode_area
    else:
        removable = np.maximum(adjusted[active_mask], 0.0) * unode_area
        removable_total = float(np.sum(removable))
        if removable_total <= 1.0e-12:
            return adjusted, float(total_mass_before - float(np.sum(adjusted) * unode_area))
        remove_mass = min(-residual_mass, removable_total)
        weights = removable / removable_total
        adjusted[active_mask] -= remove_mass * weights / unode_area
        adjusted[active_mask] = np.maximum(adjusted[active_mask], 0.0)

    final_residual = float(target_total_mass - float(np.sum(adjusted) * unode_area))
    return adjusted, final_residual


def _segment_point_weights(
    sample_points: np.ndarray,
    candidate_indices: np.ndarray,
    point_a: np.ndarray,
    *,
    point_b: np.ndarray | None = None,
    segment_area: float,
    roi: float,
    local_scale: float,
) -> np.ndarray:
    if candidate_indices.size == 0:
        return np.zeros(0, dtype=np.float64)
    deltas_a = _periodic_relative(sample_points[candidate_indices], np.asarray(point_a, dtype=np.float64))
    radial_a = np.sqrt(np.sum(deltas_a * deltas_a, axis=1))
    if point_b is not None:
        deltas_b = _periodic_relative(sample_points[candidate_indices], np.asarray(point_b, dtype=np.float64))
        radial_b = np.sqrt(np.sum(deltas_b * deltas_b, axis=1))
    else:
        radial_b = radial_a
    if roi > 0.0:
        weights_a = np.where(
            radial_a <= roi,
            (1.0 + np.cos(np.pi * radial_a / roi)) * 0.5,
            0.0,
        )
        weights_b = np.where(
            radial_b <= roi,
            (1.0 + np.cos(np.pi * radial_b / roi)) * 0.5,
            0.0,
        )
        area_factor = ((np.pi / roi) * (np.pi / roi)) * max(float(segment_area), 0.0) / max(_GBM_BELL_VOLUME, 1.0e-12)
        weights = 0.5 * (weights_a + weights_b) * area_factor
        if float(np.sum(weights)) > 1.0e-12:
            return weights.astype(np.float64)

    scale = max(float(local_scale), 1.0e-6)
    area_factor = max(float(segment_area), 0.0) / max(scale * scale, 1.0e-12)
    return (0.5 * ((1.0 / (radial_a + scale)) + (1.0 / (radial_b + scale))) * area_factor).astype(np.float64)


def _segment_line_support_weights(
    sample_points: np.ndarray,
    candidate_indices: np.ndarray,
    point_a: np.ndarray,
    *,
    point_b: np.ndarray | None = None,
    roi: float,
    local_scale: float,
) -> np.ndarray:
    if candidate_indices.size == 0:
        return np.zeros(0, dtype=np.float64)
    deltas_a = _periodic_relative(sample_points[candidate_indices], np.asarray(point_a, dtype=np.float64))
    radial_a = np.sqrt(np.sum(deltas_a * deltas_a, axis=1))
    if point_b is not None:
        deltas_b = _periodic_relative(sample_points[candidate_indices], np.asarray(point_b, dtype=np.float64))
        radial_b = np.sqrt(np.sum(deltas_b * deltas_b, axis=1))
    else:
        radial_b = radial_a
    if roi > 0.0:
        weights_a = np.where(
            radial_a <= roi,
            (1.0 + np.cos(np.pi * radial_a / roi)) * 0.5,
            0.0,
        )
        weights_b = np.where(
            radial_b <= roi,
            (1.0 + np.cos(np.pi * radial_b / roi)) * 0.5,
            0.0,
        )
        weights = 0.5 * (weights_a + weights_b)
        if float(np.sum(weights)) > 1.0e-12:
            return weights.astype(np.float64)

    scale = max(float(local_scale), 1.0e-6)
    return (0.5 * ((1.0 / (radial_a + scale)) + (1.0 / (radial_b + scale)))).astype(np.float64)


def _weighted_partition_concentration(
    total_mass: float,
    weight_total: float,
    *,
    unode_area: float,
    fallback: float,
) -> float:
    if float(weight_total) > _GBM_PARTITION_TOL and float(unode_area) > _GBM_PARTITION_TOL:
        return float(total_mass) / (float(unode_area) * float(weight_total))
    return float(fallback)


def _entry_partition_terms(
    entry: dict[str, Any],
    *,
    unode_area: float,
    partition_coeff: float = _GBM_PARTITION_COEFF,
) -> dict[str, float | bool]:
    entry_area = float(entry.get("area", 0.0))
    gb_incr_area = float(entry.get("gb_area_f", 0.0))
    gb_mass_i = float(entry.get("gb_mass_i", 0.0))
    swept_mass = float(entry.get("swept_mass_0", entry.get("swept_mass", 0.0)))
    swept_mass_1 = float(entry.get("swept_mass_1", swept_mass))
    enrich_mass = float(entry.get("enrich_mass_0", entry.get("enrich_mass", 0.0)))
    enrich_mass_1 = float(entry.get("enrich_mass_1", enrich_mass))
    sweep_weight_total = float(entry.get("sweep_weight_total_0", entry.get("sweep_weight_total", 0.0)))
    sweep_weight_total_1 = float(entry.get("sweep_weight_total_1", sweep_weight_total))
    enrich_weight_total = float(entry.get("enrich_weight_total_0", entry.get("enrich_weight_total", 0.0)))
    enrich_weight_total_1 = float(entry.get("enrich_weight_total_1", enrich_weight_total))
    sweep_line_weight_total = float(
        entry.get("sweep_weight_total_2", entry.get("sweep_line_weight_total", sweep_weight_total))
    )
    enrich_line_weight_total = float(
        entry.get("enrich_weight_total_2", entry.get("enrich_line_weight_total", enrich_weight_total))
    )
    total_support_weight = sweep_line_weight_total + enrich_line_weight_total
    if total_support_weight > _GBM_PARTITION_TOL:
        swept_area_frac = sweep_line_weight_total / total_support_weight
        enrich_area_frac = enrich_line_weight_total / total_support_weight
    else:
        swept_area_frac = 0.5
        enrich_area_frac = 0.5

    if entry_area > _GBM_PARTITION_TOL:
        overlap_frac = max(entry_area - gb_incr_area, 0.0) / entry_area
        overlap_frac = min(max(overlap_frac, 0.0), 1.0)
    else:
        overlap_frac = 0.0

    conc_b = float(gb_mass_i / gb_incr_area) if gb_incr_area > _GBM_PARTITION_TOL else 0.0
    conc_s = _weighted_partition_concentration(
        swept_mass,
        sweep_weight_total,
        unode_area=unode_area,
        fallback=conc_b,
    )
    conc_s1 = _weighted_partition_concentration(
        swept_mass_1,
        sweep_weight_total_1,
        unode_area=unode_area,
        fallback=conc_s,
    )
    conc_e = _weighted_partition_concentration(
        enrich_mass,
        enrich_weight_total,
        unode_area=unode_area,
        fallback=conc_b,
    )
    conc_e1 = _weighted_partition_concentration(
        enrich_mass_1,
        enrich_weight_total_1,
        unode_area=unode_area,
        fallback=conc_e,
    )
    partition_active = (
        abs(conc_s - conc_b) > _GBM_PARTITION_TOL
        or abs(conc_e - conc_b) > _GBM_PARTITION_TOL
    )

    combined_area = entry_area + gb_incr_area
    if combined_area > _GBM_PARTITION_TOL:
        conc_b_f = (gb_mass_i + swept_mass + enrich_mass) / combined_area
    else:
        conc_b_f = 0.0
    conc_s_f = conc_b_f * float(partition_coeff)
    conc_e_f = conc_b_f * float(partition_coeff)
    mass_chge_s = -swept_mass + (conc_s_f * entry_area * overlap_frac * swept_area_frac)
    raw_mass_chge_e = -enrich_mass + (
        conc_e_f
        * (
            ((1.0 - overlap_frac) * entry_area)
            + (entry_area * enrich_area_frac * overlap_frac)
        )
    )
    mass_chge_b = (gb_incr_area * conc_b_f) - gb_mass_i
    mass_chge_e = float(-(mass_chge_b + mass_chge_s))

    return {
        "conc_b": float(conc_b),
        "conc_s": float(conc_s),
        "conc_e": float(conc_e),
        "conc_s1": float(conc_s1),
        "conc_e1": float(conc_e1),
        "conc_b_f": float(conc_b_f),
        "conc_s_f": float(conc_s_f),
        "conc_e_f": float(conc_e_f),
        "overlap_frac": float(overlap_frac),
        "swept_area_frac": float(swept_area_frac),
        "enrich_area_frac": float(enrich_area_frac),
        "partition_active": bool(partition_active),
        "mass_chge_s": float(mass_chge_s),
        "raw_mass_chge_e": float(raw_mass_chge_e),
        "mass_chge_b": float(mass_chge_b),
        "mass_chge_e": float(mass_chge_e),
    }


def _partition_mass_node(
    entries: list[dict[str, Any]],
    *,
    partition_coeff: float = _GBM_PARTITION_COEFF,
) -> dict[str, Any]:
    total_swept_mass = float(sum(float(entry.get("swept_mass", 0.0)) for entry in entries))
    total_swept_area = float(sum(float(entry.get("area", 0.0)) for entry in entries))
    total_gb_area_i = float(sum(float(entry.get("gb_area_i", 0.0)) for entry in entries))
    total_gb_area_f = float(sum(float(entry.get("gb_area_f", 0.0)) for entry in entries))
    total_gb_mass = float(sum(float(entry.get("gb_mass_i", 0.0)) for entry in entries))

    neighbor_count = sum(1 for entry in entries if int(entry.get("neighbor_id", -1)) != -1)
    if neighbor_count <= 0:
        neighbor_count = len(entries)

    total_boundary_area = total_gb_area_f / 2.0
    total_mass = total_swept_mass + total_gb_mass
    denominator = total_swept_area + total_boundary_area
    if denominator > _GBM_PARTITION_TOL and total_boundary_area > _GBM_PARTITION_TOL:
        gb_mass_f = total_mass * total_boundary_area * float(partition_coeff) / denominator
    else:
        gb_mass_f = 0.0
    total_put_mass = float(total_mass - gb_mass_f)

    entry_put_mass: list[float] = []
    if total_swept_area > _GBM_PARTITION_TOL:
        for entry in entries:
            entry_put_mass.append(total_put_mass * float(entry.get("area", 0.0)) / total_swept_area)
    elif neighbor_count > 0:
        equal_share = total_put_mass / float(neighbor_count)
        for entry in entries:
            if int(entry.get("neighbor_id", -1)) != -1:
                entry_put_mass.append(equal_share)
            else:
                entry_put_mass.append(0.0)
    else:
        entry_put_mass = [0.0 for _ in entries]

    entry_gb_mass_f: list[float] = []
    if total_gb_area_f > _GBM_PARTITION_TOL:
        for entry in entries:
            entry_gb_mass_f.append(gb_mass_f * float(entry.get("gb_area_f", 0.0)) / total_gb_area_f)
    elif entries:
        equal_gb_mass = gb_mass_f / float(len(entries))
        entry_gb_mass_f = [equal_gb_mass for _ in entries]

    concentration = 0.0
    if total_boundary_area > _GBM_PARTITION_TOL:
        concentration = gb_mass_f / total_boundary_area

    return {
        "total_swept_mass": float(total_swept_mass),
        "total_swept_area": float(total_swept_area),
        "total_gb_area_i": float(total_gb_area_i),
        "total_gb_area_f": float(total_gb_area_f),
        "total_gb_mass": float(total_gb_mass),
        "total_mass": float(total_mass),
        "gb_mass_f": float(gb_mass_f),
        "total_put_mass": float(total_put_mass),
        "concentration": float(concentration),
        "entry_put_mass": tuple(float(value) for value in entry_put_mass),
        "entry_gb_mass_f": tuple(float(value) for value in entry_gb_mass_f),
    }


def _incremented_segment_swept_records(
    node_id: int,
    neighbor_id: int,
    old_node: np.ndarray,
    new_node: np.ndarray,
    pivot: np.ndarray,
    sample_points: np.ndarray,
    *,
    active_mask: np.ndarray | None = None,
    sample_current: np.ndarray | None = None,
    sample_target: np.ndarray | None = None,
    sweep_label: int | None = None,
    enrich_label: int | None = None,
    old_length: float,
    new_length: float,
    increments: int = _GBM_TRANSFER_INCREMENTS,
) -> list[dict[str, Any]]:
    polygon_points = [
        np.asarray(old_node, dtype=np.float64),
        np.asarray(new_node, dtype=np.float64),
        np.asarray(pivot, dtype=np.float64),
    ]
    sweep_area = abs(_polygon_signed_area_from_points(polygon_points))
    if sweep_area <= 1.0e-12:
        return []

    increments = max(int(increments), 1)
    if increments == 1:
        swept_mask = _mark_seed_points_in_polygon(
            sample_points,
            polygon_points,
            active_mask=active_mask,
        )
        if not np.any(swept_mask):
            return []
        if (
            sample_current is not None
            and sample_target is not None
            and sweep_label is not None
            and enrich_label is not None
        ):
            sweep_membership = swept_mask & (np.asarray(sample_current, dtype=np.int32) == int(sweep_label))
            enrich_membership = swept_mask & (np.asarray(sample_target, dtype=np.int32) == int(enrich_label))
            reassigned_membership = sweep_membership & enrich_membership
        else:
            sweep_membership = swept_mask.copy()
            enrich_membership = swept_mask.copy()
            reassigned_membership = np.zeros_like(swept_mask, dtype=bool)
        return [
            {
                "node_id": int(node_id),
                "neighbor_id": int(neighbor_id),
                "area": float(sweep_area),
                "mask": swept_mask,
                "sweep_mask": sweep_membership,
                "enrich_mask": enrich_membership,
                "reassigned_mask": reassigned_membership,
                "origin": np.asarray(old_node, dtype=np.float64),
                "tip": np.asarray(new_node, dtype=np.float64),
                "old_length": float(old_length),
                "new_length": float(new_length),
                "increment_index": 0,
                "increment_count": 1,
            }
        ]

    end_a = np.asarray(old_node, dtype=np.float64)
    end_b = np.asarray(new_node, dtype=np.float64)
    pivot_np = np.asarray(pivot, dtype=np.float64)
    delta_a = (end_a - pivot_np) / float(increments)
    delta_b = (end_b - pivot_np) / float(increments)

    nested_areas: list[float] = []
    ring_centers: list[tuple[np.ndarray, np.ndarray]] = []
    triangle_masks: list[np.ndarray] = []
    current_a = end_a.copy()
    current_b = end_b.copy()
    for _index in range(increments):
        triangle_points = [
            np.asarray(current_a, dtype=np.float64),
            np.asarray(current_b, dtype=np.float64),
            pivot_np,
        ]
        nested_areas.append(
            abs(
                _polygon_signed_area_from_points(triangle_points)
            )
        )
        triangle_masks.append(
            _mark_seed_points_in_polygon(
                sample_points,
                triangle_points,
                active_mask=active_mask,
            )
        )
        ring_centers.append((current_a - (delta_a * 0.5), current_b - (delta_b * 0.5)))
        current_a = current_a - delta_a
        current_b = current_b - delta_b

    ring_areas: list[float] = []
    ring_masks: list[np.ndarray] = []
    for index in range(increments):
        if index < increments - 1:
            ring_area = max(nested_areas[index] - nested_areas[index + 1], 0.0)
            ring_mask = np.asarray(triangle_masks[index], dtype=bool) & (~np.asarray(triangle_masks[index + 1], dtype=bool))
        else:
            ring_area = max(nested_areas[index], 0.0)
            ring_mask = np.asarray(triangle_masks[index], dtype=bool)
        ring_areas.append(ring_area)
        ring_masks.append(ring_mask)

    total_ring_area = float(sum(ring_areas))
    if total_ring_area <= 1.0e-12:
        return []

    scale = float(sweep_area) / total_ring_area
    records: list[dict[str, Any]] = []
    for index, ((center_a, center_b), ring_area, ring_mask) in enumerate(zip(ring_centers, ring_areas, ring_masks)):
        scaled_area = float(ring_area) * scale
        if scaled_area <= 1.0e-12:
            continue
        if (
            sample_current is not None
            and sample_target is not None
            and sweep_label is not None
            and enrich_label is not None
        ):
            sweep_membership = np.asarray(ring_mask, dtype=bool) & (np.asarray(sample_current, dtype=np.int32) == int(sweep_label))
            enrich_membership = np.asarray(ring_mask, dtype=bool) & (np.asarray(sample_target, dtype=np.int32) == int(enrich_label))
            reassigned_membership = sweep_membership & enrich_membership
        else:
            sweep_membership = np.asarray(ring_mask, dtype=bool)
            enrich_membership = np.asarray(ring_mask, dtype=bool)
            reassigned_membership = np.zeros_like(ring_mask, dtype=bool)
        records.append(
            {
                "node_id": int(node_id),
                "neighbor_id": int(neighbor_id),
                "area": float(scaled_area),
                "mask": np.asarray(ring_mask, dtype=bool),
                "sweep_mask": sweep_membership,
                "enrich_mask": enrich_membership,
                "reassigned_mask": reassigned_membership,
                "origin": np.asarray(center_a, dtype=np.float64),
                "tip": np.asarray(center_b, dtype=np.float64),
                "old_length": float(old_length) / float(increments),
                "new_length": float(new_length) / float(increments),
                "increment_index": int(index),
                "increment_count": int(increments),
            }
        )
    return records


def _apply_segment_mass_partition(
    source_values: np.ndarray,
    updated_values: np.ndarray,
    transfer_records: list[dict[str, Any]],
    *,
    changed_mask: np.ndarray,
    swept_mask: np.ndarray,
    sample_points: np.ndarray,
    unode_area: float,
    roi: float,
    node_values: np.ndarray | None = None,
    boundary_scale: float = 0.0,
    return_ledger: bool = False,
) -> tuple[np.ndarray, float] | tuple[np.ndarray, float, dict[int, dict[str, float]]]:
    adjusted = np.asarray(updated_values, dtype=np.float64).copy()
    source_np = np.asarray(source_values, dtype=np.float64)
    node_aware_partition = node_values is not None and float(boundary_scale) > 0.0
    total_mass_before = float(np.sum(np.asarray(source_values, dtype=np.float64)) * unode_area)
    total_mass_after = float(np.sum(adjusted) * unode_area)
    residual_mass = total_mass_before - total_mass_after
    if (abs(residual_mass) <= 1.0e-12 and not node_aware_partition) or not transfer_records:
        if return_ledger:
            return adjusted, residual_mass, {}
        return adjusted, residual_mass

    if node_aware_partition:
        eligible_records = [
            record
            for record in transfer_records
            if np.any(record["mask"] & (np.asarray(changed_mask, dtype=bool) | np.asarray(swept_mask, dtype=bool)))
        ]
    elif residual_mass > 0.0:
        eligible_records = [
            record for record in transfer_records if np.any(record["mask"] & np.asarray(changed_mask, dtype=bool))
        ]
    else:
        eligible_records = [
            record for record in transfer_records if np.any(record["mask"] & np.asarray(swept_mask, dtype=bool))
        ]
    if not eligible_records:
        if return_ledger:
            return adjusted, residual_mass, {}
        return adjusted, residual_mass

    total_area = float(sum(float(record["area"]) for record in eligible_records))
    if total_area <= 1.0e-12:
        if return_ledger:
            return adjusted, residual_mass, {}
        return adjusted, residual_mass

    segment_entries: list[dict[str, Any]] = []
    for record in eligible_records:
        if node_aware_partition or residual_mass > 0.0:
            if "enrich_mask" in record:
                candidate_mask = np.asarray(record["enrich_mask"], dtype=bool) & np.asarray(changed_mask, dtype=bool)
            else:
                candidate_mask = np.asarray(record["mask"], dtype=bool) & np.asarray(changed_mask, dtype=bool)
        else:
            if "sweep_mask" in record:
                candidate_mask = np.asarray(record["sweep_mask"], dtype=bool) & np.asarray(swept_mask, dtype=bool)
            else:
                candidate_mask = np.asarray(record["mask"], dtype=bool) & np.asarray(swept_mask, dtype=bool)
        candidate_indices = np.flatnonzero(candidate_mask)
        if candidate_indices.size == 0:
            continue

        local_scale = max(float(np.sqrt(record["area"])), 1.0e-6)
        weights = _segment_point_weights(
            sample_points,
            candidate_indices,
            np.asarray(record["origin"], dtype=np.float64),
            point_b=np.asarray(record["tip"], dtype=np.float64),
            segment_area=float(record["area"]),
            roi=roi,
            local_scale=local_scale,
        )
        reassigned_mask = np.asarray(
            record.get("reassigned_mask", np.zeros(sample_points.shape[0], dtype=bool)),
            dtype=bool,
        )
        if candidate_indices.size > 0:
            reassigned_local = reassigned_mask[candidate_indices]
            if np.any(reassigned_local):
                weights = weights.copy()
                weights[reassigned_local] *= 0.5
        if residual_mass < 0.0:
            removable = np.maximum(adjusted[candidate_indices], 0.0) * unode_area
            if float(np.sum(removable)) <= 1.0e-12:
                continue
            weights = weights * removable
        weight_sum = float(np.sum(weights))
        if weight_sum <= 1.0e-12:
            continue

        if "enrich_mask" in record:
            enrich_support_indices = np.flatnonzero(np.asarray(record["enrich_mask"], dtype=bool))
        else:
            enrich_support_indices = np.flatnonzero(np.asarray(record["mask"], dtype=bool))
        enrich_support_weights = np.zeros(0, dtype=np.float64)
        enrich_line_weights = np.zeros(0, dtype=np.float64)
        enrich_support_mass = 0.0
        enrich_support_weight_total = 0.0
        enrich_line_weight_total = 0.0
        if enrich_support_indices.size > 0:
            enrich_support_weights = _segment_point_weights(
                sample_points,
                enrich_support_indices,
                np.asarray(record["origin"], dtype=np.float64),
                point_b=np.asarray(record["tip"], dtype=np.float64),
                segment_area=float(record["area"]),
                roi=roi,
                local_scale=local_scale,
            )
            enrich_line_weights = _segment_line_support_weights(
                sample_points,
                enrich_support_indices,
                np.asarray(record["origin"], dtype=np.float64),
                point_b=np.asarray(record["tip"], dtype=np.float64),
                roi=roi,
                local_scale=local_scale,
            )
            reassigned_support = reassigned_mask[enrich_support_indices]
            if np.any(reassigned_support):
                enrich_support_weights = enrich_support_weights.copy()
                enrich_support_weights[reassigned_support] *= 0.5
                enrich_line_weights = enrich_line_weights.copy()
                enrich_line_weights[reassigned_support] *= 0.5
            enrich_support_weight_total = float(np.sum(enrich_support_weights))
            enrich_line_weight_total = float(np.sum(enrich_line_weights))
            if enrich_support_weight_total > 1.0e-12:
                enrich_support_mass = float(
                    np.sum(np.maximum(source_np[enrich_support_indices], 0.0) * unode_area * enrich_support_weights)
                )
        enrich_support_weight_total_1 = float(enrich_support_weight_total)
        enrich_support_mass_1 = float(enrich_support_mass)
        enrich_line_weight_total_2 = float(enrich_line_weight_total)

        if "sweep_mask" in record:
            swept_indices = np.flatnonzero(np.asarray(record["sweep_mask"], dtype=bool) & np.asarray(swept_mask, dtype=bool))
        else:
            swept_indices = np.flatnonzero(np.asarray(record["mask"], dtype=bool) & np.asarray(swept_mask, dtype=bool))
        swept_weights = np.zeros(0, dtype=np.float64)
        swept_line_weights = np.zeros(0, dtype=np.float64)
        if swept_indices.size > 0:
            swept_weights = _segment_point_weights(
                sample_points,
                swept_indices,
                np.asarray(record["origin"], dtype=np.float64),
                point_b=np.asarray(record["tip"], dtype=np.float64),
                segment_area=float(record["area"]),
                roi=roi,
                local_scale=local_scale,
            )
            swept_line_weights = _segment_line_support_weights(
                sample_points,
                swept_indices,
                np.asarray(record["origin"], dtype=np.float64),
                point_b=np.asarray(record["tip"], dtype=np.float64),
                roi=roi,
                local_scale=local_scale,
            )
            reassigned_local = reassigned_mask[swept_indices]
            if np.any(reassigned_local):
                swept_weights = swept_weights.copy()
                swept_weights[reassigned_local] *= 0.5
                swept_line_weights = swept_line_weights.copy()
                swept_line_weights[reassigned_local] *= 0.5
        sweep_weight_total_0 = float(np.sum(swept_weights)) if swept_weights.size else 0.0
        sweep_weight_total_1 = float(sweep_weight_total_0)
        sweep_line_weight_total_2 = float(np.sum(swept_line_weights)) if swept_line_weights.size else 0.0
        segment_entries.append(
            {
                "node_id": int(record.get("node_id", 0)),
                "neighbor_id": int(record.get("neighbor_id", -1)),
                "candidate_indices": candidate_indices,
                "weights": weights,
                "candidate_weight_total": float(weight_sum),
                "enrich_support_indices": enrich_support_indices,
                "enrich_support_weights_1": enrich_support_weights,
                "enrich_weight_total": float(enrich_support_weight_total),
                "enrich_weight_total_0": float(enrich_support_weight_total),
                "enrich_weight_total_1": float(enrich_support_weight_total_1),
                "enrich_weight_total_2": float(enrich_line_weight_total_2),
                "enrich_line_weight_total": float(enrich_line_weight_total_2),
                "enrich_mass": float(enrich_support_mass),
                "enrich_mass_0": float(enrich_support_mass),
                "enrich_mass_1": float(enrich_support_mass_1),
                "area": float(record["area"]),
                "old_length": float(record.get("old_length", 0.0)),
                "new_length": float(record.get("new_length", 0.0)),
                "swept_indices": swept_indices,
                "swept_weights": swept_weights,
                "sweep_weight_total": float(sweep_weight_total_0),
                "sweep_weight_total_0": float(sweep_weight_total_0),
                "sweep_weight_total_1": float(sweep_weight_total_1),
                "sweep_weight_total_2": float(sweep_line_weight_total_2),
                "sweep_line_weight_total": float(sweep_line_weight_total_2),
                "swept_mass": 0.0,
                "swept_mass_0": 0.0,
                "swept_mass_1": 0.0,
            }
        )

    if not segment_entries:
        if return_ledger:
            return adjusted, residual_mass, {}
        return adjusted, residual_mass

    point_segments: dict[int, list[tuple[dict[str, Any], float]]] = {}
    for entry in segment_entries:
        swept_indices = np.asarray(entry.get("swept_indices", ()), dtype=np.int32)
        swept_weights = np.asarray(entry.get("swept_weights", ()), dtype=np.float64)
        for local_index, point_index in enumerate(swept_indices):
            weight = float(swept_weights[local_index]) if local_index < swept_weights.size else 0.0
            if weight > 0.0:
                point_segments.setdefault(int(point_index), []).append((entry, weight))

    total_swept_mass = 0.0
    for point_index, segment_weights in point_segments.items():
        combined_weight = float(sum(weight for _, weight in segment_weights))
        if combined_weight <= 1.0e-12:
            continue
        removal_fraction = min(combined_weight, 1.0)
        source_mass = float(max(source_np[int(point_index)], 0.0) * unode_area)
        removed_mass = source_mass * removal_fraction
        total_swept_mass += removed_mass

        current_mass = float(max(adjusted[int(point_index)], 0.0) * unode_area)
        actual_removed_mass = min(removed_mass, current_mass)
        if actual_removed_mass > 0.0:
            adjusted[int(point_index)] = max(
                float(adjusted[int(point_index)] - actual_removed_mass / unode_area),
                0.0,
            )

        for entry, weight in segment_weights:
            weighted_removed_mass = removed_mass * weight / combined_weight
            entry["swept_mass"] += weighted_removed_mass
            entry["swept_mass_0"] += weighted_removed_mass
            entry["swept_mass_1"] += weighted_removed_mass

    node_partition: dict[int, dict[str, float]] = {}
    segments_by_node: dict[int, list[dict[str, Any]]] = {}
    for entry in segment_entries:
        segments_by_node.setdefault(int(entry["node_id"]), []).append(entry)

    total_put_mass = 0.0
    for node_id, entries in segments_by_node.items():
        total_swept_mass_node = float(sum(float(entry["swept_mass"]) for entry in entries))
        total_swept_area_node = float(sum(float(entry["area"]) for entry in entries))
        total_enrich_weight_node = float(sum(float(entry.get("enrich_weight_total", 0.0)) for entry in entries))
        total_candidate_weight_node = float(sum(float(entry.get("candidate_weight_total", 0.0)) for entry in entries))
        total_enrich_mass_node = float(sum(float(entry.get("enrich_mass", 0.0)) for entry in entries))
        total_put_mass_node = total_swept_mass_node
        node_concentration = None
        gb_mass_f = 0.0
        total_source_mass_node = total_swept_mass_node
        total_capacity_node = total_swept_area_node
        total_gb_area_i = 0.0
        total_gb_area_f = 0.0

        if (
            node_values is not None
            and int(node_id) < np.asarray(node_values).shape[0]
        ):
            old_node_conc = float(np.asarray(node_values, dtype=np.float64)[int(node_id)])
            for entry in entries:
                entry_gb_area_i = float(entry.get("old_length", 0.0)) * float(boundary_scale) * 0.5
                entry_gb_area_f = float(entry.get("new_length", 0.0)) * float(boundary_scale) * 0.5
                entry_gb_mass_i = old_node_conc * entry_gb_area_i
                entry_source_mass = float(entry.get("swept_mass", 0.0)) + entry_gb_mass_i
                entry_capacity = float(entry.get("area", 0.0)) + entry_gb_area_f
                entry["gb_area_i"] = float(entry_gb_area_i)
                entry["gb_area_f"] = float(entry_gb_area_f)
                entry["gb_mass_i"] = float(entry_gb_mass_i)
                entry["source_mass"] = float(entry_source_mass)
                entry["capacity"] = float(entry_capacity)
                total_gb_area_i += float(entry_gb_area_i)
                total_gb_area_f += float(entry_gb_area_f)
                total_source_mass_node += float(entry_gb_mass_i)
                total_capacity_node += float(entry_gb_area_f)

            if total_capacity_node > 1.0e-12:
                for entry in entries:
                    partition_terms = _entry_partition_terms(
                        entry,
                        unode_area=unode_area,
                    )
                    entry.update(partition_terms)
                    entry_gb_mass_i = float(entry.get("gb_mass_i", 0.0))
                    entry_gb_mass_f = entry_gb_mass_i + float(entry["mass_chge_b"])
                    entry_put_mass = float(entry["mass_chge_e"])
                    entry_total_mass = entry_gb_mass_f + max(entry_put_mass, 0.0)
                    entry["total_mass"] = float(entry_total_mass)
                    entry["gb_mass_f"] = float(entry_gb_mass_f)
                    entry["redistributed_mass"] = float(entry["raw_mass_chge_e"])
                    entry["put_mass"] = float(entry_put_mass)
                    entry["mass_balance_error"] = float(
                        float(entry["mass_chge_s"]) + float(entry["raw_mass_chge_e"]) + float(entry["mass_chge_b"])
                    )
                partition_mass = _partition_mass_node(entries)
                gb_mass_f = float(partition_mass["gb_mass_f"])
                total_put_mass_node = float(partition_mass["total_put_mass"])
                node_concentration = max(float(partition_mass["concentration"]), 0.0)
                partition_entry_put_mass = tuple(partition_mass["entry_put_mass"])
                partition_entry_gb_mass_f = tuple(partition_mass["entry_gb_mass_f"])
                for entry, entry_put_mass, entry_gb_mass_f in zip(
                    entries,
                    partition_entry_put_mass,
                    partition_entry_gb_mass_f,
                ):
                    entry["partitionmass_put_mass"] = float(entry_put_mass)
                    entry["partitionmass_gb_mass_f"] = float(entry_gb_mass_f)
                    entry["put_mass"] = float(entry_put_mass)
                    entry["gb_mass_f"] = float(entry_gb_mass_f)

        if node_concentration is None:
            segment_count = max(len(entries), 1)
            for entry in entries:
                if total_enrich_weight_node > 1.0e-12:
                    entry["put_mass"] = total_put_mass_node * float(entry.get("enrich_weight_total", 0.0)) / total_enrich_weight_node
                elif total_candidate_weight_node > 1.0e-12:
                    entry["put_mass"] = total_put_mass_node * float(entry.get("candidate_weight_total", 0.0)) / total_candidate_weight_node
                elif total_swept_area_node > 1.0e-12:
                    entry["put_mass"] = total_put_mass_node * float(entry["area"]) / total_swept_area_node
                else:
                    entry["put_mass"] = total_put_mass_node / float(segment_count)
                entry["redistributed_mass"] = float(entry["put_mass"])
                entry["mass_chge_s"] = float(-float(entry.get("swept_mass", 0.0)))
                entry["raw_mass_chge_e"] = float(entry.get("put_mass", 0.0) - float(entry.get("enrich_mass", 0.0)))
                entry["mass_chge_e"] = float(entry.get("put_mass", 0.0))
                entry["mass_chge_b"] = 0.0
                entry["mass_balance_error"] = float(entry["mass_chge_s"] + entry["raw_mass_chge_e"])
        total_put_mass += total_put_mass_node
        total_mass_chge_s_node = float(sum(float(entry.get("mass_chge_s", 0.0)) for entry in entries))
        total_mass_chge_e_node = float(sum(float(entry.get("mass_chge_e", 0.0)) for entry in entries))
        total_raw_mass_chge_e_node = float(sum(float(entry.get("raw_mass_chge_e", 0.0)) for entry in entries))
        total_mass_chge_b_node = float(sum(float(entry.get("mass_chge_b", 0.0)) for entry in entries))
        node_partition[int(node_id)] = {
            "put_mass": float(total_put_mass_node),
            "swept_mass": float(total_swept_mass_node),
            "total_swept_area": float(total_swept_area_node),
            "total_enrich_weight": float(total_enrich_weight_node),
            "total_enrich_mass": float(total_enrich_mass_node),
            "total_source_mass": float(total_source_mass_node),
            "total_capacity": float(total_capacity_node),
            "mass_chge_s": float(total_mass_chge_s_node),
            "mass_chge_e": float(total_mass_chge_e_node),
            "raw_mass_chge_e": float(total_raw_mass_chge_e_node),
            "mass_chge_b": float(total_mass_chge_b_node),
            "gb_area_i": float(total_gb_area_i),
            "gb_area_f": float(total_gb_area_f),
            "gb_mass_f": float(gb_mass_f),
            "partitionmass_put_mass": float(total_put_mass_node),
            "partitionmass_gb_mass_f": float(gb_mass_f),
            "concentration": float(node_concentration) if node_concentration is not None else np.nan,
            "partition_mode": "elle_partitionmass" if node_concentration is not None and node_values is not None else "weighted_fallback",
            "entries": tuple(
                {
                    "neighbor_id": int(entry.get("neighbor_id", -1)),
                    "swept_mass": float(entry.get("swept_mass", 0.0)),
                    "put_mass": float(entry.get("put_mass", 0.0)),
                    "redistributed_mass": float(entry.get("redistributed_mass", entry.get("put_mass", 0.0))),
                    "source_mass": float(entry.get("source_mass", entry.get("swept_mass", 0.0))),
                    "capacity": float(entry.get("capacity", entry.get("area", 0.0))),
                    "gb_area_i": float(entry.get("gb_area_i", 0.0)),
                    "gb_area_f": float(entry.get("gb_area_f", 0.0)),
                    "gb_mass_i": float(entry.get("gb_mass_i", 0.0)),
                    "gb_mass_f": float(entry.get("gb_mass_f", 0.0)),
                    "partitionmass_put_mass": float(entry.get("partitionmass_put_mass", entry.get("put_mass", 0.0))),
                    "partitionmass_gb_mass_f": float(entry.get("partitionmass_gb_mass_f", entry.get("gb_mass_f", 0.0))),
                    "conc_b": float(entry.get("conc_b", 0.0)),
                    "conc_s": float(entry.get("conc_s", 0.0)),
                    "conc_e": float(entry.get("conc_e", 0.0)),
                    "conc_s1": float(entry.get("conc_s1", 0.0)),
                    "conc_e1": float(entry.get("conc_e1", 0.0)),
                    "conc_b_f": float(entry.get("conc_b_f", 0.0)),
                    "conc_s_f": float(entry.get("conc_s_f", 0.0)),
                    "conc_e_f": float(entry.get("conc_e_f", 0.0)),
                    "overlap_frac": float(entry.get("overlap_frac", 0.0)),
                    "swept_area_frac": float(entry.get("swept_area_frac", 0.0)),
                    "enrich_area_frac": float(entry.get("enrich_area_frac", 0.0)),
                    "sweep_line_weight_total": float(entry.get("sweep_line_weight_total", 0.0)),
                    "enrich_line_weight_total": float(entry.get("enrich_line_weight_total", 0.0)),
                    "sweep_weight_total_0": float(entry.get("sweep_weight_total_0", entry.get("sweep_weight_total", 0.0))),
                    "sweep_weight_total_1": float(entry.get("sweep_weight_total_1", entry.get("sweep_weight_total", 0.0))),
                    "sweep_weight_total_2": float(entry.get("sweep_weight_total_2", entry.get("sweep_line_weight_total", 0.0))),
                    "enrich_weight_total_0": float(entry.get("enrich_weight_total_0", entry.get("enrich_weight_total", 0.0))),
                    "enrich_weight_total_1": float(entry.get("enrich_weight_total_1", entry.get("enrich_weight_total", 0.0))),
                    "enrich_weight_total_2": float(entry.get("enrich_weight_total_2", entry.get("enrich_line_weight_total", 0.0))),
                    "partition_active": bool(entry.get("partition_active", False)),
                    "mass_chge_s": float(entry.get("mass_chge_s", 0.0)),
                    "raw_mass_chge_e": float(entry.get("raw_mass_chge_e", 0.0)),
                    "mass_chge_e": float(entry.get("mass_chge_e", 0.0)),
                    "mass_chge_b": float(entry.get("mass_chge_b", 0.0)),
                    "mass_balance_error": float(entry.get("mass_balance_error", 0.0)),
                }
                for entry in entries
            ),
        }

    for entry in segment_entries:
        enrich_support_indices = np.asarray(entry.get("enrich_support_indices", ()), dtype=np.int32)
        enrich_support_weights_1 = np.asarray(entry.get("enrich_support_weights_1", ()), dtype=np.float64)
        enrich_weight_sum_1 = float(np.sum(enrich_support_weights_1))
        candidate_indices = np.asarray(entry["candidate_indices"], dtype=np.int32)
        weights = np.asarray(entry["weights"], dtype=np.float64)
        weight_sum = float(np.sum(weights))
        applied_mass = float(entry.get("put_mass", 0.0))
        if abs(applied_mass) > 1.0e-12:
            if enrich_support_indices.size and enrich_weight_sum_1 > 1.0e-12:
                adjusted[enrich_support_indices] += applied_mass * (enrich_support_weights_1 / enrich_weight_sum_1) / unode_area
                adjusted[enrich_support_indices] = np.maximum(adjusted[enrich_support_indices], 0.0)
            elif candidate_indices.size and weight_sum > 1.0e-12:
                adjusted[candidate_indices] += applied_mass * (weights / weight_sum) / unode_area
                adjusted[candidate_indices] = np.maximum(adjusted[candidate_indices], 0.0)

    target_total_mass = float(total_mass_before - total_swept_mass + total_put_mass)
    final_residual = float(target_total_mass - float(np.sum(adjusted) * unode_area))
    if return_ledger:
        return adjusted, final_residual, node_partition
    return adjusted, final_residual


def incremental_seed_unode_reassignment(
    current_labels: np.ndarray,
    target_labels: np.ndarray,
    base_mesh_state: dict[str, Any],
    moved_mesh_state: dict[str, Any],
    seed_unodes: dict[str, Any],
) -> tuple[np.ndarray, dict[str, int]]:
    updated_labels = np.asarray(current_labels, dtype=np.int32).copy()
    target_np = np.asarray(target_labels, dtype=np.int32)
    if updated_labels.shape != target_np.shape:
        raise ValueError("current_labels and target_labels must have the same shape")

    base_nodes = np.array([[node["x"], node["y"]] for node in base_mesh_state["nodes"]], dtype=np.float64)
    moved_nodes = np.array([[node["x"], node["y"]] for node in moved_mesh_state["nodes"]], dtype=np.float64)
    if base_nodes.shape != moved_nodes.shape:
        return target_np.copy(), {
            "swept_unodes": int(updated_labels.size),
            "changed_unodes": int(np.count_nonzero(target_np != updated_labels)),
        }

    sample_grid_indices = np.asarray(seed_unodes["grid_indices"], dtype=np.int32)
    active_seed_mask = _compute_swept_seed_unode_mask(base_mesh_state, moved_mesh_state, seed_unodes)

    if np.any(active_seed_mask):
        active_grid = sample_grid_indices[active_seed_mask]
        updated_labels[active_grid[:, 0], active_grid[:, 1]] = target_np[
            active_grid[:, 0],
            active_grid[:, 1],
        ]

    changed_unodes = 0
    if np.any(active_seed_mask):
        active_grid = sample_grid_indices[active_seed_mask]
        changed_unodes = int(
            np.count_nonzero(
                current_labels[active_grid[:, 0], active_grid[:, 1]]
                != updated_labels[active_grid[:, 0], active_grid[:, 1]]
            )
        )

    return updated_labels, {
        "swept_unodes": int(active_seed_mask.sum()),
        "changed_unodes": int(changed_unodes),
    }


def update_seed_unode_fields(
    current_labels: np.ndarray,
    target_labels: np.ndarray,
    base_mesh_state: dict[str, Any],
    moved_mesh_state: dict[str, Any],
    seed_unodes: dict[str, Any],
    seed_fields: dict[str, Any],
    node_fields: dict[str, Any] | None = None,
) -> tuple[dict[str, tuple[float, ...]], dict[str, int], dict[str, dict[int, dict[str, float]]]]:
    sample_points = np.asarray(seed_unodes["positions"], dtype=np.float64)
    sample_grid_indices = np.asarray(seed_unodes["grid_indices"], dtype=np.int32)
    current_np = np.asarray(current_labels, dtype=np.int32)
    target_np = np.asarray(target_labels, dtype=np.int32)
    sample_current = current_np[sample_grid_indices[:, 0], sample_grid_indices[:, 1]]
    sample_target = target_np[sample_grid_indices[:, 0], sample_grid_indices[:, 1]]
    swept_mask = _compute_swept_seed_unode_mask(base_mesh_state, moved_mesh_state, seed_unodes)
    changed_mask = swept_mask & (sample_current != sample_target)

    updated_fields: dict[str, tuple[float, ...]] = {}
    changed_field_values = 0
    mass_conserved_fields = 0
    mass_partitioned_fields = 0
    total_mass_residual = 0.0
    mass_ledgers: dict[str, dict[int, dict[str, float]]] = {}
    label_attribute = str(seed_fields.get("label_attribute", "U_ATTRIB_A"))
    source_labels = [int(value) for value in seed_fields.get("source_labels", ())]
    runtime_values = dict(seed_fields.get("values", {}))
    roi = float(seed_fields.get("roi", _estimate_seed_unode_roi(seed_unodes)))
    unode_area = float(seed_fields.get("unode_area", _estimate_seed_unode_area(seed_unodes)))
    boundary_scale = float(_boundary_area_scale(base_mesh_state, seed_unodes))
    transfer_records = _segment_swept_records(
        base_mesh_state,
        moved_mesh_state,
        sample_points,
        active_mask=swept_mask,
        sample_current=sample_current,
        sample_target=sample_target,
    )

    for field_name, field_values in runtime_values.items():
        current_values = np.asarray(field_values, dtype=np.float64).copy()
        if field_name == label_attribute and source_labels:
            mapped = np.array([source_labels[int(label)] for label in sample_target], dtype=np.float64)
            changed_field_values += int(np.count_nonzero(current_values != mapped))
            current_values = mapped
            updated_fields[field_name] = tuple(float(value) for value in current_values)
            continue

        for point_index in np.flatnonzero(changed_mask):
            target_label = int(sample_target[int(point_index)])
            donor_mask = (sample_target == target_label) & (~changed_mask)
            donor_value = _estimate_seed_field_value(
                sample_points,
                current_values,
                donor_mask,
                int(point_index),
                roi=roi,
            )
            if donor_value is None:
                donor_mask = (sample_target == target_label) & (np.arange(sample_target.size) != int(point_index))
                donor_value = _estimate_seed_field_value(
                    sample_points,
                    current_values,
                    donor_mask,
                    int(point_index),
                    roi=roi,
                )
            if donor_value is None:
                continue
            if abs(float(current_values[int(point_index)]) - float(donor_value)) > 1.0e-12:
                changed_field_values += 1
            current_values[int(point_index)] = float(donor_value)

        if _is_mass_like_unode_field(field_name):
            matching_node_field = _matching_node_field_name(field_name)
            node_values = None
            node_aware_partition = False
            if node_fields is not None and matching_node_field is not None:
                node_values_map = dict(node_fields.get("values", {}))
                if matching_node_field in node_values_map:
                    node_values = np.asarray(node_values_map[matching_node_field], dtype=np.float64)
                    node_aware_partition = True
            if transfer_records:
                current_values, target_residual, node_partition = _apply_segment_mass_partition(
                    np.asarray(field_values, dtype=np.float64),
                    current_values,
                    transfer_records,
                    changed_mask=changed_mask,
                    swept_mask=swept_mask,
                    sample_points=sample_points,
                    unode_area=unode_area,
                    roi=roi,
                    node_values=node_values,
                    boundary_scale=boundary_scale,
                    return_ledger=True,
                )
                mass_ledgers[str(field_name)] = node_partition
                mass_partitioned_fields += 1
            else:
                target_residual = 0.0
                mass_ledgers[str(field_name)] = {}
            adjustment_mask = changed_mask if np.any(changed_mask) else swept_mask
            if node_aware_partition:
                target_total_mass = float(np.sum(current_values) * unode_area + target_residual)
                current_values, residual_mass = _apply_mass_target(
                    current_values,
                    target_total_mass,
                    adjustment_mask,
                    unode_area=unode_area,
                )
            else:
                current_values, residual_mass = _apply_mass_conservation(
                    np.asarray(field_values, dtype=np.float64),
                    current_values,
                    adjustment_mask,
                    unode_area=unode_area,
                )
            total_mass_residual += float(residual_mass)
            mass_conserved_fields += 1

        updated_fields[field_name] = tuple(float(value) for value in current_values)

    return updated_fields, {
        "updated_scalar_unodes": int(changed_field_values),
        "scalar_swept_unodes": int(swept_mask.sum()),
        "mass_conserved_fields": int(mass_conserved_fields),
        "mass_partitioned_fields": int(mass_partitioned_fields),
        "scalar_mass_residual": float(total_mass_residual),
    }, mass_ledgers


def _matching_unode_field_name(node_field_name: str) -> str | None:
    if node_field_name.startswith("N_"):
        return f"U{node_field_name[1:]}"
    return None


def _matching_node_field_name(unode_field_name: str) -> str | None:
    if unode_field_name.startswith("U_"):
        return f"N{unode_field_name[1:]}"
    return None


def _is_mass_like_node_field(field_name: str) -> bool:
    return str(field_name).startswith("N_CONC_")


def _boundary_area_scale(mesh_state: dict[str, Any], seed_unodes: dict[str, Any]) -> float:
    stats = dict(mesh_state.get("stats", {}))
    boundary_width = float(stats.get("elle_option_boundarywidth", 0.0))
    unit_length = float(stats.get("elle_option_unitlength", 0.0))
    if boundary_width > 0.0 and unit_length > 0.0:
        return boundary_width / unit_length
    return _estimate_seed_unode_area(seed_unodes) ** 0.5


def _node_index_mapping(previous_positions: np.ndarray, current_positions: np.ndarray) -> dict[int, int]:
    mapping: dict[int, int] = {}
    if previous_positions.shape[0] == current_positions.shape[0]:
        return {int(index): int(index) for index in range(current_positions.shape[0])}
    if previous_positions.size == 0 or current_positions.size == 0:
        return mapping
    for current_index, point in enumerate(current_positions):
        deltas = _periodic_relative(previous_positions, point)
        distances_sq = np.sum(deltas * deltas, axis=1)
        mapping[int(current_index)] = int(np.argmin(distances_sq))
    return mapping


def _segment_swept_mask(
    old_node: np.ndarray,
    new_node: np.ndarray,
    old_neighbor: np.ndarray,
    sample_points: np.ndarray,
) -> np.ndarray:
    plotted_new = _plot_xy_polygon(new_node, old_node)
    plotted_neighbor = _plot_xy_polygon(old_neighbor, old_node)
    polygon_points = [
        np.asarray(old_node, dtype=np.float64),
        np.asarray(plotted_new, dtype=np.float64),
        np.asarray(plotted_neighbor, dtype=np.float64),
    ]
    return _mark_seed_points_in_polygon(sample_points, polygon_points)


def _segment_swept_records(
    base_mesh_state: dict[str, Any],
    moved_mesh_state: dict[str, Any],
    sample_points: np.ndarray,
    *,
    active_mask: np.ndarray | None = None,
    sample_current: np.ndarray | None = None,
    sample_target: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    base_nodes = np.array([[node["x"], node["y"]] for node in base_mesh_state["nodes"]], dtype=np.float64)
    moved_nodes = np.array([[node["x"], node["y"]] for node in moved_mesh_state["nodes"]], dtype=np.float64)
    if base_nodes.shape != moved_nodes.shape:
        return []

    node_neighbors = _node_neighbors(base_mesh_state["flynns"])
    records: list[dict[str, Any]] = []
    for node_id, neighbors in node_neighbors.items():
        old_node = base_nodes[int(node_id)]
        new_node = _plot_xy_polygon(moved_nodes[int(node_id)], old_node)
        if float(np.hypot(*(new_node - old_node))) <= 1e-9:
            continue
        for neighbor_id in sorted(int(value) for value in neighbors):
            old_neighbor = _plot_xy_polygon(base_nodes[int(neighbor_id)], old_node)
            new_neighbor = _plot_xy_polygon(moved_nodes[int(neighbor_id)], new_node)
            polygon_points = [
                np.asarray(old_node, dtype=np.float64),
                np.asarray(new_node, dtype=np.float64),
                np.asarray(old_neighbor, dtype=np.float64),
            ]
            area = abs(_polygon_signed_area_from_points(polygon_points))
            if area <= 1.0e-12:
                continue
            full_triangle_mask = _mark_seed_points_in_polygon(
                sample_points,
                polygon_points,
                active_mask=active_mask,
            )
            if not np.any(full_triangle_mask):
                continue
            sweep_label = None
            enrich_label = None
            if sample_current is not None and sample_target is not None:
                changed_indices = np.flatnonzero(
                    np.asarray(full_triangle_mask, dtype=bool)
                    & (np.asarray(sample_current, dtype=np.int32) != np.asarray(sample_target, dtype=np.int32))
                )
                if changed_indices.size > 0:
                    changed_current = np.asarray(sample_current, dtype=np.int32)[changed_indices]
                    changed_target = np.asarray(sample_target, dtype=np.int32)[changed_indices]
                    current_labels, current_counts = np.unique(changed_current, return_counts=True)
                    target_labels, target_counts = np.unique(changed_target, return_counts=True)
                    sweep_label = int(current_labels[int(np.argmax(current_counts))])
                    enrich_label = int(target_labels[int(np.argmax(target_counts))])
            records.extend(
                _incremented_segment_swept_records(
                    int(node_id),
                    int(neighbor_id),
                    np.asarray(old_node, dtype=np.float64),
                    np.asarray(new_node, dtype=np.float64),
                    np.asarray(old_neighbor, dtype=np.float64),
                    sample_points,
                    active_mask=active_mask,
                    sample_current=sample_current,
                    sample_target=sample_target,
                    sweep_label=sweep_label,
                    enrich_label=enrich_label,
                    old_length=float(np.hypot(*(old_neighbor - old_node))),
                    new_length=float(np.hypot(*(new_neighbor - new_node))),
                )
            )
    return records


def _partitioned_node_concentration(
    node_index: int,
    node: dict[str, Any],
    *,
    previous_positions: np.ndarray,
    current_positions: np.ndarray,
    previous_values: np.ndarray,
    sample_points: np.ndarray,
    donor_values: np.ndarray,
    unode_area: float,
    boundary_scale: float,
    node_mapping: dict[int, int],
) -> float | None:
    previous_index = node_mapping.get(int(node_index))
    if previous_index is None or previous_index >= previous_positions.shape[0] or previous_index >= previous_values.shape[0]:
        return None

    old_node = np.asarray(previous_positions[int(previous_index)], dtype=np.float64)
    new_node = np.asarray(current_positions[int(node_index)], dtype=np.float64)
    old_conc = float(previous_values[int(previous_index)])
    total_swept_mass = 0.0
    total_swept_area = 0.0
    total_gb_area_f = 0.0
    total_gb_mass = 0.0
    neighbor_count = 0

    for neighbor_id in node.get("neighbors", []):
        neighbor_id = int(neighbor_id)
        previous_neighbor_index = node_mapping.get(neighbor_id)
        if (
            previous_neighbor_index is None
            or previous_neighbor_index >= previous_positions.shape[0]
            or neighbor_id >= current_positions.shape[0]
        ):
            continue
        old_neighbor = np.asarray(previous_positions[int(previous_neighbor_index)], dtype=np.float64)
        new_neighbor = np.asarray(current_positions[int(neighbor_id)], dtype=np.float64)
        swept_mask = _segment_swept_mask(old_node, new_node, old_neighbor, sample_points)
        swept_area = abs(
            _polygon_signed_area_from_points(
                [
                    np.asarray(old_node, dtype=np.float64),
                    _plot_xy_polygon(new_node, old_node),
                    _plot_xy_polygon(old_neighbor, old_node),
                ]
            )
        )
        old_length = float(np.hypot(*_periodic_relative(old_neighbor, old_node)))
        new_length = float(np.hypot(*_periodic_relative(new_neighbor, new_node)))
        gb_area_i = old_length * boundary_scale
        gb_area_f = new_length * boundary_scale
        swept_mass = float(np.sum(donor_values[swept_mask]) * unode_area)

        total_swept_mass += swept_mass
        total_swept_area += swept_area
        total_gb_area_f += gb_area_f
        total_gb_mass += old_conc * gb_area_i * 0.5
        neighbor_count += 1

    if neighbor_count == 0:
        return None

    denom = total_swept_area + (total_gb_area_f * 0.5)
    if denom <= 1.0e-12 or total_gb_area_f <= 1.0e-12:
        return old_conc

    gb_mass_f = (total_swept_mass + total_gb_mass) * (total_gb_area_f * 0.5) / denom
    return float(max(gb_mass_f / (total_gb_area_f * 0.5), 0.0))


def update_seed_node_fields(
    mesh_state: dict[str, Any],
    seed_unodes: dict[str, Any],
    target_labels: np.ndarray,
    node_fields: dict[str, Any],
    unode_fields: dict[str, tuple[float, ...]],
    mass_ledgers: dict[str, dict[int, dict[str, float]]] | None = None,
) -> tuple[dict[str, tuple[float, ...]], dict[str, int]]:
    sample_points = np.asarray(seed_unodes["positions"], dtype=np.float64)
    sample_grid_indices = np.asarray(seed_unodes["grid_indices"], dtype=np.int32)
    target_np = np.asarray(target_labels, dtype=np.int32)
    sample_target = target_np[sample_grid_indices[:, 0], sample_grid_indices[:, 1]]
    roi = float(
        node_fields.get(
            "roi",
            _estimate_seed_unode_roi(seed_unodes),
        )
    )

    flynn_label_by_id = {
        int(flynn["flynn_id"]): int(flynn.get("label", flynn["flynn_id"]))
        for flynn in mesh_state.get("flynns", [])
    }
    previous_values = {
        str(name): np.asarray(values, dtype=np.float64)
        for name, values in dict(node_fields.get("values", {})).items()
    }
    previous_positions = np.asarray(node_fields.get("positions", ()), dtype=np.float64)
    updated_node_fields: dict[str, tuple[float, ...]] = {}
    changed_node_values = 0
    partitioned_node_fields = 0

    current_node_positions = np.asarray(
        [[float(node["x"]), float(node["y"])] for node in mesh_state.get("nodes", [])],
        dtype=np.float64,
    )
    node_mapping = _node_index_mapping(previous_positions, current_node_positions)
    unode_area = float(_estimate_seed_unode_area(seed_unodes))
    boundary_scale = float(_boundary_area_scale(mesh_state, seed_unodes))

    for field_name in node_fields.get("field_order", previous_values.keys()):
        field_name = str(field_name)
        field_values = previous_values.get(field_name)
        if field_values is None:
            continue
        matching_unode_field = _matching_unode_field_name(field_name)
        donor_values = None
        if matching_unode_field is not None and matching_unode_field in unode_fields:
            donor_values = np.asarray(unode_fields[matching_unode_field], dtype=np.float64)

        current_values: list[float] = []
        for node_index, node in enumerate(mesh_state.get("nodes", [])):
            node_point = current_node_positions[int(node_index)]
            adjacent_labels = [
                flynn_label_by_id.get(int(flynn_id), int(flynn_id))
                for flynn_id in node.get("flynns", [])
            ]
            donor_mask = (
                np.isin(sample_target, np.asarray(adjacent_labels, dtype=np.int32))
                if adjacent_labels
                else np.zeros(sample_target.shape, dtype=bool)
            )
            estimated_value = None
            if donor_values is not None and _is_mass_like_node_field(field_name):
                matching_unode_field = _matching_unode_field_name(field_name)
                ledger = (
                    dict(mass_ledgers.get(matching_unode_field, {}))
                    if mass_ledgers is not None and matching_unode_field is not None
                    else {}
                )
                ledger_entry = ledger.get(int(node_index))
                if ledger_entry is not None and not np.isnan(float(ledger_entry.get("concentration", np.nan))):
                    estimated_value = float(ledger_entry["concentration"])
                else:
                    estimated_value = _partitioned_node_concentration(
                        int(node_index),
                        node,
                        previous_positions=previous_positions,
                        current_positions=current_node_positions,
                        previous_values=field_values,
                        sample_points=sample_points,
                        donor_values=donor_values,
                        unode_area=unode_area,
                        boundary_scale=boundary_scale,
                        node_mapping=node_mapping,
                    )
                if estimated_value is not None:
                    partitioned_node_fields = 1

            if donor_values is not None and estimated_value is None:
                estimated_value = _estimate_field_value_from_samples(
                    sample_points,
                    donor_values,
                    donor_mask,
                    node_point,
                    roi=roi,
                )
                if estimated_value is None:
                    estimated_value = _estimate_field_value_from_samples(
                        sample_points,
                        donor_values,
                        np.ones(sample_target.shape, dtype=bool),
                        node_point,
                        roi=roi,
                    )

            if estimated_value is None and previous_positions.size and field_values.size:
                estimated_value = _estimate_field_value_from_samples(
                    previous_positions,
                    field_values,
                    np.ones(field_values.shape, dtype=bool),
                    node_point,
                    roi=roi,
                )

            if estimated_value is None:
                estimated_value = float(field_values[min(node_index, field_values.size - 1)])
            current_values.append(float(estimated_value))

        current_array = np.asarray(current_values, dtype=np.float64)
        baseline = field_values[: current_array.size]
        changed_node_values += int(
            np.count_nonzero(np.abs(current_array[: baseline.size] - baseline) > 1.0e-12)
        )
        updated_node_fields[field_name] = tuple(float(value) for value in current_array)

    return updated_node_fields, {
        "updated_node_values": int(changed_node_values),
        "partitioned_node_fields": int(partitioned_node_fields),
    }


def _sample_order_parameters(phi: np.ndarray, source_x: np.ndarray, source_y: np.ndarray) -> np.ndarray:
    nx, ny = phi.shape[1], phi.shape[2]
    grid_x = (np.asarray(source_x, dtype=np.float64) * float(nx) - 0.5) % float(nx)
    grid_y = ((1.0 - np.asarray(source_y, dtype=np.float64)) * float(ny) - 0.5) % float(ny)

    x0 = np.floor(grid_x).astype(np.int32)
    y0 = np.floor(grid_y).astype(np.int32)
    x1 = (x0 + 1) % nx
    y1 = (y0 + 1) % ny
    tx = (grid_x - x0).astype(np.float32)
    ty = (grid_y - y0).astype(np.float32)

    q00 = phi[:, x0, y0]
    q10 = phi[:, x1, y0]
    q01 = phi[:, x0, y1]
    q11 = phi[:, x1, y1]

    wx0 = (1.0 - tx)[None, :, :]
    wx1 = tx[None, :, :]
    wy0 = (1.0 - ty)[None, :, :]
    wy1 = ty[None, :, :]
    return q00 * wx0 * wy0 + q10 * wx1 * wy0 + q01 * wx0 * wy1 + q11 * wx1 * wy1


def _node_displacements(before_nodes: np.ndarray, after_nodes: np.ndarray) -> np.ndarray:
    return np.asarray(
        [_periodic_relative(after_nodes[index], before_nodes[index]) for index in range(len(before_nodes))],
        dtype=np.float32,
    )


def mesh_motion_field(
    base_mesh_state: dict[str, Any],
    moved_mesh_state: dict[str, Any],
    grid_shape: tuple[int, int] | list[int],
    *,
    active_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    nx, ny = int(grid_shape[0]), int(grid_shape[1])
    before_nodes = np.array([[node["x"], node["y"]] for node in base_mesh_state["nodes"]], dtype=np.float64)
    after_nodes = np.array([[node["x"], node["y"]] for node in moved_mesh_state["nodes"]], dtype=np.float64)
    velocity = np.zeros((nx, ny, 2), dtype=np.float32)

    if len(before_nodes) == 0 or before_nodes.shape != after_nodes.shape:
        return velocity, {"transport_pixels": 0, "max_displacement": 0.0, "mean_displacement": 0.0}

    if active_mask is None:
        active_mask = np.ones((nx, ny), dtype=bool)
    else:
        active_mask = np.asarray(active_mask, dtype=bool)

    if not np.any(active_mask):
        return velocity, {"transport_pixels": 0, "max_displacement": 0.0, "mean_displacement": 0.0}

    displacements = _node_displacements(before_nodes, after_nodes)
    displacement_norm = np.linalg.norm(displacements, axis=1)
    sample_x, sample_y = _grid_sample_coordinates((nx, ny))
    sample_points = np.stack([sample_x[active_mask], sample_y[active_mask]], axis=1)
    delta = sample_points[:, None, :] - before_nodes[None, :, :]
    delta = (delta + 0.5) % 1.0 - 0.5
    distance_sq = np.sum(delta * delta, axis=2)

    neighbor_count = min(8, len(before_nodes))
    nearest = np.argpartition(distance_sq, kth=neighbor_count - 1, axis=1)[:, :neighbor_count]
    nearest_dist = np.take_along_axis(distance_sq, nearest, axis=1)
    nearest_disp = displacements[nearest]
    base_scale = (0.5 / float(max(nx, ny))) ** 2
    weights = 1.0 / (nearest_dist + base_scale)
    weighted_disp = (nearest_disp * weights[..., None]).sum(axis=1) / weights.sum(axis=1, keepdims=True)
    velocity[active_mask] = weighted_disp.astype(np.float32)

    return velocity, {
        "transport_pixels": int(active_mask.sum()),
        "max_displacement": float(displacement_norm.max()) if displacement_norm.size else 0.0,
        "mean_displacement": float(displacement_norm.mean()) if displacement_norm.size else 0.0,
    }


def _apply_transport_field(
    phi_np: np.ndarray,
    velocity_field: np.ndarray,
    transport_mask: np.ndarray,
    *,
    strength: float,
) -> np.ndarray:
    clamped_strength = float(max(strength, 0.0))
    if clamped_strength <= 0.0 or not np.any(transport_mask):
        return phi_np

    sample_x, sample_y = _grid_sample_coordinates(transport_mask.shape)
    source_x = (sample_x - clamped_strength * velocity_field[:, :, 0]) % 1.0
    source_y = (sample_y - clamped_strength * velocity_field[:, :, 1]) % 1.0
    advected = _sample_order_parameters(phi_np, source_x, source_y)
    transported = np.where(transport_mask[None, :, :], advected, phi_np)
    total = transported.sum(axis=0, keepdims=True) + 1e-12
    return transported / total


def compute_mesh_motion_velocity(
    phi,
    feedback_config: MeshFeedbackConfig,
    *,
    tracked_topology: dict[str, Any] | None = None,
    base_mesh_state: dict[str, Any] | None = None,
    compute_velocity_field: bool = True,
) -> tuple[dict[str, Any], dict[str, Any], np.ndarray, np.ndarray, dict[str, Any]]:
    phi_np = np.asarray(phi, dtype=np.float32)
    if phi_np.ndim != 3:
        raise ValueError("expected phi to have shape (num_grains, nx, ny)")

    if base_mesh_state is None:
        base_mesh_state = build_mesh_state(phi_np, tracked_topology=tracked_topology)
    else:
        base_mesh_state = copy.deepcopy(base_mesh_state)
    motion_config = MeshRelaxationConfig(
        steps=feedback_config.relax_config.steps,
        topology_steps=0,
        speed_up=feedback_config.relax_config.speed_up,
        switch_distance=feedback_config.relax_config.switch_distance,
        movement_model=feedback_config.relax_config.movement_model,
        boundary_energy=feedback_config.relax_config.boundary_energy,
        random_seed=feedback_config.relax_config.random_seed,
        min_node_separation_factor=feedback_config.relax_config.min_node_separation_factor,
        max_node_separation_factor=feedback_config.relax_config.max_node_separation_factor,
    )
    motion_mesh_state = (
        relax_mesh_state(base_mesh_state, motion_config)
        if int(motion_config.steps) > 0
        else base_mesh_state
    )
    if not compute_velocity_field:
        empty_mask = np.zeros(phi_np.shape[1:], dtype=bool)
        zero_velocity = np.zeros((*phi_np.shape[1:], 2), dtype=np.float32)
        return base_mesh_state, motion_mesh_state, zero_velocity, empty_mask, {
            "transport_pixels": 0,
            "max_displacement": 0.0,
            "mean_displacement": 0.0,
            "velocity_field_skipped": True,
        }
    labels = dominant_grain_map(phi_np)
    transport_mask = _dilate_mask(boundary_mask(labels), feedback_config.boundary_width)
    velocity_field, transport_stats = mesh_motion_field(
        base_mesh_state,
        motion_mesh_state,
        labels.shape,
        active_mask=transport_mask,
    )
    return base_mesh_state, motion_mesh_state, velocity_field, transport_mask, transport_stats


def apply_mesh_transport(
    phi,
    base_mesh_state: dict[str, Any],
    moved_mesh_state: dict[str, Any],
    *,
    strength: float,
    boundary_width: int = 1,
) -> tuple[np.ndarray, dict[str, Any]]:
    phi_np = np.asarray(phi, dtype=np.float32)
    if phi_np.ndim != 3:
        raise ValueError("expected phi to have shape (num_grains, nx, ny)")

    clamped_strength = float(max(strength, 0.0))
    labels = dominant_grain_map(phi_np)
    transport_mask = _dilate_mask(boundary_mask(labels), boundary_width)
    velocity_field, transport_stats = mesh_motion_field(
        base_mesh_state,
        moved_mesh_state,
        labels.shape,
        active_mask=transport_mask,
    )

    transport_stats["strength"] = clamped_strength
    if clamped_strength <= 0.0 or not np.any(transport_mask):
        return phi_np, transport_stats

    return _apply_transport_field(
        phi_np,
        velocity_field,
        transport_mask,
        strength=clamped_strength,
    ), transport_stats


def apply_mesh_feedback(
    phi,
    mesh_state: dict[str, Any],
    *,
    strength: float,
    boundary_width: int = 1,
) -> tuple[np.ndarray, dict[str, Any]]:
    phi_np = np.asarray(phi, dtype=np.float32)
    if phi_np.ndim != 3:
        raise ValueError("expected phi to have shape (num_grains, nx, ny)")

    clamped_strength = float(np.clip(strength, 0.0, 1.0))
    current_labels = dominant_grain_map(phi_np)
    mesh_labels = rasterize_mesh_labels(mesh_state, current_labels.shape, fallback_labels=current_labels)
    changed = mesh_labels != current_labels
    changed_band = _dilate_mask(changed, boundary_width)
    feedback_mask = changed_band & (boundary_mask(mesh_labels) | boundary_mask(current_labels) | changed_band)

    stats = {
        "changed_pixels": int(changed.sum()),
        "feedback_pixels": int(feedback_mask.sum()),
        "strength": clamped_strength,
    }
    if clamped_strength <= 0.0 or not np.any(feedback_mask):
        return phi_np, stats

    target = np.moveaxis(np.eye(phi_np.shape[0], dtype=np.float32)[mesh_labels], -1, 0)
    blended = np.where(
        feedback_mask[None, :, :],
        (1.0 - clamped_strength) * phi_np + clamped_strength * target,
        phi_np,
    )
    total = blended.sum(axis=0, keepdims=True) + 1e-12
    return blended / total, stats


def compute_mesh_motion_target(
    phi,
    feedback_config: MeshFeedbackConfig,
    *,
    tracked_topology: dict[str, Any] | None = None,
    base_mesh_state: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any], dict[str, Any]]:
    phi_np = np.asarray(phi, dtype=np.float32)
    (
        base_mesh_state,
        motion_mesh_state,
        velocity_field,
        transport_mask,
        transport_stats,
    ) = compute_mesh_motion_velocity(
        phi_np,
        feedback_config,
        tracked_topology=tracked_topology,
        base_mesh_state=base_mesh_state,
    )
    transport_stats["strength"] = float(max(feedback_config.transport_strength, 0.0))
    phi_motion = _apply_transport_field(
        phi_np,
        velocity_field,
        transport_mask,
        strength=feedback_config.transport_strength,
    )
    return phi_motion, base_mesh_state, motion_mesh_state, transport_stats


def couple_mesh_to_order_parameters(
    phi,
    feedback_config: MeshFeedbackConfig,
    *,
    tracked_topology: dict[str, Any] | None = None,
    base_mesh_state: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    phi_np = np.asarray(phi, dtype=np.float32)
    mesh_only_update = str(feedback_config.update_mode) == "mesh_only"

    if mesh_only_update:
        (
            _base_mesh_state,
            motion_mesh_state,
            _velocity_field,
            _transport_mask,
            transport_stats,
        ) = compute_mesh_motion_velocity(
            phi_np,
            feedback_config,
            tracked_topology=tracked_topology,
            base_mesh_state=base_mesh_state,
            compute_velocity_field=False,
        )
        phi_motion = phi_np
    else:
        phi_motion, _, motion_mesh_state, transport_stats = compute_mesh_motion_target(
            phi,
            feedback_config,
            tracked_topology=tracked_topology,
            base_mesh_state=base_mesh_state,
        )

    topology_config = MeshRelaxationConfig(
        steps=0,
        topology_steps=feedback_config.relax_config.topology_steps,
        speed_up=feedback_config.relax_config.speed_up,
        switch_distance=feedback_config.relax_config.switch_distance,
        movement_model=feedback_config.relax_config.movement_model,
        boundary_energy=feedback_config.relax_config.boundary_energy,
        random_seed=feedback_config.relax_config.random_seed,
        min_node_separation_factor=feedback_config.relax_config.min_node_separation_factor,
        max_node_separation_factor=feedback_config.relax_config.max_node_separation_factor,
    )
    mesh_state = (
        relax_mesh_state(motion_mesh_state, topology_config)
        if int(topology_config.topology_steps) > 0
        else motion_mesh_state
    )

    if mesh_only_update:
        current_labels = dominant_grain_map(phi_np)
        seed_unodes = mesh_state.get("_runtime_seed_unodes")
        if seed_unodes is not None:
            mesh_labels, seed_assignment_stats = assign_seed_unodes_from_mesh(
                mesh_state,
                seed_unodes,
                fallback_labels=current_labels,
            )
            mesh_labels, incremental_stats = incremental_seed_unode_reassignment(
                current_labels,
                mesh_labels,
                base_mesh_state if base_mesh_state is not None else motion_mesh_state,
                motion_mesh_state,
                seed_unodes,
            )
            seed_assignment_stats.update(incremental_stats)
            seed_fields = mesh_state.get("_runtime_seed_unode_fields")
            node_fields = mesh_state.get("_runtime_seed_node_fields")
            mass_ledgers: dict[str, dict[int, dict[str, float]]] = {}
            if seed_fields is not None:
                updated_fields, scalar_stats, mass_ledgers = update_seed_unode_fields(
                    current_labels,
                    mesh_labels,
                    base_mesh_state if base_mesh_state is not None else motion_mesh_state,
                    motion_mesh_state,
                    seed_unodes,
                    seed_fields,
                    node_fields=node_fields,
                )
                mesh_state["_runtime_seed_unode_fields"] = {
                    **seed_fields,
                    "values": updated_fields,
                }
                seed_assignment_stats.update(scalar_stats)
            if node_fields is not None:
                updated_node_fields, node_stats = update_seed_node_fields(
                    mesh_state,
                    seed_unodes,
                    mesh_labels,
                    node_fields,
                    dict(mesh_state.get("_runtime_seed_unode_fields", {}).get("values", {})),
                    mass_ledgers=mass_ledgers,
                )
                mesh_state["_runtime_seed_node_fields"] = {
                    **node_fields,
                    "positions": tuple(
                        (float(node["x"]), float(node["y"])) for node in mesh_state.get("nodes", [])
                    ),
                    "values": updated_node_fields,
                }
                seed_assignment_stats.update(node_stats)
        else:
            mesh_labels = rasterize_mesh_labels(
                mesh_state,
                current_labels.shape,
                fallback_labels=current_labels,
            )
            seed_assignment_stats = {
                "assigned_unodes": int(mesh_labels.size),
                "unassigned_unodes": 0,
                "swept_unodes": int(mesh_labels.size),
                "changed_unodes": int(np.count_nonzero(mesh_labels != current_labels)),
                "updated_scalar_unodes": 0,
                "scalar_swept_unodes": 0,
            }
        phi_feedback = mesh_labels_to_order_parameters(mesh_labels, phi_np.shape[0])
        changed = mesh_labels != current_labels
        feedback_stats = {
            "changed_pixels": int(changed.sum()),
            "feedback_pixels": int(changed.sum()),
            "strength": 1.0,
            **seed_assignment_stats,
        }
    else:
        phi_feedback, feedback_stats = apply_mesh_feedback(
            phi_motion,
            mesh_state,
            strength=feedback_config.strength,
            boundary_width=feedback_config.boundary_width,
        )
    mesh_state["stats"]["mesh_feedback_strength"] = float(feedback_config.strength)
    mesh_state["stats"]["mesh_transport_strength"] = float(feedback_config.transport_strength)
    mesh_state["stats"]["mesh_feedback_every"] = int(feedback_config.every)
    mesh_state["stats"]["mesh_update_mode"] = str(feedback_config.update_mode)
    mesh_state["stats"]["mesh_relaxation_steps"] = int(feedback_config.relax_config.steps)
    mesh_state["stats"]["mesh_topology_steps"] = int(topology_config.topology_steps)
    mesh_state["stats"]["mesh_feedback_changed_pixels"] = int(feedback_stats["changed_pixels"])
    mesh_state["stats"]["mesh_feedback_pixels"] = int(feedback_stats["feedback_pixels"])
    mesh_state["stats"]["mesh_transport_pixels"] = int(transport_stats["transport_pixels"])
    mesh_state["stats"]["mesh_transport_max_displacement"] = float(transport_stats["max_displacement"])
    mesh_state["stats"]["mesh_transport_mean_displacement"] = float(transport_stats["mean_displacement"])
    if "updated_scalar_unodes" in feedback_stats:
        mesh_state["stats"]["mesh_updated_scalar_unodes"] = int(feedback_stats["updated_scalar_unodes"])
    if "scalar_swept_unodes" in feedback_stats:
        mesh_state["stats"]["mesh_scalar_swept_unodes"] = int(feedback_stats["scalar_swept_unodes"])
    if "mass_conserved_fields" in feedback_stats:
        mesh_state["stats"]["mesh_mass_conserved_fields"] = int(feedback_stats["mass_conserved_fields"])
    if "mass_partitioned_fields" in feedback_stats:
        mesh_state["stats"]["mesh_mass_partitioned_fields"] = int(
            feedback_stats["mass_partitioned_fields"]
        )
    if "scalar_mass_residual" in feedback_stats:
        mesh_state["stats"]["mesh_scalar_mass_residual"] = float(feedback_stats["scalar_mass_residual"])
    if "updated_node_values" in feedback_stats:
        mesh_state["stats"]["mesh_updated_node_values"] = int(feedback_stats["updated_node_values"])
    if "partitioned_node_fields" in feedback_stats:
        mesh_state["stats"]["mesh_partitioned_node_fields"] = int(
            feedback_stats["partitioned_node_fields"]
        )
    return phi_feedback, mesh_state, {
        **feedback_stats,
        "transport_pixels": int(transport_stats["transport_pixels"]),
        "transport_max_displacement": float(transport_stats["max_displacement"]),
        "transport_mean_displacement": float(transport_stats["mean_displacement"]),
    }


def _get_ray(
    node_xy: np.ndarray,
    nb2_xy: np.ndarray,
    nb3_xy: np.ndarray,
    switch_distance: float,
) -> tuple[float, np.ndarray]:
    eps = 1e-8
    xy2 = _plot_xy(nb2_xy, node_xy)
    xy3 = _plot_xy(nb3_xy, node_xy)
    dx2 = xy2[0] - node_xy[0]
    dy2 = xy2[1] - node_xy[1]
    dx3 = xy3[0] - node_xy[0]
    dy3 = xy3[1] - node_xy[1]

    if abs(dx2) <= eps:
        dx2, dx3 = dx3, dx2
        dy2, dy3 = dy3, dy2

    if abs(dx2) <= eps:
        return 0.0, np.zeros(2, dtype=np.float64)

    k = 2.0 * dx3 * dy2 / dx2 - 2.0 * dy3
    if abs(k) <= eps:
        return 0.0, np.zeros(2, dtype=np.float64)

    y0 = ((dx3 / dx2) * (dx2 * dx2 + dy2 * dy2) - dx3 * dx3 - dy3 * dy3) / k
    x0 = (dx2 * dx2 + dy2 * dy2 - 2.0 * y0 * dy2) / (2.0 * dx2)
    radius = float(np.hypot(x0, y0))
    if radius <= eps:
        return 0.0, np.zeros(2, dtype=np.float64)

    direction = np.array([x0 / radius, y0 / radius], dtype=np.float64)
    radius = max(radius, switch_distance / 3.0)
    return radius, direction


def _ordered_neighbors(node_xy: np.ndarray, neighbor_ids: list[int], nodes: np.ndarray) -> list[int]:
    angles = []
    for neighbor_id in neighbor_ids:
        plotted = _plot_xy(nodes[neighbor_id], node_xy)
        delta = plotted - node_xy
        angle = float(np.arctan2(delta[1], delta[0]))
        angles.append((angle, neighbor_id))
    angles.sort()
    return [neighbor_id for _, neighbor_id in angles]


def _ordered_neighbor_points(
    node_xy: np.ndarray,
    ordered_neighbors: list[int],
    nodes: np.ndarray,
) -> np.ndarray:
    if not ordered_neighbors:
        return np.zeros((0, 2), dtype=np.float64)
    return np.asarray(
        [_plot_xy(nodes[int(neighbor_id)], node_xy) for neighbor_id in ordered_neighbors],
        dtype=np.float64,
    )


def _move_double(
    node_xy: np.ndarray,
    ordered_neighbors: list[int],
    nodes: np.ndarray,
    switch_distance: float,
    speed_up: float,
) -> np.ndarray:
    max_v = switch_distance / 5.0
    gb_energy = speed_up * switch_distance * switch_distance * 0.02
    radius, direction = _get_ray(node_xy, nodes[ordered_neighbors[0]], nodes[ordered_neighbors[1]], switch_distance)
    if radius <= 0.0:
        return np.zeros(2, dtype=np.float64)

    vlen = gb_energy / radius
    vlen = min(vlen, max_v)
    if vlen <= 0.0:
        return np.zeros(2, dtype=np.float64)
    return direction * vlen


def _move_triple(
    node_xy: np.ndarray,
    ordered_neighbors: list[int],
    nodes: np.ndarray,
    switch_distance: float,
    speed_up: float,
) -> np.ndarray:
    max_v = switch_distance / 5.0
    gb_energy = speed_up * switch_distance * switch_distance * 0.02
    moves = []

    for first, second in (
        (ordered_neighbors[0], ordered_neighbors[1]),
        (ordered_neighbors[1], ordered_neighbors[2]),
        (ordered_neighbors[2], ordered_neighbors[0]),
    ):
        radius, direction = _get_ray(node_xy, nodes[first], nodes[second], switch_distance)
        if radius <= 0.0:
            plotted = _plot_xy(nodes[first], node_xy) - node_xy
            moves.append(plotted)
            continue
        vlen = min(gb_energy / radius, max_v)
        moves.append(direction * vlen)

    increment = np.sum(moves, axis=0)
    norm = float(np.hypot(increment[0], increment[1]))
    if norm > max_v and norm > 0.0:
        increment = increment * (max_v / norm)
    return increment


def _trial_node_energy(
    trial_xy: np.ndarray,
    neighbor_ids: list[int],
    nodes: np.ndarray,
    *,
    boundary_energy: float,
) -> float:
    neighbor_points = _ordered_neighbor_points(
        np.asarray(trial_xy, dtype=np.float64),
        neighbor_ids,
        nodes,
    )
    if neighbor_points.size == 0:
        return 0.0
    delta = neighbor_points - np.asarray(trial_xy, dtype=np.float64)[None, :]
    return float(boundary_energy) * float(np.linalg.norm(delta, axis=1).sum())


def _surface_force_from_trial_energies(
    node_xy: np.ndarray,
    ordered_neighbors: list[int],
    nodes: np.ndarray,
    *,
    switch_distance: float,
    boundary_energy: float,
    use_diagonal_trials: bool = False,
    neighbor_points: np.ndarray | None = None,
) -> np.ndarray:
    if switch_distance <= 0.0:
        return np.zeros(2, dtype=np.float64)

    node_xy = np.asarray(node_xy, dtype=np.float64)
    if neighbor_points is None:
        neighbor_points = _ordered_neighbor_points(node_xy, ordered_neighbors, nodes)
    else:
        neighbor_points = np.asarray(neighbor_points, dtype=np.float64)

    if neighbor_points.size == 0:
        return np.zeros(2, dtype=np.float64)

    trial_offsets = [
        (switch_distance, 0.0),
        (-switch_distance, 0.0),
        (0.0, switch_distance),
        (0.0, -switch_distance),
    ]
    if use_diagonal_trials:
        diagonal_distance = switch_distance / np.sqrt(2.0)
        trial_offsets.extend(
            [
                (diagonal_distance, diagonal_distance),
                (-diagonal_distance, -diagonal_distance),
                (diagonal_distance, -diagonal_distance),
                (-diagonal_distance, diagonal_distance),
            ]
        )

    trial_positions = node_xy[None, :] + np.asarray(trial_offsets, dtype=np.float64)
    delta = neighbor_points[None, :, :] - trial_positions[:, None, :]
    energies = float(boundary_energy) * np.linalg.norm(delta, axis=2).sum(axis=1)

    energy_plus_x, energy_minus_x, energy_plus_y, energy_minus_y = [float(value) for value in energies[:4]]
    surface_delta_x = energy_plus_x - energy_minus_x
    surface_delta_y = energy_plus_y - energy_minus_y

    if use_diagonal_trials:
        diagonal_delta = (
            float(energies[4])
            - float(energies[5])
            + float(energies[6])
            - float(energies[7])
        ) / np.sqrt(2.0)
        surface_delta_x = (surface_delta_x + diagonal_delta) / 2.0
        surface_delta_y = (surface_delta_y + diagonal_delta) / 2.0

    return np.array(
        [
            -surface_delta_x / (2.0 * switch_distance),
            -surface_delta_y / (2.0 * switch_distance),
        ],
        dtype=np.float64,
    )


def _elle_surface_effective_dt(
    velocity_length: float,
    switch_distance: float,
    speed_up: float,
) -> float:
    if velocity_length <= 1.0e-12 or switch_distance <= 0.0:
        return 0.0
    dt = max(float(speed_up), 0.0)
    if velocity_length * dt > switch_distance:
        if dt > 1.0:
            dt = 1.0
        if velocity_length * dt > switch_distance:
            dt = 0.9 * switch_distance / velocity_length
    return dt


def _elle_surface_velocity_from_force(
    force: np.ndarray,
    node_xy: np.ndarray,
    ordered_neighbors: list[int],
    nodes: np.ndarray,
    *,
    unit_length: float = 1.0,
    segment_mobilities: list[float] | tuple[float, ...] | np.ndarray | None = None,
    neighbor_points: np.ndarray | None = None,
) -> np.ndarray:
    force_length = float(np.hypot(force[0], force[1]))
    if force_length <= 1.0e-12:
        return np.zeros(2, dtype=np.float64)

    effective_unit_length = max(float(unit_length), 1.0e-20)
    force_unit = force / force_length
    if neighbor_points is None:
        neighbor_points = _ordered_neighbor_points(
            np.asarray(node_xy, dtype=np.float64),
            ordered_neighbors,
            nodes,
        )
    else:
        neighbor_points = np.asarray(neighbor_points, dtype=np.float64)
    denominator = 0.0
    for index, plotted in enumerate(neighbor_points):
        segment = plotted - node_xy
        normal = np.array([segment[1], -segment[0]], dtype=np.float64)
        length = float(np.hypot(normal[0], normal[1]))
        if length <= 1.0e-12:
            continue
        cosine = abs(float(np.dot(normal, force_unit) / length))
        cosine = max(cosine, 0.01745)
        mobility = 1.0
        if segment_mobilities is not None and index < len(segment_mobilities):
            mobility = max(float(segment_mobilities[index]), 1.0e-20)
        denominator += ((length * effective_unit_length) * cosine * cosine) / mobility

    if denominator <= 1.0e-5 or np.isnan(denominator):
        return np.zeros(2, dtype=np.float64)

    velocity = (2.0 * force_unit * force_length) / denominator
    return velocity / effective_unit_length


def _move_node_elle_surface(
    node_xy: np.ndarray,
    ordered_neighbors: list[int],
    nodes: np.ndarray,
    switch_distance: float,
    speed_up: float,
    *,
    boundary_energy: float,
    use_diagonal_trials: bool = False,
    unit_length: float = 1.0,
) -> np.ndarray:
    if switch_distance <= 0.0 or not ordered_neighbors:
        return np.zeros(2, dtype=np.float64)

    node_xy = np.asarray(node_xy, dtype=np.float64)
    neighbor_points = _ordered_neighbor_points(node_xy, ordered_neighbors, nodes)
    force = _surface_force_from_trial_energies(
        node_xy,
        ordered_neighbors,
        nodes,
        switch_distance=switch_distance,
        boundary_energy=boundary_energy,
        use_diagonal_trials=use_diagonal_trials,
        neighbor_points=neighbor_points,
    )
    if abs(float(force[0])) <= 1.0e-12 or abs(float(force[1])) <= 1.0e-12:
        return np.zeros(2, dtype=np.float64)
    velocity = _elle_surface_velocity_from_force(
        force,
        node_xy,
        ordered_neighbors,
        nodes,
        unit_length=unit_length,
        neighbor_points=neighbor_points,
    )
    velocity_length = float(np.hypot(velocity[0], velocity[1]))
    if velocity_length <= 1.0e-12:
        return np.zeros(2, dtype=np.float64)

    dt = _elle_surface_effective_dt(velocity_length, switch_distance, speed_up)
    if dt <= 0.0:
        return np.zeros(2, dtype=np.float64)
    return velocity * dt


def _edge_map(flynns: list[dict[str, Any]]) -> dict[tuple[int, int], list[tuple[int, int]]]:
    edges: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for flynn_index, flynn in enumerate(flynns):
        node_ids = flynn["node_ids"]
        for index, node_id in enumerate(node_ids):
            neighbor_id = node_ids[(index + 1) % len(node_ids)]
            edge = tuple(sorted((int(node_id), int(neighbor_id))))
            edges.setdefault(edge, []).append((flynn_index, index))
    return edges


def _cycle_edges(node_ids: list[int]) -> list[tuple[int, int]]:
    normalized = _normalize_cycle([int(node_id) for node_id in node_ids])
    if len(normalized) < 2:
        return []
    return [
        (int(normalized[index]), int(normalized[(index + 1) % len(normalized)]))
        for index in range(len(normalized))
    ]


def _flynn_neighbor_lengths(
    flynns: list[dict[str, Any]],
    nodes: np.ndarray,
) -> list[dict[int, float]]:
    neighbor_lengths: list[dict[int, float]] = [{} for _ in flynns]
    for edge, entries in _edge_map(flynns).items():
        if len(entries) != 2:
            continue
        first_index = int(entries[0][0])
        second_index = int(entries[1][0])
        length = _edge_length(nodes, edge[0], edge[1])
        neighbor_lengths[first_index][second_index] = (
            neighbor_lengths[first_index].get(second_index, 0.0) + length
        )
        neighbor_lengths[second_index][first_index] = (
            neighbor_lengths[second_index].get(first_index, 0.0) + length
        )
    return neighbor_lengths


def _insert_node_on_edge(
    flynns: list[dict[str, Any]],
    node_a: int,
    node_b: int,
    new_node_id: int,
    candidate_flynn_indices: list[int] | tuple[int, ...] | None = None,
) -> bool:
    changed = False
    if candidate_flynn_indices is None:
        flynn_indices = range(len(flynns))
    else:
        flynn_indices = [int(index) for index in candidate_flynn_indices if 0 <= int(index) < len(flynns)]

    for flynn_index in flynn_indices:
        flynn = flynns[int(flynn_index)]
        node_ids = flynn["node_ids"]
        count = len(node_ids)
        for index in range(count):
            current = int(node_ids[index])
            nxt = int(node_ids[(index + 1) % count])
            if {current, nxt} != {node_a, node_b}:
                continue
            flynn["node_ids"] = node_ids[: index + 1] + [int(new_node_id)] + node_ids[index + 1 :]
            changed = True
            break
    return changed


def _normalize_cycle(node_ids: list[int]) -> list[int]:
    if not node_ids:
        return []

    normalized = [int(node_ids[0])]
    for node_id in node_ids[1:]:
        node_id = int(node_id)
        if node_id != normalized[-1]:
            normalized.append(node_id)

    if len(normalized) > 1 and normalized[0] == normalized[-1]:
        normalized.pop()
    return normalized


def _compact_mesh(nodes: np.ndarray, flynns: list[dict[str, Any]]) -> tuple[np.ndarray, list[dict[str, Any]]]:
    cleaned_flynns: list[dict[str, Any]] = []
    used_node_ids: set[int] = set()

    for flynn in flynns:
        normalized = _normalize_cycle([int(node_id) for node_id in flynn["node_ids"]])
        if len(set(normalized)) < 3:
            continue
        flynn_copy = dict(flynn)
        flynn_copy["node_ids"] = normalized
        cleaned_flynns.append(flynn_copy)
        used_node_ids.update(normalized)

    if not cleaned_flynns:
        return np.empty((0, 2), dtype=np.float64), []

    kept_node_ids = sorted(used_node_ids)
    node_mapping = {old_id: new_id for new_id, old_id in enumerate(kept_node_ids)}
    compact_nodes = np.asarray([nodes[int(old_id)] for old_id in kept_node_ids], dtype=np.float64)

    for flynn in cleaned_flynns:
        flynn["node_ids"] = [int(node_mapping[int(node_id)]) for node_id in flynn["node_ids"]]

    return compact_nodes, cleaned_flynns


def _cycle_with_directed_edge(node_ids: list[int], node_a: int, node_b: int) -> list[int] | None:
    count = len(node_ids)
    for index in range(count):
        if int(node_ids[index]) == int(node_a) and int(node_ids[(index + 1) % count]) == int(node_b):
            return [int(entry) for entry in node_ids[index:]] + [int(entry) for entry in node_ids[:index]]
    return None


def _cycle_with_normalized_edge(node_ids: list[int], node_a: int, node_b: int) -> list[int] | None:
    directed = _cycle_with_directed_edge(node_ids, node_a, node_b)
    if directed is not None:
        return directed
    reversed_node_ids = [int(entry) for entry in reversed(node_ids)]
    return _cycle_with_directed_edge(reversed_node_ids, node_a, node_b)


def _cycle_with_pattern(node_ids: list[int], pattern: list[int]) -> list[int] | None:
    count = len(node_ids)
    pattern_length = len(pattern)
    for source in ([int(entry) for entry in node_ids], [int(entry) for entry in reversed(node_ids)]):
        for index in range(count):
            rotated = source[index:] + source[:index]
            if rotated[:pattern_length] == pattern:
                return rotated
    return None


def _shared_edge_context(node_ids: list[int], node_a: int, node_b: int) -> dict[str, int] | None:
    directed = _cycle_with_directed_edge(node_ids, node_a, node_b)
    if directed is not None and len(directed) >= 3:
        return {
            "a_neighbor": int(directed[-1]),
            "b_neighbor": int(directed[2]),
        }

    directed = _cycle_with_directed_edge(node_ids, node_b, node_a)
    if directed is not None and len(directed) >= 3:
        return {
            "a_neighbor": int(directed[2]),
            "b_neighbor": int(directed[-1]),
        }
    return None


def _replace_shared_edge_keep_a(
    node_ids: list[int],
    static_a: int,
    node_a: int,
    node_b: int,
    transfer_b: int,
) -> list[int] | None:
    directed = _cycle_with_pattern(node_ids, [int(static_a), int(node_a), int(node_b), int(transfer_b)])
    if directed is None or len(directed) < 4:
        return None
    return [int(static_a), int(node_a), int(transfer_b)] + [int(entry) for entry in directed[4:]]


def _replace_shared_edge_keep_b(
    node_ids: list[int],
    transfer_a: int,
    node_a: int,
    node_b: int,
    static_b: int,
) -> list[int] | None:
    directed = _cycle_with_pattern(node_ids, [int(transfer_a), int(node_a), int(node_b), int(static_b)])
    if directed is None or len(directed) < 4:
        return None
    return [int(transfer_a), int(node_b), int(static_b)] + [int(entry) for entry in directed[4:]]


def _rewrite_exclusive_keep_a(
    node_ids: list[int],
    static_a: int,
    node_a: int,
    transfer_a: int,
    node_b: int,
) -> list[int] | None:
    directed = _cycle_with_pattern(node_ids, [int(static_a), int(node_a), int(transfer_a)])
    if directed is None or len(directed) < 3:
        return None
    return [int(static_a), int(node_a), int(node_b), int(transfer_a)] + [int(entry) for entry in directed[3:]]


def _rewrite_exclusive_keep_b(
    node_ids: list[int],
    transfer_b: int,
    node_b: int,
    static_b: int,
    node_a: int,
) -> list[int] | None:
    directed = _cycle_with_pattern(node_ids, [int(transfer_b), int(node_b), int(static_b)])
    if directed is None or len(directed) < 3:
        return None
    return [int(transfer_b), int(node_a), int(node_b), int(static_b)] + [int(entry) for entry in directed[3:]]


def _find_exclusive_flynn(
    flynns: list[dict[str, Any]],
    center: int,
    neighbors: tuple[int, int],
    excluded_indices: set[int],
) -> int | None:
    for flynn_index, flynn in enumerate(flynns):
        if flynn_index in excluded_indices:
            continue
        node_ids = [int(node_id) for node_id in flynn["node_ids"]]
        count = len(node_ids)
        for index in range(count):
            if int(node_ids[index]) != int(center):
                continue
            prev_node = int(node_ids[index - 1])
            next_node = int(node_ids[(index + 1) % count])
            if {prev_node, next_node} == {int(neighbors[0]), int(neighbors[1])}:
                return int(flynn_index)
    return None


def _segments_intersect(
    point_a1: np.ndarray,
    point_a2: np.ndarray,
    point_b1: np.ndarray,
    point_b2: np.ndarray,
) -> bool:
    def orientation(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> float:
        return float((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]))

    def on_segment(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> bool:
        eps = 1e-9
        return (
            min(p[0], r[0]) - eps <= q[0] <= max(p[0], r[0]) + eps
            and min(p[1], r[1]) - eps <= q[1] <= max(p[1], r[1]) + eps
        )

    o1 = orientation(point_a1, point_a2, point_b1)
    o2 = orientation(point_a1, point_a2, point_b2)
    o3 = orientation(point_b1, point_b2, point_a1)
    o4 = orientation(point_b1, point_b2, point_a2)
    eps = 1e-9

    if (o1 > eps and o2 < -eps or o1 < -eps and o2 > eps) and (
        o3 > eps and o4 < -eps or o3 < -eps and o4 > eps
    ):
        return True
    if abs(o1) <= eps and on_segment(point_a1, point_b1, point_a2):
        return True
    if abs(o2) <= eps and on_segment(point_a1, point_b2, point_a2):
        return True
    if abs(o3) <= eps and on_segment(point_b1, point_a1, point_b2):
        return True
    if abs(o4) <= eps and on_segment(point_b1, point_a2, point_b2):
        return True
    return False


def _plot_cycle_points(node_ids: list[int], nodes: np.ndarray) -> list[np.ndarray]:
    if not node_ids:
        return []
    points = [np.asarray(nodes[int(node_ids[0])], dtype=np.float64)]
    for node_id in node_ids[1:]:
        points.append(_plot_xy(nodes[int(node_id)], points[-1]))
    return points


def _polygon_signed_area_from_points(points: list[np.ndarray]) -> float:
    if len(points) < 3:
        return 0.0
    area = 0.0
    for index, point in enumerate(points):
        next_point = points[(index + 1) % len(points)]
        area += point[0] * next_point[1] - next_point[0] * point[1]
    return 0.5 * float(area)


def _flynn_area(node_ids: list[int], nodes: np.ndarray) -> float:
    points = _plot_cycle_points(node_ids, nodes)
    return abs(_polygon_signed_area_from_points(points))


def _polygon_is_simple_points(points: list[np.ndarray]) -> bool:
    count = len(points)
    if count <= 3:
        return True

    for segment_index in range(count):
        seg_start = points[segment_index]
        seg_end = points[(segment_index + 1) % count]
        for next_index in range(segment_index + 2, count):
            next_end_index = (next_index + 1) % count
            if segment_index == 0 and next_end_index == 0:
                continue
            if next_end_index == segment_index:
                continue
            next_start = points[next_index]
            next_end = points[next_end_index]
            if _segments_intersect(seg_start, seg_end, next_start, next_end):
                return False
    return True


def _point_in_polygon_mask(
    sample_x: np.ndarray,
    sample_y: np.ndarray,
    polygon_points: list[np.ndarray],
) -> np.ndarray:
    inside = np.zeros(sample_x.shape, dtype=bool)
    x_coords = np.array([float(point[0]) for point in polygon_points], dtype=np.float64)
    y_coords = np.array([float(point[1]) for point in polygon_points], dtype=np.float64)
    next_x = np.roll(x_coords, -1)
    next_y = np.roll(y_coords, -1)

    for x0, y0, x1, y1 in zip(x_coords, y_coords, next_x, next_y):
        intersects = ((y0 > sample_y) != (y1 > sample_y)) & (
            sample_x < (x1 - x0) * (sample_y - y0) / ((y1 - y0) + 1e-12) + x0
        )
        inside ^= intersects
    return inside


def _point_in_polygon(point: np.ndarray, polygon_points: list[np.ndarray]) -> bool:
    sample_x = np.array([[float(point[0])]], dtype=np.float64)
    sample_y = np.array([[float(point[1])]], dtype=np.float64)
    return bool(_point_in_polygon_mask(sample_x, sample_y, polygon_points)[0, 0])


def _flynn_geometry_metrics(node_ids: list[int], nodes: np.ndarray) -> dict[str, float | bool]:
    points = _plot_cycle_points(node_ids, nodes)
    area = abs(_polygon_signed_area_from_points(points))
    return {
        "area": float(area),
        "is_simple": bool(_polygon_is_simple_points(points)),
    }


def _flynn_geometry_metrics_cached(
    flynns: list[dict[str, Any]],
    nodes: np.ndarray,
    cache: dict[int, dict[str, float | bool]],
    flynn_index: int,
    *,
    require_simple: bool = True,
) -> dict[str, float | bool]:
    index = int(flynn_index)
    metrics = cache.get(index)
    if metrics is not None and ((not require_simple) or "is_simple" in metrics):
        return metrics

    node_ids = [int(node_id) for node_id in flynns[index]["node_ids"]]
    if metrics is None:
        metrics = {
            "area": float(_flynn_area(node_ids, nodes)),
        }
    if require_simple and "is_simple" not in metrics:
        metrics["is_simple"] = bool(_polygon_is_simple_points(_plot_cycle_points(node_ids, nodes)))
    cache[index] = metrics
    return metrics


def _affected_flynns_are_valid(
    flynns: list[dict[str, Any]],
    nodes: np.ndarray,
    affected_indices: set[int],
    min_area: float,
    *,
    geometry_cache: dict[int, dict[str, float | bool]] | None = None,
) -> tuple[bool, str | None]:
    for flynn_index in sorted(int(index) for index in affected_indices):
        if geometry_cache is None:
            metrics = _flynn_geometry_metrics([int(node_id) for node_id in flynns[flynn_index]["node_ids"]], nodes)
        else:
            metrics = _flynn_geometry_metrics_cached(
                flynns,
                nodes,
                geometry_cache,
                int(flynn_index),
                require_simple=True,
            )
        if not metrics["is_simple"]:
            return False, "non_simple"
        if float(metrics["area"]) < min_area:
            return False, "small_flynn"
    return True, None


def _switch_variant_is_valid(
    nodes: np.ndarray,
    node_a: int,
    node_b: int,
    shared_keep_a: dict[str, int],
    shared_keep_b: dict[str, int],
) -> bool:
    anchor = nodes[int(node_a)]
    a_xy = anchor
    b_xy = _plot_xy(nodes[int(node_b)], anchor)
    a_keep_xy = _plot_xy(nodes[int(shared_keep_a["b_neighbor"])], anchor)
    a_static_xy = _plot_xy(nodes[int(shared_keep_a["a_neighbor"])], anchor)
    b_keep_xy = _plot_xy(nodes[int(shared_keep_b["a_neighbor"])], anchor)
    b_static_xy = _plot_xy(nodes[int(shared_keep_b["b_neighbor"])], anchor)

    return not (
        _segments_intersect(b_keep_xy, b_xy, a_keep_xy, a_xy)
        or _segments_intersect(b_keep_xy, b_xy, a_static_xy, a_xy)
        or _segments_intersect(a_keep_xy, a_xy, b_static_xy, b_xy)
    )


def _polygon_is_valid(node_ids: list[int]) -> bool:
    normalized = _normalize_cycle(node_ids)
    return len(normalized) >= 3 and len(set(normalized)) >= 3


def _trace_boundary_cycle(
    boundary_edges: list[tuple[int, int]],
    preferred_start_order: list[int],
) -> list[int] | None:
    if not boundary_edges:
        return None

    adjacency: dict[int, list[int]] = {}
    for node_a, node_b in boundary_edges:
        adjacency.setdefault(int(node_a), []).append(int(node_b))
        adjacency.setdefault(int(node_b), []).append(int(node_a))

    for node_id, neighbors in list(adjacency.items()):
        unique_neighbors = sorted(set(int(neighbor) for neighbor in neighbors))
        if len(unique_neighbors) != 2:
            return None
        adjacency[int(node_id)] = unique_neighbors

    start_candidates = [int(node_id) for node_id in preferred_start_order if int(node_id) in adjacency]
    if not start_candidates:
        start_candidates = [min(adjacency)]

    for start in start_candidates:
        for first_neighbor in adjacency[int(start)]:
            cycle = [int(start)]
            visited_edges: set[tuple[int, int]] = set()
            previous = int(start)
            current = int(first_neighbor)

            while True:
                edge = tuple(sorted((int(previous), int(current))))
                if edge in visited_edges:
                    break
                visited_edges.add(edge)

                if int(current) == int(start):
                    if len(visited_edges) == len(boundary_edges) and len(cycle) >= 3:
                        return _normalize_cycle(cycle)
                    break

                cycle.append(int(current))
                next_candidates = [neighbor for neighbor in adjacency[int(current)] if int(neighbor) != int(previous)]
                if len(next_candidates) != 1:
                    break
                previous, current = int(current), int(next_candidates[0])

    return None


def _nudge_switched_node(
    nodes: np.ndarray,
    node_id: int,
    anchor_neighbor: int,
    side_neighbor_a: int,
    side_neighbor_b: int,
    switch_distance: float,
) -> None:
    origin = nodes[int(node_id)]
    target = (
        _plot_xy(nodes[int(anchor_neighbor)], origin)
        + _plot_xy(nodes[int(side_neighbor_a)], origin)
        + _plot_xy(nodes[int(side_neighbor_b)], origin)
    ) / 3.0
    increment = target - origin
    norm = float(np.hypot(increment[0], increment[1]))
    if norm <= 1e-12:
        return
    max_move = switch_distance * 0.25
    nodes[int(node_id)] = (origin + increment * min(max_move / norm, 1.0)) % 1.0


def _switch_triple_edge(
    nodes: np.ndarray,
    flynns: list[dict[str, Any]],
    node_a: int,
    node_b: int,
    switch_distance: float,
    *,
    node_neighbors: dict[int, set[int]] | None = None,
    edge_map: dict[tuple[int, int], list[tuple[int, int]]] | None = None,
    geometry_cache: dict[int, dict[str, float | bool]] | None = None,
) -> tuple[bool, dict[str, Any] | None, str | None]:
    if node_neighbors is None:
        node_neighbors = _node_neighbors(flynns)
    if len(node_neighbors.get(int(node_a), set())) != 3 or len(node_neighbors.get(int(node_b), set())) != 3:
        return False, None, None

    if edge_map is None:
        edge_map = _edge_map(flynns)
    shared_entries = edge_map.get(tuple(sorted((int(node_a), int(node_b)))))
    if shared_entries is None:
        return False, None, None
    shared_indices = sorted({int(flynn_index) for flynn_index, _ in shared_entries})
    if len(shared_indices) != 2:
        return False, None, None

    shared_contexts = []
    for flynn_index in shared_indices:
        context = _shared_edge_context([int(node_id) for node_id in flynns[flynn_index]["node_ids"]], node_a, node_b)
        if context is None:
            return False, None, None
        shared_contexts.append(context)

    exclusive_a_index = _find_exclusive_flynn(
        flynns,
        node_a,
        (shared_contexts[0]["a_neighbor"], shared_contexts[1]["a_neighbor"]),
        set(shared_indices),
    )
    exclusive_b_index = _find_exclusive_flynn(
        flynns,
        node_b,
        (shared_contexts[0]["b_neighbor"], shared_contexts[1]["b_neighbor"]),
        set(shared_indices),
    )
    if exclusive_a_index is None or exclusive_b_index is None:
        return False, None, None

    involved_indices = {int(exclusive_a_index), int(exclusive_b_index), *shared_indices}
    if len(involved_indices) != 4:
        return False, None, None

    candidates = [
        {
            "name": "switch_triple_a0b1",
            "keep_a_shared": 0,
            "keep_b_shared": 1,
            "a_new_neighbor": int(shared_contexts[0]["b_neighbor"]),
            "b_new_neighbor": int(shared_contexts[1]["a_neighbor"]),
        },
        {
            "name": "switch_triple_a1b0",
            "keep_a_shared": 1,
            "keep_b_shared": 0,
            "a_new_neighbor": int(shared_contexts[1]["b_neighbor"]),
            "b_new_neighbor": int(shared_contexts[0]["a_neighbor"]),
        },
    ]

    valid_candidates = []
    for candidate in candidates:
        if not _switch_variant_is_valid(
            nodes,
            node_a,
            node_b,
            shared_contexts[int(candidate["keep_a_shared"])],
            shared_contexts[int(candidate["keep_b_shared"])],
        ):
            continue
        valid_candidates.append(candidate)

    if not valid_candidates:
        return False, None, None

    candidate = min(
        valid_candidates,
        key=lambda item: _edge_length(nodes, node_a, int(item["a_new_neighbor"]))
        + _edge_length(nodes, node_b, int(item["b_new_neighbor"])),
    )

    original_node_ids = {
        flynn_index: [int(node_id) for node_id in flynns[flynn_index]["node_ids"]]
        for flynn_index in involved_indices
    }
    original_node_positions = {
        int(node_a): nodes[int(node_a)].copy(),
        int(node_b): nodes[int(node_b)].copy(),
    }

    keep_a_index = int(shared_indices[int(candidate["keep_a_shared"])])
    keep_b_index = int(shared_indices[int(candidate["keep_b_shared"])])

    rewritten = {
        keep_a_index: _replace_shared_edge_keep_a(
            original_node_ids[keep_a_index],
            int(shared_contexts[int(candidate["keep_a_shared"])]["a_neighbor"]),
            node_a,
            node_b,
            int(candidate["a_new_neighbor"]),
        ),
        keep_b_index: _replace_shared_edge_keep_b(
            original_node_ids[keep_b_index],
            int(candidate["b_new_neighbor"]),
            node_a,
            node_b,
            int(shared_contexts[int(candidate["keep_b_shared"])]["b_neighbor"]),
        ),
        int(exclusive_a_index): _rewrite_exclusive_keep_a(
            original_node_ids[int(exclusive_a_index)],
            int(shared_contexts[int(candidate["keep_a_shared"])]["a_neighbor"]),
            node_a,
            int(candidate["b_new_neighbor"]),
            node_b,
        ),
        int(exclusive_b_index): _rewrite_exclusive_keep_b(
            original_node_ids[int(exclusive_b_index)],
            int(candidate["a_new_neighbor"]),
            node_b,
            int(shared_contexts[int(candidate["keep_b_shared"])]["b_neighbor"]),
            node_a,
        ),
    }

    if any(node_ids is None or not _polygon_is_valid(node_ids) for node_ids in rewritten.values()):
        return False, None, "invalid_polygon"

    for flynn_index, node_ids in rewritten.items():
        flynns[int(flynn_index)]["node_ids"] = [int(node_id) for node_id in node_ids]

    updated_neighbors = _node_neighbors(flynns)
    if len(updated_neighbors.get(int(node_a), set())) != 3 or len(updated_neighbors.get(int(node_b), set())) != 3:
        for flynn_index, node_ids in original_node_ids.items():
            flynns[int(flynn_index)]["node_ids"] = [int(node_id) for node_id in node_ids]
        return False, None, "bad_degree"

    _nudge_switched_node(
        nodes,
        node_a,
        node_b,
        int(shared_contexts[int(candidate["keep_a_shared"])]["a_neighbor"]),
        int(candidate["a_new_neighbor"]),
        switch_distance,
    )
    _nudge_switched_node(
        nodes,
        node_b,
        node_a,
        int(candidate["b_new_neighbor"]),
        int(shared_contexts[1]["b_neighbor"] if int(candidate["keep_b_shared"]) == 1 else shared_contexts[0]["b_neighbor"]),
        switch_distance,
    )

    min_flynn_area = switch_distance * switch_distance * np.sin(np.pi / 3.0) * 0.5
    valid, invalid_reason = _affected_flynns_are_valid(
        flynns,
        nodes,
        involved_indices,
        min_area=float(min_flynn_area),
        geometry_cache=geometry_cache,
    )
    if not valid:
        for flynn_index, node_ids in original_node_ids.items():
            flynns[int(flynn_index)]["node_ids"] = [int(node_id) for node_id in node_ids]
        for node_id, position in original_node_positions.items():
            nodes[int(node_id)] = position
        return False, None, invalid_reason

    return True, {
        "type": str(candidate["name"]),
        "edge": [int(node_a), int(node_b)],
        "shared_flynns": [int(flynns[index]["flynn_id"]) for index in shared_indices],
        "exclusive_flynns": [
            int(flynns[int(exclusive_a_index)]["flynn_id"]),
            int(flynns[int(exclusive_b_index)]["flynn_id"]),
        ],
        "new_neighbors": {
            "node_a": int(candidate["a_new_neighbor"]),
            "node_b": int(candidate["b_new_neighbor"]),
        },
    }, None


def _merge_flynn_into_neighbor(
    nodes: np.ndarray,
    flynns: list[dict[str, Any]],
    remove_index: int,
    keep_index: int,
) -> dict[str, Any] | None:
    if int(remove_index) == int(keep_index):
        return None

    remove_flynn = flynns[int(remove_index)]
    keep_flynn = flynns[int(keep_index)]
    remove_node_ids = [int(node_id) for node_id in remove_flynn["node_ids"]]
    keep_node_ids = [int(node_id) for node_id in keep_flynn["node_ids"]]

    edge_counts: dict[tuple[int, int], int] = {}
    for node_ids in (keep_node_ids, remove_node_ids):
        for node_a, node_b in _cycle_edges(node_ids):
            edge = tuple(sorted((int(node_a), int(node_b))))
            edge_counts[edge] = edge_counts.get(edge, 0) + 1

    shared_edges = [edge for edge, count in edge_counts.items() if int(count) > 1]
    if not shared_edges:
        return None

    boundary_edges = [edge for edge, count in edge_counts.items() if int(count) == 1]
    merged_node_ids = _trace_boundary_cycle(boundary_edges, keep_node_ids + remove_node_ids)
    if merged_node_ids is None or not _polygon_is_valid(merged_node_ids):
        return None

    metrics = _flynn_geometry_metrics(merged_node_ids, nodes)
    if not metrics["is_simple"] or float(metrics["area"]) <= 0.0:
        return None

    shared_boundary = float(sum(_edge_length(nodes, edge[0], edge[1]) for edge in shared_edges))
    keep_flynn["node_ids"] = [int(node_id) for node_id in merged_node_ids]
    del flynns[int(remove_index)]

    return {
        "type": "merge_small_two_sided_flynn",
        "removed_flynn": int(remove_flynn["flynn_id"]),
        "kept_flynn": int(keep_flynn["flynn_id"]),
        "shared_boundary": shared_boundary,
        "result_node_count": int(len(merged_node_ids)),
    }


def _topocheck_small_flynn_thresholds(
    grid_shape: tuple[int, int] | list[int],
    switch_distance: float,
) -> tuple[int, float]:
    nx, ny = int(grid_shape[0]), int(grid_shape[1])
    max_unodes = max(nx * ny, 1)
    if switch_distance <= 0.0:
        max_nodes = 0
    else:
        max_nodes = int(round(2.5 / float(switch_distance) * (1.0 / np.sqrt(float(max_unodes)))))
    min_area = 1.0 / float(max_unodes)
    return int(max_nodes), float(min_area)


def _resolve_coincident_nodes(
    nodes: np.ndarray,
    flynns: list[dict[str, Any]],
    *,
    switch_distance: float,
) -> tuple[np.ndarray, list[dict[str, Any]], list[dict[str, Any]], int]:
    if len(nodes) == 0 or switch_distance <= 0.0:
        return nodes, flynns, [], 0

    increment = float(switch_distance) / 5.0
    events: list[dict[str, Any]] = []
    moved_nodes = 0
    checked_pairs: set[tuple[int, int]] = set()
    node_keys = {
        int(node_id): (round(float(xy[0]), 12), round(float(xy[1]), 12))
        for node_id, xy in enumerate(nodes)
    }

    for flynn in flynns:
        node_ids = [int(node_id) for node_id in flynn["node_ids"]]
        duplicate_groups: dict[tuple[float, float], list[int]] = {}
        for node_id in node_ids:
            duplicate_groups.setdefault(node_keys[int(node_id)], []).append(int(node_id))
        for grouped_nodes in duplicate_groups.values():
            if len(grouped_nodes) < 2:
                continue
            count = len(grouped_nodes)
            for left_index in range(max(count - 1, 0)):
                left_node = int(grouped_nodes[left_index])
                for right_index in range(left_index + 1, count):
                    right_node = int(grouped_nodes[right_index])
                pair = tuple(sorted((left_node, right_node)))
                if pair in checked_pairs:
                    continue
                checked_pairs.add(pair)

                separation = _periodic_relative(nodes[right_node], nodes[left_node])
                if float(np.hypot(separation[0], separation[1])) > 1.0e-12:
                    continue

                origin = nodes[left_node].copy()
                candidate_plus = (origin + np.array([increment, increment], dtype=np.float64)) % 1.0
                candidate_minus = (origin - np.array([increment, increment], dtype=np.float64)) % 1.0
                dist_plus = float(np.hypot(*_periodic_relative(nodes[right_node], candidate_plus)))
                dist_minus = float(np.hypot(*_periodic_relative(nodes[right_node], candidate_minus)))
                nodes[left_node] = candidate_minus if dist_minus > dist_plus else candidate_plus
                moved_nodes += 1
                events.append(
                    {
                        "type": "resolve_coincident_nodes",
                        "flynn_id": int(flynn.get("flynn_id", flynn.get("label", -1))),
                        "node_ids": [int(left_node), int(right_node)],
                        "increment": float(increment),
                    }
                )

    return nodes, flynns, events, moved_nodes


def _delete_small_flynns(
    nodes: np.ndarray,
    flynns: list[dict[str, Any]],
    *,
    max_nodes: int,
    min_area: float,
) -> tuple[np.ndarray, list[dict[str, Any]], list[dict[str, Any]], int]:
    if len(flynns) < 2:
        return nodes, flynns, [], 0

    events: list[dict[str, Any]] = []
    deleted = 0

    while True:
        neighbor_lengths = _flynn_neighbor_lengths(flynns, nodes)
        geometry_cache: dict[int, dict[str, float | bool]] = {}
        candidates = []
        for flynn_index, flynn in enumerate(flynns):
            neighbor_map = neighbor_lengths[int(flynn_index)]
            if not neighbor_map:
                continue

            node_ids = [int(node_id) for node_id in flynn["node_ids"]]
            node_count = len(set(node_ids))
            area = float(
                _flynn_geometry_metrics_cached(
                    flynns,
                    nodes,
                    geometry_cache,
                    int(flynn_index),
                    require_simple=False,
                )["area"]
            )
            if not (int(node_count) <= int(max_nodes) or area < float(min_area)):
                continue
            metrics = _flynn_geometry_metrics_cached(
                flynns,
                nodes,
                geometry_cache,
                int(flynn_index),
                require_simple=True,
            )
            if not metrics["is_simple"]:
                continue

            neighbor_order = sorted(
                (int(neighbor_index) for neighbor_index in neighbor_map),
                key=lambda neighbor_index: (
                    -float(neighbor_map[int(neighbor_index)]),
                    -float(
                        _flynn_geometry_metrics_cached(
                            flynns,
                            nodes,
                            geometry_cache,
                            int(neighbor_index),
                            require_simple=False,
                        )["area"]
                    ),
                    int(flynns[int(neighbor_index)]["flynn_id"]),
                ),
            )
            candidates.append(
                {
                    "flynn_index": int(flynn_index),
                    "flynn_id": int(flynn["flynn_id"]),
                    "node_count": int(node_count),
                    "area": float(area),
                    "neighbor_order": neighbor_order,
                }
            )

        if not candidates:
            break

        deleted_this_round = False
        for candidate in sorted(candidates, key=lambda item: (float(item["area"]), int(item["node_count"]), int(item["flynn_id"]))):
            remove_index = int(candidate["flynn_index"])
            if remove_index >= len(flynns) or int(flynns[remove_index]["flynn_id"]) != int(candidate["flynn_id"]):
                continue
            for keep_index in candidate["neighbor_order"]:
                if keep_index >= len(flynns):
                    continue
                event = _merge_flynn_into_neighbor(nodes, flynns, remove_index, keep_index)
                if event is None:
                    continue
                event["type"] = "delete_small_flynn"
                event["node_count"] = int(candidate["node_count"])
                event["area"] = float(candidate["area"])
                event["threshold_max_nodes"] = int(max_nodes)
                event["threshold_min_area"] = float(min_area)
                events.append(event)
                deleted += 1
                nodes, flynns = _compact_mesh(nodes, flynns)
                deleted_this_round = True
                break
            if deleted_this_round:
                break

        if not deleted_this_round:
            break

    return nodes, flynns, events, deleted


def _cyclic_order_distance(size: int, first_index: int, second_index: int) -> int:
    difference = abs(int(second_index) - int(first_index))
    return int(min(difference, int(size) - difference))


def _rotate_cycle_to_start(node_ids: list[int], start_node: int) -> list[int] | None:
    normalized = _normalize_cycle([int(node_id) for node_id in node_ids])
    if int(start_node) not in normalized:
        return None
    start_index = normalized.index(int(start_node))
    return normalized[start_index:] + normalized[:start_index]


def _find_split_pair(
    node_ids: list[int],
    nodes: np.ndarray,
    node_neighbors: dict[int, set[int]],
    *,
    switch_distance: float,
    excluded_neighbor_order: int = 4,
) -> tuple[int, int, float] | None:
    normalized = _normalize_cycle([int(node_id) for node_id in node_ids])
    size = len(normalized)
    if size < (excluded_neighbor_order * 2 + 1):
        return None

    normalized_array = np.asarray(normalized, dtype=np.int32)
    is_double = np.asarray(
        [len(node_neighbors.get(int(node_id), set())) == 2 for node_id in normalized_array],
        dtype=bool,
    )
    if int(is_double.sum()) < 2:
        return None

    positions = np.asarray(nodes[normalized_array], dtype=np.float64)
    cycle_indices = np.arange(size, dtype=np.int32)
    best_pair: tuple[int, int, float] | None = None
    best_separation = float(switch_distance)
    for first_index, first_node in enumerate(normalized_array):
        if not bool(is_double[first_index]):
            continue
        order_distance = np.abs(cycle_indices - int(first_index))
        order_distance = np.minimum(order_distance, size - order_distance)
        candidate_mask = is_double & (order_distance > int(excluded_neighbor_order))
        if not np.any(candidate_mask):
            continue

        candidate_indices = cycle_indices[candidate_mask]
        delta = positions[candidate_mask] - positions[int(first_index)]
        delta = (delta + 0.5) % 1.0 - 0.5
        separations = np.linalg.norm(delta, axis=1)
        nearest_index = int(np.argmin(separations))
        separation = float(separations[nearest_index])
        if separation >= best_separation:
            continue

        second_index = int(candidate_indices[nearest_index])
        second_node = int(normalized_array[second_index])
        best_pair = (int(first_node), second_node, separation)
        best_separation = separation
    return best_pair


def _split_midpoint_is_inside_flynn(
    node_ids: list[int],
    nodes: np.ndarray,
    node_a: int,
    node_b: int,
) -> bool:
    rotated = _rotate_cycle_to_start(node_ids, int(node_a))
    if rotated is None or int(node_b) not in rotated:
        return False
    split_index = rotated.index(int(node_b))
    if split_index <= 0:
        return False
    polygon_points = _plot_cycle_points(rotated, nodes)
    midpoint = (polygon_points[0] + polygon_points[int(split_index)]) * 0.5
    return _point_in_polygon(midpoint, polygon_points)


def _split_cycle_between_nodes(
    node_ids: list[int],
    node_a: int,
    node_b: int,
) -> tuple[list[int], list[int]] | None:
    rotated = _rotate_cycle_to_start(node_ids, int(node_a))
    if rotated is None or int(node_b) not in rotated:
        return None
    split_index = rotated.index(int(node_b))
    first_cycle = _normalize_cycle(rotated[: split_index + 1])
    second_cycle = _normalize_cycle(rotated[split_index:] + [int(node_a)])
    if len(set(first_cycle)) < 3 or len(set(second_cycle)) < 3:
        return None
    return first_cycle, second_cycle


def _split_flynns_if_needed(
    nodes: np.ndarray,
    flynns: list[dict[str, Any]],
    *,
    switch_distance: float,
) -> tuple[np.ndarray, list[dict[str, Any]], list[dict[str, Any]], int]:
    if len(flynns) == 0 or switch_distance <= 0.0:
        return nodes, flynns, [], 0

    split_events: list[dict[str, Any]] = []
    split_count = 0
    next_flynn_id = 0
    if flynns:
        next_flynn_id = max(int(flynn["flynn_id"]) for flynn in flynns) + 1

    node_neighbors = _node_neighbors(flynns)
    flynn_index = 0
    while flynn_index < len(flynns):
        flynn = flynns[int(flynn_index)]
        node_ids = [int(node_id) for node_id in flynn["node_ids"]]
        split_pair = _find_split_pair(
            node_ids,
            nodes,
            node_neighbors,
            switch_distance=switch_distance,
        )
        if split_pair is None:
            flynn_index += 1
            continue

        node_a, node_b, separation = split_pair
        if not _split_midpoint_is_inside_flynn(node_ids, nodes, node_a, node_b):
            flynn_index += 1
            continue

        split_cycles = _split_cycle_between_nodes(node_ids, node_a, node_b)
        if split_cycles is None:
            flynn_index += 1
            continue

        first_cycle, second_cycle = split_cycles
        first_metrics = _flynn_geometry_metrics(first_cycle, nodes)
        second_metrics = _flynn_geometry_metrics(second_cycle, nodes)
        if (
            not first_metrics["is_simple"]
            or not second_metrics["is_simple"]
            or float(first_metrics["area"]) <= 0.0
            or float(second_metrics["area"]) <= 0.0
        ):
            flynn_index += 1
            continue

        original_flynn_id = int(flynn["flynn_id"])
        original_label = int(flynn["label"])
        flynns[int(flynn_index)]["node_ids"] = [int(node_id) for node_id in first_cycle]
        new_flynn = dict(flynn)
        new_flynn["flynn_id"] = int(next_flynn_id)
        new_flynn["label"] = int(original_label)
        new_flynn["node_ids"] = [int(node_id) for node_id in second_cycle]
        flynns.append(new_flynn)
        node_neighbors = _node_neighbors(flynns)

        split_count += 1
        split_events.append(
            {
                "type": "split_flynn",
                "source_flynn": int(original_flynn_id),
                "new_flynn": int(next_flynn_id),
                "label": int(original_label),
                "split_nodes": [int(node_a), int(node_b)],
                "split_distance": float(separation),
                "source_node_count": int(len(node_ids)),
                "first_cycle_nodes": int(len(first_cycle)),
                "second_cycle_nodes": int(len(second_cycle)),
            }
        )
        next_flynn_id += 1
        flynn_index += 1

    return nodes, flynns, split_events, split_count


def _node_angle(
    node_xy: np.ndarray,
    first_xy: np.ndarray,
    second_xy: np.ndarray,
) -> float:
    first = _plot_xy(first_xy, node_xy) - node_xy
    second = _plot_xy(second_xy, node_xy) - node_xy
    first_norm = float(np.hypot(first[0], first[1]))
    second_norm = float(np.hypot(second[0], second[1]))
    if first_norm <= 1.0e-12 or second_norm <= 1.0e-12:
        return float(np.pi)
    cosine = float(np.dot(first, second) / (first_norm * second_norm))
    cosine = float(np.clip(cosine, -1.0, 1.0))
    return float(np.arccos(cosine))


def _node_angle_from_plotted_points(
    node_xy: np.ndarray,
    first_point: np.ndarray,
    second_point: np.ndarray,
) -> float:
    first = np.asarray(first_point, dtype=np.float64) - np.asarray(node_xy, dtype=np.float64)
    second = np.asarray(second_point, dtype=np.float64) - np.asarray(node_xy, dtype=np.float64)
    first_norm = float(np.hypot(first[0], first[1]))
    second_norm = float(np.hypot(second[0], second[1]))
    if first_norm <= 1.0e-12 or second_norm <= 1.0e-12:
        return float(np.pi)
    cosine = float(np.dot(first, second) / (first_norm * second_norm))
    cosine = float(np.clip(cosine, -1.0, 1.0))
    return float(np.arccos(cosine))


def _angle_widening_candidate(
    node_xy: np.ndarray,
    target_xy: np.ndarray,
    switch_distance: float,
) -> np.ndarray | None:
    direction = _plot_xy(target_xy, node_xy) - node_xy
    distance = float(np.hypot(direction[0], direction[1]))
    if distance <= max(float(switch_distance) * 0.1, 1.0e-12):
        return None
    return (node_xy + direction / distance * (0.1 * float(switch_distance))) % 1.0


def _angle_widening_candidate_from_point(
    node_xy: np.ndarray,
    target_point: np.ndarray,
    switch_distance: float,
) -> np.ndarray | None:
    direction = np.asarray(target_point, dtype=np.float64) - np.asarray(node_xy, dtype=np.float64)
    distance = float(np.hypot(direction[0], direction[1]))
    if distance <= max(float(switch_distance) * 0.1, 1.0e-12):
        return None
    return (np.asarray(node_xy, dtype=np.float64) + direction / distance * (0.1 * float(switch_distance))) % 1.0


def _repeat_angle_widening_to_target(
    node_xy: np.ndarray,
    first_point: np.ndarray,
    second_point: np.ndarray,
    target_point: np.ndarray,
    *,
    switch_distance: float,
    min_angle: float,
) -> tuple[np.ndarray | None, float, int]:
    current_xy = np.asarray(node_xy, dtype=np.float64).copy()
    first_point = np.asarray(first_point, dtype=np.float64)
    second_point = np.asarray(second_point, dtype=np.float64)
    target_point = np.asarray(target_point, dtype=np.float64)

    angle = _node_angle_from_plotted_points(current_xy, first_point, second_point)
    steps_taken = 0
    min_distance = max(float(switch_distance) * 0.1, 1.0e-12)

    while angle < min_angle:
        candidate_xy = _angle_widening_candidate_from_point(
            current_xy,
            target_point,
            switch_distance,
        )
        if candidate_xy is None:
            break
        current_xy = candidate_xy
        angle = _node_angle_from_plotted_points(current_xy, first_point, second_point)
        steps_taken += 1
        distance = float(np.hypot(*(current_xy - target_point)))
        if distance <= min_distance:
            break

    if steps_taken <= 1:
        return None, angle, steps_taken
    return current_xy, angle, steps_taken


def _widen_acute_angles(
    nodes: np.ndarray,
    flynns: list[dict[str, Any]],
    *,
    switch_distance: float,
    min_angle_degrees: float,
) -> tuple[np.ndarray, list[dict[str, Any]], list[dict[str, Any]], int]:
    if len(nodes) == 0 or switch_distance <= 0.0 or min_angle_degrees <= 0.0:
        return nodes, flynns, [], 0

    min_angle = float(np.deg2rad(min_angle_degrees))
    events: list[dict[str, Any]] = []
    widened_nodes = 0
    max_passes = max(len(nodes), 1)
    node_neighbors = _node_neighbors(flynns)

    for _ in range(max_passes):
        moved_any = False
        for node_id in range(len(nodes)):
            neighbor_ids = list(node_neighbors.get(int(node_id), set()))
            if len(neighbor_ids) not in (2, 3):
                continue

            current_xy = np.asarray(nodes[int(node_id)], dtype=np.float64)
            neighbor_points = _ordered_neighbor_points(current_xy, neighbor_ids, nodes)
            if neighbor_points.size == 0:
                continue
            if len(neighbor_ids) == 2:
                ordered = [int(neighbor_ids[0]), int(neighbor_ids[1])]
                ordered_points = neighbor_points
                pairs = [(0, 1)]
            else:
                angles = np.arctan2(
                    neighbor_points[:, 1] - current_xy[1],
                    neighbor_points[:, 0] - current_xy[0],
                )
                order = np.argsort(angles)
                ordered = [int(neighbor_ids[int(index)]) for index in order]
                ordered_points = neighbor_points[order]
                pairs = [(0, 1), (1, 2), (2, 0)]

            moved_this_node = False
            for first_index, second_index in pairs:
                first_neighbor = int(ordered[first_index])
                second_neighbor = int(ordered[second_index])
                current_angle = _node_angle_from_plotted_points(
                    current_xy,
                    ordered_points[int(first_index)],
                    ordered_points[int(second_index)],
                )
                if current_angle >= min_angle:
                    continue

                best_xy = None
                best_angle = current_angle
                best_target = None
                for target_index in (int(first_index), int(second_index)):
                    candidate_xy, candidate_angle, steps_taken = _repeat_angle_widening_to_target(
                        current_xy,
                        ordered_points[int(first_index)],
                        ordered_points[int(second_index)],
                        ordered_points[int(target_index)],
                        switch_distance=switch_distance,
                        min_angle=min_angle,
                    )
                    if candidate_xy is None or steps_taken <= 1:
                        continue
                    if candidate_angle > best_angle + 1.0e-12:
                        best_xy = candidate_xy
                        best_angle = candidate_angle
                        best_target = int(ordered[int(target_index)])

                if best_xy is None:
                    continue

                nodes[int(node_id)] = best_xy
                widened_nodes += 1
                moved_any = True
                moved_this_node = True
                events.append(
                    {
                        "type": "widen_angle",
                        "node_id": int(node_id),
                        "junction_type": "double" if len(ordered) == 2 else "triple",
                        "neighbors": [int(first_neighbor), int(second_neighbor)],
                        "target_neighbor": int(best_target),
                        "angle_before_degrees": float(np.rad2deg(current_angle)),
                        "angle_after_degrees": float(np.rad2deg(best_angle)),
                        "min_angle_degrees": float(min_angle_degrees),
                    }
                )
                break

            if moved_this_node:
                continue

        if not moved_any:
            break

    return nodes, flynns, events, widened_nodes


def _cleanup_small_two_sided_flynns(
    nodes: np.ndarray,
    flynns: list[dict[str, Any]],
    *,
    min_area: float,
) -> tuple[np.ndarray, list[dict[str, Any]], list[dict[str, Any]], int]:
    if min_area <= 0.0 or len(flynns) < 2:
        return nodes, flynns, [], 0

    events: list[dict[str, Any]] = []
    merged_flynns = 0

    while True:
        node_neighbors = _node_neighbors(flynns)
        neighbor_lengths = _flynn_neighbor_lengths(flynns, nodes)
        geometry_cache: dict[int, dict[str, float | bool]] = {}
        candidates = []

        for flynn_index, flynn in enumerate(flynns):
            neighbor_map = neighbor_lengths[int(flynn_index)]
            if len(neighbor_map) != 2:
                continue

            node_ids = [int(node_id) for node_id in flynn["node_ids"]]
            area = float(
                _flynn_geometry_metrics_cached(
                    flynns,
                    nodes,
                    geometry_cache,
                    int(flynn_index),
                    require_simple=False,
                )["area"]
            )
            if area >= float(min_area):
                continue
            metrics = _flynn_geometry_metrics_cached(
                flynns,
                nodes,
                geometry_cache,
                int(flynn_index),
                require_simple=True,
            )
            if not metrics["is_simple"]:
                continue

            triple_nodes = sorted(
                int(node_id)
                for node_id in set(node_ids)
                if len(node_neighbors.get(int(node_id), set())) >= 3
            )
            if len(triple_nodes) != 2:
                continue

            neighbor_order = sorted(
                (int(neighbor_index) for neighbor_index in neighbor_map),
                key=lambda neighbor_index: (
                    -float(neighbor_map[int(neighbor_index)]),
                    -float(
                        _flynn_geometry_metrics_cached(
                            flynns,
                            nodes,
                            geometry_cache,
                            int(neighbor_index),
                            require_simple=False,
                        )["area"]
                    ),
                    int(flynns[int(neighbor_index)]["flynn_id"]),
                ),
            )
            candidates.append(
                {
                    "flynn_index": int(flynn_index),
                    "flynn_id": int(flynn["flynn_id"]),
                    "area": float(area),
                    "triple_nodes": triple_nodes,
                    "neighbor_order": neighbor_order,
                    "neighbor_flynn_ids": [
                        int(flynns[int(neighbor_index)]["flynn_id"]) for neighbor_index in neighbor_order
                    ],
                }
            )

        if not candidates:
            break

        merged_this_round = False
        for candidate in sorted(candidates, key=lambda item: (float(item["area"]), int(item["flynn_id"]))):
            remove_index = int(candidate["flynn_index"])
            if remove_index >= len(flynns) or int(flynns[remove_index]["flynn_id"]) != int(candidate["flynn_id"]):
                continue

            for keep_index in candidate["neighbor_order"]:
                if keep_index >= len(flynns):
                    continue
                event = _merge_flynn_into_neighbor(nodes, flynns, remove_index, keep_index)
                if event is None:
                    continue

                event["area"] = float(candidate["area"])
                event["triple_nodes"] = [int(node_id) for node_id in candidate["triple_nodes"]]
                event["neighbor_flynn_ids"] = [int(flynn_id) for flynn_id in candidate["neighbor_flynn_ids"]]
                events.append(event)
                merged_flynns += 1
                nodes, flynns = _compact_mesh(nodes, flynns)
                merged_this_round = True
                break

            if merged_this_round:
                break

        if not merged_this_round:
            break

    return nodes, flynns, events, merged_flynns


def _maintain_mesh_once(
    nodes: np.ndarray,
    flynns: list[dict[str, Any]],
    *,
    grid_shape: tuple[int, int] | list[int],
    switch_distance: float,
    min_angle_degrees: float,
    min_node_separation: float,
    max_node_separation: float,
) -> tuple[np.ndarray, list[dict[str, Any]], list[dict[str, Any]], int, int, int, int, int, int, int, int]:
    events: list[dict[str, Any]] = []
    switched_edges = 0
    rejected_switches = 0
    merged_flynns = 0
    split_flynns = 0
    inserted_nodes = 0
    removed_nodes = 0
    coincident_nodes_resolved = 0
    deleted_small_flynns = 0
    widened_angles = 0

    topocheck_max_nodes, topocheck_min_area = _topocheck_small_flynn_thresholds(grid_shape, switch_distance)
    nodes, flynns, coincide_events, coincident_nodes_resolved = _resolve_coincident_nodes(
        nodes,
        flynns,
        switch_distance=switch_distance,
    )
    if coincident_nodes_resolved:
        events.extend(coincide_events)

    nodes, flynns, delete_events, deleted_small_flynns = _delete_small_flynns(
        nodes,
        flynns,
        max_nodes=topocheck_max_nodes,
        min_area=topocheck_min_area,
    )
    if deleted_small_flynns:
        merged_flynns += deleted_small_flynns
        events.extend(delete_events)

    edge_map = _edge_map(flynns)
    for edge, entries in sorted(edge_map.items(), key=lambda item: item[0]):
        length = _edge_length(nodes, edge[0], edge[1])
        if length <= max_node_separation:
            continue
        new_node_id = int(len(nodes))
        midpoint = _periodic_midpoint(nodes[edge[0]], nodes[edge[1]])
        nodes = np.vstack([nodes, midpoint])
        if _insert_node_on_edge(
            flynns,
            edge[0],
            edge[1],
            new_node_id,
            candidate_flynn_indices=[int(flynn_index) for flynn_index, _ in entries],
        ):
            inserted_nodes += 1
            events.append(
                {
                    "type": "insert_double",
                    "edge": [int(edge[0]), int(edge[1])],
                    "new_node_id": int(new_node_id),
                    "edge_length": float(length),
                }
            )

    if inserted_nodes:
        nodes, flynns = _compact_mesh(nodes, flynns)

    edge_map = _edge_map(flynns)
    node_neighbors = _node_neighbors(flynns)
    switch_geometry_cache: dict[int, dict[str, float | bool]] = {}
    edge_candidates = sorted(edge_map)
    for edge in edge_candidates:
        if _edge_length(nodes, edge[0], edge[1]) >= switch_distance:
            continue
        switched, event, rejected_reason = _switch_triple_edge(
            nodes,
            flynns,
            edge[0],
            edge[1],
            switch_distance,
            node_neighbors=node_neighbors,
            edge_map=edge_map,
            geometry_cache=switch_geometry_cache,
        )
        if switched and event is not None:
            switched_edges += 1
            event["edge_length"] = float(_edge_length(nodes, edge[0], edge[1]))
            events.append(event)
            edge_map = _edge_map(flynns)
            node_neighbors = _node_neighbors(flynns)
            switch_geometry_cache = {}
        elif rejected_reason is not None:
            rejected_switches += 1

    nodes, flynns, angle_events, widened_angles = _widen_acute_angles(
        nodes,
        flynns,
        switch_distance=switch_distance,
        min_angle_degrees=min_angle_degrees,
    )
    if widened_angles:
        events.extend(angle_events)

    nodes, flynns, split_events, split_flynns = _split_flynns_if_needed(
        nodes,
        flynns,
        switch_distance=switch_distance,
    )
    if split_flynns:
        events.extend(split_events)

    cleanup_area = min_node_separation * min_node_separation * np.sin(np.pi / 3.0) * 0.5
    node_neighbors = _node_neighbors(flynns)
    removal_candidates = []
    for node_id, neighbors in node_neighbors.items():
        if len(neighbors) != 2:
            continue
        shortest_length = min(_edge_length(nodes, node_id, neighbor_id) for neighbor_id in neighbors)
        if shortest_length < min_node_separation:
            removal_candidates.append((shortest_length, int(node_id)))

    current_node_neighbors = {
        int(node_id): {int(neighbor_id) for neighbor_id in neighbors}
        for node_id, neighbors in node_neighbors.items()
    }
    for shortest_length, node_id in sorted(removal_candidates):
        incident_flynns = [flynn for flynn in flynns if node_id in flynn["node_ids"]]
        if not incident_flynns:
            continue
        if any(len(set(flynn["node_ids"])) <= 3 for flynn in incident_flynns):
            continue

        current_neighbors = current_node_neighbors.get(node_id, set())
        if len(current_neighbors) != 2:
            continue

        for flynn in incident_flynns:
            flynn["node_ids"] = _normalize_cycle(
                [int(entry) for entry in flynn["node_ids"] if int(entry) != node_id]
            )

        removed_nodes += 1
        neighbor_pair = [int(neighbor_id) for neighbor_id in sorted(current_neighbors)]
        current_node_neighbors.pop(int(node_id), None)
        if len(neighbor_pair) == 2:
            left_neighbor, right_neighbor = neighbor_pair
            if left_neighbor in current_node_neighbors:
                current_node_neighbors[left_neighbor].discard(int(node_id))
                current_node_neighbors[left_neighbor].add(int(right_neighbor))
            if right_neighbor in current_node_neighbors:
                current_node_neighbors[right_neighbor].discard(int(node_id))
                current_node_neighbors[right_neighbor].add(int(left_neighbor))
        events.append(
            {
                "type": "delete_double",
                "node_id": int(node_id),
                "edge_length": float(shortest_length),
                "neighbors": [int(neighbor_id) for neighbor_id in sorted(current_neighbors)],
            }
        )

    if removed_nodes:
        nodes, flynns = _compact_mesh(nodes, flynns)

    nodes, flynns, cleanup_events, merged = _cleanup_small_two_sided_flynns(
        nodes,
        flynns,
        min_area=float(cleanup_area),
    )
    if merged:
        merged_flynns += merged
        events.extend(cleanup_events)

    return (
        nodes,
        flynns,
        events,
        switched_edges,
        rejected_switches,
        merged_flynns,
        split_flynns,
        inserted_nodes,
        removed_nodes,
        coincident_nodes_resolved,
        deleted_small_flynns,
        widened_angles,
    )


def relax_mesh_state(mesh_state: dict[str, Any], config: MeshRelaxationConfig) -> dict[str, Any]:
    nodes = np.array([[node["x"], node["y"]] for node in mesh_state["nodes"]], dtype=np.float64)
    flynns = _copy_flynns(mesh_state["flynns"])
    stats = dict(mesh_state["stats"])
    events = [dict(event) for event in mesh_state.get("events", [])]

    grid_shape = stats["grid_shape"]
    switch_distance = (
        config.switch_distance
        if config.switch_distance is not None
        else 0.5 / float(max(grid_shape))
    )
    unit_length = 1.0
    if config.use_elle_physical_units:
        unit_length = float(stats.get("elle_option_unitlength", 1.0))
        if unit_length <= 0.0:
            unit_length = 1.0
    min_node_separation = config.min_node_separation_factor * switch_distance
    max_node_separation = config.max_node_separation_factor * switch_distance
    rng = np.random.default_rng(config.random_seed)

    inserted_total = 0
    removed_total = 0
    switched_total = 0
    rejected_switch_total = 0
    merged_total = 0
    split_total = 0
    coincident_total = 0
    deleted_small_total = 0
    widened_angle_total = 0
    total_iterations = max(int(config.steps), int(config.topology_steps))

    for iteration in range(total_iterations):
        if iteration < int(config.steps):
            node_neighbors = _node_neighbors(flynns)
            node_order = rng.permutation(len(nodes))
            for node_id in node_order:
                neighbor_ids = sorted(node_neighbors.get(int(node_id), set()))
                if len(neighbor_ids) not in (2, 3):
                    continue

                node_xy = nodes[int(node_id)].copy()
                ordered = _ordered_neighbors(node_xy, neighbor_ids, nodes)
                if config.movement_model == "elle_surface":
                    increment = _move_node_elle_surface(
                        node_xy,
                        ordered,
                        nodes,
                        switch_distance,
                        config.speed_up,
                        boundary_energy=config.boundary_energy,
                        use_diagonal_trials=config.use_diagonal_trials,
                        unit_length=unit_length,
                    )
                elif len(ordered) == 2:
                    increment = _move_double(node_xy, ordered, nodes, switch_distance, config.speed_up)
                else:
                    increment = _move_triple(node_xy, ordered, nodes, switch_distance, config.speed_up)

                nodes[int(node_id)] = (node_xy + increment) % 1.0

        if iteration < int(config.topology_steps):
            nodes, flynns, iteration_events, switched, rejected, merged, split_count, inserted, removed, coincident, deleted_small, widened_angles = _maintain_mesh_once(
                nodes,
                flynns,
                grid_shape=grid_shape,
                switch_distance=switch_distance,
                min_angle_degrees=config.min_angle_degrees,
                min_node_separation=min_node_separation,
                max_node_separation=max_node_separation,
            )
            switched_total += switched
            rejected_switch_total += rejected
            merged_total += merged
            split_total += split_count
            inserted_total += inserted
            removed_total += removed
            coincident_total += coincident
            deleted_small_total += deleted_small
            widened_angle_total += widened_angles
            events.extend(iteration_events)

    node_neighbors = _node_neighbors(flynns)
    node_flynn_membership = _node_flynn_ids(flynns)
    stats["num_nodes"] = int(len(nodes))
    stats["num_flynns"] = int(len(flynns))
    stats["double_junctions"] = int(sum(len(neighbors) == 2 for neighbors in node_neighbors.values()))
    stats["triple_junctions"] = int(sum(len(neighbors) == 3 for neighbors in node_neighbors.values()))
    stats["mesh_relaxation_steps"] = int(config.steps)
    stats["mesh_topology_steps"] = int(config.topology_steps)
    stats["mesh_movement_model"] = str(config.movement_model)
    stats["mesh_surface_diagonal_trials"] = int(bool(config.use_diagonal_trials))
    stats["mesh_surface_use_elle_physical_units"] = int(bool(config.use_elle_physical_units))
    stats["mesh_unit_length"] = float(unit_length)
    stats["mesh_boundary_energy"] = float(config.boundary_energy)
    stats["mesh_switch_distance"] = float(switch_distance)
    stats["mesh_min_angle_degrees"] = float(config.min_angle_degrees)
    stats["mesh_min_node_separation"] = float(min_node_separation)
    stats["mesh_max_node_separation"] = float(max_node_separation)
    stats["mesh_random_seed"] = int(config.random_seed)
    stats["mesh_switched_triples"] = int(switched_total)
    stats["mesh_rejected_switches"] = int(rejected_switch_total)
    stats["mesh_merged_flynns"] = int(merged_total)
    stats["mesh_split_flynns"] = int(split_total)
    stats["mesh_inserted_nodes"] = int(inserted_total)
    stats["mesh_removed_nodes"] = int(removed_total)
    stats["mesh_coincident_nodes_resolved"] = int(coincident_total)
    stats["mesh_deleted_small_flynns"] = int(deleted_small_total)
    stats["mesh_widened_angles"] = int(widened_angle_total)
    stats["mesh_event_count"] = int(len(events))
    relaxed_mesh = _serializable_mesh(
        nodes,
        flynns,
        node_neighbors,
        node_flynn_membership,
        stats,
        events=events,
    )
    if "_runtime_seed_unodes" in mesh_state:
        relaxed_mesh["_runtime_seed_unodes"] = mesh_state["_runtime_seed_unodes"]
    if "_runtime_seed_unode_fields" in mesh_state:
        relaxed_mesh["_runtime_seed_unode_fields"] = mesh_state["_runtime_seed_unode_fields"]
    if "_runtime_seed_node_fields" in mesh_state:
        relaxed_mesh["_runtime_seed_node_fields"] = mesh_state["_runtime_seed_node_fields"]
    return relaxed_mesh


def write_mesh_state(path: str | Path, mesh_state: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    json_state = {key: value for key, value in mesh_state.items() if not str(key).startswith("_runtime_")}
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_state, handle, indent=2)
    return path
