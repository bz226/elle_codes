from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np

from .mesh import (
    _enforce_connected_label_ownership,
    _estimate_seed_unode_roi,
    _mark_seed_points_in_polygon,
    _node_flynn_ids,
    _node_neighbors,
    _plot_cycle_points_polygon,
    _polygon_signed_area_from_points,
    _serializable_mesh,
    assign_seed_unodes_from_mesh,
    build_mesh_state,
)
from .recovery import (
    _orientation_matrices_from_euler_deg,
    _resolve_recovery_symmetry_operators,
    _symmetry_misorientation_from_matrices,
)


@dataclass(frozen=True)
class NucleationConfig:
    high_angle_boundary_deg: float = 5.0
    min_cluster_unodes: int = 6
    excluded_phase: int = 2
    neighbor_radius: int = 3
    capture_neighbor_order: int = 1
    symmetry_path: str | None = None
    max_mesh_rebuild_growth_factor: float = 2.0
    parent_area_crit: float = 5.0e-4


def _smallest_unused_label(labels: np.ndarray) -> int:
    used = set(int(value) for value in np.unique(np.asarray(labels, dtype=np.int32)))
    candidate = 0
    while candidate in used:
        candidate += 1
    return candidate


def _build_grid_section(
    seed_unodes: dict[str, Any],
    flat_values: np.ndarray,
) -> np.ndarray:
    grid_shape = tuple(int(value) for value in seed_unodes["grid_shape"])
    grid = np.zeros(grid_shape + flat_values.shape[1:], dtype=flat_values.dtype)
    grid_indices = np.asarray(seed_unodes["grid_indices"], dtype=np.int32)
    grid[grid_indices[:, 0], grid_indices[:, 1]] = flat_values
    return grid


def _label_boundary_mask(label_mask: np.ndarray) -> np.ndarray:
    label_mask = np.asarray(label_mask, dtype=bool)
    if label_mask.ndim != 2:
        raise ValueError("label boundary mask expects a 2D label mask")
    boundary = np.zeros_like(label_mask, dtype=bool)
    if label_mask.size == 0:
        return boundary

    boundary[0, :] |= label_mask[0, :]
    boundary[-1, :] |= label_mask[-1, :]
    boundary[:, 0] |= label_mask[:, 0]
    boundary[:, -1] |= label_mask[:, -1]

    boundary[1:, :] |= label_mask[1:, :] & ~label_mask[:-1, :]
    boundary[:-1, :] |= label_mask[:-1, :] & ~label_mask[1:, :]
    boundary[:, 1:] |= label_mask[:, 1:] & ~label_mask[:, :-1]
    boundary[:, :-1] |= label_mask[:, :-1] & ~label_mask[:, 1:]
    return boundary


def _component_touches_label_boundary(component_mask: np.ndarray, label_mask: np.ndarray) -> bool:
    component_mask = np.asarray(component_mask, dtype=bool)
    label_mask = np.asarray(label_mask, dtype=bool)
    if component_mask.shape != label_mask.shape:
        raise ValueError("component and label masks must have matching shapes")
    return bool(np.any(component_mask & _label_boundary_mask(label_mask)))


def _choose_critical_boundary_seed(
    component_mask: np.ndarray,
    label_mask: np.ndarray,
    *,
    critical_scores: np.ndarray | None = None,
    boundary_seed_mask: np.ndarray | None = None,
) -> tuple[int, int] | None:
    component_np = np.asarray(component_mask, dtype=bool)
    if boundary_seed_mask is None:
        candidate_seed_mask = component_np & _label_boundary_mask(label_mask)
    else:
        candidate_seed_mask = component_np & np.asarray(boundary_seed_mask, dtype=bool)
    candidate_coords = np.argwhere(candidate_seed_mask)
    if candidate_coords.size == 0:
        return None
    if critical_scores is None:
        return (int(candidate_coords[0, 0]), int(candidate_coords[0, 1]))

    score_grid = np.asarray(critical_scores, dtype=np.float64)
    candidate_scores = np.asarray(
        [score_grid[int(ix), int(iy)] for ix, iy in candidate_coords],
        dtype=np.float64,
    )
    candidate_scores = np.where(np.isfinite(candidate_scores), candidate_scores, -np.inf)
    best_index = int(np.argmax(candidate_scores))
    return (
        int(candidate_coords[best_index, 0]),
        int(candidate_coords[best_index, 1]),
    )


def _boundary_local_component_subset(
    component_mask: np.ndarray,
    label_mask: np.ndarray,
    *,
    critical_scores: np.ndarray | None = None,
    capture_neighbor_order: int = 1,
    single_seed: bool = False,
    boundary_seed_mask: np.ndarray | None = None,
) -> np.ndarray:
    component_np = np.asarray(component_mask, dtype=bool)
    selected = np.zeros_like(component_np, dtype=bool)
    if boundary_seed_mask is None:
        candidate_seed_mask = component_np & _label_boundary_mask(label_mask)
    else:
        candidate_seed_mask = component_np & np.asarray(boundary_seed_mask, dtype=bool)
    if not np.any(candidate_seed_mask):
        return selected

    if single_seed:
        seed_coord = _choose_critical_boundary_seed(
            component_np,
            label_mask,
            critical_scores=critical_scores,
            boundary_seed_mask=candidate_seed_mask,
        )
        if seed_coord is None:
            return selected
        selected[seed_coord] = True
        frontier = {seed_coord}
    else:
        seed_coords = np.argwhere(candidate_seed_mask)
        selected[candidate_seed_mask] = True
        frontier = {
            (int(ix), int(iy))
            for ix, iy in seed_coords
        }
    max_steps = max(int(capture_neighbor_order), 0)
    nx, ny = component_np.shape
    for _ in range(max_steps):
        if not frontier:
            break
        next_frontier: set[tuple[int, int]] = set()
        for cx, cy in frontier:
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    ix = (int(cx) + dx) % nx
                    iy = (int(cy) + dy) % ny
                    if not bool(component_np[ix, iy]) or bool(selected[ix, iy]):
                        continue
                    selected[ix, iy] = True
                    next_frontier.add((ix, iy))
        frontier = next_frontier

    return selected


def _label_boundary_seed_mask_from_mesh(
    mesh_state: dict[str, Any],
    label: int,
    label_mask: np.ndarray,
    seed_position_grid: np.ndarray,
    *,
    max_distance: float,
) -> np.ndarray:
    label_mask_np = np.asarray(label_mask, dtype=bool)
    boundary_seed_mask = np.zeros_like(label_mask_np, dtype=bool)
    if max_distance <= 0.0:
        return boundary_seed_mask

    nodes = np.asarray(
        [[float(node["x"]), float(node["y"])] for node in mesh_state.get("nodes", ())],
        dtype=np.float64,
    )
    if nodes.size == 0:
        return boundary_seed_mask

    node_ids = sorted(
        {
            int(node_id)
            for flynn in mesh_state.get("flynns", ())
            if int(flynn.get("label", -1)) == int(label)
            for node_id in flynn.get("node_ids", ())
        }
    )
    if not node_ids:
        return boundary_seed_mask

    label_coords = np.argwhere(label_mask_np)
    if label_coords.size == 0:
        return boundary_seed_mask

    label_positions = np.asarray(
        seed_position_grid[label_coords[:, 0], label_coords[:, 1]],
        dtype=np.float64,
    )
    node_positions = nodes[np.asarray(node_ids, dtype=np.int32)]
    deltas = label_positions[:, None, :] - node_positions[None, :, :]
    deltas = (deltas + 0.5) % 1.0 - 0.5
    min_distances = np.min(np.hypot(deltas[..., 0], deltas[..., 1]), axis=1)
    near_coords = label_coords[min_distances <= float(max_distance)]
    if near_coords.size == 0:
        return boundary_seed_mask
    boundary_seed_mask[near_coords[:, 0], near_coords[:, 1]] = True
    return boundary_seed_mask


def _assign_seed_point_values_from_mesh(
    mesh_state: dict[str, Any],
    seed_unodes: dict[str, Any],
    value_getter,
    *,
    fallback_value: int = -1,
) -> np.ndarray:
    sample_points = np.asarray(seed_unodes["positions"], dtype=np.float64)
    values = np.full(sample_points.shape[0], int(fallback_value), dtype=np.int32)
    if sample_points.size == 0:
        return values

    node_positions = np.array(
        [[node["x"], node["y"]] for node in mesh_state.get("nodes", ())],
        dtype=np.float64,
    )
    if node_positions.size == 0:
        return values

    unassigned_mask = np.ones(sample_points.shape[0], dtype=bool)
    flynn_entries: list[tuple[float, int, list[np.ndarray]]] = []
    for flynn in mesh_state.get("flynns", ()):
        node_ids = [int(node_id) for node_id in flynn.get("node_ids", ())]
        if len(node_ids) < 3:
            continue
        polygon_points = _plot_cycle_points_polygon(node_ids, node_positions)
        if len(polygon_points) < 3:
            continue
        flynn_entries.append(
            (
                abs(_polygon_signed_area_from_points(polygon_points)),
                int(value_getter(flynn)),
                polygon_points,
            )
        )

    for _, value, polygon_points in sorted(flynn_entries, key=lambda item: item[0], reverse=True):
        if not np.any(unassigned_mask):
            break
        inside_mask = _mark_seed_points_in_polygon(
            sample_points,
            polygon_points,
            active_mask=unassigned_mask,
        )
        if not np.any(inside_mask):
            continue
        assigned_indices = np.flatnonzero(inside_mask)
        values[assigned_indices] = int(value)
        unassigned_mask[assigned_indices] = False

    return values


def _majority_nonnegative(values: np.ndarray) -> int | None:
    values_np = np.asarray(values, dtype=np.int32)
    values_np = values_np[values_np >= 0]
    if values_np.size == 0:
        return None
    unique_values, counts = np.unique(values_np, return_counts=True)
    return int(unique_values[int(np.argmax(counts))])


def _extend_source_labels(
    seed_fields: dict[str, Any] | None,
    *,
    max_label: int,
) -> tuple[int, ...]:
    existing = []
    if isinstance(seed_fields, dict):
        existing = [int(value) for value in seed_fields.get("source_labels", ())]
    next_source_label = max(existing, default=-1) + 1
    extended: list[int] = []
    for compact_label in range(max(int(max_label), -1) + 1):
        if compact_label < len(existing):
            extended.append(int(existing[compact_label]))
        else:
            extended.append(int(next_source_label))
            next_source_label += 1
    return tuple(extended)


def _build_current_flynn_section_specs(
    seed_flynn_sections: dict[str, Any] | None,
    current_flynns: list[dict[str, Any]],
    source_labels: tuple[int, ...],
) -> dict[str, Any] | None:
    if not isinstance(seed_flynn_sections, dict):
        return None

    field_order = tuple(str(name) for name in seed_flynn_sections.get("field_order", ()))
    if not field_order:
        return None

    defaults = {
        str(name): tuple(float(value) for value in values)
        for name, values in dict(seed_flynn_sections.get("defaults", {})).items()
    }
    component_counts = {
        str(name): int(value)
        for name, value in dict(seed_flynn_sections.get("component_counts", {})).items()
    }
    source_id_order = [int(value) for value in seed_flynn_sections.get("id_order", ())]
    source_values = {
        str(name): tuple(
            tuple(float(component) for component in entry)
            for entry in dense_values
        )
        for name, dense_values in dict(seed_flynn_sections.get("values", {})).items()
    }
    source_lookup_by_field: dict[str, dict[int, tuple[float, ...]]] = {}
    for field_name in field_order:
        dense_values = source_values.get(field_name)
        if dense_values is None:
            continue
        source_lookup_by_field[field_name] = {
            int(source_id): tuple(float(component) for component in entry)
            for source_id, entry in zip(source_id_order, dense_values)
        }

    current_values: dict[str, tuple[tuple[float, ...], ...]] = {}
    for field_name in field_order:
        default_value = tuple(float(value) for value in defaults.get(field_name, (0.0,)))
        dense_values: list[tuple[float, ...]] = []
        for flynn in current_flynns:
            if field_name == "F_ATTRIB_C":
                compact_label = int(flynn.get("label", 0))
                external_label = (
                    int(source_labels[compact_label])
                    if 0 <= compact_label < len(source_labels)
                    else int(compact_label)
                )
                dense_values.append((float(external_label),))
                continue
            source_flynn_id = int(flynn.get("source_flynn_id", flynn.get("flynn_id", -1)))
            dense_values.append(
                tuple(source_lookup_by_field.get(field_name, {}).get(source_flynn_id, default_value))
            )
        current_values[field_name] = tuple(dense_values)

    return {
        "field_order": field_order,
        "id_order": tuple(int(flynn["flynn_id"]) for flynn in current_flynns),
        "defaults": defaults,
        "component_counts": component_counts,
        "values": current_values,
    }


def _label_source_context(mesh_state: dict[str, Any]) -> dict[int, tuple[int | None, bool]]:
    context: dict[int, tuple[int | None, bool]] = {}
    for flynn in mesh_state.get("flynns", ()):
        if not isinstance(flynn, dict):
            continue
        compact_label = int(flynn.get("label", -1))
        if compact_label < 0:
            continue
        source_flynn_id = flynn.get("source_flynn_id")
        retained_identity = bool(flynn.get("retained_identity", True))
        existing = context.get(compact_label)
        if existing is None:
            context[compact_label] = (
                None if source_flynn_id is None else int(source_flynn_id),
                bool(retained_identity),
            )
            continue
        existing_source, existing_retained = existing
        chosen_source = existing_source
        if chosen_source is None and source_flynn_id is not None:
            chosen_source = int(source_flynn_id)
        context[compact_label] = (chosen_source, bool(existing_retained or retained_identity))
    return context


def _label_area_lookup(
    mesh_state: dict[str, Any],
    labels: np.ndarray,
) -> dict[int, float]:
    labels_np = np.asarray(labels, dtype=np.int32)
    area_by_label: dict[int, float] = {}
    nodes = mesh_state.get("nodes", ())
    flynns = mesh_state.get("flynns", ())
    if nodes and flynns:
        node_positions = np.asarray(
            [
                [float(node["x"]), float(node["y"])]
                for node in nodes
            ],
            dtype=np.float64,
        )
        for flynn in flynns:
            if not isinstance(flynn, dict):
                continue
            label = int(flynn.get("label", -1))
            node_ids = [int(node_id) for node_id in flynn.get("node_ids", ())]
            if label < 0 or len(node_ids) < 3:
                continue
            polygon_points = _plot_cycle_points_polygon(node_ids, node_positions)
            if len(polygon_points) < 3:
                continue
            area = abs(_polygon_signed_area_from_points(polygon_points))
            area_by_label[label] = float(area_by_label.get(label, 0.0) + float(area))

    if area_by_label:
        return area_by_label

    total_cells = float(labels_np.size)
    if total_cells <= 0.0:
        return {}
    unique_labels, counts = np.unique(labels_np, return_counts=True)
    return {
        int(label): float(count) / total_cells
        for label, count in zip(unique_labels, counts)
    }


def _rebuild_mesh_state_from_nucleated_labels(
    mesh_state: dict[str, Any],
    updated_labels: np.ndarray,
    source_labels: tuple[int, ...],
) -> dict[str, Any]:
    seed_unodes = mesh_state.get("_runtime_seed_unodes")
    if seed_unodes is None:
        return mesh_state

    previous_flynn_ids = _assign_seed_point_values_from_mesh(
        mesh_state,
        seed_unodes,
        lambda flynn: int(flynn.get("flynn_id", -1)),
        fallback_value=-1,
    )
    previous_source_flynn_ids = _assign_seed_point_values_from_mesh(
        mesh_state,
        seed_unodes,
        lambda flynn: int(flynn.get("source_flynn_id", flynn.get("flynn_id", -1))),
        fallback_value=-1,
    )

    rebuilt_mesh = build_mesh_state(updated_labels)
    rebuilt_component_ids = _assign_seed_point_values_from_mesh(
        rebuilt_mesh,
        seed_unodes,
        lambda flynn: int(flynn.get("flynn_id", -1)),
        fallback_value=-1,
    )

    original_source_label_count = len(
        tuple(int(value) for value in mesh_state.get("_runtime_seed_unode_fields", {}).get("source_labels", ()))
    )
    old_max_flynn_id = max(
        (int(flynn["flynn_id"]) for flynn in mesh_state.get("flynns", ())),
        default=-1,
    )
    next_flynn_id = old_max_flynn_id + 1
    retained_ids: set[int] = set()
    rebuilt_source_labels = [int(value) for value in source_labels]
    rebuilt_flynns: list[dict[str, Any]] = []

    for flynn in rebuilt_mesh.get("flynns", ()):
        component_mask = rebuilt_component_ids == int(flynn.get("flynn_id", -1))
        majority_previous_flynn = _majority_nonnegative(previous_flynn_ids[component_mask])
        majority_source_flynn = _majority_nonnegative(previous_source_flynn_ids[component_mask])
        compact_label = int(flynn.get("label", 0))
        retain_identity = (
            compact_label < int(original_source_label_count)
            and majority_previous_flynn is not None
            and majority_previous_flynn not in retained_ids
        )
        if retain_identity:
            current_flynn_id = int(majority_previous_flynn)
            retained_ids.add(current_flynn_id)
        else:
            current_flynn_id = int(next_flynn_id)
            next_flynn_id += 1
        if compact_label >= len(rebuilt_source_labels):
            next_source_label = max(rebuilt_source_labels, default=-1) + 1
            rebuilt_source_labels.extend(
                int(value) for value in range(next_source_label, next_source_label + (compact_label - len(rebuilt_source_labels) + 1))
            )
        current_compact_label = int(compact_label)
        current_source_label = int(rebuilt_source_labels[current_compact_label])
        source_flynn_id = (
            int(majority_source_flynn)
            if majority_source_flynn is not None
            else int(majority_previous_flynn if majority_previous_flynn is not None else current_flynn_id)
        )
        rebuilt_flynn = dict(flynn)
        rebuilt_flynn["flynn_id"] = int(current_flynn_id)
        rebuilt_flynn["label"] = int(current_compact_label)
        rebuilt_flynn["source_flynn_id"] = int(source_flynn_id)
        rebuilt_flynn["source_label"] = int(current_source_label)
        rebuilt_flynn["retained_identity"] = bool(retain_identity)
        rebuilt_flynn["parents"] = (
            []
            if retain_identity or majority_previous_flynn is None
            else [int(majority_previous_flynn)]
        )
        rebuilt_flynn["neighbors"] = []
        rebuilt_flynns.append(rebuilt_flynn)

    rebuilt_flynns.sort(key=lambda flynn: int(flynn["flynn_id"]))
    relabeled_labels = np.asarray(updated_labels, dtype=np.int32).copy()
    rebuilt_nodes = np.asarray(
        [[node["x"], node["y"]] for node in rebuilt_mesh.get("nodes", ())],
        dtype=np.float64,
    )
    rebuilt_neighbors = _node_neighbors(rebuilt_flynns)
    rebuilt_membership = _node_flynn_ids(rebuilt_flynns)
    rebuilt_stats = dict(mesh_state.get("stats", {}))
    rebuilt_stats.update(dict(rebuilt_mesh.get("stats", {})))
    rebuilt_stats["mesh_seed_source"] = "nucleation_rebuild"
    rebuilt_stats["mesh_parent_flynns"] = int(len(mesh_state.get("flynns", ())))
    rebuilt_stats["num_flynns"] = int(len(rebuilt_flynns))
    rebuilt_serializable = _serializable_mesh(
        rebuilt_nodes,
        rebuilt_flynns,
        rebuilt_neighbors,
        rebuilt_membership,
        rebuilt_stats,
        events=rebuilt_mesh.get("events", ()),
    )

    if "_runtime_seed_unodes" in mesh_state:
        rebuilt_serializable["_runtime_seed_unodes"] = mesh_state["_runtime_seed_unodes"]
    if "_runtime_seed_unode_sections" in mesh_state:
        rebuilt_serializable["_runtime_seed_unode_sections"] = mesh_state["_runtime_seed_unode_sections"]
    if "_runtime_seed_flynn_sections" in mesh_state:
        rebuilt_serializable["_runtime_seed_flynn_sections"] = mesh_state["_runtime_seed_flynn_sections"]
        current_sections = _build_current_flynn_section_specs(
            mesh_state.get("_runtime_seed_flynn_sections"),
            rebuilt_flynns,
            tuple(int(value) for value in rebuilt_source_labels),
        )
        if current_sections is not None:
            rebuilt_serializable["_runtime_current_flynn_sections"] = current_sections

    rebuilt_serializable["_runtime_rebuilt_source_labels"] = tuple(
        int(value) for value in rebuilt_source_labels
    )
    rebuilt_serializable["_runtime_relabelled_labels"] = tuple(
        tuple(int(value) for value in row)
        for row in np.asarray(relabeled_labels, dtype=np.int32)
    )

    # A direct rasterize->mesh->rasterize roundtrip can perturb the freshly
    # nucleated label field more than the old ELLE-style outer step does. Give
    # the next GBM stage one incremental handoff step so only genuinely swept
    # regions are re-assigned immediately after a successful nucleation rebuild.
    workflow_gbm_steps = int(mesh_state.get("stats", {}).get("workflow_gbm_steps_per_subloop", 1))
    rebuilt_serializable["_runtime_incremental_label_remap_stages"] = int(
        max(
            int(mesh_state.get("_runtime_incremental_label_remap_stages", 0)),
            max(workflow_gbm_steps, 1),
        )
    )
    rebuilt_serializable.setdefault("stats", {})
    rebuilt_serializable["stats"]["nucleation_incremental_label_handoff"] = 1

    return rebuilt_serializable


def _connected_components_from_misorientation(
    labels: np.ndarray,
    orientation_matrices: np.ndarray,
    misorientation_cutoff_deg: float,
    neighbor_radius: int = 1,
    excluded_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    labels_np = np.asarray(labels, dtype=np.int32)
    grid_shape = labels_np.shape
    component_ids = np.full(grid_shape, -1, dtype=np.int32)
    component_sizes: list[tuple[int, int]] = []
    symmetry_ops = _resolve_recovery_symmetry_operators(
        SimpleNamespace(symmetry_path=None)
    )
    next_component = 0
    radius = max(int(neighbor_radius), 1)

    for ix in range(grid_shape[0]):
        for iy in range(grid_shape[1]):
            if component_ids[ix, iy] >= 0:
                continue
            if excluded_mask is not None and bool(excluded_mask[ix, iy]):
                component_ids[ix, iy] = next_component
                component_sizes.append((next_component, 1))
                next_component += 1
                continue

            label = int(labels_np[ix, iy])
            stack = [(ix, iy)]
            component_ids[ix, iy] = next_component
            size = 0

            while stack:
                cx, cy = stack.pop()
                size += 1
                center_matrix = orientation_matrices[cx, cy]
                for dx in range(-radius, radius + 1):
                    for dy in range(-radius, radius + 1):
                        if dx == 0 and dy == 0:
                            continue
                        nx = (cx + dx) % grid_shape[0]
                        ny = (cy + dy) % grid_shape[1]
                        if component_ids[nx, ny] >= 0:
                            continue
                        if int(labels_np[nx, ny]) != label:
                            continue
                        if excluded_mask is not None and bool(excluded_mask[nx, ny]):
                            continue
                        neighbour_matrix = orientation_matrices[nx, ny]
                        misorientation = _symmetry_misorientation_from_matrices(
                            center_matrix,
                            neighbour_matrix,
                            symmetry_ops,
                        )
                        if misorientation > float(misorientation_cutoff_deg):
                            continue
                        component_ids[nx, ny] = next_component
                        stack.append((nx, ny))

            component_sizes.append((next_component, size))
            next_component += 1

    return component_ids, component_sizes


def apply_nucleation_stage(
    mesh_state: dict[str, Any],
    current_labels: np.ndarray,
    config: NucleationConfig,
    *,
    nucleation_stage_index: int = 0,
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    seed_unodes = mesh_state.get("_runtime_seed_unodes")
    seed_sections = mesh_state.get("_runtime_seed_unode_sections")
    if seed_unodes is None or seed_sections is None:
        return (
            np.asarray(current_labels, dtype=np.int32),
            mesh_state,
            {
                "candidate_clusters": 0,
                "nucleated_clusters": 0,
                "nucleated_unodes": 0,
                "new_labels_added": 0,
            },
        )

    section_values = seed_sections.get("values", {})
    if "U_EULER_3" not in section_values:
        return (
            np.asarray(current_labels, dtype=np.int32),
            mesh_state,
            {
                "candidate_clusters": 0,
                "nucleated_clusters": 0,
                "nucleated_unodes": 0,
                "new_labels_added": 0,
            },
        )

    current_np = np.asarray(current_labels, dtype=np.int32)
    seed_position_grid = _build_grid_section(
        seed_unodes,
        np.asarray(seed_unodes["positions"], dtype=np.float64),
    )
    boundary_seed_distance = float(_estimate_seed_unode_roi(seed_unodes))
    boundary_seed_masks: dict[int, np.ndarray] = {}
    euler_values = np.asarray(section_values["U_EULER_3"], dtype=np.float64)
    if euler_values.ndim != 2 or euler_values.shape[1] != 3:
        return (
            current_np,
            mesh_state,
            {
                "candidate_clusters": 0,
                "nucleated_clusters": 0,
                "nucleated_unodes": 0,
                "new_labels_added": 0,
            },
        )

    euler_grid = _build_grid_section(seed_unodes, euler_values)
    orientation_matrices = _orientation_matrices_from_euler_deg(euler_grid)

    excluded_mask = None
    seed_fields = mesh_state.get("_runtime_seed_unode_fields")
    if seed_fields is not None and "U_VISCOSITY" in seed_fields.get("values", {}):
        viscosity_values = np.asarray(seed_fields["values"]["U_VISCOSITY"], dtype=np.float64).reshape(-1, 1)
        viscosity_grid = _build_grid_section(seed_unodes, viscosity_values)[..., 0]
        excluded_mask = np.isclose(viscosity_grid, float(config.excluded_phase))

    component_ids, component_sizes = _connected_components_from_misorientation(
        current_np,
        orientation_matrices,
        misorientation_cutoff_deg=float(config.high_angle_boundary_deg),
        neighbor_radius=int(config.neighbor_radius),
        excluded_mask=excluded_mask,
    )
    component_size_lookup = {int(component_id): int(size) for component_id, size in component_sizes}

    updated_labels = current_np.copy()
    candidate_clusters = 0
    nucleated_clusters = 0
    nucleated_unodes = 0
    new_labels_added = 0
    skipped_repeated_sources = 0
    skipped_interior_clusters = 0
    skipped_small_parent_flynns = 0
    trimmed_cluster_unodes = 0
    pre_rebuild_connectivity_reassigned_unodes = 0
    pre_rebuild_connectivity_merged_components = 0

    field_values = {} if seed_fields is None else dict(seed_fields.get("values", {}))
    critical_score_grid = None
    use_single_critical_seed = "U_DISLOCDEN" in field_values
    attr_f_values = field_values.get("U_ATTRIB_F")
    if attr_f_values is not None:
        critical_score_grid = _build_grid_section(
            seed_unodes,
            np.asarray(attr_f_values, dtype=np.float64).reshape(-1, 1),
        )[..., 0]

    next_label = _smallest_unused_label(updated_labels)
    label_source_context = _label_source_context(mesh_state)
    label_area_lookup = _label_area_lookup(mesh_state, current_np)
    already_nucleated_sources = {
        int(value) for value in mesh_state.get("_runtime_nucleated_source_flynns", ()) if value is not None
    }
    for label in sorted(int(value) for value in np.unique(current_np)):
        source_flynn_id, retained_identity = label_source_context.get(label, (None, True))
        if (source_flynn_id is not None and source_flynn_id in already_nucleated_sources) or not retained_identity:
            skipped_repeated_sources += 1
            continue
        if float(label_area_lookup.get(label, 0.0)) < float(config.parent_area_crit):
            skipped_small_parent_flynns += 1
            continue
        label_mask = updated_labels == label
        boundary_seed_mask = boundary_seed_masks.get(int(label))
        if boundary_seed_mask is None:
            boundary_seed_mask = _label_boundary_seed_mask_from_mesh(
                mesh_state,
                int(label),
                label_mask,
                seed_position_grid,
                max_distance=boundary_seed_distance,
            )
            boundary_seed_masks[int(label)] = boundary_seed_mask
        component_candidates = []
        for component_id in np.unique(component_ids[label_mask]):
            size = int(component_size_lookup.get(int(component_id), 0))
            component_candidates.append((size, int(component_id)))
        component_candidates.sort(reverse=True)
        if len(component_candidates) < 2:
            continue
        candidate_clusters += max(len(component_candidates) - 1, 0)
        component_mask = None
        for size, component_id in component_candidates[1:]:
            if size < int(config.min_cluster_unodes):
                continue
            candidate_mask = component_ids == int(component_id)
            if not np.any(np.asarray(candidate_mask, dtype=bool) & np.asarray(boundary_seed_mask, dtype=bool)):
                skipped_interior_clusters += 1
                continue
            boundary_local_mask = _boundary_local_component_subset(
                candidate_mask,
                label_mask,
                critical_scores=critical_score_grid,
                capture_neighbor_order=int(config.capture_neighbor_order),
                single_seed=bool(use_single_critical_seed),
                boundary_seed_mask=boundary_seed_mask,
            )
            if not np.any(boundary_local_mask):
                skipped_interior_clusters += 1
                continue
            trimmed_cluster_unodes += int(
                np.count_nonzero(candidate_mask) - np.count_nonzero(boundary_local_mask)
            )
            component_mask = boundary_local_mask
            break
        if component_mask is None:
            continue
        updated_labels[component_mask] = int(next_label)
        nucleated_clusters += 1
        nucleated_unodes += int(np.count_nonzero(component_mask))
        new_labels_added += 1
        if source_flynn_id is not None:
            already_nucleated_sources.add(int(source_flynn_id))
        next_label += 1

    updated_mesh_state = mesh_state
    rebuilt_source_labels_fallback = tuple(
        int(value) for value in range(int(np.max(updated_labels)) + 1)
    )
    if nucleated_clusters > 0:
        label_overrides = np.full(current_np.shape, -1, dtype=np.int32)
        label_overrides[updated_labels != current_np] = updated_labels[updated_labels != current_np]
        fallback_override_labels = {
            int(value)
            for value in mesh_state.get("_runtime_fallback_override_labels", ())
            if value is not None
        }
        fallback_override_labels.update(
            int(value) for value in np.unique(label_overrides[label_overrides >= 0])
        )
        if seed_fields is not None:
            field_values = dict(seed_fields.get("values", {}))
            label_attribute = str(seed_fields.get("label_attribute", ""))
            extended_source_labels = _extend_source_labels(
                seed_fields,
                max_label=int(np.max(updated_labels)),
            )
            grid_indices_np = np.asarray(seed_unodes["grid_indices"], dtype=np.int32)

            def _label_attribute_values_from_labels(
                label_grid: np.ndarray,
                active_source_labels: tuple[int, ...],
            ) -> tuple[float, ...]:
                flat_labels = np.asarray(label_grid, dtype=np.int32)[
                    grid_indices_np[:, 0],
                    grid_indices_np[:, 1],
                ]
                return tuple(float(active_source_labels[int(value)]) for value in flat_labels)

            if label_attribute and label_attribute in field_values:
                field_values[label_attribute] = _label_attribute_values_from_labels(
                    updated_labels,
                    extended_source_labels,
                )
            rebuilt_mesh_state = _rebuild_mesh_state_from_nucleated_labels(
                mesh_state,
                updated_labels,
                extended_source_labels,
            )
            rebuilt_flynn_count = int(len(rebuilt_mesh_state.get("flynns", ())))
            parent_flynn_count = int(len(mesh_state.get("flynns", ())))
            baseline_flynn_count = int(
                mesh_state.get("stats", {}).get("mesh_original_seed_num_flynns", parent_flynn_count)
            )
            max_allowed_flynns = max(
                int(np.ceil(float(config.max_mesh_rebuild_growth_factor) * max(baseline_flynn_count, 1))),
                max(len(extended_source_labels) * 2, 1),
            )
            if rebuilt_flynn_count > max_allowed_flynns:
                cleaned_labels, cleanup_stats = _enforce_connected_label_ownership(
                    updated_labels,
                    reference_labels=current_np,
                    merge_max_component_size=max(int(config.min_cluster_unodes), 0),
                )
                retried_mesh_state = _rebuild_mesh_state_from_nucleated_labels(
                    mesh_state,
                    cleaned_labels,
                    extended_source_labels,
                )
                retried_flynn_count = int(len(retried_mesh_state.get("flynns", ())))
                if retried_flynn_count <= max_allowed_flynns:
                    updated_labels = np.asarray(cleaned_labels, dtype=np.int32)
                    if label_attribute and label_attribute in field_values:
                        field_values[label_attribute] = _label_attribute_values_from_labels(
                            updated_labels,
                            extended_source_labels,
                        )
                    rebuilt_mesh_state = retried_mesh_state
                    rebuilt_flynn_count = retried_flynn_count
                    pre_rebuild_connectivity_reassigned_unodes += int(
                        cleanup_stats.get("connectivity_reassigned_unodes", 0)
                    )
                    pre_rebuild_connectivity_merged_components += int(
                        cleanup_stats.get("connectivity_merged_components", 0)
                    )
                    updated_mesh_state = rebuilt_mesh_state
                    updated_mesh_state.pop("_runtime_fallback_override_labels", None)
                    updated_mesh_state["_runtime_seed_unode_fields"] = {
                        **seed_fields,
                        "source_labels": tuple(int(value) for value in extended_source_labels),
                        "values": field_values,
                    }
                    updated_mesh_state.setdefault("stats", {})
                    updated_mesh_state["stats"]["nucleation_pre_rebuild_connectivity_reassigned_unodes"] = int(
                        cleanup_stats.get("connectivity_reassigned_unodes", 0)
                    )
                    updated_mesh_state["stats"]["nucleation_pre_rebuild_connectivity_merged_components"] = int(
                        cleanup_stats.get("connectivity_merged_components", 0)
                    )
                    updated_mesh_state["stats"]["nucleation_mesh_rebuild_retried"] = 1
                else:
                    fallback_labels = np.asarray(cleaned_labels, dtype=np.int32)
                    label_overrides = np.full(current_np.shape, -1, dtype=np.int32)
                    label_overrides[fallback_labels != current_np] = fallback_labels[fallback_labels != current_np]
                    if label_attribute and label_attribute in field_values:
                        field_values[label_attribute] = _label_attribute_values_from_labels(
                            fallback_labels,
                            extended_source_labels,
                        )
                    updated_mesh_state = dict(mesh_state)
                    updated_mesh_state["_runtime_label_overrides"] = label_overrides
                    updated_mesh_state["_runtime_fallback_override_labels"] = tuple(
                        sorted(int(value) for value in fallback_override_labels)
                    )
                    updated_mesh_state["_runtime_seed_unode_fields"] = {
                        **seed_fields,
                        "source_labels": tuple(int(value) for value in extended_source_labels),
                        "values": field_values,
                    }
                    updated_mesh_state.setdefault("stats", {})
                    updated_mesh_state["stats"]["nucleation_mesh_rebuilt"] = 0
                    updated_mesh_state["stats"]["nucleation_mesh_rebuild_rejected"] = 1
                    updated_mesh_state["stats"]["nucleation_fragment_merge_max_size"] = int(
                        max(int(config.min_cluster_unodes), 0)
                    )
                    updated_mesh_state["stats"]["nucleation_rebuild_candidate_flynns"] = int(rebuilt_flynn_count)
                    updated_mesh_state["stats"]["nucleation_rebuild_retry_candidate_flynns"] = int(
                        retried_flynn_count
                    )
                    updated_mesh_state["stats"]["nucleation_rebuild_max_allowed_flynns"] = int(max_allowed_flynns)
                    updated_mesh_state["stats"]["nucleation_rebuild_baseline_flynns"] = int(baseline_flynn_count)
                    updated_mesh_state["stats"]["nucleation_pre_rebuild_connectivity_reassigned_unodes"] = int(
                        cleanup_stats.get("connectivity_reassigned_unodes", 0)
                    )
                    updated_mesh_state["stats"]["nucleation_pre_rebuild_connectivity_merged_components"] = int(
                        cleanup_stats.get("connectivity_merged_components", 0)
                    )
                    updated_mesh_state["stats"]["nucleation_mesh_rebuild_retried"] = 1
                    updated_mesh_state["stats"]["export_dense_unodes"] = 1
            if rebuilt_flynn_count <= max_allowed_flynns and updated_mesh_state is mesh_state:
                updated_mesh_state = rebuilt_mesh_state
                updated_mesh_state.pop("_runtime_fallback_override_labels", None)
                updated_mesh_state["_runtime_seed_unode_fields"] = {
                    **seed_fields,
                    "source_labels": tuple(int(value) for value in extended_source_labels),
                    "values": field_values,
                }
        else:
            rebuilt_mesh_state = _rebuild_mesh_state_from_nucleated_labels(
                mesh_state,
                updated_labels,
                tuple(range(int(np.max(updated_labels)) + 1)),
            )
            rebuilt_flynn_count = int(len(rebuilt_mesh_state.get("flynns", ())))
            parent_flynn_count = int(len(mesh_state.get("flynns", ())))
            baseline_flynn_count = int(
                mesh_state.get("stats", {}).get("mesh_original_seed_num_flynns", parent_flynn_count)
            )
            max_allowed_flynns = max(
                int(np.ceil(float(config.max_mesh_rebuild_growth_factor) * max(baseline_flynn_count, 1))),
                max(int(np.max(updated_labels)) + 1, 1) * 2,
            )
            if rebuilt_flynn_count > max_allowed_flynns:
                cleaned_labels, cleanup_stats = _enforce_connected_label_ownership(
                    updated_labels,
                    reference_labels=current_np,
                    merge_max_component_size=max(int(config.min_cluster_unodes), 0),
                )
                retried_mesh_state = _rebuild_mesh_state_from_nucleated_labels(
                    mesh_state,
                    cleaned_labels,
                    tuple(range(int(np.max(cleaned_labels)) + 1)),
                )
                retried_flynn_count = int(len(retried_mesh_state.get("flynns", ())))
                if retried_flynn_count <= max_allowed_flynns:
                    updated_labels = np.asarray(cleaned_labels, dtype=np.int32)
                    rebuilt_mesh_state = retried_mesh_state
                    rebuilt_flynn_count = retried_flynn_count
                    pre_rebuild_connectivity_reassigned_unodes += int(
                        cleanup_stats.get("connectivity_reassigned_unodes", 0)
                    )
                    pre_rebuild_connectivity_merged_components += int(
                        cleanup_stats.get("connectivity_merged_components", 0)
                    )
                    updated_mesh_state = rebuilt_mesh_state
                    updated_mesh_state.pop("_runtime_fallback_override_labels", None)
                    updated_mesh_state.setdefault("stats", {})
                    updated_mesh_state["stats"]["nucleation_pre_rebuild_connectivity_reassigned_unodes"] = int(
                        cleanup_stats.get("connectivity_reassigned_unodes", 0)
                    )
                    updated_mesh_state["stats"]["nucleation_pre_rebuild_connectivity_merged_components"] = int(
                        cleanup_stats.get("connectivity_merged_components", 0)
                    )
                    updated_mesh_state["stats"]["nucleation_mesh_rebuild_retried"] = 1
                else:
                    updated_mesh_state = dict(mesh_state)
                    updated_mesh_state["_runtime_label_overrides"] = label_overrides
                    updated_mesh_state["_runtime_fallback_override_labels"] = tuple(
                        sorted(int(value) for value in fallback_override_labels)
                    )
                    updated_mesh_state.setdefault("stats", {})
                    updated_mesh_state["stats"]["nucleation_mesh_rebuilt"] = 0
                    updated_mesh_state["stats"]["nucleation_mesh_rebuild_rejected"] = 1
                    updated_mesh_state["stats"]["nucleation_fragment_merge_max_size"] = int(
                        max(int(config.min_cluster_unodes), 0)
                    )
                    updated_mesh_state["stats"]["nucleation_rebuild_candidate_flynns"] = int(rebuilt_flynn_count)
                    updated_mesh_state["stats"]["nucleation_rebuild_retry_candidate_flynns"] = int(
                        retried_flynn_count
                    )
                    updated_mesh_state["stats"]["nucleation_rebuild_max_allowed_flynns"] = int(max_allowed_flynns)
                    updated_mesh_state["stats"]["nucleation_rebuild_baseline_flynns"] = int(baseline_flynn_count)
                    updated_mesh_state["stats"]["nucleation_pre_rebuild_connectivity_reassigned_unodes"] = int(
                        cleanup_stats.get("connectivity_reassigned_unodes", 0)
                    )
                    updated_mesh_state["stats"]["nucleation_pre_rebuild_connectivity_merged_components"] = int(
                        cleanup_stats.get("connectivity_merged_components", 0)
                    )
                    updated_mesh_state["stats"]["nucleation_mesh_rebuild_retried"] = 1
                    updated_mesh_state["stats"]["export_dense_unodes"] = 1
            if rebuilt_flynn_count <= max_allowed_flynns and updated_mesh_state is mesh_state:
                updated_mesh_state = rebuilt_mesh_state
                updated_mesh_state.pop("_runtime_fallback_override_labels", None)
        rebuilt_source_labels = tuple(
            int(value)
            for value in updated_mesh_state.pop(
                "_runtime_rebuilt_source_labels",
                extended_source_labels if seed_fields is not None else rebuilt_source_labels_fallback,
            )
        )
        if int(updated_mesh_state.get("stats", {}).get("nucleation_mesh_rebuilt", 1)) == 1:
            relabeled_payload = updated_mesh_state.pop("_runtime_relabelled_labels", None)
            relabel_stats: dict[str, int] = {"assigned_unodes": 0, "unassigned_unodes": 0}
            if relabeled_payload is not None:
                updated_labels = np.asarray(relabeled_payload, dtype=np.int32)
                relabel_stats["assigned_unodes"] = int(updated_labels.size)
            elif updated_mesh_state.get("nodes") and updated_mesh_state.get("flynns"):
                relabeled_labels, relabel_stats = assign_seed_unodes_from_mesh(
                    updated_mesh_state,
                    seed_unodes,
                    fallback_labels=updated_labels,
                )
                updated_labels = np.asarray(relabeled_labels, dtype=np.int32)
            if seed_fields is not None:
                runtime_seed_fields = dict(updated_mesh_state.get("_runtime_seed_unode_fields", seed_fields))
                runtime_field_values = dict(runtime_seed_fields.get("values", {}))
                if label_attribute and label_attribute in runtime_field_values:
                    runtime_field_values[label_attribute] = _label_attribute_values_from_labels(
                        updated_labels,
                        rebuilt_source_labels,
                    )
                updated_mesh_state["_runtime_seed_unode_fields"] = {
                    **runtime_seed_fields,
                    "source_labels": tuple(int(value) for value in rebuilt_source_labels),
                    "values": runtime_field_values,
                }
            updated_mesh_state["stats"]["nucleation_rebuilt_assigned_unodes"] = int(
                relabel_stats.get("assigned_unodes", 0)
            )
            updated_mesh_state["stats"]["nucleation_rebuilt_unassigned_unodes"] = int(
                relabel_stats.get("unassigned_unodes", 0)
            )
            updated_mesh_state.pop("_runtime_label_overrides", None)
            updated_mesh_state.pop("_runtime_fallback_override_labels", None)
            updated_mesh_state.pop("_runtime_seed_node_fields", None)
            updated_mesh_state.pop("_runtime_seed_node_sections", None)
        updated_mesh_state.setdefault("stats", {})
        updated_mesh_state["stats"]["nucleation_stage_index"] = int(nucleation_stage_index)
        updated_mesh_state["stats"]["nucleated_clusters"] = int(nucleated_clusters)
        updated_mesh_state["stats"]["nucleated_unodes"] = int(nucleated_unodes)
        updated_mesh_state["stats"]["new_labels_added"] = int(new_labels_added)
        updated_mesh_state["stats"].setdefault("nucleation_mesh_rebuilt", 1)
        updated_mesh_state["stats"]["nucleation_skipped_repeated_sources"] = int(skipped_repeated_sources)
        updated_mesh_state["stats"]["nucleation_skipped_interior_clusters"] = int(skipped_interior_clusters)
        updated_mesh_state["stats"]["nucleation_skipped_small_parent_flynns"] = int(
            skipped_small_parent_flynns
        )
        updated_mesh_state["stats"]["nucleation_trimmed_cluster_unodes"] = int(trimmed_cluster_unodes)
        updated_mesh_state["stats"]["nucleation_fragment_merge_max_size"] = int(
            max(int(config.min_cluster_unodes), 0)
        )
        updated_mesh_state["stats"]["nucleation_pre_rebuild_connectivity_reassigned_unodes"] = int(
            pre_rebuild_connectivity_reassigned_unodes
        )
        updated_mesh_state["stats"]["nucleation_pre_rebuild_connectivity_merged_components"] = int(
            pre_rebuild_connectivity_merged_components
        )
        updated_mesh_state["_runtime_nucleated_source_flynns"] = tuple(sorted(int(value) for value in already_nucleated_sources))

    stats = {
        "candidate_clusters": int(candidate_clusters),
        "nucleated_clusters": int(nucleated_clusters),
        "nucleated_unodes": int(nucleated_unodes),
        "new_labels_added": int(new_labels_added),
        "skipped_repeated_sources": int(skipped_repeated_sources),
        "skipped_interior_clusters": int(skipped_interior_clusters),
        "skipped_small_parent_flynns": int(skipped_small_parent_flynns),
        "trimmed_cluster_unodes": int(trimmed_cluster_unodes),
        "pre_rebuild_connectivity_reassigned_unodes": int(pre_rebuild_connectivity_reassigned_unodes),
        "pre_rebuild_connectivity_merged_components": int(pre_rebuild_connectivity_merged_components),
    }
    return updated_labels, updated_mesh_state, stats
