from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import dominant_grain_map, snapshot_statistics


def _as_numpy(phi) -> np.ndarray:
    phi_np = np.asarray(phi, dtype=np.float32)
    if phi_np.ndim != 3:
        raise ValueError("expected phi to have shape (num_grains, nx, ny)")
    return phi_np


def _write_options(handle, nx: int, ny: int) -> None:
    switch_distance = 0.5 / float(max(nx, ny))
    max_node_separation = 2.2 * switch_distance
    min_node_separation = switch_distance

    handle.write("OPTIONS\n")
    handle.write(f"SwitchDistance {switch_distance:.8e}\n")
    handle.write(f"MaxNodeSeparation {max_node_separation:.8e}\n")
    handle.write(f"MinNodeSeparation {min_node_separation:.8e}\n")
    handle.write("SpeedUp 1.00000000e+00\n")
    handle.write("CellBoundingBox 0.00000000e+00 0.00000000e+00\n")
    handle.write("                1.00000000e+00 0.00000000e+00 \n")
    handle.write("                1.00000000e+00 1.00000000e+00 \n")
    handle.write("                0.00000000e+00 1.00000000e+00 \n")
    handle.write("SimpleShearOffset 0.00000000e+00\n")
    handle.write("CumulativeSimpleShear 0.00000000e+00\n")
    handle.write("Timestep 1.00000000e+00\n")
    handle.write("UnitLength 1.00000000e+00\n")
    handle.write("Temperature 2.50000000e+01\n")
    handle.write("Pressure 1.00000000e+00\n")


def _format_sparse_record(values: tuple[float, ...]) -> str:
    if len(values) == 1:
        return f"{float(values[0]):.8e}"
    return " ".join(f"{float(value):.8e}" for value in values)


def _iter_nondefault_records(
    ordered_ids: list[int] | tuple[int, ...],
    dense_values: list[tuple[float, ...]] | tuple[tuple[float, ...], ...],
    default_value: tuple[float, ...],
) -> list[tuple[int, tuple[float, ...]]]:
    records: list[tuple[int, tuple[float, ...]]] = []
    default_np = np.asarray(default_value, dtype=np.float64)
    for entity_id, values in zip(ordered_ids, dense_values):
        value_tuple = tuple(float(value) for value in values)
        if np.allclose(np.asarray(value_tuple, dtype=np.float64), default_np, atol=1.0e-12, rtol=0.0):
            continue
        records.append((int(entity_id), value_tuple))
    return records


def _write_sparse_numeric_section(
    handle,
    section_name: str,
    ordered_ids: list[int] | tuple[int, ...],
    dense_values: list[tuple[float, ...]] | tuple[tuple[float, ...], ...],
    default_value: tuple[float, ...],
) -> None:
    handle.write(f"{section_name}\n")
    handle.write(f"Default {_format_sparse_record(default_value)}\n")
    for entity_id, values in _iter_nondefault_records(ordered_ids, dense_values, default_value):
        handle.write(f"{int(entity_id)} {_format_sparse_record(values)}\n")


def _dense_unode_layout(nx: int, ny: int) -> tuple[tuple[int, ...], tuple[tuple[float, float], ...], tuple[tuple[int, int], ...]]:
    unode_ids: list[int] = []
    unode_positions: list[tuple[float, float]] = []
    unode_grid_indices: list[tuple[int, int]] = []
    unode_id = 0
    for iy in range(ny):
        y_coord = (iy + 0.5) / float(ny)
        for ix in range(nx):
            x_coord = (ix + 0.5) / float(nx)
            unode_ids.append(int(unode_id))
            unode_positions.append((float(x_coord), float(y_coord)))
            unode_grid_indices.append((int(ix), int(iy)))
            unode_id += 1
    return tuple(unode_ids), tuple(unode_positions), tuple(unode_grid_indices)


def _nearest_fill_dense_grid(values: np.ndarray, filled: np.ndarray) -> np.ndarray:
    if not np.any(filled) or np.all(filled):
        return values
    dense = np.asarray(values).copy()
    occupied = np.argwhere(filled)
    missing = np.argwhere(~filled)
    for ix, iy in missing:
        deltas = occupied - np.asarray([ix, iy], dtype=np.int32)
        nearest = occupied[int(np.argmin(np.einsum("ij,ij->i", deltas, deltas)))]
        dense[ix, iy] = dense[int(nearest[0]), int(nearest[1])]
    return dense


def _densify_scalar_samples(
    grid_shape: tuple[int, int],
    grid_indices: np.ndarray,
    sample_values: np.ndarray,
    *,
    categorical: bool,
) -> np.ndarray:
    nx, ny = grid_shape
    filled = np.zeros((nx, ny), dtype=bool)
    if categorical:
        dense = np.zeros((nx, ny), dtype=np.float64)
        buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
        for (ix, iy), value in zip(grid_indices, sample_values):
            buckets[(int(ix), int(iy))].append(int(round(float(value))))
        for (ix, iy), values in buckets.items():
            labels, counts = np.unique(np.asarray(values, dtype=np.int32), return_counts=True)
            dense[int(ix), int(iy)] = float(labels[int(np.argmax(counts))])
            filled[int(ix), int(iy)] = True
        return _nearest_fill_dense_grid(dense, filled)

    sums = np.zeros((nx, ny), dtype=np.float64)
    counts = np.zeros((nx, ny), dtype=np.int32)
    for (ix, iy), value in zip(grid_indices, sample_values):
        sums[int(ix), int(iy)] += float(value)
        counts[int(ix), int(iy)] += 1
    dense = np.zeros((nx, ny), dtype=np.float64)
    filled = counts > 0
    dense[filled] = sums[filled] / counts[filled]
    return _nearest_fill_dense_grid(dense, filled)


def _densify_section_samples(
    grid_shape: tuple[int, int],
    grid_indices: np.ndarray,
    sample_values: np.ndarray,
) -> np.ndarray:
    nx, ny = grid_shape
    components = int(sample_values.shape[1])
    sums = np.zeros((nx, ny, components), dtype=np.float64)
    counts = np.zeros((nx, ny), dtype=np.int32)
    for (ix, iy), values in zip(grid_indices, sample_values):
        sums[int(ix), int(iy)] += np.asarray(values, dtype=np.float64)
        counts[int(ix), int(iy)] += 1
    dense = np.zeros((nx, ny, components), dtype=np.float64)
    filled = counts > 0
    dense[filled] = sums[filled] / counts[filled, None]
    return _nearest_fill_dense_grid(dense, filled)


def _build_dense_unode_export_payload(
    labels: np.ndarray,
    runtime_seed_unodes: dict[str, Any],
    runtime_seed_fields: dict[str, Any] | None,
    runtime_seed_unode_sections: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any] | None]:
    nx, ny = (int(labels.shape[0]), int(labels.shape[1]))
    dense_ids, dense_positions, dense_grid_indices = _dense_unode_layout(nx, ny)
    dense_unodes = {
        "ids": dense_ids,
        "positions": dense_positions,
        "grid_indices": dense_grid_indices,
        "grid_shape": (nx, ny),
    }
    sample_grid_indices = np.asarray(runtime_seed_unodes.get("grid_indices", ()), dtype=np.int32)
    dense_fields = runtime_seed_fields
    dense_sections = runtime_seed_unode_sections

    if runtime_seed_fields is not None:
        label_attribute = str(runtime_seed_fields.get("label_attribute", ""))
        source_labels = [int(value) for value in runtime_seed_fields.get("source_labels", ())]
        dense_field_values: dict[str, tuple[float, ...]] = {}
        for field_name, values in dict(runtime_seed_fields.get("values", {})).items():
            sample_values = np.asarray(values, dtype=np.float64)
            if str(field_name) == label_attribute and source_labels:
                dense_field_values[str(field_name)] = tuple(
                    float(source_labels[int(labels[ix, iy])]) for ix, iy in dense_grid_indices
                )
                continue
            integer_like = bool(
                sample_values.ndim == 1
                and sample_values.size > 0
                and float(np.max(np.abs(sample_values - np.rint(sample_values)))) <= 1.0e-6
            )
            dense_grid = _densify_scalar_samples(
                (nx, ny),
                sample_grid_indices,
                sample_values.reshape(-1),
                categorical=integer_like,
            )
            dense_field_values[str(field_name)] = tuple(
                float(dense_grid[ix, iy]) for ix, iy in dense_grid_indices
            )
        dense_fields = {
            **runtime_seed_fields,
            "values": dense_field_values,
        }

    if runtime_seed_unode_sections is not None:
        component_counts = dict(runtime_seed_unode_sections.get("component_counts", {}))
        dense_section_values: dict[str, tuple[tuple[float, ...], ...]] = {}
        label_attribute = ""
        source_labels: list[int] = []
        if dense_fields is not None:
            label_attribute = str(dense_fields.get("label_attribute", ""))
            source_labels = [int(value) for value in dense_fields.get("source_labels", ())]
        for field_name, values in dict(runtime_seed_unode_sections.get("values", {})).items():
            component_count = int(component_counts.get(str(field_name), 1))
            if str(field_name) == label_attribute and source_labels and component_count == 1:
                dense_section_values[str(field_name)] = tuple(
                    (float(source_labels[int(labels[ix, iy])]),) for ix, iy in dense_grid_indices
                )
                continue
            sample_values = np.asarray(values, dtype=np.float64)
            if sample_values.ndim == 1:
                sample_values = sample_values[:, None]
            if component_count <= 1:
                dense_grid = _densify_scalar_samples(
                    (nx, ny),
                    sample_grid_indices,
                    sample_values[:, 0],
                    categorical=bool(
                        sample_values.size > 0
                        and float(np.max(np.abs(sample_values[:, 0] - np.rint(sample_values[:, 0])))) <= 1.0e-6
                    ),
                )
                dense_section_values[str(field_name)] = tuple(
                    (float(dense_grid[ix, iy]),) for ix, iy in dense_grid_indices
                )
            else:
                dense_grid = _densify_section_samples(
                    (nx, ny),
                    sample_grid_indices,
                    sample_values[:, :component_count],
                )
                dense_section_values[str(field_name)] = tuple(
                    tuple(float(value) for value in dense_grid[ix, iy, :component_count])
                    for ix, iy in dense_grid_indices
                )
        dense_sections = {
            **runtime_seed_unode_sections,
            "id_order": dense_ids,
            "values": dense_section_values,
        }

    return dense_unodes, dense_fields, dense_sections


def _connected_components(labels: np.ndarray) -> list[dict[str, Any]]:
    nx, ny = labels.shape
    visited = np.zeros((nx, ny), dtype=bool)
    components: list[dict[str, Any]] = []

    for ix in range(nx):
        for iy in range(ny):
            if visited[ix, iy]:
                continue

            label = int(labels[ix, iy])
            stack = [(ix, iy)]
            visited[ix, iy] = True
            cells: list[tuple[int, int]] = []

            while stack:
                cx, cy = stack.pop()
                cells.append((cx, cy))

                for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nx_i = cx + dx
                    ny_i = cy + dy
                    if not (0 <= nx_i < nx and 0 <= ny_i < ny):
                        continue
                    if visited[nx_i, ny_i] or int(labels[nx_i, ny_i]) != label:
                        continue
                    visited[nx_i, ny_i] = True
                    stack.append((nx_i, ny_i))

            components.append({"label": label, "cells": cells})

    return components


def _component_boundary_edges(
    cells: list[tuple[int, int]],
    nx: int,
    ny: int,
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    cell_set = set(cells)
    edges: list[tuple[tuple[int, int], tuple[int, int]]] = []

    for ix, iy_top in cells:
        iy = ny - 1 - iy_top

        if ix == 0 or (ix - 1, iy_top) not in cell_set:
            edges.append(((ix, iy + 1), (ix, iy)))
        if ix == nx - 1 or (ix + 1, iy_top) not in cell_set:
            edges.append(((ix + 1, iy), (ix + 1, iy + 1)))
        if iy_top == 0 or (ix, iy_top - 1) not in cell_set:
            edges.append(((ix + 1, iy + 1), (ix, iy + 1)))
        if iy_top == ny - 1 or (ix, iy_top + 1) not in cell_set:
            edges.append(((ix, iy), (ix + 1, iy)))

    return edges


def _trace_loops(
    edges: list[tuple[tuple[int, int], tuple[int, int]]]
) -> list[list[tuple[int, int]]]:
    outgoing: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for start, end in edges:
        outgoing[start].append(end)

    unused = set(edges)
    loops: list[list[tuple[int, int]]] = []

    for start, end in edges:
        if (start, end) not in unused:
            continue

        loop = [start]
        current_start = start
        current_end = end

        while True:
            unused.remove((current_start, current_end))
            loop.append(current_end)
            if current_end == loop[0]:
                break

            candidates = sorted(
                candidate
                for candidate in outgoing[current_end]
                if (current_end, candidate) in unused
            )
            if not candidates:
                raise ValueError("failed to close flynn boundary loop")

            current_start = current_end
            current_end = candidates[0]

        loops.append(loop)

    return loops


def _simplify_loop(loop: list[tuple[int, int]]) -> list[tuple[int, int]]:
    points = loop[:-1] if loop and loop[0] == loop[-1] else list(loop)
    changed = True

    while changed and len(points) > 3:
        changed = False
        simplified: list[tuple[int, int]] = []
        count = len(points)

        for index in range(count):
            prev_point = points[index - 1]
            curr_point = points[index]
            next_point = points[(index + 1) % count]

            dx1 = curr_point[0] - prev_point[0]
            dy1 = curr_point[1] - prev_point[1]
            dx2 = next_point[0] - curr_point[0]
            dy2 = next_point[1] - curr_point[1]

            if dx1 * dy2 == dy1 * dx2:
                changed = True
                continue
            simplified.append(curr_point)

        points = simplified

    return points


def _polygon_area(points: list[tuple[int, int]]) -> float:
    area = 0.0
    for index, (x0, y0) in enumerate(points):
        x1, y1 = points[(index + 1) % len(points)]
        area += x0 * y1 - x1 * y0
    return 0.5 * area


def extract_flynn_topology(
    labels,
) -> tuple[list[tuple[float, float]], list[dict[str, Any]], dict[str, int]]:
    labels_np = np.asarray(labels, dtype=np.int32)
    if labels_np.ndim != 2:
        raise ValueError("expected labels to have shape (nx, ny)")

    nx, ny = labels_np.shape
    nodes: list[tuple[float, float]] = []
    node_ids: dict[tuple[int, int], int] = {}
    flynns: list[dict[str, Any]] = []
    stats = {"components": 0, "holes_skipped": 0}

    for component_index, component in enumerate(_connected_components(labels_np)):
        cells = component["cells"]
        edges = _component_boundary_edges(cells, nx, ny)
        loops = _trace_loops(edges)
        stats["components"] += 1

        outer_loops: list[tuple[float, list[tuple[int, int]]]] = []
        for loop in loops:
            simplified = _simplify_loop(loop)
            area = _polygon_area(simplified)
            if area > 0:
                outer_loops.append((area, simplified))
            elif area < 0:
                stats["holes_skipped"] += 1

        if not outer_loops:
            continue

        outer_loop = max(outer_loops, key=lambda entry: entry[0])[1]
        flynn_node_ids: list[int] = []

        for vertex in outer_loop:
            if vertex not in node_ids:
                node_ids[vertex] = len(nodes)
                nodes.append((vertex[0] / float(nx), vertex[1] / float(ny)))
            flynn_node_ids.append(node_ids[vertex])

        flynns.append(
            {
                "flynn_id": len(flynns),
                "label": int(component["label"]),
                "component_index": component_index,
                "pixel_count": len(cells),
                "node_ids": flynn_node_ids,
            }
        )

    return nodes, flynns, stats


def write_unode_elle(
    path: str | Path,
    phi,
    *,
    step: int | None = None,
    include_confidence: bool = True,
    tracked_topology: dict[str, Any] | None = None,
    mesh_state: dict[str, Any] | None = None,
) -> Path:
    """Write a minimal ELLE-style file containing UNODES and unode attributes.

    This export includes a first-pass flynn topology reconstruction from the
    dominant-grain map plus the underlying unode attributes.
    """

    phi_np = _as_numpy(phi)
    labels = dominant_grain_map(phi_np)
    confidence = phi_np.max(axis=0)
    nx, ny = labels.shape
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stats = snapshot_statistics(phi_np, step=step)
    runtime_seed_unodes = None if mesh_state is None else mesh_state.get("_runtime_seed_unodes")
    runtime_seed_fields = None if mesh_state is None else mesh_state.get("_runtime_seed_unode_fields")
    runtime_seed_node_fields = None if mesh_state is None else mesh_state.get("_runtime_seed_node_fields")
    runtime_seed_unode_sections = None if mesh_state is None else mesh_state.get("_runtime_seed_unode_sections")
    runtime_seed_node_sections = None if mesh_state is None else mesh_state.get("_runtime_seed_node_sections")
    runtime_seed_flynn_sections = None if mesh_state is None else (
        mesh_state.get("_runtime_current_flynn_sections")
        or mesh_state.get("_runtime_seed_flynn_sections")
    )
    if (
        mesh_state is not None
        and runtime_seed_unodes is not None
        and int(mesh_state.get("stats", {}).get("export_dense_unodes", 0)) == 1
    ):
        runtime_seed_unodes, runtime_seed_fields, runtime_seed_unode_sections = (
            _build_dense_unode_export_payload(
                labels,
                runtime_seed_unodes,
                runtime_seed_fields,
                runtime_seed_unode_sections,
            )
        )
    if mesh_state is not None:
        node_locations = [
            (float(node["x"]), float(node["y"]))
            for node in sorted(mesh_state["nodes"], key=lambda node: int(node["node_id"]))
        ]
        flynns = [
            {
                "flynn_id": int(flynn["flynn_id"]),
                "label": int(flynn["label"]),
                "node_ids": [int(node_id) for node_id in flynn["node_ids"]],
            }
            for flynn in mesh_state["flynns"]
        ]
        topology_stats = {
            "components": int(mesh_state["stats"]["num_flynns"]),
            "holes_skipped": int(mesh_state["stats"].get("holes_skipped", 0)),
        }
    else:
        node_locations, flynns, topology_stats = extract_flynn_topology(labels)
        tracked_by_local_index = {}
        if tracked_topology is not None:
            tracked_by_local_index = {
                int(flynn["local_index"]): flynn for flynn in tracked_topology.get("flynns", [])
            }
            for flynn in flynns:
                tracked_entry = tracked_by_local_index.get(int(flynn["component_index"]))
                if tracked_entry is not None:
                    flynn["flynn_id"] = int(tracked_entry["flynn_id"])
                    flynn["tracked_neighbors"] = list(tracked_entry.get("neighbors", []))
                    flynn["tracked_parents"] = list(tracked_entry.get("parents", []))
            flynns.sort(key=lambda flynn: int(flynn["flynn_id"]))

    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Created by python_jax_model unode export\n")
        if step is not None:
            handle.write(
                f"# step={step} active_grains={stats['active_grains']} "
                f"boundary_fraction={stats['boundary_fraction']:.6f}\n"
            )
        handle.write(
            f"# flynns={len(flynns)} components={topology_stats['components']} "
            f"holes_skipped={topology_stats['holes_skipped']}\n"
        )
        if tracked_topology is not None:
            handle.write("# tracked_topology=1\n")
        if mesh_state is not None:
            mesh_stats = mesh_state.get("stats", {})
            if int(mesh_stats.get("mesh_relaxation_steps", 0)) > 0:
                handle.write("# mesh_relaxed=1\n")
            if int(mesh_stats.get("mesh_topology_steps", 0)) > 0:
                handle.write("# mesh_topology_maintained=1\n")
            if int(mesh_stats.get("mesh_event_count", 0)) > 0:
                handle.write(
                    "# mesh_events "
                    f"switched={int(mesh_stats.get('mesh_switched_triples', 0))} "
                    f"rejected={int(mesh_stats.get('mesh_rejected_switches', 0))} "
                    f"merged={int(mesh_stats.get('mesh_merged_flynns', 0))} "
                    f"inserted={int(mesh_stats.get('mesh_inserted_nodes', 0))} "
                    f"removed={int(mesh_stats.get('mesh_removed_nodes', 0))}\n"
                )

        _write_options(handle, nx, ny)

        handle.write("FLYNNS\n")
        for flynn in flynns:
            node_id_text = " ".join(str(node_id) for node_id in flynn["node_ids"])
            handle.write(
                f"{flynn['flynn_id']} {len(flynn['node_ids'])} {node_id_text}\n"
            )

        if runtime_seed_flynn_sections is not None and runtime_seed_flynn_sections.get("field_order"):
            flynn_id_order = [
                int(value)
                for value in runtime_seed_flynn_sections.get(
                    "id_order",
                    [int(flynn["flynn_id"]) for flynn in flynns],
                )
            ]
            flynn_defaults = dict(runtime_seed_flynn_sections.get("defaults", {}))
            flynn_values = dict(runtime_seed_flynn_sections.get("values", {}))
            for field_name in runtime_seed_flynn_sections.get("field_order", ()):
                dense_values = flynn_values.get(str(field_name))
                default_value = flynn_defaults.get(str(field_name))
                if dense_values is None or default_value is None:
                    continue
                _write_sparse_numeric_section(
                    handle,
                    str(field_name),
                    flynn_id_order,
                    dense_values,
                    tuple(float(value) for value in default_value),
                )
        else:
            handle.write("F_ATTRIB_A\n")
            handle.write("Default 0.00000000e+00\n")
            for flynn in flynns:
                handle.write(
                    f"{flynn['flynn_id']} {float(flynn['label']):.8e}\n"
                )

        handle.write("LOCATION\n")
        for node_id, (x_coord, y_coord) in enumerate(node_locations):
            handle.write(f"{node_id} {x_coord:.10f} {y_coord:.10f}\n")

        if runtime_seed_node_sections is not None and runtime_seed_node_sections.get("field_order"):
            node_id_order = [
                int(value)
                for value in runtime_seed_node_sections.get(
                    "id_order",
                    list(range(len(node_locations))),
                )
            ]
            node_defaults = dict(runtime_seed_node_sections.get("defaults", {}))
            node_component_counts = dict(runtime_seed_node_sections.get("component_counts", {}))
            node_section_values = dict(runtime_seed_node_sections.get("values", {}))
            runtime_node_values = (
                dict(runtime_seed_node_fields.get("values", {}))
                if runtime_seed_node_fields is not None
                else {}
            )
            for field_name in runtime_seed_node_sections.get("field_order", ()):
                dense_values = node_section_values.get(str(field_name))
                default_value = node_defaults.get(str(field_name))
                component_count = int(node_component_counts.get(str(field_name), 1))
                if dense_values is None or default_value is None:
                    continue
                if component_count == 1 and str(field_name) in runtime_node_values:
                    dense_values = tuple(
                        (float(value),)
                        for value in runtime_node_values[str(field_name)]
                    )
                _write_sparse_numeric_section(
                    handle,
                    str(field_name),
                    node_id_order,
                    dense_values,
                    tuple(float(value) for value in default_value),
                )
        elif runtime_seed_node_fields is not None:
            field_order = [
                str(name)
                for name in runtime_seed_node_fields.get(
                    "field_order",
                    runtime_seed_node_fields["values"].keys(),
                )
            ]
            field_values = dict(runtime_seed_node_fields.get("values", {}))
            for field_name in field_order:
                if field_name not in field_values:
                    continue
                handle.write(f"{field_name}\n")
                handle.write("Default 0.00000000e+00\n")
                for node_id, value in enumerate(field_values[field_name]):
                    handle.write(f"{int(node_id)} {float(value):.8e}\n")

        handle.write("UNODES\n")
        if runtime_seed_unodes is not None:
            unode_ids = [int(unode_id) for unode_id in runtime_seed_unodes["ids"]]
            unode_positions = [
                (float(position[0]), float(position[1]))
                for position in runtime_seed_unodes["positions"]
            ]
            unode_grid_indices = [
                (int(index[0]), int(index[1]))
                for index in runtime_seed_unodes["grid_indices"]
            ]
            for unode_id, (x_coord, y_coord) in zip(unode_ids, unode_positions):
                handle.write(f"{unode_id} {x_coord:.10f} {y_coord:.10f}\n")

            if runtime_seed_unode_sections is not None and runtime_seed_unode_sections.get("field_order"):
                unode_defaults = dict(runtime_seed_unode_sections.get("defaults", {}))
                unode_component_counts = dict(runtime_seed_unode_sections.get("component_counts", {}))
                unode_section_values = dict(runtime_seed_unode_sections.get("values", {}))
                runtime_field_values = (
                    dict(runtime_seed_fields.get("values", {}))
                    if runtime_seed_fields is not None
                    else {}
                )
                runtime_label_attribute = (
                    str(runtime_seed_fields.get("label_attribute", ""))
                    if runtime_seed_fields is not None
                    else ""
                )
                for field_name in runtime_seed_unode_sections.get("field_order", ()):
                    dense_values = unode_section_values.get(str(field_name))
                    default_value = unode_defaults.get(str(field_name))
                    component_count = int(unode_component_counts.get(str(field_name), 1))
                    if dense_values is None or default_value is None:
                        continue
                    if (
                        str(field_name) == runtime_label_attribute
                        and str(field_name) in runtime_field_values
                    ):
                        dense_values = tuple(
                            (float(value),)
                            for value in runtime_field_values[str(field_name)]
                        )
                    elif component_count == 1 and str(field_name) in runtime_field_values:
                        dense_values = tuple(
                            (float(value),)
                            for value in runtime_field_values[str(field_name)]
                        )
                    _write_sparse_numeric_section(
                        handle,
                        str(field_name),
                        unode_ids,
                        dense_values,
                        tuple(float(value) for value in default_value),
                    )
            elif runtime_seed_fields is not None:
                field_order = [
                    str(name) for name in runtime_seed_fields.get("field_order", runtime_seed_fields["values"].keys())
                ]
                field_values = dict(runtime_seed_fields.get("values", {}))
                for field_name in field_order:
                    if field_name not in field_values:
                        continue
                    handle.write(f"{field_name}\n")
                    handle.write("Default 0.00000000e+00\n")
                    for unode_id, value in zip(unode_ids, field_values[field_name]):
                        handle.write(f"{int(unode_id)} {float(value):.8e}\n")
            else:
                label_attribute = "U_ATTRIB_A"
                handle.write(f"{label_attribute}\n")
                handle.write("Default 0.00000000e+00\n")
                for unode_id, (ix, iy) in zip(unode_ids, unode_grid_indices):
                    handle.write(f"{int(unode_id)} {float(labels[ix, iy]):.8e}\n")
        else:
            unode_id = 0
            for iy in range(ny):
                y_coord = 1.0 - ((iy + 0.5) / float(ny))
                for ix in range(nx):
                    x_coord = (ix + 0.5) / float(nx)
                    handle.write(f"{unode_id} {x_coord:.10f} {y_coord:.10f}\n")
                    unode_id += 1

            handle.write("U_ATTRIB_A\n")
            handle.write("Default 0.00000000e+00\n")
            unode_id = 0
            for iy in range(ny):
                for ix in range(nx):
                    handle.write(f"{unode_id} {float(labels[ix, iy]):.8e}\n")
                    unode_id += 1

            if include_confidence:
                handle.write("U_ATTRIB_B\n")
                handle.write("Default 0.00000000e+00\n")
                unode_id = 0
                for iy in range(ny):
                    for ix in range(nx):
                        handle.write(f"{unode_id} {float(confidence[ix, iy]):.8e}\n")
                        unode_id += 1

    return path
