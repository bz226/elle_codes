from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .artifacts import dominant_grain_map
from .mesh import MeshFeedbackConfig
from .simulation import GrainGrowthConfig, run_simulation


def _as_labels(data) -> np.ndarray:
    array = np.asarray(data)
    if array.ndim == 3:
        return dominant_grain_map(array)
    if array.ndim == 2:
        return array.astype(np.int32)
    raise ValueError("expected phi with shape (num_grains, nx, ny) or labels with shape (nx, ny)")


def _extract_components(labels: np.ndarray) -> tuple[list[dict[str, Any]], np.ndarray]:
    nx, ny = labels.shape
    visited = np.zeros((nx, ny), dtype=bool)
    component_map = np.full((nx, ny), -1, dtype=np.int32)
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
                component_map[cx, cy] = len(components)

                for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nx_i = cx + dx
                    ny_i = cy + dy
                    if not (0 <= nx_i < nx and 0 <= ny_i < ny):
                        continue
                    if visited[nx_i, ny_i] or int(labels[nx_i, ny_i]) != label:
                        continue
                    visited[nx_i, ny_i] = True
                    stack.append((nx_i, ny_i))

            xs = np.array([cell[0] for cell in cells], dtype=np.float64)
            ys = np.array([cell[1] for cell in cells], dtype=np.float64)
            components.append(
                {
                    "local_index": len(components),
                    "label": label,
                    "cells": cells,
                    "area": len(cells),
                    "bbox": [
                        int(xs.min()),
                        int(ys.min()),
                        int(xs.max()),
                        int(ys.max()),
                    ],
                    "centroid": [
                        float(xs.mean() + 0.5),
                        float(ys.mean() + 0.5),
                    ],
                }
            )

    return components, component_map


def _component_adjacency(component_map: np.ndarray) -> dict[int, set[int]]:
    adjacency: dict[int, set[int]] = {
        int(index): set() for index in np.unique(component_map) if int(index) >= 0
    }
    nx, ny = component_map.shape

    for ix in range(nx):
        for iy in range(ny):
            current = int(component_map[ix, iy])
            if current < 0:
                continue
            for dx, dy in ((1, 0), (0, 1)):
                nx_i = ix + dx
                ny_i = iy + dy
                if not (0 <= nx_i < nx and 0 <= ny_i < ny):
                    continue
                other = int(component_map[nx_i, ny_i])
                if other >= 0 and other != current:
                    adjacency[current].add(other)
                    adjacency.setdefault(other, set()).add(current)

    return adjacency


def _to_serializable(snapshot: dict[str, Any]) -> dict[str, Any]:
    serializable = dict(snapshot)
    serializable["flynns"] = [dict(flynn) for flynn in snapshot["flynns"]]
    serializable["events"] = {
        key: [dict(item) if isinstance(item, dict) else item for item in value]
        for key, value in snapshot["events"].items()
    }
    return serializable


@dataclass
class TopologyTracker:
    next_flynn_id: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)
    _previous_flynn_id_map: np.ndarray | None = None
    _previous_active_ids: set[int] = field(default_factory=set)

    def update(self, data, step: int) -> dict[str, Any]:
        labels = _as_labels(data)
        nx, ny = labels.shape
        components, component_map = _extract_components(labels)
        adjacency = _component_adjacency(component_map)

        current_flynn_id_map = np.full((nx, ny), -1, dtype=np.int32)
        overlaps_by_component: list[dict[int, int]] = []
        previous_to_current: dict[int, list[int]] = {}

        if self._previous_flynn_id_map is None:
            overlaps_by_component = [{} for _ in components]
        else:
            for component in components:
                overlap_counts: dict[int, int] = {}
                for ix, iy in component["cells"]:
                    previous_id = int(self._previous_flynn_id_map[ix, iy])
                    if previous_id >= 0:
                        overlap_counts[previous_id] = overlap_counts.get(previous_id, 0) + 1
                overlaps_by_component.append(overlap_counts)
                for previous_id, count in overlap_counts.items():
                    if count > 0:
                        previous_to_current.setdefault(previous_id, []).append(component["local_index"])

        requested_matches: list[tuple[int, int, int]] = []
        for component in components:
            overlap_counts = overlaps_by_component[component["local_index"]]
            if overlap_counts:
                previous_id, pixels = max(
                    overlap_counts.items(),
                    key=lambda item: (item[1], -item[0]),
                )
                requested_matches.append((pixels, component["local_index"], previous_id))

        retained_matches: dict[int, int] = {}
        used_previous_ids: set[int] = set()
        for _, local_index, previous_id in sorted(requested_matches, reverse=True):
            if previous_id in used_previous_ids:
                continue
            retained_matches[local_index] = previous_id
            used_previous_ids.add(previous_id)

        assigned_ids: dict[int, int] = {}
        for component in components:
            local_index = component["local_index"]
            if local_index in retained_matches:
                assigned_ids[local_index] = retained_matches[local_index]
            else:
                assigned_ids[local_index] = self.next_flynn_id
                self.next_flynn_id += 1

        flynns: list[dict[str, Any]] = []
        births: list[dict[str, Any]] = []
        merges: list[dict[str, Any]] = []

        for component in components:
            local_index = component["local_index"]
            overlap_counts = overlaps_by_component[local_index]
            parents = sorted(
                overlap_counts,
                key=lambda previous_id: (-overlap_counts[previous_id], previous_id),
            )

            flynn_id = assigned_ids[local_index]
            retained_identity = local_index in retained_matches

            for ix, iy in component["cells"]:
                current_flynn_id_map[ix, iy] = flynn_id

            neighbor_ids = sorted(
                {
                    assigned_ids[neighbor_index]
                    for neighbor_index in adjacency.get(local_index, set())
                    if neighbor_index in assigned_ids and assigned_ids[neighbor_index] != flynn_id
                }
            )

            record = {
                "flynn_id": int(flynn_id),
                "local_index": int(local_index),
                "label": int(component["label"]),
                "area": int(component["area"]),
                "area_fraction": float(component["area"] / float(nx * ny)),
                "bbox": list(component["bbox"]),
                "centroid": list(component["centroid"]),
                "parents": [int(parent) for parent in parents],
                "retained_identity": bool(retained_identity),
                "overlap_pixels": int(overlap_counts.get(flynn_id, 0)),
                "neighbors": [int(neighbor_id) for neighbor_id in neighbor_ids],
            }
            flynns.append(record)

            if not parents:
                births.append({"flynn_id": int(flynn_id), "label": int(component["label"])})
            if len(parents) > 1:
                merges.append(
                    {
                        "flynn_id": int(flynn_id),
                        "parents": [int(parent) for parent in parents],
                    }
                )

        current_active_ids = {flynn["flynn_id"] for flynn in flynns}
        deaths = sorted(self._previous_active_ids - set(previous_to_current))
        splits = []
        for previous_id, local_indices in sorted(previous_to_current.items()):
            if len(local_indices) > 1:
                children = sorted(
                    next(
                        flynn["flynn_id"]
                        for flynn in flynns
                        if flynn["local_index"] == local_index
                    )
                    for local_index in local_indices
                )
                splits.append({"parent": int(previous_id), "children": [int(child) for child in children]})

        snapshot = {
            "step": int(step),
            "num_components": int(len(flynns)),
            "active_flynn_ids": [int(flynn_id) for flynn_id in sorted(current_active_ids)],
            "events": {
                "births": births,
                "deaths": [{"flynn_id": int(flynn_id)} for flynn_id in deaths],
                "splits": splits,
                "merges": merges,
            },
            "flynns": sorted(flynns, key=lambda flynn: flynn["flynn_id"]),
        }

        self._previous_flynn_id_map = current_flynn_id_map
        self._previous_active_ids = current_active_ids
        self.history.append(_to_serializable(snapshot))
        return snapshot


def write_topology_snapshot(path: str | Path, snapshot: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_to_serializable(snapshot), handle, indent=2)
    return path


def write_topology_history(path: str | Path, history: list[dict[str, Any]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump([_to_serializable(snapshot) for snapshot in history], handle, indent=2)
    return path


def run_simulation_with_topology(
    config: GrainGrowthConfig,
    steps: int,
    save_every: int,
    on_snapshot: Callable[[int, object, dict[str, Any], dict[str, Any] | None], None] | None = None,
    mesh_feedback: MeshFeedbackConfig | None = None,
) -> tuple[object, list[object], list[dict[str, Any]]]:
    tracker = TopologyTracker()
    runtime_history: list[dict[str, Any]] = []

    def _wrapped_snapshot(step: int, phi, mesh_feedback_context: dict[str, Any] | None = None) -> None:
        topology_snapshot = None
        if mesh_feedback_context is not None:
            topology_snapshot = mesh_feedback_context.get("tracked_topology_snapshot")
        if topology_snapshot is None:
            topology_snapshot = tracker.update(phi, step)
        else:
            runtime_history.append(_to_serializable(topology_snapshot))
        if on_snapshot is not None:
            on_snapshot(step, phi, topology_snapshot, mesh_feedback_context)

    final_state, snapshots = run_simulation(
        config=config,
        steps=steps,
        save_every=save_every,
        on_snapshot=_wrapped_snapshot,
        mesh_feedback=mesh_feedback,
    )
    return final_state, snapshots, runtime_history or tracker.history
