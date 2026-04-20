from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .elle_visualize import (
    _field_from_sparse_unodes,
    _layout_from_unodes,
    _parse_elle_sections,
    _parse_flynns,
    _parse_location,
    _parse_sparse_values,
    _parse_unodes,
)


@dataclass(frozen=True)
class FaithfulElleOptions:
    scalar_values: dict[str, float]
    cell_bounding_box: tuple[tuple[float, float], ...] = (
        (0.0, 0.0),
        (1.0, 0.0),
        (1.0, 1.0),
        (0.0, 1.0),
    )
    simple_shear_offset: float = 0.0
    cumulative_simple_shear: float = 0.0

    def get(self, name: str, default: float | None = None) -> float | None:
        value = self.scalar_values.get(str(name))
        if value is None:
            return default
        return float(value)

    def with_scalar_overrides(self, overrides: dict[str, float | None]) -> "FaithfulElleOptions":
        scalar_values = {str(name): float(value) for name, value in self.scalar_values.items()}
        for name, value in dict(overrides).items():
            if value is None:
                continue
            scalar_values[str(name)] = float(value)
        return FaithfulElleOptions(
            scalar_values=scalar_values,
            cell_bounding_box=tuple(
                (float(x_coord), float(y_coord)) for x_coord, y_coord in self.cell_bounding_box
            ),
            simple_shear_offset=float(self.simple_shear_offset),
            cumulative_simple_shear=float(self.cumulative_simple_shear),
        )

    def to_runtime_dict(self) -> dict[str, Any]:
        return {
            "scalar_values": {
                str(name): float(value) for name, value in self.scalar_values.items()
            },
            "cell_bounding_box": [
                [float(x_coord), float(y_coord)]
                for x_coord, y_coord in self.cell_bounding_box
            ],
            "simple_shear_offset": float(self.simple_shear_offset),
            "cumulative_simple_shear": float(self.cumulative_simple_shear),
        }

    @classmethod
    def from_runtime_dict(cls, payload: dict[str, Any] | None) -> "FaithfulElleOptions":
        payload = dict(payload or {})
        scalar_values = {
            str(name): float(value)
            for name, value in dict(payload.get("scalar_values", {})).items()
        }
        raw_box = payload.get("cell_bounding_box", ())
        box = tuple(
            (float(point[0]), float(point[1]))
            for point in raw_box
            if isinstance(point, (list, tuple)) and len(point) >= 2
        )
        if len(box) != 4:
            box = (
                (0.0, 0.0),
                (1.0, 0.0),
                (1.0, 1.0),
                (0.0, 1.0),
            )
        return cls(
            scalar_values=scalar_values,
            cell_bounding_box=box,
            simple_shear_offset=float(payload.get("simple_shear_offset", 0.0)),
            cumulative_simple_shear=float(payload.get("cumulative_simple_shear", 0.0)),
        )


@dataclass(frozen=True)
class FaithfulSeedData:
    path: str
    attribute: str
    label_field: np.ndarray
    source_labels: tuple[int, ...]
    grid_shape: tuple[int, int]
    num_labels: int
    unode_ids: tuple[int, ...]
    unode_positions: tuple[tuple[float, float], ...]
    unode_grid_indices: tuple[tuple[int, int], ...]
    unode_field_values: dict[str, tuple[float, ...]]
    unode_field_order: tuple[str, ...]
    elle_options: FaithfulElleOptions


@dataclass(frozen=True)
class FaithfulSolverConfig:
    nx: int
    ny: int
    num_grains: int
    seed: int
    init_elle_path: str
    init_elle_attribute: str
    seed_data: FaithfulSeedData


def _parse_float_pair(parts: list[str]) -> tuple[float, float] | None:
    if len(parts) < 2:
        return None
    try:
        return float(parts[0]), float(parts[1])
    except ValueError:
        return None


def parse_faithful_elle_options(lines: tuple[str, ...]) -> FaithfulElleOptions:
    cleaned_lines = [
        raw_line.strip()
        for raw_line in lines
        if raw_line.strip() and not raw_line.strip().startswith("#")
    ]
    scalar_values: dict[str, float] = {}
    cell_bounding_box: tuple[tuple[float, float], ...] = (
        (0.0, 0.0),
        (1.0, 0.0),
        (1.0, 1.0),
        (0.0, 1.0),
    )
    simple_shear_offset = 0.0
    cumulative_simple_shear = 0.0

    index = 0
    while index < len(cleaned_lines):
        parts = cleaned_lines[index].split()
        key = str(parts[0])
        if key == "CellBoundingBox":
            points: list[tuple[float, float]] = []
            first_pair = _parse_float_pair(parts[1:])
            if first_pair is not None:
                points.append(first_pair)
            lookahead = index + 1
            while lookahead < len(cleaned_lines) and len(points) < 4:
                candidate = _parse_float_pair(cleaned_lines[lookahead].split())
                if candidate is None:
                    break
                points.append(candidate)
                lookahead += 1
            if len(points) == 4:
                cell_bounding_box = tuple(points)
                index = lookahead
                continue
        elif len(parts) >= 2:
            try:
                value = float(parts[1])
            except ValueError:
                value = None
            if value is not None:
                if key == "SimpleShearOffset":
                    simple_shear_offset = float(value)
                elif key == "CumulativeSimpleShear":
                    cumulative_simple_shear = float(value)
                else:
                    scalar_values[key] = float(value)
        index += 1

    return FaithfulElleOptions(
        scalar_values=scalar_values,
        cell_bounding_box=cell_bounding_box,
        simple_shear_offset=float(simple_shear_offset),
        cumulative_simple_shear=float(cumulative_simple_shear),
    )


def _build_seed_mesh_state_from_sections(
    sections: dict[str, tuple[str, ...]],
    unodes: tuple[tuple[int, float, float], ...],
) -> dict[str, Any]:
    if "LOCATION" not in sections or "FLYNNS" not in sections:
        raise ValueError("polygon-derived faithful seeds require LOCATION and FLYNNS sections")

    layout = _layout_from_unodes(unodes)
    location_map = _parse_location(sections["LOCATION"])
    ordered_node_ids = sorted(int(node_id) for node_id in location_map)
    node_id_map = {int(node_id): index for index, node_id in enumerate(ordered_node_ids)}
    flynn_records = _parse_flynns(sections["FLYNNS"])

    flynn_label_values: dict[int, int] = {}
    if "F_ATTRIB_C" in sections:
        _, flynn_attr_values = _parse_sparse_values(sections["F_ATTRIB_C"])
        flynn_label_values = {
            int(flynn_id): int(round(value)) for flynn_id, value in flynn_attr_values.items()
        }

    return {
        "nodes": [
            {
                "x": float(location_map[node_id][0]),
                "y": float(location_map[node_id][1]),
            }
            for node_id in ordered_node_ids
        ],
        "flynns": [
            {
                "flynn_id": int(flynn["flynn_id"]),
                "label": int(
                    flynn_label_values.get(int(flynn["flynn_id"]), int(flynn["flynn_id"]))
                ),
                "node_ids": [
                    int(node_id_map[int(node_id)])
                    for node_id in flynn["node_ids"]
                    if int(node_id) in node_id_map
                ],
            }
            for flynn in flynn_records
            if len(
                [node_id for node_id in flynn["node_ids"] if int(node_id) in node_id_map]
            )
            >= 3
        ],
        "stats": {"grid_shape": [int(layout["grid_shape"][0]), int(layout["grid_shape"][1])]},
        "events": [],
    }


def _field_from_flynn_polygons(
    sections: dict[str, tuple[str, ...]],
    unodes: tuple[tuple[int, float, float], ...],
) -> tuple[np.ndarray, tuple[int, int], dict[str, Any]]:
    ordered_unodes = sorted((int(unode_id), float(x), float(y)) for unode_id, x, y in unodes)
    layout = _layout_from_unodes(unodes)
    seed_unodes = {
        "grid_shape": tuple(int(value) for value in layout["grid_shape"]),
        "positions": tuple((float(x), float(y)) for _, x, y in ordered_unodes),
        "grid_indices": tuple(
            (
                int(layout["id_lookup"][int(unode_id)][0]),
                int(layout["id_lookup"][int(unode_id)][1]),
            )
            for unode_id, _, _ in ordered_unodes
        ),
    }
    mesh_state = _build_seed_mesh_state_from_sections(sections, unodes)
    from .mesh import assign_seed_unodes_from_mesh

    field, _ = assign_seed_unodes_from_mesh(mesh_state, seed_unodes)
    return np.asarray(field, dtype=np.int32), tuple(int(value) for value in layout["grid_shape"]), layout


def _all_flynn_source_labels(sections: dict[str, tuple[str, ...]]) -> tuple[int, ...]:
    flynn_records = _parse_flynns(sections.get("FLYNNS", ()))
    flynn_ids = {int(record["flynn_id"]) for record in flynn_records}
    if "F_ATTRIB_C" in sections:
        _, flynn_attr_values = _parse_sparse_values(sections["F_ATTRIB_C"])
        flynn_ids.update(int(round(value)) for value in flynn_attr_values.values())
    return tuple(sorted(flynn_ids))


def load_faithful_seed(
    elle_path: str | Path,
    *,
    attribute: str = "auto",
) -> FaithfulSeedData:
    path = Path(elle_path)
    sections = _parse_elle_sections(path)
    if "UNODES" not in sections:
        raise ValueError(f"ELLE file has no UNODES section: {path}")
    elle_options = parse_faithful_elle_options(sections.get("OPTIONS", ()))

    unodes = _parse_unodes(sections["UNODES"])
    requested_attribute = str(attribute)
    if requested_attribute.lower() == "auto":
        flynn_count = len(_parse_flynns(sections.get("FLYNNS", ())))
        attr_candidates = [
            name for name in sections if name.startswith("U_ATTRIB_") or name.startswith("U_CONC_")
        ]
        if not attr_candidates:
            if "LOCATION" in sections and "FLYNNS" in sections:
                requested_attribute = "derived_from_flynns"
            else:
                raise ValueError(f"ELLE file has no unode attribute sections to seed from: {path}")

        scored_candidates: list[tuple[tuple[float, float, float, str], str]] = []
        integer_like_candidates: set[str] = set()
        if requested_attribute != "derived_from_flynns":
            for name in attr_candidates:
                field_np, _, _ = _field_from_sparse_unodes(unodes, sections[name])
                array = np.asarray(field_np, dtype=np.float64)
                rounded = np.rint(array)
                integer_residual = float(np.max(np.abs(array - rounded))) if array.size else float("inf")
                unique_labels = sorted(int(value) for value in np.unique(rounded))
                unique_count = len(unique_labels)
                if unique_count <= 1:
                    continue
                if integer_residual <= 5.1e-1:
                    integer_like_candidates.add(str(name))
                    closeness = abs(unique_count - flynn_count) if flynn_count > 0 else float(unique_count)
                    scored_candidates.append(((closeness, integer_residual, float(unique_count), name), name))

            if "U_ATTRIB_C" in integer_like_candidates:
                requested_attribute = "U_ATTRIB_C"
            elif scored_candidates:
                scored_candidates.sort(key=lambda item: item[0])
                requested_attribute = scored_candidates[0][1]
            elif "LOCATION" in sections and "FLYNNS" in sections:
                requested_attribute = "derived_from_flynns"
            else:
                raise ValueError(f"could not auto-detect an integer-like ELLE grain-label attribute in {path}")
    elif requested_attribute != "derived_from_flynns" and requested_attribute not in sections:
        raise ValueError(f"ELLE file has no {requested_attribute} section: {path}")

    if requested_attribute == "derived_from_flynns":
        field, grid_shape, layout = _field_from_flynn_polygons(sections, unodes)
    else:
        field, grid_shape, layout = _field_from_sparse_unodes(unodes, sections[requested_attribute])
    label_field = np.rint(np.asarray(field, dtype=np.float64)).astype(np.int32)
    unique_labels = sorted(int(label) for label in np.unique(label_field))
    if requested_attribute == "derived_from_flynns":
        # Raw polygon seeds should preserve the full flynn label surface from the
        # ELLE file, even if a few flynns start with zero occupied unodes in the
        # rasterized ownership map. The legacy seed still contains those flynns,
        # and dropping them here makes the initial phi state disagree with the
        # preserved mesh seed before the first faithful outer step starts.
        unique_labels = sorted(set(unique_labels) | set(_all_flynn_source_labels(sections)))
    label_to_index = {label: index for index, label in enumerate(unique_labels)}
    compact = np.vectorize(label_to_index.get, otypes=[np.int32])(label_field)
    id_lookup = layout["id_lookup"]
    ordered_unodes = sorted((int(unode_id), float(x_coord), float(y_coord)) for unode_id, x_coord, y_coord in unodes)
    unode_field_values: dict[str, tuple[float, ...]] = {}
    field_order: list[str] = []
    for section_name in sections:
        if not section_name.startswith("U_"):
            continue
        if section_name == "UNODES":
            continue
        try:
            default_value, sparse_values = _parse_sparse_values(sections[section_name])
        except Exception:
            continue
        unode_field_values[section_name] = tuple(
            float(sparse_values.get(int(unode_id), default_value))
            for unode_id, _, _ in ordered_unodes
        )
        field_order.append(section_name)

    return FaithfulSeedData(
        path=str(path),
        attribute=requested_attribute,
        label_field=compact,
        source_labels=tuple(unique_labels),
        grid_shape=tuple(int(value) for value in grid_shape),
        num_labels=int(len(unique_labels)),
        unode_ids=tuple(int(unode_id) for unode_id, _, _ in ordered_unodes),
        unode_positions=tuple((float(x_coord), float(y_coord)) for _, x_coord, y_coord in ordered_unodes),
        unode_grid_indices=tuple(
            (
                int(id_lookup[int(unode_id)][0]),
                int(id_lookup[int(unode_id)][1]),
            )
            for unode_id, _, _ in ordered_unodes
        ),
        unode_field_values=unode_field_values,
        unode_field_order=tuple(field_order),
        elle_options=elle_options,
    )
