from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .elle_visualize import (
    _field_from_sparse_unodes,
    _parse_elle_sections,
    _parse_flynns,
    _parse_sparse_values,
    _parse_unodes,
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


@dataclass(frozen=True)
class FaithfulSolverConfig:
    nx: int
    ny: int
    num_grains: int
    seed: int
    init_elle_path: str
    init_elle_attribute: str
    seed_data: FaithfulSeedData


def load_faithful_seed(
    elle_path: str | Path,
    *,
    attribute: str = "auto",
) -> FaithfulSeedData:
    path = Path(elle_path)
    sections = _parse_elle_sections(path)
    if "UNODES" not in sections:
        raise ValueError(f"ELLE file has no UNODES section: {path}")

    unodes = _parse_unodes(sections["UNODES"])
    requested_attribute = str(attribute)
    if requested_attribute.lower() == "auto":
        flynn_count = len(_parse_flynns(sections.get("FLYNNS", ())))
        attr_candidates = [
            name for name in sections if name.startswith("U_ATTRIB_") or name.startswith("U_CONC_")
        ]
        if not attr_candidates:
            raise ValueError(f"ELLE file has no unode attribute sections to seed from: {path}")

        scored_candidates: list[tuple[tuple[float, float, float, str], str]] = []
        integer_like_candidates: set[str] = set()
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
        else:
            raise ValueError(f"could not auto-detect an integer-like ELLE grain-label attribute in {path}")
    elif requested_attribute not in sections:
        raise ValueError(f"ELLE file has no {requested_attribute} section: {path}")

    field, grid_shape, layout = _field_from_sparse_unodes(unodes, sections[requested_attribute])
    label_field = np.rint(np.asarray(field, dtype=np.float64)).astype(np.int32)
    unique_labels = sorted(int(label) for label in np.unique(label_field))
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
    )
