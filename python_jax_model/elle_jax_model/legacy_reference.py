from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .elle_visualize import (
    _layout_from_unodes,
    _parse_flynns,
    _parse_location,
    _parse_sparse_values,
    _parse_unodes,
    _parse_elle_sections,
)
from .faithful_config import load_faithful_seed
from .mesh import assign_seed_unodes_from_mesh


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def _parse_numeric_sparse_records(lines: tuple[str, ...]) -> tuple[tuple[float, ...], dict[int, tuple[float, ...]]] | None:
    default: tuple[float, ...] | None = None
    values: dict[int, tuple[float, ...]] = {}
    component_count: int | None = None
    for raw_line in lines:
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if parts[0] == "Default" and len(parts) > 1:
            try:
                parsed = tuple(float(token) for token in parts[1:])
            except ValueError:
                return None
            default = parsed
            component_count = len(parsed)
        elif len(parts) >= 2:
            try:
                parsed = tuple(float(token) for token in parts[1:])
            except ValueError:
                return None
            values[int(parts[0])] = parsed
            component_count = len(parsed)
    if component_count is None:
        return None
    if default is None:
        default = tuple(0.0 for _ in range(component_count))
    return default, values


def _dense_numeric_field(
    default: tuple[float, ...],
    values: dict[int, tuple[float, ...]],
    ordered_ids: list[int],
) -> np.ndarray:
    field = np.empty((len(ordered_ids), len(default)), dtype=np.float64)
    default_array = np.asarray(default, dtype=np.float64)
    for index, entity_id in enumerate(ordered_ids):
        record = values.get(int(entity_id))
        if record is None:
            field[index] = default_array
        else:
            field[index] = np.asarray(record, dtype=np.float64)
    return field


def _component_stats(field: np.ndarray) -> list[dict[str, float]]:
    stats: list[dict[str, float]] = []
    if field.ndim != 2:
        return stats
    for comp in range(field.shape[1]):
        values = field[:, comp]
        stats.append(
            {
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
            }
        )
    return stats


def _periodic_component_areas(labels: np.ndarray) -> np.ndarray:
    labels_np = np.asarray(labels, dtype=np.int32)
    nx, ny = labels_np.shape
    visited = np.zeros((nx, ny), dtype=bool)
    areas: list[float] = []
    total_pixels = float(nx * ny)

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

            areas.append(float(pixel_count) / total_pixels)

    return np.asarray(areas, dtype=np.float64)


def _build_mesh_seed_state(
    sections: dict[str, tuple[str, ...]],
    layout: dict[str, Any],
) -> dict[str, Any]:
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
                "label": int(flynn_label_values.get(int(flynn["flynn_id"]), int(flynn["flynn_id"]))),
                "node_ids": [
                    int(node_id_map[int(node_id)])
                    for node_id in flynn["node_ids"]
                    if int(node_id) in node_id_map
                ],
            }
            for flynn in flynn_records
            if len([node_id for node_id in flynn["node_ids"] if int(node_id) in node_id_map]) >= 3
        ],
        "stats": {"grid_shape": [int(layout["grid_shape"][0]), int(layout["grid_shape"][1])]},
        "events": [],
    }


def _derive_unode_labels_from_flynns(
    sections: dict[str, tuple[str, ...]],
    unodes: tuple[tuple[int, float, float], ...],
) -> tuple[np.ndarray, str]:
    layout = _layout_from_unodes(unodes)
    ordered_unodes = sorted((int(unode_id), float(x), float(y)) for unode_id, x, y in unodes)
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
    mesh_state = _build_mesh_seed_state(sections, layout)
    labels, _ = assign_seed_unodes_from_mesh(mesh_state, seed_unodes)
    return labels, "derived_from_flynns"


def _label_summary(
    elle_path: str | Path,
    sections: dict[str, tuple[str, ...]],
    unodes: tuple[tuple[int, float, float], ...],
    attribute: str,
) -> dict[str, Any]:
    label_attribute = str(attribute)
    label_source = "attribute"
    try:
        seed = load_faithful_seed(elle_path, attribute=label_attribute)
        labels = np.asarray(seed.label_field, dtype=np.int32)
        label_attribute = str(seed.attribute)
    except Exception:
        labels, label_source = _derive_unode_labels_from_flynns(sections, unodes)
        label_attribute = "derived_from_flynns"

    areas = _periodic_component_areas(labels)
    return {
        "source": label_source,
        "attribute": label_attribute,
        "grid_shape": [int(labels.shape[0]), int(labels.shape[1])],
        "label_hash": _sha256_bytes(np.asarray(labels, dtype=np.int32).tobytes()),
        "grain_count": int(areas.size),
        "mean_grain_area": float(np.mean(areas)) if areas.size else 0.0,
        "std_grain_area": float(np.std(areas)) if areas.size else 0.0,
        "grain_area_hash": _sha256_bytes(np.asarray(areas, dtype=np.float64).tobytes()),
    }


def _field_domain_ids(
    section_name: str,
    ordered_unode_ids: list[int],
    ordered_node_ids: list[int],
    ordered_flynn_ids: list[int],
) -> tuple[str, list[int]] | None:
    if section_name.startswith("U_"):
        return "unode", ordered_unode_ids
    if section_name.startswith("N_"):
        return "node", ordered_node_ids
    if section_name.startswith("F_") or section_name in {"EULER_1", "EULER_2", "EULER_3", "DISLOCDEN"}:
        return "flynn", ordered_flynn_ids
    return None


def _extract_legacy_dense_state(
    elle_path: str | Path,
    *,
    label_attribute: str = "auto",
) -> dict[str, Any]:
    path = Path(elle_path)
    sections = _parse_elle_sections(path)
    if "LOCATION" not in sections or "FLYNNS" not in sections or "UNODES" not in sections:
        raise ValueError(f"reference snapshot requires LOCATION, FLYNNS, and UNODES sections: {path}")

    unodes = _parse_unodes(sections["UNODES"])
    layout = _layout_from_unodes(unodes)
    ordered_unodes = sorted((int(unode_id), float(x), float(y)) for unode_id, x, y in unodes)
    ordered_unode_ids = [int(unode_id) for unode_id, _, _ in ordered_unodes]

    location_map = _parse_location(sections["LOCATION"])
    ordered_node_ids = sorted(int(node_id) for node_id in location_map)

    flynn_records = _parse_flynns(sections["FLYNNS"])
    ordered_flynn_ids = sorted(int(flynn["flynn_id"]) for flynn in flynn_records)

    labels_summary = _label_summary(path, sections, unodes, label_attribute)
    label_field = None
    try:
        seed = load_faithful_seed(path, attribute=str(labels_summary["attribute"]))
        label_field = np.asarray(seed.label_field, dtype=np.int32)
    except Exception:
        if str(labels_summary["attribute"]) == "derived_from_flynns":
            label_field, _ = _derive_unode_labels_from_flynns(sections, unodes)
        else:
            label_field = None

    dense_fields: dict[str, np.ndarray] = {}
    field_metadata: dict[str, dict[str, Any]] = {}
    for section_name, lines in sections.items():
        domain = _field_domain_ids(section_name, ordered_unode_ids, ordered_node_ids, ordered_flynn_ids)
        if domain is None:
            continue
        parsed = _parse_numeric_sparse_records(lines)
        if parsed is None:
            continue
        default, values = parsed
        entity_type, ordered_ids = domain
        dense_field = _dense_numeric_field(default, values, ordered_ids)
        dense_fields[str(section_name)] = np.asarray(dense_field, dtype=np.float64)
        field_metadata[str(section_name)] = {
            "entity_type": str(entity_type),
            "component_count": int(dense_field.shape[1]),
            "record_count": int(dense_field.shape[0]),
        }

    return {
        "path": str(path),
        "sections": sections,
        "labels_summary": labels_summary,
        "labels": label_field,
        "ordered_unode_ids": ordered_unode_ids,
        "unode_grid_indices": tuple(
            (
                int(layout["id_lookup"][int(unode_id)][0]),
                int(layout["id_lookup"][int(unode_id)][1]),
            )
            for unode_id, _, _ in ordered_unodes
        ),
        "dense_fields": dense_fields,
        "field_metadata": field_metadata,
    }


def _legacy_swept_unode_mask(
    before_state: dict[str, Any],
    after_state: dict[str, Any],
) -> np.ndarray | None:
    before_labels = before_state.get("labels")
    after_labels = after_state.get("labels")
    if before_labels is None or after_labels is None:
        return None
    before_labels_np = np.asarray(before_labels, dtype=np.int32)
    after_labels_np = np.asarray(after_labels, dtype=np.int32)
    if before_labels_np.shape != after_labels_np.shape:
        return None

    unode_grid_indices = np.asarray(before_state.get("unode_grid_indices", ()), dtype=np.int32)
    if unode_grid_indices.ndim != 2 or unode_grid_indices.shape[1] != 2:
        return None
    if unode_grid_indices.shape[0] == 0:
        return np.zeros((0,), dtype=bool)

    changed_grid = before_labels_np != after_labels_np
    return np.asarray(
        changed_grid[unode_grid_indices[:, 0], unode_grid_indices[:, 1]],
        dtype=bool,
    )


def extract_legacy_reference_snapshot(
    elle_path: str | Path,
    *,
    checkpoint_name: str | None = None,
    label_attribute: str = "auto",
) -> dict[str, Any]:
    path = Path(elle_path)
    sections = _parse_elle_sections(path)
    if "LOCATION" not in sections or "FLYNNS" not in sections or "UNODES" not in sections:
        raise ValueError(f"reference snapshot requires LOCATION, FLYNNS, and UNODES sections: {path}")

    unodes = _parse_unodes(sections["UNODES"])
    ordered_unodes = sorted((int(unode_id), float(x), float(y)) for unode_id, x, y in unodes)
    ordered_unode_ids = [int(unode_id) for unode_id, _, _ in ordered_unodes]

    location_map = _parse_location(sections["LOCATION"])
    ordered_node_ids = sorted(int(node_id) for node_id in location_map)

    flynn_records = _parse_flynns(sections["FLYNNS"])
    ordered_flynn_ids = sorted(int(flynn["flynn_id"]) for flynn in flynn_records)

    node_positions = [
        [float(location_map[node_id][0]), float(location_map[node_id][1])]
        for node_id in ordered_node_ids
    ]
    flynn_connectivity = [
        {
            "flynn_id": int(flynn["flynn_id"]),
            "node_ids": [int(node_id) for node_id in flynn["node_ids"]],
        }
        for flynn in sorted(flynn_records, key=lambda item: int(item["flynn_id"]))
    ]

    field_summaries: dict[str, dict[str, Any]] = {}
    skipped_sections: list[str] = []
    for section_name, lines in sections.items():
        domain = _field_domain_ids(section_name, ordered_unode_ids, ordered_node_ids, ordered_flynn_ids)
        if domain is None:
            continue
        parsed = _parse_numeric_sparse_records(lines)
        if parsed is None:
            skipped_sections.append(str(section_name))
            continue
        default, values = parsed
        entity_type, ordered_ids = domain
        dense_field = _dense_numeric_field(default, values, ordered_ids)
        field_summaries[str(section_name)] = {
            "entity_type": entity_type,
            "component_count": int(dense_field.shape[1]),
            "default": [float(value) for value in default],
            "record_count": int(len(ordered_ids)),
            "explicit_count": int(len(values)),
            "component_stats": _component_stats(dense_field),
            "value_hash": _sha256_bytes(dense_field.tobytes()),
        }

    return {
        "source_path": str(path),
        "checkpoint_name": str(checkpoint_name or path.stem),
        "mesh": {
            "num_nodes": int(len(ordered_node_ids)),
            "num_flynns": int(len(ordered_flynn_ids)),
            "node_position_hash": _sha256_json(node_positions),
            "flynn_connectivity_hash": _sha256_json(flynn_connectivity),
        },
        "label_summary": _label_summary(path, sections, unodes, label_attribute),
        "field_summaries": field_summaries,
        "skipped_sections": sorted(skipped_sections),
        "section_names": [str(name) for name in sections.keys()],
    }


def extract_legacy_reference_transition(
    before_elle_path: str | Path,
    after_elle_path: str | Path,
    *,
    checkpoint_name: str | None = None,
    label_attribute: str = "auto",
    field_names: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    before_state = _extract_legacy_dense_state(before_elle_path, label_attribute=label_attribute)
    after_state = _extract_legacy_dense_state(after_elle_path, label_attribute=label_attribute)

    selected_fields = (
        sorted(set(before_state["dense_fields"]) & set(after_state["dense_fields"]))
        if field_names is None
        else [str(name) for name in field_names if str(name) in before_state["dense_fields"] and str(name) in after_state["dense_fields"]]
    )

    field_transitions: dict[str, dict[str, Any]] = {}
    for name in selected_fields:
        before_field = np.asarray(before_state["dense_fields"][name], dtype=np.float64)
        after_field = np.asarray(after_state["dense_fields"][name], dtype=np.float64)
        if before_field.shape != after_field.shape:
            field_transitions[str(name)] = {
                "shape_matches": False,
                "before_shape": list(before_field.shape),
                "after_shape": list(after_field.shape),
            }
            continue
        delta = np.asarray(after_field - before_field, dtype=np.float64)
        row_active = np.any(np.abs(delta) > 1.0e-12, axis=1)
        field_transitions[str(name)] = {
            "shape_matches": True,
            "entity_type": str(before_state["field_metadata"][name]["entity_type"]),
            "component_count": int(before_field.shape[1]),
            "changed_rows": int(np.count_nonzero(row_active)),
            "changed_fraction": float(np.count_nonzero(row_active)) / float(before_field.shape[0]) if before_field.shape[0] else 0.0,
            "mean_abs_delta": float(np.mean(np.abs(delta))) if delta.size else 0.0,
            "max_abs_delta": float(np.max(np.abs(delta))) if delta.size else 0.0,
            "delta_hash": _sha256_bytes(delta.tobytes()),
        }

    label_transition: dict[str, Any] = {
        "available": bool(before_state["labels"] is not None and after_state["labels"] is not None),
        "before_attribute": str(before_state["labels_summary"]["attribute"]),
        "after_attribute": str(after_state["labels_summary"]["attribute"]),
        "before_grain_count": int(before_state["labels_summary"]["grain_count"]),
        "after_grain_count": int(after_state["labels_summary"]["grain_count"]),
        "before_mean_grain_area": float(before_state["labels_summary"]["mean_grain_area"]),
        "after_mean_grain_area": float(after_state["labels_summary"]["mean_grain_area"]),
    }
    if before_state["labels"] is not None and after_state["labels"] is not None:
        before_labels = np.asarray(before_state["labels"], dtype=np.int32)
        after_labels = np.asarray(after_state["labels"], dtype=np.int32)
        label_transition["shape_matches"] = bool(before_labels.shape == after_labels.shape)
        if before_labels.shape == after_labels.shape:
            changed = before_labels != after_labels
            label_transition["changed_pixels"] = int(np.count_nonzero(changed))
            label_transition["changed_fraction"] = float(np.count_nonzero(changed)) / float(before_labels.size) if before_labels.size else 0.0
            label_transition["delta_hash"] = _sha256_bytes(changed.astype(np.uint8).tobytes())

    return {
        "checkpoint_name": str(checkpoint_name or f"{Path(before_elle_path).stem}_to_{Path(after_elle_path).stem}"),
        "before_path": str(Path(before_elle_path)),
        "after_path": str(Path(after_elle_path)),
        "field_names": selected_fields,
        "label_transition": label_transition,
        "field_transitions": field_transitions,
    }


def extract_legacy_reference_swept_unode_transition(
    before_elle_path: str | Path,
    after_elle_path: str | Path,
    *,
    checkpoint_name: str | None = None,
    label_attribute: str = "auto",
    field_names: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    before_state = _extract_legacy_dense_state(before_elle_path, label_attribute=label_attribute)
    after_state = _extract_legacy_dense_state(after_elle_path, label_attribute=label_attribute)

    swept_mask = _legacy_swept_unode_mask(before_state, after_state)
    swept_available = swept_mask is not None
    swept_rows = int(np.count_nonzero(swept_mask)) if swept_mask is not None else 0
    total_unodes = len(before_state.get("ordered_unode_ids", ()))

    candidate_fields = sorted(set(before_state["dense_fields"]) & set(after_state["dense_fields"]))
    if field_names is None:
        selected_fields = [
            str(name)
            for name in candidate_fields
            if str(before_state["field_metadata"][str(name)]["entity_type"]) == "unode"
        ]
    else:
        selected_fields = [
            str(name)
            for name in field_names
            if (
                str(name) in before_state["dense_fields"]
                and str(name) in after_state["dense_fields"]
                and str(before_state["field_metadata"][str(name)]["entity_type"]) == "unode"
            )
        ]

    field_transitions: dict[str, dict[str, Any]] = {}
    for name in selected_fields:
        before_field = np.asarray(before_state["dense_fields"][name], dtype=np.float64)
        after_field = np.asarray(after_state["dense_fields"][name], dtype=np.float64)
        if before_field.shape != after_field.shape:
            field_transitions[str(name)] = {
                "shape_matches": False,
                "before_shape": list(before_field.shape),
                "after_shape": list(after_field.shape),
            }
            continue

        if swept_mask is None:
            field_transitions[str(name)] = {
                "shape_matches": True,
                "component_count": int(before_field.shape[1]),
                "swept_rows": 0,
                "changed_swept_rows": 0,
                "changed_swept_fraction": 0.0,
                "mean_abs_swept_delta": 0.0,
                "max_abs_swept_delta": 0.0,
                "before_swept_hash": _sha256_bytes(b""),
                "after_swept_hash": _sha256_bytes(b""),
                "swept_delta_hash": _sha256_bytes(b""),
            }
            continue

        before_swept = np.asarray(before_field[swept_mask], dtype=np.float64)
        after_swept = np.asarray(after_field[swept_mask], dtype=np.float64)
        delta = np.asarray(after_swept - before_swept, dtype=np.float64)
        row_active = np.any(np.abs(delta) > 1.0e-12, axis=1) if delta.ndim == 2 else np.zeros((0,), dtype=bool)
        field_transitions[str(name)] = {
            "shape_matches": True,
            "component_count": int(before_field.shape[1]),
            "swept_rows": int(before_swept.shape[0]),
            "changed_swept_rows": int(np.count_nonzero(row_active)),
            "changed_swept_fraction": float(np.count_nonzero(row_active)) / float(before_swept.shape[0]) if before_swept.shape[0] else 0.0,
            "mean_abs_swept_delta": float(np.mean(np.abs(delta))) if delta.size else 0.0,
            "max_abs_swept_delta": float(np.max(np.abs(delta))) if delta.size else 0.0,
            "before_swept_hash": _sha256_bytes(before_swept.tobytes()),
            "after_swept_hash": _sha256_bytes(after_swept.tobytes()),
            "swept_delta_hash": _sha256_bytes(delta.tobytes()),
        }

    return {
        "checkpoint_name": str(checkpoint_name or f"{Path(before_elle_path).stem}_to_{Path(after_elle_path).stem}_swept"),
        "before_path": str(Path(before_elle_path)),
        "after_path": str(Path(after_elle_path)),
        "field_names": selected_fields,
        "swept_unodes": {
            "available": bool(swept_available),
            "swept_rows": int(swept_rows),
            "swept_fraction": float(swept_rows) / float(total_unodes) if total_unodes else 0.0,
            "swept_mask_hash": _sha256_bytes(np.asarray(swept_mask if swept_mask is not None else np.zeros((0,), dtype=np.uint8), dtype=np.uint8).tobytes()),
        },
        "field_transitions": field_transitions,
    }


def build_legacy_reference_bundle(
    checkpoints: dict[str, str | Path],
    *,
    source_name: str,
    label_attribute: str = "auto",
) -> dict[str, Any]:
    ordered_names = sorted(checkpoints)
    return {
        "source_name": str(source_name),
        "checkpoint_order": ordered_names,
        "checkpoints": {
            str(name): extract_legacy_reference_snapshot(
                checkpoints[name],
                checkpoint_name=str(name),
                label_attribute=label_attribute,
            )
            for name in ordered_names
        },
    }


def write_legacy_reference_bundle(path: str | Path, bundle: dict[str, Any]) -> Path:
    outpath = Path(path)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    outpath.write_text(json.dumps(bundle, indent=2, sort_keys=True), encoding="utf-8")
    return outpath


def load_legacy_reference_bundle(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _compare_scalar(reference: Any, candidate: Any) -> dict[str, Any]:
    matches = reference == candidate
    report = {
        "matches": bool(matches),
        "reference": reference,
        "candidate": candidate,
    }
    if isinstance(reference, (int, float)) and isinstance(candidate, (int, float)):
        report["delta"] = float(candidate) - float(reference)
    return report


def compare_legacy_reference_snapshot(
    elle_path: str | Path,
    reference_snapshot: dict[str, Any],
    *,
    label_attribute: str = "auto",
) -> dict[str, Any]:
    candidate_snapshot = extract_legacy_reference_snapshot(
        elle_path,
        checkpoint_name=str(reference_snapshot.get("checkpoint_name", Path(elle_path).stem)),
        label_attribute=label_attribute,
    )

    mesh_report = {
        key: _compare_scalar(reference_snapshot["mesh"][key], candidate_snapshot["mesh"][key])
        for key in sorted(reference_snapshot["mesh"])
    }
    label_report = {
        key: _compare_scalar(reference_snapshot["label_summary"][key], candidate_snapshot["label_summary"][key])
        for key in sorted(reference_snapshot["label_summary"])
    }

    reference_fields = reference_snapshot["field_summaries"]
    candidate_fields = candidate_snapshot["field_summaries"]
    shared_field_names = sorted(set(reference_fields) & set(candidate_fields))
    field_report = {
        name: {
            key: _compare_scalar(reference_fields[name][key], candidate_fields[name][key])
            for key in sorted(reference_fields[name])
        }
        for name in shared_field_names
    }

    mismatched_fields = sorted(
        name
        for name, values in field_report.items()
        if not all(result["matches"] for result in values.values())
    )

    return {
        "reference_checkpoint_name": str(reference_snapshot["checkpoint_name"]),
        "candidate_path": str(Path(elle_path)),
        "matches": (
            all(result["matches"] for result in mesh_report.values())
            and all(result["matches"] for result in label_report.values())
            and not mismatched_fields
            and sorted(set(reference_fields) - set(candidate_fields)) == []
            and sorted(set(candidate_fields) - set(reference_fields)) == []
        ),
        "mesh": mesh_report,
        "label_summary": label_report,
        "field_summaries": field_report,
        "missing_field_summaries": sorted(set(reference_fields) - set(candidate_fields)),
        "unexpected_field_summaries": sorted(set(candidate_fields) - set(reference_fields)),
        "mismatched_field_summaries": mismatched_fields,
    }


def compare_legacy_reference_bundle(
    bundle: dict[str, Any],
    checkpoints: dict[str, str | Path],
    *,
    label_attribute: str = "auto",
) -> dict[str, Any]:
    ordered_names = [str(name) for name in bundle.get("checkpoint_order", [])]
    checkpoint_reports: dict[str, dict[str, Any]] = {}
    missing_checkpoints: list[str] = []
    unexpected_checkpoints = sorted(set(checkpoints) - set(ordered_names))

    for name in ordered_names:
        if name not in checkpoints:
            missing_checkpoints.append(name)
            continue
        checkpoint_reports[name] = compare_legacy_reference_snapshot(
            checkpoints[name],
            bundle["checkpoints"][name],
            label_attribute=label_attribute,
        )

    return {
        "source_name": str(bundle.get("source_name", "")),
        "matches": (
            not missing_checkpoints
            and not unexpected_checkpoints
            and all(report["matches"] for report in checkpoint_reports.values())
        ),
        "checkpoint_order": ordered_names,
        "checkpoints": checkpoint_reports,
        "missing_checkpoints": missing_checkpoints,
        "unexpected_checkpoints": unexpected_checkpoints,
    }


def compare_legacy_reference_transition(
    before_elle_path: str | Path,
    after_elle_path: str | Path,
    reference_transition: dict[str, Any],
    *,
    label_attribute: str = "auto",
) -> dict[str, Any]:
    candidate_transition = extract_legacy_reference_transition(
        before_elle_path,
        after_elle_path,
        checkpoint_name=str(reference_transition.get("checkpoint_name", "")),
        label_attribute=label_attribute,
        field_names=tuple(reference_transition.get("field_names", ())),
    )

    label_report = {
        key: _compare_scalar(reference_transition["label_transition"].get(key), candidate_transition["label_transition"].get(key))
        for key in sorted(set(reference_transition.get("label_transition", {})) | set(candidate_transition.get("label_transition", {})))
    }

    reference_fields = reference_transition.get("field_transitions", {})
    candidate_fields = candidate_transition.get("field_transitions", {})
    shared_names = sorted(set(reference_fields) & set(candidate_fields))
    field_report = {
        name: {
            key: _compare_scalar(reference_fields[name].get(key), candidate_fields[name].get(key))
            for key in sorted(set(reference_fields[name]) | set(candidate_fields[name]))
        }
        for name in shared_names
    }
    mismatched_fields = sorted(
        name
        for name, values in field_report.items()
        if not all(result["matches"] for result in values.values())
    )

    return {
        "reference_checkpoint_name": str(reference_transition.get("checkpoint_name", "")),
        "candidate_before_path": str(Path(before_elle_path)),
        "candidate_after_path": str(Path(after_elle_path)),
        "matches": (
            all(result["matches"] for result in label_report.values())
            and not mismatched_fields
            and sorted(set(reference_fields) - set(candidate_fields)) == []
            and sorted(set(candidate_fields) - set(reference_fields)) == []
        ),
        "label_transition": label_report,
        "field_transitions": field_report,
        "missing_field_transitions": sorted(set(reference_fields) - set(candidate_fields)),
        "unexpected_field_transitions": sorted(set(candidate_fields) - set(reference_fields)),
        "mismatched_field_transitions": mismatched_fields,
    }


def compare_legacy_reference_swept_unode_transition(
    before_elle_path: str | Path,
    after_elle_path: str | Path,
    reference_transition: dict[str, Any],
    *,
    label_attribute: str = "auto",
) -> dict[str, Any]:
    candidate_transition = extract_legacy_reference_swept_unode_transition(
        before_elle_path,
        after_elle_path,
        checkpoint_name=str(reference_transition.get("checkpoint_name", "")),
        label_attribute=label_attribute,
        field_names=tuple(reference_transition.get("field_names", ())),
    )

    swept_report = {
        key: _compare_scalar(reference_transition["swept_unodes"].get(key), candidate_transition["swept_unodes"].get(key))
        for key in sorted(set(reference_transition.get("swept_unodes", {})) | set(candidate_transition.get("swept_unodes", {})))
    }

    reference_fields = reference_transition.get("field_transitions", {})
    candidate_fields = candidate_transition.get("field_transitions", {})
    shared_names = sorted(set(reference_fields) & set(candidate_fields))
    field_report = {
        name: {
            key: _compare_scalar(reference_fields[name].get(key), candidate_fields[name].get(key))
            for key in sorted(set(reference_fields[name]) | set(candidate_fields[name]))
        }
        for name in shared_names
    }
    mismatched_fields = sorted(
        name
        for name, values in field_report.items()
        if not all(result["matches"] for result in values.values())
    )

    return {
        "reference_checkpoint_name": str(reference_transition.get("checkpoint_name", "")),
        "candidate_before_path": str(Path(before_elle_path)),
        "candidate_after_path": str(Path(after_elle_path)),
        "matches": (
            all(result["matches"] for result in swept_report.values())
            and not mismatched_fields
            and sorted(set(reference_fields) - set(candidate_fields)) == []
            and sorted(set(candidate_fields) - set(reference_fields)) == []
        ),
        "swept_unodes": swept_report,
        "field_transitions": field_report,
        "missing_field_transitions": sorted(set(reference_fields) - set(candidate_fields)),
        "unexpected_field_transitions": sorted(set(candidate_fields) - set(reference_fields)),
        "mismatched_field_transitions": mismatched_fields,
    }
