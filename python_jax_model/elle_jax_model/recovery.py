from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_RECOVERY_SYMMETRY_PATH = (
    Path(__file__).resolve().parents[2]
    / "FS_Codes"
    / "FS_recrystallisation"
    / "FS_recovery"
    / "symmetry.symm"
)


@dataclass(frozen=True)
class RecoveryConfig:
    """First faithful slice of the original ELLE recovery stage."""

    high_angle_boundary_deg: float = 5.0
    trial_rotation_deg: float = 0.1
    rotation_mobility_length: float = 500.0
    symmetry_path: str | None = None


@lru_cache(maxsize=8)
def _load_symmetry_operators(path_str: str) -> np.ndarray:
    path = Path(path_str)
    try:
        lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except OSError:
        return np.eye(3, dtype=np.float64)[None, :, :]
    if not lines:
        return np.eye(3, dtype=np.float64)[None, :, :]
    try:
        count = int(lines[0])
    except ValueError:
        return np.eye(3, dtype=np.float64)[None, :, :]
    rows: list[list[float]] = []
    for line in lines[1: 1 + count * 3]:
        parts = line.split()
        if len(parts) != 3:
            continue
        try:
            rows.append([float(parts[0]), float(parts[1]), float(parts[2])])
        except ValueError:
            continue
    if len(rows) < count * 3:
        return np.eye(3, dtype=np.float64)[None, :, :]
    matrices = np.asarray(rows[: count * 3], dtype=np.float64).reshape(count, 3, 3)
    return np.asarray(matrices, dtype=np.float64)


def _resolve_recovery_symmetry_operators(config: RecoveryConfig) -> np.ndarray:
    path = Path(config.symmetry_path).expanduser() if config.symmetry_path else DEFAULT_RECOVERY_SYMMETRY_PATH
    return _load_symmetry_operators(str(path.resolve()))


def _approximate_misorientation_deg(a: np.ndarray, b: np.ndarray) -> float:
    ax = float(a[0])
    ay = float(a[1])
    az = float(a[2])
    bx = float(b[0])
    by = float(b[1])
    bz = float(b[2])
    dx = (ax - bx + 180.0) % 360.0 - 180.0
    dy = (ay - by + 180.0) % 360.0 - 180.0
    dz = (az - bz + 180.0) % 360.0 - 180.0
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def _orientation_matrices_from_euler_deg(euler_values_deg: np.ndarray) -> np.ndarray:
    euler_values = np.deg2rad(np.asarray(euler_values_deg, dtype=np.float64))
    phi = euler_values[..., 0]
    rho = euler_values[..., 1]
    gamma = euler_values[..., 2]
    matrices = np.empty(euler_values.shape[:-1] + (3, 3), dtype=np.float64)
    matrices[..., 0, 0] = np.cos(phi) * np.cos(gamma) - np.sin(phi) * np.sin(gamma) * np.cos(rho)
    matrices[..., 0, 1] = np.sin(phi) * np.cos(gamma) + np.cos(phi) * np.sin(gamma) * np.cos(rho)
    matrices[..., 0, 2] = np.sin(gamma) * np.sin(rho)
    matrices[..., 1, 0] = -np.cos(phi) * np.sin(gamma) - np.sin(phi) * np.cos(gamma) * np.cos(rho)
    matrices[..., 1, 1] = -np.sin(phi) * np.sin(gamma) + np.cos(phi) * np.cos(gamma) * np.cos(rho)
    matrices[..., 1, 2] = np.cos(gamma) * np.sin(rho)
    matrices[..., 2, 0] = np.sin(phi) * np.sin(rho)
    matrices[..., 2, 1] = -np.cos(phi) * np.sin(rho)
    matrices[..., 2, 2] = np.cos(rho)
    return np.asarray(matrices, dtype=np.float64)


def _legacy_recovery_trial_matrices(trial_index: int, theta_deg: float | np.ndarray) -> np.ndarray:
    theta = np.deg2rad(np.asarray(theta_deg, dtype=np.float64))
    if int(trial_index) in (1, 3, 5):
        theta = -theta
    cosine = np.cos(theta)
    sine = np.sin(theta)
    shape = np.asarray(theta, dtype=np.float64).shape
    matrices = np.zeros(shape + (3, 3), dtype=np.float64)
    if int(trial_index) in (0, 1):
        matrices[..., 0, 0] = cosine
        matrices[..., 0, 2] = sine
        matrices[..., 1, 1] = 1.0
        matrices[..., 2, 0] = -sine
        matrices[..., 2, 2] = cosine
        return np.asarray(matrices, dtype=np.float64)
    if int(trial_index) in (2, 3):
        matrices[..., 0, 0] = cosine
        matrices[..., 0, 1] = sine
        matrices[..., 1, 0] = -sine
        matrices[..., 1, 1] = cosine
        matrices[..., 2, 2] = 1.0
        return np.asarray(matrices, dtype=np.float64)
    if int(trial_index) in (4, 5):
        matrices[..., 0, 0] = 1.0
        matrices[..., 1, 1] = cosine
        matrices[..., 1, 2] = -sine
        matrices[..., 2, 1] = sine
        matrices[..., 2, 2] = cosine
        return np.asarray(matrices, dtype=np.float64)
    raise ValueError(f"unsupported recovery trial index {trial_index}")


def _euler_deg_from_orientation_matrices(matrices: np.ndarray, current_euler_deg: np.ndarray) -> np.ndarray:
    matrices = np.asarray(matrices, dtype=np.float64)
    current_euler_deg = np.asarray(current_euler_deg, dtype=np.float64)
    eps = 1.0e-10
    sign_phi = np.where(matrices[..., 2, 1] < 0.0, 1.0, -1.0)
    new_rho = np.degrees(np.arccos(np.clip(matrices[..., 2, 2], -1.0, 1.0))) * sign_phi
    new_rho = np.abs(new_rho)

    new_phi = np.zeros_like(new_rho)
    new_gamma = np.zeros_like(new_rho)
    active = np.abs(new_rho) > eps
    new_phi[active] = -np.degrees(np.arctan2(matrices[..., 2, 0][active], matrices[..., 2, 1][active]))
    new_gamma[active] = np.degrees(np.arctan2(matrices[..., 0, 2][active], matrices[..., 1, 2][active]))
    inactive = ~active
    new_phi[inactive] = 0.0
    new_gamma[inactive] = np.degrees(np.arcsin(np.clip(matrices[..., 1, 0][inactive], -1.0, 1.0)))

    updated = np.stack([new_phi, new_rho, new_gamma], axis=-1)
    diff_phi = np.abs(updated[..., 0] - current_euler_deg[..., 0]) > 5.0
    updated[..., 0] = np.where(diff_phi, updated[..., 0] + 180.0, updated[..., 0])
    updated[..., 0] = np.where(updated[..., 0] > 180.0, updated[..., 0] - 360.0, updated[..., 0])

    diff_rho = np.abs(updated[..., 1] - current_euler_deg[..., 1]) > 5.0
    updated[..., 1] = np.where(diff_rho, updated[..., 1] + 180.0, updated[..., 1])
    updated[..., 1] = np.where(updated[..., 1] > 180.0, updated[..., 1] - 360.0, updated[..., 1])
    updated[..., 1] = np.where(updated[..., 1] < 0.0, updated[..., 1] - 360.0, updated[..., 1])

    diff_gamma = np.abs(updated[..., 2] - current_euler_deg[..., 2]) > 15.0
    updated[..., 2] = np.where(diff_gamma, updated[..., 2] + 180.0, updated[..., 2])
    updated[..., 2] = np.where(updated[..., 2] > 180.0, updated[..., 2] - 360.0, updated[..., 2])
    return np.asarray(updated, dtype=np.float64)


def _apply_legacy_recovery_trial_rotation(
    euler_values_deg: np.ndarray,
    *,
    trial_index: int,
    theta_deg: float | np.ndarray,
) -> np.ndarray:
    base_matrices = _orientation_matrices_from_euler_deg(euler_values_deg)
    trial_matrices = _legacy_recovery_trial_matrices(int(trial_index), theta_deg)
    rotated_matrices = np.einsum("...ij,...jk->...ik", trial_matrices, base_matrices)
    return _euler_deg_from_orientation_matrices(rotated_matrices, np.asarray(euler_values_deg, dtype=np.float64))


def _symmetry_misorientation_deg(
    left_euler_deg: np.ndarray,
    right_euler_deg: np.ndarray,
    symmetry_operators: np.ndarray,
) -> np.ndarray:
    left = _orientation_matrices_from_euler_deg(left_euler_deg)
    right = _orientation_matrices_from_euler_deg(right_euler_deg)
    left_right = np.matmul(left, np.swapaxes(right, -1, -2))
    right_left = np.matmul(right, np.swapaxes(left, -1, -2))
    trace_left_right = np.einsum("sij,...ji->s...", symmetry_operators, left_right)
    trace_right_left = np.einsum("sij,...ji->s...", symmetry_operators, right_left)
    cosine_left_right = np.clip((trace_left_right - 1.0) / 2.0, -1.0, 1.0)
    cosine_right_left = np.clip((trace_right_left - 1.0) / 2.0, -1.0, 1.0)
    mis_left_right = np.degrees(np.arccos(cosine_left_right))
    mis_right_left = np.degrees(np.arccos(cosine_right_left))
    return np.asarray(np.min(np.minimum(mis_left_right, mis_right_left), axis=0), dtype=np.float64)


def _symmetry_misorientation_from_matrices(
    left_matrices: np.ndarray,
    right_matrices: np.ndarray,
    symmetry_operators: np.ndarray,
) -> np.ndarray:
    left = np.asarray(left_matrices, dtype=np.float64)
    right = np.asarray(right_matrices, dtype=np.float64)
    left_right = np.matmul(left, np.swapaxes(right, -1, -2))
    right_left = np.matmul(right, np.swapaxes(left, -1, -2))
    trace_left_right = np.einsum("sij,...ji->s...", symmetry_operators, left_right)
    trace_right_left = np.einsum("sij,...ji->s...", symmetry_operators, right_left)
    cosine_left_right = np.clip((trace_left_right - 1.0) / 2.0, -1.0, 1.0)
    cosine_right_left = np.clip((trace_right_left - 1.0) / 2.0, -1.0, 1.0)
    mis_left_right = np.degrees(np.arccos(cosine_left_right))
    mis_right_left = np.degrees(np.arccos(cosine_right_left))
    return np.asarray(np.min(np.minimum(mis_left_right, mis_right_left), axis=0), dtype=np.float64)


def _baseline_neighbour_misorientation_from_matrices(
    orientation_matrices: np.ndarray,
    neighbour_indices: np.ndarray,
    neighbour_mask: np.ndarray,
    hagb_deg: float,
    *,
    symmetry_operators: np.ndarray,
) -> np.ndarray:
    if neighbour_indices.size == 0:
        return np.zeros((int(np.asarray(orientation_matrices).shape[0]),), dtype=np.float64)
    base = np.asarray(orientation_matrices, dtype=np.float64)
    neighbour_orientations = base[neighbour_indices]
    misorientation = _symmetry_misorientation_from_matrices(
        base[:, None, :, :],
        neighbour_orientations,
        np.asarray(symmetry_operators, dtype=np.float64),
    )
    capped = np.minimum(misorientation, float(hagb_deg))
    weighted = capped * neighbour_mask.astype(np.float64)
    counts = np.sum(neighbour_mask, axis=1)
    result = np.zeros((base.shape[0],), dtype=np.float64)
    active = counts > 0
    result[active] = np.sum(weighted[active], axis=1) / counts[active]
    return result


def _trial_neighbour_misorientation_from_matrices(
    trial_orientation_matrices: np.ndarray,
    neighbour_orientation_matrices: np.ndarray,
    neighbour_indices: np.ndarray,
    neighbour_mask: np.ndarray,
    hagb_deg: float,
    *,
    symmetry_operators: np.ndarray,
) -> np.ndarray:
    if neighbour_indices.size == 0:
        return np.zeros((int(np.asarray(trial_orientation_matrices).shape[0]),), dtype=np.float64)
    base = np.asarray(trial_orientation_matrices, dtype=np.float64)
    neighbour_orientations = np.asarray(neighbour_orientation_matrices, dtype=np.float64)[neighbour_indices]
    misorientation = _symmetry_misorientation_from_matrices(
        base[:, None, :, :],
        neighbour_orientations,
        np.asarray(symmetry_operators, dtype=np.float64),
    )
    capped = np.minimum(misorientation, float(hagb_deg))
    weighted = capped * neighbour_mask.astype(np.float64)
    counts = np.sum(neighbour_mask, axis=1)
    result = np.zeros((base.shape[0],), dtype=np.float64)
    active = counts > 0
    result[active] = np.sum(weighted[active], axis=1) / counts[active]
    return result


def _read_shockley_recovery_energy(avg_misorientation_deg: float, hagb_deg: float) -> float:
    avg = float(max(avg_misorientation_deg, 0.0))
    hagb = float(max(hagb_deg, 1.0e-9))
    if avg <= 0.0:
        return 0.0
    if avg > hagb:
        return 1.0
    ratio = max(avg / hagb, 1.0e-12)
    return float(ratio / (1.0 - np.log(ratio)))


def _read_shockley_recovery_energy_array(
    avg_misorientation_deg: np.ndarray,
    hagb_deg: float,
) -> np.ndarray:
    avg = np.maximum(np.asarray(avg_misorientation_deg, dtype=np.float64), 0.0)
    hagb = float(max(hagb_deg, 1.0e-9))
    ratio = np.maximum(avg / hagb, 1.0e-12)
    energy = ratio / (1.0 - np.log(ratio))
    energy = np.where(avg <= 0.0, 0.0, energy)
    energy = np.where(avg > hagb, 1.0, energy)
    return np.asarray(energy, dtype=np.float64)


def _recovery_rotation_cap_deg(
    avg_local_misorientation_deg: np.ndarray | float,
    theta_deg: float,
) -> np.ndarray:
    avg = np.maximum(np.asarray(avg_local_misorientation_deg, dtype=np.float64), 0.0)
    return np.asarray(
        np.minimum(avg, 20.0 * float(theta_deg)),
        dtype=np.float64,
    )


def _grid_neighbour_indices(
    grid_indices: np.ndarray,
    label_values: np.ndarray,
    point_index: int,
    grid_shape: tuple[int, int],
    index_lookup: dict[tuple[int, int], int] | None = None,
) -> np.ndarray:
    if index_lookup is None:
        index_lookup = {
            (int(index[0]), int(index[1])): int(i)
            for i, index in enumerate(np.asarray(grid_indices, dtype=np.int32))
        }
    ix, iy = (int(value) for value in np.asarray(grid_indices[int(point_index)], dtype=np.int32))
    current_label = int(label_values[int(point_index)])
    neighbours: list[tuple[int, int, int, int]] = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            nx = (ix + dx) % int(grid_shape[0])
            ny = (iy + dy) % int(grid_shape[1])
            neighbour_index = index_lookup.get((nx, ny))
            if neighbour_index is None:
                continue
            if int(label_values[int(neighbour_index)]) != current_label:
                continue
            neighbours.append((dx * dx + dy * dy, dx, dy, int(neighbour_index)))
    if not neighbours:
        return np.zeros((0,), dtype=np.int32)
    neighbours.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
    nearest = [entry[3] for entry in neighbours[:6]]
    return np.asarray(nearest, dtype=np.int32)


def _build_same_label_neighbour_lists(
    grid_indices: np.ndarray,
    label_values: np.ndarray,
    grid_shape: tuple[int, int],
) -> list[np.ndarray]:
    index_lookup = {
        (int(index[0]), int(index[1])): int(i)
        for i, index in enumerate(np.asarray(grid_indices, dtype=np.int32))
    }
    return [
        _grid_neighbour_indices(
            grid_indices,
            label_values,
            int(point_index),
            grid_shape,
            index_lookup=index_lookup,
        )
        for point_index in range(int(np.asarray(grid_indices).shape[0]))
    ]


def _neighbour_index_matrix(neighbour_lists: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    if not neighbour_lists:
        return np.zeros((0, 0), dtype=np.int32), np.zeros((0, 0), dtype=bool)
    max_degree = max((int(neighbours.size) for neighbours in neighbour_lists), default=0)
    if max_degree <= 0:
        count = len(neighbour_lists)
        return np.zeros((count, 0), dtype=np.int32), np.zeros((count, 0), dtype=bool)
    indices = np.zeros((len(neighbour_lists), max_degree), dtype=np.int32)
    mask = np.zeros((len(neighbour_lists), max_degree), dtype=bool)
    for row, neighbours in enumerate(neighbour_lists):
        if neighbours.size == 0:
            continue
        degree = int(neighbours.size)
        indices[row, :degree] = np.asarray(neighbours, dtype=np.int32)
        mask[row, :degree] = True
    return indices, mask


def _baseline_neighbour_misorientation_deg(
    orientation_values: np.ndarray,
    neighbour_indices: np.ndarray,
    neighbour_mask: np.ndarray,
    hagb_deg: float,
    *,
    symmetry_operators: np.ndarray | None = None,
) -> np.ndarray:
    if neighbour_indices.size == 0:
        return np.zeros((int(orientation_values.shape[0]),), dtype=np.float64)
    base = np.asarray(orientation_values[:, :3], dtype=np.float64)
    neighbour_orientations = base[neighbour_indices]
    if symmetry_operators is None:
        delta = (base[:, None, :] - neighbour_orientations + 180.0) % 360.0 - 180.0
        misorientation = np.sqrt(np.sum(delta * delta, axis=2))
    else:
        misorientation = _symmetry_misorientation_deg(
            base[:, None, :],
            neighbour_orientations,
            np.asarray(symmetry_operators, dtype=np.float64),
        )
    capped = np.minimum(misorientation, float(hagb_deg))
    weighted = capped * neighbour_mask.astype(np.float64)
    counts = np.sum(neighbour_mask, axis=1)
    result = np.zeros((base.shape[0],), dtype=np.float64)
    active = counts > 0
    result[active] = np.sum(weighted[active], axis=1) / counts[active]
    return result


def _trial_neighbour_misorientation_deg(
    trial_orientation_values: np.ndarray,
    neighbour_orientation_values: np.ndarray,
    neighbour_indices: np.ndarray,
    neighbour_mask: np.ndarray,
    hagb_deg: float,
    *,
    symmetry_operators: np.ndarray | None = None,
) -> np.ndarray:
    if neighbour_indices.size == 0:
        return np.zeros((int(trial_orientation_values.shape[0]),), dtype=np.float64)
    base = np.asarray(trial_orientation_values[:, :3], dtype=np.float64)
    neighbour_orientations = np.asarray(neighbour_orientation_values[:, :3], dtype=np.float64)[neighbour_indices]
    if symmetry_operators is None:
        delta = (base[:, None, :] - neighbour_orientations + 180.0) % 360.0 - 180.0
        misorientation = np.sqrt(np.sum(delta * delta, axis=2))
    else:
        misorientation = _symmetry_misorientation_deg(
            base[:, None, :],
            neighbour_orientations,
            np.asarray(symmetry_operators, dtype=np.float64),
        )
    capped = np.minimum(misorientation, float(hagb_deg))
    weighted = capped * neighbour_mask.astype(np.float64)
    counts = np.sum(neighbour_mask, axis=1)
    result = np.zeros((base.shape[0],), dtype=np.float64)
    active = counts > 0
    result[active] = np.sum(weighted[active], axis=1) / counts[active]
    return result


def _ensure_scalar_field(
    seed_fields: dict[str, Any],
    field_name: str,
    count: int,
    default_value: float,
) -> np.ndarray:
    updated_values = dict(seed_fields.get("values", {}))
    field_order = list(seed_fields.get("field_order", ()))
    if field_name not in updated_values:
        updated_values[field_name] = np.full((int(count),), float(default_value), dtype=np.float64)
        if field_name not in field_order:
            field_order.append(field_name)
        seed_fields["values"] = updated_values
        seed_fields["field_order"] = tuple(field_order)
    return np.asarray(updated_values[field_name], dtype=np.float64)


def _update_recovery_flynn_sections(
    mesh_state: dict[str, Any],
    sample_labels: np.ndarray,
    orientation_values: dict[str, np.ndarray],
    density_values: np.ndarray | None,
) -> None:
    flynn_sections = mesh_state.get("_runtime_seed_flynn_sections")
    flynns = mesh_state.get("flynns")
    if not isinstance(flynn_sections, dict) or not isinstance(flynns, list):
        return

    id_order = [int(value) for value in flynn_sections.get("id_order", ())]
    if not id_order:
        return

    label_by_source_flynn: dict[int, int] = {}
    for flynn in flynns:
        if not isinstance(flynn, dict):
            continue
        source_flynn_id = int(flynn.get("source_flynn_id", flynn.get("flynn_id", -1)))
        compact_label = flynn.get("label")
        if compact_label is None:
            continue
        label_by_source_flynn[source_flynn_id] = int(compact_label)

    updated_values = dict(flynn_sections.get("values", {}))
    if "EULER_3" in updated_values and "U_EULER_3" in orientation_values:
        base_values = list(updated_values.get("EULER_3", ()))
        current_orientation = np.asarray(orientation_values["U_EULER_3"], dtype=np.float64)
        for index, source_flynn_id in enumerate(id_order):
            compact_label = label_by_source_flynn.get(int(source_flynn_id))
            if compact_label is None:
                continue
            mask = np.asarray(sample_labels, dtype=np.int32) == int(compact_label)
            if not np.any(mask):
                continue
            base_values[index] = tuple(
                float(value) for value in np.mean(current_orientation[mask], axis=0)
            )
        updated_values["EULER_3"] = tuple(base_values)

    if density_values is not None and "DISLOCDEN" in updated_values:
        base_values = list(updated_values.get("DISLOCDEN", ()))
        current_density = np.asarray(density_values, dtype=np.float64)
        for index, source_flynn_id in enumerate(id_order):
            compact_label = label_by_source_flynn.get(int(source_flynn_id))
            if compact_label is None:
                continue
            mask = np.asarray(sample_labels, dtype=np.int32) == int(compact_label)
            if not np.any(mask):
                continue
            base_values[index] = float(np.mean(current_density[mask]))
        updated_values["DISLOCDEN"] = tuple(base_values)

    flynn_sections["values"] = updated_values
    mesh_state["_runtime_seed_flynn_sections"] = flynn_sections


def _recovery_neighbour_context(
    mesh_state: dict[str, Any],
    grid_indices: np.ndarray,
    label_values: np.ndarray,
    grid_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cache = mesh_state.get("_runtime_recovery_cache")
    if not isinstance(cache, dict):
        cache = {}
    cached_labels = cache.get("sample_labels")
    cached_shape = cache.get("grid_shape")
    cached_indices = cache.get("grid_indices")
    if (
        isinstance(cached_labels, np.ndarray)
        and isinstance(cached_indices, np.ndarray)
        and tuple(cached_shape) == tuple(int(value) for value in grid_shape)
        and cached_indices.shape == grid_indices.shape
        and np.array_equal(cached_indices, grid_indices)
        and np.array_equal(cached_labels, label_values)
    ):
        neighbour_matrix = np.asarray(cache.get("neighbour_matrix"), dtype=np.int32)
        neighbour_mask = np.asarray(cache.get("neighbour_mask"), dtype=bool)
        counts = np.asarray(cache.get("counts"), dtype=np.int32)
        if neighbour_matrix.ndim == 2 and neighbour_mask.shape == neighbour_matrix.shape:
            mesh_state["_runtime_recovery_cache"] = cache
            return neighbour_matrix, neighbour_mask, counts

    same_label_neighbours = _build_same_label_neighbour_lists(
        grid_indices,
        label_values,
        grid_shape,
    )
    neighbour_matrix, neighbour_mask = _neighbour_index_matrix(same_label_neighbours)
    counts = (
        np.sum(neighbour_mask, axis=1, dtype=np.int32)
        if neighbour_mask.size
        else np.zeros((int(grid_indices.shape[0]),), dtype=np.int32)
    )
    cache = {
        "sample_labels": np.asarray(label_values, dtype=np.int32).copy(),
        "grid_shape": tuple(int(value) for value in grid_shape),
        "grid_indices": np.asarray(grid_indices, dtype=np.int32).copy(),
        "neighbour_matrix": neighbour_matrix,
        "neighbour_mask": neighbour_mask,
        "counts": counts,
    }
    mesh_state["_runtime_recovery_cache"] = cache
    return neighbour_matrix, neighbour_mask, counts


def apply_recovery_stage(
    mesh_state: dict[str, Any],
    current_labels: np.ndarray,
    config: RecoveryConfig,
    *,
    recovery_stage_index: int = 0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply a first faithful recovery pass to seed-unode fields."""

    seed_unodes = mesh_state.get("_runtime_seed_unodes")
    seed_sections = mesh_state.get("_runtime_seed_unode_sections")
    seed_fields = mesh_state.get("_runtime_seed_unode_fields")
    if not isinstance(seed_unodes, dict) or not isinstance(seed_sections, dict):
        return mesh_state, {
            "recovery_stage_index": int(recovery_stage_index),
            "recovery_applied": 0,
            "rotated_unodes": 0,
            "density_reduced_unodes": 0,
            "missing_orientation_sections": 1,
        }

    grid_indices = np.asarray(seed_unodes.get("grid_indices", ()), dtype=np.int32)
    if grid_indices.ndim != 2 or grid_indices.shape[0] == 0:
        return mesh_state, {
            "recovery_stage_index": int(recovery_stage_index),
            "recovery_applied": 0,
            "rotated_unodes": 0,
            "density_reduced_unodes": 0,
            "missing_orientation_sections": 1,
        }

    labels_np = np.asarray(current_labels, dtype=np.int32)
    sample_labels = labels_np[grid_indices[:, 0], grid_indices[:, 1]]
    grid_shape = tuple(int(value) for value in seed_unodes.get("grid_shape", labels_np.shape))
    symmetry_operators = _resolve_recovery_symmetry_operators(config)
    neighbour_matrix, neighbour_mask, counts = _recovery_neighbour_context(
        mesh_state,
        grid_indices,
        sample_labels,
        grid_shape,
    )

    component_counts = dict(seed_sections.get("component_counts", {}))
    updated_section_values = dict(seed_sections.get("values", {}))
    updated_field_values = {} if not isinstance(seed_fields, dict) else dict(seed_fields.get("values", {}))
    rotated_unodes = 0
    density_reduced_unodes = 0
    avg_misorientation_updates = 0

    if not isinstance(seed_fields, dict):
        seed_fields = {"field_order": (), "values": {}}

    previous_attr_f = _ensure_scalar_field(seed_fields, "U_ATTRIB_F", grid_indices.shape[0], 0.0)
    current_attr_f = previous_attr_f.copy()
    if "U_DISLOCDEN" in updated_field_values:
        current_density = np.asarray(updated_field_values["U_DISLOCDEN"], dtype=np.float64).copy()
    else:
        current_density = None

    for field_name in seed_sections.get("field_order", ()):
        component_count = int(component_counts.get(str(field_name), 1))
        if component_count < 3 or not str(field_name).startswith("U_EULER"):
            continue
        dense_values = updated_section_values.get(str(field_name))
        if dense_values is None:
            continue
        current_values = np.asarray(dense_values, dtype=np.float64).copy()
        if current_values.ndim != 2 or current_values.shape[1] < 3:
            continue
        current_euler = np.asarray(current_values[:, :3], dtype=np.float64).copy()
        current_matrices = _orientation_matrices_from_euler_deg(current_euler)
        baseline_all = _baseline_neighbour_misorientation_from_matrices(
            current_matrices,
            neighbour_matrix,
            neighbour_mask,
            config.high_angle_boundary_deg,
            symmetry_operators=symmetry_operators,
        )
        zero_misori_mask = (counts > 0) & (baseline_all <= 1.0e-12)
        if np.any(zero_misori_mask):
            current_attr_f[np.flatnonzero(zero_misori_mask)] = 0.0
            avg_misorientation_updates += int(np.count_nonzero(zero_misori_mask))

        active_mask = baseline_all > 1.0e-12
        min_energy_all = _read_shockley_recovery_energy_array(
            baseline_all,
            config.high_angle_boundary_deg,
        )
        rotated_mask = np.zeros((current_values.shape[0],), dtype=bool)

        for trial_index in range(6):
            trial_matrix = _legacy_recovery_trial_matrices(int(trial_index), float(config.trial_rotation_deg))
            trial_matrices = np.einsum("ij,njk->nik", trial_matrix, current_matrices)
            trial_misori_all = _trial_neighbour_misorientation_from_matrices(
                trial_matrices,
                current_matrices,
                neighbour_matrix,
                neighbour_mask,
                config.high_angle_boundary_deg,
                symmetry_operators=symmetry_operators,
            )
            trial_energy_all = _read_shockley_recovery_energy_array(
                trial_misori_all,
                config.high_angle_boundary_deg,
            )
            rotation_all = float(config.rotation_mobility_length) * (
                (min_energy_all - trial_energy_all) * float(config.trial_rotation_deg)
            )
            rotation_all = np.minimum(
                rotation_all,
                _recovery_rotation_cap_deg(
                    baseline_all,
                    float(config.trial_rotation_deg),
                ),
            )
            accept_mask = (
                active_mask
                & (trial_energy_all < min_energy_all)
                & (trial_misori_all < float(config.high_angle_boundary_deg))
                & (rotation_all > 0.0)
            )
            if not np.any(accept_mask):
                continue
            accepted_matrices = np.einsum(
                "nij,njk->nik",
                _legacy_recovery_trial_matrices(int(trial_index), rotation_all[accept_mask]),
                current_matrices[accept_mask],
            )
            current_matrices[accept_mask] = accepted_matrices
            current_euler[accept_mask] = _euler_deg_from_orientation_matrices(
                accepted_matrices,
                current_euler[accept_mask],
            )
            rotated_mask |= accept_mask
            baseline_all = _baseline_neighbour_misorientation_from_matrices(
                current_matrices,
                neighbour_matrix,
                neighbour_mask,
                config.high_angle_boundary_deg,
                symmetry_operators=symmetry_operators,
            )
            min_energy_all = _read_shockley_recovery_energy_array(
                baseline_all,
                config.high_angle_boundary_deg,
            )
            active_mask = baseline_all > 1.0e-12

        current_values[:, :3] = current_euler

        current_attr_f[counts > 0] = np.minimum(
            baseline_all[counts > 0],
            float(config.high_angle_boundary_deg),
        )
        avg_misorientation_updates += int(np.count_nonzero(counts > 0))
        rotated_unodes += int(np.count_nonzero(rotated_mask))
        if current_density is not None and int(recovery_stage_index) > 0:
            dummy = np.minimum(previous_attr_f, float(config.high_angle_boundary_deg))
            dummy = np.where(dummy <= 1.0e-1, 1.0e-1, dummy)
            updated_density = np.abs(current_density * current_attr_f / dummy)
            density_reduced_unodes += int(np.count_nonzero(updated_density < current_density - 1.0e-12))
            current_density = updated_density

        updated_section_values[str(field_name)] = np.asarray(current_values, dtype=np.float64)

    seed_sections["values"] = updated_section_values
    updated_field_values["U_ATTRIB_F"] = np.asarray(current_attr_f, dtype=np.float64)
    if current_density is not None:
        updated_field_values["U_DISLOCDEN"] = np.asarray(current_density, dtype=np.float64)
    seed_fields["values"] = updated_field_values
    mesh_state["_runtime_seed_unode_sections"] = seed_sections
    mesh_state["_runtime_seed_unode_fields"] = seed_fields
    _update_recovery_flynn_sections(
        mesh_state,
        sample_labels,
        {
            str(field_name): np.asarray(values, dtype=np.float64)
            for field_name, values in updated_section_values.items()
            if str(field_name).startswith("U_EULER")
        },
        current_density,
    )
    mesh_state.setdefault("stats", {})
    mesh_state["stats"]["recovery_rotated_unodes"] = int(rotated_unodes)
    mesh_state["stats"]["recovery_density_reduced_unodes"] = int(density_reduced_unodes)
    mesh_state["stats"]["recovery_stage_index"] = int(recovery_stage_index)

    return mesh_state, {
        "recovery_stage_index": int(recovery_stage_index),
        "recovery_applied": 1,
        "rotated_unodes": int(rotated_unodes),
        "density_reduced_unodes": int(density_reduced_unodes),
        "updated_avg_misorientation_unodes": int(avg_misorientation_updates),
    }
