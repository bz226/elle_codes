from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np


@dataclass(frozen=True)
class FFTSnapshotPaths:
    """Resolved file paths for one legacy ELLE <-> FFT mechanics snapshot."""

    temp_matrix_path: str
    unode_strain_path: str
    unode_euler_path: str
    tex_path: str | None = None


@dataclass(frozen=True)
class FFTMechanicsSnapshot:
    """Frozen mechanics-side fields emitted by the legacy ELLE <-> FFT bridge."""

    paths: FFTSnapshotPaths
    temp_matrix: np.ndarray
    unode_ids: np.ndarray
    unode_strain_xyz: np.ndarray
    euler_ids: np.ndarray
    unode_euler_deg: np.ndarray
    tex_columns: np.ndarray | None = None

    @property
    def normalized_strain_rate(self) -> np.ndarray | None:
        if self.tex_columns is None:
            return None
        return np.asarray(self.tex_columns[:, 4], dtype=np.float64)

    @property
    def normalized_stress(self) -> np.ndarray | None:
        if self.tex_columns is None:
            return None
        return np.asarray(self.tex_columns[:, 5], dtype=np.float64)

    @property
    def basal_activity(self) -> np.ndarray | None:
        if self.tex_columns is None:
            return None
        return np.asarray(self.tex_columns[:, 6], dtype=np.float64)

    @property
    def prismatic_activity(self) -> np.ndarray | None:
        if self.tex_columns is None:
            return None
        return np.asarray(self.tex_columns[:, 7], dtype=np.float64)

    @property
    def geometrical_dislocation_density(self) -> np.ndarray | None:
        if self.tex_columns is None:
            return None
        return np.asarray(self.tex_columns[:, 8], dtype=np.float64)

    @property
    def statistical_dislocation_density(self) -> np.ndarray | None:
        if self.tex_columns is None:
            return None
        return np.asarray(self.tex_columns[:, 9], dtype=np.float64)

    @property
    def fourier_point_ids(self) -> np.ndarray | None:
        if self.tex_columns is None:
            return None
        return np.asarray(np.rint(self.tex_columns[:, 10]), dtype=np.int32)

    @property
    def fft_grain_numbers(self) -> np.ndarray | None:
        if self.tex_columns is None:
            return None
        return np.asarray(np.rint(self.tex_columns[:, 11]), dtype=np.int32)


def _looks_like_legacy_fft_snapshot_dir(path: str | Path) -> bool:
    candidate = Path(path)
    if not candidate.is_dir():
        return False
    has_temp_matrix = (candidate / "temp-FFT.out").exists() or (candidate / "temp.out").exists()
    return (
        has_temp_matrix
        and (candidate / "unodexyz.out").exists()
        and (candidate / "unodeang.out").exists()
    )


def _legacy_fft_snapshot_sort_key(path: Path) -> tuple[object, ...]:
    parts = re.split(r"(\d+)", str(path.name))
    key: list[object] = []
    for part in parts:
        if not part:
            continue
        if part.isdigit():
            key.append(int(part))
        else:
            key.append(part.lower())
    return tuple(key)


def _align_snapshot_rows(
    seed_unode_ids: np.ndarray,
    snapshot_unode_ids: np.ndarray,
) -> tuple[np.ndarray, str]:
    seed_ids = np.asarray(seed_unode_ids, dtype=np.int32)
    snapshot_ids = np.asarray(snapshot_unode_ids, dtype=np.int32)
    if seed_ids.shape != snapshot_ids.shape:
        raise ValueError(
            f"snapshot id count {snapshot_ids.size} does not match seed unode count {seed_ids.size}"
        )
    if np.array_equal(seed_ids, snapshot_ids):
        return np.arange(seed_ids.size, dtype=np.int32), "exact_ids"
    if np.array_equal(seed_ids, snapshot_ids + 1):
        return np.arange(seed_ids.size, dtype=np.int32), "snapshot_zero_based"
    if np.array_equal(seed_ids + 1, snapshot_ids):
        return np.arange(seed_ids.size, dtype=np.int32), "snapshot_one_based"
    snapshot_lookup = {int(value): int(index) for index, value in enumerate(snapshot_ids)}
    if all(int(value) in snapshot_lookup for value in seed_ids):
        return np.asarray([snapshot_lookup[int(value)] for value in seed_ids], dtype=np.int32), "mapped_ids"
    return np.arange(seed_ids.size, dtype=np.int32), "positional"


def _load_numeric_rows(path: str | Path, *, min_columns: int) -> np.ndarray:
    rows: list[list[float]] = []
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < int(min_columns):
            raise ValueError(
                f"{path} has a row with {len(parts)} columns; expected at least {min_columns}"
            )
        rows.append([float(value) for value in parts[: int(min_columns)]])
    if not rows:
        raise ValueError(f"{path} does not contain any numeric rows")
    return np.asarray(rows, dtype=np.float64)


def _load_indexed_triplets(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    rows = _load_numeric_rows(path, min_columns=4)
    ids = np.asarray(np.rint(rows[:, 0]), dtype=np.int32)
    values = np.asarray(rows[:, 1:4], dtype=np.float64)
    return ids, values


def resolve_legacy_fft_snapshot_paths(source: str | Path) -> FFTSnapshotPaths:
    source_path = Path(source)
    if source_path.is_file():
        raise ValueError(
            "resolve_legacy_fft_snapshot_paths expects a directory containing legacy bridge files"
        )
    temp_matrix = source_path / "temp-FFT.out"
    if not temp_matrix.exists():
        temp_matrix = source_path / "temp.out"
    unode_strain = source_path / "unodexyz.out"
    unode_euler = source_path / "unodeang.out"
    tex_path = source_path / "tex.out"

    if not temp_matrix.exists():
        raise FileNotFoundError(f"missing legacy FFT temp matrix file in {source_path}")
    if not unode_strain.exists():
        raise FileNotFoundError(f"missing legacy FFT unode strain file in {source_path}")
    if not unode_euler.exists():
        raise FileNotFoundError(f"missing legacy FFT Euler file in {source_path}")

    return FFTSnapshotPaths(
        temp_matrix_path=str(temp_matrix),
        unode_strain_path=str(unode_strain),
        unode_euler_path=str(unode_euler),
        tex_path=str(tex_path) if tex_path.exists() else None,
    )


def resolve_legacy_fft_snapshot_sequence_paths(source: str | Path) -> tuple[Path, ...]:
    source_path = Path(source)
    if source_path.is_file():
        raise ValueError(
            "resolve_legacy_fft_snapshot_sequence_paths expects a directory or a directory containing snapshot subdirectories"
        )
    if not source_path.exists():
        raise FileNotFoundError(f"missing legacy FFT snapshot source {source_path}")
    if _looks_like_legacy_fft_snapshot_dir(source_path):
        return (source_path,)

    snapshot_dirs = tuple(
        sorted(
            (
                child
                for child in source_path.iterdir()
                if _looks_like_legacy_fft_snapshot_dir(child)
            ),
            key=_legacy_fft_snapshot_sort_key,
        )
    )
    if snapshot_dirs:
        return snapshot_dirs
    raise FileNotFoundError(
        f"could not find legacy FFT snapshot files or snapshot subdirectories in {source_path}"
    )


def load_legacy_fft_snapshot(source: str | Path) -> FFTMechanicsSnapshot:
    """Load one frozen legacy mechanics snapshot from ELLE/FFT bridge artifacts."""

    paths = resolve_legacy_fft_snapshot_paths(source)
    temp_rows = _load_numeric_rows(paths.temp_matrix_path, min_columns=3)
    if temp_rows.shape[0] < 3:
        raise ValueError(
            f"{paths.temp_matrix_path} has only {temp_rows.shape[0]} rows; expected at least 3"
        )
    temp_matrix = np.asarray(temp_rows[:3, :3], dtype=np.float64)
    unode_ids, unode_strain_xyz = _load_indexed_triplets(paths.unode_strain_path)
    euler_ids, unode_euler_deg = _load_indexed_triplets(paths.unode_euler_path)
    tex_columns = None
    if paths.tex_path is not None:
        tex_columns = _load_numeric_rows(paths.tex_path, min_columns=12)

    return FFTMechanicsSnapshot(
        paths=paths,
        temp_matrix=temp_matrix,
        unode_ids=unode_ids,
        unode_strain_xyz=unode_strain_xyz,
        euler_ids=euler_ids,
        unode_euler_deg=unode_euler_deg,
        tex_columns=tex_columns,
    )


def load_legacy_fft_snapshot_sequence(source: str | Path) -> tuple[FFTMechanicsSnapshot, ...]:
    """Load one or more frozen mechanics snapshots from a directory tree."""

    snapshot_dirs = resolve_legacy_fft_snapshot_sequence_paths(source)
    return tuple(load_legacy_fft_snapshot(snapshot_dir) for snapshot_dir in snapshot_dirs)


def apply_legacy_fft_snapshot_to_mesh_state(
    mesh_state: dict[str, object],
    snapshot: FFTMechanicsSnapshot,
) -> tuple[dict[str, object], dict[str, object]]:
    """Apply one frozen legacy mechanics snapshot to the faithful runtime state."""

    seed_unodes = mesh_state.get("_runtime_seed_unodes")
    if not isinstance(seed_unodes, dict):
        raise ValueError("mesh_state has no faithful seed-unode payload")
    seed_ids = np.asarray(seed_unodes.get("ids", ()), dtype=np.int32)
    if seed_ids.ndim != 1 or seed_ids.size == 0:
        raise ValueError("mesh_state has no faithful seed unode ids")

    row_order, alignment_mode = _align_snapshot_rows(seed_ids, snapshot.unode_ids)
    euler_row_order, euler_alignment_mode = _align_snapshot_rows(seed_ids, snapshot.euler_ids)
    aligned_euler = np.asarray(snapshot.unode_euler_deg[euler_row_order], dtype=np.float64)

    seed_sections = dict(mesh_state.get("_runtime_seed_unode_sections", {}))
    section_values = dict(seed_sections.get("values", {}))
    section_defaults = dict(seed_sections.get("defaults", {}))
    section_components = dict(seed_sections.get("component_counts", {}))
    section_field_order = list(seed_sections.get("field_order", ()))
    section_values["U_EULER_3"] = tuple(tuple(float(value) for value in row) for row in aligned_euler)
    section_defaults.setdefault("U_EULER_3", (0.0, 0.0, 0.0))
    section_components["U_EULER_3"] = 3
    if "U_EULER_3" not in section_field_order:
        section_field_order.append("U_EULER_3")
    mesh_state["_runtime_seed_unode_sections"] = {
        **seed_sections,
        "values": section_values,
        "defaults": section_defaults,
        "component_counts": section_components,
        "field_order": tuple(section_field_order),
    }

    seed_fields = dict(mesh_state.get("_runtime_seed_unode_fields", {}))
    field_values = dict(seed_fields.get("values", {}))
    field_order = list(seed_fields.get("field_order", ()))

    def _assign_scalar_field(name: str, values: np.ndarray) -> None:
        field_values[str(name)] = tuple(float(value) for value in np.asarray(values, dtype=np.float64))
        if str(name) not in field_order:
            field_order.append(str(name))

    if snapshot.tex_columns is not None:
        tex_rows = np.asarray(snapshot.tex_columns, dtype=np.float64)[row_order]
        _assign_scalar_field("U_ATTRIB_A", tex_rows[:, 4])
        _assign_scalar_field("U_ATTRIB_B", tex_rows[:, 5])
        _assign_scalar_field("U_ATTRIB_D", tex_rows[:, 6])
        _assign_scalar_field("U_ATTRIB_E", tex_rows[:, 7])
        previous_density = np.asarray(field_values.get("U_DISLOCDEN", np.zeros((seed_ids.size,), dtype=np.float64)), dtype=np.float64)
        updated_density = previous_density + tex_rows[:, 8]
        _assign_scalar_field("U_DISLOCDEN", updated_density)

    mesh_state["_runtime_seed_unode_fields"] = {
        **seed_fields,
        "values": field_values,
        "field_order": tuple(field_order),
    }
    mesh_state["_runtime_mechanics_snapshot"] = {
        "paths": {
            "temp_matrix_path": str(snapshot.paths.temp_matrix_path),
            "unode_strain_path": str(snapshot.paths.unode_strain_path),
            "unode_euler_path": str(snapshot.paths.unode_euler_path),
            "tex_path": snapshot.paths.tex_path,
        },
        "alignment_mode": str(alignment_mode),
        "euler_alignment_mode": str(euler_alignment_mode),
        "temp_matrix": np.asarray(snapshot.temp_matrix, dtype=np.float64),
        "unode_strain_xyz": np.asarray(snapshot.unode_strain_xyz[row_order], dtype=np.float64),
    }
    mesh_state.setdefault("stats", {})
    mesh_state["stats"]["mechanics_snapshot_applied"] = 1
    mesh_state["stats"]["mechanics_snapshot_alignment_mode"] = str(alignment_mode)
    mesh_state["stats"]["mechanics_snapshot_euler_alignment_mode"] = str(euler_alignment_mode)
    mesh_state["stats"]["mechanics_snapshot_updated_unodes"] = int(seed_ids.size)
    mesh_state["stats"]["mechanics_snapshot_has_tex"] = int(snapshot.tex_columns is not None)
    return mesh_state, {
        "mechanics_applied": 1,
        "alignment_mode": str(alignment_mode),
        "euler_alignment_mode": str(euler_alignment_mode),
        "updated_unodes": int(seed_ids.size),
        "has_tex": int(snapshot.tex_columns is not None),
    }
