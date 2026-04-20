from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np

from .faithful_config import FaithfulElleOptions
from .mesh import (
    _distance_weighted_vector_value,
    _legacy_fs_roi,
    _nearest_same_label_vector_value,
    assign_seed_unodes_from_mesh,
)


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


@dataclass(frozen=True)
class LegacyFFTBridgePayload:
    """Named legacy bridge contract derived from one frozen mechanics snapshot."""

    temp_matrix_path: str
    unode_strain_path: str
    unode_euler_path: str
    tex_path: str | None
    alignment_mode: str
    euler_alignment_mode: str
    ordered_unode_ids: np.ndarray
    temp_matrix: np.ndarray
    cell_lengths: np.ndarray
    cell_strain_triplet: np.ndarray
    cell_shear_triplet: np.ndarray
    unode_strain_xyz: np.ndarray
    unode_euler_deg: np.ndarray
    normalized_strain_rate: np.ndarray | None
    normalized_stress: np.ndarray | None
    basal_activity: np.ndarray | None
    prismatic_activity: np.ndarray | None
    geometrical_dislocation_density_increment: np.ndarray | None
    statistical_dislocation_density: np.ndarray | None
    fourier_point_ids: np.ndarray | None
    fft_grain_numbers: np.ndarray | None


@dataclass(frozen=True)
class LegacyFFTImportOptions:
    """Legacy-style mechanics-import controls from FS_fft2elle user data."""

    import_dislocation_densities: bool = True
    exclude_phase_id: int = 0
    density_update_mode: str = "increment"
    host_repair_mode: str = "fs_check_unodes"


@dataclass(frozen=True)
class LegacyElle2FFTBridgePayload:
    """Named legacy export contract for the ELLE -> FFT bridge."""

    grain_count: int
    grain_rows: tuple[tuple[float, float, float, float, int, int], ...]
    point_rows: tuple[tuple[float, float, float, int, int, int, int, int], ...]
    temp_rows: tuple[tuple[float, float, float], ...]


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


def build_legacy_fft_bridge_payload(
    seed_unode_ids: np.ndarray,
    snapshot: FFTMechanicsSnapshot,
) -> LegacyFFTBridgePayload:
    """Align one mechanics snapshot to faithful seed unodes and expose named bridge channels."""

    ordered_ids = np.asarray(seed_unode_ids, dtype=np.int32)
    if ordered_ids.ndim != 1 or ordered_ids.size == 0:
        raise ValueError("seed_unode_ids must be a non-empty 1-D array")

    row_order, alignment_mode = _align_snapshot_rows(ordered_ids, snapshot.unode_ids)
    euler_row_order, euler_alignment_mode = _align_snapshot_rows(ordered_ids, snapshot.euler_ids)
    aligned_temp = np.asarray(snapshot.temp_matrix, dtype=np.float64)
    tex_rows = None if snapshot.tex_columns is None else np.asarray(snapshot.tex_columns, dtype=np.float64)[row_order]

    def _optional_column(column_index: int) -> np.ndarray | None:
        if tex_rows is None:
            return None
        return np.asarray(tex_rows[:, int(column_index)], dtype=np.float64)

    return LegacyFFTBridgePayload(
        temp_matrix_path=str(snapshot.paths.temp_matrix_path),
        unode_strain_path=str(snapshot.paths.unode_strain_path),
        unode_euler_path=str(snapshot.paths.unode_euler_path),
        tex_path=snapshot.paths.tex_path,
        alignment_mode=str(alignment_mode),
        euler_alignment_mode=str(euler_alignment_mode),
        ordered_unode_ids=ordered_ids.copy(),
        temp_matrix=aligned_temp,
        cell_lengths=np.asarray(aligned_temp[0], dtype=np.float64),
        cell_strain_triplet=np.asarray(aligned_temp[1], dtype=np.float64),
        cell_shear_triplet=np.asarray(aligned_temp[2], dtype=np.float64),
        unode_strain_xyz=np.asarray(snapshot.unode_strain_xyz[row_order], dtype=np.float64),
        unode_euler_deg=np.asarray(snapshot.unode_euler_deg[euler_row_order], dtype=np.float64),
        normalized_strain_rate=_optional_column(4),
        normalized_stress=_optional_column(5),
        basal_activity=_optional_column(6),
        prismatic_activity=_optional_column(7),
        geometrical_dislocation_density_increment=_optional_column(8),
        statistical_dislocation_density=_optional_column(9),
        fourier_point_ids=(
            None
            if tex_rows is None
            else np.asarray(np.rint(tex_rows[:, 10]), dtype=np.int32)
        ),
        fft_grain_numbers=(
            None
            if tex_rows is None
            else np.asarray(np.rint(tex_rows[:, 11]), dtype=np.int32)
        ),
    )


def _legacy_elle2fft_resolve_paths(source: str | Path) -> tuple[Path, Path]:
    source_path = Path(source)
    if source_path.is_file():
        if source_path.name != "make.out":
            raise ValueError(
                "load_legacy_elle2fft_bridge_payload expects a directory or make.out path"
            )
        make_path = source_path
        temp_path = source_path.with_name("temp.out")
    else:
        make_path = source_path / "make.out"
        temp_path = source_path / "temp.out"
    if not make_path.exists():
        raise FileNotFoundError(f"missing legacy ELLE -> FFT make.out in {source_path}")
    if not temp_path.exists():
        raise FileNotFoundError(f"missing legacy ELLE -> FFT temp.out in {source_path}")
    return make_path, temp_path


def _parse_legacy_elle2fft_temp_rows(path: str | Path) -> tuple[tuple[float, float, float], ...]:
    rows: list[tuple[float, float, float]] = []
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        values = [float(value) for value in line.split()]
        if len(values) == 1:
            values = [values[0], 0.0, 0.0]
        elif len(values) == 2:
            values = [values[0], values[1], 0.0]
        elif len(values) >= 3:
            values = values[:3]
        rows.append((float(values[0]), float(values[1]), float(values[2])))
    if len(rows) < 3:
        raise ValueError(f"{path} has only {len(rows)} temp rows; expected at least 3")
    return tuple(rows[:3])


def load_legacy_elle2fft_bridge_payload(source: str | Path) -> LegacyElle2FFTBridgePayload:
    """Load one frozen legacy ELLE -> FFT export payload from make.out/temp.out files."""

    make_path, temp_path = _legacy_elle2fft_resolve_paths(source)
    lines = [
        raw_line.strip()
        for raw_line in make_path.read_text(encoding="utf-8").splitlines()
        if raw_line.strip() and not raw_line.strip().startswith("#")
    ]
    if not lines:
        raise ValueError(f"{make_path} is empty")
    grain_count = int(round(float(lines[0].split()[0])))
    if grain_count < 0:
        raise ValueError(f"{make_path} has a negative grain count {grain_count}")
    if len(lines) < 1 + grain_count:
        raise ValueError(
            f"{make_path} has only {len(lines) - 1} grain/point rows after the header; expected at least {grain_count}"
        )

    grain_rows: list[tuple[float, float, float, float, int, int]] = []
    for raw_line in lines[1 : 1 + grain_count]:
        parts = raw_line.split()
        if len(parts) < 6:
            raise ValueError(f"{make_path} has an invalid grain row: {raw_line}")
        grain_rows.append(
            (
                float(parts[0]),
                float(parts[1]),
                float(parts[2]),
                float(parts[3]),
                int(round(float(parts[4]))),
                int(round(float(parts[5]))),
            )
        )

    point_rows: list[tuple[float, float, float, int, int, int, int, int]] = []
    for raw_line in lines[1 + grain_count :]:
        parts = raw_line.split()
        if len(parts) < 8:
            raise ValueError(f"{make_path} has an invalid FFT point row: {raw_line}")
        point_rows.append(
            (
                float(parts[0]),
                float(parts[1]),
                float(parts[2]),
                int(round(float(parts[3]))),
                int(round(float(parts[4]))),
                int(round(float(parts[5]))),
                int(round(float(parts[6]))),
                int(round(float(parts[7]))),
            )
        )
    return LegacyElle2FFTBridgePayload(
        grain_count=int(grain_count),
        grain_rows=tuple(grain_rows),
        point_rows=tuple(point_rows),
        temp_rows=_parse_legacy_elle2fft_temp_rows(temp_path),
    )


def compare_legacy_elle2fft_bridge_payload(
    candidate: LegacyElle2FFTBridgePayload,
    reference: LegacyElle2FFTBridgePayload,
) -> dict[str, object]:
    """Compare two ELLE -> FFT bridge payloads and summarize the remaining gap."""

    candidate_grain = np.asarray(candidate.grain_rows, dtype=np.float64)
    reference_grain = np.asarray(reference.grain_rows, dtype=np.float64)
    candidate_points = np.asarray(candidate.point_rows, dtype=np.float64)
    reference_points = np.asarray(reference.point_rows, dtype=np.float64)
    candidate_temp = np.asarray(candidate.temp_rows, dtype=np.float64)
    reference_temp = np.asarray(reference.temp_rows, dtype=np.float64)

    grain_shape_match = candidate_grain.shape == reference_grain.shape
    point_shape_match = candidate_points.shape == reference_points.shape
    temp_shape_match = candidate_temp.shape == reference_temp.shape

    report: dict[str, object] = {
        "grain_count_candidate": int(candidate.grain_count),
        "grain_count_reference": int(reference.grain_count),
        "grain_count_match": int(int(candidate.grain_count) == int(reference.grain_count)),
        "grain_rows_shape_match": int(grain_shape_match),
        "point_rows_shape_match": int(point_shape_match),
        "temp_rows_shape_match": int(temp_shape_match),
    }

    if grain_shape_match and candidate_grain.size:
        flynn_id_match = np.asarray(
            np.rint(candidate_grain[:, 5]) == np.rint(reference_grain[:, 5]),
            dtype=bool,
        )
        report["grain_header_flynn_id_match_count"] = int(np.count_nonzero(flynn_id_match))
        report["grain_header_flynn_id_total"] = int(flynn_id_match.size)
        report["grain_header_euler_rmse"] = float(
            np.sqrt(
                np.mean(
                    np.square(candidate_grain[:, :3] - reference_grain[:, :3], dtype=np.float64),
                    dtype=np.float64,
                )
            )
        )

    if point_shape_match and candidate_points.size:
        point_euler_rmse = float(
            np.sqrt(
                np.mean(
                    np.square(
                        candidate_points[:, :3] - reference_points[:, :3],
                        dtype=np.float64,
                    ),
                    dtype=np.float64,
                )
            )
        )
        report["point_euler_rmse"] = point_euler_rmse
        point_grid_mismatch_count = int(
            np.count_nonzero(
                np.any(
                    np.rint(candidate_points[:, 3:6]) != np.rint(reference_points[:, 3:6]),
                    axis=1,
                )
            )
        )
        report["point_grid_mismatch_count"] = point_grid_mismatch_count
        point_grain_mismatch_count = int(
            np.count_nonzero(np.rint(candidate_points[:, 6]) != np.rint(reference_points[:, 6]))
        )
        report["point_grain_mismatch_count"] = point_grain_mismatch_count
        point_phase_mismatch_count = int(
            np.count_nonzero(np.rint(candidate_points[:, 7]) != np.rint(reference_points[:, 7]))
        )
        report["point_phase_mismatch_count"] = point_phase_mismatch_count
        point_rows_numeric_rmse = float(
            np.sqrt(
                np.mean(
                    np.square(candidate_points - reference_points, dtype=np.float64),
                    dtype=np.float64,
                )
            )
        )
        report["point_rows_numeric_rmse"] = point_rows_numeric_rmse
        report["point_contract_match"] = int(
            point_grid_mismatch_count == 0
            and point_grain_mismatch_count == 0
            and point_phase_mismatch_count == 0
            and point_euler_rmse <= 5.0e-3
        )

    if temp_shape_match and candidate_temp.size:
        temp_rows_numeric_rmse = float(
            np.sqrt(
                np.mean(
                    np.square(candidate_temp - reference_temp, dtype=np.float64),
                    dtype=np.float64,
                )
            )
        )
        report["temp_rows_numeric_rmse"] = temp_rows_numeric_rmse
        report["temp_contract_match"] = int(temp_rows_numeric_rmse <= 1.0e-12)

    grain_header_contract_match = 0
    if (
        grain_shape_match
        and candidate_grain.size
        and report.get("grain_header_flynn_id_match_count", 0) == report.get("grain_header_flynn_id_total", -1)
        and float(report.get("grain_header_euler_rmse", np.inf)) <= 5.0e-3
    ):
        grain_header_contract_match = 1
    report["grain_header_contract_match"] = int(grain_header_contract_match)
    report["bridge_contract_match_excluding_grain_headers"] = int(
        int(report.get("grain_count_match", 0)) == 1
        and int(report.get("point_contract_match", 0)) == 1
        and int(report.get("temp_contract_match", 0)) == 1
    )
    report["bridge_contract_match_full"] = int(
        int(report["bridge_contract_match_excluding_grain_headers"]) == 1
        and int(report["grain_header_contract_match"]) == 1
    )
    report["bridge_header_only_mismatch"] = int(
        int(report["bridge_contract_match_excluding_grain_headers"]) == 1
        and int(report["grain_header_contract_match"]) == 0
    )

    return report


def diagnose_legacy_elle2fft_header_sources(
    mesh_state: dict[str, object],
    reference: LegacyElle2FFTBridgePayload,
) -> dict[str, object]:
    """Compare reference grain-header Euler rows against plausible faithful source surfaces."""

    if int(reference.grain_count) <= 0 or not reference.grain_rows:
        return {
            "candidate_rmse": {},
            "best_candidate": "none",
            "best_candidate_rmse": 0.0,
        }

    seed_unodes = mesh_state.get("_runtime_seed_unodes")
    seed_sections = mesh_state.get("_runtime_seed_unode_sections")
    if not isinstance(seed_unodes, dict) or not isinstance(seed_sections, dict):
        return {
            "candidate_rmse": {},
            "best_candidate": "unavailable",
            "best_candidate_rmse": float("inf"),
        }

    raw_unode_euler = dict(seed_sections.get("values", {})).get("U_EULER_3")
    if raw_unode_euler is None:
        return {
            "candidate_rmse": {},
            "best_candidate": "unavailable",
            "best_candidate_rmse": float("inf"),
        }
    unode_euler = np.asarray(raw_unode_euler, dtype=np.float64)
    grid_indices = np.asarray(seed_unodes.get("grid_indices", ()), dtype=np.int32)
    if (
        unode_euler.ndim != 2
        or unode_euler.shape[0] == 0
        or unode_euler.shape[1] != 3
        or grid_indices.shape != (unode_euler.shape[0], 2)
    ):
        return {
            "candidate_rmse": {},
            "best_candidate": "unavailable",
            "best_candidate_rmse": float("inf"),
        }

    flynns = [
        dict(flynn)
        for flynn in mesh_state.get("flynns", ())
        if isinstance(flynn, dict) and "flynn_id" in flynn
    ]
    if not flynns:
        return {
            "candidate_rmse": {},
            "best_candidate": "unavailable",
            "best_candidate_rmse": float("inf"),
        }
    label_to_flynn_id = {
        int(flynn.get("label", index)): int(flynn["flynn_id"])
        for index, flynn in enumerate(flynns)
    }
    label_grid, _ = assign_seed_unodes_from_mesh(mesh_state, seed_unodes)
    point_flynn_ids = np.asarray(
        [label_to_flynn_id.get(int(label_grid[int(ix), int(iy)]), -1) for ix, iy in grid_indices],
        dtype=np.int32,
    )

    flynn_euler_lookup = _legacy_elle2fft_flynn_section_lookup(mesh_state, "EULER_3")
    reference_headers = {
        int(row[5]): np.asarray(row[:3], dtype=np.float64)
        for row in reference.grain_rows
    }
    candidate_errors: dict[str, list[float]] = {
        "flynn_euler_3": [],
        "mean_unode_euler_3": [],
        "median_unode_euler_3": [],
        "first_unode_euler_3": [],
    }

    for flynn_id, target in reference_headers.items():
        mask = np.asarray(point_flynn_ids == int(flynn_id), dtype=bool)
        if not np.any(mask):
            continue
        samples = np.asarray(unode_euler[mask], dtype=np.float64)
        flynn_euler = flynn_euler_lookup.get(int(flynn_id))
        if isinstance(flynn_euler, tuple) and len(flynn_euler) >= 3:
            flynn_values = np.asarray(flynn_euler[:3], dtype=np.float64)
            candidate_errors["flynn_euler_3"].append(
                float(np.sqrt(np.mean(np.square(flynn_values - target), dtype=np.float64)))
            )
        candidate_errors["mean_unode_euler_3"].append(
            float(
                np.sqrt(
                    np.mean(
                        np.square(np.mean(samples, axis=0, dtype=np.float64) - target),
                        dtype=np.float64,
                    )
                )
            )
        )
        candidate_errors["median_unode_euler_3"].append(
            float(
                np.sqrt(
                    np.mean(
                        np.square(np.median(samples, axis=0) - target),
                        dtype=np.float64,
                    )
                )
            )
        )
        candidate_errors["first_unode_euler_3"].append(
            float(np.sqrt(np.mean(np.square(samples[0] - target), dtype=np.float64)))
        )

    candidate_rmse = {
        name: (
            float(np.mean(values, dtype=np.float64))
            if values
            else float("inf")
        )
        for name, values in candidate_errors.items()
    }
    best_candidate = min(candidate_rmse, key=candidate_rmse.get) if candidate_rmse else "none"
    best_candidate_rmse = float(candidate_rmse.get(best_candidate, 0.0))
    return {
        "candidate_rmse": candidate_rmse,
        "best_candidate": str(best_candidate),
        "best_candidate_rmse": float(best_candidate_rmse),
    }


def _runtime_seed_unode_row_order(seed_unodes: dict[str, object]) -> np.ndarray:
    grid_indices = np.asarray(seed_unodes.get("grid_indices", ()), dtype=np.int32)
    if grid_indices.ndim != 2 or grid_indices.shape[0] == 0 or grid_indices.shape[1] != 2:
        raise ValueError("faithful seed unodes do not contain a valid grid_indices payload")
    return np.lexsort((grid_indices[:, 0], grid_indices[:, 1]))


def _legacy_elle2fft_flynn_section_lookup(
    mesh_state: dict[str, object],
    name: str,
) -> dict[int, tuple[float, ...] | float]:
    flynn_sections = mesh_state.get("_runtime_seed_flynn_sections")
    if not isinstance(flynn_sections, dict):
        return {}
    raw_values = dict(flynn_sections.get("values", {})).get(str(name))
    id_order = tuple(int(value) for value in flynn_sections.get("id_order", ()))
    if raw_values is None or len(id_order) != len(raw_values):
        return {}
    lookup: dict[int, tuple[float, ...] | float] = {}
    for flynn_id, value in zip(id_order, raw_values):
        if isinstance(value, (tuple, list, np.ndarray)):
            entries = tuple(float(entry) for entry in value)
            lookup[int(flynn_id)] = entries if len(entries) != 1 else float(entries[0])
        else:
            lookup[int(flynn_id)] = float(value)
    return lookup


def _legacy_elle2fft_point_phase_ids(
    mesh_state: dict[str, object],
    field_name: str = "U_VISCOSITY",
) -> np.ndarray | None:
    seed_fields = mesh_state.get("_runtime_seed_unode_fields")
    if not isinstance(seed_fields, dict):
        return None
    raw_values = dict(seed_fields.get("values", {})).get(str(field_name))
    if raw_values is None:
        return None
    phase_ids = np.asarray(np.rint(raw_values), dtype=np.int32)
    return phase_ids if phase_ids.ndim == 1 and phase_ids.size else None


def _legacy_elle2fft_phase_lookup(
    mesh_state: dict[str, object],
    *,
    phase_attribute: str = "auto",
) -> tuple[dict[int, int], np.ndarray | None, str]:
    requested = str(phase_attribute)
    attribute_candidates: list[str]
    if requested.lower() == "auto":
        attribute_candidates = ["VISCOSITY", "DISLOCDEN", "U_VISCOSITY", "U_DISLOCDEN"]
    else:
        attribute_candidates = [requested]

    for name in attribute_candidates:
        if str(name).startswith("U_"):
            unode_phase_ids = _legacy_elle2fft_point_phase_ids(mesh_state, field_name=str(name))
            if unode_phase_ids is not None:
                return {}, unode_phase_ids, str(name)
            continue
        flynn_lookup = _legacy_elle2fft_flynn_section_lookup(mesh_state, str(name))
        if flynn_lookup:
            phase_lookup: dict[int, int] = {}
            for flynn_id, raw_value in flynn_lookup.items():
                if isinstance(raw_value, tuple):
                    if not raw_value:
                        continue
                    numeric_value = float(raw_value[0])
                else:
                    numeric_value = float(raw_value)
                phase_lookup[int(flynn_id)] = int(round(numeric_value))
            if phase_lookup:
                return phase_lookup, None, str(name)
    return {}, None, "default_1"


def _mean_legacy_elle2fft_flynn_euler(
    euler_rows: np.ndarray,
    point_flynn_ids: np.ndarray,
    target_flynn_id: int,
) -> tuple[float, float, float] | None:
    mask = np.asarray(point_flynn_ids == int(target_flynn_id), dtype=bool)
    if not np.any(mask):
        return None
    mean_euler = np.mean(np.asarray(euler_rows[mask], dtype=np.float64), axis=0, dtype=np.float64)
    return (
        float(mean_euler[0]),
        float(mean_euler[1]),
        float(mean_euler[2]),
    )


def build_legacy_elle2fft_bridge_payload(
    mesh_state: dict[str, object],
    *,
    phase_attribute: str = "auto",
    include_grain_headers: bool = True,
) -> LegacyElle2FFTBridgePayload:
    """Build a faithful ELLE -> FFT bridge payload from the current runtime state."""

    seed_unodes = mesh_state.get("_runtime_seed_unodes")
    if not isinstance(seed_unodes, dict):
        raise ValueError("mesh_state has no faithful seed-unode payload")
    row_order = _runtime_seed_unode_row_order(seed_unodes)
    grid_indices = np.asarray(seed_unodes.get("grid_indices", ()), dtype=np.int32)
    grid_shape = tuple(int(value) for value in seed_unodes.get("grid_shape", ()))
    if len(grid_shape) != 2 or min(grid_shape) <= 0:
        raise ValueError("mesh_state has no valid faithful seed grid shape")

    seed_sections = mesh_state.get("_runtime_seed_unode_sections")
    if not isinstance(seed_sections, dict):
        raise ValueError("mesh_state has no faithful seed-unode section payload")
    section_values = dict(seed_sections.get("values", {}))
    raw_unode_euler = section_values.get("U_EULER_3")
    if raw_unode_euler is None:
        raise ValueError("mesh_state has no U_EULER_3 unode section for ELLE -> FFT export")
    unode_euler = np.asarray(raw_unode_euler, dtype=np.float64)
    if unode_euler.ndim != 2 or unode_euler.shape[0] != grid_indices.shape[0] or unode_euler.shape[1] != 3:
        raise ValueError("mesh_state has an invalid U_EULER_3 payload for ELLE -> FFT export")

    flynns = [
        dict(flynn)
        for flynn in mesh_state.get("flynns", ())
        if isinstance(flynn, dict) and len(tuple(flynn.get("node_ids", ()))) >= 3
    ]
    if not flynns:
        raise ValueError("mesh_state has no active flynns for ELLE -> FFT export")
    flynns.sort(key=lambda flynn: int(flynn.get("flynn_id", -1)))

    label_to_flynn_id = {
        int(flynn.get("label", index)): int(flynn.get("flynn_id", index))
        for index, flynn in enumerate(flynns)
    }
    active_flynn_ids = tuple(int(flynn["flynn_id"]) for flynn in flynns)
    flynn_to_grain_index = {
        int(flynn_id): int(index + 1) for index, flynn_id in enumerate(active_flynn_ids)
    }

    label_grid, _ = assign_seed_unodes_from_mesh(mesh_state, seed_unodes)
    point_labels = np.asarray(
        [label_grid[int(ix), int(iy)] for ix, iy in grid_indices],
        dtype=np.int32,
    )
    point_flynn_ids = np.asarray(
        [label_to_flynn_id.get(int(label), -1) for label in point_labels],
        dtype=np.int32,
    )

    flynn_euler_lookup = _legacy_elle2fft_flynn_section_lookup(mesh_state, "EULER_3")
    flynn_phase_lookup, unode_phase_ids, _phase_source = _legacy_elle2fft_phase_lookup(
        mesh_state,
        phase_attribute=str(phase_attribute),
    )

    grain_rows: list[tuple[float, float, float, float, int, int]] = []
    if include_grain_headers:
        for flynn in flynns:
            flynn_id = int(flynn["flynn_id"])
            source_flynn_id = int(flynn.get("source_flynn_id", flynn_id))
            raw_flynn_euler = flynn_euler_lookup.get(flynn_id, flynn_euler_lookup.get(source_flynn_id))
            if isinstance(raw_flynn_euler, tuple) and len(raw_flynn_euler) >= 3:
                flynn_euler = (
                    float(raw_flynn_euler[0]),
                    float(raw_flynn_euler[1]),
                    float(raw_flynn_euler[2]),
                )
            else:
                flynn_euler = _mean_legacy_elle2fft_flynn_euler(unode_euler, point_flynn_ids, flynn_id)
                if flynn_euler is None:
                    flynn_euler = (0.0, 0.0, 0.0)
            grain_rows.append(
                (
                    float(flynn_euler[0]),
                    float(flynn_euler[1]),
                    float(flynn_euler[2]),
                    float(flynn_euler[0]),
                    0,
                    int(flynn_id),
                )
            )

    point_rows: list[tuple[float, float, float, int, int, int, int, int]] = []
    for point_index in row_order:
        point_index = int(point_index)
        x_index = int(grid_indices[point_index, 0])
        y_index = int(grid_indices[point_index, 1])
        flynn_id = int(point_flynn_ids[point_index])
        grain_index = int(flynn_to_grain_index.get(flynn_id, 0)) if include_grain_headers else 0
        raw_phase_value = flynn_phase_lookup.get(flynn_id)
        if raw_phase_value is None and unode_phase_ids is not None and point_index < unode_phase_ids.size:
            phase_id = int(unode_phase_ids[point_index])
        elif isinstance(raw_phase_value, tuple):
            phase_id = int(round(float(raw_phase_value[0])))
        elif raw_phase_value is not None:
            phase_id = int(round(float(raw_phase_value)))
        else:
            phase_id = 1
        euler = np.asarray(unode_euler[point_index], dtype=np.float64)
        point_rows.append(
            (
                float(euler[0]),
                float(euler[1]),
                float(euler[2]),
                int(x_index + 1),
                int(y_index + 1),
                1,
                int(grain_index),
                int(phase_id),
            )
        )

    elle_options = FaithfulElleOptions.from_runtime_dict(
        mesh_state.get("_runtime_elle_options") if isinstance(mesh_state, dict) else None
    )
    box = tuple((float(x), float(y)) for x, y in elle_options.cell_bounding_box)
    xlength = float(box[1][0] - box[0][0])
    ylength = float(box[3][1] - box[0][1])
    temp_rows = (
        (float(xlength), float(ylength), 1.0),
        (0.0, 0.0, 0.0),
        (float(elle_options.simple_shear_offset), 0.0, 0.0),
    )
    return LegacyElle2FFTBridgePayload(
        grain_count=int(len(grain_rows)) if include_grain_headers else 0,
        grain_rows=tuple(grain_rows),
        point_rows=tuple(point_rows),
        temp_rows=temp_rows,
    )


def _format_legacy_make_float(value: float) -> str:
    return format(float(value), ".5g")


def write_legacy_elle2fft_bridge_payload(
    destination: str | Path,
    payload: LegacyElle2FFTBridgePayload,
) -> tuple[str, str]:
    """Write one faithful ELLE -> FFT bridge payload as make.out and temp.out."""

    destination_path = Path(destination)
    destination_path.mkdir(parents=True, exist_ok=True)
    make_path = destination_path / "make.out"
    temp_path = destination_path / "temp.out"

    with make_path.open("w", encoding="utf-8") as handle:
        handle.write(f"{int(payload.grain_count)}\n")
        for row in payload.grain_rows:
            handle.write(
                "\t".join(
                    (
                        _format_legacy_make_float(row[0]),
                        _format_legacy_make_float(row[1]),
                        _format_legacy_make_float(row[2]),
                        _format_legacy_make_float(row[3]),
                        str(int(row[4])),
                        str(int(row[5])),
                    )
                )
                + "\n"
            )
        for row in payload.point_rows:
            handle.write(
                "\t".join(
                    (
                        _format_legacy_make_float(row[0]),
                        _format_legacy_make_float(row[1]),
                        _format_legacy_make_float(row[2]),
                        str(int(row[3])),
                        str(int(row[4])),
                        str(int(row[5])),
                        str(int(row[6])),
                        str(int(row[7])),
                    )
                )
                + "\n"
            )

    with temp_path.open("w", encoding="utf-8") as handle:
        for row in payload.temp_rows:
            handle.write(
                " ".join(f"{float(value): .6f}" for value in row).rstrip() + "\n"
            )

    return str(make_path), str(temp_path)


def _point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    x_coord = float(point[0])
    y_coord = float(point[1])
    inside = False
    count = int(polygon.shape[0])
    if count < 3:
        return False
    prev_x = float(polygon[-1, 0])
    prev_y = float(polygon[-1, 1])
    for index in range(count):
        curr_x = float(polygon[index, 0])
        curr_y = float(polygon[index, 1])
        intersects = ((curr_y > y_coord) != (prev_y > y_coord)) and (
            x_coord
            < (prev_x - curr_x) * (y_coord - curr_y) / ((prev_y - curr_y) + 1.0e-12)
            + curr_x
        )
        if intersects:
            inside = not inside
        prev_x = curr_x
        prev_y = curr_y
    return bool(inside)


def _apply_legacy_unode_position_update(
    seed_unodes: dict[str, object],
    payload: LegacyFFTBridgePayload,
) -> tuple[dict[str, object], dict[str, object]]:
    positions = np.asarray(seed_unodes.get("positions", ()), dtype=np.float64)
    if positions.ndim != 2 or positions.shape[0] == 0 or positions.shape[1] != 2:
        return dict(seed_unodes), {
            "updated_unode_positions": 0,
            "position_update_mode": "unavailable",
        }

    target_xy = np.asarray(payload.unode_strain_xyz[:, :2], dtype=np.float64)
    if target_xy.shape != positions.shape:
        return dict(seed_unodes), {
            "updated_unode_positions": 0,
            "position_update_mode": "shape_mismatch",
        }

    # FS_fft2elle only updates Y from unodexyz.out in pure-shear style input;
    # in simple shear the imported file primarily carries the X transport.
    pure_shear_y = bool(abs(float(payload.cell_strain_triplet[1])) > 1.0e-12)
    if not pure_shear_y:
        target_xy[:, 1] = positions[:, 1]
    target_xy = np.mod(target_xy, 1.0)

    updated_seed_unodes = dict(seed_unodes)
    updated_seed_unodes["positions"] = tuple(
        (float(x_coord), float(y_coord)) for x_coord, y_coord in target_xy
    )
    return updated_seed_unodes, {
        "updated_unode_positions": int(target_xy.shape[0]),
        "position_update_mode": "pure_shear_xy" if pure_shear_y else "simple_shear_x_only",
    }


def _apply_legacy_node_position_update(
    mesh_state: dict[str, object],
    original_seed_unodes: dict[str, object],
    updated_seed_unodes: dict[str, object],
    previous_sample_labels: np.ndarray | None,
) -> tuple[dict[str, object], dict[str, object]]:
    raw_nodes = mesh_state.get("nodes")
    raw_flynns = mesh_state.get("flynns")
    if not isinstance(raw_nodes, list) or not isinstance(raw_flynns, list):
        return mesh_state, {
            "updated_node_positions": 0,
            "node_position_update_mode": "unavailable",
        }

    sample_points = np.asarray(original_seed_unodes.get("positions", ()), dtype=np.float64)
    updated_sample_points = np.asarray(updated_seed_unodes.get("positions", ()), dtype=np.float64)
    if (
        previous_sample_labels is None
        or sample_points.ndim != 2
        or updated_sample_points.shape != sample_points.shape
        or sample_points.shape[0] == 0
    ):
        return mesh_state, {
            "updated_node_positions": 0,
            "node_position_update_mode": "invalid_seed_grid",
        }

    unode_displacements = np.asarray(updated_sample_points - sample_points, dtype=np.float64)
    unode_displacements = np.mod(unode_displacements + 0.5, 1.0) - 0.5
    roi = float(_legacy_fs_roi(original_seed_unodes, mesh_state.get("_runtime_elle_options"), factor=3))

    node_to_labels: dict[int, list[int]] = {}
    for flynn in raw_flynns:
        if not isinstance(flynn, dict):
            continue
        if "label" not in flynn:
            continue
        for node_id in flynn.get("node_ids", ()):
            node_to_labels.setdefault(int(node_id), []).append(int(flynn["label"]))

    updated_nodes: list[dict[str, object]] = []
    updated_count = 0
    for node_index, raw_node in enumerate(raw_nodes):
        if not isinstance(raw_node, dict) or "x" not in raw_node or "y" not in raw_node:
            updated_nodes.append(raw_node)
            continue
        node_xy = np.asarray([float(raw_node["x"]), float(raw_node["y"])], dtype=np.float64)
        weighted_sum = np.zeros((2,), dtype=np.float64)
        weight_total = 0.0
        for label in node_to_labels.get(int(node_index), ()):
            donor_mask = np.asarray(previous_sample_labels == int(label), dtype=bool)
            donor_indices = np.flatnonzero(donor_mask)
            if donor_indices.size == 0:
                continue
            deltas = sample_points[donor_indices] - node_xy[None, :]
            deltas = np.mod(deltas + 0.5, 1.0) - 0.5
            distances = np.linalg.norm(deltas, axis=1)
            within_roi = distances < float(roi)
            if not np.any(within_roi):
                continue
            weights = float(roi) - distances[within_roi]
            weighted_sum += np.sum(
                unode_displacements[donor_indices[within_roi]] * weights[:, None],
                axis=0,
                dtype=np.float64,
            )
            weight_total += float(np.sum(weights, dtype=np.float64))

        if weight_total <= 1.0e-12:
            donor_indices = np.arange(sample_points.shape[0], dtype=np.int32)
            deltas = sample_points[donor_indices] - node_xy[None, :]
            deltas = np.mod(deltas + 0.5, 1.0) - 0.5
            distances = np.linalg.norm(deltas, axis=1)
            within_roi = distances < float(roi)
            if np.any(within_roi):
                weights = float(roi) - distances[within_roi]
                weighted_sum += np.sum(
                    unode_displacements[donor_indices[within_roi]] * weights[:, None],
                    axis=0,
                    dtype=np.float64,
                )
                weight_total += float(np.sum(weights, dtype=np.float64))

        node_displacement = (
            weighted_sum / weight_total if weight_total > 1.0e-12 else np.zeros((2,), dtype=np.float64)
        )
        if np.max(np.abs(node_displacement)) > 1.0e-12:
            updated_count += 1
        updated_xy = np.mod(node_xy + node_displacement, 1.0)
        updated_node = dict(raw_node)
        updated_node["x"] = float(updated_xy[0])
        updated_node["y"] = float(updated_xy[1])
        updated_nodes.append(updated_node)

    updated_mesh_state = dict(mesh_state)
    updated_mesh_state["nodes"] = updated_nodes
    return updated_mesh_state, {
        "updated_node_positions": int(updated_count),
        "node_position_update_mode": "legacy_bnode_strain_weighted",
    }


def _apply_legacy_cell_reset(
    runtime_elle_options: dict[str, object] | None,
    payload: LegacyFFTBridgePayload,
) -> tuple[dict[str, object], dict[str, object]]:
    elle_options = FaithfulElleOptions.from_runtime_dict(
        runtime_elle_options if isinstance(runtime_elle_options, dict) else None
    )
    box = tuple(
        (float(x_coord), float(y_coord)) for x_coord, y_coord in elle_options.cell_bounding_box
    )
    xstrain = float(payload.cell_strain_triplet[0])
    ystrain = float(payload.cell_strain_triplet[1])
    shear_increment = float(payload.cell_shear_triplet[0])

    updated_box = (
        (float(box[0][0]), float(box[0][1])),
        (float(box[1][0] + xstrain), float(box[1][1])),
        (float(box[2][0] + shear_increment + xstrain), float(box[2][1] + ystrain)),
        (float(box[3][0] + shear_increment), float(box[3][1] + ystrain)),
    )
    updated_cumulative_simple_shear = float(elle_options.cumulative_simple_shear + shear_increment)
    updated_simple_shear_offset = float(np.modf(updated_cumulative_simple_shear)[0])
    updated_options = FaithfulElleOptions(
        scalar_values={str(name): float(value) for name, value in elle_options.scalar_values.items()},
        cell_bounding_box=updated_box,
        simple_shear_offset=updated_simple_shear_offset,
        cumulative_simple_shear=updated_cumulative_simple_shear,
    )
    return updated_options.to_runtime_dict(), {
        "cell_reset_applied": 1,
        "simple_shear_increment": float(shear_increment),
        "simple_shear_offset": float(updated_simple_shear_offset),
        "cumulative_simple_shear": float(updated_cumulative_simple_shear),
    }


def _legacy_box_height_from_runtime_options(runtime_elle_options: dict[str, object] | None) -> float:
    elle_options = FaithfulElleOptions.from_runtime_dict(
        runtime_elle_options if isinstance(runtime_elle_options, dict) else None
    )
    box = tuple(
        (float(x_coord), float(y_coord)) for x_coord, y_coord in elle_options.cell_bounding_box
    )
    return float(box[3][1] - box[0][1])


def _legacy_direct_strain_axis_summary(
    reference_box_height: float,
    updated_elle_options: dict[str, object],
    cell_reset_stats: dict[str, object],
) -> dict[str, object]:
    cumulative_simple_shear = float(cell_reset_stats.get("cumulative_simple_shear", 0.0))
    simple_shear_increment = float(cell_reset_stats.get("simple_shear_increment", 0.0))
    if (
        abs(cumulative_simple_shear) > 1.0e-12
        or abs(simple_shear_increment) > 1.0e-12
    ):
        return {
            "direct_strain_axis": float(cumulative_simple_shear),
            "strain_axis_source": "cumulative_simple_shear",
        }

    current_box_height = _legacy_box_height_from_runtime_options(updated_elle_options)
    if abs(float(reference_box_height)) > 1.0e-12:
        vertical_shortening_pct = float(
            (1.0 - (current_box_height / float(reference_box_height))) * 100.0
        )
        return {
            "direct_strain_axis": vertical_shortening_pct,
            "strain_axis_source": "vertical_shortening_pct",
        }

    return {
        "direct_strain_axis": float(cumulative_simple_shear),
        "strain_axis_source": "cumulative_simple_shear",
    }


def _seed_flynn_phase_lookup(mesh_state: dict[str, object]) -> dict[int, int]:
    flynn_sections = mesh_state.get("_runtime_seed_flynn_sections")
    if not isinstance(flynn_sections, dict):
        return {}
    section_values = dict(flynn_sections.get("values", {}))
    raw_values = section_values.get("VISCOSITY")
    id_order = tuple(int(value) for value in flynn_sections.get("id_order", ()))
    if raw_values is None or len(id_order) != len(raw_values):
        return {}
    lookup: dict[int, int] = {}
    for flynn_id, raw_value in zip(id_order, raw_values):
        if isinstance(raw_value, (tuple, list, np.ndarray)):
            if len(raw_value) == 0:
                continue
            numeric_value = float(raw_value[0])
        else:
            numeric_value = float(raw_value)
        lookup[int(flynn_id)] = int(round(numeric_value))
    return lookup


def _seed_unode_phase_ids(mesh_state: dict[str, object]) -> np.ndarray | None:
    seed_unodes = mesh_state.get("_runtime_seed_unodes")
    if not isinstance(seed_unodes, dict):
        return None
    sample_points = np.asarray(seed_unodes.get("positions", ()), dtype=np.float64)
    if sample_points.ndim != 2 or sample_points.shape[0] == 0 or sample_points.shape[1] != 2:
        return None

    seed_fields = mesh_state.get("_runtime_seed_unode_fields")
    if isinstance(seed_fields, dict):
        field_values = dict(seed_fields.get("values", {}))
        raw_unode_viscosity = field_values.get("U_VISCOSITY")
        if raw_unode_viscosity is not None:
            phase_ids = np.asarray(np.rint(raw_unode_viscosity), dtype=np.int32)
            if phase_ids.shape == (sample_points.shape[0],):
                return phase_ids

    nodes = mesh_state.get("nodes")
    flynns = mesh_state.get("flynns")
    if not isinstance(nodes, list) or not isinstance(flynns, list):
        return None

    node_positions = np.asarray(
        [[float(node["x"]), float(node["y"])] for node in nodes],
        dtype=np.float64,
    )
    if node_positions.ndim != 2 or node_positions.shape[0] == 0 or node_positions.shape[1] != 2:
        return None

    phase_lookup = _seed_flynn_phase_lookup(mesh_state)
    if not phase_lookup:
        return None

    assigned_phase_ids = np.full(sample_points.shape[0], -1, dtype=np.int32)
    unassigned_mask = np.ones(sample_points.shape[0], dtype=bool)
    flynn_entries: list[tuple[float, int, np.ndarray]] = []
    for flynn in flynns:
        if not isinstance(flynn, dict):
            continue
        source_flynn_id = int(flynn.get("source_flynn_id", flynn.get("flynn_id", -1)))
        phase_id = phase_lookup.get(source_flynn_id)
        if phase_id is None:
            continue
        node_ids = [int(node_id) for node_id in flynn.get("node_ids", ())]
        if len(node_ids) < 3:
            continue
        try:
            polygon = np.asarray([node_positions[node_id] for node_id in node_ids], dtype=np.float64)
        except Exception:
            continue
        if polygon.ndim != 2 or polygon.shape[0] < 3 or polygon.shape[1] != 2:
            continue
        shifted = np.roll(polygon, -1, axis=0)
        signed_area = 0.5 * np.sum(
            polygon[:, 0] * shifted[:, 1] - shifted[:, 0] * polygon[:, 1],
            dtype=np.float64,
        )
        flynn_entries.append((abs(float(signed_area)), int(phase_id), polygon))

    for _, phase_id, polygon in sorted(flynn_entries, key=lambda item: item[0], reverse=True):
        if not np.any(unassigned_mask):
            break
        candidate_indices = np.flatnonzero(unassigned_mask)
        if candidate_indices.size == 0:
            break
        inside_indices = [
            int(index)
            for index in candidate_indices
            if _point_in_polygon(sample_points[int(index)], polygon)
        ]
        if not inside_indices:
            continue
        assigned_phase_ids[np.asarray(inside_indices, dtype=np.int32)] = int(phase_id)
        unassigned_mask[np.asarray(inside_indices, dtype=np.int32)] = False

    return assigned_phase_ids


def _legacy_density_update_mode(mode: str) -> str:
    normalized = str(mode).strip().lower()
    if normalized in {"increment", "add", "fs_increment"}:
        return "increment"
    if normalized in {"overwrite", "replace", "set", "legacy_overwrite"}:
        return "overwrite"
    raise ValueError(f"unsupported legacy FFT density_update_mode {mode!r}")


def _legacy_host_repair_mode(mode: str) -> str:
    normalized = str(mode).strip().lower().replace("-", "_")
    if normalized in {"fs_check_unodes", "fs_check", "fs"}:
        return "fs_check_unodes"
    if normalized in {"check_error", "legacy_check_error"}:
        return "check_error"
    raise ValueError(f"unsupported legacy FFT host_repair_mode {mode!r}")


def _sample_host_flynn_labels(
    mesh_state: dict[str, object],
    seed_unodes: dict[str, object],
) -> tuple[np.ndarray | None, dict[str, int]]:
    if not isinstance(mesh_state.get("nodes"), (list, tuple)) or not isinstance(
        mesh_state.get("flynns"), (list, tuple)
    ):
        return None, {
            "assigned_unodes": 0,
            "unassigned_unodes": 0,
            "raster_filled_unassigned": 0,
        }
    sample_grid_indices = np.asarray(seed_unodes.get("grid_indices", ()), dtype=np.int32)
    sample_points = np.asarray(seed_unodes.get("positions", ()), dtype=np.float64)
    if (
        sample_grid_indices.ndim != 2
        or sample_points.ndim != 2
        or sample_grid_indices.shape[0] == 0
        or sample_grid_indices.shape != (sample_points.shape[0], 2)
    ):
        return None, {
            "assigned_unodes": 0,
            "unassigned_unodes": 0,
            "raster_filled_unassigned": 0,
        }

    label_grid, assign_stats = assign_seed_unodes_from_mesh(mesh_state, seed_unodes)
    sample_labels = np.asarray(
        label_grid[sample_grid_indices[:, 0], sample_grid_indices[:, 1]],
        dtype=np.int32,
    )
    return sample_labels, {
        "assigned_unodes": int(assign_stats.get("assigned_unodes", 0)),
        "unassigned_unodes": int(assign_stats.get("unassigned_unodes", 0)),
        "raster_filled_unassigned": int(assign_stats.get("raster_filled_unassigned", 0)),
    }


def _sample_host_flynn_ids(
    mesh_state: dict[str, object],
    seed_unodes: dict[str, object],
) -> tuple[np.ndarray | None, dict[str, int]]:
    sample_labels, assign_stats = _sample_host_flynn_labels(mesh_state, seed_unodes)
    if sample_labels is None:
        return None, dict(assign_stats)

    label_to_flynn_id: dict[int, int] = {}
    for flynn in mesh_state.get("flynns", ()):
        if not isinstance(flynn, dict):
            continue
        if "label" not in flynn or "flynn_id" not in flynn:
            continue
        label_to_flynn_id[int(flynn["label"])] = int(flynn["flynn_id"])
    if not label_to_flynn_id:
        return None, dict(assign_stats)

    flynn_ids = np.asarray(
        [label_to_flynn_id.get(int(label), -1) for label in np.asarray(sample_labels, dtype=np.int32)],
        dtype=np.int32,
    )
    if np.any(flynn_ids < 0):
        return None, dict(assign_stats)
    return flynn_ids, dict(assign_stats)


def _legacy_initialize_mechanics_tracers(
    mesh_state: dict[str, object],
    *,
    field_values: dict[str, object],
    field_order: list[str],
) -> tuple[dict[str, object], dict[str, object]]:
    tracer_stats = {
        "unode_tracer_initialized": 0,
        "flynn_tracer_initialized": 0,
        "tracer_assignment_mode": "unavailable",
    }

    seed_unodes = mesh_state.get("_runtime_seed_unodes")
    if isinstance(seed_unodes, dict) and "U_ATTRIB_C" not in field_values:
        sample_flynn_ids, _assign_stats = _sample_host_flynn_ids(mesh_state, seed_unodes)
        if sample_flynn_ids is not None:
            field_values["U_ATTRIB_C"] = tuple(float(value) for value in sample_flynn_ids)
            if "U_ATTRIB_C" not in field_order:
                field_order.append("U_ATTRIB_C")
            tracer_stats["unode_tracer_initialized"] = 1
            tracer_stats["tracer_assignment_mode"] = "mesh_host_flynn_ids"

    flynn_sections = dict(mesh_state.get("_runtime_seed_flynn_sections", {}))
    flynn_field_order = list(flynn_sections.get("field_order", ()))
    flynn_defaults = dict(flynn_sections.get("defaults", {}))
    flynn_component_counts = dict(flynn_sections.get("component_counts", {}))
    flynn_values = dict(flynn_sections.get("values", {}))

    if "F_ATTRIB_C" not in flynn_values:
        ordered_flynn_ids: tuple[int, ...]
        existing_id_order = tuple(int(value) for value in flynn_sections.get("id_order", ()))
        if existing_id_order:
            ordered_flynn_ids = existing_id_order
        else:
            ordered_flynn_ids = tuple(
                int(flynn["flynn_id"])
                for flynn in mesh_state.get("flynns", ())
                if isinstance(flynn, dict) and "flynn_id" in flynn
            )
        if ordered_flynn_ids:
            flynn_values["F_ATTRIB_C"] = tuple((float(flynn_id),) for flynn_id in ordered_flynn_ids)
            flynn_defaults.setdefault("F_ATTRIB_C", (0.0,))
            flynn_component_counts["F_ATTRIB_C"] = 1
            if "F_ATTRIB_C" not in flynn_field_order:
                flynn_field_order.append("F_ATTRIB_C")
            flynn_sections = {
                **flynn_sections,
                "field_order": tuple(flynn_field_order),
                "id_order": tuple(int(value) for value in ordered_flynn_ids),
                "defaults": flynn_defaults,
                "component_counts": flynn_component_counts,
                "values": flynn_values,
            }
            tracer_stats["flynn_tracer_initialized"] = 1
    return flynn_sections, tracer_stats


def _legacy_mechanics_label_assignment(
    mesh_state: dict[str, object],
    field_values: dict[str, object] | None = None,
) -> tuple[str | None, np.ndarray | None, dict[str, object]]:
    seed_unodes = mesh_state.get("_runtime_seed_unodes")
    seed_fields = mesh_state.get("_runtime_seed_unode_fields")
    if not isinstance(seed_unodes, dict) or not isinstance(seed_fields, dict):
        return None, None, {
            "label_update_applied": 0,
            "label_changed_unodes": 0,
            "label_assignment_mode": "unavailable",
        }
    label_attribute = str(seed_fields.get("label_attribute", ""))
    source_labels = tuple(int(value) for value in seed_fields.get("source_labels", ()))
    if not label_attribute or not source_labels:
        return None, None, {
            "label_update_applied": 0,
            "label_changed_unodes": 0,
            "label_assignment_mode": "unavailable",
        }

    active_field_values = (
        field_values
        if isinstance(field_values, dict)
        else dict(seed_fields.get("values", {}))
    )
    sample_grid_indices = np.asarray(seed_unodes.get("grid_indices", ()), dtype=np.int32)
    sample_points = np.asarray(seed_unodes.get("positions", ()), dtype=np.float64)
    grid_shape = tuple(int(value) for value in seed_unodes.get("grid_shape", ()))
    if (
        sample_grid_indices.ndim != 2
        or sample_points.ndim != 2
        or sample_grid_indices.shape[0] == 0
        or sample_grid_indices.shape != (sample_points.shape[0], 2)
        or len(grid_shape) != 2
    ):
        return label_attribute, None, {
            "label_update_applied": 0,
            "label_changed_unodes": 0,
            "label_assignment_mode": "invalid_seed_grid",
        }

    fallback_labels = None
    raw_current = active_field_values.get(label_attribute)
    if raw_current is not None:
        current_values = np.asarray(raw_current, dtype=np.float64)
        if current_values.shape == (sample_points.shape[0],):
            source_to_compact = {int(value): int(index) for index, value in enumerate(source_labels)}
            fallback_labels = np.zeros(grid_shape, dtype=np.int32)
            for point_index, (ix, iy) in enumerate(sample_grid_indices):
                fallback_labels[int(ix), int(iy)] = int(
                    source_to_compact.get(int(round(float(current_values[int(point_index)]))), 0)
                )

    mesh_labels, assign_stats = assign_seed_unodes_from_mesh(
        mesh_state,
        seed_unodes,
        fallback_labels=fallback_labels,
    )
    sample_labels = mesh_labels[
        sample_grid_indices[:, 0],
        sample_grid_indices[:, 1],
    ]
    mapped_values = np.asarray(
        [float(source_labels[int(label)]) for label in sample_labels],
        dtype=np.float64,
    )
    changed_unodes = 0
    if raw_current is not None:
        current_values = np.asarray(raw_current, dtype=np.float64)
        if current_values.shape == mapped_values.shape:
            changed_unodes = int(np.count_nonzero(np.abs(current_values - mapped_values) > 1.0e-12))
    return label_attribute, mapped_values, {
        "label_update_applied": 1,
        "label_changed_unodes": int(changed_unodes),
        "label_assignment_mode": "mesh_host_flynn",
        "label_assigned_unodes": int(assign_stats.get("assigned_unodes", 0)),
        "label_unassigned_unodes": int(assign_stats.get("unassigned_unodes", 0)),
        "label_raster_filled_unassigned": int(assign_stats.get("raster_filled_unassigned", 0)),
    }


def _legacy_reassign_mechanics_swept_unodes(
    mesh_state: dict[str, object],
    previous_sample_labels: np.ndarray | None,
    target_sample_labels: np.ndarray | None,
    *,
    field_values: dict[str, object],
    section_values: dict[str, object],
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    seed_unodes = mesh_state.get("_runtime_seed_unodes")
    if (
        previous_sample_labels is None
        or target_sample_labels is None
        or not isinstance(seed_unodes, dict)
    ):
        return section_values, field_values, {
            "swept_unodes": 0,
            "swept_reassignment_applied": 0,
            "swept_reassignment_mode": "unavailable",
            "updated_orientation_unodes": 0,
            "fallback_orientation_unodes": 0,
            "old_value_fallback_orientation_unodes": 0,
            "density_reset_unodes": 0,
        }

    sample_points = np.asarray(seed_unodes.get("positions", ()), dtype=np.float64)
    if (
        sample_points.ndim != 2
        or sample_points.shape[0] == 0
        or previous_sample_labels.shape != target_sample_labels.shape
        or previous_sample_labels.shape[0] != sample_points.shape[0]
    ):
        return section_values, field_values, {
            "swept_unodes": 0,
            "swept_reassignment_applied": 0,
            "swept_reassignment_mode": "invalid_seed_grid",
            "updated_orientation_unodes": 0,
            "fallback_orientation_unodes": 0,
            "old_value_fallback_orientation_unodes": 0,
            "density_reset_unodes": 0,
        }

    changed_mask = np.asarray(previous_sample_labels != target_sample_labels, dtype=bool)
    swept_unodes = int(np.count_nonzero(changed_mask))
    if swept_unodes == 0:
        return section_values, field_values, {
            "swept_unodes": 0,
            "swept_reassignment_applied": 0,
            "swept_reassignment_mode": "no_host_flynn_change",
            "updated_orientation_unodes": 0,
            "fallback_orientation_unodes": 0,
            "old_value_fallback_orientation_unodes": 0,
            "density_reset_unodes": 0,
        }

    current_euler_rows = section_values.get("U_EULER_3")
    current_density_values = field_values.get("U_DISLOCDEN")
    if current_euler_rows is None or current_density_values is None:
        return section_values, field_values, {
            "swept_unodes": int(swept_unodes),
            "swept_reassignment_applied": 0,
            "swept_reassignment_mode": "missing_euler_or_density",
            "updated_orientation_unodes": 0,
            "fallback_orientation_unodes": 0,
            "old_value_fallback_orientation_unodes": 0,
            "density_reset_unodes": 0,
        }

    euler_values = np.asarray(current_euler_rows, dtype=np.float64)
    density_values = np.asarray(current_density_values, dtype=np.float64)
    if (
        euler_values.ndim != 2
        or euler_values.shape[0] != sample_points.shape[0]
        or euler_values.shape[1] < 3
        or density_values.shape != (sample_points.shape[0],)
    ):
        return section_values, field_values, {
            "swept_unodes": int(swept_unodes),
            "swept_reassignment_applied": 0,
            "swept_reassignment_mode": "invalid_euler_or_density_shape",
            "updated_orientation_unodes": 0,
            "fallback_orientation_unodes": 0,
            "old_value_fallback_orientation_unodes": 0,
            "density_reset_unodes": 0,
        }

    updated_sections = dict(section_values)
    updated_fields = dict(field_values)
    updated_eulers = np.asarray(euler_values, dtype=np.float64).copy()
    updated_density = np.asarray(density_values, dtype=np.float64).copy()
    reassign_roi = float(_legacy_fs_roi(seed_unodes, mesh_state.get("_runtime_elle_options"), factor=8))

    updated_orientation_unodes = 0
    fallback_orientation_unodes = 0
    old_value_fallback_orientation_unodes = 0

    for point_index in np.flatnonzero(changed_mask):
        target_label = int(target_sample_labels[int(point_index)])
        donor_mask = np.asarray((target_sample_labels == target_label) & (~changed_mask), dtype=bool)
        donor_value = _nearest_same_label_vector_value(
            sample_points,
            updated_eulers,
            donor_mask,
            int(point_index),
            roi=reassign_roi,
        )
        if donor_value is None:
            donor_value = _distance_weighted_vector_value(
                sample_points,
                updated_eulers,
                donor_mask,
                int(point_index),
            )
            if donor_value is None:
                donor_value = np.asarray(updated_eulers[int(point_index)], dtype=np.float64).copy()
                old_value_fallback_orientation_unodes += 1
            else:
                fallback_orientation_unodes += 1
        if np.max(np.abs(updated_eulers[int(point_index), :3] - donor_value[:3])) > 1.0e-12:
            updated_orientation_unodes += 1
        updated_eulers[int(point_index), :3] = donor_value[:3]

    density_reset_unodes = int(np.count_nonzero(changed_mask))
    updated_density[changed_mask] = 0.0

    updated_sections["U_EULER_3"] = tuple(
        tuple(float(component) for component in row)
        for row in updated_eulers
    )
    updated_fields["U_DISLOCDEN"] = tuple(float(value) for value in updated_density)
    return updated_sections, updated_fields, {
        "swept_unodes": int(swept_unodes),
        "swept_reassignment_applied": 1,
        "swept_reassignment_mode": "fs_check_unodes",
        "updated_orientation_unodes": int(updated_orientation_unodes),
        "fallback_orientation_unodes": int(fallback_orientation_unodes),
        "old_value_fallback_orientation_unodes": int(old_value_fallback_orientation_unodes),
        "density_reset_unodes": int(density_reset_unodes),
    }


def _legacy_check_error_mechanics_unodes(
    mesh_state: dict[str, object],
    target_host_flynn_ids: np.ndarray | None,
    *,
    field_values: dict[str, object],
    section_values: dict[str, object],
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    seed_unodes = mesh_state.get("_runtime_seed_unodes")
    if target_host_flynn_ids is None or not isinstance(seed_unodes, dict):
        return section_values, field_values, {
            "swept_unodes": 0,
            "swept_reassignment_applied": 0,
            "swept_reassignment_mode": "unavailable",
            "updated_orientation_unodes": 0,
            "fallback_orientation_unodes": 0,
            "old_value_fallback_orientation_unodes": 0,
            "density_reset_unodes": 0,
        }

    seed_fields = mesh_state.get("_runtime_seed_unode_fields")
    if isinstance(seed_fields, dict) and str(seed_fields.get("label_attribute", "")) == "U_ATTRIB_C":
        return section_values, field_values, {
            "swept_unodes": 0,
            "swept_reassignment_applied": 0,
            "swept_reassignment_mode": "label_attribute_conflict",
            "updated_orientation_unodes": 0,
            "fallback_orientation_unodes": 0,
            "old_value_fallback_orientation_unodes": 0,
            "density_reset_unodes": 0,
        }

    sample_points = np.asarray(seed_unodes.get("positions", ()), dtype=np.float64)
    tracer_values = field_values.get("U_ATTRIB_C")
    current_euler_rows = section_values.get("U_EULER_3")
    if tracer_values is None or current_euler_rows is None:
        return section_values, field_values, {
            "swept_unodes": 0,
            "swept_reassignment_applied": 0,
            "swept_reassignment_mode": "missing_tracer_or_euler",
            "updated_orientation_unodes": 0,
            "fallback_orientation_unodes": 0,
            "old_value_fallback_orientation_unodes": 0,
            "density_reset_unodes": 0,
        }

    tracer_array = np.asarray(tracer_values, dtype=np.float64)
    euler_values = np.asarray(current_euler_rows, dtype=np.float64)
    host_ids = np.asarray(target_host_flynn_ids, dtype=np.int32)
    if (
        sample_points.ndim != 2
        or sample_points.shape[0] == 0
        or tracer_array.shape != (sample_points.shape[0],)
        or host_ids.shape != (sample_points.shape[0],)
        or euler_values.ndim != 2
        or euler_values.shape[0] != sample_points.shape[0]
        or euler_values.shape[1] < 3
    ):
        return section_values, field_values, {
            "swept_unodes": 0,
            "swept_reassignment_applied": 0,
            "swept_reassignment_mode": "invalid_tracer_or_euler_shape",
            "updated_orientation_unodes": 0,
            "fallback_orientation_unodes": 0,
            "old_value_fallback_orientation_unodes": 0,
            "density_reset_unodes": 0,
        }

    mismatched_mask = np.asarray(
        np.rint(tracer_array).astype(np.int32) != host_ids,
        dtype=bool,
    )
    swept_unodes = int(np.count_nonzero(mismatched_mask))
    if swept_unodes == 0:
        return section_values, field_values, {
            "swept_unodes": 0,
            "swept_reassignment_applied": 0,
            "swept_reassignment_mode": "no_tracer_mismatch",
            "updated_orientation_unodes": 0,
            "fallback_orientation_unodes": 0,
            "old_value_fallback_orientation_unodes": 0,
            "density_reset_unodes": 0,
        }

    updated_sections = dict(section_values)
    updated_fields = dict(field_values)
    updated_eulers = np.asarray(euler_values, dtype=np.float64).copy()
    updated_tracers = np.asarray(tracer_array, dtype=np.float64).copy()

    updated_orientation_unodes = 0
    old_value_fallback_orientation_unodes = 0
    for point_index in np.flatnonzero(mismatched_mask):
        host_id = int(host_ids[int(point_index)])
        donor_mask = np.asarray(
            (host_ids == host_id)
            & (np.rint(updated_tracers).astype(np.int32) == host_id),
            dtype=bool,
        )
        donor_mask[int(point_index)] = False
        donor_value = _nearest_same_label_vector_value(
            sample_points,
            updated_eulers,
            donor_mask,
            int(point_index),
        )
        if donor_value is None:
            old_value_fallback_orientation_unodes += 1
            continue
        if np.max(np.abs(updated_eulers[int(point_index), :3] - donor_value[:3])) > 1.0e-12:
            updated_orientation_unodes += 1
        updated_eulers[int(point_index), :3] = donor_value[:3]
        updated_tracers[int(point_index)] = float(host_id)

    updated_sections["U_EULER_3"] = tuple(
        tuple(float(component) for component in row)
        for row in updated_eulers
    )
    updated_fields["U_ATTRIB_C"] = tuple(float(value) for value in updated_tracers)
    return updated_sections, updated_fields, {
        "swept_unodes": int(swept_unodes),
        "swept_reassignment_applied": 1,
        "swept_reassignment_mode": "check_error",
        "updated_orientation_unodes": int(updated_orientation_unodes),
        "fallback_orientation_unodes": 0,
        "old_value_fallback_orientation_unodes": int(old_value_fallback_orientation_unodes),
        "density_reset_unodes": 0,
    }


def _legacy_repair_mechanics_unodes(
    mesh_state: dict[str, object],
    previous_sample_labels: np.ndarray | None,
    target_sample_labels: np.ndarray | None,
    target_host_flynn_ids: np.ndarray | None,
    *,
    field_values: dict[str, object],
    section_values: dict[str, object],
    host_repair_mode: str,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    if str(host_repair_mode) == "check_error":
        return _legacy_check_error_mechanics_unodes(
            mesh_state,
            target_host_flynn_ids,
            field_values=field_values,
            section_values=section_values,
        )
    return _legacy_reassign_mechanics_swept_unodes(
        mesh_state,
        previous_sample_labels,
        target_sample_labels,
        field_values=field_values,
        section_values=section_values,
    )


def _legacy_expected_density_after_import(
    mesh_state: dict[str, object],
    field_values: dict[str, object],
    payload: LegacyFFTBridgePayload,
    import_options: LegacyFFTImportOptions,
    *,
    seed_count: int,
) -> tuple[np.ndarray, int, int]:
    previous_density = np.asarray(
        field_values.get("U_DISLOCDEN", np.zeros((int(seed_count),), dtype=np.float64)),
        dtype=np.float64,
    )
    updated_density = np.asarray(previous_density, dtype=np.float64).copy()
    density_imported_unodes = 0
    density_excluded_unodes = 0
    if (
        bool(import_options.import_dislocation_densities)
        and payload.geometrical_dislocation_density_increment is not None
    ):
        density_update_mode = _legacy_density_update_mode(import_options.density_update_mode)
        incoming_density = np.asarray(
            payload.geometrical_dislocation_density_increment,
            dtype=np.float64,
        )
        if density_update_mode == "overwrite":
            updated_density = np.asarray(incoming_density, dtype=np.float64)
        else:
            updated_density = previous_density + incoming_density
        exclude_phase_id = int(import_options.exclude_phase_id)
        if exclude_phase_id != 0:
            phase_ids = _seed_unode_phase_ids(mesh_state)
            if phase_ids is not None and phase_ids.shape == (int(seed_count),):
                excluded_mask = np.asarray(phase_ids == exclude_phase_id, dtype=bool)
                density_excluded_unodes = int(np.count_nonzero(excluded_mask))
                if density_excluded_unodes:
                    updated_density = np.asarray(updated_density, dtype=np.float64)
                    updated_density[excluded_mask] = 0.0
        density_imported_unodes = int(seed_count) - int(density_excluded_unodes)
    return np.asarray(updated_density, dtype=np.float64), int(density_imported_unodes), int(density_excluded_unodes)


def apply_legacy_fft_snapshot_to_mesh_state(
    mesh_state: dict[str, object],
    snapshot: FFTMechanicsSnapshot,
    *,
    import_options: LegacyFFTImportOptions | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    """Apply one frozen legacy mechanics snapshot to the faithful runtime state."""

    seed_unodes = mesh_state.get("_runtime_seed_unodes")
    if not isinstance(seed_unodes, dict):
        raise ValueError("mesh_state has no faithful seed-unode payload")
    seed_ids = np.asarray(seed_unodes.get("ids", ()), dtype=np.int32)
    if seed_ids.ndim != 1 or seed_ids.size == 0:
        raise ValueError("mesh_state has no faithful seed unode ids")
    previous_sample_labels, _previous_label_assign_stats = _sample_host_flynn_labels(
        mesh_state,
        seed_unodes,
    )

    payload = build_legacy_fft_bridge_payload(seed_ids, snapshot)
    if import_options is None:
        import_options = LegacyFFTImportOptions()
    host_repair_mode = _legacy_host_repair_mode(import_options.host_repair_mode)
    aligned_euler = np.asarray(payload.unode_euler_deg, dtype=np.float64)
    updated_seed_unodes, position_update_stats = _apply_legacy_unode_position_update(
        seed_unodes,
        payload,
    )
    mesh_state["_runtime_seed_unodes"] = updated_seed_unodes
    mesh_state, node_position_update_stats = _apply_legacy_node_position_update(
        mesh_state,
        seed_unodes,
        updated_seed_unodes,
        previous_sample_labels,
    )
    reference_box_height = float(
        mesh_state.get("_mechanics_reference_box_height")
        if "_mechanics_reference_box_height" in mesh_state
        else _legacy_box_height_from_runtime_options(mesh_state.get("_runtime_elle_options"))
    )
    if abs(reference_box_height) <= 1.0e-12:
        reference_box_height = 1.0
    mesh_state["_mechanics_reference_box_height"] = float(reference_box_height)
    updated_elle_options, cell_reset_stats = _apply_legacy_cell_reset(
        mesh_state.get("_runtime_elle_options"),
        payload,
    )
    mesh_state["_runtime_elle_options"] = updated_elle_options
    direct_strain_summary = _legacy_direct_strain_axis_summary(
        reference_box_height,
        updated_elle_options,
        cell_reset_stats,
    )
    target_sample_labels, _target_label_assign_stats = _sample_host_flynn_labels(
        mesh_state,
        updated_seed_unodes,
    )
    target_host_flynn_ids, _target_host_assign_stats = _sample_host_flynn_ids(
        mesh_state,
        updated_seed_unodes,
    )

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
    updated_flynn_sections, tracer_stats = _legacy_initialize_mechanics_tracers(
        mesh_state,
        field_values=field_values,
        field_order=field_order,
    )
    if updated_flynn_sections:
        mesh_state["_runtime_seed_flynn_sections"] = updated_flynn_sections
    density_field_preexists = "U_DISLOCDEN" in field_values

    def _assign_scalar_field(name: str, values: np.ndarray) -> None:
        field_values[str(name)] = tuple(float(value) for value in np.asarray(values, dtype=np.float64))
        if str(name) not in field_order:
            field_order.append(str(name))

    exclude_phase_id = int(import_options.exclude_phase_id)
    density_update_mode = _legacy_density_update_mode(import_options.density_update_mode)
    if payload.normalized_strain_rate is not None:
        _assign_scalar_field("U_ATTRIB_A", payload.normalized_strain_rate)
        _assign_scalar_field("U_ATTRIB_B", payload.normalized_stress)
        _assign_scalar_field("U_ATTRIB_D", payload.basal_activity)
        _assign_scalar_field("U_ATTRIB_E", payload.prismatic_activity)
    updated_density, density_imported_unodes, density_excluded_unodes = (
        _legacy_expected_density_after_import(
            mesh_state,
            field_values,
            payload,
            import_options,
            seed_count=int(seed_ids.size),
        )
    )
    if density_field_preexists or (
        bool(import_options.import_dislocation_densities)
        and payload.geometrical_dislocation_density_increment is not None
    ):
        _assign_scalar_field("U_DISLOCDEN", updated_density)
    section_values, field_values, swept_reassignment_stats = _legacy_repair_mechanics_unodes(
        mesh_state,
        previous_sample_labels,
        target_sample_labels,
        target_host_flynn_ids,
        field_values=field_values,
        section_values=section_values,
        host_repair_mode=str(host_repair_mode),
    )
    label_attribute, remapped_label_values, label_update_stats = _legacy_mechanics_label_assignment(
        mesh_state,
        field_values=field_values,
    )
    if label_attribute is not None and remapped_label_values is not None:
        _assign_scalar_field(label_attribute, remapped_label_values)

    mesh_state["_runtime_seed_unode_sections"] = {
        **seed_sections,
        "values": section_values,
        "defaults": section_defaults,
        "component_counts": section_components,
        "field_order": tuple(section_field_order),
    }

    mesh_state["_runtime_seed_unode_fields"] = {
        **seed_fields,
        "values": field_values,
        "field_order": tuple(field_order),
    }
    mesh_state["_runtime_mechanics_snapshot"] = {
        "paths": {
            "temp_matrix_path": str(payload.temp_matrix_path),
            "unode_strain_path": str(payload.unode_strain_path),
            "unode_euler_path": str(payload.unode_euler_path),
            "tex_path": payload.tex_path,
        },
        "alignment_mode": str(payload.alignment_mode),
        "euler_alignment_mode": str(payload.euler_alignment_mode),
        "temp_matrix": np.asarray(payload.temp_matrix, dtype=np.float64),
        "cell_lengths": np.asarray(payload.cell_lengths, dtype=np.float64),
        "cell_strain_triplet": np.asarray(payload.cell_strain_triplet, dtype=np.float64),
        "cell_shear_triplet": np.asarray(payload.cell_shear_triplet, dtype=np.float64),
        "unode_strain_xyz": np.asarray(payload.unode_strain_xyz, dtype=np.float64),
        "normalized_strain_rate": None if payload.normalized_strain_rate is None else np.asarray(payload.normalized_strain_rate, dtype=np.float64),
        "normalized_stress": None if payload.normalized_stress is None else np.asarray(payload.normalized_stress, dtype=np.float64),
        "basal_activity": None if payload.basal_activity is None else np.asarray(payload.basal_activity, dtype=np.float64),
        "prismatic_activity": None if payload.prismatic_activity is None else np.asarray(payload.prismatic_activity, dtype=np.float64),
        "geometrical_dislocation_density_increment": None if payload.geometrical_dislocation_density_increment is None else np.asarray(payload.geometrical_dislocation_density_increment, dtype=np.float64),
        "statistical_dislocation_density": None if payload.statistical_dislocation_density is None else np.asarray(payload.statistical_dislocation_density, dtype=np.float64),
        "fourier_point_ids": None if payload.fourier_point_ids is None else np.asarray(payload.fourier_point_ids, dtype=np.int32),
        "fft_grain_numbers": None if payload.fft_grain_numbers is None else np.asarray(payload.fft_grain_numbers, dtype=np.int32),
        "import_dislocation_densities": bool(import_options.import_dislocation_densities),
        "exclude_phase_id": int(exclude_phase_id),
        "density_update_mode": str(density_update_mode),
        "host_repair_mode": str(host_repair_mode),
        "label_attribute": None if label_attribute is None else str(label_attribute),
        "label_values": None if remapped_label_values is None else np.asarray(remapped_label_values, dtype=np.float64),
        "unode_tracer_initialized": int(tracer_stats["unode_tracer_initialized"]),
        "flynn_tracer_initialized": int(tracer_stats["flynn_tracer_initialized"]),
        "tracer_assignment_mode": str(tracer_stats["tracer_assignment_mode"]),
        "swept_unodes": int(swept_reassignment_stats["swept_unodes"]),
        "swept_reassignment_applied": int(swept_reassignment_stats["swept_reassignment_applied"]),
        "swept_reassignment_mode": str(swept_reassignment_stats["swept_reassignment_mode"]),
        "updated_orientation_unodes": int(swept_reassignment_stats["updated_orientation_unodes"]),
        "fallback_orientation_unodes": int(swept_reassignment_stats["fallback_orientation_unodes"]),
        "old_value_fallback_orientation_unodes": int(
            swept_reassignment_stats["old_value_fallback_orientation_unodes"]
        ),
        "density_reset_unodes": int(swept_reassignment_stats["density_reset_unodes"]),
        "unode_target_positions_xy": np.asarray(
            mesh_state["_runtime_seed_unodes"]["positions"],
            dtype=np.float64,
        ),
        "position_update_mode": str(position_update_stats["position_update_mode"]),
        "node_position_update_mode": str(node_position_update_stats["node_position_update_mode"]),
        "cell_reset_runtime_options": updated_elle_options,
        "direct_strain_axis": float(direct_strain_summary["direct_strain_axis"]),
        "strain_axis_source": str(direct_strain_summary["strain_axis_source"]),
    }
    mean_normalized_strain_rate = (
        None
        if payload.normalized_strain_rate is None
        else float(np.mean(np.asarray(payload.normalized_strain_rate, dtype=np.float64), dtype=np.float64))
    )
    mean_normalized_stress = (
        None
        if payload.normalized_stress is None
        else float(np.mean(np.asarray(payload.normalized_stress, dtype=np.float64), dtype=np.float64))
    )
    mean_basal_activity = (
        None
        if payload.basal_activity is None
        else float(np.mean(np.asarray(payload.basal_activity, dtype=np.float64), dtype=np.float64))
    )
    mean_prismatic_activity = (
        None
        if payload.prismatic_activity is None
        else float(np.mean(np.asarray(payload.prismatic_activity, dtype=np.float64), dtype=np.float64))
    )
    mean_total_activity = None
    mean_prismatic_fraction = None
    prismatic_to_basal_ratio = None
    if mean_basal_activity is not None and mean_prismatic_activity is not None:
        mean_total_activity = float(mean_basal_activity + mean_prismatic_activity)
        if abs(mean_total_activity) > 1.0e-12:
            mean_prismatic_fraction = float(mean_prismatic_activity / mean_total_activity)
        if abs(mean_basal_activity) > 1.0e-12:
            prismatic_to_basal_ratio = float(mean_prismatic_activity / mean_basal_activity)
    mesh_state["mechanics_payload_summary"] = {
        "source": "legacy_fft_snapshot",
        "has_tex": int(payload.normalized_strain_rate is not None),
        "mean_normalized_strain_rate": mean_normalized_strain_rate,
        "mean_normalized_stress": mean_normalized_stress,
        "mean_basal_activity": mean_basal_activity,
        "mean_prismatic_activity": mean_prismatic_activity,
        "mean_total_activity": mean_total_activity,
        "mean_prismatic_fraction": mean_prismatic_fraction,
        "prismatic_to_basal_ratio": prismatic_to_basal_ratio,
        "cell_lengths": np.asarray(payload.cell_lengths, dtype=np.float64).tolist(),
        "cell_strain_triplet": np.asarray(payload.cell_strain_triplet, dtype=np.float64).tolist(),
        "cell_shear_triplet": np.asarray(payload.cell_shear_triplet, dtype=np.float64).tolist(),
        "simple_shear_increment": float(cell_reset_stats["simple_shear_increment"]),
        "simple_shear_offset": float(cell_reset_stats["simple_shear_offset"]),
        "cumulative_simple_shear": float(cell_reset_stats["cumulative_simple_shear"]),
        "direct_strain_axis": float(direct_strain_summary["direct_strain_axis"]),
        "strain_axis_source": str(direct_strain_summary["strain_axis_source"]),
    }
    mesh_state.setdefault("stats", {})
    mesh_state["stats"]["mechanics_snapshot_applied"] = 1
    mesh_state["stats"]["mechanics_snapshot_alignment_mode"] = str(payload.alignment_mode)
    mesh_state["stats"]["mechanics_snapshot_euler_alignment_mode"] = str(payload.euler_alignment_mode)
    mesh_state["stats"]["mechanics_snapshot_updated_unodes"] = int(seed_ids.size)
    mesh_state["stats"]["mechanics_snapshot_has_tex"] = int(payload.normalized_strain_rate is not None)
    mesh_state["stats"]["mechanics_snapshot_has_cell_reset_payload"] = 1
    mesh_state["stats"]["mechanics_snapshot_updated_unode_positions"] = int(
        position_update_stats["updated_unode_positions"]
    )
    mesh_state["stats"]["mechanics_snapshot_position_update_mode"] = str(
        position_update_stats["position_update_mode"]
    )
    mesh_state["stats"]["mechanics_snapshot_updated_node_positions"] = int(
        node_position_update_stats["updated_node_positions"]
    )
    mesh_state["stats"]["mechanics_snapshot_node_position_update_mode"] = str(
        node_position_update_stats["node_position_update_mode"]
    )
    mesh_state["stats"]["mechanics_snapshot_cell_reset_applied"] = int(
        cell_reset_stats["cell_reset_applied"]
    )
    mesh_state["stats"]["mechanics_snapshot_simple_shear_increment"] = float(
        cell_reset_stats["simple_shear_increment"]
    )
    mesh_state["stats"]["mechanics_snapshot_simple_shear_offset"] = float(
        cell_reset_stats["simple_shear_offset"]
    )
    mesh_state["stats"]["mechanics_snapshot_cumulative_simple_shear"] = float(
        cell_reset_stats["cumulative_simple_shear"]
    )
    mesh_state["stats"]["mechanics_snapshot_direct_strain_axis"] = float(
        direct_strain_summary["direct_strain_axis"]
    )
    mesh_state["stats"]["mechanics_snapshot_strain_axis_source"] = str(
        direct_strain_summary["strain_axis_source"]
    )
    mesh_state["stats"]["mechanics_snapshot_import_dislocation_densities"] = int(
        bool(import_options.import_dislocation_densities)
    )
    mesh_state["stats"]["mechanics_snapshot_exclude_phase_id"] = int(exclude_phase_id)
    mesh_state["stats"]["mechanics_snapshot_density_update_mode"] = str(density_update_mode)
    mesh_state["stats"]["mechanics_snapshot_host_repair_mode"] = str(host_repair_mode)
    mesh_state["stats"]["mechanics_snapshot_density_imported_unodes"] = int(
        density_imported_unodes
    )
    mesh_state["stats"]["mechanics_snapshot_density_excluded_unodes"] = int(
        density_excluded_unodes
    )
    mesh_state["stats"]["mechanics_snapshot_unode_tracer_initialized"] = int(
        tracer_stats["unode_tracer_initialized"]
    )
    mesh_state["stats"]["mechanics_snapshot_flynn_tracer_initialized"] = int(
        tracer_stats["flynn_tracer_initialized"]
    )
    mesh_state["stats"]["mechanics_snapshot_tracer_assignment_mode"] = str(
        tracer_stats["tracer_assignment_mode"]
    )
    mesh_state["stats"]["mechanics_snapshot_swept_unodes"] = int(
        swept_reassignment_stats["swept_unodes"]
    )
    mesh_state["stats"]["mechanics_snapshot_swept_reassignment_applied"] = int(
        swept_reassignment_stats["swept_reassignment_applied"]
    )
    mesh_state["stats"]["mechanics_snapshot_swept_reassignment_mode"] = str(
        swept_reassignment_stats["swept_reassignment_mode"]
    )
    mesh_state["stats"]["mechanics_snapshot_updated_orientation_unodes"] = int(
        swept_reassignment_stats["updated_orientation_unodes"]
    )
    mesh_state["stats"]["mechanics_snapshot_fallback_orientation_unodes"] = int(
        swept_reassignment_stats["fallback_orientation_unodes"]
    )
    mesh_state["stats"]["mechanics_snapshot_old_value_fallback_orientation_unodes"] = int(
        swept_reassignment_stats["old_value_fallback_orientation_unodes"]
    )
    mesh_state["stats"]["mechanics_snapshot_density_reset_unodes"] = int(
        swept_reassignment_stats["density_reset_unodes"]
    )
    mesh_state["stats"]["mechanics_snapshot_label_update_applied"] = int(
        label_update_stats["label_update_applied"]
    )
    mesh_state["stats"]["mechanics_snapshot_label_changed_unodes"] = int(
        label_update_stats["label_changed_unodes"]
    )
    mesh_state["stats"]["mechanics_snapshot_label_assignment_mode"] = str(
        label_update_stats["label_assignment_mode"]
    )
    if "label_assigned_unodes" in label_update_stats:
        mesh_state["stats"]["mechanics_snapshot_label_assigned_unodes"] = int(
            label_update_stats["label_assigned_unodes"]
        )
        mesh_state["stats"]["mechanics_snapshot_label_unassigned_unodes"] = int(
            label_update_stats["label_unassigned_unodes"]
        )
        mesh_state["stats"]["mechanics_snapshot_label_raster_filled_unassigned"] = int(
            label_update_stats["label_raster_filled_unassigned"]
        )
    return mesh_state, {
        "mechanics_applied": 1,
        "alignment_mode": str(payload.alignment_mode),
        "euler_alignment_mode": str(payload.euler_alignment_mode),
        "updated_unodes": int(seed_ids.size),
        "has_tex": int(payload.normalized_strain_rate is not None),
        "has_cell_reset_payload": 1,
        "updated_unode_positions": int(position_update_stats["updated_unode_positions"]),
        "position_update_mode": str(position_update_stats["position_update_mode"]),
        "updated_node_positions": int(node_position_update_stats["updated_node_positions"]),
        "node_position_update_mode": str(node_position_update_stats["node_position_update_mode"]),
        "cell_reset_applied": int(cell_reset_stats["cell_reset_applied"]),
        "simple_shear_increment": float(cell_reset_stats["simple_shear_increment"]),
        "simple_shear_offset": float(cell_reset_stats["simple_shear_offset"]),
        "cumulative_simple_shear": float(cell_reset_stats["cumulative_simple_shear"]),
        "import_dislocation_densities": int(bool(import_options.import_dislocation_densities)),
        "exclude_phase_id": int(exclude_phase_id),
        "density_update_mode": str(density_update_mode),
        "host_repair_mode": str(host_repair_mode),
        "density_imported_unodes": int(density_imported_unodes),
        "density_excluded_unodes": int(density_excluded_unodes),
        "unode_tracer_initialized": int(tracer_stats["unode_tracer_initialized"]),
        "flynn_tracer_initialized": int(tracer_stats["flynn_tracer_initialized"]),
        "tracer_assignment_mode": str(tracer_stats["tracer_assignment_mode"]),
        "swept_unodes": int(swept_reassignment_stats["swept_unodes"]),
        "swept_reassignment_applied": int(swept_reassignment_stats["swept_reassignment_applied"]),
        "swept_reassignment_mode": str(swept_reassignment_stats["swept_reassignment_mode"]),
        "updated_orientation_unodes": int(swept_reassignment_stats["updated_orientation_unodes"]),
        "fallback_orientation_unodes": int(swept_reassignment_stats["fallback_orientation_unodes"]),
        "old_value_fallback_orientation_unodes": int(
            swept_reassignment_stats["old_value_fallback_orientation_unodes"]
        ),
        "density_reset_unodes": int(swept_reassignment_stats["density_reset_unodes"]),
        "label_update_applied": int(label_update_stats["label_update_applied"]),
        "label_changed_unodes": int(label_update_stats["label_changed_unodes"]),
        "label_assignment_mode": str(label_update_stats["label_assignment_mode"]),
    }


def _runtime_unode_scalar_field_array(
    mesh_state: dict[str, object],
    name: str,
) -> np.ndarray | None:
    seed_fields = mesh_state.get("_runtime_seed_unode_fields")
    if not isinstance(seed_fields, dict):
        return None
    raw_values = dict(seed_fields.get("values", {})).get(str(name))
    if raw_values is None:
        return None
    values = np.asarray(raw_values, dtype=np.float64)
    if values.ndim != 1:
        return None
    return values


def _runtime_flynn_scalar_section_array(
    mesh_state: dict[str, object],
    name: str,
) -> np.ndarray | None:
    flynn_sections = mesh_state.get("_runtime_current_flynn_sections")
    if not isinstance(flynn_sections, dict):
        flynn_sections = mesh_state.get("_runtime_seed_flynn_sections")
    if not isinstance(flynn_sections, dict):
        return None
    raw_values = dict(flynn_sections.get("values", {})).get(str(name))
    if raw_values is None:
        return None
    values = np.asarray(raw_values, dtype=np.float64)
    if values.ndim == 2 and values.shape[1] >= 1:
        return np.asarray(values[:, 0], dtype=np.float64)
    if values.ndim == 1:
        return np.asarray(values, dtype=np.float64)
    return None


def _runtime_unode_section_array(
    mesh_state: dict[str, object],
    name: str,
) -> np.ndarray | None:
    seed_sections = mesh_state.get("_runtime_seed_unode_sections")
    if not isinstance(seed_sections, dict):
        return None
    raw_values = dict(seed_sections.get("values", {})).get(str(name))
    if raw_values is None:
        return None
    values = np.asarray(raw_values, dtype=np.float64)
    if values.ndim != 2:
        return None
    return values


def _runtime_positions_array(mesh_state: dict[str, object]) -> np.ndarray | None:
    seed_unodes = mesh_state.get("_runtime_seed_unodes")
    if not isinstance(seed_unodes, dict):
        return None
    positions = np.asarray(seed_unodes.get("positions", ()), dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 2:
        return None
    return positions


def _runtime_node_positions_array(mesh_state: dict[str, object]) -> np.ndarray | None:
    raw_nodes = mesh_state.get("nodes")
    if not isinstance(raw_nodes, list) or not raw_nodes:
        return None
    positions = np.asarray(
        [
            (float(node["x"]), float(node["y"]))
            for node in raw_nodes
            if isinstance(node, dict) and "x" in node and "y" in node
        ],
        dtype=np.float64,
    )
    if positions.ndim != 2 or positions.shape[1] != 2:
        return None
    return positions


def _runtime_mechanics_snapshot_array(
    mesh_state: dict[str, object],
    name: str,
) -> np.ndarray | None:
    runtime_snapshot = mesh_state.get("_runtime_mechanics_snapshot")
    if not isinstance(runtime_snapshot, dict):
        return None
    raw_values = runtime_snapshot.get(str(name))
    if raw_values is None:
        return None
    return np.asarray(raw_values)


def _runtime_elle_options_summary(mesh_state: dict[str, object]) -> tuple[np.ndarray, float, float]:
    options = FaithfulElleOptions.from_runtime_dict(
        mesh_state.get("_runtime_elle_options") if isinstance(mesh_state, dict) else None
    )
    bounding_box = np.asarray(options.cell_bounding_box, dtype=np.float64)
    return (
        bounding_box,
        float(options.simple_shear_offset),
        float(options.cumulative_simple_shear),
    )


def _float_rmse(candidate: np.ndarray | None, reference: np.ndarray | None) -> float:
    if candidate is None or reference is None:
        return float("inf")
    candidate_array = np.asarray(candidate, dtype=np.float64)
    reference_array = np.asarray(reference, dtype=np.float64)
    if candidate_array.shape != reference_array.shape:
        return float("inf")
    if candidate_array.size == 0:
        return 0.0
    return float(
        np.sqrt(
            np.mean(
                np.square(candidate_array - reference_array, dtype=np.float64),
                dtype=np.float64,
            )
        )
    )


def _integer_mismatch_count(candidate: np.ndarray | None, reference: np.ndarray | None) -> int:
    if candidate is None or reference is None:
        return -1
    candidate_array = np.asarray(candidate)
    reference_array = np.asarray(reference)
    if candidate_array.shape != reference_array.shape:
        return -1
    return int(
        np.count_nonzero(
            np.asarray(candidate_array, dtype=np.int64) != np.asarray(reference_array, dtype=np.int64)
        )
    )


def compare_applied_legacy_fft_snapshot_to_mesh_state(
    before_mesh_state: dict[str, object],
    after_mesh_state: dict[str, object],
    snapshot: FFTMechanicsSnapshot,
    *,
    import_options: LegacyFFTImportOptions | None = None,
) -> dict[str, object]:
    """Compare faithful runtime state after mechanics import against the frozen bridge snapshot."""

    seed_unodes = before_mesh_state.get("_runtime_seed_unodes")
    if not isinstance(seed_unodes, dict):
        raise ValueError("before_mesh_state has no faithful seed-unode payload")
    seed_ids = np.asarray(seed_unodes.get("ids", ()), dtype=np.int32)
    if seed_ids.ndim != 1 or seed_ids.size == 0:
        raise ValueError("before_mesh_state has no faithful seed unode ids")
    if import_options is None:
        import_options = LegacyFFTImportOptions()
    density_update_mode = _legacy_density_update_mode(import_options.density_update_mode)
    host_repair_mode = _legacy_host_repair_mode(import_options.host_repair_mode)
    previous_sample_labels, _previous_label_assign_stats = _sample_host_flynn_labels(
        before_mesh_state,
        seed_unodes,
    )

    payload = build_legacy_fft_bridge_payload(seed_ids, snapshot)
    expected_seed_unodes, position_update_stats = _apply_legacy_unode_position_update(
        dict(seed_unodes),
        payload,
    )
    position_phase_mesh_state = dict(before_mesh_state)
    position_phase_mesh_state["_runtime_seed_unodes"] = expected_seed_unodes
    position_phase_mesh_state, expected_node_position_stats = _apply_legacy_node_position_update(
        position_phase_mesh_state,
        seed_unodes,
        expected_seed_unodes,
        previous_sample_labels,
    )
    expected_elle_options, cell_reset_stats = _apply_legacy_cell_reset(
        before_mesh_state.get("_runtime_elle_options"),
        payload,
    )
    position_phase_mesh_state["_runtime_elle_options"] = expected_elle_options
    expected_target_sample_labels, _expected_target_label_assign_stats = _sample_host_flynn_labels(
        position_phase_mesh_state,
        expected_seed_unodes,
    )
    expected_target_host_flynn_ids, _expected_target_host_assign_stats = _sample_host_flynn_ids(
        position_phase_mesh_state,
        expected_seed_unodes,
    )
    before_seed_fields = before_mesh_state.get("_runtime_seed_unode_fields")
    before_field_values = dict(before_seed_fields.get("values", {})) if isinstance(before_seed_fields, dict) else {}
    before_field_order = list(before_seed_fields.get("field_order", ())) if isinstance(before_seed_fields, dict) else []
    _expected_flynn_sections, expected_tracer_stats = _legacy_initialize_mechanics_tracers(
        position_phase_mesh_state,
        field_values=before_field_values,
        field_order=before_field_order,
    )
    expected_unode_tracer_values = (
        np.asarray(before_field_values.get("U_ATTRIB_C"), dtype=np.float64)
        if int(expected_tracer_stats["unode_tracer_initialized"]) == 1 and "U_ATTRIB_C" in before_field_values
        else None
    )
    expected_flynn_tracer_values = None
    if isinstance(_expected_flynn_sections, dict):
        raw_expected_flynn_tracer = dict(_expected_flynn_sections.get("values", {})).get("F_ATTRIB_C")
        if raw_expected_flynn_tracer is not None:
            expected_flynn_tracer_values = np.asarray(raw_expected_flynn_tracer, dtype=np.float64)
            if expected_flynn_tracer_values.ndim == 2 and expected_flynn_tracer_values.shape[1] >= 1:
                expected_flynn_tracer_values = np.asarray(
                    expected_flynn_tracer_values[:, 0],
                    dtype=np.float64,
                )
    before_seed_sections = before_mesh_state.get("_runtime_seed_unode_sections")
    expected_section_values = (
        dict(before_seed_sections.get("values", {}))
        if isinstance(before_seed_sections, dict)
        else {}
    )
    expected_section_values["U_EULER_3"] = tuple(
        tuple(float(value) for value in row)
        for row in np.asarray(payload.unode_euler_deg, dtype=np.float64)
    )
    density_field_expected = bool(
        "U_DISLOCDEN" in before_field_values
        or (
            bool(import_options.import_dislocation_densities)
            and payload.geometrical_dislocation_density_increment is not None
        )
    )
    expected_density, density_imported_unodes, density_excluded_unodes = (
        _legacy_expected_density_after_import(
            position_phase_mesh_state,
            before_field_values,
            payload,
            import_options,
            seed_count=int(seed_ids.size),
        )
    )
    if density_field_expected:
        before_field_values["U_DISLOCDEN"] = tuple(float(value) for value in expected_density)
    expected_section_values, before_field_values, expected_swept_stats = (
        _legacy_repair_mechanics_unodes(
            position_phase_mesh_state,
            previous_sample_labels,
            expected_target_sample_labels,
            expected_target_host_flynn_ids,
            field_values=before_field_values,
            section_values=expected_section_values,
            host_repair_mode=str(host_repair_mode),
        )
    )
    expected_label_attribute, expected_label_values, expected_label_stats = _legacy_mechanics_label_assignment(
        position_phase_mesh_state,
        field_values=before_field_values,
    )
    expected_final_euler = np.asarray(
        expected_section_values.get("U_EULER_3", ()),
        dtype=np.float64,
    )
    expected_final_density = (
        np.asarray(before_field_values.get("U_DISLOCDEN"), dtype=np.float64)
        if density_field_expected and "U_DISLOCDEN" in before_field_values
        else None
    )

    actual_euler = _runtime_unode_section_array(after_mesh_state, "U_EULER_3")
    actual_positions = _runtime_positions_array(after_mesh_state)
    actual_node_positions = _runtime_node_positions_array(after_mesh_state)
    expected_node_positions = _runtime_node_positions_array(position_phase_mesh_state)
    actual_attr_a = _runtime_unode_scalar_field_array(after_mesh_state, "U_ATTRIB_A")
    actual_attr_b = _runtime_unode_scalar_field_array(after_mesh_state, "U_ATTRIB_B")
    actual_attr_d = _runtime_unode_scalar_field_array(after_mesh_state, "U_ATTRIB_D")
    actual_attr_e = _runtime_unode_scalar_field_array(after_mesh_state, "U_ATTRIB_E")
    actual_density = _runtime_unode_scalar_field_array(after_mesh_state, "U_DISLOCDEN")
    actual_unode_tracer_values = _runtime_unode_scalar_field_array(after_mesh_state, "U_ATTRIB_C")
    actual_flynn_tracer_values = _runtime_flynn_scalar_section_array(after_mesh_state, "F_ATTRIB_C")
    actual_label_values = (
        _runtime_unode_scalar_field_array(after_mesh_state, expected_label_attribute)
        if expected_label_attribute is not None
        else None
    )

    expected_box, expected_offset, expected_cumulative_shear = _runtime_elle_options_summary(
        {"_runtime_elle_options": expected_elle_options}
    )
    expected_reference_box_height = float(
        before_mesh_state.get("_mechanics_reference_box_height")
        if "_mechanics_reference_box_height" in before_mesh_state
        else _legacy_box_height_from_runtime_options(before_mesh_state.get("_runtime_elle_options"))
    )
    if abs(expected_reference_box_height) <= 1.0e-12:
        expected_reference_box_height = 1.0
    expected_direct_strain_summary = _legacy_direct_strain_axis_summary(
        expected_reference_box_height,
        expected_elle_options,
        cell_reset_stats,
    )
    actual_box, actual_offset, actual_cumulative_shear = _runtime_elle_options_summary(after_mesh_state)

    runtime_snapshot_alignment_mode = _runtime_mechanics_snapshot_array(after_mesh_state, "alignment_mode")
    runtime_snapshot_euler_alignment_mode = _runtime_mechanics_snapshot_array(after_mesh_state, "euler_alignment_mode")
    runtime_snapshot_temp_matrix = _runtime_mechanics_snapshot_array(after_mesh_state, "temp_matrix")
    runtime_snapshot_unode_strain_xyz = _runtime_mechanics_snapshot_array(after_mesh_state, "unode_strain_xyz")
    runtime_snapshot_statistical_density = _runtime_mechanics_snapshot_array(
        after_mesh_state,
        "statistical_dislocation_density",
    )
    runtime_snapshot_fourier_point_ids = _runtime_mechanics_snapshot_array(after_mesh_state, "fourier_point_ids")
    runtime_snapshot_fft_grain_numbers = _runtime_mechanics_snapshot_array(after_mesh_state, "fft_grain_numbers")

    actual_stats = (
        dict(after_mesh_state.get("stats", {}))
        if isinstance(after_mesh_state.get("stats"), dict)
        else {}
    )
    actual_runtime_snapshot = after_mesh_state.get("_runtime_mechanics_snapshot")
    actual_payload_summary = (
        dict(after_mesh_state.get("mechanics_payload_summary", {}))
        if isinstance(after_mesh_state.get("mechanics_payload_summary"), dict)
        else {}
    )
    actual_position_mode = (
        str(actual_runtime_snapshot.get("position_update_mode"))
        if isinstance(actual_runtime_snapshot, dict) and "position_update_mode" in actual_runtime_snapshot
        else "missing"
    )
    node_position_rmse = (
        0.0
        if actual_node_positions is None and expected_node_positions is None
        else _float_rmse(actual_node_positions, expected_node_positions)
    )

    report: dict[str, object] = {
        "alignment_mode_expected": str(payload.alignment_mode),
        "euler_alignment_mode_expected": str(payload.euler_alignment_mode),
        "euler_rmse": _float_rmse(actual_euler, expected_final_euler),
        "position_xy_rmse": _float_rmse(
            actual_positions,
            np.asarray(expected_seed_unodes.get("positions", ()), dtype=np.float64),
        ),
        "position_update_mode_expected": str(position_update_stats["position_update_mode"]),
        "position_update_mode_actual": str(actual_position_mode),
        "node_position_xy_rmse": float(node_position_rmse),
        "updated_node_positions_expected": int(expected_node_position_stats["updated_node_positions"]),
        "updated_node_positions_actual": int(
            actual_stats.get("mechanics_snapshot_updated_node_positions", -1)
        ),
        "node_position_update_mode_expected": str(expected_node_position_stats["node_position_update_mode"]),
        "node_position_update_mode_actual": str(
            actual_stats.get("mechanics_snapshot_node_position_update_mode", "missing")
        ),
        "cell_box_rmse": _float_rmse(actual_box, expected_box),
        "simple_shear_offset_abs_error": float(abs(actual_offset - expected_offset)),
        "cumulative_simple_shear_abs_error": float(
            abs(actual_cumulative_shear - expected_cumulative_shear)
        ),
        "direct_strain_axis_expected": float(expected_direct_strain_summary["direct_strain_axis"]),
        "direct_strain_axis_actual": (
            float(actual_payload_summary["direct_strain_axis"])
            if "direct_strain_axis" in actual_payload_summary
            else float("nan")
        ),
        "direct_strain_axis_abs_error": (
            float(
                abs(
                    float(actual_payload_summary["direct_strain_axis"])
                    - float(expected_direct_strain_summary["direct_strain_axis"])
                )
            )
            if "direct_strain_axis" in actual_payload_summary
            else float("inf")
        ),
        "strain_axis_source_expected": str(expected_direct_strain_summary["strain_axis_source"]),
        "strain_axis_source_actual": str(
            actual_payload_summary.get("strain_axis_source", "missing")
        ),
        "u_attrib_a_rmse": _float_rmse(actual_attr_a, payload.normalized_strain_rate),
        "u_attrib_b_rmse": _float_rmse(actual_attr_b, payload.normalized_stress),
        "u_attrib_d_rmse": _float_rmse(actual_attr_d, payload.basal_activity),
        "u_attrib_e_rmse": _float_rmse(actual_attr_e, payload.prismatic_activity),
        "density_field_expected": int(density_field_expected),
        "density_field_present": int(actual_density is not None),
        "u_dislocden_rmse": (
            _float_rmse(actual_density, expected_final_density)
            if density_field_expected
            else (0.0 if actual_density is None else float("inf"))
        ),
        "runtime_snapshot_temp_matrix_rmse": _float_rmse(runtime_snapshot_temp_matrix, payload.temp_matrix),
        "runtime_snapshot_unode_strain_xyz_rmse": _float_rmse(
            runtime_snapshot_unode_strain_xyz,
            payload.unode_strain_xyz,
        ),
        "runtime_snapshot_statistical_density_rmse": _float_rmse(
            None if payload.statistical_dislocation_density is None else runtime_snapshot_statistical_density,
            payload.statistical_dislocation_density,
        ),
        "runtime_snapshot_fourier_point_id_mismatch_count": _integer_mismatch_count(
            None if payload.fourier_point_ids is None else runtime_snapshot_fourier_point_ids,
            payload.fourier_point_ids,
        ),
        "runtime_snapshot_fft_grain_number_mismatch_count": _integer_mismatch_count(
            None if payload.fft_grain_numbers is None else runtime_snapshot_fft_grain_numbers,
            payload.fft_grain_numbers,
        ),
        "density_imported_unodes_expected": int(density_imported_unodes),
        "density_excluded_unodes_expected": int(density_excluded_unodes),
        "density_imported_unodes_actual": int(
            actual_stats.get("mechanics_snapshot_density_imported_unodes", -1)
        ),
        "density_excluded_unodes_actual": int(
            actual_stats.get("mechanics_snapshot_density_excluded_unodes", -1)
        ),
        "mechanics_snapshot_import_dislocation_densities_actual": int(
            actual_stats.get("mechanics_snapshot_import_dislocation_densities", -1)
        ),
        "mechanics_snapshot_exclude_phase_id_actual": int(
            actual_stats.get("mechanics_snapshot_exclude_phase_id", -1)
        ),
        "density_update_mode_expected": str(density_update_mode),
        "density_update_mode_actual": (
            str(actual_stats.get("mechanics_snapshot_density_update_mode"))
            if "mechanics_snapshot_density_update_mode" in actual_stats
            else "missing"
        ),
        "host_repair_mode_expected": str(host_repair_mode),
        "host_repair_mode_actual": (
            str(actual_stats.get("mechanics_snapshot_host_repair_mode"))
            if "mechanics_snapshot_host_repair_mode" in actual_stats
            else "missing"
        ),
        "unode_tracer_initialized_expected": int(expected_tracer_stats["unode_tracer_initialized"]),
        "unode_tracer_initialized_actual": int(
            actual_stats.get("mechanics_snapshot_unode_tracer_initialized", -1)
        ),
        "flynn_tracer_initialized_expected": int(expected_tracer_stats["flynn_tracer_initialized"]),
        "flynn_tracer_initialized_actual": int(
            actual_stats.get("mechanics_snapshot_flynn_tracer_initialized", -1)
        ),
        "tracer_assignment_mode_expected": str(expected_tracer_stats["tracer_assignment_mode"]),
        "tracer_assignment_mode_actual": str(
            actual_stats.get("mechanics_snapshot_tracer_assignment_mode", "missing")
        ),
        "unode_tracer_rmse": (
            0.0
            if int(expected_tracer_stats["unode_tracer_initialized"]) == 0
            else (
                0.0
                if actual_unode_tracer_values is None and expected_unode_tracer_values is None
                else _float_rmse(actual_unode_tracer_values, expected_unode_tracer_values)
            )
        ),
        "flynn_tracer_rmse": (
            0.0
            if int(expected_tracer_stats["flynn_tracer_initialized"]) == 0
            else (
                0.0
                if actual_flynn_tracer_values is None and expected_flynn_tracer_values is None
                else _float_rmse(actual_flynn_tracer_values, expected_flynn_tracer_values)
            )
        ),
        "swept_unodes_expected": int(expected_swept_stats["swept_unodes"]),
        "swept_unodes_actual": int(actual_stats.get("mechanics_snapshot_swept_unodes", -1)),
        "swept_reassignment_applied_expected": int(expected_swept_stats["swept_reassignment_applied"]),
        "swept_reassignment_applied_actual": int(
            actual_stats.get("mechanics_snapshot_swept_reassignment_applied", -1)
        ),
        "swept_reassignment_mode_expected": str(expected_swept_stats["swept_reassignment_mode"]),
        "swept_reassignment_mode_actual": str(
            actual_stats.get("mechanics_snapshot_swept_reassignment_mode", "missing")
        ),
        "updated_orientation_unodes_expected": int(expected_swept_stats["updated_orientation_unodes"]),
        "updated_orientation_unodes_actual": int(
            actual_stats.get("mechanics_snapshot_updated_orientation_unodes", -1)
        ),
        "fallback_orientation_unodes_expected": int(expected_swept_stats["fallback_orientation_unodes"]),
        "fallback_orientation_unodes_actual": int(
            actual_stats.get("mechanics_snapshot_fallback_orientation_unodes", -1)
        ),
        "old_value_fallback_orientation_unodes_expected": int(
            expected_swept_stats["old_value_fallback_orientation_unodes"]
        ),
        "old_value_fallback_orientation_unodes_actual": int(
            actual_stats.get("mechanics_snapshot_old_value_fallback_orientation_unodes", -1)
        ),
        "density_reset_unodes_expected": int(expected_swept_stats["density_reset_unodes"]),
        "density_reset_unodes_actual": int(
            actual_stats.get("mechanics_snapshot_density_reset_unodes", -1)
        ),
        "label_attribute_expected": None if expected_label_attribute is None else str(expected_label_attribute),
        "label_update_applied_expected": int(expected_label_stats["label_update_applied"]),
        "label_update_applied_actual": int(
            actual_stats.get("mechanics_snapshot_label_update_applied", 0)
        ),
        "label_changed_unodes_expected": int(expected_label_stats["label_changed_unodes"]),
        "label_changed_unodes_actual": int(
            actual_stats.get("mechanics_snapshot_label_changed_unodes", -1)
        ),
        "label_assignment_mode_expected": str(expected_label_stats["label_assignment_mode"]),
        "label_assignment_mode_actual": str(
            actual_stats.get("mechanics_snapshot_label_assignment_mode", "missing")
        ),
        "label_field_rmse": (
            _float_rmse(actual_label_values, expected_label_values)
            if expected_label_attribute is not None and expected_label_values is not None
            else 0.0
        ),
        "tex_payload_present": int(payload.normalized_strain_rate is not None),
    }
    report["euler_contract_match"] = int(float(report["euler_rmse"]) <= 1.0e-12)
    report["position_contract_match"] = int(
        float(report["position_xy_rmse"]) <= 1.0e-12
        and str(report["position_update_mode_expected"]) == str(report["position_update_mode_actual"])
        and float(report["node_position_xy_rmse"]) <= 1.0e-12
        and int(report["updated_node_positions_expected"]) == int(report["updated_node_positions_actual"])
        and str(report["node_position_update_mode_expected"])
        == str(report["node_position_update_mode_actual"])
    )
    report["cell_reset_contract_match"] = int(
        float(report["cell_box_rmse"]) <= 1.0e-12
        and float(report["simple_shear_offset_abs_error"]) <= 1.0e-12
        and float(report["cumulative_simple_shear_abs_error"]) <= 1.0e-12
        and float(report["direct_strain_axis_abs_error"]) <= 1.0e-12
        and str(report["strain_axis_source_expected"]) == str(report["strain_axis_source_actual"])
    )
    report["tex_contract_match"] = int(
        int(report["tex_payload_present"]) == 0
        or (
            float(report["u_attrib_a_rmse"]) <= 1.0e-12
            and float(report["u_attrib_b_rmse"]) <= 1.0e-12
            and float(report["u_attrib_d_rmse"]) <= 1.0e-12
            and float(report["u_attrib_e_rmse"]) <= 1.0e-12
        )
    )
    report["density_contract_match"] = int(
        int(report["density_field_expected"]) == int(report["density_field_present"])
        and float(report["u_dislocden_rmse"]) <= 1.0e-12
        and int(report["density_imported_unodes_expected"]) == int(report["density_imported_unodes_actual"])
        and int(report["density_excluded_unodes_expected"]) == int(report["density_excluded_unodes_actual"])
        and int(bool(import_options.import_dislocation_densities))
        == int(report["mechanics_snapshot_import_dislocation_densities_actual"])
        and int(import_options.exclude_phase_id) == int(report["mechanics_snapshot_exclude_phase_id_actual"])
        and str(report["density_update_mode_expected"]) == str(report["density_update_mode_actual"])
    )
    report["tracer_contract_match"] = int(
        int(report["unode_tracer_initialized_expected"]) == int(report["unode_tracer_initialized_actual"])
        and int(report["flynn_tracer_initialized_expected"]) == int(report["flynn_tracer_initialized_actual"])
        and str(report["tracer_assignment_mode_expected"]) == str(report["tracer_assignment_mode_actual"])
        and float(report["unode_tracer_rmse"]) <= 1.0e-12
        and float(report["flynn_tracer_rmse"]) <= 1.0e-12
    )
    report["runtime_snapshot_contract_match"] = int(
        isinstance(actual_runtime_snapshot, dict)
        and str(actual_runtime_snapshot.get("alignment_mode", "")) == str(payload.alignment_mode)
        and str(actual_runtime_snapshot.get("euler_alignment_mode", "")) == str(payload.euler_alignment_mode)
        and float(report["runtime_snapshot_temp_matrix_rmse"]) <= 1.0e-12
        and float(report["runtime_snapshot_unode_strain_xyz_rmse"]) <= 1.0e-12
        and (
            payload.statistical_dislocation_density is None
            or float(report["runtime_snapshot_statistical_density_rmse"]) <= 1.0e-12
        )
        and (
            payload.fourier_point_ids is None
            or int(report["runtime_snapshot_fourier_point_id_mismatch_count"]) == 0
        )
        and (
            payload.fft_grain_numbers is None
            or int(report["runtime_snapshot_fft_grain_number_mismatch_count"]) == 0
        )
    )
    report["label_contract_match"] = int(
        int(report["label_update_applied_expected"]) == int(report["label_update_applied_actual"])
        and int(report["label_changed_unodes_expected"]) == int(report["label_changed_unodes_actual"])
        and str(report["label_assignment_mode_expected"]) == str(report["label_assignment_mode_actual"])
        and float(report["label_field_rmse"]) <= 1.0e-12
    )
    report["swept_reassignment_contract_match"] = int(
        int(report["swept_unodes_expected"]) == int(report["swept_unodes_actual"])
        and int(report["swept_reassignment_applied_expected"])
        == int(report["swept_reassignment_applied_actual"])
        and str(report["host_repair_mode_expected"]) == str(report["host_repair_mode_actual"])
        and str(report["swept_reassignment_mode_expected"])
        == str(report["swept_reassignment_mode_actual"])
        and int(report["updated_orientation_unodes_expected"])
        == int(report["updated_orientation_unodes_actual"])
        and int(report["fallback_orientation_unodes_expected"])
        == int(report["fallback_orientation_unodes_actual"])
        and int(report["old_value_fallback_orientation_unodes_expected"])
        == int(report["old_value_fallback_orientation_unodes_actual"])
        and int(report["density_reset_unodes_expected"]) == int(report["density_reset_unodes_actual"])
    )
    report["mechanics_import_contract_match"] = int(
        int(report["euler_contract_match"]) == 1
        and int(report["position_contract_match"]) == 1
        and int(report["cell_reset_contract_match"]) == 1
        and int(report["tex_contract_match"]) == 1
        and int(report["density_contract_match"]) == 1
        and int(report["tracer_contract_match"]) == 1
        and int(report["swept_reassignment_contract_match"]) == 1
        and int(report["label_contract_match"]) == 1
        and int(report["runtime_snapshot_contract_match"]) == 1
    )
    return report
