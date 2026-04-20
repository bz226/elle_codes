from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


LEGACY_ALLOUT_COLUMNS = (
    "SVM",
    "DVM",
    "diffStress",
    "stressFieldErr",
    "strainrateFieldErr",
    "basalact",
    "prismact",
    "pyramact",
    "s11",
    "s22",
    "s33",
    "s23",
    "s13",
    "s12",
    "d11",
    "d22",
    "d33",
    "d23",
    "d13",
    "d12",
)


@dataclass(frozen=True)
class LegacyAllOutStatisticsRow:
    source: str
    mean_von_mises_stress: float
    mean_von_mises_strain_rate: float
    mean_differential_stress: float
    stress_field_error: float
    strain_rate_field_error: float
    mean_basal_activity: float
    mean_prismatic_activity: float
    mean_pyramidal_activity: float
    mean_total_activity: float
    mean_prismatic_fraction: float
    prismatic_to_basal_ratio: float
    stress_tensor: tuple[float, float, float, float, float, float]
    strain_rate_tensor: tuple[float, float, float, float, float, float]

    def to_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "mean_von_mises_stress": float(self.mean_von_mises_stress),
            "mean_von_mises_strain_rate": float(self.mean_von_mises_strain_rate),
            "mean_differential_stress": float(self.mean_differential_stress),
            "stress_field_error": float(self.stress_field_error),
            "strain_rate_field_error": float(self.strain_rate_field_error),
            "mean_basal_activity": float(self.mean_basal_activity),
            "mean_prismatic_activity": float(self.mean_prismatic_activity),
            "mean_pyramidal_activity": float(self.mean_pyramidal_activity),
            "mean_total_activity": float(self.mean_total_activity),
            "mean_prismatic_fraction": float(self.mean_prismatic_fraction),
            "prismatic_to_basal_ratio": float(self.prismatic_to_basal_ratio),
            "stress_tensor": list(self.stress_tensor),
            "strain_rate_tensor": list(self.strain_rate_tensor),
        }


@dataclass(frozen=True)
class LegacyTmpStatsSummary:
    max_subgrain_count: int
    total_grain_number: int
    average_grain_size: float
    second_moment_grain_size: float
    total_boundary_length: float | None = None
    ratio: float | None = None
    max_ang: float | None = None
    min_ang: float | None = None
    accuracy: float | None = None
    min_max_orientation_bins: dict[str, float] | None = None
    max_orientation_bins: dict[str, float] | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "max_subgrain_count": int(self.max_subgrain_count),
            "total_grain_number": int(self.total_grain_number),
            "average_grain_size": float(self.average_grain_size),
            "second_moment_grain_size": float(self.second_moment_grain_size),
            "total_boundary_length": (
                None if self.total_boundary_length is None else float(self.total_boundary_length)
            ),
            "ratio": None if self.ratio is None else float(self.ratio),
            "max_ang": None if self.max_ang is None else float(self.max_ang),
            "min_ang": None if self.min_ang is None else float(self.min_ang),
            "accuracy": None if self.accuracy is None else float(self.accuracy),
            "min_max_orientation_bins": None if self.min_max_orientation_bins is None else dict(self.min_max_orientation_bins),
            "max_orientation_bins": None if self.max_orientation_bins is None else dict(self.max_orientation_bins),
        }


@dataclass(frozen=True)
class LegacyOldStatsRow:
    mineral: str
    flynn_number: int
    grain_number: int
    split: int
    area: float
    cycle: float
    age: float

    def to_dict(self) -> dict[str, object]:
        return {
            "mineral": self.mineral,
            "flynn_number": int(self.flynn_number),
            "grain_number": int(self.grain_number),
            "split": int(self.split),
            "area": float(self.area),
            "cycle": float(self.cycle),
            "age": float(self.age),
        }


@dataclass(frozen=True)
class LegacyOldStatsSummary:
    max_subgrain_count: int
    flynn_rows: tuple[LegacyOldStatsRow, ...]
    total_grain_number: int
    average_grain_size: float
    second_moment_grain_size: float
    total_boundary_length: float | None = None
    ratio: float | None = None
    max_ang: float | None = None
    min_ang: float | None = None
    accuracy: float | None = None
    min_max_orientation_bins: dict[str, float] | None = None
    max_orientation_bins: dict[str, float] | None = None

    @property
    def mapped_flynn_count(self) -> int:
        return sum(1 for row in self.flynn_rows if int(row.grain_number) >= 0)

    @property
    def orphan_flynn_count(self) -> int:
        return sum(1 for row in self.flynn_rows if int(row.grain_number) < 0)

    @property
    def split_flynn_count(self) -> int:
        return sum(1 for row in self.flynn_rows if int(row.split) != 0)

    @property
    def active_cycle_flynn_count(self) -> int:
        return sum(1 for row in self.flynn_rows if float(row.cycle) > 0.0)

    def to_dict(self) -> dict[str, object]:
        return {
            "max_subgrain_count": int(self.max_subgrain_count),
            "flynn_rows": [row.to_dict() for row in self.flynn_rows],
            "total_grain_number": int(self.total_grain_number),
            "average_grain_size": float(self.average_grain_size),
            "second_moment_grain_size": float(self.second_moment_grain_size),
            "mapped_flynn_count": int(self.mapped_flynn_count),
            "orphan_flynn_count": int(self.orphan_flynn_count),
            "split_flynn_count": int(self.split_flynn_count),
            "active_cycle_flynn_count": int(self.active_cycle_flynn_count),
            "total_boundary_length": (
                None if self.total_boundary_length is None else float(self.total_boundary_length)
            ),
            "ratio": None if self.ratio is None else float(self.ratio),
            "max_ang": None if self.max_ang is None else float(self.max_ang),
            "min_ang": None if self.min_ang is None else float(self.min_ang),
            "accuracy": None if self.accuracy is None else float(self.accuracy),
            "min_max_orientation_bins": None if self.min_max_orientation_bins is None else dict(self.min_max_orientation_bins),
            "max_orientation_bins": None if self.max_orientation_bins is None else dict(self.max_orientation_bins),
        }


@dataclass(frozen=True)
class LegacyLastStatsRow:
    mineral: str
    flynn_number: int
    area: float
    cycle: float
    age: float

    def to_dict(self) -> dict[str, object]:
        return {
            "mineral": self.mineral,
            "flynn_number": int(self.flynn_number),
            "area": float(self.area),
            "cycle": float(self.cycle),
            "age": float(self.age),
        }


@dataclass(frozen=True)
class LegacyLastStatsSummary:
    max_subgrain_count: int
    flynn_rows: tuple[LegacyLastStatsRow, ...]
    total_grain_number: int
    average_grain_size: float
    second_moment_grain_size: float
    total_boundary_length: float | None = None
    ratio: float | None = None
    max_ang: float | None = None
    min_ang: float | None = None
    accuracy: float | None = None
    min_max_orientation_bins: dict[str, float] | None = None
    max_orientation_bins: dict[str, float] | None = None

    @property
    def active_flynn_count(self) -> int:
        return len(self.flynn_rows)

    @property
    def active_cycle_flynn_count(self) -> int:
        return sum(1 for row in self.flynn_rows if float(row.cycle) > 0.0)

    def to_dict(self) -> dict[str, object]:
        return {
            "max_subgrain_count": int(self.max_subgrain_count),
            "flynn_rows": [row.to_dict() for row in self.flynn_rows],
            "active_flynn_count": int(self.active_flynn_count),
            "active_cycle_flynn_count": int(self.active_cycle_flynn_count),
            "total_grain_number": int(self.total_grain_number),
            "average_grain_size": float(self.average_grain_size),
            "second_moment_grain_size": float(self.second_moment_grain_size),
            "total_boundary_length": (
                None if self.total_boundary_length is None else float(self.total_boundary_length)
            ),
            "ratio": None if self.ratio is None else float(self.ratio),
            "max_ang": None if self.max_ang is None else float(self.max_ang),
            "min_ang": None if self.min_ang is None else float(self.min_ang),
            "accuracy": None if self.accuracy is None else float(self.accuracy),
            "min_max_orientation_bins": None if self.min_max_orientation_bins is None else dict(self.min_max_orientation_bins),
            "max_orientation_bins": None if self.max_orientation_bins is None else dict(self.max_orientation_bins),
        }


@dataclass(frozen=True)
class CurrentMeshBookkeepingSummary:
    source: str
    total_flynn_count: int
    retained_flynn_count: int
    nonretained_flynn_count: int
    source_mapped_flynn_count: int
    source_orphan_flynn_count: int
    unique_source_flynn_count: int
    multi_parent_flynn_count: int
    mesh_split_flynn_count: int
    mesh_merged_flynn_count: int
    mesh_stats_num_flynns: int | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "total_flynn_count": int(self.total_flynn_count),
            "retained_flynn_count": int(self.retained_flynn_count),
            "nonretained_flynn_count": int(self.nonretained_flynn_count),
            "source_mapped_flynn_count": int(self.source_mapped_flynn_count),
            "source_orphan_flynn_count": int(self.source_orphan_flynn_count),
            "unique_source_flynn_count": int(self.unique_source_flynn_count),
            "multi_parent_flynn_count": int(self.multi_parent_flynn_count),
            "mesh_split_flynn_count": int(self.mesh_split_flynn_count),
            "mesh_merged_flynn_count": int(self.mesh_merged_flynn_count),
            "mesh_stats_num_flynns": (
                None if self.mesh_stats_num_flynns is None else int(self.mesh_stats_num_flynns)
            ),
        }


@dataclass(frozen=True)
class LegacyOldStatsBookkeepingComparison:
    source: str
    legacy_total_flynn_count: int
    current_total_flynn_count: int
    legacy_mapped_flynn_count: int
    current_source_mapped_flynn_count: int
    legacy_orphan_flynn_count: int
    current_source_orphan_flynn_count: int
    legacy_split_flynn_count: int
    current_multi_parent_flynn_count: int
    current_nonretained_flynn_count: int
    legacy_total_grain_number: int
    current_unique_source_flynn_count: int
    total_flynn_count_match: bool
    mapped_flynn_count_match: bool
    orphan_flynn_count_match: bool
    total_grain_number_match: bool
    split_count_match_via_multi_parent: bool
    split_count_match_via_mesh_stats: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "legacy_total_flynn_count": int(self.legacy_total_flynn_count),
            "current_total_flynn_count": int(self.current_total_flynn_count),
            "legacy_mapped_flynn_count": int(self.legacy_mapped_flynn_count),
            "current_source_mapped_flynn_count": int(self.current_source_mapped_flynn_count),
            "legacy_orphan_flynn_count": int(self.legacy_orphan_flynn_count),
            "current_source_orphan_flynn_count": int(self.current_source_orphan_flynn_count),
            "legacy_split_flynn_count": int(self.legacy_split_flynn_count),
            "current_multi_parent_flynn_count": int(self.current_multi_parent_flynn_count),
            "current_nonretained_flynn_count": int(self.current_nonretained_flynn_count),
            "legacy_total_grain_number": int(self.legacy_total_grain_number),
            "current_unique_source_flynn_count": int(self.current_unique_source_flynn_count),
            "total_flynn_count_match": bool(self.total_flynn_count_match),
            "mapped_flynn_count_match": bool(self.mapped_flynn_count_match),
            "orphan_flynn_count_match": bool(self.orphan_flynn_count_match),
            "total_grain_number_match": bool(self.total_grain_number_match),
            "split_count_match_via_multi_parent": bool(self.split_count_match_via_multi_parent),
            "split_count_match_via_mesh_stats": bool(self.split_count_match_via_mesh_stats),
            "count_deltas": {
                "total_flynn_count": int(self.current_total_flynn_count - self.legacy_total_flynn_count),
                "mapped_flynn_count": int(self.current_source_mapped_flynn_count - self.legacy_mapped_flynn_count),
                "orphan_flynn_count": int(self.current_source_orphan_flynn_count - self.legacy_orphan_flynn_count),
                "total_grain_number": int(self.current_unique_source_flynn_count - self.legacy_total_grain_number),
                "split_count_via_multi_parent": int(self.current_multi_parent_flynn_count - self.legacy_split_flynn_count),
            },
        }


@dataclass(frozen=True)
class LegacyStatisticsSummaryComparison:
    source: str
    legacy_statistics_kind: str
    current_grain_count: int
    legacy_total_grain_number: int
    current_mean_grain_area: float
    legacy_average_grain_size: float
    current_second_moment_grain_size: float
    legacy_second_moment_grain_size: float
    grain_count_match: bool
    mean_grain_area_match: bool
    second_moment_grain_size_match: bool
    grain_count_delta: int
    mean_grain_area_delta: float
    second_moment_grain_size_delta: float

    def to_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "legacy_statistics_kind": self.legacy_statistics_kind,
            "current_grain_count": int(self.current_grain_count),
            "legacy_total_grain_number": int(self.legacy_total_grain_number),
            "current_mean_grain_area": float(self.current_mean_grain_area),
            "legacy_average_grain_size": float(self.legacy_average_grain_size),
            "current_second_moment_grain_size": float(self.current_second_moment_grain_size),
            "legacy_second_moment_grain_size": float(self.legacy_second_moment_grain_size),
            "grain_count_match": bool(self.grain_count_match),
            "mean_grain_area_match": bool(self.mean_grain_area_match),
            "second_moment_grain_size_match": bool(self.second_moment_grain_size_match),
            "grain_count_delta": int(self.grain_count_delta),
            "mean_grain_area_delta": float(self.mean_grain_area_delta),
            "second_moment_grain_size_delta": float(self.second_moment_grain_size_delta),
        }


def load_legacy_allout_statistics(path: str | Path) -> list[LegacyAllOutStatisticsRow]:
    allout_path = Path(path)
    rows: list[LegacyAllOutStatisticsRow] = []
    expect_numeric_row = False
    for raw_line in allout_path.read_text(encoding="utf-8").splitlines():
        tokens = raw_line.strip().split()
        if not tokens:
            continue

        value_tokens: list[str] | None = None
        if len(tokens) >= 2 * len(LEGACY_ALLOUT_COLUMNS) and tuple(tokens[: len(LEGACY_ALLOUT_COLUMNS)]) == LEGACY_ALLOUT_COLUMNS:
            value_tokens = list(tokens[len(LEGACY_ALLOUT_COLUMNS) : 2 * len(LEGACY_ALLOUT_COLUMNS)])
            expect_numeric_row = False
        elif tuple(tokens) == LEGACY_ALLOUT_COLUMNS:
            expect_numeric_row = True
            continue
        elif expect_numeric_row and len(tokens) >= len(LEGACY_ALLOUT_COLUMNS):
            value_tokens = list(tokens[: len(LEGACY_ALLOUT_COLUMNS)])
            expect_numeric_row = False
        elif len(tokens) == len(LEGACY_ALLOUT_COLUMNS):
            value_tokens = list(tokens)

        if value_tokens is None:
            continue

        values = np.asarray([float(value) for value in value_tokens], dtype=np.float64)
        mean_basal_activity = float(values[5])
        mean_prismatic_activity = float(values[6])
        mean_pyramidal_activity = float(values[7])
        mean_total_activity = float(mean_basal_activity + mean_prismatic_activity + mean_pyramidal_activity)
        mean_prismatic_fraction = (
            float(mean_prismatic_activity / mean_total_activity)
            if abs(mean_total_activity) > 1.0e-12
            else 0.0
        )
        prismatic_to_basal_ratio = (
            float(mean_prismatic_activity / mean_basal_activity)
            if abs(mean_basal_activity) > 1.0e-12
            else float("nan")
        )
        rows.append(
            LegacyAllOutStatisticsRow(
                source="legacy_allout_statistics",
                mean_von_mises_stress=float(values[0]),
                mean_von_mises_strain_rate=float(values[1]),
                mean_differential_stress=float(values[2]),
                stress_field_error=float(values[3]),
                strain_rate_field_error=float(values[4]),
                mean_basal_activity=mean_basal_activity,
                mean_prismatic_activity=mean_prismatic_activity,
                mean_pyramidal_activity=mean_pyramidal_activity,
                mean_total_activity=mean_total_activity,
                mean_prismatic_fraction=mean_prismatic_fraction,
                prismatic_to_basal_ratio=prismatic_to_basal_ratio,
                stress_tensor=tuple(float(value) for value in values[8:14]),
                strain_rate_tensor=tuple(float(value) for value in values[14:20]),
            )
        )

    return rows


def load_legacy_tmpstats_summary(path: str | Path) -> LegacyTmpStatsSummary:
    lines = [line.rstrip("\n") for line in Path(path).read_text(encoding="utf-8").splitlines()]
    if not lines:
        raise ValueError(f"{path} is empty")

    max_subgrain_count = int(lines[0].strip())
    total_grain_number: int | None = None
    average_grain_size: float | None = None
    second_moment_grain_size: float | None = None
    total_boundary_length: float | None = None
    ratio: float | None = None
    max_ang: float | None = None
    min_ang: float | None = None
    accuracy: float | None = None
    min_max_orientation_bins: dict[str, float] | None = None
    max_orientation_bins: dict[str, float] | None = None

    def _looks_like_orientation_bin(label: str) -> bool:
        if not label:
            return False
        first = label[0]
        return first == "<" or first.isdigit()

    index = 1
    while index < len(lines):
        stripped = lines[index].strip()
        index += 1
        if not stripped:
            continue

        if stripped == "min_max orientation statistics":
            bins: dict[str, float] = {}
            while index < len(lines):
                entry = lines[index].strip()
                if not entry or "\t" not in entry:
                    break
                label, value = entry.split("\t", 1)
                if not _looks_like_orientation_bin(label):
                    break
                bins[str(label)] = float(value)
                index += 1
            min_max_orientation_bins = bins
            continue

        if stripped == "max orientation statistics":
            bins = {}
            while index < len(lines):
                entry = lines[index].strip()
                if not entry or "\t" not in entry:
                    break
                label, value = entry.split("\t", 1)
                if not _looks_like_orientation_bin(label):
                    break
                bins[str(label)] = float(value)
                index += 1
            max_orientation_bins = bins
            continue

        if "\t" not in stripped:
            continue
        key, value = stripped.split("\t", 1)
        key = key.strip()
        value = value.strip()
        if key == "total grain number":
            total_grain_number = int(value)
        elif key == "average grain size":
            average_grain_size = float(value)
        elif key == "second moment grain size":
            second_moment_grain_size = float(value)
        elif key == "total_b_length":
            total_boundary_length = float(value)
        elif key == "ratio":
            ratio = float(value)
        elif key == "maxAng":
            max_ang = float(value)
        elif key == "minAng":
            min_ang = float(value)
        elif key == "accuracy":
            accuracy = float(value)

    if total_grain_number is None or average_grain_size is None or second_moment_grain_size is None:
        raise ValueError(f"{path} is missing required tmpstats summary fields")

    return LegacyTmpStatsSummary(
        max_subgrain_count=max_subgrain_count,
        total_grain_number=total_grain_number,
        average_grain_size=average_grain_size,
        second_moment_grain_size=second_moment_grain_size,
        total_boundary_length=total_boundary_length,
        ratio=ratio,
        max_ang=max_ang,
        min_ang=min_ang,
        accuracy=accuracy,
        min_max_orientation_bins=min_max_orientation_bins,
        max_orientation_bins=max_orientation_bins,
    )


def load_legacy_old_stats_summary(path: str | Path) -> LegacyOldStatsSummary:
    lines = [line.rstrip("\n") for line in Path(path).read_text(encoding="utf-8").splitlines()]
    if not lines:
        raise ValueError(f"{path} is empty")

    max_subgrain_count = int(lines[0].strip())
    row_index = 1
    while row_index < len(lines) and not lines[row_index].strip():
        row_index += 1
    if row_index >= len(lines):
        raise ValueError(f"{path} is missing old.stats header")

    header = lines[row_index].strip().split("\t")
    if tuple(header) != ("mineral", "number", "grain", "split", "area", "cycle", "age"):
        raise ValueError(f"{path} has unexpected old.stats header {header!r}")
    row_index += 1

    flynn_rows: list[LegacyOldStatsRow] = []
    while row_index < len(lines):
        stripped = lines[row_index].strip()
        if not stripped:
            row_index += 1
            continue
        if stripped.startswith("total grain number\t"):
            break
        parts = stripped.split("\t")
        if len(parts) != 7:
            row_index += 1
            continue
        flynn_rows.append(
            LegacyOldStatsRow(
                mineral=str(parts[0]),
                flynn_number=int(parts[1]),
                grain_number=int(parts[2]),
                split=int(parts[3]),
                area=float(parts[4]),
                cycle=float(parts[5]),
                age=float(parts[6]),
            )
        )
        row_index += 1

    if row_index >= len(lines):
        raise ValueError(f"{path} is missing required old.stats summary fields")

    summary = load_legacy_tmpstats_summary(path)
    return LegacyOldStatsSummary(
        max_subgrain_count=max_subgrain_count,
        flynn_rows=tuple(flynn_rows),
        total_grain_number=summary.total_grain_number,
        average_grain_size=summary.average_grain_size,
        second_moment_grain_size=summary.second_moment_grain_size,
        total_boundary_length=summary.total_boundary_length,
        ratio=summary.ratio,
        max_ang=summary.max_ang,
        min_ang=summary.min_ang,
        accuracy=summary.accuracy,
        min_max_orientation_bins=summary.min_max_orientation_bins,
        max_orientation_bins=summary.max_orientation_bins,
    )


def load_legacy_last_stats_summary(path: str | Path) -> LegacyLastStatsSummary:
    lines = [line.rstrip("\n") for line in Path(path).read_text(encoding="utf-8").splitlines()]
    if not lines:
        raise ValueError(f"{path} is empty")

    max_subgrain_count = int(lines[0].strip())
    row_index = 1
    while row_index < len(lines) and not lines[row_index].strip():
        row_index += 1
    if row_index >= len(lines):
        raise ValueError(f"{path} is missing last.stats header")

    header = lines[row_index].strip().split("\t")
    if tuple(header) != ("mineral", "number", "area", "cycle", "age"):
        raise ValueError(f"{path} has unexpected last.stats header {header!r}")
    row_index += 1

    flynn_rows: list[LegacyLastStatsRow] = []
    while row_index < len(lines):
        stripped = lines[row_index].strip()
        if not stripped:
            row_index += 1
            continue
        if stripped.startswith("total grain number\t") or stripped.startswith("grain number\t") or stripped.startswith("min_max orientation statistics"):
            break
        parts = stripped.split("\t")
        if len(parts) != 5:
            row_index += 1
            continue
        flynn_rows.append(
            LegacyLastStatsRow(
                mineral=str(parts[0]),
                flynn_number=int(parts[1]),
                area=float(parts[2]),
                cycle=float(parts[3]),
                age=float(parts[4]),
            )
        )
        row_index += 1

    summary = load_legacy_tmpstats_summary(path)
    return LegacyLastStatsSummary(
        max_subgrain_count=max_subgrain_count,
        flynn_rows=tuple(flynn_rows),
        total_grain_number=summary.total_grain_number,
        average_grain_size=summary.average_grain_size,
        second_moment_grain_size=summary.second_moment_grain_size,
        total_boundary_length=summary.total_boundary_length,
        ratio=summary.ratio,
        max_ang=summary.max_ang,
        min_ang=summary.min_ang,
        accuracy=summary.accuracy,
        min_max_orientation_bins=summary.min_max_orientation_bins,
        max_orientation_bins=summary.max_orientation_bins,
    )


def load_legacy_statistics_summary(
    path: str | Path,
) -> LegacyTmpStatsSummary | LegacyOldStatsSummary | LegacyLastStatsSummary:
    path_obj = Path(path)
    name = path_obj.name.lower()
    if name == "tmpstats.dat":
        return load_legacy_tmpstats_summary(path_obj)
    if name == "old.stats":
        return load_legacy_old_stats_summary(path_obj)
    if name == "last.stats":
        return load_legacy_last_stats_summary(path_obj)

    lines = [line.rstrip("\n") for line in path_obj.read_text(encoding="utf-8").splitlines()]
    if not lines:
        raise ValueError(f"{path} is empty")

    header_index = 1
    while header_index < len(lines) and not lines[header_index].strip():
        header_index += 1
    if header_index >= len(lines):
        raise ValueError(f"{path} has no detectable legacy statistics header")

    header = tuple(lines[header_index].strip().split("\t"))
    if header == ("mineral", "number", "grain", "split", "area", "cycle", "age"):
        return load_legacy_old_stats_summary(path)
    if header == ("mineral", "number", "area", "cycle", "age"):
        return load_legacy_last_stats_summary(path_obj)
    return load_legacy_tmpstats_summary(path)


def summarize_current_mesh_bookkeeping(mesh_state_or_path: str | Path | dict[str, Any]) -> CurrentMeshBookkeepingSummary:
    if isinstance(mesh_state_or_path, (str, Path)):
        source = str(Path(mesh_state_or_path))
        mesh_state = json.loads(Path(mesh_state_or_path).read_text(encoding="utf-8"))
    else:
        source = "current_mesh_bookkeeping"
        mesh_state = dict(mesh_state_or_path)

    flynns = list(mesh_state.get("flynns", []))
    stats = dict(mesh_state.get("stats", {}))

    retained_flynn_count = 0
    source_mapped_flynn_count = 0
    source_orphan_flynn_count = 0
    multi_parent_flynn_count = 0
    unique_source_flynn_ids: set[int] = set()

    for flynn in flynns:
        if bool(flynn.get("retained_identity", True)):
            retained_flynn_count += 1
        parents = [int(parent) for parent in flynn.get("parents", [])]
        if len(parents) > 1:
            multi_parent_flynn_count += 1
        source_flynn_id = flynn.get("source_flynn_id")
        if source_flynn_id is None or int(source_flynn_id) < 0:
            source_orphan_flynn_count += 1
        else:
            source_mapped_flynn_count += 1
            unique_source_flynn_ids.add(int(source_flynn_id))

    total_flynn_count = len(flynns)
    return CurrentMeshBookkeepingSummary(
        source=source,
        total_flynn_count=total_flynn_count,
        retained_flynn_count=retained_flynn_count,
        nonretained_flynn_count=int(total_flynn_count - retained_flynn_count),
        source_mapped_flynn_count=source_mapped_flynn_count,
        source_orphan_flynn_count=source_orphan_flynn_count,
        unique_source_flynn_count=len(unique_source_flynn_ids),
        multi_parent_flynn_count=multi_parent_flynn_count,
        mesh_split_flynn_count=int(stats.get("mesh_split_flynns", 0)),
        mesh_merged_flynn_count=int(stats.get("mesh_merged_flynns", 0)),
        mesh_stats_num_flynns=(
            None if "num_flynns" not in stats else int(stats["num_flynns"])
        ),
    )


def compare_mesh_bookkeeping_to_legacy_old_stats(
    mesh_state_or_path: str | Path | dict[str, Any],
    legacy_old_stats_or_path: str | Path | LegacyOldStatsSummary,
) -> LegacyOldStatsBookkeepingComparison:
    current_summary = summarize_current_mesh_bookkeeping(mesh_state_or_path)
    if isinstance(legacy_old_stats_or_path, LegacyOldStatsSummary):
        legacy_summary = legacy_old_stats_or_path
    else:
        legacy_summary = load_legacy_old_stats_summary(legacy_old_stats_or_path)

    return LegacyOldStatsBookkeepingComparison(
        source="legacy_old_stats_bookkeeping_comparison",
        legacy_total_flynn_count=len(legacy_summary.flynn_rows),
        current_total_flynn_count=current_summary.total_flynn_count,
        legacy_mapped_flynn_count=legacy_summary.mapped_flynn_count,
        current_source_mapped_flynn_count=current_summary.source_mapped_flynn_count,
        legacy_orphan_flynn_count=legacy_summary.orphan_flynn_count,
        current_source_orphan_flynn_count=current_summary.source_orphan_flynn_count,
        legacy_split_flynn_count=legacy_summary.split_flynn_count,
        current_multi_parent_flynn_count=current_summary.multi_parent_flynn_count,
        current_nonretained_flynn_count=current_summary.nonretained_flynn_count,
        legacy_total_grain_number=legacy_summary.total_grain_number,
        current_unique_source_flynn_count=current_summary.unique_source_flynn_count,
        total_flynn_count_match=bool(len(legacy_summary.flynn_rows) == current_summary.total_flynn_count),
        mapped_flynn_count_match=bool(legacy_summary.mapped_flynn_count == current_summary.source_mapped_flynn_count),
        orphan_flynn_count_match=bool(legacy_summary.orphan_flynn_count == current_summary.source_orphan_flynn_count),
        total_grain_number_match=bool(legacy_summary.total_grain_number == current_summary.unique_source_flynn_count),
        split_count_match_via_multi_parent=bool(legacy_summary.split_flynn_count == current_summary.multi_parent_flynn_count),
        split_count_match_via_mesh_stats=bool(legacy_summary.split_flynn_count == current_summary.mesh_split_flynn_count),
    )


def compare_snapshot_summary_to_legacy_statistics(
    snapshot_summary: dict[str, Any],
    legacy_summary_or_path: str | Path | LegacyTmpStatsSummary | LegacyOldStatsSummary | LegacyLastStatsSummary,
    *,
    atol: float = 1.0e-12,
) -> LegacyStatisticsSummaryComparison:
    if isinstance(
        legacy_summary_or_path,
        (LegacyTmpStatsSummary, LegacyOldStatsSummary, LegacyLastStatsSummary),
    ):
        legacy_summary = legacy_summary_or_path
    else:
        legacy_summary = load_legacy_statistics_summary(legacy_summary_or_path)

    if isinstance(legacy_summary, LegacyOldStatsSummary):
        legacy_kind = "old.stats"
    elif isinstance(legacy_summary, LegacyLastStatsSummary):
        legacy_kind = "last.stats"
    else:
        legacy_kind = "tmpstats.dat"

    current_grain_count = int(snapshot_summary["grain_count"])
    current_mean_grain_area = float(snapshot_summary["mean_grain_area"])
    current_second_moment_grain_size = float(snapshot_summary["second_moment_grain_size"])
    legacy_total_grain_number = int(legacy_summary.total_grain_number)
    legacy_average_grain_size = float(legacy_summary.average_grain_size)
    legacy_second_moment_grain_size = float(legacy_summary.second_moment_grain_size)

    return LegacyStatisticsSummaryComparison(
        source="legacy_statistics_summary_comparison",
        legacy_statistics_kind=legacy_kind,
        current_grain_count=current_grain_count,
        legacy_total_grain_number=legacy_total_grain_number,
        current_mean_grain_area=current_mean_grain_area,
        legacy_average_grain_size=legacy_average_grain_size,
        current_second_moment_grain_size=current_second_moment_grain_size,
        legacy_second_moment_grain_size=legacy_second_moment_grain_size,
        grain_count_match=bool(current_grain_count == legacy_total_grain_number),
        mean_grain_area_match=bool(abs(current_mean_grain_area - legacy_average_grain_size) <= atol),
        second_moment_grain_size_match=bool(
            abs(current_second_moment_grain_size - legacy_second_moment_grain_size) <= atol
        ),
        grain_count_delta=int(current_grain_count - legacy_total_grain_number),
        mean_grain_area_delta=float(current_mean_grain_area - legacy_average_grain_size),
        second_moment_grain_size_delta=float(
            current_second_moment_grain_size - legacy_second_moment_grain_size
        ),
    )
