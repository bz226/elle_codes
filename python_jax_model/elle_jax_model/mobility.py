from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


_GBM_R_GAS_CONSTANT = 8.314472


@dataclass(frozen=True)
class PhaseBoundaryProperty:
    mobility: float
    boundary_energy: float
    activation_energy: float


@dataclass(frozen=True)
class PhaseProperty:
    cluster_diffusion: int
    diff_times: int
    kappa: float
    stored_energy: float
    disscale: float
    disbondscale: float


@dataclass(frozen=True)
class FaithfulMobilityDB:
    path: str
    first_phase: int
    no_phases: int
    phase_properties: dict[int, PhaseProperty]
    pairs: dict[tuple[int, int], PhaseBoundaryProperty]
    cluster_multiplier_a: float
    cluster_multiplier_b: float
    cluster_multiplier_c: float
    cluster_multiplier_d: float


def default_phase_db_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "FS_Codes"
        / "FS_recrystallisation"
        / "FS_gbm_pp_fft"
        / "phase_db.txt"
    )


@lru_cache(maxsize=8)
def load_phase_boundary_db(phase_db_path: str | None = None) -> FaithfulMobilityDB:
    path = default_phase_db_path() if phase_db_path is None else Path(phase_db_path)
    if not path.exists():
        raise FileNotFoundError(f"phase_db.txt not found: {path}")

    first_phase: int | None = None
    no_phases: int | None = None
    section = 0
    phase_properties: dict[int, PhaseProperty] = {}
    pairs: dict[tuple[int, int], PhaseBoundaryProperty] = {}
    cluster_multiplier_a = 0.0
    cluster_multiplier_b = 0.0
    cluster_multiplier_c = 0.0
    cluster_multiplier_d = 0.0

    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                if "PHASE PROPERTIES" in raw_line:
                    section = 1
                elif "PHASE BOUNDARY PROPERTIES" in raw_line:
                    section = 2
                elif "MELT TRACKING" in raw_line:
                    section = 3
                elif "CLUSTER_TRACKING" in raw_line:
                    section = 4
                continue

            parts = line.split()
            if first_phase is None and no_phases is None:
                if len(parts) < 2:
                    raise ValueError(f"invalid phase_db header line in {path}: {line!r}")
                first_phase = int(parts[0])
                no_phases = int(parts[1])
                continue

            if section != 2:
                if section == 1 and len(parts) >= 7:
                    phase_number = int(parts[0])
                    phase_properties[phase_number] = PhaseProperty(
                        cluster_diffusion=int(parts[1]),
                        diff_times=int(parts[2]),
                        kappa=float(parts[3]),
                        stored_energy=float(parts[4]),
                        disscale=float(parts[5]),
                        disbondscale=float(parts[6]),
                    )
                elif section == 4 and len(parts) >= 4:
                    cluster_multiplier_a = float(parts[0])
                    cluster_multiplier_b = float(parts[1])
                    cluster_multiplier_c = float(parts[2])
                    cluster_multiplier_d = float(parts[3])
                continue
            if len(parts) >= 5:
                phase_a = int(parts[0])
                phase_b = int(parts[1])
                prop = PhaseBoundaryProperty(
                    mobility=float(parts[2]),
                    boundary_energy=float(parts[3]),
                    activation_energy=float(parts[4]),
                )
                pairs[(phase_a, phase_b)] = prop
                pairs[(phase_b, phase_a)] = prop

    if first_phase is None or no_phases is None:
        raise ValueError(f"phase_db.txt is missing phase header: {path}")
    if not pairs:
        raise ValueError(f"phase_db.txt is missing boundary-pair data: {path}")

    return FaithfulMobilityDB(
        path=str(path),
        first_phase=int(first_phase),
        no_phases=int(no_phases),
        phase_properties=phase_properties,
        pairs=pairs,
        cluster_multiplier_a=float(cluster_multiplier_a),
        cluster_multiplier_b=float(cluster_multiplier_b),
        cluster_multiplier_c=float(cluster_multiplier_c),
        cluster_multiplier_d=float(cluster_multiplier_d),
    )


def get_phase_boundary_property(
    db: FaithfulMobilityDB,
    phase_a: int,
    phase_b: int,
) -> PhaseBoundaryProperty | None:
    return db.pairs.get((int(phase_a), int(phase_b)))


def arrhenius_boundary_mobility(
    base_mobility: float,
    activation_energy: float,
    *,
    temperature_c: float,
) -> float:
    temperature_k = float(temperature_c) + 273.15
    if temperature_k <= 0.0:
        return 0.0
    return float(base_mobility) * math.exp(-(float(activation_energy)) / (_GBM_R_GAS_CONSTANT * temperature_k))


def misorientation_mobility_reduction(misorientation_degrees: float) -> float:
    theta_ha = 15.0
    exponent = 4
    scale = 5.0
    misorientation = abs(float(misorientation_degrees))
    if misorientation > 90.0:
        misorientation = 180.0 - misorientation
    if misorientation <= theta_ha:
        reduced = misorientation / theta_ha
        reduced = scale * (reduced**exponent)
        return 1.0 - math.exp(-reduced)
    return 1.0


def boundary_segment_mobility(
    db: FaithfulMobilityDB,
    phase_a: int,
    phase_b: int,
    *,
    temperature_c: float,
    misorientation_degrees: float | None = None,
    default_mobility: float = 1.0,
) -> float:
    prop = get_phase_boundary_property(db, phase_a, phase_b)
    if prop is None:
        mobility = float(default_mobility)
    else:
        mobility = arrhenius_boundary_mobility(
            prop.mobility,
            prop.activation_energy,
            temperature_c=temperature_c,
        )
    if misorientation_degrees is not None:
        mobility *= misorientation_mobility_reduction(float(misorientation_degrees))
    return float(mobility)


def caxis_from_euler3(alpha: float, beta: float, gamma: float) -> tuple[float, float, float]:
    alpha_rad = math.radians(float(alpha))
    beta_rad = math.radians(float(beta))
    return (
        math.sin(beta_rad) * math.cos(alpha_rad),
        -math.sin(beta_rad) * math.sin(alpha_rad),
        math.cos(beta_rad),
    )


def caxis_misorientation_degrees(
    euler_a: tuple[float, float, float],
    euler_b: tuple[float, float, float],
) -> float:
    axis_a = caxis_from_euler3(*euler_a)
    axis_b = caxis_from_euler3(*euler_b)
    dot = (axis_a[0] * axis_b[0]) + (axis_a[1] * axis_b[1]) + (axis_a[2] * axis_b[2])
    dot = max(min(float(dot), 1.0), -1.0)
    if abs(dot - 1.0) < 1.0e-4:
        misorientation = 0.0
    else:
        misorientation = abs(math.degrees(math.acos(dot)))
    if misorientation > 90.0:
        misorientation = 180.0 - misorientation
    return float(misorientation)
