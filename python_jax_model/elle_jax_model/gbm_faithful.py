from __future__ import annotations

"""Faithful ELLE GBM translation entrypoints.

This module is the supported parity target for the faithful branch.
It intentionally avoids depending on the archived phase-field runtime.
Prototype modules remain importable only for history and comparison.
"""

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .faithful_config import FaithfulSeedData, FaithfulSolverConfig, load_faithful_seed
from .fft_bridge import (
    FFTMechanicsSnapshot,
    LegacyFFTImportOptions,
    load_legacy_fft_snapshot_sequence,
)
from .faithful_runtime import run_faithful_simulation_with_topology
from .mesh import MeshFeedbackConfig, MeshRelaxationConfig, load_elle_mesh_seed
from .nucleation import NucleationConfig
from .recovery import RecoveryConfig

FAITHFUL_GBM_DEFAULTS = {
    "movement_model": "elle_surface",
    "motion_passes": 1,
    "topology_passes": 1,
    "use_diagonal_trials": True,
    "use_elle_physical_units": True,
    "temperature_c": None,
    "random_seed": 0,
    "stage_interval": 1,
    "subloops_per_snapshot": 1,
    "gbm_steps_per_subloop": 1,
    "nucleation_steps_per_subloop": 0,
    "nucleation_hagb_deg": 5.0,
    "nucleation_min_cluster_unodes": 10,
    "nucleation_parent_area_crit": 5.0e-4,
    "recovery_steps_per_subloop": 0,
    "recovery_hagb_deg": 5.0,
    "recovery_trial_rotation_deg": 0.1,
    "recovery_rotation_mobility_length": 500.0,
    "raster_boundary_band": 1,
    "mechanics_import_dislocation_densities": True,
    "mechanics_exclude_phase_id": 0,
    "mechanics_density_update_mode": "increment",
    "mechanics_host_repair_mode": "fs_check_unodes",
}


@dataclass(frozen=True)
class FaithfulGBMSetup:
    """Concrete runtime setup for the NumPy-first faithful GBM branch."""

    config: FaithfulSolverConfig
    mesh_feedback: MeshFeedbackConfig
    seed_info: FaithfulSeedData
    mesh_seed: dict[str, Any]
    mechanics_snapshot: FFTMechanicsSnapshot | None
    mechanics_snapshots: tuple[FFTMechanicsSnapshot, ...]
    mechanics_import_options: LegacyFFTImportOptions
    mesh_relax_overrides: dict[str, float]
    subloops_per_snapshot: int
    gbm_steps_per_subloop: int
    nucleation_steps_per_subloop: int
    nucleation_config: NucleationConfig
    recovery_steps_per_subloop: int
    recovery_config: RecoveryConfig


def _derive_raster_boundary_band(
    mesh_seed: dict[str, Any],
    fallback: int,
) -> int:
    stats = dict(mesh_seed.get("stats", {}))
    boundary_width = float(stats.get("elle_option_boundarywidth", 0.0))
    unit_length = float(stats.get("elle_option_unitlength", 0.0))
    grid_shape = tuple(int(value) for value in stats.get("grid_shape", (0, 0)))
    if boundary_width <= 0.0 or unit_length <= 0.0 or len(grid_shape) != 2:
        return max(int(fallback), 1)

    grid_scale = min(max(grid_shape[0], 0), max(grid_shape[1], 0))
    if grid_scale <= 0:
        return max(int(fallback), 1)

    band = int(np.ceil((boundary_width / unit_length) * float(grid_scale)))
    return max(band, 1)


def build_faithful_gbm_setup(
    init_elle_path: str | Path,
    *,
    init_elle_attribute: str = "auto",
    seed: int = 0,
    movement_model: str = str(FAITHFUL_GBM_DEFAULTS["movement_model"]),
    motion_passes: int = int(FAITHFUL_GBM_DEFAULTS["motion_passes"]),
    topology_passes: int = int(FAITHFUL_GBM_DEFAULTS["topology_passes"]),
    use_diagonal_trials: bool = bool(FAITHFUL_GBM_DEFAULTS["use_diagonal_trials"]),
    use_elle_physical_units: bool = bool(FAITHFUL_GBM_DEFAULTS["use_elle_physical_units"]),
    temperature_c: float | None = FAITHFUL_GBM_DEFAULTS["temperature_c"],
    random_seed: int = int(FAITHFUL_GBM_DEFAULTS["random_seed"]),
    stage_interval: int = int(FAITHFUL_GBM_DEFAULTS["stage_interval"]),
    subloops_per_snapshot: int = int(FAITHFUL_GBM_DEFAULTS["subloops_per_snapshot"]),
    gbm_steps_per_subloop: int = int(FAITHFUL_GBM_DEFAULTS["gbm_steps_per_subloop"]),
    nucleation_steps_per_subloop: int = int(FAITHFUL_GBM_DEFAULTS["nucleation_steps_per_subloop"]),
    nucleation_hagb_deg: float = float(FAITHFUL_GBM_DEFAULTS["nucleation_hagb_deg"]),
    nucleation_min_cluster_unodes: int = int(FAITHFUL_GBM_DEFAULTS["nucleation_min_cluster_unodes"]),
    nucleation_parent_area_crit: float = float(FAITHFUL_GBM_DEFAULTS["nucleation_parent_area_crit"]),
    recovery_steps_per_subloop: int = int(FAITHFUL_GBM_DEFAULTS["recovery_steps_per_subloop"]),
    recovery_hagb_deg: float = float(FAITHFUL_GBM_DEFAULTS["recovery_hagb_deg"]),
    recovery_trial_rotation_deg: float = float(FAITHFUL_GBM_DEFAULTS["recovery_trial_rotation_deg"]),
    recovery_rotation_mobility_length: float = float(
        FAITHFUL_GBM_DEFAULTS["recovery_rotation_mobility_length"]
    ),
    raster_boundary_band: int | None = None,
    phase_db_path: str | None = None,
    mesh_relax_steps: int | None = None,
    mesh_topology_steps: int | None = None,
    mesh_movement_model: str | None = None,
    mesh_surface_diagonal_trials: bool | None = None,
    mesh_use_elle_physical_units: bool | None = None,
    mesh_random_seed: int | None = None,
    mesh_feedback_every: int | None = None,
    mesh_feedback_boundary_width: int | None = None,
    mechanics_snapshot_dir: str | Path | None = None,
    mechanics_import_dislocation_densities: bool = bool(
        FAITHFUL_GBM_DEFAULTS["mechanics_import_dislocation_densities"]
    ),
    mechanics_exclude_phase_id: int = int(FAITHFUL_GBM_DEFAULTS["mechanics_exclude_phase_id"]),
    mechanics_density_update_mode: str = str(FAITHFUL_GBM_DEFAULTS["mechanics_density_update_mode"]),
    mechanics_host_repair_mode: str = str(FAITHFUL_GBM_DEFAULTS["mechanics_host_repair_mode"]),
) -> FaithfulGBMSetup:
    """Build the original-ELLE-style mesh-only GBM runtime configuration."""

    init_elle_path = Path(init_elle_path)
    if mesh_relax_steps is not None:
        motion_passes = int(mesh_relax_steps)
    if mesh_topology_steps is not None:
        topology_passes = int(mesh_topology_steps)
    if mesh_movement_model is not None:
        movement_model = str(mesh_movement_model)
    if mesh_surface_diagonal_trials is not None:
        use_diagonal_trials = bool(mesh_surface_diagonal_trials)
    if mesh_use_elle_physical_units is not None:
        use_elle_physical_units = bool(mesh_use_elle_physical_units)
    if mesh_random_seed is not None:
        random_seed = int(mesh_random_seed)
    if mesh_feedback_every is not None:
        stage_interval = int(mesh_feedback_every)
    if mesh_feedback_boundary_width is not None:
        raster_boundary_band = int(mesh_feedback_boundary_width)

    seed_info = load_faithful_seed(init_elle_path, attribute=init_elle_attribute)
    mesh_seed, relax_overrides = load_elle_mesh_seed(
        init_elle_path,
        {
            "source_labels": seed_info.source_labels,
            "label_field": seed_info.label_field,
            "grid_shape": seed_info.grid_shape,
            "unode_ids": seed_info.unode_ids,
            "unode_positions": seed_info.unode_positions,
            "unode_grid_indices": seed_info.unode_grid_indices,
            "unode_field_values": seed_info.unode_field_values,
            "unode_field_order": seed_info.unode_field_order,
            "attribute": seed_info.attribute,
        },
    )

    config = FaithfulSolverConfig(
        nx=int(seed_info.grid_shape[0]),
        ny=int(seed_info.grid_shape[1]),
        num_grains=int(seed_info.num_labels),
        seed=int(seed),
        init_elle_path=str(init_elle_path),
        init_elle_attribute=str(seed_info.attribute),
        seed_data=seed_info,
    )

    if raster_boundary_band is None:
        raster_boundary_band = _derive_raster_boundary_band(
            mesh_seed,
            fallback=int(FAITHFUL_GBM_DEFAULTS["raster_boundary_band"]),
        )

    inherited_temperature_c = seed_info.elle_options.get("Temperature", 25.0)
    effective_temperature_c = (
        float(inherited_temperature_c)
        if temperature_c is None
        else float(temperature_c)
    )

    mesh_config = MeshRelaxationConfig(
        steps=int(motion_passes),
        topology_steps=int(topology_passes),
        movement_model=str(movement_model),
        use_diagonal_trials=bool(use_diagonal_trials),
        use_elle_physical_units=bool(use_elle_physical_units),
        temperature_c=float(effective_temperature_c),
        random_seed=int(random_seed),
        phase_db_path=phase_db_path,
    )
    if relax_overrides:
        mesh_config = replace(
            mesh_config,
            time_step=float(relax_overrides.get("time_step", mesh_config.time_step)),
            speed_up=float(relax_overrides.get("speed_up", mesh_config.speed_up)),
            switch_distance=relax_overrides.get("switch_distance", mesh_config.switch_distance),
            min_node_separation_factor=float(
                relax_overrides.get(
                    "min_node_separation_factor",
                    mesh_config.min_node_separation_factor,
                )
            ),
            max_node_separation_factor=float(
                relax_overrides.get(
                    "max_node_separation_factor",
                    mesh_config.max_node_separation_factor,
                )
            ),
        )

    effective_elle_options = seed_info.elle_options.with_scalar_overrides(
        {
            "SwitchDistance": mesh_config.switch_distance,
            "MinNodeSeparation": (
                float(mesh_config.min_node_separation_factor) * float(mesh_config.switch_distance)
                if mesh_config.switch_distance is not None
                else None
            ),
            "MaxNodeSeparation": (
                float(mesh_config.max_node_separation_factor) * float(mesh_config.switch_distance)
                if mesh_config.switch_distance is not None
                else None
            ),
            "SpeedUp": float(mesh_config.speed_up),
            "Timestep": float(mesh_config.time_step),
            "Temperature": float(mesh_config.temperature_c),
        }
    )
    mesh_seed["_runtime_elle_options"] = effective_elle_options.to_runtime_dict()
    for key, value in effective_elle_options.scalar_values.items():
        mesh_seed["stats"][f"elle_option_{str(key).lower()}"] = float(value)

    mesh_feedback = MeshFeedbackConfig(
        every=int(stage_interval),
        strength=0.0,
        transport_strength=0.0,
        update_mode="mesh_only",
        boundary_width=int(raster_boundary_band),
        initial_mesh_state=mesh_seed,
        relax_config=mesh_config,
    )
    mechanics_snapshots = (
        load_legacy_fft_snapshot_sequence(mechanics_snapshot_dir)
        if mechanics_snapshot_dir is not None
        else ()
    )
    mechanics_snapshot = mechanics_snapshots[0] if mechanics_snapshots else None
    return FaithfulGBMSetup(
        config=config,
        mesh_feedback=mesh_feedback,
        seed_info=seed_info,
        mesh_seed=mesh_seed,
        mechanics_snapshot=mechanics_snapshot,
        mechanics_snapshots=tuple(mechanics_snapshots),
        mechanics_import_options=LegacyFFTImportOptions(
            import_dislocation_densities=bool(mechanics_import_dislocation_densities),
            exclude_phase_id=int(mechanics_exclude_phase_id),
            density_update_mode=str(mechanics_density_update_mode),
            host_repair_mode=str(mechanics_host_repair_mode),
        ),
        mesh_relax_overrides=relax_overrides,
        subloops_per_snapshot=int(subloops_per_snapshot),
        gbm_steps_per_subloop=int(gbm_steps_per_subloop),
        nucleation_steps_per_subloop=int(nucleation_steps_per_subloop),
        nucleation_config=NucleationConfig(
            high_angle_boundary_deg=float(nucleation_hagb_deg),
            min_cluster_unodes=int(nucleation_min_cluster_unodes),
            parent_area_crit=float(nucleation_parent_area_crit),
        ),
        recovery_steps_per_subloop=int(recovery_steps_per_subloop),
        recovery_config=RecoveryConfig(
            high_angle_boundary_deg=float(recovery_hagb_deg),
            trial_rotation_deg=float(recovery_trial_rotation_deg),
            rotation_mobility_length=float(recovery_rotation_mobility_length),
        ),
    )


def run_faithful_gbm_simulation(
    *,
    init_elle_path: str | Path | None = None,
    init_elle_attribute: str = "auto",
    steps: int,
    save_every: int,
    on_snapshot: Callable[[int, object, dict[str, Any], dict[str, Any] | None], None] | None = None,
    setup: FaithfulGBMSetup | None = None,
    include_initial_snapshot: bool = False,
    **setup_kwargs: Any,
) -> tuple[object, list[object], list[dict[str, Any]]]:
    """Run the dedicated NumPy-first faithful GBM translation path."""

    if setup is None:
        if init_elle_path is None:
            raise ValueError("init_elle_path is required when setup is not provided")
        setup = build_faithful_gbm_setup(
            init_elle_path,
            init_elle_attribute=init_elle_attribute,
            **setup_kwargs,
        )

    return run_faithful_simulation_with_topology(
        config=setup.config,
        steps=int(steps),
        save_every=int(save_every),
        on_snapshot=on_snapshot,
        mesh_feedback=setup.mesh_feedback,
        mechanics_snapshot=setup.mechanics_snapshot,
        mechanics_snapshots=setup.mechanics_snapshots,
        mechanics_import_options=setup.mechanics_import_options,
        include_initial_snapshot=bool(include_initial_snapshot),
        subloops_per_snapshot=int(setup.subloops_per_snapshot),
        gbm_steps_per_subloop=int(setup.gbm_steps_per_subloop),
        nucleation_steps_per_subloop=int(setup.nucleation_steps_per_subloop),
        nucleation_config=setup.nucleation_config,
        recovery_steps_per_subloop=int(setup.recovery_steps_per_subloop),
        recovery_config=setup.recovery_config,
    )
