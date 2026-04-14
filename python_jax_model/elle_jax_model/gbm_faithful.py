from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .mesh import MeshFeedbackConfig, MeshRelaxationConfig, load_elle_mesh_seed
from .simulation import GrainGrowthConfig, load_elle_label_seed
from .topology import run_simulation_with_topology

FAITHFUL_GBM_DEFAULTS = {
    "movement_model": "elle_surface",
    "motion_passes": 1,
    "topology_passes": 1,
    "use_diagonal_trials": True,
    "use_elle_physical_units": True,
    "random_seed": 0,
    "stage_interval": 1,
    "raster_boundary_band": 1,
}

_LEGACY_PHASEFIELD_DEFAULTS = {
    "dt": 0.05,
    "mobility": 1.0,
    "gradient_penalty": 1.0,
    "interaction_strength": 2.0,
    "init_smoothing_steps": 0,
    "init_noise": 0.0,
}


@dataclass(frozen=True)
class FaithfulGBMSetup:
    """Concrete runtime setup for the NumPy-first faithful GBM branch."""

    config: GrainGrowthConfig
    mesh_feedback: MeshFeedbackConfig
    seed_info: dict[str, object]
    mesh_seed: dict[str, Any]
    mesh_relax_overrides: dict[str, float]


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
    random_seed: int = int(FAITHFUL_GBM_DEFAULTS["random_seed"]),
    stage_interval: int = int(FAITHFUL_GBM_DEFAULTS["stage_interval"]),
    raster_boundary_band: int | None = None,
    dt: float | None = None,
    mobility: float | None = None,
    gradient_penalty: float | None = None,
    interaction_strength: float | None = None,
    init_smoothing_steps: int | None = None,
    init_noise: float | None = None,
    mesh_relax_steps: int | None = None,
    mesh_topology_steps: int | None = None,
    mesh_movement_model: str | None = None,
    mesh_surface_diagonal_trials: bool | None = None,
    mesh_use_elle_physical_units: bool | None = None,
    mesh_random_seed: int | None = None,
    mesh_feedback_every: int | None = None,
    mesh_feedback_strength: float | None = None,
    mesh_transport_strength: float | None = None,
    mesh_feedback_boundary_width: int | None = None,
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

    dt = float(_LEGACY_PHASEFIELD_DEFAULTS["dt"] if dt is None else dt)
    mobility = float(_LEGACY_PHASEFIELD_DEFAULTS["mobility"] if mobility is None else mobility)
    gradient_penalty = float(
        _LEGACY_PHASEFIELD_DEFAULTS["gradient_penalty"] if gradient_penalty is None else gradient_penalty
    )
    interaction_strength = float(
        _LEGACY_PHASEFIELD_DEFAULTS["interaction_strength"]
        if interaction_strength is None
        else interaction_strength
    )
    init_smoothing_steps = int(
        _LEGACY_PHASEFIELD_DEFAULTS["init_smoothing_steps"]
        if init_smoothing_steps is None
        else init_smoothing_steps
    )
    init_noise = float(_LEGACY_PHASEFIELD_DEFAULTS["init_noise"] if init_noise is None else init_noise)
    feedback_strength = 0.0 if mesh_feedback_strength is None else float(mesh_feedback_strength)
    transport_strength = 0.0 if mesh_transport_strength is None else float(mesh_transport_strength)

    seed_info = load_elle_label_seed(init_elle_path, attribute=init_elle_attribute)
    mesh_seed, relax_overrides = load_elle_mesh_seed(init_elle_path, seed_info)

    config = GrainGrowthConfig(
        nx=int(seed_info["grid_shape"][0]),
        ny=int(seed_info["grid_shape"][1]),
        num_grains=int(seed_info["num_labels"]),
        dt=float(dt),
        mobility=float(mobility),
        gradient_penalty=float(gradient_penalty),
        interaction_strength=float(interaction_strength),
        seed=int(seed),
        init_mode="elle",
        init_elle_path=str(init_elle_path),
        init_elle_attribute=str(seed_info["attribute"]),
        init_smoothing_steps=int(init_smoothing_steps),
        init_noise=float(init_noise),
    )

    if raster_boundary_band is None:
        raster_boundary_band = _derive_raster_boundary_band(
            mesh_seed,
            fallback=int(FAITHFUL_GBM_DEFAULTS["raster_boundary_band"]),
        )

    mesh_config = MeshRelaxationConfig(
        steps=int(motion_passes),
        topology_steps=int(topology_passes),
        movement_model=str(movement_model),
        use_diagonal_trials=bool(use_diagonal_trials),
        use_elle_physical_units=bool(use_elle_physical_units),
        random_seed=int(random_seed),
    )
    if relax_overrides:
        mesh_config = replace(
            mesh_config,
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

    mesh_feedback = MeshFeedbackConfig(
        every=int(stage_interval),
        strength=float(feedback_strength),
        transport_strength=float(transport_strength),
        update_mode="mesh_only",
        boundary_width=int(raster_boundary_band),
        initial_mesh_state=mesh_seed,
        relax_config=mesh_config,
    )
    return FaithfulGBMSetup(
        config=config,
        mesh_feedback=mesh_feedback,
        seed_info=seed_info,
        mesh_seed=mesh_seed,
        mesh_relax_overrides=relax_overrides,
    )


def run_faithful_gbm_simulation(
    *,
    init_elle_path: str | Path | None = None,
    init_elle_attribute: str = "auto",
    steps: int,
    save_every: int,
    on_snapshot: Callable[[int, object, dict[str, Any], dict[str, Any] | None], None] | None = None,
    setup: FaithfulGBMSetup | None = None,
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

    return run_simulation_with_topology(
        config=setup.config,
        steps=int(steps),
        save_every=int(save_every),
        on_snapshot=on_snapshot,
        mesh_feedback=setup.mesh_feedback,
    )
