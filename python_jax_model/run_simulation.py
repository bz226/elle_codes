from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from elle_jax_model.artifacts import save_snapshot_artifacts
from elle_jax_model.calibration import (
    BEST_KNOWN_FINE_FOAM_CALIBRATED_PRESET,
    BEST_KNOWN_FINE_FOAM_PARAMS,
    BEST_KNOWN_FINE_FOAM_PRESET,
    BEST_KNOWN_FINE_FOAM_TRUTHFUL_MESH_PRESET,
)
from elle_jax_model.gbm_faithful import FAITHFUL_GBM_DEFAULTS
from elle_jax_model.mesh import (
    MeshFeedbackConfig,
    MeshRelaxationConfig,
    build_mesh_state,
    load_elle_mesh_seed,
    relax_mesh_state,
)
from elle_jax_model.simulation import GrainGrowthConfig
from elle_jax_model.simulation import load_elle_label_seed
from elle_jax_model.topology import run_simulation_with_topology, write_topology_history


def resolve_runtime_preset(
    preset: str,
    *,
    dt: float,
    mobility: float,
    gradient_penalty: float,
    interaction_strength: float,
    init_smoothing_steps: int,
    init_noise: float,
) -> dict[str, float | int]:
    params = {
        "dt": float(dt),
        "mobility": float(mobility),
        "gradient_penalty": float(gradient_penalty),
        "interaction_strength": float(interaction_strength),
        "init_smoothing_steps": int(init_smoothing_steps),
        "init_noise": float(init_noise),
    }
    if preset == "none":
        return params
    if preset in {"fine-foam-best", "fine-foam-calibrated"}:
        return dict(BEST_KNOWN_FINE_FOAM_CALIBRATED_PRESET)
    if preset == "gbm-faithful-default":
        return {
            "dt": float(dt),
            "mobility": float(mobility),
            "gradient_penalty": float(gradient_penalty),
            "interaction_strength": float(interaction_strength),
            "init_smoothing_steps": 0,
            "init_noise": 0.0,
        }
    if preset == "fine-foam-truthful-mesh":
        return {
            key: BEST_KNOWN_FINE_FOAM_TRUTHFUL_MESH_PRESET[key]
            for key in ("dt", "mobility", "gradient_penalty", "interaction_strength", "init_smoothing_steps", "init_noise")
        }
    raise ValueError(f"unsupported preset: {preset}")


def resolve_mesh_preset(
    preset: str,
    *,
    mesh_relax_steps: int,
    mesh_topology_steps: int,
    mesh_movement_model: str,
    mesh_surface_diagonal_trials: bool,
    mesh_use_elle_physical_units: bool,
    mesh_update_mode: str,
    mesh_random_seed: int,
    mesh_feedback_every: int,
    mesh_feedback_strength: float,
    mesh_transport_strength: float,
    mesh_kernel_every: int,
    mesh_kernel_strength: float,
    mesh_kernel_corrector: bool,
    mesh_feedback_boundary_width: int,
) -> dict[str, float | int | bool]:
    params: dict[str, float | int | bool] = {
        "mesh_relax_steps": int(mesh_relax_steps),
        "mesh_topology_steps": int(mesh_topology_steps),
        "mesh_movement_model": str(mesh_movement_model),
        "mesh_surface_diagonal_trials": bool(mesh_surface_diagonal_trials),
        "mesh_use_elle_physical_units": bool(mesh_use_elle_physical_units),
        "mesh_update_mode": str(mesh_update_mode),
        "mesh_random_seed": int(mesh_random_seed),
        "mesh_feedback_every": int(mesh_feedback_every),
        "mesh_feedback_strength": float(mesh_feedback_strength),
        "mesh_transport_strength": float(mesh_transport_strength),
        "mesh_kernel_every": int(mesh_kernel_every),
        "mesh_kernel_strength": float(mesh_kernel_strength),
        "mesh_kernel_corrector": bool(mesh_kernel_corrector),
        "mesh_feedback_boundary_width": int(mesh_feedback_boundary_width),
    }
    if preset in {"none", "fine-foam-best", "fine-foam-calibrated"}:
        return params
    if preset == "gbm-faithful-default":
        return {
            "mesh_relax_steps": int(FAITHFUL_GBM_DEFAULTS["motion_passes"]),
            "mesh_topology_steps": int(FAITHFUL_GBM_DEFAULTS["topology_passes"]),
            "mesh_movement_model": str(FAITHFUL_GBM_DEFAULTS["movement_model"]),
            "mesh_surface_diagonal_trials": bool(FAITHFUL_GBM_DEFAULTS["use_diagonal_trials"]),
            "mesh_use_elle_physical_units": bool(FAITHFUL_GBM_DEFAULTS["use_elle_physical_units"]),
            "mesh_update_mode": "mesh_only",
            "mesh_random_seed": int(FAITHFUL_GBM_DEFAULTS["random_seed"]),
            "mesh_feedback_every": int(FAITHFUL_GBM_DEFAULTS["stage_interval"]),
            "mesh_feedback_strength": 0.0,
            "mesh_transport_strength": 0.0,
            "mesh_kernel_every": 0,
            "mesh_kernel_strength": 0.0,
            "mesh_kernel_corrector": False,
            "mesh_feedback_boundary_width": int(FAITHFUL_GBM_DEFAULTS["raster_boundary_band"]),
        }
    if preset == "fine-foam-truthful-mesh":
        return {
            key: BEST_KNOWN_FINE_FOAM_TRUTHFUL_MESH_PRESET[key]
            for key in (
                "mesh_relax_steps",
                "mesh_topology_steps",
                "mesh_movement_model",
                "mesh_surface_diagonal_trials",
                "mesh_use_elle_physical_units",
                "mesh_update_mode",
                "mesh_random_seed",
                "mesh_feedback_every",
                "mesh_feedback_strength",
                "mesh_transport_strength",
                "mesh_kernel_every",
                "mesh_kernel_strength",
                "mesh_kernel_corrector",
                "mesh_feedback_boundary_width",
            )
        }
    raise ValueError(f"unsupported preset: {preset}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run JAX grain-growth prototype")
    parser.add_argument("--nx", type=int, default=128)
    parser.add_argument("--ny", type=int, default=128)
    parser.add_argument("--num-grains", type=int, default=12)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--preset",
        choices=(
            "none",
            "fine-foam-best",
            "fine-foam-calibrated",
            "gbm-faithful-default",
            "fine-foam-truthful-mesh",
        ),
        default="none",
    )
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--mobility", type=float, default=1.0)
    parser.add_argument("--gradient-penalty", type=float, default=1.0)
    parser.add_argument("--interaction-strength", type=float, default=2.0)
    parser.add_argument("--init-mode", choices=("random", "voronoi", "elle"), default="voronoi")
    parser.add_argument("--init-elle", type=Path, help="Seed the simulation from an ELLE file via an unode attribute")
    parser.add_argument("--init-elle-attribute", default="auto")
    parser.add_argument("--init-smoothing-steps", type=int, default=2)
    parser.add_argument("--init-noise", type=float, default=0.02)
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument("--save-elle", action="store_true")
    parser.add_argument("--track-topology", action="store_true")
    parser.add_argument("--mesh-relax-steps", type=int, default=0)
    parser.add_argument("--mesh-topology-steps", type=int, default=0)
    parser.add_argument("--mesh-movement-model", choices=("legacy", "elle_surface"), default="legacy")
    parser.add_argument("--mesh-surface-diagonal-trials", action="store_true")
    parser.add_argument("--mesh-use-elle-physical-units", action="store_true")
    parser.add_argument("--mesh-update-mode", choices=("blend", "mesh_only"), default="blend")
    parser.add_argument("--mesh-random-seed", type=int, default=0)
    parser.add_argument("--mesh-feedback-every", type=int, default=0)
    parser.add_argument("--mesh-feedback-strength", type=float, default=0.2)
    parser.add_argument("--mesh-transport-strength", type=float, default=1.0)
    parser.add_argument("--mesh-kernel-every", type=int, default=0)
    parser.add_argument("--mesh-kernel-strength", type=float, default=0.0)
    parser.add_argument("--mesh-kernel-corrector", action="store_true")
    parser.add_argument("--mesh-feedback-boundary-width", type=int, default=1)
    parser.add_argument("--outdir", type=Path, default=Path("python_jax_model/output"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    runtime_params = resolve_runtime_preset(
        args.preset,
        dt=args.dt,
        mobility=args.mobility,
        gradient_penalty=args.gradient_penalty,
        interaction_strength=args.interaction_strength,
        init_smoothing_steps=args.init_smoothing_steps,
        init_noise=args.init_noise,
    )
    mesh_params = resolve_mesh_preset(
        args.preset,
        mesh_relax_steps=args.mesh_relax_steps,
        mesh_topology_steps=args.mesh_topology_steps,
        mesh_movement_model=args.mesh_movement_model,
        mesh_surface_diagonal_trials=args.mesh_surface_diagonal_trials,
        mesh_use_elle_physical_units=args.mesh_use_elle_physical_units,
        mesh_update_mode=args.mesh_update_mode,
        mesh_random_seed=args.mesh_random_seed,
        mesh_feedback_every=args.mesh_feedback_every,
        mesh_feedback_strength=args.mesh_feedback_strength,
        mesh_transport_strength=args.mesh_transport_strength,
        mesh_kernel_every=args.mesh_kernel_every,
        mesh_kernel_strength=args.mesh_kernel_strength,
        mesh_kernel_corrector=args.mesh_kernel_corrector,
        mesh_feedback_boundary_width=args.mesh_feedback_boundary_width,
    )
    if args.preset != "none":
        print(
            f"applied preset {args.preset}: "
            f"dt={runtime_params['dt']} mobility={runtime_params['mobility']} "
            f"gradient_penalty={runtime_params['gradient_penalty']} "
            f"interaction_strength={runtime_params['interaction_strength']} "
            f"init_smoothing_steps={runtime_params['init_smoothing_steps']} "
            f"init_noise={runtime_params['init_noise']}"
        )
        if args.preset in {"gbm-faithful-default", "fine-foam-truthful-mesh"}:
            print(
                "applied faithful GBM stage: "
                f"movement_model={mesh_params['mesh_movement_model']} "
                f"motion_passes={mesh_params['mesh_relax_steps']} "
                f"topology_passes={mesh_params['mesh_topology_steps']} "
                f"stage_interval={mesh_params['mesh_feedback_every']} "
                f"raster_boundary_band={mesh_params['mesh_feedback_boundary_width']} "
                f"diagonal_trials={int(bool(mesh_params['mesh_surface_diagonal_trials']))} "
                f"use_elle_physical_units={int(bool(mesh_params['mesh_use_elle_physical_units']))} "
                f"update_mode={mesh_params['mesh_update_mode']}"
            )

    nx = args.nx
    ny = args.ny
    num_grains = args.num_grains
    init_elle_path = None
    elle_mesh_state = None
    elle_mesh_relax_overrides: dict[str, float] = {}
    if args.init_mode == "elle":
        if args.init_elle is None:
            raise ValueError("--init-elle is required when --init-mode elle")
        seed = load_elle_label_seed(args.init_elle, attribute=args.init_elle_attribute)
        nx, ny = seed["grid_shape"]
        num_grains = seed["num_labels"]
        init_elle_path = str(args.init_elle)
        print(
            f"loaded ELLE seed: {args.init_elle.name} "
            f"attribute={seed['attribute']} grid={nx}x{ny} labels={num_grains}"
        )
        elle_mesh_state, elle_mesh_relax_overrides = load_elle_mesh_seed(args.init_elle, seed)
        if elle_mesh_relax_overrides:
            override_text = " ".join(
                f"{key}={value:g}" for key, value in sorted(elle_mesh_relax_overrides.items())
            )
            print(
                f"loaded ELLE mesh seed: flynns={elle_mesh_state['stats']['num_flynns']} "
                f"nodes={elle_mesh_state['stats']['num_nodes']} {override_text}"
            )

    config = GrainGrowthConfig(
        nx=nx,
        ny=ny,
        num_grains=num_grains,
        dt=runtime_params["dt"],
        mobility=runtime_params["mobility"],
        gradient_penalty=runtime_params["gradient_penalty"],
        interaction_strength=runtime_params["interaction_strength"],
        seed=args.seed,
        init_mode=args.init_mode,
        init_elle_path=init_elle_path,
        init_elle_attribute=str(seed["attribute"]) if args.init_mode == "elle" else args.init_elle_attribute,
        init_smoothing_steps=int(runtime_params["init_smoothing_steps"]),
        init_noise=float(runtime_params["init_noise"]),
    )

    track_topology = args.track_topology or args.save_elle
    mesh_config = MeshRelaxationConfig(
        steps=int(mesh_params["mesh_relax_steps"]),
        topology_steps=int(mesh_params["mesh_topology_steps"]),
        movement_model=str(mesh_params["mesh_movement_model"]),
        use_diagonal_trials=bool(mesh_params["mesh_surface_diagonal_trials"]),
        use_elle_physical_units=bool(mesh_params["mesh_use_elle_physical_units"]),
        random_seed=int(mesh_params["mesh_random_seed"]),
    )
    if elle_mesh_relax_overrides:
        mesh_config = replace(
            mesh_config,
            speed_up=float(elle_mesh_relax_overrides.get("speed_up", mesh_config.speed_up)),
            switch_distance=elle_mesh_relax_overrides.get("switch_distance", mesh_config.switch_distance),
            min_node_separation_factor=float(
                elle_mesh_relax_overrides.get(
                    "min_node_separation_factor",
                    mesh_config.min_node_separation_factor,
                )
            ),
            max_node_separation_factor=float(
                elle_mesh_relax_overrides.get(
                    "max_node_separation_factor",
                    mesh_config.max_node_separation_factor,
                )
            ),
        )
    mesh_feedback = None
    enable_mesh_feedback = (
        int(mesh_params["mesh_feedback_every"]) > 0
        and (
            str(mesh_params["mesh_update_mode"]) == "mesh_only"
            or float(mesh_params["mesh_feedback_strength"]) > 0.0
            or float(mesh_params["mesh_transport_strength"]) > 0.0
        )
    )
    enable_mesh_kernel = int(mesh_params["mesh_kernel_every"]) > 0 and float(mesh_params["mesh_kernel_strength"]) > 0.0
    mesh_enabled = (
        int(mesh_params["mesh_relax_steps"]) > 0
        or int(mesh_params["mesh_topology_steps"]) > 0
        or enable_mesh_feedback
        or enable_mesh_kernel
    )
    if enable_mesh_feedback or enable_mesh_kernel:
        mesh_feedback = MeshFeedbackConfig(
            every=int(mesh_params["mesh_feedback_every"]),
            strength=float(mesh_params["mesh_feedback_strength"]),
            transport_strength=float(mesh_params["mesh_transport_strength"]),
            update_mode=str(mesh_params["mesh_update_mode"]),
            kernel_advection_every=int(mesh_params["mesh_kernel_every"]),
            kernel_advection_strength=float(mesh_params["mesh_kernel_strength"]),
            kernel_predictor_corrector=bool(mesh_params["mesh_kernel_corrector"]),
            boundary_width=int(mesh_params["mesh_feedback_boundary_width"]),
            initial_mesh_state=elle_mesh_state,
            relax_config=mesh_config,
        )

    def save_snapshot(step: int, phi, topology_snapshot, mesh_feedback_context=None) -> None:
        mesh_state = None
        if mesh_feedback_context is not None:
            mesh_state = mesh_feedback_context.get("mesh_state")
        elif mesh_enabled:
            mesh_state = relax_mesh_state(
                build_mesh_state(phi, tracked_topology=topology_snapshot),
                mesh_config,
            )
        written = save_snapshot_artifacts(
            args.outdir,
            step,
            phi,
            save_preview=not args.no_preview,
            save_elle=args.save_elle,
            tracked_topology=topology_snapshot,
            save_topology=track_topology,
            mesh_state=mesh_state,
            save_mesh=mesh_enabled,
        )
        names = ", ".join(path.name for path in written.values())
        print(f"saved step {step:05d}: {names}")

    _, snapshots, topology_history = run_simulation_with_topology(
        config=config,
        steps=args.steps,
        save_every=args.save_every,
        on_snapshot=save_snapshot,
        mesh_feedback=mesh_feedback,
    )

    if track_topology:
        history_path = args.outdir / "topology_history.json"
        write_topology_history(history_path, topology_history)
        print(f"saved topology history: {history_path.name}")

    print(f"done: wrote {len(snapshots)} snapshots")


if __name__ == "__main__":
    main()
