from __future__ import annotations

import argparse
from pathlib import Path

from elle_jax_model.artifacts import save_snapshot_artifacts
from elle_jax_model.gbm_faithful import (
    FAITHFUL_GBM_DEFAULTS,
    build_faithful_gbm_setup,
    run_faithful_gbm_simulation,
)
from elle_jax_model.topology import write_topology_history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the dedicated faithful NumPy-first ELLE GBM translation path"
    )
    parser.add_argument("--init-elle", type=Path, required=True, help="Original ELLE file to seed from")
    parser.add_argument("--init-elle-attribute", default="auto")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--movement-model",
        choices=("legacy", "elle_surface"),
        default=FAITHFUL_GBM_DEFAULTS["movement_model"],
    )
    parser.add_argument(
        "--motion-passes",
        type=int,
        default=FAITHFUL_GBM_DEFAULTS["motion_passes"],
        help="Number of ELLE-style boundary-motion passes per saved stage",
    )
    parser.add_argument(
        "--topology-passes",
        type=int,
        default=FAITHFUL_GBM_DEFAULTS["topology_passes"],
        help="Number of topology-cleanup passes per saved stage",
    )
    parser.add_argument(
        "--stage-interval",
        type=int,
        default=FAITHFUL_GBM_DEFAULTS["stage_interval"],
        help="How often to apply one faithful GBM stage in the outer loop",
    )
    parser.add_argument(
        "--raster-boundary-band",
        type=int,
        default=FAITHFUL_GBM_DEFAULTS["raster_boundary_band"],
        help="Integer raster-band support used only in the rewritten grid ownership path",
    )
    parser.add_argument(
        "--no-diagonal-trials",
        action="store_true",
        help="Disable ELLE-style diagonal trial positions in the faithful mover",
    )
    parser.add_argument(
        "--no-elle-physical-units",
        action="store_true",
        help="Disable UnitLength-aware physical scaling in the faithful mover",
    )
    parser.add_argument("--mesh-relax-steps", dest="motion_passes", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--mesh-topology-steps", dest="topology_passes", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--mesh-movement-model", dest="movement_model", help=argparse.SUPPRESS)
    parser.add_argument("--mesh-feedback-every", dest="stage_interval", type=int, help=argparse.SUPPRESS)
    parser.add_argument(
        "--mesh-feedback-boundary-width",
        dest="raster_boundary_band",
        type=int,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--dt", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--mobility", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--gradient-penalty", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--interaction-strength", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--mesh-feedback-strength", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--mesh-transport-strength", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--save-elle", action="store_true")
    parser.add_argument("--track-topology", action="store_true")
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("python_jax_model/validation/gbm_faithful_output"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    setup = build_faithful_gbm_setup(
        args.init_elle,
        init_elle_attribute=args.init_elle_attribute,
        seed=int(args.seed),
        movement_model=str(args.movement_model),
        motion_passes=int(args.motion_passes),
        topology_passes=int(args.topology_passes),
        stage_interval=int(args.stage_interval),
        raster_boundary_band=int(args.raster_boundary_band),
        use_diagonal_trials=not bool(args.no_diagonal_trials),
        use_elle_physical_units=not bool(args.no_elle_physical_units),
        dt=float(args.dt) if args.dt is not None else None,
        mobility=float(args.mobility) if args.mobility is not None else None,
        gradient_penalty=float(args.gradient_penalty) if args.gradient_penalty is not None else None,
        interaction_strength=float(args.interaction_strength) if args.interaction_strength is not None else None,
        mesh_feedback_strength=(
            float(args.mesh_feedback_strength) if args.mesh_feedback_strength is not None else None
        ),
        mesh_transport_strength=(
            float(args.mesh_transport_strength) if args.mesh_transport_strength is not None else None
        ),
    )

    print(
        "running faithful GBM translation: "
        f"attribute={setup.seed_info['attribute']} "
        f"grid={setup.seed_info['grid_shape'][0]}x{setup.seed_info['grid_shape'][1]} "
        f"labels={setup.seed_info['num_labels']} "
        f"movement_model={setup.mesh_feedback.relax_config.movement_model} "
        f"motion_passes={setup.mesh_feedback.relax_config.steps} "
        f"topology_passes={setup.mesh_feedback.relax_config.topology_steps} "
        f"stage_interval={setup.mesh_feedback.every} "
        f"raster_boundary_band={setup.mesh_feedback.boundary_width} "
        "update_mode=mesh_only"
    )

    save_topology = bool(args.track_topology or args.save_elle)

    def save_snapshot(step: int, phi, topology_snapshot, mesh_feedback_context=None) -> None:
        mesh_state = None if mesh_feedback_context is None else mesh_feedback_context.get("mesh_state")
        written = save_snapshot_artifacts(
            args.outdir,
            step,
            phi,
            save_preview=not args.no_preview,
            save_elle=args.save_elle,
            tracked_topology=topology_snapshot,
            save_topology=save_topology,
            mesh_state=mesh_state,
            save_mesh=True,
        )
        names = ", ".join(path.name for path in written.values())
        print(f"saved step {step:05d}: {names}")

    _, snapshots, topology_history = run_faithful_gbm_simulation(
        steps=int(args.steps),
        save_every=int(args.save_every),
        on_snapshot=save_snapshot,
        setup=setup,
    )

    if save_topology:
        history_path = args.outdir / "topology_history.json"
        write_topology_history(history_path, topology_history)
        print(f"saved topology history: {history_path.name}")

    print(f"done: wrote {len(snapshots)} snapshots with solver_backend=numpy_mesh_only")


if __name__ == "__main__":
    main()
