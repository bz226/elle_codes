from __future__ import annotations

import argparse
from pathlib import Path

from elle_jax_model.artifacts import save_snapshot_artifacts
from elle_jax_model.mesh import (
    MeshFeedbackConfig,
    MeshRelaxationConfig,
    build_mesh_state,
    relax_mesh_state,
)
from elle_jax_model.simulation import GrainGrowthConfig
from elle_jax_model.topology import run_simulation_with_topology, write_topology_history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run JAX grain-growth prototype")
    parser.add_argument("--nx", type=int, default=128)
    parser.add_argument("--ny", type=int, default=128)
    parser.add_argument("--num-grains", type=int, default=12)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--mobility", type=float, default=1.0)
    parser.add_argument("--gradient-penalty", type=float, default=1.0)
    parser.add_argument("--interaction-strength", type=float, default=2.0)
    parser.add_argument("--init-mode", choices=("random", "voronoi"), default="voronoi")
    parser.add_argument("--init-smoothing-steps", type=int, default=2)
    parser.add_argument("--init-noise", type=float, default=0.02)
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument("--save-elle", action="store_true")
    parser.add_argument("--track-topology", action="store_true")
    parser.add_argument("--mesh-relax-steps", type=int, default=0)
    parser.add_argument("--mesh-topology-steps", type=int, default=0)
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

    config = GrainGrowthConfig(
        nx=args.nx,
        ny=args.ny,
        num_grains=args.num_grains,
        dt=args.dt,
        mobility=args.mobility,
        gradient_penalty=args.gradient_penalty,
        interaction_strength=args.interaction_strength,
        seed=args.seed,
        init_mode=args.init_mode,
        init_smoothing_steps=args.init_smoothing_steps,
        init_noise=args.init_noise,
    )

    track_topology = args.track_topology or args.save_elle
    mesh_config = MeshRelaxationConfig(
        steps=args.mesh_relax_steps,
        topology_steps=args.mesh_topology_steps,
        random_seed=args.mesh_random_seed,
    )
    mesh_feedback = None
    enable_mesh_feedback = (
        args.mesh_feedback_every > 0
        and (args.mesh_feedback_strength > 0.0 or args.mesh_transport_strength > 0.0)
    )
    enable_mesh_kernel = args.mesh_kernel_every > 0 and args.mesh_kernel_strength > 0.0
    mesh_enabled = (
        args.mesh_relax_steps > 0
        or args.mesh_topology_steps > 0
        or enable_mesh_feedback
        or enable_mesh_kernel
    )
    if enable_mesh_feedback or enable_mesh_kernel:
        mesh_feedback = MeshFeedbackConfig(
            every=args.mesh_feedback_every,
            strength=args.mesh_feedback_strength,
            transport_strength=args.mesh_transport_strength,
            kernel_advection_every=args.mesh_kernel_every,
            kernel_advection_strength=args.mesh_kernel_strength,
            kernel_predictor_corrector=args.mesh_kernel_corrector,
            boundary_width=args.mesh_feedback_boundary_width,
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
