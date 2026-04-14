from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

from elle_jax_model.artifacts import save_snapshot_artifacts, snapshot_statistics
from elle_jax_model.mesh import MeshRelaxationConfig, build_mesh_state, relax_mesh_state
from elle_jax_model.topology import TopologyTracker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a saved order-parameter snapshot into simple viewable artifacts"
    )
    parser.add_argument("snapshot", type=Path, help="Path to an order_parameter_XXXXX.npy file")
    parser.add_argument("--outdir", type=Path, help="Directory for rendered artifacts")
    parser.add_argument("--step", type=int, help="Override the step number used in output names")
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument("--save-elle", action="store_true")
    parser.add_argument("--track-topology", action="store_true")
    parser.add_argument("--mesh-relax-steps", type=int, default=0)
    parser.add_argument("--mesh-topology-steps", type=int, default=0)
    parser.add_argument("--mesh-random-seed", type=int, default=0)
    return parser.parse_args()


def infer_step(path: Path) -> int:
    match = re.search(r"(\d+)$", path.stem)
    return int(match.group(1)) if match else 0


def main() -> None:
    args = parse_args()
    phi = np.load(args.snapshot)

    outdir = args.outdir if args.outdir is not None else args.snapshot.parent
    step = args.step if args.step is not None else infer_step(args.snapshot)
    track_topology = args.track_topology or args.save_elle
    topology_snapshot = TopologyTracker().update(phi, step) if track_topology else None
    mesh_enabled = args.mesh_relax_steps > 0 or args.mesh_topology_steps > 0
    mesh_state = None
    if mesh_enabled:
        mesh_state = relax_mesh_state(
            build_mesh_state(phi, tracked_topology=topology_snapshot),
            MeshRelaxationConfig(
                steps=args.mesh_relax_steps,
                topology_steps=args.mesh_topology_steps,
                random_seed=args.mesh_random_seed,
            ),
        )
    written = save_snapshot_artifacts(
        outdir,
        step,
        phi,
        save_order_parameter=False,
        save_preview=not args.no_preview,
        save_elle=args.save_elle,
        tracked_topology=topology_snapshot,
        save_topology=track_topology,
        mesh_state=mesh_state,
        save_mesh=mesh_enabled,
    )

    stats = snapshot_statistics(phi, step=step)
    print(
        f"rendered step {step:05d}: active_grains={stats['active_grains']} "
        f"boundary_fraction={stats['boundary_fraction']:.4f}"
    )
    for name, path in written.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
