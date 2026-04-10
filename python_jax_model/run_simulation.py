from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from elle_jax_model.simulation import GrainGrowthConfig, run_simulation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run JAX grain-growth prototype")
    parser.add_argument("--nx", type=int, default=128)
    parser.add_argument("--ny", type=int, default=128)
    parser.add_argument("--num-grains", type=int, default=12)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--outdir", type=Path, default=Path("python_jax_model/output"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    config = GrainGrowthConfig(
        nx=args.nx,
        ny=args.ny,
        num_grains=args.num_grains,
        seed=args.seed,
    )

    def save_snapshot(step: int, phi) -> None:
        outpath = args.outdir / f"order_parameter_{step:05d}.npy"
        np.save(outpath, np.array(phi))
        print(f"saved {outpath}")

    _, snapshots = run_simulation(
        config=config,
        steps=args.steps,
        save_every=args.save_every,
        on_snapshot=save_snapshot,
    )
    print(f"done: wrote {len(snapshots)} snapshots")


if __name__ == "__main__":
    main()
