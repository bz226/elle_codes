from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from elle_jax_model.elle_phasefield import (
    EllePhaseFieldConfig,
    load_elle_phasefield_state,
    run_elle_phasefield_simulation,
    save_elle_phasefield_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a direct Python/JAX port of ELLE's phasefield process")
    parser.add_argument("--nx", type=int, default=300)
    parser.add_argument("--ny", type=int, default=300)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--latent-heat", type=float, default=1.8)
    parser.add_argument("--tau", type=float, default=0.0003)
    parser.add_argument("--eps", type=float, default=0.01)
    parser.add_argument("--delta", type=float, default=0.02)
    parser.add_argument("--angle0", type=float, default=1.57)
    parser.add_argument("--aniso", type=float, default=6.0)
    parser.add_argument("--alpha", type=float, default=0.9)
    parser.add_argument("--gamma", type=float, default=10.0)
    parser.add_argument("--teq", type=float, default=1.0)
    parser.add_argument("--spatial-step", type=float, default=0.03)
    parser.add_argument("--dt", type=float, default=2.0e-4)
    parser.add_argument("--initial-radius-sq", type=float, default=10.0)
    parser.add_argument("--initial-temperature", type=float, default=0.0)
    parser.add_argument("--input-elle", type=Path)
    parser.add_argument("--save-elle", action="store_true")
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument("--outdir", type=Path, default=Path("python_jax_model/output_phasefield"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    config = EllePhaseFieldConfig(
        nx=args.nx,
        ny=args.ny,
        latent_heat=args.latent_heat,
        tau=args.tau,
        eps=args.eps,
        delta=args.delta,
        angle0=args.angle0,
        aniso=args.aniso,
        alpha=args.alpha,
        gamma=args.gamma,
        teq=args.teq,
        spatial_step=args.spatial_step,
        dt=args.dt,
        initial_radius_sq=args.initial_radius_sq,
        initial_temperature=args.initial_temperature,
    )

    initial_state = None
    elle_template = None
    if args.input_elle is not None:
        theta_init, temperature_init, elle_template = load_elle_phasefield_state(args.input_elle)
        initial_state = (theta_init, temperature_init)
        config = replace(config, nx=int(theta_init.shape[0]), ny=int(theta_init.shape[1]))

    def save_snapshot(step: int, theta, temperature) -> None:
        written = save_elle_phasefield_artifacts(
            args.outdir,
            step,
            theta,
            temperature,
            save_preview=not args.no_preview,
            save_elle=args.save_elle,
            elle_template=elle_template,
        )
        names = ", ".join(path.name for path in written.values())
        print(f"saved step {step:05d}: {names}")

    _, _, snapshots = run_elle_phasefield_simulation(
        config=config,
        steps=args.steps,
        save_every=args.save_every,
        on_snapshot=save_snapshot,
        initial_state=initial_state,
    )
    print(f"done: wrote {len(snapshots)} snapshots")


if __name__ == "__main__":
    main()
