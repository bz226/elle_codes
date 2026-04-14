from __future__ import annotations

import argparse
from pathlib import Path

from elle_jax_model.elle_phasefield import EllePhaseFieldConfig
from elle_jax_model.phasefield_compare import (
    compare_elle_phasefield_sequences,
    inspect_elle_phasefield_binary,
    run_original_elle_phasefield_sequence,
    run_python_elle_phasefield_sequence,
    write_phasefield_comparison_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Python ELLE phasefield snapshots and optionally compare them against the original binary"
    )
    parser.add_argument("--input-elle", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--python-outdir", type=Path, default=Path("python_jax_model/validation/python"))
    parser.add_argument("--original-outdir", type=Path, default=Path("python_jax_model/validation/original"))
    parser.add_argument("--json-out", type=Path, default=Path("python_jax_model/validation/report.json"))
    parser.add_argument("--binary", type=Path, default=Path("binwx/elle_phasefield"))
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = EllePhaseFieldConfig(
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
    )

    python_run = run_python_elle_phasefield_sequence(
        args.input_elle,
        args.python_outdir,
        config=config,
        steps=args.steps,
        save_every=args.save_every,
    )
    binary_status = inspect_elle_phasefield_binary(args.binary)
    original_run = run_original_elle_phasefield_sequence(
        args.binary,
        args.input_elle,
        args.original_outdir,
        steps=args.steps,
        save_every=args.save_every,
    )

    report = {
        "input_elle": str(args.input_elle),
        "python_run": python_run,
        "binary_status": binary_status,
        "original_run": original_run,
    }

    if original_run.get("ran"):
        report["sequence_comparison"] = compare_elle_phasefield_sequences(
            args.original_outdir,
            args.python_outdir,
        )

    write_phasefield_comparison_report(args.json_out, report)
    print(f"wrote validation report: {args.json_out}")
    print(f"python snapshots: {len(python_run['snapshots'])}")
    print(f"binary ready: {int(bool(binary_status['ready']))}")
    if original_run.get("ran"):
        summary = report["sequence_comparison"]["summary"]
        print(
            "sequence comparison: "
            f"matched_steps={summary['num_matched_steps']} "
            f"theta_rmse_mean={summary['theta_rmse_mean']:.6e} "
            f"temperature_rmse_mean={summary['temperature_rmse_mean']:.6e}"
        )
    else:
        missing = ", ".join(binary_status["missing_libraries"]) if binary_status["missing_libraries"] else "none"
        print(f"original binary unavailable: missing_libraries={missing}")


if __name__ == "__main__":
    main()
