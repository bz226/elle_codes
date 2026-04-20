from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from elle_jax_model.fft_bridge import (
    LegacyFFTImportOptions,
    apply_legacy_fft_snapshot_to_mesh_state,
    compare_applied_legacy_fft_snapshot_to_mesh_state,
    load_legacy_fft_snapshot,
)
from elle_jax_model.gbm_faithful import build_faithful_gbm_setup


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate one frozen legacy FFT -> ELLE mechanics import against the faithful runtime state"
    )
    parser.add_argument("--init-elle", type=Path, required=True, help="Initial ELLE file used to seed the faithful runtime state")
    parser.add_argument(
        "--mechanics-snapshot-dir",
        type=Path,
        required=True,
        help="Directory containing one legacy FFT bridge snapshot",
    )
    parser.add_argument(
        "--skip-dd-import",
        action="store_true",
        help="Mirror ImportDDs=0 by skipping geometrical dislocation-density increments from tex.out",
    )
    parser.add_argument(
        "--exclude-phase-id",
        type=int,
        default=0,
        help="Mirror ExcludePhaseID by zeroing imported DD increments in the selected phase",
    )
    parser.add_argument(
        "--mechanics-density-update-mode",
        choices=("increment", "overwrite"),
        default="increment",
        help="Choose whether tex.out DD values increment the current U_DISLOCDEN field or overwrite it, mirroring different legacy fft2elle branches",
    )
    parser.add_argument(
        "--mechanics-host-repair-mode",
        choices=("fs_check_unodes", "check_error"),
        default="fs_check_unodes",
        help="Choose whether post-import host repair follows the later FS_CheckUnodes path or the older check_error path",
    )
    parser.add_argument("--json-out", type=Path, help="Optional path to write the bridge comparison report JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup = build_faithful_gbm_setup(args.init_elle)
    before_mesh_state = copy.deepcopy(setup.mesh_seed)
    after_mesh_state = copy.deepcopy(setup.mesh_seed)
    snapshot = load_legacy_fft_snapshot(args.mechanics_snapshot_dir)
    import_options = LegacyFFTImportOptions(
        import_dislocation_densities=not bool(args.skip_dd_import),
        exclude_phase_id=int(args.exclude_phase_id),
        density_update_mode=str(args.mechanics_density_update_mode),
        host_repair_mode=str(args.mechanics_host_repair_mode),
    )
    after_mesh_state, apply_stats = apply_legacy_fft_snapshot_to_mesh_state(
        after_mesh_state,
        snapshot,
        import_options=import_options,
    )
    report = compare_applied_legacy_fft_snapshot_to_mesh_state(
        before_mesh_state,
        after_mesh_state,
        snapshot,
        import_options=import_options,
    )
    report["apply_stats"] = apply_stats

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print(f"wrote FFT -> ELLE bridge report: {args.json_out}")

    print(f"mechanics import contract match: {report['mechanics_import_contract_match']}")
    print(
        "subcontracts: "
        f"euler={report['euler_contract_match']} "
        f"positions={report['position_contract_match']} "
        f"cell={report['cell_reset_contract_match']} "
        f"tex={report['tex_contract_match']} "
        f"density={report['density_contract_match']} "
        f"runtime_snapshot={report['runtime_snapshot_contract_match']}"
    )


if __name__ == "__main__":
    main()
