from __future__ import annotations

import argparse
import json
from pathlib import Path

from elle_jax_model.fft_bridge import (
    build_legacy_elle2fft_bridge_payload,
    compare_legacy_elle2fft_bridge_payload,
    diagnose_legacy_elle2fft_header_sources,
    load_legacy_elle2fft_bridge_payload,
)
from elle_jax_model.gbm_faithful import build_faithful_gbm_setup


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare a faithful ELLE runtime state against a legacy make.out/temp.out bridge snapshot.",
    )
    parser.add_argument("--init-elle", required=True, help="Seed ELLE file to load into the faithful runtime.")
    parser.add_argument(
        "--reference-dir",
        required=True,
        help="Directory containing legacy make.out and temp.out files.",
    )
    parser.add_argument(
        "--attribute",
        default="auto",
        help="Seed label attribute to use when loading the faithful ELLE state.",
    )
    parser.add_argument(
        "--json-out",
        default="",
        help="Optional path for a JSON comparison report.",
    )
    args = parser.parse_args()

    setup = build_faithful_gbm_setup(args.init_elle, init_elle_attribute=args.attribute)
    candidate = build_legacy_elle2fft_bridge_payload(setup.mesh_seed)
    reference = load_legacy_elle2fft_bridge_payload(args.reference_dir)
    report = compare_legacy_elle2fft_bridge_payload(candidate, reference)
    report["grain_header_source_diagnostics"] = diagnose_legacy_elle2fft_header_sources(
        setup.mesh_seed,
        reference,
    )
    report.update(
        {
            "init_elle": str(Path(args.init_elle)),
            "reference_dir": str(Path(args.reference_dir)),
        }
    )
    print(json.dumps(report, indent=2, sort_keys=True))

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
