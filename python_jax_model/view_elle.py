from __future__ import annotations

import argparse
from pathlib import Path

from portable_elle_viewer import write_viewer_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a portable interactive HTML viewer for an ELLE file")
    parser.add_argument("elle_file", type=Path, help="Path to an .elle file")
    parser.add_argument("--out", type=Path, help="Output HTML path")
    parser.add_argument("--attribute", default="auto", help="Attribute to render, e.g. CONC_A or U_ATTRIB_A")
    parser.add_argument("--palette", choices=("auto", "gray", "heat", "labels"), default="auto")
    parser.add_argument("--scale", type=int, default=2, help="Initial zoom factor for the HTML viewer")
    parser.add_argument("--no-legend", action="store_true", help="Disable the initial scalar legend state")
    parser.add_argument("--label-flynns", action="store_true", help="Enable flynn labels by default")
    parser.add_argument("--showelle-in", type=Path, help="Optional showelle.in file to reuse attribute/range settings")
    parser.add_argument("--no-boundaries", action="store_true", help="Disable flynn-boundary overlay by default")
    parser.add_argument("--single-file", action="store_true", help="Embed the data directly into the HTML file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = write_viewer_bundle(
        args.elle_file,
        outpath=args.out,
        attribute=args.attribute,
        palette=args.palette,
        scale=args.scale,
        legend=not args.no_legend,
        label_flynns=True if args.label_flynns else None,
        showelle_in=args.showelle_in,
        overlay_boundaries=not args.no_boundaries,
        single_file=args.single_file,
    )
    summary = (
        f"viewer {result['elle_path']} -> {result['outpath']} "
        f"attribute={result['attribute']} palette={result['palette']} "
        f"grid={tuple(result['grid_shape'])} flynns={result['num_flynns']} "
        f"boundaries={int(result['overlay_boundaries'])} "
        f"labels={int(result['flynn_labels'])} legend={int(result['legend'])} scale={result['scale']}"
    )
    if result["data_outpath"] is not None:
        summary += f" data={result['data_outpath']}"
    print(summary)


if __name__ == "__main__":
    main()
