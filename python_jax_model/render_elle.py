from __future__ import annotations

import argparse
from pathlib import Path

from elle_jax_model.elle_visualize import render_elle_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render an ELLE file into a showelle-like preview image")
    parser.add_argument("elle_file", type=Path, help="Path to an .elle file")
    parser.add_argument("--out", type=Path, help="Output PPM path")
    parser.add_argument("--attribute", default="auto", help="Attribute to render, e.g. CONC_A or U_ATTRIB_A")
    parser.add_argument("--palette", choices=("auto", "gray", "heat", "labels"), default="auto")
    parser.add_argument("--scale", type=int, default=1, help="Nearest-neighbor scale factor for the output image")
    parser.add_argument("--legend", action="store_true", help="Append a scalar legend for gray/heat palettes")
    parser.add_argument(
        "--label-flynns",
        action="store_true",
        help="Draw flynn IDs at approximate region centroids",
    )
    parser.add_argument("--showelle-in", type=Path, help="Optional showelle.in file to reuse attribute/range settings")
    parser.add_argument("--no-boundaries", action="store_true", help="Disable flynn-boundary overlay")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = render_elle_file(
        args.elle_file,
        outpath=args.out,
        attribute=args.attribute,
        palette=args.palette,
        scale=args.scale,
        legend=args.legend,
        label_flynns=True if args.label_flynns else None,
        showelle_in=args.showelle_in,
        overlay_boundaries=not args.no_boundaries,
    )
    print(
        f"rendered {result['elle_path']} -> {result['outpath']} "
        f"attribute={result['attribute']} palette={result['palette']} "
        f"grid={tuple(result['grid_shape'])} image={tuple(result['image_shape'])} "
        f"boundaries={int(result['overlay_boundaries'])} "
        f"labels={int(result['flynn_labels'])} legend={int(result['legend'])} scale={result['scale']}"
    )


if __name__ == "__main__":
    main()
