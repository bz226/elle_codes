# ELLE Python/JAX Prototype Rewrite

This folder contains a **clean-room Python rewrite prototype** inspired by ELLE's grain-growth style process drivers (for example `processes/growth/gg.main.cc`).

It does **not** modify or replace the original C/C++ codebase. Instead, it demonstrates a modern, vectorized modeling path using JAX.

## What this prototype includes

- A small 2D phase-field grain-growth simulator in JAX (with NumPy fallback if JAX is unavailable).
- A direct Python/JAX port of the original `processes/phasefield/phasefield.elle.cc` single-order-parameter process, including latent heat, anisotropic interface width, and the coupled temperature field.
- Reproducible initial-condition generators, including a Voronoi-style grain map that is closer to ELLE-style microstructure layouts.
- A simulation runner that exports raw fields plus derived grain-ID, boundary, stats, and preview artifacts.
- Stable flynn tracking across saved timesteps, with per-step topology snapshots and history.
- A mesh-evolution layer that moves double and triple junctions and applies ELLE-style node-spacing checks after topology extraction.
- A standalone renderer for existing `.npy` snapshots.
- Lightweight unit tests for initialization, simulation, and artifact generation.

## Why this is a practical migration path

A full rewrite of the entire ELLE ecosystem is large, but **yes, it is possible** to incrementally rewrite process-by-process.
This prototype shows one such process-level migration:

1. Keep old C/C++ binaries untouched.
2. Reimplement one physics kernel in Python/JAX.
3. Validate behavior against known ELLE outputs.
4. Expand coverage process-by-process.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r python_jax_model/requirements.txt
python python_jax_model/run_simulation.py --steps 300 --save-every 60 --init-mode voronoi --save-elle --track-topology --mesh-relax-steps 3 --mesh-topology-steps 1 --mesh-kernel-every 5 --mesh-kernel-strength 1.0 --mesh-kernel-corrector --mesh-feedback-every 20 --mesh-transport-strength 1.0 --mesh-feedback-strength 0.2 --outdir python_jax_model/output
```

To run the more faithful direct port of ELLE's original `phasefield` process:

```bash
python python_jax_model/run_elle_phasefield.py --steps 200 --save-every 50 --outdir python_jax_model/output_phasefield
```

To start that direct port from an existing ELLE phasefield file and write updated ELLE unode states back out:

```bash
python python_jax_model/run_elle_phasefield.py --input-elle processes/phasefield/inphase.elle --steps 20 --save-every 10 --save-elle --outdir python_jax_model/output_phasefield
```

To compare two ELLE phasefield states numerically:

```bash
python python_jax_model/compare_elle_phasefield.py processes/phasefield/inphase.elle python_jax_model/output_phasefield_from_elle/phasefield_state_00002.elle --json-out python_jax_model/output_phasefield_from_elle/comparison.json --binary binwx/elle_phasefield
```

To run the Python port over a whole saved sequence and, when the original binary is usable, compare both directories step by step:

```bash
python python_jax_model/validate_elle_phasefield.py --input-elle processes/phasefield/inphase.elle --steps 5 --save-every 1 --python-outdir python_jax_model/validation/python --original-outdir python_jax_model/validation/original --json-out python_jax_model/validation/report.json --binary binwx/elle_phasefield
```

To render an ELLE file into a showelle-like preview image:

```bash
python python_jax_model/render_elle.py processes/phasefield/inphase.elle --showelle-in processes/phasefield/showelle.in
```

For a more inspection-friendly view with zoom, flynn labels, and a scalar legend:

```bash
python python_jax_model/render_elle.py processes/phasefield/inphase.elle --showelle-in processes/phasefield/showelle.in --scale 2 --legend --label-flynns
```

To build a self-contained interactive HTML viewer you can open in a browser:

```bash
python python_jax_model/view_elle.py processes/phasefield/inphase.elle --showelle-in processes/phasefield/showelle.in --scale 2 --label-flynns
```

By default this now writes a small HTML shell plus a sidecar `.data.js` file. Keep both files in the same folder when you move or open them. If you really want one larger self-contained file, add `--single-file`.

There is also a portable standalone script, [portable_elle_viewer.py](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model/portable_elle_viewer.py), that uses only the Python standard library. You can copy that one file to another machine and run:

```bash
python portable_elle_viewer.py your_file.elle --label-flynns
```

## Output

Each saved step now produces a small bundle of files:

- `order_parameter_XXXXX.npy`: raw `(num_grains, nx, ny)` order-parameter tensor
- `grain_ids_XXXXX.npy`: dominant grain label per grid cell
- `boundary_mask_XXXXX.npy`: binary grain-boundary mask
- `grain_stats_XXXXX.json`: per-grain areas and summary statistics
- `grain_preview_XXXXX.ppm`: viewable color image with boundaries overlaid
- `topology_XXXXX.json`: tracked flynn state for that saved step
- `mesh_XXXXX.json`: evolved boundary-network state with node positions, degrees, flynn membership, and topology events
- `grain_unodes_XXXXX.elle`: optional ELLE-style export when `--save-elle` is enabled
- `topology_history.json`: tracked flynn history across all saved steps

The direct `phasefield` port writes a separate bundle:

- `theta_XXXXX.npy`: the original ELLE `CONC_A`-style order parameter field
- `temperature_XXXXX.npy`: the coupled temperature field corresponding to `U_ATTRIB_A`
- `phasefield_stats_XXXXX.json`: summary ranges, solid fraction, and interface fraction
- `theta_preview_XXXXX.ppm`: grayscale view of the phase field
- `temperature_preview_XXXXX.ppm`: false-color temperature view
- `phasefield_state_XXXXX.elle`: optional ELLE-style unode export with `UNODES`, `U_CONC_A`, and `U_ATTRIB_A`

The `.elle` export now includes a first-pass flynn reconstruction from the dominant-grain map:

- `FLYNNS` and `LOCATION` are rebuilt from connected grain-region boundaries on the grid
- `F_ATTRIB_A` stores the dominant grain label for each exported flynn
- `U_ATTRIB_A` stores the dominant grain ID per unode
- `U_ATTRIB_B` stores the dominant order-parameter confidence per unode

To reduce the old limitation where topology was rebuilt independently on every export, the rewrite now tracks flynn identities across saved timesteps. Each tracked flynn gets a stable ID, neighbor list, and event history including births, deaths, splits, and merges. This is still an approximation rather than a full ELLE-native process rewrite: topology is sampled from the phase-field state at save points rather than driving the PDE update directly.

The next layer toward ELLE faithfulness is now present as well: `--mesh-relax-steps` runs ELLE-style double- and triple-junction motion based on curvature radius (`GetRay`) and capped node velocity, while `--mesh-topology-steps` applies ELLE-like topology checks that can switch short triple-triple edges, split overly long boundary segments, and collapse overly short double junctions. The triple-switch path is now also guarded by simple-polygon and minimum-area checks inspired by `ElleRegionIsSimple` and `ElleCheckSmallFlynn`, so unstable local rewires are rejected instead of corrupting the mesh. After local rewires, the mesh pass also performs a conservative cleanup of tiny two-sided flynns by merging the degenerate lens into the stronger neighboring grain and compacting orphaned nodes, which is a closer match to ELLE's post-switch/post-delete cleanup flow. The runtime can now take a stronger coupling step too: `--mesh-feedback-every` uses the relaxed mesh in two ways. First, `--mesh-kernel-every` plus `--mesh-kernel-strength` inject a mesh-derived velocity field directly into the phase-field update law as an advection term, and `--mesh-kernel-corrector` upgrades that path to a predictor-corrector step that re-estimates mesh motion from a predicted post-step state before committing the PDE update. Then `--mesh-transport-strength` turns continuous node motion into a smooth boundary-band transport field, and `--mesh-feedback-strength` rasterizes the mesh back to the grid to blend the order parameters toward mesh-consistent labels where topology has changed. The coupling loop now also keeps a live tracked-topology snapshot internally, so the mesh used during runtime and the topology exported at save points share the same stable flynn IDs instead of being reconstructed independently. This is still short of full ELLE topology maintenance because the transport and kernel forcing are still derived from relaxed rasterized boundaries rather than a native moving front inside a single coupled solver, but the mesh no longer affects only exported artifacts or only discrete cell ownership.

Alongside that multiphase rewrite path, the new `run_elle_phasefield.py` driver is a much closer translation of the original ELLE `phasefield` process itself. It carries over the single `theta` field, the coupled temperature field `T`, the 9-point laplacian, latent-heat source term, and the anisotropic `epsilon(theta)` interface-width logic from the original C++ process. It can now also read original ELLE phasefield `.elle` files through the `UNODES`, `U_CONC_A`, and `U_ATTRIB_A` sections and write those sections back out, so we can start running direct Python/JAX updates from existing ELLE phasefield states instead of only from a synthetic circular seed.

The new `compare_elle_phasefield.py` tool adds the validation side of that workflow. It loads two ELLE phasefield states and reports RMSE, max-absolute error, solid-fraction drift, interface-fraction drift, and solid-region overlap for `theta`, plus the same core error metrics for temperature. If you point it at the original `binwx/elle_phasefield` binary with `--binary`, it also reports whether the local environment has the shared libraries needed to run the original executable.

The new `validate_elle_phasefield.py` script builds on that by generating a Python/JAX snapshot sequence from an input ELLE phasefield state, attempting the same run through the original `elle_phasefield` executable, and then comparing the two sequences step by step when the original binary is available. In environments like this one where the original executable is present but missing runtime libraries, it still writes the Python sequence and a machine-readable report that lists the blocking shared-library dependencies.

The new `render_elle.py` script covers the visualization side. It reads `.elle` files directly, detects common unode attributes like `U_CONC_A`, `U_ATTRIB_A`, and `U_ATTRIB_B`, and writes a static PPM preview with optional flynn-boundary overlay. When you pass `--showelle-in`, it reuses the `Unode_Attribute` selection and numeric range from the original `showelle.in` file, so the preview behavior is much closer to the original ELLE viewing workflow. It also supports nearest-neighbor zoom with `--scale`, scalar legends with `--legend`, and region ID overlays with `--label-flynns`, which makes it much more practical for inspecting reconstructed flynn topology from the Python rewrite.

The new `view_elle.py` script goes one step further and writes a portable browser viewer. To keep the `.html` file small, it now defaults to a split bundle: a reusable HTML shell plus a sibling `.data.js` payload file containing the encoded field and flynn geometry. The browser viewer still gives you zoom, palette switching, boundary toggles, flynn label toggles, scalar legend toggles, and a hover readout for cell values, but the HTML itself stays much smaller than before. When you need a single artifact instead, `--single-file` switches back to an all-in-one HTML export. The separate `portable_elle_viewer.py` script uses only the Python standard library, so it is the easiest thing to copy to another machine when you want to inspect `.elle` files locally without recreating the full rewrite environment.

## Render Existing Snapshots

To generate preview artifacts from an existing `.npy` snapshot:

```bash
python python_jax_model/render_snapshot.py python_jax_model/output/order_parameter_00060.npy
```

To also write an ELLE-style unode file:

```bash
python python_jax_model/render_snapshot.py python_jax_model/output/order_parameter_00060.npy --save-elle --track-topology --mesh-relax-steps 3 --mesh-topology-steps 1
```

## Tests

The tests are written so they can run with the standard library test runner:

```bash
python -m unittest discover -s python_jax_model/tests -v
```
