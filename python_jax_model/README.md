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

To seed the multiphase rewrite from an existing ELLE grain map instead of a synthetic Voronoi layout:

```bash
python python_jax_model/run_simulation.py --init-mode elle --init-elle /path/to/fine_foam_step001.elle --steps 10 --save-every 1 --save-elle --track-topology --outdir python_jax_model/output_from_elle
```

By default the loader now tries to auto-detect the unode attribute that behaves like a grain-ID field. If you already know the correct attribute for a file, you can override it with `--init-elle-attribute U_ATTRIB_C` or another section name.
When mesh coupling is enabled on an ELLE-seeded run, the runtime now also imports the original `LOCATION`, `FLYNNS`, and key `OPTIONS` values (`SwitchDistance`, `MinNodeSeparation`, `MaxNodeSeparation`, `SpeedUp`) so the mesh-feedback path starts from the original ELLE boundary network instead of reconstructing the first mesh from raster labels alone.

To rerun the current best-known calibrated `fine_foam` branch directly from the CLI, add the calibrated preset:

```bash
python python_jax_model/run_simulation.py --preset fine-foam-calibrated --init-mode elle --init-elle /path/to/fine_foam_step001.elle --steps 10 --save-every 1 --save-elle --track-topology --outdir python_jax_model/output_from_elle_calibrated
```

Right now that preset expands to `dt=0.025`, `mobility=1.0`, `gradient_penalty=1.25`, and `interaction_strength=1.5`, which is the best `fine_foam` match we have measured so far.
It also overrides the ELLE seeding knobs to `init_smoothing_steps=0` and `init_noise=0.0`, so the preset reproduces the calibrated run instead of the noisier CLI defaults.
For backward compatibility, `--preset fine-foam-best` still points to this same older calibrated branch.

To run the newer, more structurally faithful ELLE-mesh branch separately:

```bash
python python_jax_model/run_simulation.py --preset fine-foam-truthful-mesh --init-mode elle --init-elle /path/to/fine_foam_step001.elle --steps 10 --save-every 1 --save-elle --track-topology --outdir python_jax_model/output_from_elle_truthful_mesh
```

That truthful preset keeps the same calibrated phase-field coefficients, but it also turns on the ELLE-seeded mesh path with `mesh_relax_steps=1`, `mesh_topology_steps=1`, `mesh_movement_model=elle_surface`, `mesh_update_mode=mesh_only`, and `mesh_feedback_every=1`, so we can improve the mesh-faithful branch separately from the older raster-first calibrated branch.
The `elle_surface` movement model is a more literal nod to the original ELLE GBM code: it probes surface energy at `±SwitchDistance` in `x/y`, can now also include the original ELLE-style diagonal trial pivots, forms a trial-difference force vector, and then advances double and triple junctions with an ELLE-style denominator based on incident segment lengths and projected normals. Its timestep handling is also closer now: the truthful branch follows the `GetMoveDir` gate that requires both force components to be active, and it resets `speedup` to `1.0` before falling back to the local 90%-of-maximum movement clamp. The truthful preset also now enables an ELLE-style physical-unit path for that denominator, so `UnitLength` from the seed file participates in the motion law instead of being dropped after import. The `mesh_only` update mode then skips the phase-field blend for that branch and reassigns the original ELLE-seeded unode points from the moved flynn polygons before writing grain IDs back to the grid, which is closer to the original “move boundaries, topocheck, update unodes” sequence than the older PDE-plus-feedback approximation. It now also preserves and rewrites original ELLE `U_*` unode sections and `N_ATTRIB_*` / `N_CONC_*` node sections when those are present in the seed file, and matching `U_CONC_*` / `N_CONC_*` fields now share one segment ledger: swept unode mass is removed from incremented swept-triangle shell records, those increments use area-scaled endpoint cosine-bell weights closer to ELLE's `Weights[j][i]` construction, and each increment now carries separate `sweep`, `enrich`, and `reassigned` masks so reassigned overlap points can be treated separately instead of using one coarse triangle mask. Boundary-side bookkeeping is now also carried per increment rather than only at the aggregated node level, so each increment keeps its own old/new boundary-area share, source mass, capacity, weighted `conc_s / conc_s1 / conc_e / conc_e1 / conc_b` state, the ELLE-style `Total_Sweep_Weights[0/1/2]` and `Total_Enrich_Weights[0/1/2]` channels, a `partition_active` flag, final boundary-mass contribution, and explicit `mass_chge_s / mass_chge_e / mass_chge_b` terms before the node totals are assembled. In the node-aware branch, the literal `PartitionMass(...)`-style node solve now determines final node concentration and `put_mass`, the enrich-side delta (`mass_chge_e`) is what gets applied back to unodes, and the raw redistributed share is kept separately in the ledger for inspection. Node concentrations are updated from that same ledger, and the unode enrichment uses the same segment bookkeeping before the final correction.
Under the hood that truthful branch now dispatches to an explicit NumPy runner instead of the JAX phase-field loop, so the faithful path is separated in code as well as in behavior.

If you want the same faithful GBM stage structure without the `fine_foam`-specific benchmark tuning, use the general faithful preset instead:

```bash
python python_jax_model/run_simulation.py --preset gbm-faithful-default --init-mode elle --init-elle /path/to/example.elle --steps 10 --save-every 1 --save-elle --track-topology --outdir python_jax_model/output_from_elle_gbm_default
```

That preset keeps the ELLE-seeded `mesh_only` GBM path, but it only enforces the general stage defaults:
- zero-noise ELLE seeding
- one movement pass per saved stage
- one topology-cleanup pass per saved stage
- `elle_surface` movement
- diagonal trial pivots
- `UnitLength`-aware motion
- raster boundary support derived from the seed file's `BoundaryWidth` and `UnitLength` when available, with a stable one-cell fallback

Unlike the `fine-foam-truthful-mesh` preset, it does not advertise the old benchmark-shaped phase-field coefficients as meaningful faithful controls.

If you want to call that faithful path explicitly rather than through the mixed prototype CLI, use the dedicated NumPy runner:

```bash
python python_jax_model/run_gbm_faithful.py --init-elle /path/to/fine_foam_step001.elle --steps 10 --save-every 1 --save-elle --track-topology --outdir python_jax_model/output_truthful_numpy
```

The faithful GBM branch now also has a dedicated module boundary in [gbm_faithful.py](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model/elle_jax_model/gbm_faithful.py), so the original-ELLE-style NumPy path can evolve separately from the older mixed simulation runner. [run_truthful_numpy.py](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model/run_truthful_numpy.py) remains as a compatibility alias to the new dedicated runner.

That dedicated runner now exposes more honest GBM-stage controls such as `--motion-passes`, `--topology-passes`, `--stage-interval`, and `--raster-boundary-band` instead of foregrounding analogue-era phase-field coefficients. The older hidden flags still work for compatibility, but they are intentionally de-emphasized because the faithful `mesh_only` branch is driven mainly by ELLE seed geometry, node motion, topology repair, and unode reassignment rather than by PDE coefficients.

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

To extract benchmark targets directly from the Llorens and Liu/Suckale papers and combine them with the local ELLE/FNO release data:

```bash
python python_jax_model/validate_papers.py --llorens-pdf /path/to/llorens2016jg.pdf --liu-pdf /path/to/liu_suckale.pdf --reference-dir /path/to/TwoWayIceModel_Release/elle/example/results --pattern 'fine_foam_step*.elle' --data-dir /path/to/TwoWayIceModel_Release/data --json-out python_jax_model/validation/paper_validation_report.json
```

To run a first benchmark-oriented validation pass against the local `fine_foam` ELLE sequence and the shipped Liu/Suckale release datasets:

```bash
python python_jax_model/validate_benchmarks.py --reference-dir /path/to/TwoWayIceModel_Release/elle/example/results --pattern 'fine_foam_step*.elle' --data-dir /path/to/TwoWayIceModel_Release/data --json-out python_jax_model/validation/benchmark_validation_report.json
```

This benchmark report checks whether the ELLE reference sequence shows the expected coarsening trend and whether the warm/high-strain release datasets evolve more strongly than the cold/low-strain release datasets, which is one of the clearest paper-level expectations we can verify locally right now.

The benchmark report now includes two comparison tracks:
- `static_grain_growth`: polygon/flynn-based ELLE comparison
- `rasterized_grain_growth`: periodic connected-component comparison on the unode grain-ID field, which is often the fairer comparison for the grid-based Python rewrite

To calibrate the ELLE-seeded rewrite directly against the `fine_foam` reference trajectory and rank a small parameter sweep automatically:

```bash
python python_jax_model/calibrate_fine_foam.py --reference-dir /path/to/TwoWayIceModel_Release/elle/example/results --pattern 'fine_foam_step*.elle' --output-dir python_jax_model/validation/fine_foam_calibration --json-out python_jax_model/validation/fine_foam_calibration_report.json --dt-grid 0.01 0.02 0.03 --mobility-grid 0.5 0.75 1.0
```

This calibration report writes one candidate ELLE sequence per parameter set, scores each run against the rasterized and polygon `fine_foam` benchmarks, and highlights the current best match.
Calibration now reuses existing candidate ELLE sequences in the output directory by default, so you can rerun or extend a search without regenerating completed runs. Add `--no-reuse-existing` if you want to force regeneration.

To score an existing calibration directory, including partially completed runs, and rank what is already on disk:

```bash
python python_jax_model/score_calibration_runs.py --reference-dir /path/to/TwoWayIceModel_Release/elle/example/results --candidate-root python_jax_model/validation/fine_foam_calibration_refined --pattern 'fine_foam_step*.elle' --json-out python_jax_model/validation/fine_foam_calibration_refined_scores.json
```

By default this applies a coverage penalty so incomplete runs do not outrank fully matched candidates just because they only reached the easy early steps. Add `--complete-only` if you want to rank only fully completed runs.

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
