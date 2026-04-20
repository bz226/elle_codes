# ELLE Python Faithful Translation

This folder now has two tracks:

- supported solver-parity track: the NumPy-first faithful ELLE + FFT translation centered on [run_gbm_faithful.py](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model/run_gbm_faithful.py)
- archive/prototype track: older phase-field and mixed-runner experiments kept for history, debugging, and comparison

If the goal is **faithful translation**, start with:

- [run_gbm_faithful.py](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model/run_gbm_faithful.py)
- [elle_jax_model/gbm_faithful.py](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model/elle_jax_model/gbm_faithful.py)
- [elle_jax_model/faithful_runtime.py](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model/elle_jax_model/faithful_runtime.py)
- [elle_jax_model/faithful_config.py](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model/elle_jax_model/faithful_config.py)

The faithful branch is trying to mirror the original solver structure:

- ELLE flynns and bnodes plus regular-grid unodes
- outer loop with mechanics followed by DRX subloops
- GBM, recovery, and mechanics replay stages as explicit runtime steps
- ELLE option round-trip instead of synthetic export defaults
- legacy-reference and benchmark tooling for parity checks

The older phase-field path is still importable, but it is not the solver-parity target anymore.

## Faithful Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r python_jax_model/requirements.txt
python python_jax_model/run_gbm_faithful.py --init-elle /path/to/fine_foam_step001.elle --steps 10 --save-every 1 --include-step0 --save-elle --track-topology --outdir python_jax_model/validation/faithful_run
```

Useful faithful controls:

- `--subloops-per-snapshot`, `--gbm-steps-per-subloop`, `--recovery-steps-per-subloop`
- `--motion-passes`, `--topology-passes`, `--stage-interval`
- `--temperature-c` and `--phase-db` for legacy mobility semantics
- `--mechanics-snapshot-dir` and `--mechanics-only` for explicit FFT-bridge replay cases
- `--mechanics-density-update-mode` to mirror legacy `fft2elle` DD overwrite vs increment branches

For faithful benchmark parity work, keep the guide rules in mind:

- do not tune phase-field coefficients to chase parity
- do not treat nucleation as part of benchmark mode
- do not use the archived mixed runner as the fidelity target

## Faithful Validation And Workflow

To run the automated checkpoint workflow from the guide:

```bash
python python_jax_model/run_faithful_workflow.py
```

That runner follows the guide order, records checkpoint state under `python_jax_model/validation/workflow/`, and reuses the existing targeted tests and validation scripts. The checkpoint flow is documented in [docs/faithful_workflow.md](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model/docs/faithful_workflow.md).

To run a mechanics-only parity replay against a legacy before/after ELLE pair:

```bash
python python_jax_model/validate_faithful_mechanics_transition.py --init-elle /path/to/before.elle --mechanics-snapshot-dir /path/to/fft_snapshot_sequence --reference-before /path/to/before.elle --reference-after /path/to/after.elle --outdir python_jax_model/validation/faithful_mechanics_transition --json-out python_jax_model/validation/faithful_mechanics_transition/report.json
```

To validate one frozen FFT -> ELLE mechanics import directly against the faithful runtime state:

```bash
python python_jax_model/validate_faithful_fft2elle_bridge.py --init-elle /path/to/inifft001.elle --mechanics-snapshot-dir /path/to/fft_snapshot --json-out python_jax_model/validation/faithful_fft2elle_bridge.json
```

That report checks the import-side bridge contract explicitly:
- Euler import
- unode position transfer
- host-flynn / label-field sync after moved unode positions
- `FS_CheckUnodes`-style swept-unode repair for `U_EULER_3` and `U_DISLOCDEN`
- legacy tracer initialization for `U_ATTRIB_C` / `F_ATTRIB_C` when those mechanics-side host-flynn fields are missing
- cell reset / shear update
- `tex.out` strain, stress, and activity columns
- DD increment import and phase exclusion semantics
- DD overwrite vs increment mode
- stored runtime snapshot payload

The bridge layer also now covers the export side of the old contract. `elle_jax_model.fft_bridge` can build/load/write faithful `make.out` / `temp.out` payloads, and the shipped `processes/fft/example/step0` files are used as the current real export-side parity anchor.
The export adapter can also mirror both legacy phase-ID conventions: `VISCOSITY` for `FS_elle2fft`, or `DISLOCDEN` for the shipped `processes/fft/elle2fft` path.
It also supports the old `FS_ExcludeFlynns` style export by omitting grain headers and writing point grain IDs as `0`.

To compare a faithful ELLE seed directly against a legacy `make.out` / `temp.out` bridge snapshot:

```bash
python python_jax_model/validate_faithful_elle2fft_bridge.py --init-elle /path/to/inifft001.elle --reference-dir /path/to/step0 --json-out python_jax_model/validation/faithful_elle2fft_bridge.json
```

That report distinguishes between:
- full bridge match
- bridge match excluding grain-header Euler rows
- header-only mismatch

To run the current benchmark and Figure-2-style validation helpers on a candidate ELLE sequence:

```bash
python python_jax_model/validate_benchmarks.py --reference-dir /path/to/TwoWayIceModel_Release/elle/example/results --candidate-dir /path/to/your_candidate_sequence --data-dir /path/to/TwoWayIceModel_Release/data --pattern '*.elle' --json-out python_jax_model/validation/benchmark_validation_report.json
python python_jax_model/validate_figure2_line.py --reference-dir /path/to/TwoWayIceModel_Release/elle/example/results --candidate-dir /path/to/your_candidate_sequence --pattern '*.elle' --json-out python_jax_model/validation/figure2_line_validation.json --html-out python_jax_model/validation/figure2_line_validation.html
```

To score a local experiment-family suite for the paper-style `0`, `1`, `10`, `25` runs, you can now provide either repeated explicit directory mappings, a JSON manifest, a root directory with discoverable family subdirectories, or precomputed benchmark report JSONs for each family:

```bash
python python_jax_model/validate_benchmarks.py --reference-dir /path/to/TwoWayIceModel_Release/elle/example/results --data-dir /path/to/TwoWayIceModel_Release/data --experiment-family 0=/path/to/family0 --experiment-family 1=/path/to/family1 --experiment-family 10=/path/to/family10 --experiment-family 25=/path/to/family25
python python_jax_model/validate_benchmarks.py --reference-dir /path/to/TwoWayIceModel_Release/elle/example/results --data-dir /path/to/TwoWayIceModel_Release/data --experiment-family-manifest /path/to/families.json
python python_jax_model/validate_benchmarks.py --reference-dir /path/to/TwoWayIceModel_Release/elle/example/results --data-dir /path/to/TwoWayIceModel_Release/data --experiment-family-root /path/to/family_suite
python python_jax_model/validate_benchmarks.py --reference-dir /path/to/TwoWayIceModel_Release/elle/example/results --data-dir /path/to/TwoWayIceModel_Release/data --experiment-family-report 0=/path/to/family0_report.json --experiment-family-report 1=/path/to/family1_report.json --experiment-family-report 10=/path/to/family10_report.json --experiment-family-report 25=/path/to/family25_report.json
python python_jax_model/validate_benchmarks.py --reference-dir /path/to/TwoWayIceModel_Release/elle/example/results --data-dir /path/to/TwoWayIceModel_Release/data --experiment-family-report-manifest /path/to/family_reports.json
python python_jax_model/validate_benchmarks.py --reference-dir /path/to/TwoWayIceModel_Release/elle/example/results --data-dir /path/to/TwoWayIceModel_Release/data --experiment-family-report-root /path/to/family_report_suite
```

If you only want the family-suite report itself, there is now a dedicated entrypoint:

```bash
python python_jax_model/validate_experiment_families.py --experiment-family-root /path/to/family_suite
python python_jax_model/validate_experiment_families.py --experiment-family-manifest /path/to/families.json
python python_jax_model/validate_experiment_families.py --experiment-family 0=/path/to/family0 --experiment-family 1=/path/to/family1 --experiment-family 10=/path/to/family10 --experiment-family 25=/path/to/family25
python python_jax_model/validate_experiment_families.py --experiment-family-report-manifest /path/to/family_reports.json
python python_jax_model/validate_experiment_families.py --experiment-family-report-root /path/to/family_report_suite
```

## Archive / Prototype Tools

The following modules remain available for history and side-by-side experimentation, but they are **not** the supported fidelity path:

- [run_simulation.py](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model/run_simulation.py)
- [elle_jax_model/simulation.py](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model/elle_jax_model/simulation.py)
- [elle_jax_model/elle_phasefield.py](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model/elle_jax_model/elle_phasefield.py)
- [run_elle_phasefield.py](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model/run_elle_phasefield.py)
- older calibrated presets and parameter-search tooling around the mixed runner

Those tools are still useful for:

- understanding earlier prototype behavior
- reproducing older calibration experiments
- debugging viewer and export utilities against saved prototype outputs

They should not drive faithful solver design decisions.

To run the archived direct port of ELLE's original `phasefield` process:

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

## Validation And Analysis Tools

To extract benchmark targets directly from the Llorens and Liu/Suckale papers and combine them with the local ELLE/FNO release data:

```bash
python python_jax_model/validate_papers.py --llorens-pdf /path/to/llorens2016jg.pdf --liu-pdf /path/to/liu_suckale.pdf --reference-dir /path/to/TwoWayIceModel_Release/elle/example/results --pattern 'fine_foam_step*.elle' --data-dir /path/to/TwoWayIceModel_Release/data --json-out python_jax_model/validation/paper_validation_report.json
```

To run a first benchmark-oriented validation pass against the local `fine_foam` ELLE sequence and the shipped Liu/Suckale release datasets:

```bash
python python_jax_model/validate_benchmarks.py --reference-dir /path/to/TwoWayIceModel_Release/elle/example/results --pattern 'fine_foam_step*.elle' --data-dir /path/to/TwoWayIceModel_Release/data --json-out python_jax_model/validation/benchmark_validation_report.json
```

To build a Figure-2-style grain-area validation line, comparing the mean grain area over time with one-standard-deviation bands and exporting the underlying grain-area KDE data:

```bash
python python_jax_model/validate_figure2_line.py --reference-dir /path/to/TwoWayIceModel_Release/elle/example/results --candidate-dir /path/to/your_candidate_sequence --pattern 'fine_foam_step*.elle' --json-out python_jax_model/validation/figure2_line_validation.json --html-out python_jax_model/validation/figure2_line_validation.html
```

This follows the spirit of Liu & Suckale Figure 2(c): it writes a standalone HTML line chart for the matched ELLE snapshots and also stores per-step grain-area distributions and KDE curves in the JSON report, so the distribution-side validation can be extended later without rerunning the sequence scan.

This benchmark report checks whether the ELLE reference sequence shows the expected coarsening trend and whether the warm/high-strain release datasets evolve more strongly than the cold/low-strain release datasets, which is one of the clearest paper-level expectations we can verify locally right now.

The benchmark report now includes two comparison tracks:
- `static_grain_growth`: polygon/flynn-based ELLE comparison
- `rasterized_grain_growth`: periodic connected-component comparison on the unode grain-ID field, which is often the fairer comparison for the grid-based Python rewrite

## Archive Calibration Tools

To rerun the archived calibration-oriented mixed-runner sweep against the `fine_foam` reference trajectory:

```bash
python python_jax_model/calibrate_fine_foam.py --reference-dir /path/to/TwoWayIceModel_Release/elle/example/results --pattern 'fine_foam_step*.elle' --output-dir python_jax_model/validation/fine_foam_calibration --json-out python_jax_model/validation/fine_foam_calibration_report.json --dt-grid 0.01 0.02 0.03 --mobility-grid 0.5 0.75 1.0
```

This calibration report writes one candidate ELLE sequence per parameter set, scores each run against the rasterized and polygon `fine_foam` benchmarks, and highlights the current best match for that older mixed-runner branch.
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
