# Project-Wide Rewrite Map

This document is the working map for the ELLE rewrite effort in
[python_jax_model](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model).
It is meant to answer three questions:

1. What does the original ELLE project contain?
2. What has already been rewritten faithfully enough to trust?
3. What still needs revision, and in what order?

## 1. Original ELLE Project Structure

Top-level original code areas:

- Core platform:
  - [basecode](/home/bz1229682991/research/Elle/newcode/elle/basecode)
  - [lib](/home/bz1229682991/research/Elle/newcode/elle/lib)
- Process drivers:
  - [processes](/home/bz1229682991/research/Elle/newcode/elle/processes)
- Frederic-style / FFT-linked process chain:
  - [FS_Codes](/home/bz1229682991/research/Elle/newcode/elle/FS_Codes)
- Utilities and viewers:
  - [utilities](/home/bz1229682991/research/Elle/newcode/elle/utilities)
  - [plotcode](/home/bz1229682991/research/Elle/newcode/elle/plotcode)
  - [wxplotcode](/home/bz1229682991/research/Elle/newcode/elle/wxplotcode)

The original ELLE model has two major layers:

- Boundary-network / topology layer:
  - flynns
  - nodes / bnodes
  - topology checks
  - switching / splitting / deletion
- Regular material-data layer:
  - unodes
  - unode attributes and concentrations
  - local material state updates

The main original workflows we care about are:

- `phasefield` process:
  - [processes/phasefield](/home/bz1229682991/research/Elle/newcode/elle/processes/phasefield)
- Grain-boundary migration / recrystallization / `fine_foam` style chain:
  - [FS_Codes/FS_recrystallisation](/home/bz1229682991/research/Elle/newcode/elle/FS_Codes/FS_recrystallisation)
  - [FS_Codes/FS_utilities](/home/bz1229682991/research/Elle/newcode/elle/FS_Codes/FS_utilities)
  - [basecode/GBMUnodeUpdate.cc](/home/bz1229682991/research/Elle/newcode/elle/basecode/GBMUnodeUpdate.cc)

## 2. Python Rewrite Structure

Main Python modules:

- Main multiphase / prototype branch:
  - [simulation.py](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model/elle_jax_model/simulation.py)
- Faithful mesh / unode branch:
  - [mesh.py](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model/elle_jax_model/mesh.py)
- Topology tracking:
  - [topology.py](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model/elle_jax_model/topology.py)
- ELLE import/export:
  - [elle_export.py](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model/elle_jax_model/elle_export.py)
- Direct `phasefield` port:
  - [elle_phasefield.py](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model/elle_jax_model/elle_phasefield.py)
- Viewing:
  - [elle_visualize.py](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model/elle_jax_model/elle_visualize.py)
  - [elle_html_viewer.py](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model/elle_jax_model/elle_html_viewer.py)
- Validation / calibration:
  - [benchmark_validation.py](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model/elle_jax_model/benchmark_validation.py)
  - [calibration.py](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model/elle_jax_model/calibration.py)
  - [microstructure_validation.py](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model/elle_jax_model/microstructure_validation.py)
  - [paper_validation.py](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model/elle_jax_model/paper_validation.py)
  - [phasefield_compare.py](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model/elle_jax_model/phasefield_compare.py)

Important CLIs:

- Main mixed runner:
  - [run_simulation.py](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model/run_simulation.py)
- Direct phasefield runner:
  - [run_elle_phasefield.py](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model/run_elle_phasefield.py)
- Viewer:
  - [view_elle.py](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model/view_elle.py)
  - [render_elle.py](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model/render_elle.py)

## 3. Rewrite Branches

There are really three rewrite branches now.

### A. Calibrated Multiphase Branch

Purpose:
- match `fine_foam` numerically as well as possible

Main files:
- [simulation.py](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model/elle_jax_model/simulation.py)
- [calibration.py](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model/elle_jax_model/calibration.py)

Strengths:
- currently the best `fine_foam` benchmark match
- easy to run
- good for numerical comparison and validation

Weakness:
- structurally still an analogue, not a literal ELLE GBM rewrite

### B. Truthful Mesh / NumPy Branch

Purpose:
- follow the original ELLE GBM structure more honestly

Main files:
- [mesh.py](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model/elle_jax_model/mesh.py)
- [simulation.py](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model/elle_jax_model/simulation.py)

Current philosophy:
- move mesh
- run topocheck-style maintenance
- update unodes
- avoid relying on the old JAX PDE loop

Strengths:
- closer to original ELLE structure
- better place to keep porting `GBMUnodeUpdate` and topological logic

Weakness:
- not yet as numerically close to `fine_foam` as the calibrated analogue branch

### C. Direct Phasefield Port

Purpose:
- faithfully port the original single-order-parameter `phasefield` process

Main file:
- [elle_phasefield.py](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model/elle_jax_model/elle_phasefield.py)

Strengths:
- much closer to a real process port
- already reads and writes ELLE-style phasefield files
- easier to validate directly

Weakness:
- only covers the `phasefield` process, not the full GBM / FFT chain

## 4. Current Fidelity by Subsystem

### 4.1 ELLE File I/O

Status: fairly faithful

Files:
- [elle_export.py](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model/elle_jax_model/elle_export.py)
- [elle_phasefield.py](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model/elle_jax_model/elle_phasefield.py)

What is solid:
- `UNODES`
- `FLYNNS`
- `LOCATION`
- preservation of original `U_*`
- preservation of original `N_*` in truthful branch

What still needs work:
- make export/import semantics match original ELLE more literally for all edge cases

### 4.2 Viewer / Showelle Replacement

Status: practical and working, but not a literal rewrite

Files:
- [elle_visualize.py](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model/elle_jax_model/elle_visualize.py)
- [elle_html_viewer.py](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model/elle_jax_model/elle_html_viewer.py)

What is solid:
- renders original `.elle` files
- periodic boundary splitting handled much better now
- compact HTML bundle works

What still needs work:
- closer visual parity with original `showelle`
- better support for every display mode the old viewer used

### 4.3 Direct Phasefield Process

Status: one of the most faithful rewritten parts

Files:
- [elle_phasefield.py](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model/elle_jax_model/elle_phasefield.py)
- [phasefield_compare.py](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model/elle_jax_model/phasefield_compare.py)

What is strong:
- direct state import/export
- anisotropic interface width
- coupled temperature
- latent heat
- benchmark comparison tooling exists

What still needs work:
- full timestep-by-timestep validation against original binary outputs

### 4.4 Mesh Motion / Topocheck Sequence

Status: partially faithful and improving

Files:
- [mesh.py](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model/elle_jax_model/mesh.py)

What is already close:
- ELLE mesh seeding from original files
- topocheck-style stage order
- `elle_surface` now includes the original optional diagonal trial pivots
- the truthful mover now follows `GetMoveDir` more closely by requiring both
  force components and by resetting `speedup` to `1.0` before the local 90%
  clamp
- the truthful mover now has an ELLE-style physical-unit denominator path using
  seed `UnitLength`, with mobility slots in the formula even though per-segment
  mobility is still simplified
- acute-angle widening
- small-flynn deletion
- split-if-neck-too-narrow analogue
- node-spacing maintenance

What still needs work:
- exact node-motion law beyond the current diagonal-aware surface probe
- hidden ELLE helper behavior
- full equivalence of topological event ordering

### 4.5 Unode Update / Partition Path

Status: now substantially closer, but not finished

Files:
- [mesh.py](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model/elle_jax_model/mesh.py)
- [PARTITION_REWRITE_MAP.md](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model/PARTITION_REWRITE_MAP.md)

What is now close:
- incremented swept records
- separate `sweep`, `enrich`, `reassigned` masks
- explicit ELLE-style `[0] / [1] / [2]` channels
- `conc_s`, `conc_s1`, `conc_e`, `conc_e1`
- boundary concentration bookkeeping
- a literal Python `PartitionMass(...)`-style node solve is now used in the
  truthful node-aware branch for final node concentration and total `put_mass`
- `put_mass` now applied using enrich support, closer to `EnrichUnodes`

What still needs work:
- literal `Weights[j][i]`-style swept-mass removal
- literal `p_mass[j] / Total_Weights[j]` enrich redistribution

### 4.6 Main Grain-Growth Physics

Status: mixed

Files:
- [simulation.py](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model/elle_jax_model/simulation.py)

Current reality:
- the calibrated branch is still an analogue
- the truthful branch is now much less PDE-dependent and now uses an explicit
  `PartitionMass(...)`-style node solve, but it still is not a full port of the
  original GBM process chain
- the faithful GBM translation path now has a dedicated home in
  [gbm_faithful.py](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model/elle_jax_model/gbm_faithful.py)
  and [run_gbm_faithful.py](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model/run_gbm_faithful.py),
  with [run_truthful_numpy.py](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model/run_truthful_numpy.py)
  kept as a compatibility alias

Main missing pieces:
- exact stored-energy coupling
- FFT-driven forcing path
- exact GBM process translation beyond node motion and unode reassignment

## 5. Validation Status

### `fine_foam`

Best numerical match right now:
- calibrated branch

Best structural match direction:
- truthful mesh / NumPy branch

Main benchmark tools:
- [benchmark_validation.py](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model/elle_jax_model/benchmark_validation.py)
- [calibration.py](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model/elle_jax_model/calibration.py)

### Phasefield

Best validation path:
- direct `phasefield` port
- compare ELLE state files and sequences

Main tools:
- [phasefield_compare.py](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model/elle_jax_model/phasefield_compare.py)
- [validate_elle_phasefield.py](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model/validate_elle_phasefield.py)

### Paper-level validation

Available:
- [paper_validation.py](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model/elle_jax_model/paper_validation.py)
- [microstructure_validation.py](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model/elle_jax_model/microstructure_validation.py)

Still missing:
- exact reproduction of the full Liu/Suckale multiscale chain
- exact reproduction of the full ELLE FFT/recovery/GBM coupling

## 6. Highest-Priority Revision Order

### Tier 1: Finish Faithful Unode Transfer

1. Complete the remaining `Partition` gaps in
   [mesh.py](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model/elle_jax_model/mesh.py)
2. Port a denser `RemoveSweptMass(...)` / `EnrichUnodes(...)` style matrix path
3. Keep checking the literal node solve against the richer per-entry partition
   diagnostics so the truthful branch stays aligned with both ELLE routines

Why first:
- this is now the cleanest path toward a truly faithful GBM translation

### Tier 2: Tighten Truthful Mesh Motion

1. Compare the current `elle_surface` mover directly against original ELLE formulas
2. Match node event ordering and edge cases more literally
3. Re-benchmark truthful branch on `fine_foam`

Why next:
- after unode transfer, mesh motion is the next biggest reason the truthful branch still diverges

### Tier 3: Port More of the GBM Process Chain

1. Identify which original driver should be ported next:
   - [FS_gbm_pp_fft.elle.cc](/home/bz1229682991/research/Elle/newcode/elle/FS_Codes/FS_recrystallisation/FS_gbm_pp_fft/FS_gbm_pp_fft.elle.cc)
   - [FS_movenode_pp_fft.cc](/home/bz1229682991/research/Elle/newcode/elle/FS_Codes/FS_recrystallisation/FS_gbm_pp_fft/FS_movenode_pp_fft.cc)
   - [FS_topocheck.elle.cc](/home/bz1229682991/research/Elle/newcode/elle/FS_Codes/FS_utilities/FS_topocheck/FS_topocheck.elle.cc)
2. Separate stored-energy and FFT-coupled forcing from the analogue branch

Why:
- this is where the faithful branch becomes a real process translation instead of a structural imitation

### Tier 4: Keep the Calibrated Branch Stable

1. Preserve the calibrated `fine_foam` branch as the numerical reference
2. Keep using it for benchmark sanity while the truthful branch improves

Why:
- it gives us a strong baseline and prevents “getting more faithful” from feeling like losing ground everywhere

## 7. Lower-Priority Revisions

- Viewer parity improvements
- Cleaner separation of “faithful” vs “analogue” CLI workflows
- More dedicated docs for each branch
- More benchmark cases besides `fine_foam`

## 8. Recommended Reading Order

If you want the fastest orientation path:

1. [python_jax_model/README.md](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model/README.md)
2. [PROJECT_REWRITE_MAP.md](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model/PROJECT_REWRITE_MAP.md)
3. [PARTITION_REWRITE_MAP.md](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model/PARTITION_REWRITE_MAP.md)
4. [elle_jax_model/simulation.py](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model/elle_jax_model/simulation.py)
5. [elle_jax_model/mesh.py](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model/elle_jax_model/mesh.py)
6. [tests/test_simulation.py](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model/tests/test_simulation.py)

## 9. Short Honest Summary

The rewrite is no longer one thing.

- The direct `phasefield` port is already fairly faithful.
- The calibrated multiphase branch is the best numerical `fine_foam` match.
- The truthful mesh/NumPy branch is the best path toward a real ELLE GBM rewrite.

The highest-value next work is on the truthful branch, especially:
- finishing the unode transfer path
- tightening mesh motion
- then porting more of the real GBM driver logic
