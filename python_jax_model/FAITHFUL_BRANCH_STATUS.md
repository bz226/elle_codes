# Faithful Branch Status

This file tracks our real position against `/home/bz1229682991/FAITHFUL_BRANCH_AGENT_GUIDE.md`.

It is intentionally blunt: the goal is to keep the faithful branch honest about what is already close to the old ELLE + FFT solver, what is still approximate, and what the next gating steps are.

## Current Position

We are no longer at the "prototype mixed with phase-field" stage.

The faithful branch now has:
- a dedicated runtime path via [run_gbm_faithful.py](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model/run_gbm_faithful.py)
- a faithful setup/runtime split via [faithful_config.py](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model/elle_jax_model/faithful_config.py), [faithful_runtime.py](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model/elle_jax_model/faithful_runtime.py), and [gbm_faithful.py](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model/elle_jax_model/gbm_faithful.py)
- exact step-0 roundtrip parity on the old FFT example through the legacy-reference harness
- a mostly faithful swept-unode update path for ownership, Euler-angle donor logic, and `U_DISLOCDEN` reset
- a real phase-pair mobility module driven by `phase_db.txt`

But we are not yet at solver parity.

The biggest remaining fidelity gaps are still:
- node-motion and topology event ordering
- stored strain energy in the GBM driving force
- recovery
- FFT/mechanics bridge
- paper-level benchmark validation

## Guide Step Status

### Step 0 — Docs split + faithful-only public story

Status: `done`

What is in place:
- [README.md](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model/README.md) presents [run_gbm_faithful.py](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model/run_gbm_faithful.py) as the parity target.
- faithful CLI examples no longer advertise hidden phase-field coefficients as meaningful faithful controls.

Residual caveat:
- the repo still contains the archived prototype modules, but they are now a side path rather than the public fidelity story.

### Step 1 — Faithful-only runtime skeleton

Status: `done`

What is in place:
- [faithful_config.py](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model/elle_jax_model/faithful_config.py)
- [faithful_runtime.py](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model/elle_jax_model/faithful_runtime.py)
- [gbm_faithful.py](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model/elle_jax_model/gbm_faithful.py)

Meaning:
- the faithful public path is no longer architected around the phase-field runtime.
- the faithful CLI works as a NumPy-first `mesh_only` branch.

Residual caveat:
- some shared package exports still include archived prototype modules, but the faithful runner itself no longer depends on them.

### Step 2 — Golden-reference harness from old code

Status: `partial`

What is in place:
- [legacy_reference.py](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model/elle_jax_model/legacy_reference.py)
- [build_legacy_reference_bundle.py](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model/build_legacy_reference_bundle.py)
- [validate_legacy_reference_bundle.py](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model/validate_legacy_reference_bundle.py)
- committed fixture:
  [fft_example_step0_reference.json](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model/legacy_reference/testdata/fft_example_step0_reference.json)

What is proven:
- exact step-0 faithful roundtrip parity is achieved for the old FFT example.

What is still missing:
- one deformation step reference
- one GBM-stage reference
- one recovery-stage reference
- one full outer-loop reference

Current blocker:
- old binaries need missing shared libraries for direct automated post-step0 extraction in this environment.

### Step 3 — Node-motion and topology event ordering

Status: `partial`

What is true now:
- [mesh.py](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model/elle_jax_model/mesh.py) already randomizes node order.
- topology maintenance is substantially more ELLE-like than before.
- in faithful `elle_surface` mode, topology repair now runs immediately after each moved node instead of only after a whole movement sweep.
- the immediate path is now local to the moved node and its neighborhood, with one stage-end global cleanup pass kept as a safety net.
- the local path now includes a real `DeleteSingleJ`-style single-node cleanup before the broader stage-end topocheck pass.
- the local path now runs node-centric double/triple checks instead of only edge-centric maintenance heuristics.
- local triple switching now respects the old phase-boundary gate, only forcing some phase-boundary switches in a much smaller gap regime.

What is still wrong:
- the local immediate-repair path is still a Python-side approximation of the ELLE event sequence, not yet a literal `CrossingsCheck -> DeleteSingleJ -> CheckDoubleJ / CheckTripleJ` port.
- local triple-node handling still reuses our current switch helper instead of a closer direct translation of the old triple-node routines.
- node-identity handling after topology changes is still a Python-side approximation, not a direct old-code node bookkeeping port.

This is still one of the main gating fidelity gaps.

### Step 4 — Literal GBM driving force

Status: `partial`

What is true now:
- `elle_surface` is much closer to ELLE than the old analogue branch.
- boundary-energy trials, diagonal trials, ELLE-style denominator, timestep gating, and physical-unit scaling are in place.
- the faithful mover now includes a first stored-energy contribution derived from current `U_DISLOCDEN`, phase properties from `phase_db.txt`, and local swept-area estimates.
- stored-energy density lookup now uses an ELLE-style same-flynn ROI-weighted average around the trial point before broader fallback, instead of a nearest-sample proxy.
- swept-area lookup now uses a small union-style construction across the local swept triangles, which is closer to the old `area_swept_gpclip` behavior than the earlier simple triangle-sum proxy.
- stored-energy fallback now distinguishes between “same flynn has unodes but none inside ROI” and “target flynn has no unodes”, matching the old solver more closely by using dummy density in the first case and broader all-unode ROI averaging only in the second.
- when a trial point cannot be assigned to any neighbouring flynn, the stored-energy path now uses the old-style dummy-density fallback with the maximum incident phase stored energy.
- the faithful mover now includes a first cluster-area energy term driven by original `phase_db.txt` cluster multipliers, cluster phases, and the preserved `F_ATTRIB_B` flynn target-area field.

What is still wrong:
- the stored-energy term is still not a literal port of the original `gpcclip` swept-area path.
- the density fallback behavior is closer to ELLE now, but it is still not a full direct port of every old `FS_density_unodes` edge case.
- cluster-area energy is still only a first faithful translation, not yet a direct port of the full old cluster-tracking subsystem and its merge/split bookkeeping.

This is another primary gating fidelity gap.

### Step 5 — Literal mobility law

Status: `partial to strong partial`

What is now in place:
- [mobility.py](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model/elle_jax_model/mobility.py)
- `phase_db.txt` parsing
- Arrhenius scaling with temperature
- Holm-style low-angle mobility reduction
- per-segment mobility threading into `elle_surface` in [mesh.py](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model/elle_jax_model/mesh.py)

What this means:
- the faithful mover is no longer effectively uniform-mobility when the seed carries enough phase/orientation information.

What is still missing:
- stronger old-code parity validation after one real motion stage
- more confidence that phase identifiers and orientation sources match the exact old-code semantics for all reference cases

### Step 6 — Literal swept-unode update semantics

Status: `strong partial`

What is now in place:
- full seed-unode remap from moved flynns
- separate handling for ownership, scalar fields, node fields, and section payloads
- nearest same-label donor for Euler fields
- flynn-mean fallback when no safe donor exists
- swept `U_DISLOCDEN` reset to zero
- concentration-like fields on a dedicated path

Meaning:
- this part of the faithful branch is much closer to the old solver now than the earlier generic transport path.

What is still missing:
- direct stage-1 old-code comparison for this behavior
- more field-class-specific parity checks beyond current tiny regressions

### Step 7 — Recovery module

Status: `partial`

What is now in place:
- [recovery.py](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model/elle_jax_model/recovery.py)
- first faithful local recovery pass over seed unodes
- local recovery now uses the original six `rot_matrix(...)` trial directions about `(010)`, `(001)`, and `(100)` in both senses instead of perturbing raw Euler components
- recovery misorientation now reads the original `symmetry.symm` operators and uses a `CME_hex`-style symmetry-aware misorientation instead of plain Euler-angle distance
- `U_ATTRIB_F` updated as average local misorientation
- `U_DISLOCDEN` reduced proportionally to accepted local misorientation reduction after the first recovery stage

What is still missing:
- exact parity validation against a saved old recovery-stage checkpoint
- direct parity checks against one old recovery-stage output

Current honest benchmark read:
- recovery is now runnable and more faithful, but the first outer-step `fine_foam` benchmark still does not show the full expected damping of GBM.
- after wiring recovery state back into flynn `EULER_3`/`DISLOCDEN`, the one-step raster grain-count result improved from `164` to `167` versus the original `178`.
- after switching recovery to symmetry-aware misorientation, the same one-step result moved to `166`, which is slightly worse numerically but more faithful to the old recovery mechanism.
- after switching to the literal six-direction `rot_matrix(...)` trial basis, the recovery-enabled outer-step runtime became slower again, so the next iteration should likely focus on keeping this more faithful basis while recovering enough performance to benchmark it cleanly.

### Step 8 — FFT bridge / mechanics snapshot interface

Status: `partial`

What is now in place:
- [fft_bridge.py](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model/elle_jax_model/fft_bridge.py)
- typed snapshot ingest for the legacy bridge artifacts:
  - `temp-FFT.out` / `temp.out`
  - `unodexyz.out`
  - `unodeang.out`
  - optional `tex.out`
- faithful-side parsing for legacy mechanics payload channels such as:
  - normalized strain rate
  - normalized stress
  - slip-system activities
  - dislocation-density proxies
  - Fourier-point / FFT-grain identifiers
- faithful outer-loop support for an explicit mechanics stage per saved outer step when one frozen snapshot or a sequence of legacy bridge snapshots is provided
- mechanics-only replay is now possible, so frozen legacy mechanics snapshots can be replayed across saved outer steps without forcing an artificial GBM stage into the parity case
- a dedicated mechanics-only transition validator now exists, so a frozen mechanics replay can be compared directly against a legacy before/after ELLE pair without hand-assembling the replay and transition report

What is still missing:
- faithful deformation-to-ELLE coupling path
- direct one-step parity tests driven by frozen old mechanics snapshots

### Step 9 — Faithful outer loop

Status: `partial`

What is in place:
- faithful runtime stages are explicit enough for a mesh-only GBM branch.
- one saved faithful snapshot can now represent multiple original-style inner stages via `subloops_per_snapshot` and `gbm_steps_per_subloop`.
- a first explicit mechanics stage can now be injected before the GBM/recovery subloops using a frozen legacy FFT snapshot
- a first nucleation stage now exists via [nucleation.py](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model/elle_jax_model/nucleation.py), using `U_EULER_3` subgrain clustering and a first real mesh rebuild from the nucleated label grid instead of only a raster label override.
- recovery stages can now be interleaved after GBM inside each subloop via `recovery_steps_per_subloop`.

What is missing:
- deformation stage
- faithful mesh/flynn split for nucleated grains instead of the current label-override approximation
- explicit remap for next FFT step

### Step 10 — Paper-level observables

Status: `partial`

What is in place:
- geometry validation
- benchmark reports
- Figure-2-style line / histogram / KDE validation views

What is missing:
- fabric tensor diagnostics
- P/G/R indices
- slip-system activity curves
- differential stress curves
- benchmark acceptance logic based on paper conclusions rather than mostly structural checks

### Step 11 — Benchmark against paper experiments

Status: `not started`

We have not yet reproduced Experiments 0, 1, 10, and 25 as a faithful benchmark suite.

### Step 12 — CI that protects solver fidelity

Status: `partial`

What is in place:
- strong local regression coverage in [test_simulation.py](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model/tests/test_simulation.py)

What is missing:
- faithful test split into dedicated subtrees
- explicit CI structure around golden references and benchmark smoke tests

### Step 13 — Optimize after parity

Status: `not applicable yet`

We are still before parity.

## Honest Summary Against the Guide PR Order

Guide PR order says:
1. docs split
2. faithful runtime split
3. legacy reference
4. node-ordering + topology parity
5. stored-energy GBM force
6. mobility law
7. swept-unode literal update
8. recovery
9. FFT bridge
10. validation
11. benchmark reproduction
12. optimization

Our real position is:
- `1` done
- `2` done
- `3` partial but strong at step 0
- `4` partial
- `5` partial
- `6` partial and now meaningfully implemented
- `7` strong partial
- `8` not started
- `9` not started
- `10` partial

So the branch has advanced usefully, but the two biggest gating items are still not fully literal:
- event ordering
- stored-energy GBM force

That means our next fidelity effort should stay focused there.

## Current Short-Term Goal Status

Short-term goal:
- produce an untuned but successful `fine_foam` result
- only allow artificial parameters that already existed in old ELLE or are unlikely to vary much across examples

Where we are:
- we do have an honest untuned faithful `fine_foam` run
- it looks promising visually
- geometry metrics improved a lot after the label-remap fix
- but it is not yet a full success by benchmark standards

The main remaining issue is:
- the mesh evolution is now much closer than before
- but the full faithful solver path still lacks stored-energy driving and full old-event ordering

## Recommended Next Move

The next guide-aligned move should be:

1. Finish Step 3 before going broader.
   Make faithful movement and topology repair happen immediately per node.

2. Then do Step 4.
   Add stored strain energy to the trial energy, so GBM is not surface-only anymore.

3. After that, rerun:
   - one tiny legacy reference comparison if possible
   - the untuned `fine_foam` sequence

That is the cleanest path to moving from “promising faithful branch” to “old solver would recognize this as the same algorithm.”
