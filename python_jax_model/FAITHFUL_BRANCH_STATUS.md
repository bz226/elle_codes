# Faithful Branch Status

This file tracks our real position against `/home/bz1229682991/UPDATED_FAITHFUL_BRANCH_AGENT_GUIDE.md`.

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
- recovery
- faithful outer-loop coupling, especially nucleation/cluster bookkeeping
- FFT/mechanics bridge
- paper-level benchmark validation

## Guide Step Status

### Step 0 — Docs split + faithful-only public story

Status: `done`

What is in place:
- [README.md](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model/README.md) now leads with [run_gbm_faithful.py](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model/run_gbm_faithful.py) as the supported parity target.
- the README now explicitly marks [run_simulation.py](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model/run_simulation.py), [simulation.py](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model/elle_jax_model/simulation.py), [elle_phasefield.py](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model/elle_jax_model/elle_phasefield.py), and related calibration tooling as archive/prototype paths.
- [elle_jax_model/__init__.py](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model/elle_jax_model/__init__.py) now exports faithful modules first, with prototype modules retained but grouped as archive/history.
- [run_gbm_faithful.py](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model/run_gbm_faithful.py), [gbm_faithful.py](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model/elle_jax_model/gbm_faithful.py), and [faithful_runtime.py](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model/elle_jax_model/faithful_runtime.py) now carry explicit banner text that they are the fidelity targets.

Residual caveat:
- the repo still contains extensive historical prototype documentation deeper in the README, but the first-contact public story is now aligned with the faithful branch.

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

### Step 1 — ELLE option round-trip and runtime defaults

Status: `done`

What is now in place:
- [faithful_config.py](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model/elle_jax_model/faithful_config.py) now parses and retains seed ELLE options explicitly, including:
  - `SwitchDistance`
  - `MinNodeSeparation`
  - `MaxNodeSeparation`
  - `SpeedUp`
  - `Timestep`
  - `UnitLength`
  - `Temperature`
  - `Pressure`
  - `BoundaryWidth`
  - `MassIncrement`
- the carried option payload now also preserves:
  - `CellBoundingBox`
  - `SimpleShearOffset`
  - `CumulativeSimpleShear`
- [gbm_faithful.py](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model/elle_jax_model/gbm_faithful.py) now defaults the faithful runtime temperature to the seed ELLE `Temperature` option instead of hard-coding `25.0`.
- effective runtime option values are now threaded into the faithful mesh state, so export writes original seed values unless a faithful runtime override explicitly changed them.
- [elle_export.py](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model/elle_jax_model/elle_export.py) now writes carried ELLE option values instead of the old synthetic `Timestep 1`, `UnitLength 1`, `Temperature 25`, and `Pressure 1` placeholders.

What is now proven:
- the shipped [inifft001.elle](/home/bz1229682991/research/Elle/newcode/elle/processes/fft/example/step0/inifft001.elle) seed preserves its key option values across zero-stage faithful export.
- explicit runtime temperature overrides now update the faithful export instead of being silently dropped.

What is still missing:
- a fuller option round-trip check against more than one real seed file
- confirmation that every exported option line matches old ELLE ordering and formatting where that matters for downstream tooling

### Step 2 — Golden-reference harness from old code

Status: `strong partial`

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

Status: `done`

What is true now:
- [mesh.py](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model/elle_jax_model/mesh.py) already randomizes node order.
- topology maintenance is substantially more ELLE-like than before.
- in faithful `elle_surface` mode, topology repair now runs immediately after each moved node instead of only after a whole movement sweep.
- the immediate path is now local to the moved node and its neighborhood, with one stage-end global cleanup pass kept as a safety net.
- the local path now includes a real `DeleteSingleJ`-style single-node cleanup before the broader stage-end topocheck pass.
- the local path now runs node-centric double/triple checks instead of only edge-centric maintenance heuristics.
- local triple switching now respects the old phase-boundary gate, only forcing some phase-boundary switches in a much smaller gap regime.
- the faithful local topology path is now split into explicit ELLE-style sub-steps in [mesh.py](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model/elle_jax_model/mesh.py):
  - `crossings_check`
  - `delete_single_j`
  - `check_double_j`
  - `check_triple_j`
  - `update_topology_state`
- local topology events now carry ordered `faithful_topology_stage` markers, and [test_simulation.py](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model/tests/test_simulation.py) now includes a deterministic regression that checks the emitted stage order instead of only the final mesh geometry.
- the `crossings_check` stage now has a first literal old-style movement veto for double nodes in 2-sided flynns: if the moved node path would overtake a flynn edge, the move is rolled back locally instead of being allowed through to later cleanup.
- the `crossings_check` stage now also mirrors the old ELLE triple-node safety gate: if a moved triple node gets within switch distance of the nearest neighbouring triple node but the switch would be impossible, the move is rolled back before later topology checks.
- the local triple-node path now also carries an explicit old-style `FS_TripleSwitchPossible` gate: if two neighboring triple nodes have the same three incident flynns, the switch is rejected immediately instead of being left to fail later inside generic switch logic.
- when local topology changes invalidate node indices, the faithful movement loop now rebuilds only the remaining randomized node order instead of restarting a fresh whole-mesh permutation, which is closer to the old sequential node-processing rhythm.
- the triple-node path now records explicit structured rejection events for failed switch attempts in both the immediate faithful path and the stage-end topocheck path, including same-flynn-neighbourhood and phase-boundary vetoes plus lower-level switch-helper rejection reasons, so event-log comparison is no longer limited to successful topology changes.
- triple-node candidate selection is now explicitly node-centric in both the local `check_triple_j` path and the stage-end topocheck path: each triple node selects only its nearest short triple neighbour, which is closer to old `ElleCheckTripleJ` behaviour than the earlier global short-edge sweep.
- successful triple-switch events now also carry explicit old-style neighbourhood roles (`node3/node4/node5/node6`) so event-log parity can compare which local triple-node rewrite was chosen, not just whether a switch happened.
- the triple-switch helper now derives an explicit old-style local neighbourhood context before rewriting (`node3/node4/node5/node6` plus `full_id_0..3`-style flynn roles), so the switch path itself is less generic and closer to the old `ElleSwitchTripleNodes` setup logic.
- the actual triple-switch polygon rewrite now runs through an explicit legacy-role helper that rewrites the four affected flynns from `full_id_0..3` and `node3..6`, so the switch body is starting to follow the old `ElleSwitchTripleNodes` structure instead of assembling shared/exclusive rewrites ad hoc.
- post-switch node positioning now follows the old `ElleFindCentre` / `ElleMoveToCentre` pattern for the switched triple pair, replacing the previous generic nudge with a more literal centre-of-triangle repositioning based on the legacy `node3/node4/node5/node6` neighbourhood.
- the switch path now also has a local old-style two-sided-grain cleanup step scoped to the switched pair and the four involved flynns before the final centre move, instead of relying only on the later broad cleanup pass.
- post-switch validation is now less index-fragile: the triple-switch context carries stable legacy flynn IDs alongside legacy role indices, local post-switch validation now checks the active `full_id_0..3` roles by flynn ID before the broad affected-flynn fallback, and successful switch events now keep their shared/exclusive/legacy flynn payloads anchored to the original legacy roles even if local cleanup merges shift flynn indices.
- the triple-switch routine now also runs a direct local post-switch topology pass on the switched node before returning, which is closer to the old `ElleSwitchTripleNodes -> ElleNodeTopologyCheck(node1)` rhythm than the previous “return and let broader cleanup rediscover it later” behaviour.

Residual caveat:
- this is still a faithful Python translation of the old ELLE topology order rather than a line-by-line C port, so future parity work may still tighten low-level details, but Step 3 is no longer the main gating gap.

### Step 4 — Literal GBM driving force

Status: `done`

What is true now:
- `elle_surface` is much closer to ELLE than the old analogue branch.
- boundary-energy trials, diagonal trials, ELLE-style denominator, timestep gating, and physical-unit scaling are in place.
- the stored-energy trial path now follows the old `GetNodeStoredEnergy` structure directly:
  - target flynn chosen from neighbouring flynns at the trial position
  - same-phase internal case uses `disscale` as a gate only, not a multiplier
  - mixed-phase boundary case applies `disbondscale`
  - trial density is taken from a flynn-based ROI-weighted `U_DISLOCDEN` average, with the same main fallback split as `FS_density_unodes`
  - the ROI radius now follows the old `FS_GetROI(3)` box-based formula instead of a count-only shortcut
  - when a trial point is in none of the neighbouring flynns, the path now uses the old dummy-density / max-incident-phase-energy fallback
- swept-area lookup now uses the exact local union geometry of the swept triangles for this node-motion configuration, which is the faithful Python equivalent of the old `area_swept_gpclip` union for these local trials.
- the faithful mover now includes a first cluster-area energy term driven by original `phase_db.txt` cluster multipliers, cluster phases, and the preserved `F_ATTRIB_B` flynn target-area field.
- small two-grain acceptance tests now cover:
  - curvature-only collapse when stored-energy contrast is removed
  - density-driven motion when curvature contrast is weak
  - mixed-force superposition

Residual caveat:
- cluster-area energy is still only a first faithful translation of the old cluster-tracking subsystem and its merge/split bookkeeping, but that is now better treated as an outer-loop / cluster-tracking fidelity gap than as the main GBM stored-energy gap.

This is no longer one of the main gating fidelity gaps.

### Step 5 — Literal mobility law

Status: `done`

What is now in place:
- [mobility.py](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model/elle_jax_model/mobility.py)
- `phase_db.txt` parsing
- Arrhenius scaling with temperature
- Holm-style low-angle mobility reduction
- per-segment mobility threading into `elle_surface` in [mesh.py](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model/elle_jax_model/mesh.py)
- segment misorientation sourcing is now closer to the old mover:
  - if `U_EULER_3` is present, edge mobility now uses nearest-unode orientation sampling at the segment midpoint, with the same flynn-first then global-within-ROI fallback shape as the legacy `SearchUnode` path
  - if the unode orientation path is not available, the faithful branch falls back to flynn `EULER_3`
- segment boundary energy now also follows the legacy `CheckPair(..., 1)` path instead of a single scalar everywhere:
  - the `elle_surface` surface term now weights each incident segment with the phase-pair boundary energy from `phase_db.txt`
  - this uses the same phase-pair lookup family as legacy mobility and activation-energy reads
- the faithful branch now explicitly matches the old misorientation semantics:
  - `GetMisorientation` in the legacy mover uses c-axis angle derived from `EULER_3`, not a full crystal-symmetry orientation distance
  - the faithful mobility path intentionally stays on that c-axis contract for Step 5
- direct regressions now prove that:
  - phase-pair boundary energy comes from the legacy pair table
  - temperature and misorientation change speed through mobility, but do not change the force direction

What this means:
- the faithful mover is no longer effectively uniform-mobility when the seed carries enough phase/orientation information.
- the mobility path now matches the old source not only in formula, but also in:
  - which orientation data layer supplies the low-angle reduction
  - where phase-pair boundary energy enters the surface term
  - the legacy c-axis-only misorientation contract used by `GetMisorientation`

Residual caveat:
- broader one-stage motion parity is still worth validating, but that is now better treated as a whole-mover / reference-harness task than as an unfinished mobility-law translation gap.

### Step 6 — Literal swept-unode update semantics

Status: `strong partial`

What is now in place:
- full seed-unode remap from moved flynns
- separate handling for ownership, scalar fields, node fields, and section payloads
- nearest same-label donor for Euler fields
- legacy-style Euler fallback is now closer to `FS_fft2elle`:
  - nearest same-label donor is now constrained by a legacy-style reassignment ROI instead of searching the whole grain without limit
  - when no donor exists inside that ROI but unswept donors remain in the new grain, the fallback is now a distance-weighted whole-flynn mean, matching the old code’s unusual weighting
  - when no valid unswept donor remains at all, the swept unode now keeps its old orientation instead of inventing a synthetic mean
- swept `U_DISLOCDEN` reset to zero
- concentration-like fields on a dedicated path

Meaning:
- this part of the faithful branch is much closer to the old solver now than the earlier generic transport path.
- the branch now also has a real dumped-transition parity path for swept unodes through [legacy_reference.py](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model/elle_jax_model/legacy_reference.py), and the shipped old `fine_foam_step001 -> fine_foam_step002` pair now exercises both swept `U_EULER_3` and swept `U_DISLOCDEN` directly.
- a legacy parser bug around composite section headers like `U_FINITE_STRAIN START_S_X ...` is now fixed, so old `U_DISLOCDEN` sections survive dense-state extraction instead of being truncated by the next header line.

### Step 7 — Recovery module

Status: `done`

What is now in place:
- [recovery.py](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model/elle_jax_model/recovery.py)
- first faithful local recovery pass over seed unodes
- local recovery now uses the original six `rot_matrix(...)` trial directions about `(010)`, `(001)`, and `(100)` in both senses instead of perturbing raw Euler components
- recovery misorientation now reads the original `symmetry.symm` operators and uses a `CME_hex`-style symmetry-aware misorientation instead of plain Euler-angle distance
- `U_ATTRIB_F` updated as average local misorientation
- `U_DISLOCDEN` reduced proportionally to accepted local misorientation reduction after the first recovery stage
- a deterministic legacy-style inner-loop helper now checks:
  - which trial rotation is accepted
  - final Euler values
  - final average local misorientation
  - final `U_DISLOCDEN`
- the live recovery stage now matches that sequential legacy-style response on a tiny seeded fixture, which is the practical acceptance target while the old standalone recovery binary is not runnable in this environment

Current honest benchmark read:
- recovery is now runnable and more faithful, but the first outer-step `fine_foam` benchmark still does not show the full expected damping of GBM.
- after wiring recovery state back into flynn `EULER_3`/`DISLOCDEN`, the one-step raster grain-count result improved from `164` to `167` versus the original `178`.
- after switching recovery to symmetry-aware misorientation, the same one-step result moved to `166`, which is slightly worse numerically but more faithful to the old recovery mechanism.
- after switching to the literal six-direction `rot_matrix(...)` trial basis, the recovery-enabled outer-step runtime became slower again, so the next iteration should likely focus on keeping this more faithful basis while recovering enough performance to benchmark it cleanly.

### Step 8 — FFT bridge / mechanics snapshot interface

Status: `strong partial`

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
- the import-side mechanics path now also has a direct bridge-contract validator:
  - `compare_applied_legacy_fft_snapshot_to_mesh_state(...)`
  - [validate_faithful_fft2elle_bridge.py](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model/validate_faithful_fft2elle_bridge.py)
  - it checks imported Euler fields, unode positions, boundary-node positions, cell reset/shear state, `tex.out` channels, DD import semantics, and the stored runtime snapshot payload against one frozen mechanics snapshot
- a named bridge adapter layer now mirrors the old import contract more explicitly:
  - cell reset payload from `temp-FFT.out` / `temp.out`
  - aligned Euler import
  - aligned unode strain increments
  - explicit `tex.out` channel mapping for strain/stress/activity/DD/FFT identifiers
  - a first `FS_CheckUnodes`-style host-flynn sync for the faithful label field after moved unode positions, so mechanics import can now update the runtime tracer/ownership field from the current mesh instead of leaving it stale
  - a first literal `FS_CheckUnodes`-style swept-unode repair after mechanics import, so moved unodes can now inherit `U_EULER_3` from the nearest valid donor in the new host flynn and have `U_DISLOCDEN` reset to zero instead of keeping stale pre-sweep values
  - legacy tracer initialization when missing, so mechanics import can now materialize `U_ATTRIB_C` and `F_ATTRIB_C` host-flynn surfaces instead of assuming they already exist in the seed
- the mechanics import path now also mirrors the old `FS_fft2elle` DD controls more closely:
  - explicit `ImportDDs`-style enable/disable behavior for `tex.out` dislocation-density increments
  - explicit `ExcludePhaseID`-style zeroing of imported DD in a chosen phase during the mechanics stage
  - explicit DD overwrite vs increment mode, matching the split between the older `processes/fft/fft2elle` path and the later utility / `FS_fft2elle` additive path
  - faithful runtime/CLI threading of those bridge controls through saved outer steps
- the mechanics import path now also mirrors both legacy host-repair families:
  - later `FS_CheckUnodes`-style swept-unode repair, including nearest-donor Euler reassignment and DD reset for swept sites
  - older `check_error`-style tracer repair, including host-flynn `U_ATTRIB_C` correction and donor Euler reassignment without DD reset
  - faithful runtime/CLI threading and validator coverage for the selected host-repair mode
- the mechanics stage now also applies the first geometry-side bridge effects from `FS_fft2elle`:
  - seed-unode positions are updated from `unodexyz.out` using the old pure-shear vs simple-shear Y-update gate
  - mesh boundary-node positions are now updated from nearby imported unode displacements using a first `SetBnodeStrain(...)`-style weighted interpolation over neighbouring flynns
  - runtime ELLE cell-box / shear-offset state is updated from the `temp-FFT.out` payload in the old `ResetCell(...)` style
  - exported ELLE snapshots can therefore carry the imported mechanics geometry state instead of only imported mechanics scalar fields
- the export-side half of the bridge now also exists:
  - `build_legacy_elle2fft_bridge_payload(...)`
  - `load_legacy_elle2fft_bridge_payload(...)`
  - `compare_legacy_elle2fft_bridge_payload(...)`
  - `write_legacy_elle2fft_bridge_payload(...)`
- the export adapter now mirrors both legacy phase-ID conventions:
  - `VISCOSITY`-based phase IDs for `FS_elle2fft`
  - `DISLOCDEN`-based phase IDs for the shipped `processes/fft/elle2fft` utility path
- the export adapter now also mirrors the legacy `FS_ExcludeFlynns` branch:
  - `include_grain_headers=False` writes `grain_count = 0`
  - FFT point grain numbers are exported as `0`
- the shipped legacy `processes/fft/example/step0` files now anchor a real export-side parity check:
  - faithful point rows match the shipped `make.out` point block
  - faithful `temp.out` rows match the shipped `temp.out` payload
  - the bridge comparison helper now classifies this as a header-only mismatch instead of leaving it as a manual diff
  - the adapter can round-trip its own `make.out` / `temp.out` output cleanly
- faithful outer-loop support for an explicit mechanics stage per saved outer step when one frozen snapshot or a sequence of legacy bridge snapshots is provided
- mechanics-only replay is now possible, so frozen legacy mechanics snapshots can be replayed across saved outer steps without forcing an artificial GBM stage into the parity case
- a dedicated mechanics-only transition validator now exists, so a frozen mechanics replay can be compared directly against a legacy before/after ELLE pair without hand-assembling the replay and transition report
- a dedicated frozen-mechanics outer-step validator now exists, so one saved outer step with faithful GBM/recovery can be compared directly against a legacy ELLE transition from the same replay inputs

What is still missing:
- faithful deformation-to-ELLE coupling path
- direct one-step parity tests driven by real frozen old mechanics snapshots
- the guide acceptance is still not fully met because the repo does not yet ship a real `unodexyz.out` / `unodeang.out` / `tex.out` outer-step pair that can drive one full frozen-mechanics legacy transition end-to-end

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

Recent translation support added for this seam:
- [legacy_statistics.py](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model/elle_jax_model/legacy_statistics.py) now parses the committed legacy [old.stats](/home/bz1229682991/research/Elle/newcode/elle/processes/statistics/old.stats) bookkeeping fixture, including:
  - flynn number
  - legacy grain number mapping
  - split flags
  - cycle/age payloads
  - aggregate mapped/orphan flynn counts
- the same module now also summarizes saved modern `mesh_*.json` bookkeeping and compares it conservatively against the `old.stats` contract, side by side, for:
  - total flynn count
  - source-mapped vs orphan-like flynns
  - unique source-flynn count versus legacy grain count
  - multi-parent / split-related flynns
  - retained versus non-retained identities
- [benchmark_validation.py](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model/elle_jax_model/benchmark_validation.py) and [validate_benchmarks.py](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model/validate_benchmarks.py) now expose that bookkeeping comparison as an optional report section when a saved candidate `mesh_*.json` and legacy `old.stats` file are both provided.
- this does not fix Step 9 by itself, but it gives the branch a real legacy bookkeeping contract for future outer-loop and nucleation/cluster checks instead of relying only on visual or geometric symptoms

### Step 10 — Paper-level observables

Status: `partial`

What is in place:
- geometry validation
- benchmark reports
- Figure-2-style line / histogram / KDE validation views
- fabric tensor diagnostics from multicomponent Euler fields in [microstructure_validation.py](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model/elle_jax_model/microstructure_validation.py)
- `P`, `G`, and `R` fabric indices for c-axis, a-axis, and prism-normal summaries
- first c-axis pole-figure summaries:
  - normalized hemisphere histogram
  - mean colatitude
  - peak bin location
  - fraction within `15` degrees of the vertical axis
- aspect-ratio summaries and distributions for flynn polygons
- `second moment grain size` is now computed directly in [microstructure_validation.py](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model/elle_jax_model/microstructure_validation.py) from the old `processes/statistics` contract in [statistics.elle.cc](/home/bz1229682991/research/Elle/newcode/elle/processes/statistics/statistics.elle.cc)
- mechanics-side unode summaries when imported FFT fields are present:
  - mean normalized strain-rate proxy
  - mean normalized stress proxy
  - mean basal activity
  - mean prismatic activity
  - prismatic-to-basal and prismatic-fraction activity summaries
- first explicit mechanics-history curves in [benchmark_validation.py](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model/elle_jax_model/benchmark_validation.py):
  - cumulative normalized strain from snapshot mean strain-rate
  - stress-vs-strain curves
  - basal/prismatic/prismatic-fraction activity-vs-strain curves
- saved-run mechanics sidecars can now contribute a direct strain axis from outer-step mechanics payload state:
  - `mechanics_payload_summary` is preserved in saved mesh artifacts
  - validation prefers direct sidecar strain history when it exists
  - current direct strain-axis sources include:
    - `cumulative_simple_shear`
    - `vertical_shortening_pct` from the imported cell-reset path
  - benchmark curves can therefore use saved outer-step mechanics strain history instead of only reintegrating ELLE snapshot means
- saved-run mechanics sidecars can now also contribute direct stress history:
  - optional `mean_differential_stress` is carried through the microstructure summary and benchmark report
  - stress-strain curves prefer direct differential stress when it exists, and otherwise fall back to the older normalized-stress proxy
- saved-run mechanics sidecars can now also contribute optional pyramidal activity history:
  - `mean_pyramidal_activity` is carried through the microstructure summary and benchmark report when present
  - activity-vs-strain curves include pyramidal activity when that sidecar field exists
  - activity-curve acceptance and trend checks include pyramidal activity when it is available
- legacy `FS_statistics` mechanics outputs are now part of the validation surface:
  - the old `AllOutData.txt` contract is now factored into a reusable parser in [legacy_statistics.py](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model/elle_jax_model/legacy_statistics.py), rather than being embedded only inside one validator
  - the old `tmpstats.dat` summary contract from `processes/statistics` is now also parsed in [legacy_statistics.py](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model/elle_jax_model/legacy_statistics.py), including:
    - total grain number
    - average grain size
    - second moment grain size
    - orientation-bin summaries
    - total boundary length
    - Panozzo ratio / angle / accuracy fields
- the old `old.stats` contract is now parsed too, which exposes the legacy grain-number bookkeeping surface that sits between flynn activity and aggregate grain statistics
- the old [last.stats](/home/bz1229682991/research/Elle/newcode/elle/processes/statistics/last.stats) contract is now parsed too, and the benchmark path can optionally compare the final candidate snapshot directly against legacy committed summary statistics (`tmpstats.dat`, `last.stats`, or `old.stats`) on the shared metrics that the old statistics process actually wrote:
  - total grain number
  - average grain size
  - second moment grain size
  - [microstructure_validation.py](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model/elle_jax_model/microstructure_validation.py) now reads [AllOutData.txt](/home/bz1229682991/research/Elle/TwoWayIceModel_Release/elle/example/results/AllOutData.txt) when it is present beside an ELLE snapshot sequence
  - the row contract is source-backed by [FS_statistics.cc](/home/bz1229682991/research/Elle/newcode/elle/FS_Codes/FS_utilities/FS_statistics/FS_statistics.cc), where `ReadAllOutData(...)` and the appended `AllOutData.txt` rows carry:
    - von Mises stress / strain rate
    - differential stress
    - basal / prismatic / pyramidal activity
    - stress / strain-rate tensors
  - sequence validation now overlays those legacy mechanics rows onto the ordered saved ELLE snapshots when the row count matches the saved-step count, or matches `step1..N` after a `step0` seed snapshot
  - benchmark trend building can therefore use a real legacy mechanics-history source, not only saved faithful sidecars
  - benchmark comparison reports now also compare the legacy statistics fields directly when they exist:
    - von Mises stress
    - von Mises strain rate
    - stress-field error
    - strain-rate-field error
- benchmark mechanics-history curves now have a legacy-statistics fallback path:
  - if no direct strain axis or normalized strain-rate series exists, stress/activity curves can now integrate the legacy von Mises strain-rate history from `AllOutData.txt`
  - if no direct stress history exists, stress curves can now fall back to legacy von Mises stress after preferring differential stress and then normalized stress
- benchmark trend/comparison reports now also carry `second moment grain size`, so the old `tmpstats.dat` grain-size spread observable has a modern comparison surface instead of living only in ad hoc process output
- benchmark-side comparison/trend hooks for:
  - c-axis largest eigenvalue
  - c-axis `P` index
  - c-axis vertical-fraction pole metric
  - a-axis `P` index
  - mean aspect ratio
  - mean normalized stress
  - mean differential stress
  - mean prismatic activity
  - mean pyramidal activity
  - prismatic-to-basal activity ratio
- first explicit paper-signature assessment layer in [benchmark_validation.py](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model/elle_jax_model/benchmark_validation.py), with directional checks for:
  - coarsening
  - grain-count decline
  - equidimensionalization
  - c-axis strengthening / vertical clustering
  - mechanics-side stress and prismatic-activity trend agreement when those fields exist
- first grain-survival diagnostics in [microstructure_validation.py](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model/elle_jax_model/microstructure_validation.py):
  - per-initial-grain retention to final step
  - first zero-area step
  - correlation of survival with initial grain size
  - correlation of survival with an initial basal-Schmid proxy under the benchmark shortening axis
  - explicit `size stronger than Schmid` summary when the data support it
- benchmark reports now carry reference/candidate survival diagnostics and survival-summary deltas when the ELLE sequence contains a stable grain-label field
- release-dataset validation now has an explicit signature-assessment layer in [benchmark_validation.py](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model/elle_jax_model/benchmark_validation.py):
  - grain-size hotter-case-more-active check
  - per-Euler hotter-case-more-active checks
  - aggregate release-dataset pass counts
- the top-level benchmark report now emits aggregate acceptance across:
  - static grain-growth paper-signature checks
  - release-dataset signature checks
- benchmark validation now also has an explicit experiment-family suite layer for local `0`, `1`, `10`, `25` runs in [benchmark_validation.py](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model/elle_jax_model/benchmark_validation.py):
  - family-0 near-constant mean-area check
  - higher-DRX monotonic checks for end mean grain area, aspect ratio, and stress
  - higher-DRX earlier-peak-stress check from mechanics stress-strain curves
  - all-family c-axis strengthening / vertical-clustering checks
  - all-family `size stronger than Schmid` survival check
- [validate_benchmarks.py](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model/validate_benchmarks.py) can now ingest repeated `--experiment-family FAMILY=DIR` entries and emit the family-suite acceptance summary in the benchmark report
- the benchmark CLI can now also build that family suite from:
  - a JSON family manifest
  - a root directory with discoverable family subdirectories
  - repeated `--experiment-family-report FAMILY=JSON` benchmark-report inputs
  - a JSON manifest mapping family IDs to benchmark report JSONs
  - a root directory with discoverable family benchmark report JSONs
  which makes the family-suite path usable against real local run layouts instead of only synthetic test fixtures
- there is now a dedicated family-suite entrypoint in [validate_experiment_families.py](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model/validate_experiment_families.py), so the `0/1/10/25` report can be generated directly without also running the generic static/reference/release benchmark stack
- the family-suite benchmark layer in [benchmark_validation.py](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model/elle_jax_model/benchmark_validation.py) can now consume either live family run directories or saved benchmark report JSONs, which gives Step 10 a real non-directory path to end-to-end family acceptance on the current repo state

What is missing:
- real bridge-fed pyramidal activity data beyond the current optional sidecar path
- broader use of legacy statistics outputs beyond `AllOutData.txt`, if additional committed old-output fixtures appear
- broader experiment-family acceptance beyond the current local family-suite checks, especially once real benchmark runs for `0`, `1`, `10`, and `25` exist in a stable committed layout

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
