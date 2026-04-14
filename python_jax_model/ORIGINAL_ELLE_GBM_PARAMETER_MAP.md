# Original ELLE GBM Parameter Map

This note maps the parameters we currently expose in the Python rewrite to the
way original ELLE GBM actually treats them.

The main reference files are:

- [FS_gbm_pp_fft.main.cc](/home/bz1229682991/research/Elle/newcode/elle/FS_Codes/FS_recrystallisation/FS_gbm_pp_fft/FS_gbm_pp_fft.main.cc)
- [FS_gbm_pp_fft.elle.cc](/home/bz1229682991/research/Elle/newcode/elle/FS_Codes/FS_recrystallisation/FS_gbm_pp_fft/FS_gbm_pp_fft.elle.cc)
- [FS_movenode_pp_fft.cc](/home/bz1229682991/research/Elle/newcode/elle/FS_Codes/FS_recrystallisation/FS_gbm_pp_fft/FS_movenode_pp_fft.cc)
- [FS_topocheck.elle.cc](/home/bz1229682991/research/Elle/newcode/elle/FS_Codes/FS_utilities/FS_topocheck/FS_topocheck.elle.cc)
- [GBMUnodeUpdate.cc](/home/bz1229682991/research/Elle/newcode/elle/basecode/GBMUnodeUpdate.cc)

## Quick Take

Original ELLE GBM is a moving-boundary method, not a phase-field PDE. So some
rewrite parameters have a clean ELLE counterpart, while others are only
analogue-era leftovers and should not be treated as physically meaningful for
the faithful branch.

## 1. Rewrite Parameters With A Real ELLE Counterpart

| Rewrite parameter | Original ELLE counterpart | How ELLE treats it | Current faithful rewrite status |
| --- | --- | --- | --- |
| `mesh_movement_model="elle_surface"` | `GetMoveDir`, `MoveDNode`, `MoveTNode` | Node motion is computed from trial energy differences, local segment geometry, mobility, timestep, speedup, and switch distance. | Approximate but increasingly faithful. |
| `mesh_surface_diagonal_trials` | `diagonal` user option | `FS_gbm_pp_fft.main.cc` exposes `diagonal` through `ElleSetOptNames(...)`. This controls whether diagonal trial positions are used. | Real mapping. |
| `mesh_use_elle_physical_units` | `ElleUnitLength()` usage in motion law | ELLE scales segment lengths and final node velocity by physical unit length directly in `MoveDNode` and `MoveTNode`. | Real intent, but our flag exists only because the rewrite originally dropped this path. |
| `mesh_relax_steps` | one GBM node-motion pass per stage | ELLE loops over active nodes during a growth stage. There is no exact same named parameter, but repeated motion passes are the closest scheduling analogue. | Approximate scheduling control. |
| `mesh_topology_steps` | `ElleGGTopoChecks()` / `FS_topocheck` pass count | ELLE runs topology correction after movement. Again, not exposed as the same parameter, but conceptually similar. | Approximate scheduling control. |
| `dt` | `ElleTimestep()` | ELLE uses a real physical timestep in node movement. It is part of `dt = ElleTimestep() * ElleSpeedup()`, then clamped against `SwitchDistance`. | In the current `mesh_only` faithful branch this is mostly legacy and not the active driver yet. |
| `SwitchDistance` | `ElleSwitchdistance()` | Hard movement cap. If a node would move too far, ELLE resets speedup and then locally reduces timestep to 90% of the maximum allowed. | Real mapping, currently imported from the seed case. |
| `SpeedUp` | `ElleSpeedup()` | Runtime multiplier on the timestep. ELLE can reset it to `1.0` if motion would be too large. | Real mapping, currently imported from the seed case. |
| `UnitLength` | `ElleUnitLength()` | Converts ELLE-length units to physical units in movement and mass transfer. | Real mapping, currently imported from the seed case. |
| `BoundaryWidth` | `ElleBndWidth()` | Used by `GBMUnodeUpdate.cc` to define the physical grain-boundary band for swept/enriched mass transfer. | Real mapping on the faithful unode-transfer side. |
| `phase-pair mobility` | `phases->pairs[p1][p2].mobility` from `phase_db.txt` | ELLE stores base mobility per phase pair and then adjusts it in `GetBoundaryMobility(...)`. | Only partially ported; our faithful branch does not yet use the full phase-db path. |
| `activation energy` | `phases->pairs[p1][p2].dGbActEn` from `phase_db.txt` | ELLE uses an Arrhenius law in `GetBoundaryMobility(...)`. | Not yet fully ported into the faithful branch. |
| `temperature` | `ElleTemperature()` | ELLE computes mobility from temperature-dependent Arrhenius scaling. | Not yet fully ported into the faithful branch. |
| `MinNodeSeparation`, `MaxNodeSeparation` | ELLE node-spacing controls | Used in topology maintenance and node splitting/collapse behavior. | Real mapping, currently imported from the seed case. |

## 2. Rewrite Parameters That Are Mostly Analogue Or Legacy

These come from the older phase-field/JAX prototype. They are meaningful for the
analogue branch, but they are not really part of original ELLE GBM.

| Rewrite parameter | ELLE equivalent | What this means in practice |
| --- | --- | --- |
| `gradient_penalty` | none | Phase-field-only. Original ELLE GBM does not evolve a diffuse interface PDE. |
| `interaction_strength` | none | Phase-field-only. No direct ELLE GBM counterpart. |
| global PDE `mobility` | none as a single scalar | ELLE uses per-boundary mobility, not one global phase-field coefficient. |
| `mesh_feedback_strength` | none in `mesh_only` | In the faithful `mesh_only` branch this is effectively inert metadata, not a real physical control. |
| `mesh_transport_strength` | none in `mesh_only` | Same story: useful for the analogue branch, not an ELLE GBM control. |
| `mesh_kernel_every` / `mesh_kernel_strength` / `mesh_kernel_corrector` | none | JAX/PDE coupling machinery, not part of original GBM. |
| `init_noise` | none | Useful for synthetic phase-field initialization, but not for faithful ELLE seeding. |
| `init_smoothing_steps` | none | Same as above. For a faithful ELLE seed, this should normally stay `0`. |

## 3. Important Original ELLE Controls We Still Need To Treat More Literally

These are real original GBM controls or behaviors that matter, but are still
missing or only partially translated in the faithful rewrite.

| Original ELLE control or behavior | Where it lives | Rewrite status |
| --- | --- | --- |
| per-phase-pair mobility table | `phase_db.txt`, `Read2PhaseDb(...)` in [FS_gbm_pp_fft.elle.cc](/home/bz1229682991/research/Elle/newcode/elle/FS_Codes/FS_recrystallisation/FS_gbm_pp_fft/FS_gbm_pp_fft.elle.cc) | Not fully ported. |
| Arrhenius mobility scaling with temperature and activation energy | `GetBoundaryMobility(...)` in [FS_movenode_pp_fft.cc](/home/bz1229682991/research/Elle/newcode/elle/FS_Codes/FS_recrystallisation/FS_gbm_pp_fft/FS_movenode_pp_fft.cc) | Not fully ported. |
| misorientation-dependent mobility reduction | `Get2BoundaryMobility(...)` in [FS_movenode_pp_fft.cc](/home/bz1229682991/research/Elle/newcode/elle/FS_Codes/FS_recrystallisation/FS_gbm_pp_fft/FS_movenode_pp_fft.cc) | Only partially represented. |
| `ExcludePhase` behavior | user option in [FS_gbm_pp_fft.main.cc](/home/bz1229682991/research/Elle/newcode/elle/FS_Codes/FS_recrystallisation/FS_gbm_pp_fft/FS_gbm_pp_fft.main.cc) | Not yet surfaced cleanly. |
| `Start Timestep` behavior | user option in [FS_gbm_pp_fft.main.cc](/home/bz1229682991/research/Elle/newcode/elle/FS_Codes/FS_recrystallisation/FS_gbm_pp_fft/FS_gbm_pp_fft.main.cc) | Not yet surfaced cleanly. |
| `afactor` / redistribution control | user option in [FS_gbm_pp_fft.main.cc](/home/bz1229682991/research/Elle/newcode/elle/FS_Codes/FS_recrystallisation/FS_gbm_pp_fft/FS_gbm_pp_fft.main.cc) | Not yet ported. |
| full `PartitionMass` / `EnrichUnodes` bookkeeping | [GBMUnodeUpdate.cc](/home/bz1229682991/research/Elle/newcode/elle/basecode/GBMUnodeUpdate.cc) | Much closer now, but still not fully literal. |
| exact topocheck event ordering and thresholds | [FS_topocheck.elle.cc](/home/bz1229682991/research/Elle/newcode/elle/FS_Codes/FS_utilities/FS_topocheck/FS_topocheck.elle.cc) | Partially ported. |

## 4. What This Means For Validation

If we want a general GBM rewrite, we should avoid leaning on the analogue-era
phase-field knobs when evaluating the faithful branch.

The best current interpretation is:

- trusted ELLE-native inputs:
  - seed mesh geometry
  - seed unodes
  - `SwitchDistance`
  - `SpeedUp`
  - `UnitLength`
  - `BoundaryWidth`
  - node-separation controls
- partially faithful controls:
  - node movement law
  - topocheck ordering
  - unode mass/ownership update
- not trustworthy as general GBM controls:
  - phase-field coefficients
  - mesh feedback strength
  - mesh transport strength

## 5. Current Faithful Runner Naming

The dedicated faithful CLI at
[run_gbm_faithful.py](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model/run_gbm_faithful.py)
now uses names that are meant to reflect the rewrite's real GBM-stage role more
honestly:

| Faithful CLI name | Meaning in the rewrite | Closest ELLE interpretation |
| --- | --- | --- |
| `motion_passes` | number of node-motion passes per outer stage | one GBM movement sweep over active bnodes |
| `topology_passes` | number of cleanup passes after motion | one topology-repair stage after movement |
| `stage_interval` | how often a faithful GBM stage is applied in the outer loop | closest stand-in for “perform one GBM stage each saved step” |
| `raster_boundary_band` | integer grid support band used by the rewritten raster ownership path | no literal ELLE parameter; only a rewrite-side raster aid |

The older names like `mesh_relax_steps`, `mesh_topology_steps`, and
`mesh_feedback_every` are still accepted internally for compatibility, but they
should be read as rewrite scheduling controls, not original ELLE file
parameters.

## 6. Current Preset Split

The faithful branch now has two clearer roles:

1. `fine-foam-truthful-mesh`
   This stays as the benchmark-oriented reproduction preset.
2. `gbm-faithful-default`
   This is the cleaner general preset in the mixed runner. It keeps the
   ELLE-seeded `mesh_only` GBM structure and honest stage controls, while
   leaving the old phase-field coefficients out of the public faithful story.
   Its raster boundary support now derives from the seed file's physical
   `BoundaryWidth` and `UnitLength` when those are available, instead of being
   presented only as a hand-picked integer.
   In the current faithful `mesh_only` path, changing the old phase-field
   coefficients does not change the evolved state, which is exactly the
   separation we want short-term.

## 7. Recommended Next Cleanup

For the faithful branch, the next clean parameter/modeling steps should be:

1. avoid advertising phase-field-only knobs as meaningful for the faithful path
2. port the remaining original mobility inputs before any more case-by-case tuning
3. tie more of the faithful public CLI directly to original ELLE names where possible
