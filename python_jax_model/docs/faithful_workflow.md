# Faithful Workflow Runner

`python_jax_model/run_faithful_workflow.py` is a checkpoint runner for the faithful branch guide at `/home/bz1229682991/FAITHFUL_BRANCH_AGENT_GUIDE.md`.

It is meant to automate the part we can reliably automate:
- ordered checkpoint execution
- dependency closure when you target a later slice like Step 9 to 10
- targeted verification commands
- resumable state
- in-progress state updates while long commands are still running
- benchmark smoke runs
- machine-readable checkpoint results under `python_jax_model/validation/workflow/`
- parallel command execution for checkpoints that are explicitly marked safe to split

It is intentionally not an autonomous code-writing agent.
Guide steps that still require human or agent implementation judgment are recorded as `manual` checkpoints instead of being faked.

## Examples

List the automated checkpoints:

```bash
python python_jax_model/run_faithful_workflow.py --list
```

Run the default automated chain and keep going after failures:

```bash
python python_jax_model/run_faithful_workflow.py
```

Run only the recovery-through-validation slice:

```bash
python python_jax_model/run_faithful_workflow.py --from-step 7 --to-step 10
```

Run the Step 9 to 10 slice and let Step 10 validations fan out in parallel:

```bash
python python_jax_model/run_faithful_workflow.py --from-step 9 --to-step 10 --command-workers 2
```

Dry-run the plan without executing commands:

```bash
python python_jax_model/run_faithful_workflow.py --dry-run --include-manual
```

Resume from an existing state file:

```bash
python python_jax_model/run_faithful_workflow.py --resume-json python_jax_model/validation/workflow/<session>/faithful_workflow_state.json
```

## What it currently automates

- `step01_runtime_skeleton`
- `step02_legacy_reference`
- `step03_topology_parity`
- `step04_gbm_force`
- `step05_mobility`
- `step06_swept_unodes`
- `step07_recovery`
- `step09_outer_loop_smoke`
- `step10_validation`
- `step12_ci_guardrails`

## What it records as manual

- `step00_docs_split`
- `step08_fft_bridge`
- `step11_benchmark_reproduction`
- `step13_optimize_after_parity`

These remain in the workflow state so the run log still matches the guide order.

## Parallel behavior

The runner stays conservative by default.
It only runs commands in parallel when a checkpoint is explicitly marked safe for that, and you opt in with `--command-workers > 1`.

Right now that mainly helps validation checkpoints such as Step 10, where the Figure-2 report and legacy-transition comparison both depend on the Step 9 candidate but not on each other.

## Progress visibility

The workflow state JSON is written before a checkpoint starts, while each command is running, and after each command finishes.
That means a long Step 9 outer-loop smoke should show:

- checkpoint `status: "running"`
- per-command `status: "running"` for the active command
- final command output once the command finishes

This is mainly there so long faithful smoke runs do not look stalled when they are simply busy.
