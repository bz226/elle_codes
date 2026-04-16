from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Callable


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_path(path: Path) -> Path:
    return Path(path).expanduser().absolute()


@dataclass(frozen=True)
class WorkflowContext:
    repo_root: Path
    project_root: Path
    python_bin: Path
    guide_path: Path
    status_path: Path
    session_dir: Path
    fine_foam_init_elle: Path | None
    fine_foam_reference_dir: Path | None
    data_dir: Path | None
    legacy_bundle_json: Path | None
    legacy_transition_json: Path | None


@dataclass(frozen=True)
class WorkflowCommand:
    name: str
    argv: tuple[str, ...]
    cwd: Path
    timeout_sec: int = 1800


CommandBuilder = Callable[[WorkflowContext], list[WorkflowCommand]]


@dataclass(frozen=True)
class WorkflowCheckpoint:
    step_number: int
    checkpoint_id: str
    title: str
    description: str
    references: tuple[str, ...]
    automated: bool
    dependencies: tuple[str, ...] = ()
    parallel_commands: bool = False
    command_builder: CommandBuilder | None = None


def default_workflow_context(
    *,
    project_root: Path | None = None,
    python_bin: Path | None = None,
    session_dir: Path | None = None,
    guide_path: Path | None = None,
    status_path: Path | None = None,
    fine_foam_init_elle: Path | None = None,
    fine_foam_reference_dir: Path | None = None,
    data_dir: Path | None = None,
    legacy_bundle_json: Path | None = None,
    legacy_transition_json: Path | None = None,
) -> WorkflowContext:
    resolved_project_root = (
        _normalize_path(Path(project_root))
        if project_root is not None
        else _normalize_path(Path(__file__).parents[1])
    )
    repo_root = resolved_project_root.parent
    resolved_session_dir = (
        _normalize_path(Path(session_dir))
        if session_dir is not None
        else resolved_project_root / "validation" / "workflow" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    preferred_python = None
    if python_bin is None:
        venv_python = repo_root / ".venv" / "bin" / "python"
        if venv_python.exists():
            preferred_python = _normalize_path(venv_python)
    return WorkflowContext(
        repo_root=repo_root,
        project_root=resolved_project_root,
        python_bin=(
            _normalize_path(Path(python_bin))
            if python_bin is not None
            else preferred_python if preferred_python is not None else _normalize_path(Path(sys.executable))
        ),
        guide_path=_normalize_path(Path(guide_path)) if guide_path is not None else Path("/home/bz1229682991/FAITHFUL_BRANCH_AGENT_GUIDE.md"),
        status_path=_normalize_path(Path(status_path)) if status_path is not None else resolved_project_root / "FAITHFUL_BRANCH_STATUS.md",
        session_dir=resolved_session_dir,
        fine_foam_init_elle=(
            _normalize_path(Path(fine_foam_init_elle))
            if fine_foam_init_elle is not None
            else Path("/home/bz1229682991/research/Elle/TwoWayIceModel_Release/elle/example/results/fine_foam_step001.elle")
        ),
        fine_foam_reference_dir=(
            _normalize_path(Path(fine_foam_reference_dir))
            if fine_foam_reference_dir is not None
            else Path("/home/bz1229682991/research/Elle/TwoWayIceModel_Release/elle/example/results")
        ),
        data_dir=(
            _normalize_path(Path(data_dir))
            if data_dir is not None
            else Path("/home/bz1229682991/research/Elle/TwoWayIceModel_Release/data")
        ),
        legacy_bundle_json=(
            _normalize_path(Path(legacy_bundle_json))
            if legacy_bundle_json is not None
            else resolved_project_root / "legacy_reference" / "testdata" / "fft_example_step0_reference.json"
        ),
        legacy_transition_json=(
            _normalize_path(Path(legacy_transition_json))
            if legacy_transition_json is not None
            else resolved_project_root / "legacy_reference" / "testdata" / "fine_foam_outerstep_001_to_002_transition.json"
        ),
    )


def _unittest_command(context: WorkflowContext, name: str, *tests: str) -> list[WorkflowCommand]:
    return [
        WorkflowCommand(
            name=name,
            argv=(str(context.python_bin), "-m", "unittest", *tests, "-v"),
            cwd=context.repo_root,
            timeout_sec=1800,
        )
    ]


def _step09_commands(context: WorkflowContext) -> list[WorkflowCommand]:
    if context.fine_foam_init_elle is None or context.fine_foam_reference_dir is None or context.data_dir is None:
        return []
    outdir = context.session_dir / "step09_outer_loop_smoke"
    benchmark_json = context.session_dir / "step09_outer_loop_smoke_benchmark.json"
    return [
        WorkflowCommand(
            name="faithful outer-loop smoke",
            argv=(
                str(context.python_bin),
                "python_jax_model/run_gbm_faithful.py",
                "--init-elle",
                str(context.fine_foam_init_elle),
                "--steps",
                "1",
                "--save-every",
                "1",
                "--include-step0",
                "--save-elle",
                "--track-topology",
                "--no-preview",
                "--subloops-per-snapshot",
                "10",
                "--nucleation-steps-per-subloop",
                "2",
                "--gbm-steps-per-subloop",
                "2",
                "--recovery-steps-per-subloop",
                "2",
                "--outdir",
                str(outdir),
            ),
            cwd=context.repo_root,
            timeout_sec=7200,
        ),
        WorkflowCommand(
            name="outer-loop benchmark validation",
            argv=(
                str(context.python_bin),
                "python_jax_model/validate_benchmarks.py",
                "--reference-dir",
                str(context.fine_foam_reference_dir),
                "--candidate-dir",
                str(outdir),
                "--data-dir",
                str(context.data_dir),
                "--pattern",
                "*.elle",
                "--json-out",
                str(benchmark_json),
            ),
            cwd=context.repo_root,
            timeout_sec=3600,
        ),
    ]


def _step10_commands(context: WorkflowContext) -> list[WorkflowCommand]:
    if context.fine_foam_reference_dir is None or context.legacy_transition_json is None:
        return []
    candidate_dir = context.session_dir / "step09_outer_loop_smoke"
    before = candidate_dir / "grain_unodes_00000.elle"
    after = candidate_dir / "grain_unodes_00001.elle"
    figure2_html = context.session_dir / "step10_figure2_line.html"
    figure2_json = context.session_dir / "step10_figure2_line.json"
    transition_json = context.session_dir / "step10_transition_compare.json"
    return [
        WorkflowCommand(
            name="figure-2 validation",
            argv=(
                str(context.python_bin),
                "python_jax_model/validate_figure2_line.py",
                "--reference-dir",
                str(context.fine_foam_reference_dir),
                "--candidate-dir",
                str(candidate_dir),
                "--pattern",
                "*.elle",
                "--json-out",
                str(figure2_json),
                "--html-out",
                str(figure2_html),
            ),
            cwd=context.repo_root,
            timeout_sec=3600,
        ),
        WorkflowCommand(
            name="legacy outer-step transition comparison",
            argv=(
                str(context.python_bin),
                "python_jax_model/validate_legacy_reference_transition.py",
                "--reference-json",
                str(context.legacy_transition_json),
                "--before",
                str(before),
                "--after",
                str(after),
                "--json-out",
                str(transition_json),
            ),
            cwd=context.repo_root,
            timeout_sec=1800,
        ),
    ]


def build_faithful_workflow(context: WorkflowContext) -> list[WorkflowCheckpoint]:
    return [
        WorkflowCheckpoint(
            step_number=0,
            checkpoint_id="step00_docs_split",
            title="Docs split + faithful-only public story",
            description="Documentation split is mostly human-reviewed; this checkpoint is recorded but not fully auto-verifiable.",
            references=(str(context.guide_path), str(context.status_path)),
            automated=False,
        ),
        WorkflowCheckpoint(
            step_number=1,
            checkpoint_id="step01_runtime_skeleton",
            title="Faithful-only runtime skeleton",
            description="Verify faithful setup/runtime decoupling from the prototype path.",
            references=("python_jax_model/run_gbm_faithful.py", "python_jax_model/elle_jax_model/faithful_runtime.py"),
            automated=True,
            command_builder=lambda ctx: _unittest_command(
                ctx,
                "step01 runtime skeleton tests",
                "python_jax_model.tests.test_simulation.SimulationTests.test_build_faithful_gbm_setup_uses_truthful_mesh_defaults",
                "python_jax_model.tests.test_simulation.SimulationTests.test_run_faithful_gbm_simulation_exposes_numpy_mesh_only_backend",
                "python_jax_model.tests.test_simulation.SimulationTests.test_run_faithful_gbm_simulation_can_emit_initial_snapshot",
            ),
        ),
        WorkflowCheckpoint(
            step_number=2,
            checkpoint_id="step02_legacy_reference",
            title="Golden-reference harness",
            description="Verify committed legacy snapshot and transition fixtures still compare cleanly.",
            references=("python_jax_model/build_legacy_reference_bundle.py", "python_jax_model/legacy_reference/testdata"),
            automated=True,
            command_builder=lambda ctx: _unittest_command(
                ctx,
                "step02 legacy reference tests",
                "python_jax_model.tests.test_simulation.SimulationTests.test_compare_legacy_reference_bundle_matches_fft_example_fixture",
                "python_jax_model.tests.test_simulation.SimulationTests.test_fine_foam_outerstep_transition_fixture_matches_reference_pair",
            ),
        ),
        WorkflowCheckpoint(
            step_number=3,
            checkpoint_id="step03_topology_parity",
            title="Node motion + topology ordering",
            description="Run focused topology/order regressions for faithful mesh motion.",
            references=("python_jax_model/elle_jax_model/mesh.py",),
            automated=True,
            command_builder=lambda ctx: _unittest_command(
                ctx,
                "step03 topology tests",
                "python_jax_model.tests.test_simulation.SimulationTests.test_relax_mesh_state_elle_surface_runs_local_topology_after_each_moved_node",
                "python_jax_model.tests.test_simulation.SimulationTests.test_mesh_topology_maintenance_switches_short_triple_edge",
                "python_jax_model.tests.test_simulation.SimulationTests.test_mesh_topology_maintenance_collapses_short_double_nodes",
                "python_jax_model.tests.test_simulation.SimulationTests.test_mesh_topology_merges_tiny_two_sided_flynn",
            ),
        ),
        WorkflowCheckpoint(
            step_number=4,
            checkpoint_id="step04_gbm_force",
            title="Literal GBM driving force",
            description="Verify stored-energy and swept-area force regressions.",
            references=("python_jax_model/elle_jax_model/mesh.py",),
            automated=True,
            command_builder=lambda ctx: _unittest_command(
                ctx,
                "step04 gbm force tests",
                "python_jax_model.tests.test_simulation.SimulationTests.test_surface_force_from_trial_energies_can_include_stored_energy_term",
                "python_jax_model.tests.test_simulation.SimulationTests.test_trial_swept_area_unions_same_side_triangles",
                "python_jax_model.tests.test_simulation.SimulationTests.test_roi_weighted_label_density_prefers_same_label_support_before_fallback",
            ),
        ),
        WorkflowCheckpoint(
            step_number=5,
            checkpoint_id="step05_mobility",
            title="Literal mobility law",
            description="Verify phase-db-driven mobility behavior.",
            references=("python_jax_model/elle_jax_model/mobility.py",),
            automated=True,
            command_builder=lambda ctx: _unittest_command(
                ctx,
                "step05 mobility tests",
                "python_jax_model.tests.test_simulation.SimulationTests.test_boundary_segment_mobility_matches_old_arrhenius_formula",
                "python_jax_model.tests.test_simulation.SimulationTests.test_build_edge_mobility_lookup_uses_phase_pairs_and_misorientation",
                "python_jax_model.tests.test_simulation.SimulationTests.test_misorientation_mobility_reduction_matches_holm_style_cutoff",
            ),
        ),
        WorkflowCheckpoint(
            step_number=6,
            checkpoint_id="step06_swept_unodes",
            title="Literal swept-unode update",
            description="Verify ownership, donor, and reset semantics for swept unodes.",
            references=("python_jax_model/elle_jax_model/mesh.py",),
            automated=True,
            command_builder=lambda ctx: _unittest_command(
                ctx,
                "step06 swept-unode tests",
                "python_jax_model.tests.test_simulation.SimulationTests.test_update_seed_unode_fields_resets_swept_u_dislocden",
                "python_jax_model.tests.test_simulation.SimulationTests.test_update_seed_unode_sections_uses_nearest_same_label_donor_for_u_euler_3",
                "python_jax_model.tests.test_simulation.SimulationTests.test_update_seed_unode_sections_uses_flynn_mean_fallback_when_no_safe_donor_exists",
                "python_jax_model.tests.test_simulation.SimulationTests.test_assign_seed_unodes_from_mesh_fills_polygon_gaps_from_rasterized_mesh",
            ),
        ),
        WorkflowCheckpoint(
            step_number=7,
            checkpoint_id="step07_recovery",
            title="Recovery module",
            description="Verify faithful recovery behavior and symmetry-aware rotations.",
            references=("python_jax_model/elle_jax_model/recovery.py",),
            automated=True,
            command_builder=lambda ctx: _unittest_command(
                ctx,
                "step07 recovery tests",
                "python_jax_model.tests.test_simulation.SimulationTests.test_apply_recovery_stage_adds_u_attrib_f_without_reducing_density_on_first_stage",
                "python_jax_model.tests.test_simulation.SimulationTests.test_apply_recovery_stage_reduces_u_dislocden_after_first_stage",
                "python_jax_model.tests.test_simulation.SimulationTests.test_symmetry_aware_recovery_misorientation_collapses_hex_equivalent_rotation",
                "python_jax_model.tests.test_simulation.SimulationTests.test_legacy_recovery_trial_rotation_matches_old_trial_matrix_for_identity_orientation",
            ),
        ),
        WorkflowCheckpoint(
            step_number=8,
            checkpoint_id="step08_fft_bridge",
            title="FFT bridge / mechanics snapshot ingest",
            description="Guide step exists, but there is not yet a stable automated verifier in-repo.",
            references=(str(context.guide_path),),
            automated=False,
        ),
        WorkflowCheckpoint(
            step_number=9,
            checkpoint_id="step09_outer_loop_smoke",
            title="Faithful outer loop smoke",
            description="Run a one-step faithful fine_foam smoke plus benchmark scoring.",
            references=("python_jax_model/run_gbm_faithful.py", "python_jax_model/validate_benchmarks.py"),
            automated=True,
            dependencies=("step07_recovery",),
            command_builder=_step09_commands,
        ),
        WorkflowCheckpoint(
            step_number=10,
            checkpoint_id="step10_validation",
            title="Paper-level validation smoke",
            description="Run figure-style validation and legacy transition comparison against the step-9 candidate.",
            references=("python_jax_model/validate_figure2_line.py", "python_jax_model/validate_legacy_reference_transition.py"),
            automated=True,
            dependencies=("step09_outer_loop_smoke",),
            parallel_commands=True,
            command_builder=_step10_commands,
        ),
        WorkflowCheckpoint(
            step_number=11,
            checkpoint_id="step11_benchmark_reproduction",
            title="Benchmark reproduction",
            description="This is still a campaign-level task; the runner records it as manual until a stable suite exists.",
            references=(str(context.guide_path),),
            automated=False,
        ),
        WorkflowCheckpoint(
            step_number=12,
            checkpoint_id="step12_ci_guardrails",
            title="CI-style full regression suite",
            description="Run the full faithful test suite as a checkpoint gate.",
            references=("python_jax_model/tests/test_simulation.py",),
            automated=True,
            dependencies=("step10_validation",),
            command_builder=lambda ctx: [
                WorkflowCommand(
                    name="step12 full simulation suite",
                    argv=(str(ctx.python_bin), "-m", "unittest", "python_jax_model.tests.test_simulation", "-v"),
                    cwd=ctx.repo_root,
                    timeout_sec=7200,
                )
            ],
        ),
        WorkflowCheckpoint(
            step_number=13,
            checkpoint_id="step13_optimize_after_parity",
            title="Optimize after parity",
            description="Optimization stays manual until parity gates are green.",
            references=(str(context.guide_path),),
            automated=False,
        ),
    ]


def select_checkpoints(
    checkpoints: list[WorkflowCheckpoint],
    *,
    from_step: int | None = None,
    to_step: int | None = None,
    only_ids: tuple[str, ...] | None = None,
    include_manual: bool = False,
) -> list[WorkflowCheckpoint]:
    lookup = {checkpoint.checkpoint_id: checkpoint for checkpoint in checkpoints}
    selected = checkpoints
    if from_step is not None:
        selected = [checkpoint for checkpoint in selected if checkpoint.step_number >= int(from_step)]
    if to_step is not None:
        selected = [checkpoint for checkpoint in selected if checkpoint.step_number <= int(to_step)]
    if only_ids:
        only_set = {str(value) for value in only_ids}
        selected = [checkpoint for checkpoint in selected if checkpoint.checkpoint_id in only_set]
    selected_ids = {checkpoint.checkpoint_id for checkpoint in selected}
    pending_dependency_ids = [
        dependency
        for checkpoint in selected
        for dependency in checkpoint.dependencies
    ]
    while pending_dependency_ids:
        dependency_id = pending_dependency_ids.pop()
        if dependency_id in selected_ids:
            continue
        dependency_checkpoint = lookup.get(dependency_id)
        if dependency_checkpoint is None:
            continue
        if not include_manual and not dependency_checkpoint.automated:
            continue
        selected_ids.add(dependency_id)
        pending_dependency_ids.extend(dependency_checkpoint.dependencies)
    selected = [checkpoint for checkpoint in checkpoints if checkpoint.checkpoint_id in selected_ids]
    if not include_manual:
        selected = [checkpoint for checkpoint in selected if checkpoint.automated]
    return selected


def _truncate_output(value: str, limit: int = 8000) -> str:
    if len(value) <= limit:
        return value
    return value[-limit:]


def _load_resume_state(resume_json: Path | None) -> dict[str, object]:
    if resume_json is None or not resume_json.exists():
        return {}
    return json.loads(resume_json.read_text(encoding="utf-8"))


def _write_state_json(state_json: Path, state: dict[str, object]) -> None:
    state["updated_at"] = _utc_now()
    state_json.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def _new_command_record(command: WorkflowCommand) -> dict[str, object]:
    return {
        "name": command.name,
        "argv": list(command.argv),
        "cwd": str(command.cwd),
        "timeout_sec": int(command.timeout_sec),
        "started_at": _utc_now(),
    }


def _execute_command(
    command: WorkflowCommand,
    *,
    dry_run: bool,
    command_record: dict[str, object] | None = None,
) -> dict[str, object]:
    if command_record is None:
        command_record = _new_command_record(command)
    if dry_run:
        command_record["status"] = "dry_run"
        command_record["returncode"] = 0
        command_record["ended_at"] = _utc_now()
        return command_record
    try:
        completed = subprocess.run(
            command.argv,
            cwd=str(command.cwd),
            capture_output=True,
            text=True,
            timeout=int(command.timeout_sec),
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        command_record["status"] = "failed"
        command_record["returncode"] = None
        command_record["stdout"] = _truncate_output(exc.stdout or "")
        command_record["stderr"] = _truncate_output(exc.stderr or "")
        command_record["failure_reason"] = "timeout"
    else:
        command_record["returncode"] = int(completed.returncode)
        command_record["stdout"] = _truncate_output(completed.stdout)
        command_record["stderr"] = _truncate_output(completed.stderr)
        command_record["status"] = "passed" if completed.returncode == 0 else "failed"
    command_record["ended_at"] = _utc_now()
    return command_record


def run_workflow(
    checkpoints: list[WorkflowCheckpoint],
    context: WorkflowContext,
    *,
    state_json: Path,
    stop_on_failure: bool = False,
    dry_run: bool = False,
    command_workers: int = 1,
    resume_json: Path | None = None,
) -> dict[str, object]:
    context.session_dir.mkdir(parents=True, exist_ok=True)
    previous_state = _load_resume_state(resume_json)
    previous_results = {
        str(name): value
        for name, value in dict(previous_state.get("results", {})).items()
    }

    state: dict[str, object] = {
        "guide_path": str(context.guide_path),
        "status_path": str(context.status_path),
        "project_root": str(context.project_root),
        "repo_root": str(context.repo_root),
        "python_bin": str(context.python_bin),
        "session_dir": str(context.session_dir),
        "started_at": previous_state.get("started_at", _utc_now()),
        "updated_at": _utc_now(),
        "stop_on_failure": bool(stop_on_failure),
        "dry_run": bool(dry_run),
        "command_workers": max(1, int(command_workers)),
        "results": dict(previous_results),
        "checkpoint_order": [checkpoint.checkpoint_id for checkpoint in checkpoints],
    }

    for checkpoint in checkpoints:
        result_key = checkpoint.checkpoint_id
        previous_result = previous_results.get(result_key)
        if isinstance(previous_result, dict) and str(previous_result.get("status")) == "passed":
            state["results"][result_key] = previous_result
            continue

        checkpoint_record: dict[str, object] = {
            "step_number": int(checkpoint.step_number),
            "title": checkpoint.title,
            "description": checkpoint.description,
            "references": list(checkpoint.references),
            "automated": bool(checkpoint.automated),
            "dependencies": list(checkpoint.dependencies),
            "started_at": _utc_now(),
            "commands": [],
        }

        if not checkpoint.automated:
            checkpoint_record["status"] = "manual"
            checkpoint_record["ended_at"] = _utc_now()
            state["results"][result_key] = checkpoint_record
            _write_state_json(state_json, state)
            continue

        satisfied_statuses = {"passed"}
        if dry_run:
            satisfied_statuses.add("dry_run")
        blocked = [
            dependency
            for dependency in checkpoint.dependencies
            if str(dict(state["results"]).get(dependency, {}).get("status")) not in satisfied_statuses
        ]
        if blocked:
            checkpoint_record["status"] = "blocked"
            checkpoint_record["blocked_by"] = blocked
            checkpoint_record["ended_at"] = _utc_now()
            state["results"][result_key] = checkpoint_record
            _write_state_json(state_json, state)
            if stop_on_failure:
                break
            continue

        commands = [] if checkpoint.command_builder is None else checkpoint.command_builder(context)
        if not commands:
            checkpoint_record["status"] = "skipped"
            checkpoint_record["skip_reason"] = "no_commands_available"
            checkpoint_record["ended_at"] = _utc_now()
            state["results"][result_key] = checkpoint_record
            _write_state_json(state_json, state)
            continue

        checkpoint_status = "passed"
        checkpoint_record["status"] = "running"
        state["results"][result_key] = checkpoint_record
        _write_state_json(state_json, state)
        if checkpoint.parallel_commands and len(commands) > 1 and max(1, int(command_workers)) > 1:
            command_records = [_new_command_record(command) for command in commands]
            for command_record in command_records:
                command_record["status"] = "running"
            checkpoint_record["commands"] = command_records
            _write_state_json(state_json, state)
            with ThreadPoolExecutor(max_workers=min(len(commands), max(1, int(command_workers)))) as executor:
                future_to_index = {
                    executor.submit(
                        _execute_command,
                        command,
                        dry_run=dry_run,
                        command_record=command_records[index],
                    ): index
                    for index, command in enumerate(commands)
                }
                for future in as_completed(future_to_index):
                    index = future_to_index[future]
                    command_record = future.result()
                    command_records[index] = command_record
                    if str(command_record.get("status")) not in ("passed", "dry_run"):
                        checkpoint_status = "failed"
                    _write_state_json(state_json, state)
            checkpoint_record["commands"] = command_records
        else:
            for command in commands:
                command_record = _new_command_record(command)
                command_record["status"] = "running"
                checkpoint_record["commands"].append(command_record)
                _write_state_json(state_json, state)
                command_record = _execute_command(command, dry_run=dry_run, command_record=command_record)
                _write_state_json(state_json, state)
                if str(command_record.get("status")) not in ("passed", "dry_run"):
                    checkpoint_status = "failed"
                    break

        checkpoint_record["status"] = checkpoint_status if not dry_run else "dry_run"
        checkpoint_record["ended_at"] = _utc_now()
        state["results"][result_key] = checkpoint_record
        _write_state_json(state_json, state)
        if checkpoint_record["status"] == "failed" and stop_on_failure:
            break

    return state
