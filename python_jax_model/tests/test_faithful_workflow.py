from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
from datetime import datetime
from pathlib import Path

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from elle_jax_model.faithful_workflow import (  # noqa: E402
    WorkflowCheckpoint,
    WorkflowCommand,
    build_faithful_workflow,
    default_workflow_context,
    run_workflow,
    select_checkpoints,
)


class FaithfulWorkflowTests(unittest.TestCase):
    def test_default_workflow_context_prefers_repo_venv_python_without_resolving_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            context = default_workflow_context(
                project_root=PROJECT_ROOT,
                session_dir=Path(tmpdir) / "session",
            )

        self.assertEqual(
            context.python_bin,
            (PROJECT_ROOT.parent / ".venv" / "bin" / "python").absolute(),
        )

    def test_build_faithful_workflow_contains_key_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            context = default_workflow_context(
                project_root=PROJECT_ROOT,
                session_dir=Path(tmpdir) / "session",
            )
            checkpoints = build_faithful_workflow(context)

        checkpoint_ids = {checkpoint.checkpoint_id for checkpoint in checkpoints}
        self.assertIn("step01_runtime_skeleton", checkpoint_ids)
        self.assertIn("step09_outer_loop_smoke", checkpoint_ids)
        self.assertIn("step12_ci_guardrails", checkpoint_ids)
        manual_ids = {
            checkpoint.checkpoint_id
            for checkpoint in checkpoints
            if not checkpoint.automated
        }
        self.assertIn("step08_fft_bridge", manual_ids)

    def test_select_checkpoints_respects_step_range_and_manual_filter(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            context = default_workflow_context(
                project_root=PROJECT_ROOT,
                session_dir=Path(tmpdir) / "session",
            )
            checkpoints = build_faithful_workflow(context)
            selected = select_checkpoints(
                checkpoints,
                from_step=7,
                to_step=10,
                include_manual=False,
            )

        selected_ids = [checkpoint.checkpoint_id for checkpoint in selected]
        self.assertEqual(
            selected_ids,
            [
                "step07_recovery",
                "step09_outer_loop_smoke",
                "step10_validation",
            ],
        )

    def test_select_checkpoints_includes_required_dependencies_outside_requested_range(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            context = default_workflow_context(
                project_root=PROJECT_ROOT,
                session_dir=Path(tmpdir) / "session",
            )
            checkpoints = build_faithful_workflow(context)
            selected = select_checkpoints(
                checkpoints,
                from_step=9,
                to_step=10,
                include_manual=False,
            )

        selected_ids = [checkpoint.checkpoint_id for checkpoint in selected]
        self.assertEqual(
            selected_ids,
            [
                "step07_recovery",
                "step09_outer_loop_smoke",
                "step10_validation",
            ],
        )

    def test_step09_and_step10_commands_use_session_dir_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            session_dir = Path(tmpdir) / "workflow-session"
            context = default_workflow_context(
                project_root=PROJECT_ROOT,
                session_dir=session_dir,
            )
            checkpoints = {
                checkpoint.checkpoint_id: checkpoint
                for checkpoint in build_faithful_workflow(context)
            }
            step09_commands = checkpoints["step09_outer_loop_smoke"].command_builder(context)
            step10_commands = checkpoints["step10_validation"].command_builder(context)

        step09_args = " ".join(step09_commands[0].argv)
        self.assertIn(str(session_dir / "step09_outer_loop_smoke"), step09_args)
        step10_args = " ".join(step10_commands[0].argv)
        self.assertIn(str(session_dir / "step09_outer_loop_smoke"), step10_args)
        self.assertIn(str(session_dir / "step10_figure2_line.json"), step10_args)

    def test_step10_validation_checkpoint_marks_commands_parallel_safe(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            context = default_workflow_context(
                project_root=PROJECT_ROOT,
                session_dir=Path(tmpdir) / "session",
            )
            checkpoints = {
                checkpoint.checkpoint_id: checkpoint
                for checkpoint in build_faithful_workflow(context)
            }

        self.assertTrue(checkpoints["step10_validation"].parallel_commands)

    def test_run_workflow_can_overlap_parallel_safe_commands(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            context = default_workflow_context(
                project_root=PROJECT_ROOT,
                session_dir=Path(tmpdir) / "session",
            )
            checkpoint = WorkflowCheckpoint(
                step_number=99,
                checkpoint_id="parallel_probe",
                title="Parallel probe",
                description="Synthetic checkpoint for verifying parallel command overlap.",
                references=(),
                automated=True,
                parallel_commands=True,
                command_builder=lambda ctx: [
                    WorkflowCommand(
                        name="sleep-a",
                        argv=(str(ctx.python_bin), "-c", "import time; time.sleep(0.35)"),
                        cwd=ctx.repo_root,
                        timeout_sec=30,
                    ),
                    WorkflowCommand(
                        name="sleep-b",
                        argv=(str(ctx.python_bin), "-c", "import time; time.sleep(0.35)"),
                        cwd=ctx.repo_root,
                        timeout_sec=30,
                    ),
                ],
            )
            state_json = context.session_dir / "state.json"
            started = time.monotonic()
            state = run_workflow(
                [checkpoint],
                context,
                state_json=state_json,
                command_workers=2,
            )
            elapsed = time.monotonic() - started

        record = state["results"]["parallel_probe"]
        commands = record["commands"]
        start_a = datetime.fromisoformat(commands[0]["started_at"])
        end_a = datetime.fromisoformat(commands[0]["ended_at"])
        start_b = datetime.fromisoformat(commands[1]["started_at"])
        end_b = datetime.fromisoformat(commands[1]["ended_at"])
        self.assertLess(elapsed, 0.65)
        self.assertLess(start_a, end_b)
        self.assertLess(start_b, end_a)

    def test_run_workflow_writes_running_status_while_command_is_in_progress(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            context = default_workflow_context(
                project_root=PROJECT_ROOT,
                session_dir=Path(tmpdir) / "session",
            )
            checkpoint = WorkflowCheckpoint(
                step_number=98,
                checkpoint_id="running_probe",
                title="Running probe",
                description="Synthetic checkpoint for verifying in-progress state writes.",
                references=(),
                automated=True,
                command_builder=lambda ctx: [
                    WorkflowCommand(
                        name="sleep-once",
                        argv=(str(ctx.python_bin), "-c", "import time; time.sleep(0.4)"),
                        cwd=ctx.repo_root,
                        timeout_sec=30,
                    )
                ],
            )
            state_json = context.session_dir / "state.json"
            holder: dict[str, object] = {}

            def _runner() -> None:
                holder["state"] = run_workflow(
                    [checkpoint],
                    context,
                    state_json=state_json,
                )

            thread = threading.Thread(target=_runner)
            thread.start()

            saw_running = False
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline and thread.is_alive():
                if state_json.exists():
                    state = json.loads(state_json.read_text(encoding="utf-8"))
                    record = state.get("results", {}).get("running_probe", {})
                    commands = record.get("commands", [])
                    if (
                        record.get("status") == "running"
                        and commands
                        and commands[0].get("status") == "running"
                    ):
                        saw_running = True
                        break
                time.sleep(0.02)
            thread.join()

        self.assertTrue(saw_running)
        self.assertEqual(holder["state"]["results"]["running_probe"]["status"], "passed")


if __name__ == "__main__":
    unittest.main()
