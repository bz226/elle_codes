from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from elle_jax_model.faithful_workflow import (
    build_faithful_workflow,
    default_workflow_context,
    run_workflow,
    select_checkpoints,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the faithful-branch guide checkpoints with automated verification and resumable state"
    )
    parser.add_argument("--list", action="store_true", help="List available checkpoints and exit")
    parser.add_argument("--from-step", type=int, help="First guide step to include")
    parser.add_argument("--to-step", type=int, help="Last guide step to include")
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        help="Specific checkpoint id to run; may be passed multiple times",
    )
    parser.add_argument(
        "--include-manual",
        action="store_true",
        help="Also include non-automated guide checkpoints in the state report",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write the selected checkpoint plan without executing commands",
    )
    parser.add_argument(
        "--stop-on-failure",
        action="store_true",
        help="Stop the workflow after the first failed or blocked checkpoint",
    )
    parser.add_argument(
        "--command-workers",
        type=int,
        default=1,
        help="Maximum number of parallel command jobs within a checkpoint when that checkpoint is marked parallel-safe",
    )
    parser.add_argument(
        "--resume-json",
        type=Path,
        help="Resume from an existing workflow state JSON and skip previously passed checkpoints",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("python_jax_model/validation/workflow"),
        help="Base directory for workflow session artifacts",
    )
    parser.add_argument(
        "--run-name",
        default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        help="Session name under the workflow run root",
    )
    parser.add_argument("--project-root", type=Path, default=Path("python_jax_model"))
    parser.add_argument("--python-bin", type=Path, default=Path())
    parser.add_argument("--guide-path", type=Path)
    parser.add_argument("--status-path", type=Path)
    parser.add_argument("--fine-foam-init-elle", type=Path)
    parser.add_argument("--fine-foam-reference-dir", type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--legacy-bundle-json", type=Path)
    parser.add_argument("--legacy-transition-json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.expanduser().absolute()
    session_dir = (args.run_root / args.run_name).expanduser().absolute()
    python_bin = (
        None
        if str(args.python_bin) in ("", ".")
        else args.python_bin.expanduser().absolute()
    )
    context = default_workflow_context(
        project_root=project_root,
        python_bin=python_bin,
        session_dir=session_dir,
        guide_path=args.guide_path,
        status_path=args.status_path,
        fine_foam_init_elle=args.fine_foam_init_elle,
        fine_foam_reference_dir=args.fine_foam_reference_dir,
        data_dir=args.data_dir,
        legacy_bundle_json=args.legacy_bundle_json,
        legacy_transition_json=args.legacy_transition_json,
    )
    checkpoints = build_faithful_workflow(context)
    selected = select_checkpoints(
        checkpoints,
        from_step=args.from_step,
        to_step=args.to_step,
        only_ids=tuple(args.only) if args.only else None,
        include_manual=bool(args.include_manual),
    )

    if args.list:
        for checkpoint in selected:
            kind = "auto" if checkpoint.automated else "manual"
            print(
                f"{checkpoint.checkpoint_id} step={checkpoint.step_number} kind={kind} "
                f"title={checkpoint.title}"
            )
        return

    if not selected:
        print("no checkpoints selected")
        return

    context.session_dir.mkdir(parents=True, exist_ok=True)
    state_json = (
        args.resume_json.expanduser().absolute()
        if args.resume_json is not None
        else context.session_dir / "faithful_workflow_state.json"
    )
    state = run_workflow(
        selected,
        context,
        state_json=state_json,
        stop_on_failure=bool(args.stop_on_failure),
        dry_run=bool(args.dry_run),
        command_workers=max(1, int(args.command_workers)),
        resume_json=args.resume_json.expanduser().absolute() if args.resume_json is not None else None,
    )

    results = dict(state.get("results", {}))
    passed = sum(1 for item in results.values() if str(item.get("status")) == "passed")
    failed = sum(1 for item in results.values() if str(item.get("status")) == "failed")
    blocked = sum(1 for item in results.values() if str(item.get("status")) == "blocked")
    manual = sum(1 for item in results.values() if str(item.get("status")) == "manual")
    dry_run = sum(1 for item in results.values() if str(item.get("status")) == "dry_run")

    print(f"wrote workflow state: {state_json}")
    print(
        f"workflow summary: passed={passed} failed={failed} blocked={blocked} "
        f"manual={manual} dry_run={dry_run}"
    )
    if failed:
        failed_ids = [
            checkpoint_id
            for checkpoint_id, item in results.items()
            if str(item.get("status")) == "failed"
        ]
        print(f"failed checkpoints: {', '.join(failed_ids)}")
    if blocked:
        blocked_ids = [
            checkpoint_id
            for checkpoint_id, item in results.items()
            if str(item.get("status")) == "blocked"
        ]
        print(f"blocked checkpoints: {', '.join(blocked_ids)}")
    print(json.dumps(state, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
