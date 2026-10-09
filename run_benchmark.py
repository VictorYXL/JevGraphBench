"""One configuration-driven entry point for graph tasks and result workflows.

Use --check to validate and preview without invoking a workflow. Live inference
and download-capable actions require explicit, independent CLI permissions.
"""

import argparse
import asyncio
from dataclasses import replace
import json
from pathlib import Path

from src.benchmark.workflows import WorkflowConfig, load_config
from src.benchmark.runner import run_experiment


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run a configured GraphDecisionBench task workflow")
    parser.add_argument("--config", type=Path, required=True, help="YAML benchmark configuration (.yaml/.yml)")
    parser.add_argument("--action", help="Select an action declared in the configuration")
    parser.add_argument("--check", action="store_true", help="Validate configuration and print the selected action without executing it")
    parser.add_argument("--output", type=Path, help="Override experiment root (relative to current directory; not supported for results)")
    parser.add_argument("--allow-inference", action="store_true", help="Allow the selected action to call model providers; existing approvals still apply")
    parser.add_argument("--allow-downloads", action="store_true", help="Allow the selected download-capable action to fetch public data")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress display for schema 1 topology evaluation")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        if isinstance(config, WorkflowConfig):
            invocation = config.select(args.action, args.output)
            if args.no_progress:
                parser.error("--no-progress is supported only for schema 1 topology evaluation")
            if args.check:
                preview = {**invocation.snapshot(), "action_invoked": False}
                print(json.dumps(preview, ensure_ascii=False, allow_nan=False, indent=2))
                return
            if invocation.requires_inference and not args.allow_inference:
                parser.error("this action calls model providers; pass --allow-inference to authorize it")
            if invocation.requires_downloads and not args.allow_downloads:
                parser.error("this action may download public data; pass --allow-downloads to authorize it")
            # Dispatch is outside configuration-error handling: runner failures
            # retain their existing diagnostics, exit status and sealed evidence.
        else:
            invocation = None
            if args.action not in (None, "run"):
                parser.error("schema 1 topology configurations support only the run action")
    except ValueError as exc:
        parser.error(str(exc))
    if invocation is not None:
        status = invocation.dispatch()
        if status:
            raise SystemExit(status)
        return
    if args.output:
        config = replace(config, run=replace(config.run, output_dir=args.output.resolve()))
    if args.check:
        preview = {
            "workflow": "topology", "action": "run", "action_invoked": False,
            "requires_inference": True, "requires_downloads": False,
            "planned_graphs": config.planned_graphs, "planned_questions": config.planned_questions,
            "config": config.snapshot(),
        }
        print(json.dumps(preview, ensure_ascii=False, allow_nan=False, indent=2))
        return
    if not args.allow_inference:
        parser.error("this action calls model providers; pass --allow-inference to authorize it")
    result = asyncio.run(run_experiment(config, progress=not args.no_progress))
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2))


if __name__ == "__main__":
    main()