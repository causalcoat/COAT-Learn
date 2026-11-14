"""Command-line access to the same task configuration and execution functions."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="coatlearn", description="Discover and measure factors in text.")
    commands = parser.add_subparsers(dest="method", required=True)
    for method in ("mb", "pag", "adj"):
        command = commands.add_parser(method, help={"mb": "Find factors related to a target",
                                                   "pag": "Build a graph of variable relationships",
                                                   "adj": "Find adjustment sets and estimate effects"}[method])
        command.add_argument("--config", type=Path, required=True, help="TOML task configuration")
        command.add_argument("--data", type=Path, help="Use another local data file")
        command.add_argument("--decision-model", help="Named model for annotation, judge and dedup roles")
        command.add_argument("--output-dir", type=Path, help="New directory for this run")
        command.add_argument("--env-file", type=Path, help="Load only required keys from this dotenv, on --run")
        command.add_argument("--run", action="store_true", help="Run models; may send text and incur charges")
    args = parser.parse_args(argv)
    from coatlearn.configuration import ConfigError, load_task
    from coatlearn.execution import TaskError, check_task, execute_task

    try:
        overrides = {"data": str(args.data.resolve())} if args.data is not None else None
        plan = load_task(args.config, task_overrides=overrides, decision_model=args.decision_model)
        if plan.task.method != args.method:
            raise ConfigError("Command method does not match task.method")
        checked = check_task(plan)
        if not args.run:
            print(json.dumps(checked.summary, ensure_ascii=False, indent=2, allow_nan=False))
            return 0
        if plan.placeholder:
            raise TaskError("Replace placeholder models before execution")
        if args.env_file is not None:
            from dotenv import dotenv_values
            selected = dotenv_values(args.env_file, interpolate=False)
            for name in plan.required_env_names():
                value = selected.get(name)
                if name not in os.environ and isinstance(value, str) and value.strip():
                    os.environ[name] = value
        missing = [name for name in plan.required_env_names() if not os.environ.get(name)]
        if missing:
            raise TaskError("Set required environment variables: " + ", ".join(missing))
        result, output = execute_task(checked, output_dir=args.output_dir)
        print(json.dumps({"status": result["status"], "result": str(output / "result.json")}, indent=2))
        return 0 if result["status"] == "completed" else 3
    except (ConfigError, TaskError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except OSError:
        print("Cannot access a local file; check paths and permissions", file=sys.stderr)
        return 2
    except Exception as exc:
        print("Execution stopped: " + type(exc).__name__, file=sys.stderr)
        return 1
