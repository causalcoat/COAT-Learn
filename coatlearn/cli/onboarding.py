"""Setup and run adapters over the shared workspace interface."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def _ask(label, default=None):
    if not sys.stdin.isatty():
        raise ValueError("Interactive mode needs a terminal; supply command arguments instead")
    answer = input(f"{label}" + (f" [{default}]" if default is not None else "") + ": ").strip()
    return answer or default


def _choose(label, choices, default):
    print(label + ": " + " / ".join(choices))
    value = _ask("Choose", default)
    if value not in choices:
        raise ValueError("Choose one of the listed options")
    return value


def _setup_wizard():
    from coatlearn.workspace import save_connection
    name = _ask("Connection name", "main")
    provider = _choose("Model source", ["deepseek", "openrouter", "opencode", "vllm", "jev", "kev"], "deepseek")
    from coatlearn.catalog import get_catalog
    catalog = get_catalog()
    preset = {**catalog['providers'], **catalog['decision_models']}[provider]
    backend, address, env = preset['backend'], preset['address'], preset['api_key_env']
    choices = preset['models'] + ['custom']
    selected = _choose('Model', choices, choices[0])
    model = _ask('Custom model name') if selected == 'custom' else selected
    address = preset.get('addresses', {}).get(model, address)
    kwargs = dict(backend=backend, model=model, address=_ask("Endpoint", address), provider=provider if backend == "openai" else None)
    if preset.get('efforts'):
        effort = _choose('Reasoning effort', ['default', *preset['efforts']], 'default')
        kwargs['effort'] = None if effort == 'default' else effort
    if env:
        kwargs["api_key_env"] = _ask("API key environment variable name", env)
    kwargs["connections"] = _ask("Connections file", "coatlearn.connections.json")
    from coatlearn.workspace import list_connections
    existing = list_connections(kwargs['connections']).get(name)
    if existing:
        print(json.dumps({'previous': existing, 'replacement': {'name': name, **kwargs}}, indent=2))
        kwargs['replace'] = True
    else:
        print(json.dumps({"name": name, **kwargs}, indent=2))
    if _choose("Update this connection?" if existing else "Save this connection?", ["yes", "no"], "no" if existing else "yes") == "no":
        return {"status": "cancelled"}
    result = save_connection(name, **kwargs)
    if backend == 'kev':
        action = _choose('Local Kev service', ['new', 'existing'], 'new')
        if action == 'new':
            from coatlearn.local_setup import prepare
            plan = prepare(model, kwargs['address'])
            print(json.dumps(plan, indent=2))
            if _choose('Install dependencies and download weights?', ['yes', 'no'], 'no') == 'yes':
                prepare(model, kwargs['address'], execute=True)
            print('Start in another terminal:')
            import shlex
            print(shlex.join(['coatlearn', 'setup', 'local', 'start', '--model', model, '--address', kwargs['address']]))
    return result


def _run_wizard():
    method = _choose("Analysis method", ["mb", "pag", "adj"], "mb")
    source = _choose('Data source', ['example', 'custom'], 'example')
    kwargs = {}
    default_rounds = 2
    if source == 'example':
        from coatlearn.preparation import prepare_resources
        print(json.dumps(prepare_resources(method=method), indent=2))
        kwargs['config'] = _ask('Example configuration', f'config/{method}.toml')
        from coatlearn.configuration import load_task
        default_rounds = load_task(kwargs['config']).run_config().rounds
    else:
        kwargs = {"data": _ask("Local data path"), "text_col": _ask("Text column", "Review")}
        if method == "mb":
            kwargs["target"] = _ask("Target column", "Score")
        elif method == "pag":
            kwargs["known_variables"] = [v.strip() for v in _ask("Known columns (comma separated)", "Taste,Score").split(",")]
        else:
            kwargs["treatment"] = _ask("Treatment column", "email_type")
            kwargs["outcome"] = _ask("Outcome column", "visit")
    kwargs.update(main_connection=_ask("Main connection", "main"),
                  decision_kind=_choose("Annotation type", ["llm", "model"], "model"),
                  decision_connection=_ask("Annotation connection", "decision"),
                  connections=_ask("Connections file", "coatlearn.connections.json"),
                  rounds=int(_ask("Factor discovery iterations", str(default_rounds))),
                  output=_ask("Output directory (blank creates a new run automatically)", "") or None)
    if kwargs['decision_kind'] == 'llm':
        from coatlearn.preparation import prepare_resources
        print(json.dumps(prepare_resources(method='none', decision_kind='llm'), indent=2))
    from coatlearn.workspace import run_task
    summary = run_task(method, **kwargs)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if _choose("Execute with these models?", ["yes", "no"], "no") == "yes":
        return run_task(method, execute=True, env_file=_ask("Credentials file (optional)", "") or None, **kwargs)
    return summary


def main(argv):
    parser = argparse.ArgumentParser(prog="coatlearn")
    commands = parser.add_subparsers(dest="command", required=True)
    setup = commands.add_parser("setup", help="Save named model connections or prepare local Kev")
    setup_commands = setup.add_subparsers(dest="setup_command")
    connection = setup_commands.add_parser("connection")
    connection.add_argument("name")
    connection.add_argument("--backend", choices=["openai", "jev", "kev"], default="openai")
    connection.add_argument("--provider", choices=["deepseek", "openrouter", "opencode", "vllm"])
    connection.add_argument("--model", required=True)
    connection.add_argument("--address", "--base-url", dest="address", required=True)
    connection.add_argument("--api-key-env")
    connection.add_argument("--effort")
    connection.add_argument("--timeout", type=float, default=60.0)
    connection.add_argument("--max-concurrency", type=int, default=3)
    connection.add_argument("--max-tokens", type=int)
    connection.add_argument("--connections", default="coatlearn.connections.json")
    connection.add_argument("--replace", action="store_true")
    listing = setup_commands.add_parser("list")
    listing.add_argument("--connections", default="coatlearn.connections.json")
    resources = setup_commands.add_parser('prepare', help='Prepare example files and optional WordNet resources')
    resources.add_argument('--method', choices=['mb', 'pag', 'adj', 'all', 'none'], default='mb')
    resources.add_argument('--directory', default='.')
    resources.add_argument('--decision-kind', choices=['llm', 'model'], default='model')
    resources.add_argument('--wordnet', action=argparse.BooleanOptionalAction, default=None)
    local = setup_commands.add_parser("local")
    local_commands = local.add_subparsers(dest="local_command", required=True)
    for action in ("prepare", "start"):
        cmd = local_commands.add_parser(action)
        cmd.add_argument("--model", required=True, choices=["kev-9b-v1.0", "kev-4b-v1.0"])
        cmd.add_argument("--address")
        for key in ("work-dir", "venv-dir", "source-dir", "adapter-dir", "base-dir"):
            cmd.add_argument("--" + key, type=Path)
        if action == "prepare":
            cmd.add_argument("--execute", action="store_true", help="Install dependencies and download weights")
    commands.add_parser("catalog", help="Print shared website and CLI choices as JSON")
    run = commands.add_parser("run", help="Prepare or execute an analysis using saved connections")
    run.add_argument("method", nargs="?", choices=["mb", "pag", "adj"])
    for key in ("config", "data", "text-col", "target", "treatment", "outcome", "output", "env-file"):
        run.add_argument("--" + key)
    run.add_argument("--known-variable", action="append", dest="known_variables")
    run.add_argument("--main-connection")
    run.add_argument("--decision-connection")
    run.add_argument("--decision-kind", choices=["llm", "model"])
    run.add_argument("--connections", default="coatlearn.connections.json")
    run.add_argument("--rounds", type=int)
    run.add_argument("--concurrency", type=int)
    run.add_argument("--execute", action="store_true", help="Explicitly allow model calls")
    run.add_argument("--option", action="append", default=[], metavar="NAME=JSON", help="Analysis option override")
    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    try:
        if command == "catalog":
            from coatlearn.catalog import get_catalog
            result = get_catalog()
        elif command == "setup":
            action = args.pop("setup_command")
            if action is None:
                result = _setup_wizard()
            elif action == 'prepare':
                from coatlearn.preparation import prepare_resources
                result = prepare_resources(**args)
            elif action == "local":
                from coatlearn import local_setup
                local_action = args.pop("local_command")
                result = getattr(local_setup, local_action)(**args)
            elif action == "list":
                from coatlearn.workspace import list_connections
                result = list_connections(**args)
            else:
                from coatlearn.workspace import save_connection
                result = save_connection(**args)
        else:
            method = args.pop("method")
            if method is None:
                result = _run_wizard()
            else:
                from coatlearn.workspace import run_task
                options = {}
                for item in args.pop("option"):
                    key, separator, raw = item.partition("=")
                    if not separator:
                        raise ValueError("--option requires NAME=JSON")
                    options[key] = json.loads(raw)
                if options:
                    args["analysis_options"] = options
                result = run_task(method, **args)
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str, allow_nan=False))
        return 0 if not isinstance(result, dict) or result.get("status") not in {"failed", "completed_with_errors"} else 3
    except (EOFError, KeyboardInterrupt):
        print("Cancelled", file=sys.stderr)
        return 130
    except (ValueError, OSError, RuntimeError) as exc:
        # Helpers sanitize validation errors; never print network/provider payloads.
        from coatlearn.configuration import ConfigError
        from coatlearn.execution import TaskError
        if isinstance(exc, (ConfigError, TaskError)):
            print(str(exc), file=sys.stderr)
        else:
            print("Cannot complete operation: " + type(exc).__name__ + ". Check arguments, local files and configuration.", file=sys.stderr)
        return 2
