"""Named connections and in-memory plans for CLI, Python and web callers."""
from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import re
import tempfile
import tomllib
from urllib.parse import urlsplit

from .configuration import ConfigError, TaskPlan, validate_document

_DEFAULT = "coatlearn.connections.json"
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\Z")


def _name(value):
    if not isinstance(value, str) or not _NAME.fullmatch(value):
        raise ConfigError("Connection name must contain only letters, digits, dot, underscore or hyphen")
    return value


def _connection_model(entry):
    if not isinstance(entry, dict) or set(entry) - {"model", "provider"} or not isinstance(entry.get("model"), dict):
        raise ConfigError("Invalid connection store")
    if entry.get("provider") is not None and entry["provider"] not in {"deepseek", "openrouter", "opencode", "vllm"}:
        raise ConfigError("Invalid connection provider")
    model = entry["model"]
    # Reuse exactly the same strict validation and credential/URL checks as tasks.
    backend = model.get("backend")
    main = model if backend == "openai" else {"backend": "openai", "model": "validation"}
    roles = {"main": "main", "annotation": "selected"}
    if backend in {"jev", "kev"}:
        roles.update(judge="selected", dedup="selected")
    validate_document({"schema_version": 1, "task": {"method": "mb", "data": "validation.csv", "text_col": "text", "target": "target"}, "models": {"main": main, "selected": model}, "roles": roles}, Path.cwd() / "workspace.toml")
    return deepcopy(model)


def list_connections(connections=_DEFAULT) -> dict:
    """Read and validate a credential-free connection mapping."""
    path = Path(connections)
    try:
        if not path.exists():
            return {}
        with path.open(encoding="utf-8") as handle:
            store = json.load(handle)
        if not isinstance(store, dict):
            raise ConfigError("Invalid connection store")
        for key, value in store.items():
            _name(key)
            _connection_model(value)
        return store
    except ConfigError:
        raise
    except (OSError, ValueError, TypeError):
        raise ConfigError("Connection store cannot be read or parsed") from None


def save_connection(name, *, backend="openai", model, address, api_key_env=None, effort=None,
                    connections=_DEFAULT, replace=False, timeout=60.0, max_concurrency=3,
                    max_tokens=None, provider=None) -> dict:
    """Atomically save a named connection; only environment variable names are stored."""
    _name(name)
    if provider is not None and provider not in {"deepseek", "openrouter", "opencode", "vllm"}:
        raise ConfigError("Unsupported connection provider")
    if backend not in {"openai", "jev", "kev"}:
        raise ConfigError("Unsupported connection backend")
    if not isinstance(address, str):
        raise ConfigError("Connection address must be a valid HTTP(S) URL")
    try:
        url = urlsplit(address)
        if url.scheme not in {"http", "https"} or not url.hostname or url.username is not None or url.password is not None or url.query or url.fragment:
            raise ValueError
        url.port
    except ValueError:
        raise ConfigError("Connection address must be an HTTP(S) URL without credentials, query or fragment") from None
    spec = {"backend": backend, "model": model, "base_url": address, "timeout": timeout, "max_concurrency": max_concurrency}
    if backend == "kev":
        if api_key_env is not None or effort is not None or max_tokens is not None:
            raise ConfigError("KEV does not accept credential, effort or token options")
        spec["max_retries"] = 0
    else:
        from .catalog import get_catalog
        catalog = get_catalog()
        preset = catalog['decision_models'].get(backend, {}) if backend != 'openai' else catalog['providers'].get(provider, {})
        spec["api_key_env"] = api_key_env if api_key_env is not None else preset.get('api_key_env') or 'OPENAI_API_KEY'
        if backend == "jev" and (effort is not None or max_tokens is not None):
            raise ConfigError("Jev does not accept effort or token options")
        if backend == "openai":
            if max_tokens is not None:
                spec["max_tokens"] = max_tokens
            if effort is not None:
                if not isinstance(effort, str) or not effort.strip():
                    raise ConfigError("Invalid effort")
                deepseek = provider == "deepseek" or url.hostname.lower() == "api.deepseek.com"
                if deepseek:
                    if effort not in {"none", "low", "high", "max"}:
                        raise ConfigError("Unsupported DeepSeek effort")
                    spec["thinking"] = "disabled" if effort == "none" else "enabled"
                    if effort != "none":
                        spec["reasoning_effort"] = effort
                else:
                    spec["reasoning_effort"] = effort
    entry = {"model": spec, "provider": provider}
    _connection_model(entry)
    path = Path(connections)
    store = list_connections(path)
    if name in store and not replace:
        raise ConfigError("Connection already exists; use replace to overwrite")
    store[name] = entry
    temp = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, prefix=".coatlearn-", delete=False) as handle:
            temp = handle.name
            json.dump(store, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    except (OSError, ValueError, TypeError):
        raise ConfigError("Cannot save connection store") from None
    finally:
        if temp and os.path.exists(temp):
            os.unlink(temp)
    return deepcopy(entry)


def prepare_task(method, *, config=None, data=None, text_col=None, target=None, known_variables=None,
                 treatment=None, outcome=None, main_connection=None, decision_connection=None,
                 decision_kind=None, rounds=None, concurrency=None, output=None,
                 connections=_DEFAULT, analysis_options=None) -> TaskPlan:
    """Compose and strictly validate a task in memory, without clients or credentials."""
    override_main = main_connection is not None or config is None
    override_decision = decision_connection is not None or config is None
    main_connection = main_connection or "main"
    decision_connection = decision_connection or "decision"
    if method not in {"mb", "pag", "adj"} or decision_kind not in {None, "model", "llm"}:
        raise ConfigError("Unsupported task method or decision kind")
    source = Path(config).resolve() if config is not None else Path.cwd() / "workspace.toml"
    if config is not None:
        try:
            if source.suffix.lower() != ".toml":
                raise ConfigError("Workspace configuration must be TOML")
            with source.open("rb") as handle:
                document = tomllib.load(handle)
        except ConfigError:
            raise
        except (OSError, ValueError, TypeError):
            raise ConfigError("Configuration file cannot be read or parsed") from None
        original = validate_document(document, source)
        if original.task.method != method:
            raise ConfigError("Command method does not match task.method")
    else:
        document = {"schema_version": 1, "task": {"method": method}, "models": {}, "roles": {}, "analysis": {"rounds": 2}}
    task = document["task"]
    for key, value in {"data": data, "text_col": text_col, "target": target, "known_variables": known_variables, "treatment": treatment, "outcome": outcome}.items():
        if value is not None:
            task[key] = str(Path(value).resolve()) if key == "data" else value
    analysis = document.setdefault("analysis", {})
    for key, value in {"rounds": rounds, "concurrency": concurrency, "output_dir": str(Path(output).resolve()) if output is not None else None}.items():
        if value is not None:
            analysis[key] = value
    if analysis_options is not None:
        if not isinstance(analysis_options, dict):
            raise ConfigError("Analysis options must be a mapping")
        analysis.setdefault("options", {}).update(deepcopy(analysis_options))
    store = list_connections(connections) if override_main or override_decision else {}
    models = document.setdefault('models', {})
    roles = document.setdefault('roles', {})
    role_options = document.setdefault('role_options', {})
    options = analysis.setdefault('options', {})
    def selected_connection(name):
        _name(name)
        if name not in store:
            raise ConfigError('Selected connection is not defined')
        return _connection_model(store[name])
    if override_main:
        main = selected_connection(main_connection)
        if main['backend'] != 'openai':
            raise ConfigError('Main connection must use the openai chat backend')
        models['workspace_main'] = main
        roles['main'] = 'workspace_main'
    if override_decision:
        decision = selected_connection(decision_connection)
        actual_kind = 'model' if decision['backend'] in {'jev', 'kev'} else 'llm'
        if decision_kind is not None and decision_kind != actual_kind:
            raise ConfigError('Decision kind does not match the selected connection')
        models['workspace_decision'] = decision
        roles['annotation'] = 'workspace_decision'
        # Connection replacement replaces its transport settings, not analysis rules.
        dedup_context = role_options.get('dedup', {}).get('task_context')
        for role in ('annotation', 'judge', 'dedup'):
            role_options.pop(role, None)
        if actual_kind == 'model' and dedup_context is not None:
            role_options['dedup'] = {'task_context': dedup_context}
        for role in ('judge', 'dedup'):
            options.pop('factor_' + role, None)
            roles.pop(role, None)
            if actual_kind == 'model':
                roles[role] = 'workspace_decision'
        if actual_kind == 'llm':
            options.pop('annotation_batching', None)
    elif decision_kind is not None:
        current = models[roles['annotation']]
        actual_kind = 'model' if current['backend'] in {'jev', 'kev'} else 'llm'
        if actual_kind != decision_kind:
            raise ConfigError('Decision kind does not match the configured annotation model')
    return validate_document(document, source)


def run_task(method, *, execute=False, env_file=None, **same_kwargs) -> dict:
    """Check locally by default; execute only after explicit opt-in."""
    from .execution import check_task, execute_task, TaskError
    plan = prepare_task(method, **same_kwargs)
    checked = check_task(plan)
    if not execute:
        return checked.summary
    if plan.placeholder:
        raise TaskError("Replace placeholder models before execution")
    if env_file is not None:
        try:
            from dotenv import dotenv_values
            selected = dotenv_values(env_file, interpolate=False)
        except (OSError, ValueError, TypeError):
            raise TaskError("Cannot load the selected environment file") from None
        for name in plan.required_env_names():
            value = selected.get(name)
            if name not in os.environ and isinstance(value, str) and value.strip():
                os.environ[name] = value
    missing = [name for name in plan.required_env_names() if not os.environ.get(name)]
    if missing:
        raise TaskError("Set required environment variables: " + ", ".join(missing))
    result, output = execute_task(checked)
    return {"status": result["status"], "result": result, "output_dir": str(output)}
