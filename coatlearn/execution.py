"""Local task checking and explicit execution over the stable COAT interface.

Checking never loads dotenv, imports a model SDK, creates clients, or writes
artifacts. Execution owns its output directory; raw data and private core audits
are never copied into the public configuration snapshot.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from io import BytesIO
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd

from .configuration import TaskPlan


class TaskError(ValueError):
    """An actionable task failure whose message contains no source data."""


@dataclass
class CheckedTask:
    """Validated projected data and its non-sensitive local-check summary."""

    plan: TaskPlan
    frame: pd.DataFrame
    summary: dict[str, Any]
    mapping: dict[str, Any]


def load_data(task: Any) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read only the named analysis columns and apply declared mappings.

    Rows, duplicates and order are preserved. No implicit treatment recoding or
    truth-column inference is performed. Errors omit raw cell values.
    """
    numeric = ([task.target] if task.method == "mb" else
               list(task.known_variables) if task.method == "pag" else
               [task.treatment, task.outcome])
    columns = numeric + [task.text_col]
    path = Path(task.data)
    try:
        # Parse and fingerprint the exact same bytes, so a file replaced after
        # preflight cannot change the recorded identity of the cached frame.
        source = path.read_bytes()
        if path.suffix.lower() == ".csv":
            frame = pd.read_csv(BytesIO(source), usecols=columns)
        elif path.suffix.lower() == ".xlsx":
            frame = pd.read_excel(BytesIO(source), usecols=columns)
        else:
            raise TaskError("Data must be a .csv or .xlsx file")
    except (OSError, ValueError, ImportError, KeyError) as exc:
        if isinstance(exc, TaskError):
            raise
        raise TaskError("Cannot load data: check the path, format, required columns and reader dependencies") from None
    frame = frame.loc[:, columns].copy()
    if frame.empty:
        raise TaskError("Data must contain at least one row")
    if not frame[task.text_col].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
        raise TaskError("The text column must contain nonempty strings without missing values")

    mapping: dict[str, Any] = {"columns_used": columns, "n_rows": len(frame),
                               "source_csv_sha256": hashlib.sha256(source).hexdigest()}
    treatment_mapping = getattr(task, "treatment_mapping", None)
    outcome_mapping = getattr(task, "outcome_mapping", None)
    if treatment_mapping == "minus_one_one":
        values = frame[task.treatment]
        if values.isna().any() or not values.isin([-1, 1]).all() or set(values) != {-1, 1}:
            raise TaskError("Treatment mapping requires exact -1/+1 values and both groups; no truncation or missing values")
        frame[task.treatment] = values.map({-1: 0, 1: 1}).astype(int)
        mapping["treatment_mapping"] = {"-1": 0, "1": 1}
    elif treatment_mapping == "yes_missing":
        values = frame[task.treatment]
        if not values.dropna().eq("Yes").all():
            raise TaskError("raw treatment must be Yes or missing; missing means control")
        frame[task.treatment] = values.notna().astype(int)
        mapping["treatment_mapping"] = {"Yes": 1, "missing": 0}
    if outcome_mapping == "yes_no":
        if not frame[task.outcome].isin(["Yes", "No"]).all():
            raise TaskError("outcome must contain only Yes/No labels")
        frame[task.outcome] = frame[task.outcome].map({"Yes": 1, "No": 0}).astype(int)
        mapping["outcome_mapping"] = {"Yes": 1, "No": 0}
    for column in numeric:
        try:
            values = pd.to_numeric(frame[column], errors="raise")
            valid = np.isfinite(values.to_numpy(dtype=float)).all()
        except (ValueError, TypeError):
            valid = False
        if not valid or frame[column].map(lambda value: isinstance(value, (bool, np.bool_))).any():
            raise TaskError("Numeric analysis columns must contain finite numbers without missing values or booleans")
        frame[column] = values
    if task.method == "adj":
        if set(frame[task.treatment]) != {0, 1}:
            raise TaskError("Both treatment groups with exact 0/1 values are required; declare an explicit mapping if needed")
        mapping["treatment_value_counts"] = {
            str(int(key)): int(value) for key, value in frame[task.treatment].value_counts().sort_index().items()
        }
    return frame, mapping


def check_task(plan: TaskPlan, *, check_dependencies: bool = True) -> CheckedTask:
    """Validate an entire task locally without reading credentials or writing files."""
    frame, mapping = load_data(plan.task)
    run = plan.run_config()
    main, annotation = plan.model_configs()
    modules = ["numpy", "pandas", "scipy", "sklearn", "statsmodels", "tqdm",
               "unidecode", "causallearn", "nltk", "pydot"]
    modules.append("openai")
    if run.options.get("use_lingam"):
        modules.append("lingam")
    if type(annotation).__name__ == "JevConfig" or run.options.get("factor_judge") or run.options.get("factor_dedup"):
        modules.append("httpx")
    if plan.task.data.suffix.lower() == ".xlsx":
        modules.append("openpyxl")
    missing = [name for name in modules if importlib.util.find_spec(name) is None] if check_dependencies else []
    if missing:
        hint = "; install coatlearn[lingam] for LiNGAM" if "lingam" in missing else "; install coatlearn with its declared dependencies"
        raise TaskError("Missing dependencies: " + ", ".join(missing) + hint)
    # WordNet matters only on the legacy overlap path. Never download it.
    if check_dependencies and not run.options.get("factor_dedup"):
        try:
            import nltk
            nltk.corpus.wordnet.ensure_loaded()
        except (ImportError, LookupError):
            raise TaskError("WordNet corpus NOT found; install it explicitly with python -m nltk.downloader wordnet omw-1.4") from None
    summary = {
        "mode": "preflight", "method": plan.task.method, "rows": len(frame),
        "columns_used": list(frame), "required_env_names": plan.required_env_names(),
        "credentials_checked": False, "network_requests": 0,
        "annotation_backend": "kev" if type(annotation).__name__ == "KevConfig" else "jev" if type(annotation).__name__ == "JevConfig" else "openai",
        "cost_note": "Annotation scales with proposed factors × rows × rounds; factors and exact price are unknown before discovery.",
        "warnings": list(plan.warnings),
        "configuration": plan.snapshot(),
    }
    return CheckedTask(plan, frame, summary, mapping)


def execute_task(checked: CheckedTask, *, output_dir: str | Path | None = None,
                 metadata: dict[str, Any] | None = None) -> tuple[dict[str, Any], Path]:
    """Run an already checked task once, refusing existing attempt directories.

    This is an explicit paid-operation interface. Callers own credential loading;
    adapters read their named environment variable lazily on actual first use.
    Input projection is reused from preflight, rather than rereading changed data.
    """
    import coatlearn
    from coatlearn.models.jev import JevConfig

    plan = checked.plan
    if plan.placeholder:
        raise TaskError("REFUSING execution: CONFIG NOT CONFIRMED; replace placeholder models/endpoints")
    config = plan.run_config()
    output = Path(output_dir or config.output_dir or
                  (Path.cwd() / "runs" / (plan.task.method + "-" + uuid4().hex))).resolve()
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        raise TaskError("REFUSING: output directory already exists; choose a fresh attempt directory") from None
    config = replace(config, output_dir=str(output))
    snapshot = plan.snapshot()
    snapshot["run"]["output_dir"] = str(output)
    (output / "config.resolved.json").write_text(json.dumps(snapshot, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    mapping = dict(checked.mapping)
    if metadata:
        # Only the dataset identifier is accepted, never arbitrary caller data.
        if metadata.get("dataset") in {"hillstrom", "retailhero"}:
            mapping["dataset"] = metadata["dataset"]
    (output / "input_mapping.json").write_text(json.dumps(mapping, indent=2) + "\n", encoding="utf-8")
    try:
        main, annotation = plan.model_configs()
        if isinstance(annotation, JevConfig):
            from coatlearn.annotation.measurement import decision_annotator
            annotation = decision_annotator(annotation)
        kwargs = {"text_col": plan.task.text_col, "main_model": main,
                  "annotation_model": annotation, "config": config}
        if plan.task.method == "mb":
            result = coatlearn.run_mb(checked.frame, plan.task.target, **kwargs)
        elif plan.task.method == "pag":
            result = coatlearn.run_pag(checked.frame, list(plan.task.known_variables), **kwargs)
        else:
            result = coatlearn.run_adj(checked.frame, plan.task.treatment, plan.task.outcome, **kwargs)
    except Exception as exc:
        # Wrapper imports/construction/configuration can fail before the core's
        # own exception handler. Preserve a terminal result without raw text.
        result = {"method": plan.task.method, "status": "failed", "factors": {},
                  "graph": None, "adjustment_sets": None, "estimates": None,
                  "diagnostics": {"error": {"stage": "task_execution",
                                              "type": type(exc).__name__,
                                              "message": None}}, "provenance": {}}
    result["provenance"]["task_configuration"] = snapshot
    result["provenance"]["input_mapping"] = mapping
    (output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result, output
