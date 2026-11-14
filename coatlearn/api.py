"""Programmatic entry points over the COAT core experiment classes.

This module is a thin wrapper around the core classes ``COATMB``,
``COATPAG`` and ``COATADJ``.  It does not re-implement any algorithm:
factor proposal, annotation, graph construction and effect estimation
all happen in the core modules.  The wrapper is responsible only for

* strict input validation *before* any model client is created,
* lazy resolution of model specifications (``ModelConfig`` -> client via
  ``coatlearn.models.openai.create_model``, or pass-through of injected objects),
* mapping :class:`RunConfig` onto each core class's own configuration
  surface (``set_config`` / ``exp_config``) without hiding any parameter
  that was actually used,
* returning a strictly JSON-safe result dictionary with conservative
  statuses, and
* never returning or logging API keys, raw prompts, or model responses.

Nothing runs at import time: no client, no experiment, no NLTK corpus
download, no dataset access.  Heavy core modules are imported lazily
inside :func:`_load_core`, which unit tests may monkeypatch.

Status semantics (never claim success beyond wrapper-visible evidence):

* ``"completed"``             - the wrapped core call returned normally and
                                no degraded conditions were recorded.
* ``"completed_with_errors"`` - the core call returned, but annotation
                                failures, unknown option keys, or other
                                degraded conditions were recorded.
* ``"failed"``                - the wrapped core call raised; the result
                                still carries every diagnostic collected
                                up to the failure.

Integration notes (see ``README.md``): the wrapper reads core diagnostics
defensively (``annotation_errors`` / ``proposal_errors`` / ``judge_errors``
registries, ``last_adjustment_status``, and the fail-closed
``legacy_adjustment_estimates`` switch).  The PAG graph comes from the core
result's ``final_graph_matrix``, ``node_order`` and ``edge_endpoint_encoding``,
with a legacy ``g`` graph fallback.  Where a diagnostic is unavailable the
result says so instead of guessing.
"""

from __future__ import annotations

import dataclasses
import importlib.metadata
import inspect
from io import BytesIO
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from urllib.parse import urlsplit

import numpy as np
import pandas as pd

logger = logging.getLogger("coatlearn.api")

__all__ = ["RunConfig", "run_mb", "run_pag", "run_adj"]

_CAUSAL_LEARN_NOTE = (
    "The legacy requirements file for the original experiments pinned "
    "causal-learn 0.1.4.0; the current author development environment has "
    "been observed at 0.1.4.5. The original runtime has not been fully "
    "reconstructed, and behavioural equivalence across causal-learn "
    "versions is NOT established. See provenance.library_versions for the "
    "versions this Python environment actually resolves."
)

_ENDPOINT_LEGEND = {
    -1: "TAIL",
    0: "NULL",
    1: "ARROW",
    2: "CIRCLE",
    3: "STAR",
    4: "TAIL_AND_ARROW",
    5: "ARROW_AND_ARROW",
}

# Options that the core experiment surface understands.  Used only to warn
# about likely-typo option keys; unknown keys are still forwarded to the core
# (which ignores them) and recorded in provenance.
_KNOWN_CORE_OPTION_KEYS = {
    "num_workers",
    "vif_threshold",
    "alpha",
    "test_method",
    "prompt_sample_size",
    "save_llm_conversation",
    "save_factor_audit",
    "factor_judge",
    "factor_dedup",
    "fci_depth",
    "proposal_top_k",
    "stability_B",
    "stability_pi",
    "score_threshold",
    "num_round",
    "exp_uid",
    "use_lingam",
    "sample_minimum",
    "legacy_adjustment_estimates",
    "break_cycles",
    "assume_treatment_causes_outcome",
    "factor_template",
    "factor_template_repair",
    "assume_factors_precede_treatment",
    "annotation_batching",
    "support_gate",
    "factor_prediction_gate",
    "observed_factor_definitions",
    "coverage_gate",
    "cell_coverage_min",
    "adjustment_selection",
    "sparse_cell_warning_below",
    "amenability_exempt",
    "amenability_exempt_reason",
}

_RESULT_KEYS = (
    "method",
    "status",
    "factors",
    "graph",
    "adjustment_sets",
    "estimates",
    "diagnostics",
    "provenance",
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class RunConfig:
    """Common run configuration for ``run_mb`` / ``run_pag`` / ``run_adj``.

    Every field is forwarded to the core layer and echoed verbatim in
    ``result["provenance"]["run_config"]`` and (as applied) in
    ``provenance["core_effective_config"]`` - nothing that influences a run
    is hidden.

    Attributes:
        rounds: For ``run_mb``, the number of ``COATMB.single_round`` calls.
            For ``run_pag``, the per-target consecutive MB iteration cap
            (early stop if unchanged), also the outer extension-batch cap.
            For ``run_adj``, it becomes ``exp_config["num_round"]``.
        concurrency: Forwarded as ``num_workers`` (annotation thread pool).
        alpha: Significance level for conditional-independence tests.
        test_method: Conditional-independence test name passed to
            ``causallearn``'s ``CIT`` (e.g. ``"kci"``, ``"rcit"``, ``"fisherz"``).
        seed: Seed for ``numpy.random`` (set before the core call; PAG/ADJ
            subsampling uses the global numpy generator).
        output_dir: Where core artifacts (logs, saved conversations) go.
            ``None`` lets each core class use its own default location
            (``./exp_log_folders/...`` relative to the current working
            directory) - the actual path is reported in provenance.
        save_conversations: Forwarded as ``save_llm_conversation``; when
            true (and ``output_dir`` is set) the core writes raw prompts
            and responses under ``<output_dir>/llm_prompts/``.
        options: Extra keyword arguments passed through to the core's
            configuration (e.g. ``vif_threshold``, ``stability_pi``, or
            ``exp_uid`` for ``run_adj``).  Keys that the core is not known
            to understand are flagged in ``diagnostics`` but still forwarded.
    """

    rounds: int = 1
    concurrency: int = 16
    alpha: float = 0.01
    test_method: str = "kci"
    seed: int = 0
    output_dir: Optional[str] = None
    save_conversations: bool = False
    options: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def validate(self) -> None:
        if not isinstance(self.rounds, int) or isinstance(self.rounds, bool) or self.rounds < 1:
            raise ValueError(f"rounds must be an int >= 1, got {self.rounds!r}")
        if not isinstance(self.concurrency, int) or isinstance(self.concurrency, bool) or self.concurrency < 1:
            raise ValueError(f"concurrency must be an int >= 1, got {self.concurrency!r}")
        if not isinstance(self.alpha, (int, float)) or isinstance(self.alpha, bool) or not (0 < float(self.alpha) < 1):
            raise ValueError(f"alpha must be a float in (0, 1), got {self.alpha!r}")
        if not isinstance(self.test_method, str) or not self.test_method:
            raise ValueError(f"test_method must be a non-empty string, got {self.test_method!r}")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError(f"seed must be an int, got {self.seed!r}")
        if self.output_dir is not None and (not isinstance(self.output_dir, str) or not self.output_dir):
            raise ValueError("output_dir must be None or a non-empty string")
        if not isinstance(self.save_conversations, bool):
            raise ValueError(f"save_conversations must be a bool, got {self.save_conversations!r}")
        if not isinstance(self.options, dict):
            raise ValueError(f"options must be a dict, got {type(self.options).__name__}")


# ---------------------------------------------------------------------------
# JSON-safety and error sanitisation
# ---------------------------------------------------------------------------


def _json_safe(obj: Any, _nulled: Optional[List[str]] = None) -> Any:
    """Recursively convert ``obj`` into strictly JSON-serialisable data.

    numpy scalars/arrays, sets, tuples, Paths and non-finite floats are all
    handled; non-finite floats become ``None`` and are counted (the count is
    reported by the caller so the loss is never silent).
    """
    if _nulled is None:
        _nulled = []
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, (int,)):
        return int(obj)
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            _nulled.append("nonfinite")
            return None
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        if np.isnan(value) or np.isinf(value):
            _nulled.append("nonfinite")
            return None
        return value
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist(), _nulled)
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return _json_safe(dataclasses.asdict(obj), _nulled)
    if isinstance(obj, (set, frozenset)):
        return sorted(str(item) for item in obj)
    if isinstance(obj, (list, tuple)):
        return [_json_safe(item, _nulled) for item in obj]
    if isinstance(obj, dict):
        return {str(key): _json_safe(value, _nulled) for key, value in obj.items()}
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (bytes, bytearray)):
        return obj.decode("utf-8", errors="replace")
    return str(obj)


def _error_diagnostic(exc: BaseException, stage: str) -> Dict[str, Any]:
    """Describe an exception without leaking secrets, prompts or responses.

    Only the exception class name and the stage are recorded - in results
    AND in coatlearn's own log lines. No exception text is emitted anywhere
    by this wrapper, so prompt bodies, model responses and key material
    cannot leak through error reporting.
    """
    return {
        "type": type(exc).__name__,
        "stage": stage,
        "message": None,
        "note": "Exception text is intentionally omitted from results and "
        "log lines; only the exception type and stage are recorded.",
    }


# ---------------------------------------------------------------------------
# Model resolution
# ---------------------------------------------------------------------------


def _resolve_model(spec: Any, role: str) -> Tuple[Any, Dict[str, Any]]:
    """Turn ``spec`` into a model object plus provenance metadata.

    ``spec`` may be a ``coatlearn.models.openai.ModelConfig``, a plain string model
    name, a dict of ``ModelConfig`` fields, or an already-constructed object.
    Client construction happens only here (never at import time);
    ``create_model`` reads credentials lazily on first use - the API key
    itself is never seen, stored, or logged by this wrapper; only the
    *name* of the environment variable is recorded in provenance.
    """
    if spec is None:
        raise ValueError(f"{role}_model is required (pass a ModelConfig or an object with a .chat method)")

    from coatlearn.models.presets import ModelPreset

    if isinstance(spec, ModelPreset):
        # Presets opt into execution-time dotenv loading only after all public
        # data validation. Resolve fresh state for every role and every run.
        resolved = spec._resolve(role)
        model, provenance = _resolve_model(resolved, role)
        provenance["preset"] = spec.name
        provenance["backend"] = spec.backend
        provenance["configuration"] = spec.config.redacted()
        return model, provenance

    if isinstance(spec, (str, dict)) or type(spec).__name__ == "ModelConfig":
        from coatlearn import models as _models

        if isinstance(spec, str):
            config = _models.ModelConfig(model=spec)
        elif isinstance(spec, dict):
            config = _models.ModelConfig(**spec)
        else:
            config = spec
        model = _models.create_model(config)
        provenance = {
            "kind": "ModelConfig",
            "model": getattr(config, "model", None),
            "base_url": getattr(config, "base_url", None),
            "api_key_env": getattr(config, "api_key_env", None),
            "temperature": _json_safe(getattr(config, "temperature", None)),
            "max_tokens": _json_safe(getattr(config, "max_tokens", None)),
            "reasoning_effort": _json_safe(getattr(config, "reasoning_effort", None)),
            "timeout": _json_safe(getattr(config, "timeout", None)),
            "max_retries": _json_safe(getattr(config, "max_retries", None)),
            "max_concurrency": _json_safe(getattr(config, "max_concurrency", None)),
            "requests_per_minute": _json_safe(getattr(config, "requests_per_minute", None)),
        }
        _check_protocol(model, role)
        return model, provenance

    # Injected object path: used as-is, never inspected further.
    model = spec
    provenance = {"kind": "injected_object", "class": type(spec).__name__}
    if hasattr(spec, "annotate_factor"):
        provenance["protocol"] = "annotate_factor"
    _check_protocol(model, role)
    return model, provenance


def _check_protocol(model: Any, role: str) -> None:
    if role == "main":
        if not callable(getattr(model, "chat", None)):
            raise TypeError(
                "main_model must expose .chat(prompt, return_text=True); "
                "Jev-style annotators (annotate_factor/prepare_factor) can only "
                "be used as annotation_model"
            )
        return
    if role == "annotation":
        has_chat = callable(getattr(model, "chat", None))
        has_factor_protocol = callable(getattr(model, "annotate_factor", None))
        if not (has_chat or has_factor_protocol):
            raise TypeError(
                "annotation_model must expose .chat(prompt, return_text=True) or the "
                "factor protocol (.annotate_factor(name, definition, text, *, sample_id=None))"
            )


_SNAPSHOT_METHODS = ("annotation_metadata", "metadata_snapshot", "finalized_specs", "archived_factor_snapshot", "stats")

_SOFT_MEASUREMENT_NOTE = (
    "The annotation backend exposes measurement metadata; where annotations "
    "are soft-scale measurements (probability-weighted expectations, "
    "value_policy, probabilities, confidence), they are NOT equivalent to "
    "the original hard-label annotations."
)


def _snapshot_model_runtime(model: Any) -> Optional[Dict[str, Any]]:
    """Read-only, non-IO snapshot of a model object's measurement metadata.

    Calls only zero-argument snapshot methods (``annotation_metadata``,
    ``metadata_snapshot``, ``finalized_specs``, ``archived_factor_snapshot``,
    ``stats``) that the model exposes; methods requiring arguments are skipped and recorded
    as such.  Snapshots are converted with the strict JSON-safe serialiser;
    prompts, responses and keys are never requested.  Failures are recorded
    by exception type only.
    """
    snapshots: Dict[str, Any] = {}
    for name in _SNAPSHOT_METHODS:
        method = getattr(model, name, None)
        if not callable(method):
            continue
        try:
            requires_args = any(
                parameter.default is inspect.Parameter.empty
                and parameter.kind
                in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    inspect.Parameter.KEYWORD_ONLY,
                )
                for parameter in inspect.signature(method).parameters.values()
            )
        except (TypeError, ValueError):
            requires_args = False
        if requires_args:
            snapshots[name] = {"skipped": "method requires arguments"}
            continue
        try:
            snapshots[name] = _json_safe(method())
        except Exception as exc:  # noqa: BLE001 - snapshot is diagnostic only
            snapshots[name] = {"unavailable_error_type": type(exc).__name__}
    return snapshots or None


def _augment_models_provenance(
    models_provenance: Dict[str, Any], models: Dict[str, Any]
) -> Dict[str, Any]:
    """Attach post-run runtime snapshots to per-role provenance entries."""
    augmented: Dict[str, Any] = {}
    for role, provenance in models_provenance.items():
        entry = dict(provenance)
        snapshot = _snapshot_model_runtime(models.get(role))
        if snapshot is not None:
            entry["runtime_metadata"] = snapshot
        augmented[role] = entry
    return augmented


def _annotation_backend_diagnostics(models: Dict[str, Any]) -> Dict[str, Any]:
    """Flag soft-measurement annotation backends in diagnostics."""
    annotation_model = models.get("annotation")
    exposes_specs = any(
        callable(getattr(annotation_model, name, None))
        for name in ("annotation_metadata", "metadata_snapshot", "finalized_specs")
    )
    return {
        "main_class": type(models.get("main")).__name__ if models.get("main") is not None else None,
        "annotation_class": type(annotation_model).__name__ if annotation_model is not None else None,
        "exposes_measurement_metadata": exposes_specs,
        "note": _SOFT_MEASUREMENT_NOTE if exposes_specs else None,
    }


# ---------------------------------------------------------------------------
# Core loading and configuration
# ---------------------------------------------------------------------------


def _load_core() -> Any:
    """Import the heavy core modules lazily.

    Returns a simple namespace exposing ``COATMB``, ``COATPAG``,
    ``COATADJ``, ``CausalEffectEstimator`` and ``is_visible``.  Unit tests
    monkeypatch this function to inject fakes and keep the suite offline.
    """
    from coatlearn.discovery import adj as coat_adj, mb as coat_mb, pag as coat_pag
    from coatlearn.causality.graph import is_visible as _is_visible

    class _CoreModules:
        COATMB = coat_mb.COATMB
        COATPAG = coat_pag.COATPAG
        COATADJ = coat_adj.COATADJ
        CausalEffectEstimator = coat_adj.CausalEffectEstimator
        is_visible = staticmethod(_is_visible)

    return _CoreModules


def _apply_core_config(core: Any, explicit: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
    """Forward ``explicit`` fields then user ``options`` to ``core.set_config``.

    The core's own pre-call configuration keys are snapshotted so that
    unknown option keys can be flagged (they are still forwarded - the core
    stores and silently ignores unknown keys).  The effective configuration
    is returned for provenance.
    """
    default_keys = set(getattr(core, "config", {}).keys())
    merged = dict(explicit)
    merged.update(options)
    core.set_config(**merged)
    unknown = sorted(set(options) - (default_keys | _KNOWN_CORE_OPTION_KEYS))
    return {
        "unknown_option_keys": unknown,
        "effective_config": _json_safe(getattr(core, "config", {})),
    }


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def _validate_frame(data: Any) -> pd.DataFrame:
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame or local CSV path (str or pathlib.Path)")
    if len(data) == 0:
        raise ValueError("data is empty")
    return data


def _load_input(data: Any, columns: Sequence[str]) -> pd.DataFrame:
    """Normalize a DataFrame or local CSV without model/credential side effects.

    The CSV is read once as bytes, headers checked, and only the named columns
    parsed. Remote URLs, implicit recoding, row dropping and truth-column
    inference are deliberately not supported. Parser errors never echo cells.
    """
    if not isinstance(data, (str, Path)):
        return _validate_frame(data)
    if isinstance(data, str) and (urlsplit(data).scheme or data.startswith("//")):
        raise ValueError("data must be a local CSV path; remote URLs are not supported")
    path = Path(data)
    if path.suffix.lower() != ".csv":
        raise ValueError("File input supports .csv only; prepare a pandas DataFrame for other formats or custom parsing")
    try:
        contents = path.read_bytes()
    except OSError:
        raise ValueError("Cannot read local CSV; check that the file exists and is readable") from None
    try:
        header = pd.read_csv(BytesIO(contents), nrows=0)
        _require_columns(header, columns)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeError):
        raise ValueError("Cannot parse CSV; use a UTF-8, comma-separated file with a header row") from None
    try:
        frame = pd.read_csv(BytesIO(contents), usecols=list(columns))
    except (ValueError, UnicodeError):
        raise ValueError("Cannot parse selected CSV columns; check formatting or pass a prepared DataFrame") from None
    return _validate_frame(frame.loc[:, list(columns)].copy())


def _require_columns(data: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [col for col in columns if col not in data.columns]
    if missing:
        raise ValueError(f"data is missing required column(s): {missing}; available: {list(data.columns)}")


def _validate_text_column(data: pd.DataFrame, text_col: str) -> None:
    series = data[text_col]
    if series.isna().any():
        raise ValueError(f"text column {text_col!r} contains missing values; drop or fill them before calling")
    non_str = series.map(lambda v: not isinstance(v, str))
    if non_str.any():
        raise ValueError(
            f"text column {text_col!r} must contain only strings "
            f"({int(non_str.sum())} non-string value(s) found)"
        )


def _numeric_values(data: pd.DataFrame, column: str) -> np.ndarray:
    series = pd.to_numeric(data[column], errors="coerce")
    if series.isna().any():
        raise ValueError(f"column {column!r} must be numeric without missing values ({int(series.isna().sum())} invalid cells)")
    return series.to_numpy(dtype=float)


def _validate_run_config(config: Optional[RunConfig]) -> RunConfig:
    if config is None:
        config = RunConfig()
    if not isinstance(config, RunConfig):
        raise TypeError(f"config must be a RunConfig or None, got {type(config).__name__}")
    config.validate()
    return config


def _base_limitations() -> List[str]:
    return [
        "No formal success or quality criteria for discovered factors/graphs "
        "have been established for this artifact; 'status' describes "
        "wrapper-visible execution state only.",
        _CAUSAL_LEARN_NOTE,
    ]


# ---------------------------------------------------------------------------
# Result assembly
# ---------------------------------------------------------------------------


def _new_result(method: str) -> Dict[str, Any]:
    return {
        "method": method,
        "status": "failed",
        "factors": None,
        "graph": None,
        "adjustment_sets": None,
        "estimates": None,
        "diagnostics": {},
        "provenance": {},
    }


def _summarize_error_registry(errors: Any) -> Optional[Dict[str, Any]]:
    """Conservative summary of a core error registry (annotation/proposal/judge).

    The core documents these as ``{key: exception_type}`` (keys may be factor
    names or ``(factor_name, sample_idx)`` tuples).  Only keys and
    type-like strings are reported - never exception messages, which could
    embed prompt/response fragments.  Unknown shapes are summarised by count
    only.
    """
    if errors is None:
        return None
    if isinstance(errors, dict):
        entries: Dict[str, Any] = {}
        for key, value in list(errors.items())[:50]:
            if isinstance(value, str) and value and not any(c.isspace() for c in value):
                entries[str(key)] = value
            elif isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
                entries[str(key)] = _json_safe(value)
            else:
                entries[str(key)] = {"recorded": True}
        return {"shape": "dict", "count_entries": len(errors), "entries": entries}
    if isinstance(errors, (list, tuple)):
        return {"shape": "list", "count_entries": len(errors)}
    return {"shape": type(errors).__name__}


def _serialize_graph(
    g: Any,
    node_order: Optional[Sequence[str]],
    node_order_source: str,
    extra_notes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Serialise a causal-learn graph object without mutating it.

    The endpoint matrix is the raw ``g.graph`` integer matrix; codes follow
    causal-learn's ``Endpoint`` enum (see ``endpoint_legend``).  ``node_order``
    is the label order assumed for the matrix rows/columns; when the length
    does not match the matrix the graph is reported as unavailable rather
    than guessing an order.
    """
    notes = list(extra_notes or [])
    if g is None:
        return {
            "available": False,
            "reason": "core did not expose a final graph object",
            "node_order": None,
            "endpoint_matrix": None,
            "endpoint_legend": _ENDPOINT_LEGEND,
            "node_order_source": node_order_source,
            "notes": notes,
        }
    matrix = getattr(g, "graph", None)
    if matrix is None:
        return {
            "available": False,
            "reason": "graph object has no .graph endpoint matrix (e.g. use_lingam output)",
            "node_order": None,
            "endpoint_matrix": None,
            "endpoint_legend": _ENDPOINT_LEGEND,
            "node_order_source": node_order_source,
            "notes": notes,
        }
    array = np.asarray(matrix)
    order = None
    if node_order is not None:
        order = [str(name) for name in node_order]
        if array.ndim != 2 or array.shape[0] != len(order) or array.shape[1] != len(order):
            return {
                "available": False,
                "reason": (
                    f"endpoint matrix shape {tuple(array.shape)} does not match "
                    f"node_order length {len(order)} ({node_order_source})"
                ),
                "node_order": order,
                "endpoint_matrix": None,
                "endpoint_legend": _ENDPOINT_LEGEND,
                "node_order_source": node_order_source,
                "notes": notes,
            }
    endpoint_matrix = [[int(value) for value in row] for row in array.tolist()]
    notes.append(
        "Endpoint codes follow causal-learn's Endpoint enum; confirm them "
        "against the causal-learn version recorded in provenance."
    )
    return {
        "available": True,
        "reason": None,
        "node_order": order,
        "endpoint_matrix": endpoint_matrix,
        "endpoint_legend": _ENDPOINT_LEGEND,
        "node_order_source": node_order_source,
        "notes": notes,
    }


def _graph_from_run_result(
    matrix: Any,
    node_order: Any,
    encoding: Any,
    direction_constraints: Any,
    fallback_g: Any = None,
) -> Dict[str, Any]:
    """Serialise the graph the core returned in its run result.

    Primary source: ``final_graph_matrix`` + ``node_order`` +
    ``edge_endpoint_encoding`` (the core copies the matrix, so reading it
    cannot mutate the core's graph and no discovery step is re-run).  If the
    result carries no matrix, fall back to a legacy ``g`` object's
    ``.graph``; otherwise report the graph as unavailable with the reason.
    """
    if matrix is not None:
        array = np.asarray(matrix)
        order = [str(name) for name in node_order] if node_order is not None else None
        if order is not None and (array.ndim != 2 or array.shape[0] != len(order) or array.shape[1] != len(order)):
            return {
                "available": False,
                "reason": (
                    f"final_graph_matrix shape {tuple(array.shape)} does not "
                    f"match node_order length {len(order)}"
                ),
                "node_order": order,
                "endpoint_matrix": None,
                "endpoint_legend": _ENDPOINT_LEGEND,
                "edge_endpoint_encoding": _json_safe(encoding),
                "node_order_source": "core result node_order",
                "notes": [],
            }
        return {
            "available": True,
            "reason": None,
            "node_order": order,
            "endpoint_matrix": [[int(value) for value in row] for row in array.tolist()],
            "endpoint_legend": _ENDPOINT_LEGEND,
            "edge_endpoint_encoding": _json_safe(encoding),
            "node_order_source": "core result node_order",
            "notes": [
                "Matrix is the core's copied final_graph_matrix; node_order is "
                "the core-provided row/column order. No discovery step was "
                "re-run to obtain it.",
            ],
        }
    if fallback_g is not None:
        serialized = _serialize_graph(fallback_g, None, "legacy core g.graph")
        serialized["edge_endpoint_encoding"] = _json_safe(encoding)
        return serialized
    return {
        "available": False,
        "reason": "core result contained neither final_graph_matrix nor a graph object",
        "node_order": _ordered_names(node_order),
        "endpoint_matrix": None,
        "endpoint_legend": _ENDPOINT_LEGEND,
        "edge_endpoint_encoding": _json_safe(encoding),
        "node_order_source": None,
        "notes": [],
    }


def _safe_library_versions() -> Dict[str, Optional[str]]:
    versions = {}
    for distribution in ("causal-learn", "numpy", "pandas", "scikit-learn", "statsmodels", "openai", "lingam"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except Exception:
            versions[distribution] = None
    return versions


def _build_provenance(
    method: str,
    config: RunConfig,
    core: Any,
    models_provenance: Dict[str, Any],
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    core_config = getattr(core, "config", None)
    if core_config is None:
        core_config = getattr(core, "exp_config", None)
    provenance = {
        "method": method,
        "core_class": type(core).__name__,
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": config.seed,
        "run_config": dataclasses.asdict(config),
        "core_effective_config": _json_safe(core_config),
        "models": models_provenance,
        "library_versions": _safe_library_versions(),
        "library_version_notes": _CAUSAL_LEARN_NOTE,
    }
    if extra:
        provenance.update(_json_safe(extra))
    return provenance


def _final_status(degraded: bool) -> str:
    return "completed_with_errors" if degraded else "completed"


def _finish(
    result: Dict[str, Any],
    factors: Optional[Dict[str, Any]],
    graph: Optional[Dict[str, Any]],
    adjustment_sets: Optional[Any],
    estimates: Optional[Any],
    diagnostics: Dict[str, Any],
    provenance: Dict[str, Any],
    degraded: bool,
    failed: bool = False,
) -> Dict[str, Any]:
    """Assemble the final result payload.

    ``failed`` has strictly higher priority than ``degraded``: a wrapped core
    call that raised is reported as ``"failed"`` and is never softened into
    ``"completed_with_errors"`` by degraded conditions collected around it.
    """
    diagnostics.setdefault("limitations", []).extend(_base_limitations())
    nulled: List[str] = []
    factors_s = _json_safe(factors, nulled)
    graph_s = _json_safe(graph, nulled)
    adjustment_sets_s = _json_safe(adjustment_sets, nulled)
    estimates_s = _json_safe(estimates, nulled)
    diagnostics_s = _json_safe(diagnostics, nulled)
    provenance_s = _json_safe(provenance, nulled)
    if nulled and isinstance(diagnostics_s, dict):
        diagnostics_s["nonfinite_values_nulled"] = len(nulled)
    payload = {
        "method": result["method"],
        "status": "failed" if failed else _final_status(degraded),
        "factors": factors_s,
        "graph": graph_s,
        "adjustment_sets": adjustment_sets_s,
        "estimates": estimates_s,
        "diagnostics": diagnostics_s,
        "provenance": provenance_s,
    }
    result.clear()
    result.update(payload)
    try:
        json.dumps(result)
    except (TypeError, ValueError) as exc:  # pragma: no cover - programmer error
        raise RuntimeError("assembled result is not JSON-safe") from exc
    return result


def _factor_descriptions(factor_names: Sequence[str], factor_pools: Any) -> Dict[str, Optional[str]]:
    pools = factor_pools if isinstance(factor_pools, dict) else {}
    return {str(name): _json_safe(pools.get(name)) for name in factor_names}


def _sorted_names(names: Any) -> List[str]:
    """Names of an unordered collection (a set); sorted for stable output."""
    if names is None:
        return []
    try:
        return sorted(str(name) for name in names)
    except TypeError:
        return [str(name) for name in names]


def _ordered_names(names: Any) -> Optional[List[str]]:
    """Names of an ORDERED core structure (final_V, W_list).

    The order is significant: endpoint-matrix rows/columns are indexed in
    this order, so it must never be sorted or otherwise rearranged.
    """
    if names is None:
        return None
    return [str(name) for name in names]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_mb(
    data: Union[pd.DataFrame, str, Path],
    target: str,
    *,
    text_col: str = "X",
    main_model: Any,
    annotation_model: Any,
    config: Optional[RunConfig] = None,
) -> Dict[str, Any]:
    """Markov-blanket discovery for ``target`` from text via ``COATMB``.

    Only the ``target`` and ``text_col`` columns of ``data`` are passed to
    the core; every other column (e.g. ground-truth factors used for
    evaluation) is deliberately excluded from discovery inputs.

    Args:
        data: DataFrame or local CSV path (str/Path), one row per sample;
            must contain ``target`` (numeric) and ``text_col`` (non-missing
            strings). Relative CSV paths use the current working directory.
            Only named columns are parsed; URLs are not supported.
        target: Name of the numeric target column.
        text_col: Name of the free-text column.
        main_model: ``ModelConfig`` or object with
            ``.chat(prompt, return_text=True)``; used for factor proposal
            and LLM-judge checks.
        annotation_model: ``ModelConfig``, chat object, or Jev-style
            annotator (``annotate_factor``/``prepare_factor``) used to score
            factors on every text sample.
        config: Optional :class:`RunConfig`.

    Returns:
        JSON-safe dict with keys ``method, status, factors, graph,
        adjustment_sets, estimates, diagnostics, provenance``.  ``graph``
        is always unavailable for MB (the core MB routine returns a set, not
        a graph); ``adjustment_sets`` and ``estimates`` are ``None`` (use
        ``run_adj`` for estimation).
    """
    result = _new_result("mb")
    config = _validate_run_config(config)
    if not isinstance(target, str) or not target:
        raise ValueError("target must be a non-empty string")
    if not isinstance(text_col, str) or not text_col:
        raise ValueError("text_col must be a non-empty string")
    if target == text_col:
        raise ValueError("target and text_col must be different columns")
    data = _load_input(data, [target, text_col])
    _require_columns(data, [target, text_col])
    _validate_text_column(data, text_col)
    _numeric_values(data, target)

    main, main_prov = _resolve_model(main_model, "main")
    annotation, annotation_prov = _resolve_model(annotation_model, "annotation")
    annotation_protocol_note = None
    if annotation_prov.get("protocol") == "annotate_factor":
        annotation_protocol_note = (
            "annotation_model uses the factor protocol (annotate_factor); the core "
            "prefers it over .chat for annotation - recorded here for "
            "traceability of the annotation backend."
        )

    core_modules = _load_core()
    meta_df = data.loc[:, [target, text_col]].copy()
    core = core_modules.COATMB(
        main,
        annotation,
        target_name=target,
        meta_df=meta_df,
        text_col=text_col,
    )
    config_info = _apply_core_config(
        core,
        {
            "num_workers": config.concurrency,
            "alpha": float(config.alpha),
            "test_method": config.test_method,
            "save_llm_conversation": config.save_conversations,
        },
        config.options,
    )

    np.random.seed(config.seed)
    output_dir = Path(config.output_dir) if config.output_dir else None
    v_history: List[List[str]] = []
    error = None
    try:
        for _ in range(config.rounds):
            blanket = core.single_round(output_dir=output_dir)
            v_history.append(_sorted_names(blanket))
    except Exception as exc:  # noqa: BLE001 - reported, not raised
        error = _error_diagnostic(exc, stage="COATMB.single_round")
        logger.error("run_mb failed in COATMB.single_round: %s", type(exc).__name__)

    final_blanket = _sorted_names(getattr(core, "V", None))
    annotation_errors = _summarize_error_registry(getattr(core, "annotation_errors", None))
    proposal_errors = _summarize_error_registry(getattr(core, "proposal_errors", None))
    judge_errors = _summarize_error_registry(getattr(core, "judge_errors", None))
    rejected = getattr(core, "rejected_factors", None)
    unstable = getattr(core, "unstable_factors", None)

    diagnostics: Dict[str, Any] = {
        "v_history": v_history,
        "final_blanket": final_blanket,
        "rejected_factors": _json_safe(rejected) if isinstance(rejected, dict) else None,
        "unstable_factors": _sorted_names(unstable) if unstable is not None else None,
        "annotation_errors": annotation_errors,
        "proposal_errors": proposal_errors,
        "judge_errors": judge_errors,
        "unknown_option_keys": config_info["unknown_option_keys"],
        "limitations": [
            "run_mb returns the Markov blanket set; no graph, adjustment sets, "
            "or effect estimates are produced by the core MB routine.",
        ],
    }
    if annotation_protocol_note:
        diagnostics["limitations"].append(annotation_protocol_note)
    if error:
        diagnostics["error"] = error

    degraded = bool(
        error is not None
        or config_info["unknown_option_keys"]
        or any(
            summary is not None and summary.get("count_entries")
            for summary in (annotation_errors, proposal_errors, judge_errors)
        )
    )
    graph = _serialize_graph(None, None, "not_applicable")
    provenance = _build_provenance(
        "mb",
        config,
        core,
        {"main": main_prov, "annotation": annotation_prov},
        {
            "data": {
                "rows": int(len(data)),
                "columns_used": [target, text_col],
                "target": target,
                "text_col": text_col,
            },
            "output_dir": str(output_dir) if output_dir else None,
        },
    )
    provenance["models"] = _augment_models_provenance(
        provenance["models"], {"main": main, "annotation": annotation}
    )
    backend_diagnostics = _annotation_backend_diagnostics({"main": main, "annotation": annotation})
    diagnostics["annotation_backend"] = backend_diagnostics
    if backend_diagnostics.get("exposes_measurement_metadata"):
        diagnostics["limitations"].append(_SOFT_MEASUREMENT_NOTE)
    return _finish(
        result,
        factors=_factor_descriptions(final_blanket, getattr(core, "factor_pools", None)),
        graph=graph,
        adjustment_sets=None,
        estimates=None,
        diagnostics=diagnostics,
        provenance=provenance,
        degraded=degraded,
        failed=error is not None,
    )


def run_pag(
    data: Union[pd.DataFrame, str, Path],
    known_variables: List[str],
    *,
    text_col: str = "X",
    main_model: Any,
    annotation_model: Any,
    config: Optional[RunConfig] = None,
) -> Dict[str, Any]:
    """PAG construction over ``known_variables`` plus discovered factors via ``COATPAG``.

    ``known_variables`` are the initial numeric columns with observed values
    (``init_kv``); they are the *only* data columns fed to the core.  All
    other columns of ``data`` - including ground-truth factors kept for
    evaluation - are excluded from discovery inputs.

    Args:
        data: DataFrame or local CSV path (str/Path), one row per sample;
            must contain every ``known_variables`` column (numeric) and
            ``text_col`` (strings). CSV paths are local and cwd-relative;
            only named columns are parsed.
        known_variables: Non-empty list of initial numeric column names
            (typically the treatment-like and outcome-like variables of the
            setting, e.g. ``["Taste", "Score"]`` for the apple PAG graph 1).
        text_col: Name of the free-text column.
        main_model / annotation_model: As in :func:`run_mb`.
        config: Optional :class:`RunConfig`. ``rounds`` caps each target's
            consecutive MB search (initial and uncertain targets); an unchanged
            MB stops that search early. It also retains the existing cap on
            PAG extension batches, with at most three targets per batch.
            Saved measurements are shared, but each target has its own MB.

    Returns:
        JSON-safe dict as in :func:`run_mb`.  ``graph`` is serialised from
        the core result's ``final_graph_matrix`` / ``node_order`` /
        ``edge_endpoint_encoding`` (the core copies the matrix; nothing is
        mutated and no discovery step is re-run).  ``adjustment_sets``/
        ``estimates`` are ``None`` (PAG is a discovery stage; use ``run_adj``
        for estimation).
    """
    result = _new_result("pag")
    config = _validate_run_config(config)
    if not isinstance(known_variables, (list, tuple)) or len(known_variables) == 0:
        raise ValueError("known_variables must be a non-empty list of column names")
    known_variables = list(known_variables)
    if any(not isinstance(name, str) or not name for name in known_variables):
        raise ValueError("known_variables entries must be non-empty strings")
    if len(set(known_variables)) != len(known_variables):
        raise ValueError("known_variables contains duplicates")
    if not isinstance(text_col, str) or not text_col:
        raise ValueError("text_col must be a non-empty string")
    if text_col in known_variables:
        raise ValueError("text_col must not appear in known_variables")
    data = _load_input(data, known_variables + [text_col])
    _require_columns(data, known_variables + [text_col])
    _validate_text_column(data, text_col)
    init_kv = {name: _numeric_values(data, name).tolist() for name in known_variables}
    text_values = data[text_col].to_numpy()

    main, main_prov = _resolve_model(main_model, "main")
    annotation, annotation_prov = _resolve_model(annotation_model, "annotation")
    annotation_protocol_note = None
    if annotation_prov.get("protocol") == "annotate_factor":
        annotation_protocol_note = (
            "annotation_model uses the factor protocol (annotate_factor); the core "
            "prefers it over .chat for annotation - recorded here for "
            "traceability of the annotation backend."
        )

    core_modules = _load_core()
    core = core_modules.COATPAG(
        main,
        annotation,
        U_list=known_variables,
        init_kv=init_kv,
        text_values=text_values,
        log_root_path=config.output_dir,
    )
    config_info = _apply_core_config(
        core,
        {
            "num_workers": config.concurrency,
            "alpha": float(config.alpha),
            "test_method": config.test_method,
            "save_llm_conversation": config.save_conversations,
        },
        config.options,
    )

    np.random.seed(config.seed)
    error = None
    pag_result: Dict[str, Any] = {}
    try:
        pag_result = core.run_algorithm(num_round=config.rounds) or {}
    except Exception as exc:  # noqa: BLE001 - reported, not raised
        error = _error_diagnostic(exc, stage="COATPAG.run_algorithm")
        logger.error("run_pag failed in COATPAG.run_algorithm: %s", type(exc).__name__)

    final_v = pag_result.get("final_V")
    if final_v is None:
        final_v = getattr(core, "V", None)
    final_v_names = _ordered_names(final_v)
    graph = _graph_from_run_result(
        pag_result.get("final_graph_matrix"),
        pag_result.get("node_order", final_v_names),
        pag_result.get("edge_endpoint_encoding"),
        pag_result.get("direction_constraints"),
        fallback_g=getattr(core, "g", None),
    )

    def _pag_from_result_or_core(key: str) -> Any:
        if isinstance(pag_result, dict) and key in pag_result:
            return pag_result[key]
        return getattr(core, key, None)

    annotation_errors = _summarize_error_registry(_pag_from_result_or_core("annotation_errors"))
    proposal_errors = _summarize_error_registry(_pag_from_result_or_core("proposal_errors"))
    judge_errors = _summarize_error_registry(_pag_from_result_or_core("judge_errors"))
    diagnostics: Dict[str, Any] = {
        "stages": _json_safe(pag_result.get("stages")),
        "llm_proposed_count": pag_result.get("llm_proposed_count"),
        "final_V": final_v_names,
        "annotation_errors": annotation_errors,
        "proposal_errors": proposal_errors,
        "judge_errors": judge_errors,
        "unknown_option_keys": config_info["unknown_option_keys"],
        "limitations": [
            "The PAG is a partial ancestral graph learned from LLM-annotated "
            "text factors; endpoint marks are not guarantees of orientability.",
            "adjustment_sets and estimates are None: PAG discovery does not "
            "estimate effects; chain run_adj on the discovered graph instead.",
        ],
    }
    if annotation_protocol_note:
        diagnostics["limitations"].append(annotation_protocol_note)
    if error:
        diagnostics["error"] = error

    degraded = bool(
        error is not None
        or config_info["unknown_option_keys"]
        or any(
            summary is not None and summary.get("count_entries")
            for summary in (annotation_errors, proposal_errors, judge_errors)
        )
        or not graph["available"]
    )
    provenance = _build_provenance(
        "pag",
        config,
        core,
        {"main": main_prov, "annotation": annotation_prov},
        {
            "data": {
                "rows": int(len(data)),
                "columns_used": known_variables + [text_col],
                "known_variables": known_variables,
                "text_col": text_col,
            },
            "log_root_path": getattr(core, "log_root_path", None),
            "direction_constraints": _json_safe(pag_result.get("direction_constraints")),
        },
    )
    provenance["models"] = _augment_models_provenance(
        provenance["models"], {"main": main, "annotation": annotation}
    )
    diagnostics["annotation_backend"] = _annotation_backend_diagnostics(
        {"main": main, "annotation": annotation}
    )
    factor_names = final_v_names or []
    return _finish(
        result,
        factors=_factor_descriptions(factor_names, getattr(core, "factor_pools", None)),
        graph=graph,
        adjustment_sets=None,
        estimates=None,
        diagnostics=diagnostics,
        provenance=provenance,
        degraded=degraded,
        failed=error is not None,
    )


def run_adj(
    data: Union[pd.DataFrame, str, Path],
    treatment: str,
    outcome: str,
    *,
    text_col: str = "X",
    main_model: Any,
    annotation_model: Any,
    config: Optional[RunConfig] = None,
) -> Dict[str, Any]:
    """Adjustment-set discovery and effect estimation via ``COATADJ``.

    The treatment column must already be binary with values in ``{0, 1}``.
    The wrapper never remaps values silently: if your data codes treatment
    as ``{-1, 1}`` (e.g. the discrete food-review stress tests), map it
    explicitly first, e.g. ``data[t] = data[t].map({-1: 0, 1: 1})``.

    ``data`` accepts a DataFrame or local CSV path (str/Path); relative paths
    use the current working directory. Only ``treatment``, ``outcome`` and
    ``text_col`` columns are parsed/passed to the core; other columns
    (ground truth for evaluation) are excluded. URLs are not supported.

    Returns:
        JSON-safe dict as in :func:`run_mb`. By default ``adjustment_sets``
        contains sets passing GAC on the supplied PAG, conditional on graph
        validity and no selection bias. ``[]`` means no valid set was found;
        ``[[]]`` means the only valid set is empty, never a failure fallback.
        Inspect ``estimates.adjustment_criterion`` and ``adjustment_status``.
        Each ``per_set`` record keeps its own estimate or skip/failure reason;
        numerical checks may prevent estimation despite a valid graph set.
        ``0.0`` is distinct from absent estimates. Explicit
        ``options={"legacy_adjustment_estimates": True}`` retains the old
        unverified comparison. GAC does not certify the discovered graph.
    """
    result = _new_result("adj")
    config = _validate_run_config(config)
    for name in (treatment, outcome):
        if not isinstance(name, str) or not name:
            raise ValueError("treatment and outcome must be non-empty strings")
    if not isinstance(text_col, str) or not text_col:
        raise ValueError("text_col must be a non-empty string")
    if len({treatment, outcome, text_col}) != 3:
        raise ValueError("treatment, outcome and text_col must be three distinct columns")
    data = _load_input(data, [treatment, outcome, text_col])
    _require_columns(data, [treatment, outcome, text_col])
    _validate_text_column(data, text_col)
    t_values = _numeric_values(data, treatment)
    _numeric_values(data, outcome)
    unique_t = {float(v) for v in np.unique(t_values)}
    if not unique_t.issubset({0.0, 1.0}):
        raise ValueError(
            f"treatment column {treatment!r} must be binary 0/1 after mapping; "
            f"found values {sorted(unique_t)}. Map explicitly, e.g. "
            f"data[{treatment!r}] = data[{treatment!r}].map({{-1: 0, 1: 1}}), "
            "before calling run_adj (the wrapper never remaps silently)."
        )

    exp_uid = config.options.get("exp_uid") or (
        f"coat_adj_{time.strftime('%Y%m%d_%H%M%S')}_seed{config.seed}"
    )
    exp_config: Dict[str, Any] = {
        "exp_uid": str(exp_uid),
        "num_round": config.rounds,
        "test_method": config.test_method,
        "alpha": float(config.alpha),
        "num_workers": config.concurrency,
        "save_llm_conversation": config.save_conversations,
    }
    passthrough_options = {k: v for k, v in config.options.items() if k != "exp_uid"}
    exp_config.update(passthrough_options)
    unknown_option_keys = sorted(set(passthrough_options) - _KNOWN_CORE_OPTION_KEYS)
    if not isinstance(exp_config.get('break_cycles', False), bool):
        raise ValueError('options.break_cycles must be a bool')
    if exp_config.get('break_cycles') and exp_config.get('legacy_adjustment_estimates'):
        raise ValueError('break_cycles cannot be combined with legacy adjustment estimates')
    from coatlearn.annotation.prediction import (
        validate_prediction_options, validate_prediction_references, validate_observed_definitions,
    )
    validate_observed_definitions(exp_config.get("observed_factor_definitions") or {})
    if set(exp_config.get("observed_factor_definitions") or {}) - {treatment, outcome}:
        raise ValueError("ADJ observed_factor_definitions may describe only treatment and outcome")
    prediction_options = exp_config.get("factor_prediction_gate")
    validate_prediction_options(prediction_options)
    if prediction_options is not None and prediction_options.get("enabled", True):
        validate_prediction_references({treatment: t_values, outcome: _numeric_values(data, outcome)})
    if exp_config.get("amenability_exempt") and not exp_config.get(
            "amenability_exempt_reason"):
        raise ValueError(
            "options.amenability_exempt requires options.amenability_exempt_reason; "
            "the amenability check is weakened deliberately and must be recorded"
        )
    cell_min = exp_config.get("cell_coverage_min")
    if cell_min is not None and not (
            isinstance(cell_min, (int, float)) and not isinstance(cell_min, bool)
            and 0 < float(cell_min) <= 1):
        raise ValueError(
            f"options.cell_coverage_min must be a float in (0, 1] or omitted, got {cell_min!r}")
    selection = exp_config.get("adjustment_selection", "max_cardinality")
    if selection not in ("max_cardinality", "size_then_propensity_range"):
        raise ValueError(
            "options.adjustment_selection must be 'max_cardinality' or "
            f"'size_then_propensity_range', got {selection!r}")

    main, main_prov = _resolve_model(main_model, "main")
    annotation, annotation_prov = _resolve_model(annotation_model, "annotation")
    annotation_protocol_note = None
    if annotation_prov.get("protocol") == "annotate_factor":
        annotation_protocol_note = (
            "annotation_model uses the factor protocol (annotate_factor); the core "
            "prefers it over .chat for annotation - recorded here for "
            "traceability of the annotation backend."
        )
    core_modules = _load_core()
    init_kv = {
        treatment: _numeric_values(data, treatment).tolist(),
        outcome: _numeric_values(data, outcome).tolist(),
    }
    text_values = data[text_col].to_numpy()
    core = core_modules.COATADJ(
        main,
        annotation,
        T_name=treatment,
        Y_name=outcome,
        init_kv=init_kv,
        text_values=text_values,
        exp_config=exp_config,
        log_root_path=config.output_dir,
    )

    np.random.seed(config.seed)
    error = None
    adj_result: Dict[str, Any] = {}
    try:
        adj_result = core.run() or {}
    except Exception as exc:  # noqa: BLE001 - reported, not raised
        error = _error_diagnostic(exc, stage="COATADJ.run")
        logger.error("run_adj failed in COATADJ.run: %s", type(exc).__name__)

    w_list = _ordered_names(getattr(core, "W_list", None)) or []
    g = getattr(core, "g", None)
    graph = _serialize_graph(
        g,
        w_list if w_list else None,
        "core.W_list",
        extra_notes=[
            "One-sided visibility (T -> Y) is reported in estimates as a "
            "diagnostic only; it is not a claim of general identifiability.",
        ],
    )

    est_ate_list = adj_result.get("est_ate_list")
    ate_by_size = adj_result.get("ate_by_size")
    t_idx = w_list.index(treatment) if treatment in w_list else None
    y_idx = w_list.index(outcome) if outcome in w_list else None

    def _from_result_or_core(key: str) -> Any:
        if isinstance(adj_result, dict) and key in adj_result:
            return adj_result[key]
        return getattr(core, key, None)

    adjustment_status = _from_result_or_core("last_adjustment_status")
    legacy_flag = bool(_from_result_or_core("legacy_adjustment_estimates") or False)
    direction_constraints = _from_result_or_core("direction_constraints")
    core_diagnostics = _from_result_or_core("diagnostics")
    if not isinstance(core_diagnostics, dict):
        core_diagnostics = {}

    visible: Optional[bool] = None
    candidates: Optional[List[List[str]]] = None
    candidates_source: Optional[str] = None
    candidate_note: Optional[str] = None

    core_visible = core_diagnostics.get("is_visible")
    if isinstance(core_visible, bool):
        visible = core_visible
    elif g is not None and t_idx is not None and y_idx is not None:
        try:
            visible = bool(core_modules.is_visible(np.asarray(g.graph), t_idx, y_idx))
        except Exception as exc:  # noqa: BLE001 - diagnostic only
            candidate_note = f"visibility diagnostic unavailable ({type(exc).__name__})"

    criterion = core_diagnostics.get('adjustment_criterion') or {}
    invalid_graph = (not legacy_flag and adjustment_status == 'gac_invalid_graph_abstain'
                     and criterion.get('status') == 'invalid_graph'
                     and criterion.get('reason') == 'directed_or_almost_directed_cycle')
    if invalid_graph:
        # None means GAC sets were not evaluated, not an evaluated empty list.
        # Do not fall back to the same failing search and lose this diagnosis.
        candidates_source = 'not evaluated: supplied graph violates ancestral assumptions'
    elif isinstance(core_diagnostics.get("candidate_adjustment_sets"), list):
        # Preserve exactly the core's sets. Their interpretation comes from
        # adjustment_criterion (GAC) or the explicit legacy/unverified status.
        candidates = [list(entry) if isinstance(entry, (list, tuple)) else entry
                      for entry in core_diagnostics["candidate_adjustment_sets"]]
        candidates_source = "core result diagnostics.candidate_adjustment_sets"
        core_caveat = core_diagnostics.get("caveat")
        if core_caveat:
            candidate_note = str(core_caveat)
    elif g is not None and t_idx is not None and y_idx is not None:
        # Fallback for older core results: read-only recompute via the core's
        # own estimator interface (never a GAC certification either).
        try:
            data_interface = pd.DataFrame(getattr(core, "annotation", {}))
            estimator = core_modules.CausalEffectEstimator(
                data_interface=data_interface,
                T_name=treatment,
                Y_name=outcome,
                legacy_adjustment_estimates=False,
            )
            index_sets = estimator.find_adjustment_sets(
                getattr(core, "W_list"), g, t_idx, y_idx, verbose=False
            )
            candidates = [
                [w_list[i] for i in index_set] if index_set is not None else None
                for index_set in index_sets
            ]
            candidates_source = "fallback: core CausalEffectEstimator.find_adjustment_sets"
            heuristic_caveat = (getattr(estimator, "diagnostics", {}) or {}).get("caveat")
            if heuristic_caveat:
                candidate_note = str(heuristic_caveat)
        except Exception as exc:  # noqa: BLE001 - diagnostic only
            candidate_note = (
                f"adjustment-set candidates unavailable ({type(exc).__name__}); "
                "core may not expose a matching estimator interface"
            )

    cycle_processed = bool((core_diagnostics.get('graph_postprocessing') or {}).get('applied'))
    gac_checked = (not legacy_flag
                   and criterion.get('criterion') == 'generalized_adjustment_criterion'
                   and criterion.get('status') in {
                       'not_amenable', 'no_adjustment_set', 'adjustment_sets_found'})
    per_set: Optional[List[Dict[str, Any]]] = None
    if gac_checked:
        # Use actual per-set records, including skips. Never zip a shortened
        # list of estimates with the original (longer) list of sets.
        per_set = core_diagnostics.get('sets', [])
    elif candidates is not None and isinstance(est_ate_list, list) and len(candidates) == len(est_ate_list):
        per_set = [
            {'adjustment_set': nodes, 'estimate': _json_safe(value),
             'kind': 'unadjusted_mean_difference_legacy' if nodes == [] else 'ipw_legacy_unverified'}
            for nodes, value in zip(candidates, est_ate_list)
        ]

    presentations = {
        'gac_invalid_graph_abstain': 'The discovered graph has a directed or almost-directed cycle. GAC does not apply; adjustment sets and per-set estimates are not evaluated. The graph is preserved, not repaired.',
        'gac_estimated': 'Estimates use sets satisfying GAC on the supplied graph; graph and estimation assumptions still apply.',
        'gac_partially_estimated': 'Some GAC sets were estimated; see each per_set status for skipped or failed estimates.',
        'gac_all_sets_skipped': 'GAC sets exist, but none met the numerical estimation requirements; no effect number is reported.',
        'gac_not_amenable_abstain': 'The supplied PAG is not adjustment amenable; this rules out covariate adjustment on this graph, not all identification methods.',
        'gac_no_adjustment_set': 'No set satisfies GAC on the supplied PAG; this is conditional on the graph, not a proof of real-world non-identifiability.',
        'adjustment_criterion_unverified_abstain': 'fail-closed abstention: candidate adjustment sets are listed but the adjustment criterion is not independently certified, so no effect number is produced',
        'not_visible_abstain': 'abstained: single-edge T -> Y visibility is False - an abstention signal, NOT a proof of non-identifiability',
        'no_adjustment_set_found': 'the legacy heuristic search found no candidate adjustment set',
        'legacy_heuristic_estimates_unverified': 'legacy comparison mode: numbers are heuristic and unverified',
        'legacy_all_sets_failed': 'legacy comparison mode: every candidate estimate failed',
    }
    estimates: Dict[str, Any] = {
        'est_ate_list': _json_safe(est_ate_list) if est_ate_list is not None else None,
        'ate_by_size': _json_safe(ate_by_size) if ate_by_size is not None else None,
        'per_set': _json_safe(per_set),
        'largest_adjustment_summary': (
            _json_safe(core_diagnostics.get('largest_adjustment_summary'))
            if gac_checked else None),
        'primary_adjustment_summary': (
            _json_safe(core_diagnostics.get('primary_adjustment_summary'))
            if gac_checked else None),
        'adjustment_status': adjustment_status,
        'adjustment_criterion': _json_safe(criterion) if criterion else None,
        'legacy_adjustment_estimates': legacy_flag,
        't_to_y_visible_diagnostic': visible,
        'candidates_source': candidates_source,
        'notes': [],
    }
    if adjustment_status in presentations:
        estimates['notes'].append(presentations[adjustment_status])
    only_empty_candidate = candidates is not None and len(candidates) == 1 and candidates[0] == []
    if cycle_processed:
        estimates['graph_postprocessing'] = _json_safe(core_diagnostics['graph_postprocessing'])
        estimates['notes'].append(
            'Estimates use adjustment sets common to all minimum edge-deletion hypotheses, '
            'not GAC on the original cyclic graph. Original graph is preserved; '
            'graph realizability and causal assumptions are not certified.')
    if gac_checked and not cycle_processed:
        estimates['notes'].append(
            'GAC checks the supplied PAG under no-selection assumptions; it does not certify '
            'discovered factors, graph accuracy, positivity or propensity-model correctness.')
        if only_empty_candidate:
            estimates['notes'].append('The empty adjustment set passed GAC; it is not a fallback for a failed search.')
    elif not cycle_processed:
        if candidates == []:
            estimates['notes'].append(
                'the heuristic search returned no candidate adjustment set; this does not prove that no valid set exists.')
        if only_empty_candidate:
            estimates['notes'].append(
                'The only candidate is the empty set; without GAC evidence it is unverified. '
                'Older revisions silently fell back to the empty set; check the core revision.')
    if per_set is None and isinstance(est_ate_list, list) and candidates is not None and len(candidates) != len(est_ate_list):
        estimates['notes'].append('Core reports estimates without per-set attribution; per_set is None.')
    if candidate_note:
        estimates['notes'].append(candidate_note)
    if est_ate_list == [] and error is None:
        estimates['notes'].append('no estimates were produced; inspect adjustment_status and per_set. An empty list is not a zero effect.')
    if legacy_flag:
        estimates['notes'].append('legacy_adjustment_estimates is ON: numbers are legacy/unverified, not GAC-checked estimates.')

    annotation_errors = _summarize_error_registry(_from_result_or_core('annotation_errors'))
    proposal_errors = _summarize_error_registry(_from_result_or_core('proposal_errors'))
    judge_errors = _summarize_error_registry(_from_result_or_core('judge_errors'))
    prior_injected = bool(isinstance(direction_constraints, dict)
                          and direction_constraints.get('assume_treatment_causes_outcome'))
    diagnostics: Dict[str, Any] = {
        'W_list_history': _json_safe(adj_result.get('W_list_history')),
        'path_extension_history': _json_safe(adj_result.get('path_extension_history')),
        'pag_refinement': _json_safe(adj_result.get('pag_refinement')),
        'final_W_list': w_list,
        'annotation_errors': annotation_errors,
        'proposal_errors': proposal_errors,
        'judge_errors': judge_errors,
        'last_adjustment_status': adjustment_status,
        'legacy_adjustment_estimates': legacy_flag,
        'core_diagnostics': _json_safe(core_diagnostics),
        'unknown_option_keys': unknown_option_keys,
        'limitations': [
            'Execution status is distinct from identification and estimation status; valid abstention is not a backend error.',
            'GAC validity is conditional on the supplied PAG and no selection bias; it does not certify graph or measurement accuracy.'
            if gac_checked else 'Adjustment candidates lack complete GAC evidence or use explicit legacy comparison mode.',
        ],
    }
    if annotation_protocol_note:
        diagnostics['limitations'].append(annotation_protocol_note)
    if prior_injected:
        diagnostics['limitations'].append(
            'assume_treatment_causes_outcome is ON: T -> Y was injected as a researcher prior, '
            'not discovered; see provenance.direction_constraints.')
    if error:
        diagnostics['error'] = error

    degraded = bool(
        error is not None or unknown_option_keys or legacy_flag
        or any(summary is not None and summary.get('count_entries')
               for summary in (annotation_errors, proposal_errors, judge_errors))
        or any(row.get('status') == 'estimation_failed' for row in (per_set or []))
        or (not gac_checked and not invalid_graph
            and (visible is False or candidates == [] or only_empty_candidate))
    )
    t_counts = {str(int(v)): int((t_values == v).sum()) for v in np.unique(t_values)}
    provenance = _build_provenance(
        "adj",
        config,
        core,
        {"main": main_prov, "annotation": annotation_prov},
        {
            "data": {
                "rows": int(len(data)),
                "columns_used": [treatment, outcome, text_col],
                "treatment": treatment,
                "outcome": outcome,
                "text_col": text_col,
                "treatment_value_counts": t_counts,
            },
            "exp_uid": str(exp_uid),
            "log_root_path": getattr(core, "log_root_path", None),
            "direction_constraints": _json_safe(direction_constraints),
        },
    )
    provenance["models"] = _augment_models_provenance(
        provenance["models"], {"main": main, "annotation": annotation}
    )
    backend_diagnostics = _annotation_backend_diagnostics({"main": main, "annotation": annotation})
    diagnostics["annotation_backend"] = backend_diagnostics
    if backend_diagnostics.get("exposes_measurement_metadata"):
        diagnostics["limitations"].append(_SOFT_MEASUREMENT_NOTE)
    factor_names = [name for name in w_list if name not in (treatment, outcome)]
    return _finish(
        result,
        factors=_factor_descriptions(factor_names, getattr(core, "factor_pools", None)),
        graph=graph,
        adjustment_sets=_json_safe(candidates),
        estimates=estimates,
        diagnostics=diagnostics,
        provenance=provenance,
        degraded=degraded,
        failed=error is not None,
    )
