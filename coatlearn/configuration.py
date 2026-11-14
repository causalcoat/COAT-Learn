"""Validated, credential-free task plans; parsing never constructs a model client.

TOML v1 is deliberately strict. Historical JSON keeps unknown core options (with
warnings), and resolves its recorded repository-relative paths against this
checkout's root. Paths explicitly supplied as overrides are config-relative.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import tomllib
from typing import TYPE_CHECKING, Annotated, Any, Literal, Mapping, Union

if TYPE_CHECKING:
    from .api import RunConfig
    from coatlearn.models.openai import ModelConfig
    from coatlearn.models.jev import JevConfig
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

__all__ = ["ConfigError", "TaskPlan", "load_task"]


class ConfigError(ValueError):
    """A configuration failure whose message never includes supplied values."""


Nonempty = Annotated[str, Field(min_length=1, pattern=r"\S")]
PositiveInt = Annotated[int, Field(ge=1)]
NonnegativeInt = Annotated[int, Field(ge=0)]
PositiveFloat = Annotated[float, Field(gt=0, allow_inf_nan=False)]
FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]
Probability = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, validate_default=True, hide_input_in_errors=True)


class _Transport(_Strict):
    model: Nonempty
    api_key_env: Nonempty
    timeout: PositiveFloat = 60.0
    max_retries: NonnegativeInt = 3
    max_concurrency: PositiveInt = 16
    requests_per_minute: PositiveFloat | None = None

    @field_validator("api_key_env")
    @classmethod
    def env_name(cls, value: str) -> str:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
            raise ValueError("api_key_env must name an environment variable")
        return value

    @field_validator("base_url", check_fields=False)
    @classmethod
    def safe_url(cls, value: str | None) -> str | None:
        if value is not None:
            try:
                parts = urlsplit(value)
                if (parts.scheme not in {"http", "https"} or not parts.hostname
                        or parts.username is not None or parts.password is not None
                        or parts.query or parts.fragment):
                    raise ValueError
                parts.port
            except ValueError:
                raise ValueError("base_url requires HTTP(S), no credentials, query or fragment") from None
        return value


class _OpenAI(_Transport):
    backend: Literal["openai"] = "openai"
    base_url: Nonempty | None = None
    api_key_env: Nonempty = "OPENAI_API_KEY"
    temperature: FiniteFloat | None = 0.0
    omit_temperature: bool = Field(default=False, exclude=True)
    max_tokens: PositiveInt | None = None
    reasoning_effort: Nonempty | None = None

    @model_validator(mode="after")
    def sampling_policy(self):
        if self.omit_temperature:
            self.temperature = None
        return self


class _Jev(_Transport):
    backend: Literal["jev"] = "jev"
    base_url: Nonempty = "https://api.typesafe.ai"
    api_key_env: Nonempty = "TYPESAFE_API_KEY"


class _Kev(_Strict):
    backend: Literal["kev"] = "kev"
    model: Literal["kev-9b-v1.0", "kev-4b-v1.0"] = "kev-9b-v1.0"
    base_url: Nonempty = "http://127.0.0.1:8019"
    timeout: PositiveFloat = 60.0
    max_retries: Literal[0] = 0
    max_concurrency: PositiveInt = 3
    requests_per_minute: PositiveFloat | None = None

    @model_validator(mode="after")
    def local_endpoint(self):
        from coatlearn.models.kev import KevConfig
        KevConfig(**self.model_dump(exclude={"backend", "task_context"}))
        return self


class _KevDedup(_Kev):
    task_context: Nonempty | None = None


ModelSpec = Annotated[Union[_OpenAI, _Jev, _Kev], Field(discriminator="backend")]


class TaskSpec(_Strict):
    method: Literal["mb", "pag", "adj"]
    data: Path
    text_col: Nonempty
    target: Nonempty | None = None
    known_variables: list[Nonempty] = Field(default_factory=list)
    treatment: Nonempty | None = None
    outcome: Nonempty | None = None
    treatment_mapping: Literal["minus_one_one", "yes_missing"] | None = None
    outcome_mapping: Literal["yes_no"] | None = None

    @model_validator(mode="after")
    def columns(self) -> TaskSpec:
        if self.method == "mb":
            if self.target is None:
                raise ValueError("MB requires target")
            columns = [self.target, self.text_col]
        elif self.method == "pag":
            if not self.known_variables:
                raise ValueError("PAG requires known_variables")
            columns = [*self.known_variables, self.text_col]
        else:
            if self.treatment is None or self.outcome is None:
                raise ValueError("ADJ requires treatment and outcome")
            columns = [self.treatment, self.outcome, self.text_col]
        if len(columns) != len(set(columns)):
            raise ValueError("Task columns must be distinct")
        if self.method != "adj" and (self.treatment_mapping or self.outcome_mapping):
            raise ValueError("Mappings require ADJ")
        return self


class _Batching(_Strict):
    enabled: bool = True
    max_factors_per_request: PositiveInt = 8
    shuffle_questions: bool = False


class _SupportGate(_Strict):
    enabled: bool = True
    min_mention_rate_per_arm: Probability = 0.05
    min_count_per_arm: NonnegativeInt = 0


class _PredictionGate(_Strict):
    enabled: bool = True
    accuracy_threshold: Annotated[float, Field(gt=0, le=1, allow_inf_nan=False)] = 0.9


class _Dedup(_Jev):
    task_context: Nonempty | None = None


class _Options(_Strict):
    vif_threshold: PositiveFloat | None = None
    fci_depth: Annotated[int, Field(ge=-1)] | None = None
    prompt_sample_size: PositiveInt | None = None
    proposal_top_k: PositiveInt | None = None
    score_threshold: FiniteFloat | None = None
    stability_B: PositiveInt | None = None
    stability_pi: Annotated[float, Field(gt=0, lt=1, allow_inf_nan=False)] | None = None
    use_lingam: bool | None = None
    factor_template: Literal["simple", "general"] | None = None
    factor_template_repair: bool | None = None
    exp_uid: Nonempty | None = None
    save_factor_audit: bool | None = None
    factor_judge: Union[_Jev, _Kev] | None = None
    factor_dedup: Union[_Dedup, _KevDedup] | None = None
    annotation_batching: _Batching | None = None
    support_gate: _SupportGate | None = None
    factor_prediction_gate: _PredictionGate | None = None
    observed_factor_definitions: dict[Nonempty, Nonempty] | None = None
    sample_minimum: Annotated[float, Field(gt=0, le=1, allow_inf_nan=False)] | None = None
    cell_coverage_min: Annotated[float, Field(gt=0, le=1, allow_inf_nan=False)] | None = None
    adjustment_selection: Literal["max_cardinality", "size_then_propensity_range"] | None = None
    sparse_cell_warning_below: NonnegativeInt | None = None
    legacy_adjustment_estimates: bool | None = None
    break_cycles: bool | None = None
    assume_treatment_causes_outcome: bool | None = None
    assume_factors_precede_treatment: bool | None = None
    amenability_exempt: bool | None = None
    amenability_exempt_reason: Nonempty | None = None
    # Historical overrides are typed here, but forbidden in new TOML below.
    num_workers: PositiveInt | None = None
    alpha: Annotated[float, Field(gt=0, lt=1, allow_inf_nan=False)] | None = None
    test_method: Nonempty | None = None
    save_llm_conversation: bool | None = None
    num_round: PositiveInt | None = None


class _Analysis(_Strict):
    rounds: PositiveInt = 1
    concurrency: PositiveInt = 16
    alpha: Annotated[float, Field(gt=0, lt=1, allow_inf_nan=False)] = 0.01
    test_method: Nonempty = "kci"
    seed: Annotated[int, Field(ge=0, le=2**32 - 1)] = 0
    output_dir: Nonempty | None = None
    save_conversations: bool = False
    options: _Options = Field(default_factory=_Options)


class _Roles(_Strict):
    main: Nonempty
    annotation: Nonempty
    judge: Nonempty | None = None
    dedup: Nonempty | None = None


class _RoleOptions(_Strict):
    timeout: PositiveFloat | None = None
    max_retries: NonnegativeInt | None = None
    max_concurrency: PositiveInt | None = None
    requests_per_minute: PositiveFloat | None = None
    temperature: FiniteFloat | None = None
    max_tokens: PositiveInt | None = None
    reasoning_effort: Nonempty | None = None


class _DedupOptions(_RoleOptions):
    task_context: Nonempty | None = None


class _RoleOverrides(_Strict):
    annotation: _RoleOptions | None = None
    judge: _RoleOptions | None = None
    dedup: _DedupOptions | None = None


class _Document(_Strict):
    schema_version: Literal[1]
    task: TaskSpec
    models: dict[Nonempty, ModelSpec]
    roles: _Roles
    role_options: _RoleOverrides = Field(default_factory=_RoleOverrides)
    analysis: _Analysis = Field(default_factory=_Analysis)

    @field_validator("schema_version", mode="before")
    @classmethod
    def exact_version(cls, value: Any) -> Any:
        if type(value) is not int:
            raise ValueError("schema_version must be integer 1")
        return value


_DUPLICATES = {"alpha", "test_method", "num_workers", "save_llm_conversation", "num_round",
               "rounds", "concurrency", "save_conversations", "seed", "output_dir"}
_SECRET_KEYS = {"api_key", "apikey", "api-key", "password", "access_token", "api_token",
                "authorization", "client_secret", "secret_key", "token"}
# Recognize the standalone source checkout, not a wheel's site-packages depth.
_REPO_ROOT = next((parent for parent in Path(__file__).resolve().parents
                   if (parent / "pyproject.toml").is_file()
                   and (parent / "coatlearn" / "configuration.py").is_file()), None)


def _check_payload(value: Any) -> None:
    """Reject credential material and non-finite JSON extensions recursively."""
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str) or key.lower() in _SECRET_KEYS:
                raise ConfigError("Credential values are forbidden; use api_key_env names")
            _check_payload(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _check_payload(child)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ConfigError("Configuration numbers must be finite")
    elif isinstance(value, str) and value.strip().startswith(("http://", "https://")):
        try:
            parts = urlsplit(value.strip())
            if parts.username is not None or parts.password is not None or parts.query or parts.fragment:
                raise ValueError
        except ValueError:
            raise ConfigError("URLs cannot contain credentials, query strings or fragments") from None


_SAFE_LOCATION_FIELDS = frozenset().union(*(
    schema.model_fields.keys() for schema in
    (_Transport, _OpenAI, _Jev, _Kev, _KevDedup, TaskSpec, _Batching, _SupportGate, _Dedup,
     _Options, _Analysis, _Roles, _RoleOptions, _DedupOptions, _RoleOverrides, _Document)
)) | {"openai", "jev", "kev"}


def _safe_location(location: tuple[Any, ...]) -> str:
    """Only schema vocabulary is printable; named model keys are always hidden."""
    parts = []
    for index, part in enumerate(location):
        if index and location[index - 1] == "models":
            parts.append("<model>")
        elif isinstance(part, int):
            parts.append("<item>")
        else:
            parts.append(part if part in _SAFE_LOCATION_FIELDS else "<unknown>")
    return ".".join(parts) or "configuration"


def _validate(cls: type[BaseModel], value: Any) -> Any:
    try:
        return cls.model_validate(value)
    except ValidationError as exc:
        # Never stringify ValidationError or its message/context: values and
        # arbitrary mapping keys can contain secrets. Render only schema words
        # and Pydantic's fixed error codes, with a bounded number of locations.
        issues = sorted({f"{_safe_location(error['loc'])}: {error['type']}"
                         for error in exc.errors(include_input=False, include_context=False)})
        raise ConfigError("Invalid configuration (" + "; ".join(issues[:8]) + ")") from None


def _model(value: Any, *, legacy: bool = False, dedup: bool = False) -> _OpenAI | _Jev:
    if not isinstance(value, dict):
        raise ConfigError("Model configuration must be a table")
    value = deepcopy(value)
    backend = value.get("backend", "openai" if legacy else None)
    if backend not in {"openai", "jev", "kev"}:
        raise ConfigError("Unsupported backend; supported backends are openai, jev and kev")
    value["backend"] = backend
    return _validate((_KevDedup if dedup else _Kev) if backend == "kev" else _Dedup if dedup and backend == "jev" else _OpenAI if backend == "openai" else _Jev, value)


def _path(value: Any, base: Path) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ConfigError("Task data must be a non-empty path")
    path = Path(value)
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def _cross_validate(task: TaskSpec, options: dict[str, Any], effective: dict[str, dict]) -> None:
    for role in ("judge", "dedup"):
        if role in effective and effective[role]["backend"] not in {"jev", "kev"}:
            raise ConfigError("Judge and dedup roles require a typed decision backend")
    if effective["main"]["backend"] != "openai":
        raise ConfigError("Main role requires the openai chat backend")
    batching = options.get("annotation_batching") or {}
    if batching.get("enabled", False):
        if effective["annotation"]["backend"] not in {"jev", "kev"}:
            raise ConfigError("Enabled annotation batching requires typed decision annotation")
    prior_keys = ("assume_treatment_causes_outcome", "assume_factors_precede_treatment", "amenability_exempt")
    if task.method != "adj" and any(options.get(key) for key in prior_keys):
        raise ConfigError("Direction and amenability priors require an ADJ task")
    if options.get("assume_factors_precede_treatment") and not options.get("assume_treatment_causes_outcome"):
        raise ConfigError("Factor temporal prior requires explicit treatment-to-outcome prior")
    if options.get("amenability_exempt") and not options.get("amenability_exempt_reason"):
        raise ConfigError("Amenability exemption requires an explicit reason")


@dataclass(frozen=True)
class TaskPlan:
    """Effective configuration only; each conversion returns independent state."""

    task: TaskSpec
    source: Path
    placeholder: bool
    warnings: tuple[str, ...]
    _models: dict[str, dict]
    _roles: dict[str, str]
    _effective_roles: dict[str, dict]
    _run: dict[str, Any]

    def run_config(self) -> RunConfig:
        from .api import RunConfig
        return RunConfig(**deepcopy(self._run))

    def model_configs(self) -> tuple[ModelConfig, ModelConfig | JevConfig]:
        from coatlearn.models.openai import ModelConfig
        from coatlearn.models.jev import JevConfig
        configs = []
        for role in ("main", "annotation"):
            fields = deepcopy(self._effective_roles[role])
            backend = fields.pop("backend")
            if backend == "kev":
                from coatlearn.models.kev import KevConfig
                configs.append(KevConfig(**fields))
            else:
                configs.append((JevConfig if backend == "jev" else ModelConfig)(**fields))
        return tuple(configs)

    def required_env_names(self) -> list[str]:
        return sorted({model["api_key_env"] for model in self._effective_roles.values() if model["backend"] != "kev"})

    def snapshot(self) -> dict[str, Any]:
        return deepcopy({"schema_version": 1, "source": str(self.source),
                         "task": self.task.model_dump(mode="json"), "models": self._models,
                         "roles": self._roles, "effective_roles": self._effective_roles,
                         "run": self._run, "placeholder": self.placeholder,
                         "warnings": list(self.warnings)})


def _finish(source, task, models, roles, effective, analysis, options, warnings):
    _cross_validate(task, options, effective)
    run = analysis.model_dump(exclude={"options"})
    run["options"] = options
    if run["output_dir"] is not None:
        run["output_dir"] = str(_path(run["output_dir"], source.parent))
    placeholder = any("PLEASE-SET" in model["model"].upper()
                      or ".example.invalid" in (model.get("base_url") or "").lower()
                      for model in effective.values())
    return TaskPlan(task, source, placeholder, tuple(warnings), models, roles, effective, run)


def _new(document, source, overrides):
    raw_task = document.get("task")
    if not isinstance(raw_task, dict):
        raise ConfigError("A task table is required")
    document = deepcopy(document)
    document["task"].update(overrides)
    document["task"]["data"] = _path(document["task"].get("data"), source.parent)
    # Reject override aliases before strict schema checking, not after precedence.
    raw_options = document.get("analysis", {}).get("options", {})
    if isinstance(raw_options, dict) and _DUPLICATES.intersection(raw_options):
        raise ConfigError("analysis.options cannot override top-level analysis fields")
    for value in document.get("models", {}).values():
        _model(value)
    schema = _validate(_Document, document)
    irrelevant = {
        "mb": ("known_variables", "treatment", "outcome"),
        "pag": ("target", "treatment", "outcome"),
        "adj": ("target", "known_variables"),
    }[schema.task.method]
    if any(getattr(schema.task, key) for key in irrelevant):
        raise ConfigError("Task column fields do not match task.method")
    models = {name: model.model_dump() for name, model in schema.models.items()}
    roles = schema.roles.model_dump(exclude_none=True)
    effective = {}
    role_options = schema.role_options.model_dump(exclude_unset=True)
    options = {}
    for key in schema.analysis.options.model_fields_set:
        value = getattr(schema.analysis.options, key)
        options[key] = value.model_dump() if isinstance(value, BaseModel) else value
        if key == "factor_dedup" and isinstance(options[key], dict) and options[key].get("task_context") is None:
            options[key].pop("task_context", None)
    for role in role_options:
        if role not in roles:
            raise ConfigError("Role options require an explicitly enabled role")
    for role, name in roles.items():
        if name not in models:
            raise ConfigError("Role references an undefined model")
        value = deepcopy(models[name])
        extra = deepcopy(role_options.get(role) or {})
        context = extra.pop("task_context", None)
        value.update(extra)
        value = _model(value).model_dump()
        if context is not None:
            value["task_context"] = context
        effective[role] = value
        if role in {"judge", "dedup"}:
            key = "factor_" + role
            if key in options:
                raise ConfigError("Role references conflict with manual factor role options")
            options[key] = deepcopy(value)
    for role in ("judge", "dedup"):
        if role not in effective and options.get("factor_" + role) is not None:
            effective[role] = deepcopy(options["factor_" + role])
    return _finish(source, schema.task, models, roles, effective, schema.analysis, options, [])


def _legacy(document, source, overrides):
    warnings = []
    models = {role: _model(document.get(role + "_model"), legacy=True).model_dump()
              for role in ("main", "annotation")}
    roles = {role: role for role in models}
    effective = deepcopy(models)
    data = document.get("data", {})
    if not isinstance(data, dict):
        raise ConfigError("Legacy data metadata must be an object")
    task = {key: data[key] for key in TaskSpec.model_fields if key in data}
    if "data" not in task:
        recorded = data.get("path", data.get("source"))
        if recorded is not None:
            task["data"] = _path(recorded, _REPO_ROOT or source.parent)
    task.update(overrides)
    if "method" not in task:
        if task.get("treatment") and task.get("outcome"):
            task["method"] = "adj"
        elif task.get("known_variables"):
            task["method"] = "pag"
        elif task.get("target"):
            task["method"] = "mb"
    if "data" not in task:
        raise ConfigError("Legacy configuration requires task data metadata or task_overrides")
    task["data"] = _path(task["data"], source.parent)
    task = _validate(TaskSpec, task)
    raw_run = deepcopy(document.get("run", {}))
    if not isinstance(raw_run, dict):
        raise ConfigError("Legacy run must be an object")
    options = raw_run.pop("options", {})
    if not isinstance(options, dict):
        raise ConfigError("Legacy run.options must be an object")
    known = {key: value for key, value in options.items() if key in _Options.model_fields}
    _validate(_Options, known)
    if set(options) - _Options.model_fields.keys():
        warnings.append("Unknown legacy core options retained and forwarded")
    if _DUPLICATES.intersection(options):
        warnings.append("Legacy core option overrides retained with historical API precedence")
    for role in ("judge", "dedup"):
        if options.get("factor_" + role) is not None:
            fields = _model(options["factor_" + role], legacy=True, dedup=role == "dedup").model_dump()
            effective[role] = fields
    analysis = _validate(_Analysis, raw_run)
    if analysis.output_dir is not None:
        analysis.output_dir = str(_path(analysis.output_dir, _REPO_ROOT or source.parent))
    return _finish(source, task, models, roles, effective, analysis, options, warnings)


def load_task(path: str | Path, *, task_overrides: Mapping[str, Any] | None = None,
              decision_model: str | None = None) -> TaskPlan:
    """Read TOML v1 or a legacy JSON config, without datasets or credentials.

    Task overrides use the task field names; relative override paths are resolved
    against the config parent. No environment values, dotenv files, SDKs, clients
    or core algorithms are accessed by this function.
    """
    try:
        source = Path(path).resolve()
        if source.suffix.lower() == ".toml":
            with source.open("rb") as handle:
                document = tomllib.load(handle)
            legacy = False
        elif source.suffix.lower() == ".json":
            with source.open(encoding="utf-8") as handle:
                document = json.load(handle)
            legacy = True
        else:
            raise ConfigError("Configuration file must be TOML or JSON")
    except (OSError, ValueError, TypeError) as exc:
        if isinstance(exc, ConfigError):
            raise
        raise ConfigError("Configuration file cannot be read or parsed") from None
    if not isinstance(document, dict):
        raise ConfigError("Configuration must contain an object")
    if task_overrides is not None and not isinstance(task_overrides, Mapping):
        raise ConfigError("task_overrides must be a mapping")
    overrides = dict(task_overrides or {})
    _check_payload(document)
    _check_payload(overrides)
    if decision_model is not None:
        if legacy:
            raise ConfigError("Model selection requires a TOML configuration")
        if decision_model not in document.get("models", {}):
            raise ConfigError("Selected decision model is not defined in models")
        document = deepcopy(document)
        roles = document.get("roles", {})
        for role in ("annotation", "judge", "dedup"):
            if role in roles:
                roles[role] = decision_model
    try:
        return (_legacy if legacy else _new)(document, source, overrides)
    except ConfigError:
        raise
    except (ValueError, TypeError, AttributeError, KeyError):
        raise ConfigError("Invalid configuration structure or fields") from None
