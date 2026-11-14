"""Ready-to-use, immutable model presets with execution-time dotenv support.

Importing presets performs no file/environment access and creates no clients.
Pass a preset directly to run_mb/run_pag/run_adj. Each resolution makes a fresh
backend: credentials, measurement caches and counters are never global models.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import os
from pathlib import Path
import threading
from typing import Any, Literal

from coatlearn.models.openai import ModelConfig
from coatlearn.models.jev import JevConfig
from coatlearn.models.kev import KevConfig

__all__ = ["dpsk41f", "jev", "kev9b", "kev4b", "ModelPreset", "PresetConfigurationError"]
_ENV_LOCK = threading.Lock()
_UNSET = object()


class PresetConfigurationError(ValueError):
    """A local preset setup error, with instructions but no credential values."""


@dataclass(frozen=True)
class ModelPreset:
    """A reusable model description, not a shared live model instance.

    Call a preset to override existing configuration fields or env_file, e.g.
    dpsk41f(max_concurrency=3). Only the required key is copied from the selected
    dotenv into the process environment, without overwriting an existing value.
    Relative dotenv paths resolve at execution; None disables dotenv entirely.
    """

    name: str
    backend: Literal["openai", "jev", "kev"]
    config: ModelConfig | JevConfig
    env_file: str | Path | None = ".env"

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise PresetConfigurationError("Preset name must be nonempty")
        expected = {"openai": ModelConfig, "jev": JevConfig, "kev": KevConfig}.get(self.backend)
        if expected is None or not isinstance(self.config, expected):
            raise PresetConfigurationError("Preset backend and configuration type must agree: openai/ModelConfig or jev/JevConfig")
        if self.env_file is not None and not isinstance(self.env_file, (str, Path)):
            raise PresetConfigurationError("env_file must be a path or None")

    def __call__(self, *, env_file: str | Path | None | object = _UNSET, **options: Any) -> ModelPreset:
        """Return an independently customized preset; never load credentials."""
        # Omission retains a customized file; explicit None disables dotenv.
        # Config rejects literal api_key fields.
        selected = self.env_file if env_file is _UNSET else env_file
        try:
            config = replace(self.config, **options)
        except (TypeError, ValueError):
            raise PresetConfigurationError("Invalid preset options; use ModelConfig/JevConfig field names, never literal credentials") from None
        if selected is not None and not isinstance(selected, (str, Path)):
            raise PresetConfigurationError("env_file must be a path or None")
        return replace(self, config=config, env_file=selected)

    def _resolve(self, role: str) -> Any:
        """Resolve after input validation, before a client or request exists."""
        if self.backend in {"jev", "kev"} and role != "annotation":
            raise PresetConfigurationError("This is an annotation preset; use dpsk41f or another chat model for main_model")
        if self.backend == "kev":
            from coatlearn.annotation.measurement import KevAnnotator
            return KevAnnotator(self.config)
        self._require_key()
        if self.backend == "openai":
            return self.config
        from coatlearn.annotation.measurement import JevAnnotator
        return JevAnnotator(self.config)

    def _require_key(self) -> None:
        name = self.config.api_key_env
        with _ENV_LOCK:
            if name not in os.environ and self.env_file is not None:
                from dotenv import dotenv_values
                path = Path(self.env_file)
                try:
                    # Explicit path: no find_dotenv/parent search, no variable
                    # interpolation, no unrelated dotenv settings imported.
                    if path.is_file():
                        value = dotenv_values(path, interpolate=False).get(name)
                        if isinstance(value, str) and value.strip():
                            os.environ.setdefault(name, value)
                except (OSError, UnicodeError, ValueError):
                    raise PresetConfigurationError(
                        f"Cannot load the selected dotenv for {self.name}; check its path, permissions and encoding. Contents were not printed."
                    ) from None
            value = os.environ.get(name)
            if not isinstance(value, str) or not value.strip():
                location = "the selected dotenv file" if self.env_file is not None else "your process environment"
                empty_hint = (f"The existing process variable {name} is empty; set a nonempty value or unset it to allow dotenv loading. "
                              if name in os.environ else "")
                raise PresetConfigurationError(
                    f"{self.name} requires {name}. {empty_hint}Set {name} in {location} or export it before running. "
                    f"For the default preset, create .env in the current working directory with {name}=<your-key>. "
                    "No client was created and no request was made; never paste your key into logs or source code."
                )


# deepseek-flash is the provider's rolling V4.1 Flash alias, not pinned weights.
dpsk41f = ModelPreset(
    name="dpsk41f",
    backend="openai",
    config=ModelConfig(model="deepseek-flash", base_url="https://api.deepseek.com",
                       api_key_env="DEEPSEEK_API_KEY", temperature=None),
)
jev = ModelPreset(
    name="jev",
    backend="jev",
    config=JevConfig(model="jev-1.14.0", api_key_env="TYPESAFE_API_KEY"),
)
kev9b = ModelPreset(name="kev9b", backend="kev", config=KevConfig(), env_file=None)
kev4b = ModelPreset(name="kev4b", backend="kev",
                    config=KevConfig(model="kev-4b-v1.0", base_url="http://127.0.0.1:8014"), env_file=None)
