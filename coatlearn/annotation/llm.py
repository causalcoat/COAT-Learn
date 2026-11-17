"""Single-factor LLM measurement for the public annotation API."""
from __future__ import annotations

import math

from .text import get_value_from_responce
from .types import FactorAnnotation, MeasurementSpec


class _LLMAnnotation(FactorAnnotation):
    def to_metadata(self):
        metadata = super().to_metadata()
        metadata["jev_type"] = None
        metadata["value_mapping"] = "Direct numeric answer from the LLM; no System One probability mapping."
        return metadata


class LLMAnnotator:
    """Use the discovery prompt and numeric parser without System One metadata."""

    def __init__(self, model):
        if not callable(getattr(model, "chat", None)):
            raise TypeError("model must expose chat")
        self.model = model

    def prepare_factor(self, name, spec):
        if not isinstance(spec, MeasurementSpec) or spec.factor_name != name:
            raise ValueError("factor name and MeasurementSpec must agree")
        return spec.validate()

    @staticmethod
    def _definition(spec):
        lines = [spec.factor_name, spec.instructions]
        if spec.jev_type == "score":
            lines.extend(f"{value}: {criterion}" for value, criterion in zip(spec.value_map, spec.criteria))
        elif spec.jev_type == "choice":
            lines.extend(f"{spec.choice_values[option]} ({option}): {description or option}"
                         for option, description in spec.criteria)
        else:
            values = spec.value_map or (0, 1)
            criteria = spec.criteria or ("yes", "no")
            lines.extend((f"{values[1]}: {criteria[0]}", f"{values[0]}: {criteria[1]}"))
        if spec.gate is not None:
            lines.append(f"Mentioned when: {spec.gate.criteria[0] if spec.gate.criteria else spec.gate.instructions}")
            lines.append("If the attribute is not mentioned, output The value is: NaN; do not guess or use zero.")
        return "\n".join(lines)

    def annotate_record(self, text, specs, *, sample_id=None):
        # Reuse the exact prompt used by COATMB; do not change discovery's path.
        from coatlearn.discovery.mb import COATMB

        results = {}
        for name, spec in specs.items():
            spec = self.prepare_factor(name, spec)
            prompt = COATMB.prompt_factor_annotation(None, self._definition(spec), text)
            response = self.model.chat(prompt, return_text=True)
            value = get_value_from_responce(response)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                value = float("nan")
            results[name] = _LLMAnnotation(
                factor_name=name, value=float(value), spec=spec, raw={},
                sample_id=sample_id, backend_model=getattr(getattr(self.model, "config", None), "model", None),
                value_policy="llm_numeric" if math.isfinite(value) else "llm_missing",
            )
        return results
