"""Batch measurement of fixed factor definitions, with no discovery or screening."""
from __future__ import annotations

from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import pandas as pd

from .definitions import factor_definition_to_spec
from .types import MeasurementSpec


@dataclass
class AnnotationResult:
    """Aligned values, per-measurement metadata and the finalized definitions."""

    values: pd.DataFrame
    metadata: pd.DataFrame
    specs: dict[str, MeasurementSpec]


def resolve_annotator(model):
    """Resolve a preset or model configuration after local data validation."""
    from coatlearn.models.presets import ModelPreset
    from coatlearn.models.jev import JevConfig
    from coatlearn.models.systemone import SystemOneClient
    from .measurement import JevAnnotator, decision_annotator

    if isinstance(model, ModelPreset):
        model = model._resolve("annotation")
    if isinstance(model, JevConfig):
        return decision_annotator(model)
    if callable(getattr(model, "annotate_record", None)):
        return model
    if isinstance(model, SystemOneClient):
        annotator = JevAnnotator(model.config)
        annotator._system_one = model._system_one
        annotator.stats = model.stats
        return annotator
    raise TypeError("model must be a Jev/Kev configuration, preset, System One client or factor annotator")


def annotate(texts, definitions: Mapping[str, str | MeasurementSpec], *, model,
             concurrency: int = 16, batch_size: int = 8) -> AnnotationResult:
    """Measure fixed definitions in row order, without judging or deduplicating them.

    Each request contains one text and up to batch_size factors. A pandas Series
    retains its index, including repeated labels. Transport failures propagate;
    a mention gate can independently record a missing measurement as NaN.
    """
    if not isinstance(concurrency, int) or isinstance(concurrency, bool) or concurrency < 1:
        raise ValueError("concurrency must be a positive integer")
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or not 1 <= batch_size <= 8:
        raise ValueError("batch_size must be between 1 and 8")
    if isinstance(texts, str):
        raise TypeError("texts must be a sequence, not a single string")
    series = texts.copy() if isinstance(texts, pd.Series) else pd.Series(list(texts), dtype=object)
    if series.empty or not series.map(lambda text: isinstance(text, str) and bool(text.strip())).all():
        raise ValueError("texts must contain nonempty strings")
    if not isinstance(definitions, Mapping) or not definitions:
        raise ValueError("definitions must map factor names to definitions")
    specs = {}
    for name, definition in definitions.items():
        spec = definition if isinstance(definition, MeasurementSpec) else factor_definition_to_spec(name, definition)
        if spec.factor_name != name:
            raise ValueError("Definition names must match their mapping keys")
        specs[name] = spec.validate()
    backend = resolve_annotator(model)
    prepared = {name: backend.prepare_factor(name, spec) for name, spec in specs.items()}
    names = list(prepared)

    def measure(item):
        row, text = item
        values, metadata = {}, []
        for start in range(0, len(names), batch_size):
            chunk = {name: prepared[name] for name in names[start:start + batch_size]}
            answers = backend.annotate_record(text, chunk, sample_id=row)
            for name in chunk:
                answer = answers[name]
                values[name] = answer.value
                snapshot = getattr(backend, "annotation_metadata", None)
                record = snapshot(row, name) if callable(snapshot) else None
                metadata.append({**(record if record is not None else answer.to_metadata()), "row": row})
        return values, metadata

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        measured = list(pool.map(measure, enumerate(series.tolist())))
    values = pd.DataFrame([row for row, _ in measured], index=series.index, columns=names)
    metadata = pd.DataFrame([record for _, records in measured for record in records])
    return AnnotationResult(values, metadata, dict(prepared))
