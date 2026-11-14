"""Offline single-factor prediction filter for explicitly binary T/Y references.

A high raw accuracy is an exclusion policy, not proof of semantic identity.
Measurements and definitions stay intact; callers filter candidate names only.
"""
from __future__ import annotations

import hashlib

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from coatlearn.annotation.definitions import factor_definition_to_spec


def validate_prediction_options(options):
    if options is None:
        return
    if not isinstance(options, dict) or set(options) - {"enabled", "accuracy_threshold"}:
        raise ValueError("factor_prediction_gate accepts enabled and accuracy_threshold")
    if not isinstance(options.get("enabled", True), bool):
        raise ValueError("factor_prediction_gate.enabled must be a bool")
    threshold = options.get("accuracy_threshold", 0.9)
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not 0 < threshold <= 1:
        raise ValueError("factor_prediction_gate.accuracy_threshold must be in (0, 1]")


def validate_observed_definitions(definitions):
    if not isinstance(definitions, dict) or any(
        not isinstance(name, str) or not name.strip()
        or not isinstance(text, str) or not text.strip()
        for name, text in definitions.items()
    ):
        raise ValueError("observed_factor_definitions must map names to nonempty definitions")
    for name, text in definitions.items():
        factor_definition_to_spec(name, text)


def validate_prediction_references(references):
    """Validate and copy binary observed vectors without creating filter state."""
    copied = {
        name: np.asarray(values, dtype=float).copy()
        for name, values in references.items()
    }
    for name, values in copied.items():
        if not np.isfinite(values).all() or set(np.unique(values)) != {0.0, 1.0}:
            raise ValueError(f"factor_prediction_gate requires binary 0/1 reference: {name}")
    return copied


class FactorPredictionFilter:
    """One fixed policy shared by ADJ and all of its MB/PAG children."""

    def __init__(self, references, options):
        validate_prediction_options(options)
        self.threshold = float(options.get("accuracy_threshold", 0.9))
        self.references = validate_prediction_references(references)
        self.records = {}
        self._cache = {}

    def policy(self):
        return {"enabled": True, "accuracy_threshold": self.threshold,
                "comparison": ">=", "prediction_direction": "single F predicts T or Y",
                "folds": 5, "seed": 42, "estimator": "LogisticRegression(random_state=42, max_iter=1000)",
                "features": "choice one-hot; score standardized numeric; no class weighting",
                "scoring": "out-of-fold raw accuracy; probability threshold 0.5",
                "missing": "complete pairs only; no zero fill or missingness predictor",
                "references": list(self.references),
                "caveat": "Exclusion policy, not semantic identity or pre-treatment certification; raw accuracy may reflect majority prevalence."}

    def filter(self, names, annotation, definitions):
        kept = []
        for name in names:
            if name in self.references:
                kept.append(name)
                continue
            if name.startswith("@ "):
                continue
            values = np.asarray(annotation[name], dtype=float)
            definition = definitions.get(name)
            fingerprint = hashlib.sha256(values.tobytes() + str(definition).encode()).hexdigest()
            key = (name, fingerprint)
            if key not in self._cache:
                categorical = factor_definition_to_spec(name, definition).jev_type == "choice"
                self._cache[key] = [
                    self._score_reference(name, values, target, reference, categorical, fingerprint)
                    for target, reference in self.references.items()
                ]
            records = self._cache[key]
            for record in records:
                self.records[(name, record["target"])] = record
            if not any(record["excluded"] for record in records):
                kept.append(name)
        return kept

    def _score_reference(self, name, values, target, reference, categorical, fingerprint):
        """Score one complete-pair vector; cache and candidate selection stay outside."""
        if values.shape != reference.shape:
            raise ValueError("Factor and T/Y measurements must have matching shapes")
        valid = np.isfinite(values) & np.isfinite(reference)
        x, y = values[valid, None], reference[valid].astype(int)
        counts = np.bincount(y, minlength=2)
        record = {
            "factor": name, "target": target, "n": len(y),
            "n0": int(counts[0]), "n1": int(counts[1]),
            "excluded": False, "measurement_sha256": fingerprint,
        }
        if counts.min() < 5:
            record["status"] = "insufficient_class_support"
            return record

        transform = OneHotEncoder(handle_unknown="ignore") if categorical else StandardScaler()
        model = make_pipeline(transform, LogisticRegression(random_state=42, max_iter=1000))
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        probability = cross_val_predict(model, x, y, cv=cv, method="predict_proba")[:, 1]
        prediction = (probability >= 0.5).astype(int)
        accuracy = float(accuracy_score(y, prediction))
        record.update(
            status="scored", type="choice" if categorical else "score",
            accuracy=accuracy, majority_baseline=float(counts.max() / len(y)),
            balanced_accuracy=float(balanced_accuracy_score(y, prediction)),
            auc=float(roc_auc_score(y, probability)), excluded=accuracy >= self.threshold,
        )
        return record

    def report(self):
        return {"policy": self.policy(),
                "records": [dict(self.records[key]) for key in sorted(self.records)],
                "excluded_factors": sorted({record["factor"] for record in self.records.values()
                                             if record["excluded"]})}
