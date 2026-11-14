"""Deterministic translation of COAT's extracted ternary factor definitions."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from coatlearn.annotation.types import MeasurementSpec, UnsupportedMeasurement

__all__ = ["factor_definition_to_spec", "general_definition_to_spec"]

_GENERAL_TYPE = re.compile(r"^[-+*•]?\s*type\s*:\s*(score|choice)\s*$", re.I)
_VALUES = re.compile(r"^[-+*•]?\s*values\s*:\s*(.+)$", re.I)
_MENTIONED = re.compile(r"^[-+*•]?\s*mentioned\s+when\s*:\s*(.+)$", re.I)
_LEVEL = re.compile(r"^[-+*•]?\s*(\d+)\s*:\s*(.+)$")
_OPTION = re.compile(r"^[-+*•]?\s*([^:]+?)\s*:\s*(.+)$")


def _unwrap(text: str) -> str:
    text = text.strip()
    while True:
        for marker in ("**", "__", "`", "*", "_"):
            if (
                text.startswith(marker)
                and text.endswith(marker)
                and len(text) >= 2 * len(marker)
                and text.find(marker, len(marker)) == len(text) - len(marker)
            ):
                text = text[len(marker) : -len(marker)].strip()
                break
        else:
            return text


def _unbullet(text: str) -> str:
    return re.sub(r"^(?:[-+*•]\s+|\d+[.)]\s+)", "", text.strip()).strip()


def _name(text: str) -> str:
    text = re.sub(r"^#{1,6}\s+", "", _unbullet(_unwrap(text)))
    return " ".join(_unwrap(text.rstrip(":")).rstrip(":").split())


def factor_definition_to_spec(factor_name: str, factor_definition: Any) -> MeasurementSpec:
    """Parse exactly one named {-1, 0, 1} factor without inference or requests.

    Accept the extracted four-line text or its exact ``name``/``description``
    record. Markdown decoration and whitespace are presentation only; criterion
    wording and label associations are retained. Unsupported input raises
    :class:`UnsupportedMeasurement` before any backend is involved.
    """
    if not isinstance(factor_name, str) or not factor_name.strip():
        raise UnsupportedMeasurement(factor_name, "factor_name must be a nonempty string")
    name = _name(factor_name)
    if not name:
        raise UnsupportedMeasurement(factor_name, "factor_name must contain a name")
    if isinstance(factor_definition, Mapping):
        if set(factor_definition) != {"name", "description"}:
            raise UnsupportedMeasurement(factor_name, "expected exactly name and description")
        record_name = factor_definition["name"]
        if not isinstance(record_name, str) or _name(record_name).casefold() != name.casefold():
            raise UnsupportedMeasurement(factor_name, "factor record name does not match")
        factor_definition = factor_definition["description"]
    if not isinstance(factor_definition, str):
        raise UnsupportedMeasurement(factor_name, "factor definition must be text or a factor record")
    if any(_GENERAL_TYPE.match(line.strip()) for line in factor_definition.splitlines()):
        return general_definition_to_spec(factor_name, factor_definition)
    lines = [line.strip() for line in factor_definition.splitlines() if line.strip()]
    if len(lines) >= 2 and lines[0] in ("```", "```text", "```markdown") and lines[-1] == "```":
        lines = lines[1:-1]
    if len(lines) != 4:
        raise UnsupportedMeasurement(factor_name, "expected a name and exactly three criterion lines")
    if _name(lines[0]).casefold() != name.casefold():
        raise UnsupportedMeasurement(factor_name, "definition name does not match factor_name")
    criteria = {}
    for line in lines[1:]:
        line = _unwrap(_unbullet(line))
        line = re.sub(r"^(\*\*|__|`|\*|_)([-−]?\d+\s*:?)\1", r"\2", line)
        match = re.fullmatch(r"([-−]?\d+)\s*:\s*(.*)", line)
        if match is None:
            raise UnsupportedMeasurement(factor_name, "expected a numeric label and criterion")
        label = match.group(1).replace("−", "-")
        description = " ".join(match.group(2).split())
        if label not in ("1", "0", "-1") or label in criteria:
            raise UnsupportedMeasurement(factor_name, "labels must be exactly 1, 0 and -1 without duplicates")
        if not _unwrap(description):
            raise UnsupportedMeasurement(factor_name, "criteria must not be empty")
        criteria[label] = description
    if set(criteria) != {"1", "0", "-1"}:
        raise UnsupportedMeasurement(factor_name, "labels must be exactly 1, 0 and -1")
    return MeasurementSpec(
        factor_name=factor_name,
        jev_type="choice",
        instructions=f"Factor: {name}",
        criteria=tuple((label, criteria[label]) for label in ("1", "0", "-1")),
        choice_values={"1": 1.0, "0": 0.0, "-1": -1.0},
        source="parsed",
        notes="COAT ternary criteria parsed verbatim; value is the selected label, not probability or confidence.",
    ).validate()


def general_definition_to_spec(factor_name: str, factor_definition: str) -> MeasurementSpec:
    """Parse a general-template factor with an explicit measurement type.

    The proposal declares ``Type: score|choice``, ``Values: ...`` and
    ``Mentioned when: ...``; the mention sentence becomes a noul gate so that
    "text does not mention this attribute" is recorded as missing rather than
    folded into a value. Parsing is deterministic and makes no model calls.
    """
    if not isinstance(factor_name, str) or not factor_name.strip():
        raise UnsupportedMeasurement(factor_name, "factor_name must be a nonempty string")
    name = _name(factor_name)
    if not name:
        raise UnsupportedMeasurement(factor_name, "factor_name must contain a name")
    if not isinstance(factor_definition, str):
        raise UnsupportedMeasurement(factor_name, "factor definition must be text")
    lines = [line.strip() for line in factor_definition.splitlines() if line.strip()]
    if len(lines) >= 2 and lines[0] in ("```", "```text", "```markdown") and lines[-1] == "```":
        lines = lines[1:-1]
    if len(lines) < 4 or _name(lines[0]).casefold() != name.casefold():
        raise UnsupportedMeasurement(factor_name, "expected a name line and a general definition")
    kind = values_text = mentioned = None
    body = []
    for line in lines[1:]:
        stripped = _unwrap(_unbullet(line))
        match = _GENERAL_TYPE.match(stripped)
        if match and kind is None:
            kind = match.group(1).lower()
            continue
        match = _VALUES.match(stripped)
        if match and values_text is None:
            values_text = match.group(1).strip()
            continue
        match = _MENTIONED.match(stripped)
        if match and mentioned is None:
            mentioned = " ".join(match.group(1).split())
            continue
        body.append(stripped)
    if kind is None or values_text is None or mentioned is None:
        raise UnsupportedMeasurement(
            factor_name, "general definition needs Type, Values and Mentioned when"
        )
    if not _unwrap(mentioned):
        raise UnsupportedMeasurement(factor_name, "Mentioned when must not be empty")

    gate = MeasurementSpec(
        factor_name=factor_name,
        jev_type="noul",
        instructions=f"Does the text explicitly mention this attribute? Factor: {name}",
        criteria=(mentioned, "The text does not explicitly mention this attribute."),
        source="parsed",
        notes="Mention gate: text without an explicit statement is recorded as missing.",
    )

    if kind == "score":
        try:
            value_map = tuple(float(v.strip()) for v in values_text.split(","))
        except ValueError:
            raise UnsupportedMeasurement(factor_name, "score Values must be comma-separated numbers") from None
        levels = {}
        for line in body:
            match = _LEVEL.match(line)
            if match is None:
                raise UnsupportedMeasurement(factor_name, "expected numbered score levels")
            index = int(match.group(1))
            if index in levels:
                raise UnsupportedMeasurement(factor_name, "duplicate score level")
            levels[index] = " ".join(match.group(2).split())
        if len(value_map) < 2 or len(levels) != len(value_map):
            raise UnsupportedMeasurement(
                factor_name, "score needs one level per Values entry")
        # Level labels may be 0..n-1 or the numeric values themselves; either way
        # they are ordered ascending and paired with Values in that order.
        ordered = sorted(levels)
        return MeasurementSpec(
            factor_name=factor_name,
            jev_type="score",
            instructions=f"Factor: {name}",
            criteria=tuple(levels[key] for key in ordered),
            value_map=value_map,
            source="parsed",
            notes="General score parsed verbatim; value is the probability-weighted expectation over Values.",
            gate=gate,
        ).validate()

    options = {}
    for line in body:
        match = _OPTION.match(line)
        if match is None:
            raise UnsupportedMeasurement(factor_name, "expected named choice options")
        option, description = match.group(1).strip(), " ".join(match.group(2).split())
        if option in options:
            raise UnsupportedMeasurement(factor_name, "duplicate choice option")
        options[option] = description
    choice_values = {}
    for item in values_text.split(","):
        if "=" not in item:
            raise UnsupportedMeasurement(factor_name, "choice Values must use option=number")
        option, value = item.split("=", 1)
        try:
            choice_values[option.strip()] = float(value.strip())
        except ValueError:
            raise UnsupportedMeasurement(factor_name, "choice value must be numeric") from None
    if set(choice_values) != set(options):
        raise UnsupportedMeasurement(factor_name, "Values must cover exactly the declared options")
    return MeasurementSpec(
        factor_name=factor_name,
        jev_type="choice",
        instructions=f"Factor: {name}",
        criteria=tuple((option, options[option]) for option in options),
        choice_values=choice_values,
        source="parsed",
        notes="General choice parsed verbatim; value is the selected option's declared number.",
        gate=gate,
    ).validate()
