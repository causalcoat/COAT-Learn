"""Opt-in Jev factor-definition judge; separate from per-review annotation.

One native request, two independent Choice questions, both must PASS. The
wording-clarity instruction is the author's exact text. No schema/main LLM,
confidence threshold, or implicit rewrite of the proposed definition is used.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Any

from coatlearn.annotation.types import MeasurementSpec, UnsupportedMeasurement
from coatlearn.models.jev import JevConfig
from coatlearn.annotation.definitions import factor_definition_to_spec

CLARITY_INSTRUCTIONS = "结合任务示例，因子名称和各取值说明是否有明确、可理解的含义？仅因实质性的文字歧义拒绝，例如指代不明、修饰语或否定范围存在不同合理解释。"
FORMAT_INSTRUCTIONS = "因子定义的格式是否正确：一个非空的因子名称，随后是标记为 1、0、-1 的三条非空取值说明，三个标签各出现一次且没有额外取值或内容？Markdown 修饰和空白不影响格式。"
FORMAT_INSTRUCTIONS_GENERAL = "因子定义的格式是否正确：一个非空的因子名称，随后声明 Type（score 或 choice）、Values、Mentioned when，并且每个取值或选项都有非空的说明？Markdown 修饰和空白不影响格式。"


def judge_questions(template: str = "simple") -> dict:
    """Fresh native question objects, safe to freeze in audit/provenance."""
    format_instructions = (
        FORMAT_INSTRUCTIONS_GENERAL if template == "general" else FORMAT_INSTRUCTIONS
    )
    return {
        "clarity": {"type": "choice", "instructions": CLARITY_INSTRUCTIONS, "criteria": {
            "PASS": "因子名称和各取值说明有明确、可理解的含义。",
            "REJECT": "存在实质性的文字歧义。"}},
        "format": {"type": "choice", "instructions": format_instructions, "criteria": {
            "PASS": "格式正确。", "REJECT": "格式不正确。"}},
    }


def proposal_examples(prompt: str) -> str:
    """Extract the actual data section, never regenerate/resample examples."""
    examples = prompt.split("# Data\n\n", 1)[1].split("# Tasks: Factor Abstraction.", 1)[0]
    for heading in ("\n# Existing Factors", "\n# Previously Rejected Factors"):
        examples = examples.split(heading, 1)[0]
    return examples


def configured_judge(fields, template="simple"):
    """Choose a typed backend without changing judge questions or acceptance."""
    if fields is None:
        return None
    if not isinstance(fields, dict) or fields.get("backend") not in {"jev", "kev"}:
        raise ValueError("factor_judge requires backend='jev' or 'kev'")
    from coatlearn.models.kev import decision_config
    return JevFactorJudge(decision_config(fields), template=template)


class JevFactorJudge:
    """Reuse the existing native transport and strict Choice response parser.

    ``evaluate`` returns a local audit record, which can contain example text
    and final responses. It must NOT be put into the public result dictionary.
    API/response errors are ERROR, not a semantic rejection or a normal zero.
    """

    def __init__(self, config: JevConfig, template: str = "simple"):
        from coatlearn.annotation.measurement import decision_annotator
        self.backend = decision_annotator(config)
        self.template = template
        self.specs = {
            name: MeasurementSpec(factor_name=name, jev_type="choice", instructions=q["instructions"],
                                  criteria=tuple(q["criteria"].items()), choice_values={"PASS": 1, "REJECT": 0}).validate()
            for name, q in judge_questions(template).items()
        }

    def policy(self) -> dict:
        backend = self.backend.config.redacted().get("backend", "jev")
        return {"backend": backend, "version": "clarity-format-v1", "template": self.template,
                "config": self.backend.config.redacted(),
                "questions": judge_questions(self.template), "acceptance": "both_PASS", "trials": 1,
                "replaces": "legacy_quality_prescoring_and_three_LLM_judges"}

    def stats(self) -> dict:
        return asdict(self.backend.stats())

    def evaluate(self, factor_name: str, definition: str, examples: str) -> dict[str, Any]:
        result = {"status": "ok", "decision": "REJECT", "local_format": "PASS", "response": None,
                  "request": {"model": self.backend.config.model,
                              "state": "# Candidate factor definition (verbatim)\n\n" + definition
                              + "\n\n# Original proposal examples (verbatim data section)\n" + examples,
                              "questions": judge_questions(self.template)}}
        try:
            factor_definition_to_spec(factor_name, definition)
        except UnsupportedMeasurement:
            result.update(local_format="REJECT", rejected_checks=["local_format"])
            return result
        try:
            if not isinstance(examples, str) or not examples.strip():
                raise ValueError("Original proposal examples are required")
            response = self.backend._system_one(result["request"])
            result["response"] = response  # retain even malformed final JSON in local audit
            if response.get("model") != self.backend.config.model or set(response.get("answers", {})) != set(self.specs):
                raise ValueError("Unexpected judge model or answer keys")
            answers = {}
            for name, spec in self.specs.items():
                annotation = self.backend._interpret_answer(name, spec, response["answers"][name], None)
                answers[name] = {"decision": annotation.raw["choice"], "probabilities": annotation.probabilities,
                                 "confidence": annotation.confidence}
            rejected = [name for name, a in answers.items() if a["decision"] == "REJECT"]
            result.update(answers=answers, rejected_checks=rejected, decision="REJECT" if rejected else "PASS")
        except Exception as exc:
            result.update(status="error", decision="ERROR", error_type=type(exc).__name__)
        return result


def review_definition(name, definition, *, examples, model, template="general"):
    """Review one definition independently of annotation or discovery."""
    from .batch import resolve_annotator
    backend = resolve_annotator(model)
    judge = JevFactorJudge(backend.config, template=template)
    judge.backend = backend
    return judge.evaluate(name, definition, examples)
