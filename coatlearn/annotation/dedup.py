"""Jev concept-level comparison, usable standalone or via opt-in COAT dedup.

Compare original definitions without rewriting them or reusing labels across
non-equivalent rubrics. UNCERTAIN and technical ERROR both retain a candidate,
but are distinct audit outcomes. This is not a definition/label merge engine.
"""
from __future__ import annotations

from dataclasses import asdict

from coatlearn.annotation.types import MeasurementSpec
from coatlearn.models.jev import JevConfig

DEFAULT_TASK_CONTEXT = "从非结构化文本中识别候选因子；比较完整定义对应的核心属性，而非预测目标值或判断因果关系。"


def normalize_factor_name(name: str) -> str:
    """Legacy exact-name normalization only; no substring or WordNet veto."""
    return " ".join(name.lower().replace("_", " ").replace("-", " ").split())


def configured_deduplicator(fields):
    """Validate optional core config without constructing an HTTP client."""
    if fields is None:
        return None, DEFAULT_TASK_CONTEXT
    if not isinstance(fields, dict) or fields.get("backend") not in {"jev", "kev"}:
        raise ValueError("factor_dedup requires backend='jev' or 'kev'")
    fields = dict(fields)
    context = fields.pop("task_context", DEFAULT_TASK_CONTEXT)
    if not isinstance(context, str) or not context.strip():
        raise ValueError("factor_dedup.task_context must be a nonempty string")
    from coatlearn.models.kev import decision_config
    return JevFactorDeduplicator(decision_config(fields)), context


def core_dedup_policy(comparator, context):
    """Record actual orchestration separately from the standalone pair policy."""
    if comparator is None:
        return None
    return {**comparator.policy(), "integrated_into_coat": True,
            "task_context": context, "comparison_order": "existing_A_new_B",
            "simple_rules": ["exact_normalized_name", "cross_cluster_exact_definition"],
            "representative": "first_valid_measured_factor",
            "candidate_order": "cluster_index_then_proposal_order",
            "reverse_check": False, "confidence_threshold": None}

DEDUP_INSTRUCTIONS = """判断两个候选因子是否描述同一个核心属性。
同一属性的措辞、评价角度、取值范围或判定细节有所变化，不足以构成新的独立因子。例如，属性本身与针对该属性的评价，可以属于概念重复。
但仅仅讨论同一对象、主题相关、存在因果联系或统计相关，不构成重复。不同主体的评价或不同结果维度，也不能仅因相关而合并。
比较两份完整定义，不仅比较名称。
不改写定义，不判断哪个定义更好。"""


def dedup_question() -> dict:
    return {
        "type": "choice",
        "instructions": DEDUP_INSTRUCTIONS,
        "criteria": {
            "DUPLICATE": "能够明确判断为同一核心属性的重复表达。",
            "DISTINCT": "能够明确指出不同的核心属性。",
            "UNCERTAIN": "现有定义不足以明确判断。",
        },
    }


def build_state(definition_a: str, definition_b: str, task_context: str) -> str:
    """Only task context and verbatim definitions enter state; no reference labels."""
    for text in (definition_a, definition_b, task_context):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Two nonempty definitions and task context are required")
    return ("# Task context\n\n" + task_context
            + "\n\n# Factor A (verbatim)\n\n" + definition_a
            + "\n\n# Factor B (verbatim)\n\n" + definition_b)


class JevFactorDeduplicator:
    """One native Choice request per pair; no main LLM or discovery calls.

    Original request and response are local audit artifacts, not public results.
    No confidence cutoff, automatic retries beyond config, or label rewriting.
    """

    def __init__(self, config: JevConfig):
        from coatlearn.annotation.measurement import decision_annotator
        self.backend = decision_annotator(config)
        question = dedup_question()
        self.spec = MeasurementSpec(
            factor_name="relation", jev_type="choice",
            instructions=question["instructions"],
            criteria=tuple(question["criteria"].items()),
            choice_values={"DUPLICATE": 1, "DISTINCT": 0, "UNCERTAIN": -1},
        ).validate()

    def policy(self) -> dict:
        backend = self.backend.config.redacted().get("backend", "jev")
        return {"version": "concept-dedup-v1", "backend": backend,
                "config": self.backend.config.redacted(),
                "questions": {"relation": dedup_question()},
                "uncertain_action": "retain", "error_action": "retain",
                "rewrites_definitions": False, "integrated_into_coat": False}

    def stats(self) -> dict:
        return asdict(self.backend.stats())

    def evaluate(self, definition_a: str, definition_b: str, task_context: str) -> dict:
        request = {"model": self.backend.config.model,
                   "state": build_state(definition_a, definition_b, task_context),
                   "questions": {"relation": dedup_question()}}
        result = {"status": "error", "decision": "ERROR", "action": "retain",
                  "request": request, "response": None}
        try:
            response = self.backend._system_one(request)
            result["response"] = response
            if (response.get("model") != self.backend.config.model
                    or set(response.get("answers", {})) != {"relation"}):
                raise ValueError("Unexpected comparator model or answer keys")
            answer = self.backend._interpret_answer(
                "relation", self.spec, response["answers"]["relation"], None)
            decision = answer.raw["choice"]
            result.update(status="ok", decision=decision,
                          action="deduplicate" if decision == "DUPLICATE" else "retain",
                          probabilities=answer.probabilities, confidence=answer.confidence)
        except Exception as exc:
            # Transport exceptions may contain secrets: record type only.
            result["error_type"] = type(exc).__name__
        return result


def compare_definitions(definition_a, definition_b, *, model, task_context=DEFAULT_TASK_CONTEXT):
    """Compare two full definitions without rewriting them or running discovery."""
    from .batch import resolve_annotator
    backend = resolve_annotator(model)
    comparator = JevFactorDeduplicator(backend.config)
    comparator.backend = backend
    return comparator.evaluate(definition_a, definition_b, task_context)
