"""Factor measurement: definitions, value mappings and per-row metadata."""
from __future__ import annotations

import json
import math
import threading
from collections.abc import Mapping as AbcMapping
from typing import TYPE_CHECKING, Any, Dict, Mapping, Optional, Tuple

from coatlearn.annotation.types import (
    MAX_CHOICE_OPTIONS,
    MAX_SCORE_LEVELS,
    MIN_SCORE_LEVELS,
    MENTION_SUFFIX,
    FactorAnnotation,
    MeasurementSpec,
    UnsupportedMeasurement,
)
from coatlearn.annotation.definitions import factor_definition_to_spec

if TYPE_CHECKING:  # 仅用于类型标注；运行时不依赖 models，保持 Jev 后端独立
    from coatlearn.models.openai import ChatModel

__all__ = [
    "SYSTEM_ONE_PATH",
    "JevConfig",
    "JevStats",
    "JevAPIError",
    "JevAnnotator",
]

from coatlearn.models.systemone import (
    JevClient, JevConfig, JevStats, JevAPIError, SYSTEM_ONE_PATH,
    _is_finite_number, _as_confidence, _normalize_probabilities, _optional_float,
)
# 缺失门阈值：noul 的“已提到”概率低于该值时，该样本记为缺失，不采用取值题答案。
MENTION_THRESHOLD = 0.5

_COMPILATION_PROMPT = """You compile a text-factor definition into ONE TypeSafe (Jev) annotation question. Output ONLY one JSON object: no prose, no code fences.

Factor definition (verbatim; never invent scale values absent from it):
{definition}

Allowed outputs:
- {"type": "score", "instructions": "...", "criteria": ["<low>", "...", "<high>"], "value_map": [...]}
  * criteria: {min_levels}..{max_levels} concrete situational level descriptions ordered low to high (no bare numbers, no degree words like "moderately").
  * value_map: same length as criteria, strictly increasing finite numbers taken from the numeric scale declared in the factor definition. Never discretize a continuous or scale-free definition.
- {"type": "noul", "instructions": "...", "criteria": {"true": "...", "false": "..."}} — only if the factor value is inherently a 0..1 probability / boolean.
- {"type": "choice", "instructions": "...", "criteria": {"<option>": "<desc> or null"}, "choice_values": {"<option>": <number>}} — only if the definition enumerates categories together with their numeric values.
- {"type": "unsupported", "reason": "..."} — whenever no faithful encoding exists.

If the definition states explicit numeric bounds, also copy them into optional keys "declared_min" / "declared_max".
"""


class JevAnnotator(JevClient):
    """Jev 因子标注器：prepare/annotate 两段式，线程安全。

    典型用法（core 侧）::

        annotator = JevAnnotator(config)                            # 默认纯解析
        for name, definition in factors:                            # 线程池之前
            annotator.prepare_factor(name, definition)
        with ThreadPoolExecutor(max_workers=n) as pool:             # 跨样本并发
            values = [pool.submit(annotator.annotate_factor, name, definition, text) ...]
    """

    def __init__(
        self,
        config: JevConfig,
        *,
        measurements: Optional[Mapping[str, MeasurementSpec]] = None,
        schema_model: Optional["ChatModel"] = None,
    ) -> None:
        if not isinstance(config, JevConfig):
            raise TypeError("config 必须是 JevConfig")
        super().__init__(config)
        self.schema_model = schema_model
        self._specs: Dict[str, MeasurementSpec] = {}
        for name, spec in (measurements or {}).items():
            if spec.factor_name != name:
                raise ValueError(
                    f"measurements 的键 {name!r} 与 spec.factor_name={spec.factor_name!r} 不一致"
                )
            try:
                spec.validate()
            except ValueError as exc:
                raise ValueError(f"显式 MeasurementSpec 无效：{exc}") from None
            self._specs[name] = spec

        self._explicit_names = set(self._specs)
        self._compiled_definitions: Dict[str, str] = {}
        self._compile_lock = threading.Lock()
        self._metadata: Dict[Tuple[Any, str], Dict[str, Any]] = {}
        self._metadata_lock = threading.Lock()
        self._archived_factors = []  # rejected batches; immutable through snapshot copies
        self._auto_sample_counter = 0

    # ------------------------------------------------------------------
    # 因子准备（线程池之前一次）
    # ------------------------------------------------------------------

    def prepare_factor(self, factor_name: str, factor_definition: Any) -> MeasurementSpec:
        """优先使用显式规格，否则纯解析标准定义；旧版编译须显式启用。

        等价的已解析定义复用同一 spec；同名定义变更明确拒绝，避免陈旧缓存。
        解析、编译与缓存注册均在线程锁内完成。
        """
        if isinstance(factor_definition, MeasurementSpec):
            return self._resolve_spec(factor_name, factor_definition)
        with self._compile_lock:
            existing = self._specs.get(factor_name)
            if factor_name in self._explicit_names:
                return existing
            try:
                spec = factor_definition_to_spec(factor_name, factor_definition)
            except UnsupportedMeasurement:
                if existing is not None and existing.source == "parsed":
                    raise UnsupportedMeasurement(factor_name, "同名因子定义已更改") from None
                if self.schema_model is None:
                    raise
                if isinstance(factor_definition, str):
                    definition_key = factor_definition
                elif isinstance(factor_definition, AbcMapping):
                    definition_key = json.dumps(dict(factor_definition), ensure_ascii=False, sort_keys=True)
                else:
                    raise UnsupportedMeasurement(factor_name, "factor_definition 必须是字符串或映射") from None
                if existing is not None:
                    if self._compiled_definitions.get(factor_name) != definition_key:
                        raise UnsupportedMeasurement(factor_name, "同名因子定义已更改") from None
                    return existing
                spec = self._compile_spec(factor_name, factor_definition)
                self._compiled_definitions[factor_name] = definition_key
            if existing is not None:
                if existing != spec:
                    raise UnsupportedMeasurement(factor_name, "同名因子定义已更改")
                return existing
            self._specs[factor_name] = spec
            return spec

    def finalized_specs(self) -> Mapping[str, MeasurementSpec]:
        """当前有效缓存的只读快照；退役批次的规格见 archived_factor_snapshot。"""
        with self._compile_lock:
            return dict(self._specs)

    def archive_rejected_factor(self, factor_name: str, factor_definition: Any,
                                rejection_reason: str) -> str:
        """Archive a numerically rejected batch AFTER its workers have joined.

        Release only that batch's auto-prepared cache entry, so a later proposal
        can be prepared normally. The default prepare_factor conflict guard is
        unchanged. Explicit user-supplied measurement specs remain pinned.
        Original definitions/specs/metadata survive even repeated rejected names;
        no review text, credentials or HTTP headers are copied into the archive.
        """
        from copy import deepcopy

        if not isinstance(rejection_reason, str) or not rejection_reason.strip():
            raise ValueError("A numerical rejection reason is required")
        with self._compile_lock:
            existing = self._specs.get(factor_name)
            if existing is None:
                raise ValueError("No prepared factor to archive")
            if existing.source == "parsed" and factor_name not in self._explicit_names:
                if factor_definition_to_spec(factor_name, factor_definition) != existing:
                    raise UnsupportedMeasurement(factor_name, "被拒批次定义与当前缓存不一致")
            elif factor_name in self._compiled_definitions:
                key = (factor_definition if isinstance(factor_definition, str)
                       else json.dumps(dict(factor_definition), ensure_ascii=False, sort_keys=True))
                if key != self._compiled_definitions[factor_name]:
                    raise UnsupportedMeasurement(factor_name, "被拒批次定义与当前缓存不一致")
            with self._metadata_lock:
                archive_id = f"rejected-{len(self._archived_factors) + 1:06d}"
                keys = [key for key in self._metadata if key[1] == factor_name]
                self._archived_factors.append(deepcopy({
                    "archive_id": archive_id, "factor": factor_name,
                    "definition": factor_definition, "spec": existing,
                    "rejection_reason": rejection_reason,
                    "metadata": [self._metadata[key] for key in keys],
                }))
                for key in keys:
                    del self._metadata[key]
                if factor_name not in self._explicit_names:
                    del self._specs[factor_name]
                    self._compiled_definitions.pop(factor_name, None)
                return archive_id

    def archived_factor_snapshot(self) -> list:
        """Independent copies of rejected batches; current metadata stays separate."""
        from copy import deepcopy

        with self._metadata_lock:
            return deepcopy(self._archived_factors)

    # ------------------------------------------------------------------
    # 标注接口
    # ------------------------------------------------------------------

    def annotate_factor(
        self,
        factor_name: str,
        factor_definition: Any,
        text: str,
        *,
        sample_id: Any = None,
    ) -> float:
        """标注单个因子，返回因子原刻度上的数值。

        未 prepare 时先按同一规则准备。默认 choice 返回所选标签；概率与
        置信度只作元数据。可选 score/noul 仍按显式规格返回软测量值。
        """
        annotations = self._annotate_many(text, {factor_name: factor_definition}, sample_id)
        return annotations[factor_name].value

    def annotate_record(
        self,
        text: str,
        specs: Mapping[str, Any],
        *,
        sample_id: Any = None,
    ) -> Dict[str, FactorAnnotation]:
        """同一条 state 一次请求标注多个因子（Jev 多 questions 并行评估）。

        specs 为 ``{因子名: factor_definition 或 MeasurementSpec}``；
        返回 ``{因子名: FactorAnnotation}``。仅是可选的批量优化，
        不改变 core 的单因子接口。
        """
        return self._annotate_many(text, dict(specs), sample_id)

    def metadata_snapshot(self) -> Dict[Tuple[Any, str], Dict[str, Any]]:
        """(sample_id, factor) -> 元数据（概率/置信度/映射说明）的只读快照。"""
        with self._metadata_lock:
            return {key: dict(value) for key, value in self._metadata.items()}

    def annotation_metadata(self, sample_id: Any, factor_name: str) -> Optional[Dict[str, Any]]:
        """查询单条 (sample_id, factor) 元数据；不存在时返回 None。"""
        with self._metadata_lock:
            stored = self._metadata.get((sample_id, factor_name))
            return dict(stored) if stored is not None else None


    # ------------------------------------------------------------------
    # 内部：请求编排
    # ------------------------------------------------------------------

    def _annotate_many(
        self,
        text: str,
        specs: Mapping[str, Any],
        sample_id: Any,
    ) -> Dict[str, FactorAnnotation]:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text 必须是非空字符串")
        if not specs:
            raise ValueError("specs 不能为空")
        resolved = {name: self._resolve_spec(name, spec) for name, spec in specs.items()}
        questions: Dict[str, Any] = {}
        for spec in resolved.values():
            questions.update(spec.to_request_questions())
        payload = {"model": self.config.model, "state": text, "questions": questions}
        response = self._system_one(payload)
        answers = response.get("answers")
        if not isinstance(answers, dict):
            raise JevAPIError("Jev 响应缺少 answers 对象")
        missing = [qid for qid in questions if qid not in answers]
        if missing:
            raise JevAPIError(f"Jev 响应缺少问题答案：{missing}")
        stored_sample_id = self._resolve_sample_id(sample_id)
        results: Dict[str, FactorAnnotation] = {}
        for name, spec in resolved.items():
            mention_probability = None
            missing_reason = None
            if spec.gate is None:
                annotation = self._interpret_answer(name, spec, answers[name], sample_id)
            else:
                gate_name = name + MENTION_SUFFIX
                gate_annotation = self._interpret_answer(
                    gate_name, spec.gate, answers[gate_name], sample_id
                )
                mention_probability = float(gate_annotation.value)
                if mention_probability < MENTION_THRESHOLD:
                    # 文本未明确提及该属性：记为缺失，绝不把缺失压成 0。
                    missing_reason = "not_mentioned"
                    annotation = FactorAnnotation(
                        factor_name=name,
                        value=float("nan"),
                        spec=spec,
                        raw={"mention": {"noul": mention_probability}},
                        sample_id=sample_id,
                        backend_model=self.config.model,
                        value_policy="missing_below_mention_threshold",
                    )
                else:
                    annotation = self._interpret_answer(name, spec, answers[name], sample_id)
            metadata = annotation.to_metadata()
            if mention_probability is not None:
                metadata["mention_probability"] = mention_probability
                metadata["mention_threshold"] = MENTION_THRESHOLD
                metadata["missing_reason"] = missing_reason
            metadata["sample_id"] = stored_sample_id
            with self._metadata_lock:
                self._metadata[(stored_sample_id, name)] = metadata
            results[name] = annotation
        return results

    def _resolve_spec(self, factor_name: str, spec_or_definition: Any) -> MeasurementSpec:
        if isinstance(spec_or_definition, MeasurementSpec):
            spec = spec_or_definition
            if spec.factor_name != factor_name:
                raise ValueError(
                    f"MeasurementSpec.factor_name={spec.factor_name!r} 与请求的因子名 {factor_name!r} 不一致"
                )
            try:
                spec.validate()
            except ValueError as exc:
                raise ValueError(f"显式 MeasurementSpec 无效：{exc}") from None
            with self._compile_lock:
                existing = self._specs.get(factor_name)
                if existing is not None and existing != spec:
                    raise ValueError("显式 MeasurementSpec 与已准备的同名规格冲突")
                if existing is not None:
                    return existing
                self._explicit_names.add(factor_name)
                self._specs[factor_name] = spec
                return spec
        return self.prepare_factor(factor_name, spec_or_definition)

    def _interpret_answer(
        self,
        factor_name: str,
        spec: MeasurementSpec,
        answer: Any,
        sample_id: Any,
    ) -> FactorAnnotation:
        """按 spec 类型解释应答；因子值一律经显式映射回原刻度。"""
        if not isinstance(answer, dict):
            raise JevAPIError(f"answers[{factor_name!r}] 不是 JSON 对象")
        if spec.jev_type == "score":
            levels = len(spec.criteria)
            raw_score = answer.get("score")
            if (
                not _is_finite_number(raw_score)
                or not (0.0 <= float(raw_score) <= levels - 1)
            ):
                raise JevAPIError(
                    f"answers[{factor_name!r}] 的 score 缺失、非数值或越界 [0, {levels - 1}]"
                )
            score = float(raw_score)
            try:
                probabilities = _normalize_probabilities(answer.get("probabilities"))
                value, value_policy = spec.resolve_score_factor_value(score, probabilities)
            except ValueError as exc:
                raise JevAPIError(
                    f"answers[{factor_name!r}] 的概率分布不可用：{exc}"
                ) from None
            return FactorAnnotation(
                factor_name=factor_name,
                value=value,
                spec=spec,
                raw={"score": score},
                probabilities=probabilities,
                confidence=_as_confidence(answer.get("confidence")),
                sample_id=sample_id,
                backend_model=self.config.model,
                value_policy=value_policy,
            )
        if spec.jev_type == "noul":
            raw_noul = answer.get("noul")
            if not _is_finite_number(raw_noul) or not (0.0 <= float(raw_noul) <= 1.0):
                raise JevAPIError(f"answers[{factor_name!r}] 的 noul 缺失、非数值或越界 [0, 1]")
            yes_probability = float(raw_noul)
            return FactorAnnotation(
                factor_name=factor_name,
                value=spec.noul_to_factor_value(yes_probability),
                spec=spec,
                raw={"noul": yes_probability},
                probabilities={"yes": yes_probability, "no": 1.0 - yes_probability},
                confidence=None,  # Jev 的 noul 没有独立置信度
                sample_id=sample_id,
                backend_model=self.config.model,
                value_policy="noul_soft_yes_probability",
            )
        if answer.get("type") != "choice":
            raise JevAPIError("Jev choice 应答的 type 缺失或不匹配")
        option = answer.get("choice")
        if not isinstance(option, str) or option not in spec.choice_values:
            raise JevAPIError("Jev choice 应答的所选标签缺失或未在规格中声明")
        probabilities = answer.get("probabilities")
        confidence = answer.get("confidence")
        try:
            valid_probabilities = (
                isinstance(probabilities, dict)
                and set(probabilities) == set(spec.choice_values)
                and all(_is_finite_number(p) and 0.0 <= p <= 1.0 for p in probabilities.values())
                and abs(sum(probabilities.values()) - 1.0) <= 1e-6
            )
            valid_confidence = _is_finite_number(confidence) and 0.0 <= confidence <= 1.0
        except (ValueError, OverflowError):
            raise JevAPIError("Jev choice 应答的概率或置信度不是有效有限数值") from None
        if not valid_probabilities:
            raise JevAPIError("Jev choice 应答必须包含全部已声明标签的 [0, 1] 归一概率")
        if not valid_confidence:
            raise JevAPIError("Jev choice 应答必须包含 [0, 1] 内的有限 confidence")
        return FactorAnnotation(
            factor_name=factor_name,
            value=spec.choice_to_factor_value(option),
            spec=spec,
            raw={"choice": option},
            probabilities={label: float(p) for label, p in probabilities.items()},
            confidence=float(confidence),
            sample_id=sample_id,
            backend_model=self.config.model,
            value_policy="choice_selected_label_mapped",
        )

    # ------------------------------------------------------------------
    # 内部：schema_model 编译（每因子一次）
    # ------------------------------------------------------------------

    def _compile_spec(self, factor_name: str, factor_definition: Any) -> MeasurementSpec:
        if isinstance(factor_definition, str):
            definition_text = factor_definition
        elif isinstance(factor_definition, AbcMapping):
            definition_text = json.dumps(dict(factor_definition), ensure_ascii=False, sort_keys=True)
        else:
            raise UnsupportedMeasurement(factor_name, "factor_definition 必须是字符串或映射")
        prompt = (
            _COMPILATION_PROMPT
            .replace("{definition}", definition_text)
            .replace("{min_levels}", str(MIN_SCORE_LEVELS))
            .replace("{max_levels}", str(MAX_SCORE_LEVELS))
        )
        try:
            response_text = self.schema_model.chat(prompt, return_text=True)
        except Exception as exc:  # 脱敏：不带出 provider 原始错误
            raise UnsupportedMeasurement(
                factor_name, f"schema_model 编译调用失败：{type(exc).__name__}（详情已省略）"
            ) from None
        if not isinstance(response_text, str):
            raise UnsupportedMeasurement(factor_name, "schema_model 未返回字符串")
        data = self._parse_json_object(factor_name, response_text)

        spec_type = data.get("type")
        if spec_type == "unsupported":
            reason = data.get("reason")
            raise UnsupportedMeasurement(
                factor_name,
                reason if isinstance(reason, str) and reason else "schema_model 判定该因子不适合 Jev",
            )
        instructions = data.get("instructions")
        if not isinstance(instructions, str) or not instructions.strip():
            instructions = f"Factor: {factor_name}"
        if spec_type == "score":
            return self._build_score_spec(factor_name, instructions, data)
        if spec_type == "noul":
            return self._build_noul_spec(factor_name, instructions, data)
        if spec_type == "choice":
            return self._build_choice_spec(factor_name, instructions, data)
        raise UnsupportedMeasurement(factor_name, f"schema_model 返回了未知问题类型：{spec_type!r}")

    @staticmethod
    def _parse_json_object(factor_name: str, text: str) -> Dict[str, Any]:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise UnsupportedMeasurement(factor_name, "schema_model 输出中找不到 JSON 对象")
        try:
            data = json.loads(text[start : end + 1])
        except ValueError:
            raise UnsupportedMeasurement(factor_name, "schema_model 输出不是合法 JSON") from None
        if not isinstance(data, dict):
            raise UnsupportedMeasurement(factor_name, "schema_model 输出的 JSON 顶层必须是对象")
        return data

    def _build_score_spec(
        self, factor_name: str, instructions: str, data: Dict[str, Any]
    ) -> MeasurementSpec:
        criteria = data.get("criteria")
        value_map = data.get("value_map")
        if not isinstance(criteria, list) or not all(
            isinstance(level, str) and level.strip() for level in criteria
        ):
            raise UnsupportedMeasurement(factor_name, "score 编译结果缺少合法的 criteria 级别描述")
        if not (MIN_SCORE_LEVELS <= len(criteria) <= MAX_SCORE_LEVELS):
            raise UnsupportedMeasurement(
                factor_name, f"score 级别数必须在 {MIN_SCORE_LEVELS}..{MAX_SCORE_LEVELS}"
            )
        if not isinstance(value_map, list) or len(value_map) != len(criteria):
            raise UnsupportedMeasurement(factor_name, "value_map 必须与 criteria 等长")
        if not all(_is_finite_number(v) for v in value_map):
            raise UnsupportedMeasurement(factor_name, "value_map 必须全为有限数值")
        values = tuple(float(v) for v in value_map)
        if not all(math.isfinite(v) for v in values):
            raise UnsupportedMeasurement(factor_name, "value_map 必须全为有限数值")
        if any(values[i] >= values[i + 1] for i in range(len(values) - 1)):
            raise UnsupportedMeasurement(
                factor_name,
                "value_map 必须严格递增（对应 criteria 从低到高），且不得发明因子定义之外的刻度",
            )
        declared_min = _optional_float(data.get("declared_min"))
        declared_max = _optional_float(data.get("declared_max"))
        if declared_min is not None and values[0] < declared_min - 1e-9:
            raise UnsupportedMeasurement(
                factor_name, f"value_map 最低值 {values[0]} 低于定义声明的刻度下界 {declared_min}"
            )
        if declared_max is not None and values[-1] > declared_max + 1e-9:
            raise UnsupportedMeasurement(
                factor_name, f"value_map 最高值 {values[-1]} 高于定义声明的刻度上界 {declared_max}"
            )
        notes = (
            f"编译自因子定义：因子值 = Jev 级别概率对 value_map={list(values)} 的加权期望；"
            "分布缺失/不完整时仅当 value_map 等距才从级别位置线性回退（软测量，非硬标签）"
        )
        return self._validated(
            factor_name,
            MeasurementSpec(
                factor_name=factor_name,
                jev_type="score",
                instructions=instructions,
                criteria=tuple(criteria),
                value_map=values,
                source="compiled",
                notes=notes,
            ),
        )

    def _build_noul_spec(
        self, factor_name: str, instructions: str, data: Dict[str, Any]
    ) -> MeasurementSpec:
        criteria_data = data.get("criteria")
        criteria: Optional[Tuple[str, str]] = None
        if criteria_data is not None:
            if not (
                isinstance(criteria_data, dict)
                and isinstance(criteria_data.get("true"), str)
                and criteria_data["true"].strip()
                and isinstance(criteria_data.get("false"), str)
                and criteria_data["false"].strip()
            ):
                raise UnsupportedMeasurement(
                    factor_name, "noul 的 criteria 需要 true/false 两个非空描述"
                )
            criteria = (criteria_data["true"], criteria_data["false"])
        value_map: Optional[Tuple[float, ...]] = None
        value_map_data = data.get("value_map")
        if value_map_data is not None:
            if not isinstance(value_map_data, list) or len(value_map_data) != 2:
                raise UnsupportedMeasurement(factor_name, "noul 的 value_map 必须是 [no 值, yes 值]")
            if not all(_is_finite_number(v) for v in value_map_data):
                raise UnsupportedMeasurement(factor_name, "noul 的 value_map 必须全为有限数值")
            value_map = tuple(float(v) for v in value_map_data)
        notes = "编译自因子定义：Jev yes 概率" + (
            "经 value_map（no, yes）线性映射回因子刻度（软测量）"
            if value_map is not None
            else "直接作为因子值（0..1，软测量，非硬标签）"
        )
        return self._validated(
            factor_name,
            MeasurementSpec(
                factor_name=factor_name,
                jev_type="noul",
                instructions=instructions,
                criteria=criteria,
                value_map=value_map,
                source="compiled",
                notes=notes,
            ),
        )

    def _build_choice_spec(
        self, factor_name: str, instructions: str, data: Dict[str, Any]
    ) -> MeasurementSpec:
        criteria_data = data.get("criteria")
        choice_values_data = data.get("choice_values")
        if not isinstance(criteria_data, dict) or not criteria_data:
            raise UnsupportedMeasurement(factor_name, "choice 编译结果缺少合法的 criteria 选项映射")
        if len(criteria_data) > MAX_CHOICE_OPTIONS:
            raise UnsupportedMeasurement(
                factor_name, f"choice 选项数 {len(criteria_data)} 超过上限 {MAX_CHOICE_OPTIONS}"
            )
        if not isinstance(choice_values_data, dict) or set(choice_values_data) != set(criteria_data):
            raise UnsupportedMeasurement(factor_name, "choice_values 必须与 criteria 选项一一对应")
        if not all(_is_finite_number(v) for v in choice_values_data.values()):
            raise UnsupportedMeasurement(factor_name, "choice_values 必须全为有限数值")
        choice_values = {str(option): float(v) for option, v in choice_values_data.items()}
        criteria = tuple(
            (str(option), desc if isinstance(desc, str) else None)
            for option, desc in criteria_data.items()
        )
        notes = (
            f"编译自因子定义：Jev 所选选项经 choice_values={choice_values} 映射为因子值"
        )
        return self._validated(
            factor_name,
            MeasurementSpec(
                factor_name=factor_name,
                jev_type="choice",
                instructions=instructions,
                criteria=criteria,
                choice_values=choice_values,
                source="compiled",
                notes=notes,
            ),
        )

    @staticmethod
    def _validated(factor_name: str, spec: MeasurementSpec) -> MeasurementSpec:
        try:
            return spec.validate()
        except ValueError as exc:
            raise UnsupportedMeasurement(factor_name, f"编译出的 spec 未通过校验：{exc}") from None

    def _resolve_sample_id(self, sample_id: Any) -> Any:
        if sample_id is not None:
            return sample_id
        with self._metadata_lock:
            self._auto_sample_counter += 1
            return f"auto-{self._auto_sample_counter:06d}"


class KevAnnotator(JevAnnotator):
    """Factor measurement using the same System One questions on local Kev."""

    def __init__(self, config=None, **kwargs):
        from coatlearn.models.kev import KevConfig
        super().__init__(config or KevConfig(), **kwargs)

    def _ensure_client(self):
        from coatlearn.models.kev import KevClient
        return KevClient._ensure_client(self)


FactorAnnotator = JevAnnotator


def decision_annotator(config):
    """Create fresh measurement state for one explicitly chosen model."""
    from coatlearn.models.kev import KevConfig
    return KevAnnotator(config) if isinstance(config, KevConfig) else JevAnnotator(config)
