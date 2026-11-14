"""
标注后端共享类型：MeasurementSpec（显式、纯解析或经 schema_model 编译出的规格）、
FactorAnnotation（带概率/置信度元数据的单因子标注结果），以及
UnsupportedMeasurement（表示"该因子不能由当前后端度量"）。

设计红线（对 core 的承诺）：
- 因子值必须保持原定义刻度：Jev score 只以「级别概率对 value_map 的加权期望」
  映射回原刻度。非等距刻度下，级别位置期望 E[level] 的插值是**有偏**的
  （例：value_map=(0,1,10)、p=(.5,0,.5) ⇒ E[level]=1 但原刻度期望=5），
  绝不使用；分布缺失/不完整时，仅等距 value_map 允许从 score 线性回退，
  否则明确报错，绝不以级别位置猜测；
- 非法概率、未知/重复级别、完整但不归一的分布、NaN/inf 一律显式拒绝，
  不静默 clamp、不静默重归一；
- score/noul 是**软测量**（分布期望 / yes 概率），与传统实验的硬标签标注
  不等价；元数据以 value_policy 明示测量模式，置信度绝不当作因子值；
- probabilities 与 confidence 分开存放；
- 本模块不包含任何网络代码。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "SUPPORTED_JEV_TYPES",
    "MIN_SCORE_LEVELS",
    "MAX_SCORE_LEVELS",
    "MAX_CHOICE_OPTIONS",
    "UnsupportedMeasurement",
    "MeasurementSpec",
    "FactorAnnotation",
]

# Jev/TypeSafe 原语：score（有序量表）、choice（有限选项）、noul（yes/no 概率）
SUPPORTED_JEV_TYPES: Tuple[str, ...] = ("score", "choice", "noul")
# 缺失门：先问“文本是否提到该属性”(noul)，未提到则不取取值题的答案。
MENTION_SUFFIX = "::mentioned"
MIN_SCORE_LEVELS = 2
MAX_SCORE_LEVELS = 10  # TypeSafe Score criteria 官方上限
MAX_CHOICE_OPTIONS = 255  # TypeSafe Choice 官方上限


class UnsupportedMeasurement(RuntimeError):
    """当前因子无法由本标注后端度量。

    典型场景：因子定义不符合标准三标签格式且未指定显式规格或旧版编译、
    定义没有可忠实表达的离散刻度、schema_model 判定不适合 Jev。调用方应把它当作
    明确失败处理，而不是回退到其他后端。
    """

    def __init__(self, factor_name: str, reason: str) -> None:
        super().__init__(f"因子 {factor_name!r} 无法用当前后端标注：{reason}")
        self.factor_name = factor_name
        self.reason = reason


@dataclass(frozen=True)
class MeasurementSpec:
    """一个因子的标注规格（对 Jev 原生 question 的受验证包装）。

    - jev_type="score"：criteria 为从低到高的 2..10 条级别描述；value_map 为
      等长的原因子刻度值（严格递增）。因子值 = Σ p(level)·value_map[level]
      （完整概率分布对原刻度的加权期望）；分布缺失/不完整时，仅当 value_map
      等距才允许从 score（级别位置期望）线性回退，见
      :meth:`resolve_score_factor_value`。
    - jev_type="noul"：criteria 可选 (true 描述, false 描述)。因子值为 yes
      概率本身（0..1，软测量）；若 value_map=(no 值, yes 值) 给出则线性映射。
    - jev_type="choice"：criteria 为 (选项名, 描述或 None) 序列（2..255 个）；
      choice_values 给出选项名到数值的映射，因子值 = 所选选项的映射值。

    source: "explicit"（调用方显式给出）、"parsed"（标准三标签定义纯解析）
    或 "compiled"（调用方显式启用 schema_model 后编译产出）。
    notes: 刻度映射说明，随元数据一起保存。
    """

    factor_name: str
    jev_type: str
    instructions: str
    criteria: Optional[Tuple[Any, ...]] = None
    value_map: Optional[Tuple[float, ...]] = None
    choice_values: Optional[Mapping[str, float]] = None
    source: str = "explicit"
    notes: str = ""
    gate: Optional["MeasurementSpec"] = None

    def validate(self) -> "MeasurementSpec":
        """校验规格内部一致性；不合法时抛 ValueError（含因子名）。"""
        if not isinstance(self.factor_name, str) or not self.factor_name:
            raise ValueError("factor_name 必须是非空字符串")
        if self.jev_type not in SUPPORTED_JEV_TYPES:
            raise ValueError(f"jev_type 必须是 {SUPPORTED_JEV_TYPES} 之一，得到 {self.jev_type!r}")
        if not isinstance(self.instructions, str) or not self.instructions.strip():
            raise ValueError("instructions 必须是非空字符串")
        if self.jev_type == "score":
            self._validate_score()
        elif self.jev_type == "noul":
            self._validate_noul()
        else:
            self._validate_choice()
        if self.gate is not None:
            if not isinstance(self.gate, MeasurementSpec):
                raise ValueError("gate 必须是 MeasurementSpec")
            if self.gate.jev_type != "noul":
                raise ValueError("缺失门必须是 noul")
            self.gate.validate()
        return self

    def _validate_score(self) -> None:
        levels = self.criteria
        if not isinstance(levels, tuple) or not (
            MIN_SCORE_LEVELS <= len(levels) <= MAX_SCORE_LEVELS
        ):
            raise ValueError(
                f"score 的 criteria 需要 {MIN_SCORE_LEVELS}..{MAX_SCORE_LEVELS} 条级别描述"
            )
        if not all(isinstance(level, str) and level.strip() for level in levels):
            raise ValueError("score 的每条 criteria 都必须是非空字符串")
        values = self.value_map
        if not isinstance(values, tuple) or len(values) != len(levels):
            raise ValueError("score 的 value_map 必须与 criteria 等长")
        if not all(_is_finite_number(v) for v in values):
            raise ValueError("score 的 value_map 必须全为有限数值")
        if any(values[i] >= values[i + 1] for i in range(len(values) - 1)):
            raise ValueError("score 的 value_map 必须严格递增（与 criteria 从低到高对应）")

    def _validate_noul(self) -> None:
        if self.criteria is not None:
            criteria = self.criteria
            if not (
                isinstance(criteria, tuple)
                and len(criteria) == 2
                and all(isinstance(c, str) and c.strip() for c in criteria)
            ):
                raise ValueError("noul 的 criteria 必须是 (true 描述, false 描述) 或 None")
        if self.value_map is not None:
            values = self.value_map
            if not (isinstance(values, tuple) and len(values) == 2):
                raise ValueError("noul 的 value_map 必须是 (no 值, yes 值)")
            if not all(_is_finite_number(v) for v in values):
                raise ValueError("noul 的 value_map 必须全为有限数值")

    def _validate_choice(self) -> None:
        options = self.criteria
        if not isinstance(options, tuple) or not (2 <= len(options) <= MAX_CHOICE_OPTIONS):
            raise ValueError(f"choice 的 criteria 需要 2..{MAX_CHOICE_OPTIONS} 个 (选项名, 描述或 None)")
        names = []
        for option in options:
            if not (
                isinstance(option, tuple)
                and len(option) == 2
                and isinstance(option[0], str)
                and option[0]
            ):
                raise ValueError("choice 的每个 criteria 项必须是 (选项名, 描述或 None)")
            names.append(option[0])
        if len(set(names)) != len(names):
            raise ValueError("choice 的选项名不能重复")
        values = self.choice_values
        if not isinstance(values, Mapping) or set(values) != set(names):
            raise ValueError("choice_values 必须恰好覆盖全部选项")
        if not all(_is_finite_number(v) for v in values.values()):
            raise ValueError("choice_values 的取值必须全为有限数值")

    def to_request_question(self) -> Dict[str, Any]:
        """转成 Jev systemone 请求中 questions[id] 的原生结构。"""
        self.validate()
        question: Dict[str, Any] = {"type": self.jev_type, "instructions": self.instructions}
        if self.jev_type == "score":
            question["criteria"] = list(self.criteria)
        elif self.jev_type == "choice":
            question["criteria"] = {name: description for name, description in self.criteria}
        elif self.jev_type == "noul" and self.criteria is not None:
            question["criteria"] = {"true": self.criteria[0], "false": self.criteria[1]}
        return question

    def to_request_questions(self) -> Dict[str, Dict[str, Any]]:
        """一次请求里的全部 question：可选缺失门 + 取值题。"""
        self.validate()
        if self.gate is None:
            return {self.factor_name: self.to_request_question()}
        return {
            self.factor_name + MENTION_SUFFIX: self.gate.to_request_question(),
            self.factor_name: self.to_request_question(),
        }

    def is_equidistant_value_map(self) -> bool:
        """value_map 是否等距（score 位置回退的唯一前提）。"""
        values = self.value_map
        if len(values) < 2:
            return True
        step = float(values[1]) - float(values[0])
        tolerance = 1e-9 * max(1.0, abs(step))
        return all(
            abs((float(values[i + 1]) - float(values[i])) - step) <= tolerance
            for i in range(len(values) - 1)
        )

    def resolve_score_factor_value(
        self,
        score: float,
        probabilities: Optional[Mapping[Any, Any]] = None,
    ) -> Tuple[float, str]:
        """把 Jev score 应答解析为因子值，返回 (value, value_policy)。

        - probabilities 覆盖全部级别且严格归一：
          value = Σ p(level)·value_map[level]，policy=
          ``"score_probability_weighted_expectation"``。对非等距刻度这是
          唯一无损的映射（级别位置期望的插值 ≠ 原刻度期望，有偏）。
        - 分布缺失/不完整（部分级别缺失）：仅当 value_map 等距才允许从
          score 线性回退，policy=``"score_equidistant_position_fallback"``
          （前提：级别等距 ⇒ E[value] = value_map[0] + step·E[level]）；
          非等距时不得猜测，抛 ValueError。
        - 非法内容（未知/重复级别、负值、NaN/inf、完整但不归一）一律抛
          ValueError：绝不静默 clamp、绝不静默重归一。

        两者均应保留 raw 的原 score / probabilities / confidence 作元数据。
        """
        self.validate()
        n = len(self.value_map)
        weights = _validated_level_weights(probabilities, n)
        if weights is not None:
            value = sum(w * float(v) for w, v in zip(weights, self.value_map))
            return value, "score_probability_weighted_expectation"
        if self.is_equidistant_value_map():
            step = (float(self.value_map[-1]) - float(self.value_map[0])) / (n - 1)
            return (
                float(self.value_map[0]) + float(score) * step,
                "score_equidistant_position_fallback",
            )
        raise ValueError(
            f"因子 {self.factor_name!r} 的 value_map 非等距，且概率分布缺失或不完整；"
            "无法无损映射到原刻度，拒绝以级别位置猜测"
        )

    def noul_to_factor_value(self, yes_probability: float) -> float:
        """Jev yes 概率 -> 因子原刻度（无 value_map 时保持 0..1）。

        软测量：返回的是 yes 的概率（或其在 value_map 上的线性期望），
        不等价于硬标签标注；value_policy 记为 ``"noul_soft_yes_probability"``。
        """
        self.validate()
        p = _clamp01(float(yes_probability))
        if self.value_map is None:
            return p
        return _interpolate(p, self.value_map)

    def choice_to_factor_value(self, option: str) -> float:
        """Jev 所选选项 -> 因子数值（choice_values 映射）。"""
        self.validate()
        if option not in self.choice_values:
            raise UnsupportedMeasurement(self.factor_name, f"Jev 返回了规格中不存在的选项 {option!r}")
        return float(self.choice_values[option])

    def mapping_note(self) -> str:
        """供元数据记录的刻度映射说明。"""
        self.validate()
        if self.notes:
            return self.notes
        if self.jev_type == "score":
            return (
                f"Jev 级别概率对 value_map={list(self.value_map)} 加权期望映射回因子原刻度；"
                "分布缺失/不完整时仅当 value_map 等距才从级别位置线性回退"
            )
        if self.jev_type == "noul":
            if self.value_map is None:
                return "Jev yes 概率(0..1)直接作为因子值（软测量）"
            return f"Jev yes 概率经 value_map={list(self.value_map)}（no, yes）线性映射回因子刻度（软测量）"
        return f"Jev 所选选项经 choice_values={dict(self.choice_values)} 映射为因子值"


@dataclass(frozen=True)
class FactorAnnotation:
    """一次因子标注结果。

    默认解析路径的 value 是 choice 所选标签的映射值（硬标签），并非
    概率期望。可选 score/noul 才是软测量（分布期望 / yes 概率），与硬标签
    不等价；value_policy 明示测量模式。probabilities / confidence / raw
    是独立诊断元数据。raw 只含 Jev 所选标签或数值，不含被标注的原始文本。
    """

    factor_name: str
    value: float
    spec: MeasurementSpec
    raw: Mapping[str, Any]
    probabilities: Optional[Mapping[str, float]] = None
    confidence: Optional[float] = None
    sample_id: Optional[Any] = None
    backend_model: Optional[str] = None
    value_policy: Optional[str] = None

    def to_metadata(self) -> Dict[str, Any]:
        """无敏感原始文本的元数据快照（概率、置信度、测量模式与映射说明）。"""
        return {
            "factor": self.factor_name,
            "sample_id": self.sample_id,
            "value": float(self.value),
            "value_policy": self.value_policy,
            "backend_model": self.backend_model,
            "jev_type": self.spec.jev_type,
            "probabilities": dict(self.probabilities) if self.probabilities is not None else None,
            "confidence": self.confidence,
            "raw_answer": dict(self.raw),
            "value_mapping": self.spec.mapping_note(),
            "spec_source": self.spec.source,
        }


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _level_index(key: Any) -> int:
    """Jev 的级别键（JSON 中为字符串数字）-> int；非法即抛 ValueError。"""
    if isinstance(key, bool):
        raise ValueError(f"非法级别键 {key!r}")
    if isinstance(key, int):
        level = key
    elif isinstance(key, str) and key.strip().isdigit():
        level = int(key.strip())
    else:
        raise ValueError(f"非法级别键 {key!r}（需为级别序号的字符串或整数）")
    if level < 0:
        raise ValueError(f"非法级别键 {key!r}（级别从 0 开始）")
    return level


def _validated_level_weights(
    probabilities: Any, n: int
) -> Optional[Tuple[float, ...]]:
    """校验 Jev 级别概率分布，返回等长权重元组。

    返回 None 仅表示「缺失或不完整」（部分级别缺失），由调用方按等距前提
    决定是否回退；非法内容（未知/重复级别、负值、NaN/inf、完整但不归一）
    直接抛 ValueError，不做任何静默修补。
    """
    if probabilities is None:
        return None
    if not isinstance(probabilities, Mapping):
        raise ValueError("probabilities 必须是级别到概率的映射")
    if len(probabilities) == 0:
        return None
    weights: List[Optional[float]] = [None] * n
    seen = set()
    for key, probability in probabilities.items():
        level = _level_index(key)
        if level in seen:
            raise ValueError(f"probabilities 存在重复级别 {level}")
        if level >= n:
            raise ValueError(f"probabilities 含未知级别 {level}（合法范围 0..{n - 1}）")
        if not _is_finite_number(probability) or probability < 0:
            raise ValueError(f"级别 {level} 的概率非法（需为非负有限数值）")
        seen.add(level)
        weights[level] = float(probability)
    if any(w is None for w in weights):
        return None  # 不完整分布：交回调用方按等距前提处理
    total = sum(weights)
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"概率分布不归一（Σp={total:.9g}），拒绝静默归一")
    return tuple(weights)


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def _interpolate(position: float, values: Sequence[float]) -> float:
    """position ∈ [0, n-1] 在 values 上的分段线性插值（两端截断）。"""
    n = len(values)
    if n == 1:
        return float(values[0])
    if position <= 0.0:
        return float(values[0])
    if position >= n - 1:
        return float(values[-1])
    lower = int(math.floor(position))
    fraction = position - lower
    lower_value = float(values[lower])
    return lower_value + (float(values[lower + 1]) - lower_value) * fraction
