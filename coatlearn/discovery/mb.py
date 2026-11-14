"""
COAT-MB: COAT Markov Blanket Discovery Algorithm

该模块实现了从非结构化文本中发现 Markov Blanket 的核心算法。
参考：StressTests/mb/stress_test.py 的 SimpleCOATMB 类
"""

import sys
import json
import time
import logging
import itertools
import random
import numpy as np
import pandas as pd
from pathlib import Path
from copy import deepcopy
from typing import Dict, Set, List, Tuple, Any, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from sklearn.cluster import KMeans
from scipy import stats
from causallearn.utils.cit import CIT
from coatlearn.annotation.text import check_factor_overlapping, get_factor_name, get_value_from_responce
from coatlearn.causality.statistics import vif_or_inf

from unidecode import unidecode
import re


def default_annotation_batching(annotator):
    """Native Jev batches by default; unrelated annotator protocols stay opt-in."""
    from coatlearn.annotation.measurement import JevAnnotator
    return ({"enabled": True, "max_factors_per_request": 8,
             "shuffle_questions": False} if isinstance(annotator, JevAnnotator) else None)


class COATMB:
    """
    COAT Markov Blanket 发现算法（核心实现）

    功能：
    - 从文本数据中发现目标变量的 Markov Blanket
    - 使用 LLM 提议因子并标注
    - 使用条件独立性测试和 VIF 过滤因子
    """

    def __init__(
        self,
        llm_proposer,
        llm_annotator,
        target_name: str,
        meta_df: pd.DataFrame,
        text_col: str = "X",
        factor_quality_gate: Optional[List[Callable[[np.ndarray], bool]]] = None,
    ):
        """
        初始化 COAT-MB 算法

        Args:
            llm_proposer: LLM 提议器（用于提议新因子）
            llm_annotator: LLM 标注器（用于标注文本）
            target_name: 目标变量名称
            meta_df: 包含目标变量和文本的 DataFrame
            text_col: 文本列名（默认 'X'）
        """
        self.llm_proposer = llm_proposer
        self.llm_annotator = llm_annotator
        self.target_name = target_name
        self.meta_df = meta_df
        self.text_col = text_col
        self.factor_quality_gate = factor_quality_gate
        self.protected_factors = set()  # 保护因子（不允许被过滤掉）

        # 提取文本数据
        self.text_values = self.meta_df[self.text_col].values

        # 算法状态
        self.V = set()  # 当前 Markov Blanket
        self.annotation = {}  # 所有因子的标注值
        self.proposed_factors_history = []  # LLM 提议历史
        self.factor_pools = {}  # 因子池（存储所有历史因子及其描述）
        self.rejected_factors = {}  # 被质量检查拒绝的因子 {name: reason}
        self.unstable_factors = set()  # 出现过但未达到稳定性阈值的因子

        # 错误登记（供 facade 检查，不参与算法计算）：
        # - annotation_errors: {(factor_name, sample_idx): "失败原因"}
        #   仅登记真实故障（API/adapter 异常、annotate_factor 返回非数值类型）；
        #   普通缺失标注（解析不到数值、NaN/Inf）不登记，保持 NaN 交由质量门控。
        # - proposal_errors: {"iter_<i>/cluster_<j>" | "scoring/factor_<k>": "失败原因"}
        # - judge_errors: {"iter_<i>/<factor_name>": "LLM-judge 失败原因（fail-open 默认通过时也在此登记）"}
        self.annotation_errors = {}
        self.proposal_errors = {}
        self.judge_errors = {}

        # 初始化目标变量
        self.annotation[self.target_name] = self.meta_df[self.target_name].values
        self.protected_factors.add(self.target_name)

        # 迭代计数
        self.iteration = 0
        self.factor_judge = None
        self.factor_dedup = None
        self.factor_dedup_context = None
        self.factor_prediction_filter = None  # shared by ADJ and its MB/PAG children
        self._proposal_contexts = {}  # exact definition -> original examples + audit event id
        self.factor_audit_records = []  # local audit only; never copied into public results

        # 配置参数（默认值，可通过 set_config 修改）
        self.config = {
            "num_workers": 16,
            "vif_threshold": 2.0,
            "alpha": 0.01,
            "test_method": "kci",
            "prompt_sample_size": 5,
            "save_llm_conversation": False,
            "save_factor_audit": True,
            "factor_judge": None,  # opt-in {backend: "jev", ...JevConfig fields}
            "factor_dedup": None,  # opt-in simple matching + forward-only Jev
            "annotation_batching": default_annotation_batching(llm_annotator),
            "fci_depth": -1,
            "proposal_top_k": 3,
            "stability_B": 10,
            "stability_pi": None,
            "score_threshold": 5,
        }

    def set_config(self, **kwargs):
        """更新配置参数；Jev 审核与去重分别 opt-in，未启用时保留旧路径。"""
        if "stability_pi" in kwargs and kwargs["stability_pi"] is not None:
            pi = kwargs["stability_pi"]
            if not isinstance(pi, (int, float)) or not (0 < pi < 1):
                raise ValueError(f"stability_pi must be a float in (0, 1), got {pi}")
        if "save_factor_audit" in kwargs and not isinstance(kwargs["save_factor_audit"], bool):
            raise ValueError("save_factor_audit must be a bool")
        if kwargs.get("factor_template") not in (None, "simple", "general"):
            raise ValueError("factor_template must be 'simple' or 'general'")
        if "factor_judge" in kwargs:
            from coatlearn.annotation.judge import configured_judge

            template = kwargs.get("factor_template") or self.config.get("factor_template", "simple")
            judge = configured_judge(kwargs["factor_judge"], template)
            self.factor_judge = judge
            self.config["factor_judge_policy"] = judge.policy() if judge else None
        if "factor_dedup" in kwargs:
            from coatlearn.annotation.dedup import configured_deduplicator, core_dedup_policy

            comparator, context = configured_deduplicator(kwargs["factor_dedup"])
            self.factor_dedup, self.factor_dedup_context = comparator, context
            self.config["factor_dedup_policy"] = core_dedup_policy(comparator, context)
        if "observed_factor_definitions" in kwargs:
            from coatlearn.annotation.prediction import validate_observed_definitions
            validate_observed_definitions(kwargs["observed_factor_definitions"])
        self.config.update(kwargs)

    def _filter_prediction_candidates(self, names):
        if self.factor_prediction_filter is None:
            return set(names)
        return set(self.factor_prediction_filter.filter(
            sorted(names), self.annotation, self.factor_pools))

    def _audit_factor_event(self, kind, output_dir=None, **payload):
        """Persist a collision-free local event independently of full conversations."""
        from uuid import uuid4

        event = {"event_id": uuid4().hex, "kind": kind, "target": self.target_name,
                 "iteration": self.iteration, **payload}
        self.factor_audit_records.append(event)
        if output_dir and self.config.get("save_factor_audit"):
            directory = Path(output_dir) / "factor_audit"
            directory.mkdir(parents=True, exist_ok=True)
            with (directory / f"{event['event_id']}.json").open("x", encoding="utf-8") as handle:
                json.dump(event, handle, ensure_ascii=False, indent=2, allow_nan=False)
        return event["event_id"]

    def _jev_judge_factor(self, definition, output_dir=None):
        """A valid rejection gates annotation; technical errors are separate and fail closed."""
        name = get_factor_name(definition)
        context = self._proposal_contexts.get(definition)
        if context is None:
            result = {"status": "error", "decision": "ERROR", "error_type": "MissingProposalContext"}
        else:
            result = self.factor_judge.evaluate(name, definition, context["examples"])
        self._audit_factor_event("judge", output_dir, factor=name, definition=definition,
                                 proposal_id=context["proposal_id"] if context else None,
                                 policy=self.factor_judge.policy(), result=result)
        if result["status"] == "error":
            self.judge_errors[f"iter_{self.iteration}/{name}"] = result["error_type"]
            return True, "error", result["error_type"]
        rejected = result["decision"] == "REJECT"
        checks = ", ".join(result.get("rejected_checks", []))
        logging.info(f"  ⚖ Jev judge: {name}: {result['decision']} ({checks})")
        return rejected, "quality" if rejected else "", checks

    def _deduplicate_candidate(self, definition, output_dir=None, pending=None):
        """Simple names first, then one existing-A/new-B comparison per valid pair.

        Only measured, non-rejected representatives can block annotation. A
        same-batch candidate joins this set only AFTER its numerical gate, so
        a rejected rubric cannot permanently shadow a later version.

        ``pending`` lists (name, definition) pairs already accepted for the
        current annotation batch. Batched annotation defers the numerical gate
        until the batch completes, so pending definitions are compared here to
        keep in-round duplicate detection unchanged.
        """
        from coatlearn.annotation.dedup import normalize_factor_name

        name = get_factor_name(definition)
        context = self._proposal_contexts.get(definition, {})
        pending_definitions = dict(pending or [])
        existing_names = [n for n in self.annotation if not n.startswith("@ ")]
        existing_names += [n for n in pending_definitions if n not in existing_names]
        for existing in existing_names:
            if normalize_factor_name(name) == normalize_factor_name(existing):
                self._audit_factor_event(
                    "gate", output_dir, factor=name, definition=definition,
                    proposal_id=context.get("proposal_id"), gate="simple_name_match",
                    decision="SKIP", overlapping_with=existing,
                    existing_definition=self.factor_pools.get(existing))
                return True

        missing_definitions = []
        for existing in existing_names:
            old_definition = (self.factor_pools.get(existing)
                              or pending_definitions.get(existing))
            # Input numeric variables often have a name but no written rubric.
            # Do not fabricate a definition or ask Jev to infer one from a name.
            if not isinstance(old_definition, str) or "\n" not in old_definition:
                missing_definitions.append(existing)
                continue
            result = self.factor_dedup.evaluate(
                old_definition, definition, self.factor_dedup_context)
            event_id = self._audit_factor_event(
                "dedup", output_dir, factor=name, definition=definition,
                existing_factor=existing, existing_definition=old_definition,
                proposal_id=context.get("proposal_id"),
                policy=self.config["factor_dedup_policy"], result=result)
            if result["status"] == "error":
                self.judge_errors[f"dedup/iter_{self.iteration}/{name}/{existing}"] = result["error_type"]
            elif result["decision"] == "DUPLICATE":
                self._audit_factor_event(
                    "gate", output_dir, factor=name, definition=definition,
                    proposal_id=context.get("proposal_id"), gate="jev_concept_duplicate",
                    decision="SKIP", overlapping_with=existing,
                    comparison_event_id=event_id)
                return True
        self._audit_factor_event(
            "gate", output_dir, factor=name, definition=definition,
            proposal_id=context.get("proposal_id"), gate="concept_dedup",
            decision="RETAIN", unavailable_definitions=missing_definitions)
        return False

    def _make_ci_test(self, data, test_method, fci_depth=None):
        """
        构建 CI 测试函数，支持 conditioning set 大小限制
        注意：不再支持，这样做会有问题。

        Args:
            data: 数据矩阵
            test_method: 独立性测试方法
            fci_depth: 条件集最大子集大小，-1 表示不限制（使用全部条件变量）

        Returns:
            ci_test(x, y, cond_set) -> p_value
        """
        if fci_depth is None:
            fci_depth = self.config["fci_depth"]

        ci_test_func = CIT(data, test_method)

        return ci_test_func

        if fci_depth == -1:
            return ci_test_func

        def ci_test(x, y, cond):
            max_p = 0.0
            for subset in itertools.combinations(cond, min(fci_depth, len(cond))):
                p = ci_test_func(x, y, list(subset))
                max_p = max(max_p, p)
            return max_p

        return ci_test

    # ========================================================================
    # 核心算法流程
    # ========================================================================

    def select_samples_via_clustering(
        self,
        focus_factor: str = None,
        n_clusters: int = None,
        max_samples: int = 100,
        top_k: int = 3,
    ) -> List[np.ndarray]:
        """
        通过聚类选择高信息增益的样本

        Args:
            focus_factor: 目标因子名称（默认使用 target_name）
            n_clusters: 聚类数量（默认为当前 MB 大小）
            max_samples: 最大样本数（默认 100）
            top_k: 选取熵最高的 top_k 个 cluster

        Returns:
            List[chosen_indices]: 每个 cluster 对应的样本索引数组列表（按熵降序）
        """
        if focus_factor is None:
            focus_factor = self.target_name

        if n_clusters is None:
            n_clusters = max(1, len(self.V))  # 聚类数 = 当前 MB 大小

        # 获取背景因子
        background_factors = [f for f in self.V if f != focus_factor]
        if len(background_factors) == 0:
            chosen = np.random.choice(
                len(self.text_values),
                size=min(10, len(self.text_values)),
                replace=False,
            )
            return [chosen]

        # 获取条件向量和标签
        data_interface = pd.DataFrame(self.annotation).fillna(0)
        cond_vectors = data_interface[background_factors].values[:max_samples]
        focus_label = data_interface[focus_factor].values[:max_samples]

        # K-means 聚类
        g = (
            KMeans(n_clusters=n_clusters, random_state=0, n_init="auto")
            .fit(cond_vectors)
            .labels_
        )

        # 计算每个 cluster 的熵
        cluster_info = []
        set_g = list(set(g))

        for g_idx in set_g:
            cluster_labels = focus_label[g == g_idx]
            _, counts = np.unique(cluster_labels, return_counts=True)
            n = counts.sum()
            pk = counts / n

            if n < len(pk) * 3:
                this_entropy = -1
            else:
                this_entropy = stats.entropy(pk)

            cluster_info.append((g_idx, this_entropy, int(n)))

        # 按熵降序排列
        cluster_info.sort(key=lambda x: x[1], reverse=True)

        # 选 top_k 个有效 cluster（熵 > 0）
        valid_clusters = [c for c in cluster_info if c[1] > 0]
        selected = valid_clusters[:top_k]

        if len(selected) == 0:
            logging.warning("所有 cluster 熵为 0，回退到随机采样")
            return [np.arange(len(focus_label))]

        result_indices = []
        for g_idx, entropy_val, n_samples in selected:
            chosen_idx = np.arange(len(focus_label))[g == g_idx]
            result_indices.append(chosen_idx)
            logging.info(f"  cluster {g_idx}: entropy={entropy_val:.2f}, n={n_samples}")

        logging.info(f"✓ 聚类选择: top-{len(selected)}/{len(set_g)} clusters")

        return result_indices

    def generate_proposal_prompt(
        self, chosen_indices: np.ndarray, existing_factors: Set[str] = None
    ) -> str:
        """
        生成 LLM 因子提议 prompt

        Args:
            chosen_indices: 被选中的样本索引
            existing_factors: 已存在的因子（需要避免重复）

        Returns:
            格式化的 prompt
        """
        if existing_factors is None:
            existing_factors = self.V

        # 过滤掉 @ 前缀的低质量因子和 target_name
        existing_factors = {f for f in existing_factors if not f.startswith("@ ")}

        # 按目标变量值分组
        data = ""

        for f_v in np.sort(
            list(set(self.meta_df[self.target_name].values[chosen_indices]))
        ):
            data += f"\n\n## Group with '{self.target_name}' = {f_v}\n"
            these_indices = np.arange(len(chosen_indices))[
                (self.meta_df[self.target_name].values[chosen_indices] == f_v)
            ]

            # 每组最多选 3 个样本
            if len(these_indices) > 3:
                these_indices = np.random.choice(these_indices, 3, replace=False)

            for idx in these_indices:
                actual_idx = chosen_indices[idx]
                text = self.meta_df.iloc[actual_idx][self.text_col]

                # 清理文本
                if isinstance(text, str):
                    text = text.strip()
                else:
                    text = str(text)

                data += f"\n- {text}"

        # 构建已有因子描述
        existing_factors_section = ""
        observed_definitions = self.config.get("observed_factor_definitions") or {}
        existing_factors = set(existing_factors) | set(observed_definitions)
        if existing_factors:
            existing_descriptions = []
            for fname in sorted(existing_factors):
                if fname in observed_definitions:
                    # T/Y remain visible even when one is the current MB target.
                    existing_descriptions.append(observed_definitions[fname])
                elif fname == self.target_name:
                    continue
                elif fname in self.factor_pools:
                    existing_descriptions.append(self.factor_pools[fname])
                else:
                    existing_descriptions.append(fname)
            if existing_descriptions:
                factors_text = "\n\n".join(existing_descriptions)
                existing_factors_section = f"""
# Existing Factors (DO NOT propose factors that are similar or overlapping with these)

{factors_text}
"""

        # 构建被拒绝因子列表（反面教材）
        rejected_section = ""
        if self.rejected_factors:
            rejected_items = []
            for rname, reason in self.rejected_factors.items():
                if reason not in ("Imbalanced", "Low valid rate"):
                    continue
                clean_name = rname.lstrip("@ ").strip()
                definition = self.factor_pools.get(rname, clean_name)
                rejected_items.append(
                    f"- **{clean_name}**: {reason}\n  Definition: {definition}"
                )
            rejected_text = "\n".join(rejected_items)
            rejected_section = f"""
# Previously Rejected Factors (learn from these failures, do NOT propose similar ones)

These factors were proposed before but failed during annotation due to poor definition quality:
{rejected_text}
"""

        if (getattr(self, "config", None) or {}).get("factor_template") == "general":
            return self._general_proposal_prompt(
                data, existing_factors_section, rejected_section
            )

        prompt = f"""
You are an excellently helpful AI assistant for analyzing, abstracting, and processing data. Now try your best.

# Data

{data}
{existing_factors_section}{rejected_section}
# Tasks: Factor Abstraction.

**What are the high-level factors behind the text that contribute to the allocation of groups?**

- Propose your candidate factors based on your observation on given data.
- The proposed factors should be diverse and different semantically. Their overlapping should be minimized.
- **IMPORTANT**: Do NOT propose factors that are semantically similar to or overlapping with the existing factors listed above. Propose genuinely new and distinct factors.
- **IMPORTANT**: Do NOT propose factors similar to the rejected ones above. They failed because their criteria were too vague, ambiguous, or could not be consistently applied. Make sure your criteria are sharp, concrete, and observable.

# About Output

Your output should contain parts described as follows.

**Part 1**: Consideration.

In this part, feel free to write down the process of your considerations.
Hint: You need to abstract, identify, and design suitable factors with corresponding criteria, and each factor is only allowed to take value from [-1, 0, 1].

**Part 2**: Final Output.

In this part, you are required to report the factors that you decided to use.
- Each factors' value should be from **[-1, 0, 1]**
- **You should give specific criterion for each value of each factor.**
- Keep the criterion for value 0 fixed as "Otherwise; or not mentioned".
- **Factor Naming Rules**:
  - Each factor name must be **concise and highly summarizing** (2-5 words).
  - **Do NOT use** parentheses `()`, brackets `[]`, braces `{{}}`, semicolons `;`, slashes `/` or `\\`, or any special punctuation in factor names.
  - Use only plain words separated by spaces. Avoid long descriptive names.

Report the factors **in following template:**

```
**Factor Name**
- 1: [Positive Criterion]
- 0: [Otherwise; or not mentioned].
- -1: [Negative Criterion]
```
"""
        return prompt

    def _general_proposal_prompt(self, data, existing_factors_section, rejected_section) -> str:
        """General template: the proposal chooses score/choice and a mention gate."""
        return f"""
You are an excellently helpful AI assistant for analyzing, abstracting, and processing data. Now try your best.

# Data

{data}
{existing_factors_section}{rejected_section}
# Tasks: Factor Abstraction.

**What are the high-level factors behind the text that contribute to the allocation of groups?**

- Propose candidate factors based on the given data.
- The proposed factors should be diverse and semantically distinct.
- **IMPORTANT**: Do NOT propose factors similar to the existing or rejected ones above.
- For each factor choose the measurement type that fits the attribute:
  - `score`: an ordered attribute with 2 to 10 levels. Describe every level from low to
    high, and give each level's numeric value in `Values`. Numeric attributes (spending
    amount, recency in months, number of items) must use `score`, with values that keep
    the magnitude instead of collapsing to three labels.
  - `choice`: an unordered categorical attribute. Give every category and its numeric
    value in `Values`.
- **Do NOT put "not mentioned" into a value.** Instead, write in `Mentioned when` the
  condition under which the text explicitly mentions this attribute. Text that does not
  mention it is treated as missing, not as a level.

# About Output

**Part 1**: Consideration.

Write down how you identified the factors and why each measurement type fits.

**Part 2**: Final Output.

Report each factor in exactly one of these templates:

```
**Factor Name**
- Type: score
- Values: 0, 50, 150, 400, 1000
- Mentioned when: text states a spending amount
- 0: no spending, or under $25
- 1: $25 to $99
- 2: $100 to $299
- 3: $300 to $699
- 4: $700 or more
```

```
**Factor Name**
- Type: choice
- Values: urban=1, suburban=-1, rural=-1
- Mentioned when: text states where the customer lives
- urban: text states living in a city or urban area
- suburban: text states living in a suburb
- rural: text states living in a rural area or small town
```

**Factor Naming Rules**:
- Each factor name must be concise (2-5 words), plain words separated by spaces.
- Do NOT use parentheses, brackets, braces, semicolons, slashes or special punctuation.
"""

    @staticmethod
    def _parse_general_factors(response: str) -> List[Dict[str, str]]:
        """Split a general-template response into name/definition blocks verbatim."""
        text = "\n".join(line for line in response.splitlines() if "```" not in line)
        lowered = text.lower()
        if "final output" in lowered:
            text = text[lowered.rfind("final output") + len("final output"):]
        factors = []
        for block in re.split(r"\n\s*\n", text):
            lines = [line.strip() for line in block.splitlines() if line.strip()]
            if len(lines) < 4:
                continue
            if not any(re.match(r"^[-+*•]?\s*type\s*:\s*(score|choice)\s*$", line, re.I)
                       for line in lines):
                continue
            name = " ".join(lines[0].strip().strip("*_` ").rstrip(":").split())
            if not name:
                continue
            factors.append({"name": name, "description": "\n".join([name] + lines[1:])})
        return factors

    def parse_llm_factor_response(self, response: str) -> List[Dict[str, str]]:
        """
        解析 LLM 的因子提议响应

        Args:
            response: LLM 的原始响应

        Returns:
            List of {"name": str, "description": str}
        """

        if (getattr(self, "config", None) or {}).get("factor_template") == "general":
            return self._parse_general_factors(response)

        # 统一处理
        # Unicode 规范化
        response = unidecode(response)
        ## 删除含有 ``` 的代码行
        response = "\n".join(
            [line for line in response.split("\n") if "```" not in line]
        )
        ## 取消markdown 加粗语法，把 ` **xxx** ` 变成 ` xxx `
        response = re.sub(r"\*\*(.*?)\*\*", r"\1", response, flags=re.DOTALL)
        ## 全小写
        response = response.lower()

        # 提取 <Factor Begin> 之后的内容
        for seperator in ["final output", "<factor begin>"]:
            if seperator in response:
                response = response.split(seperator)[-1]
                break

        # 分割成多个 factor
        factor_list = response.split("\n\n")

        # 过滤和清理
        parsed_factors = []
        for factor_str in factor_list:
            factor_str = factor_str.strip()

            # 基本过滤
            if len(factor_str) < 10:
                continue

            # 应该恰好是4行，否则过滤
            if factor_str.count("\n") != 3:
                continue

            # 检查是否有 1:, 0:, -1: 的标记
            if not ("1:" in factor_str and "-1:" in factor_str and "0:" in factor_str):
                continue

            # 提取因子名称（第一行）
            lines = factor_str.split("\n")
            name_line = lines[0]

            # 清理名称
            for i in range(20):
                if f"{i}." in name_line:
                    name_line = name_line.replace(f"{i}.", "").strip()
                if f"{i} " in name_line:
                    name_line = name_line.replace(f"{i} ", "").strip()
                if f"{i}:" in name_line:
                    name_line = name_line.replace(f"{i}:", "").strip()

            for s in [":", "*", "factor", "name", "`"]:
                if s in name_line:
                    name_line = name_line.replace(s, "").strip()

            name_line = name_line.lower()
            if name_line == "":
                continue

            # 验证名称至少包含一个字母或数字
            if len(name_line) > 0 and not any(c.isalnum() for c in name_line):
                continue

            # 组合描述
            description = "\n".join(lines[1:]) if len(lines) > 1 else ""

            if len(name_line) > 0:
                parsed_factors.append(
                    {"name": name_line, "description": name_line + "\n" + description}
                )

        return parsed_factors

    def _propose_single_cluster(
        self,
        cluster_idx: int,
        chosen_indices: np.ndarray,
        existing_factors: Set[str],
        output_dir: Optional[Path] = None,
    ) -> List[Dict[str, str]]:
        """
        对单个 cluster 生成 prompt 并调用 LLM 提议因子

        Args:
            cluster_idx: cluster 编号
            chosen_indices: 该 cluster 的样本索引
            existing_factors: 已有因子
            output_dir: 输出目录

        Returns:
            解析后的因子列表
        """
        from hashlib import sha256
        from coatlearn.annotation.judge import proposal_examples

        prompt = self.generate_proposal_prompt(chosen_indices, existing_factors)
        examples = proposal_examples(prompt)
        audit_context = {
            "cluster_idx": cluster_idx,
            "selected_row_indices": [int(i) for i in chosen_indices],
            "prompt_sha256": sha256(prompt.encode()).hexdigest(),
            "examples": examples,
            "existing_factors": {name: self.factor_pools.get(name, name) for name in sorted(existing_factors or self.V)},
        }

        if output_dir and self.config.get("save_llm_conversation"):
            prompt_dir = output_dir / "llm_prompts"
            prompt_dir.mkdir(parents=True, exist_ok=True)
            with open(
                prompt_dir
                / f"iter_{self.iteration}_cluster{cluster_idx}_proposal_prompt.txt",
                "w",
                encoding="utf-8",
            ) as f:
                f.write(prompt)

        try:
            response = self.llm_proposer.chat(prompt, return_text=True)
        except (RuntimeError, ValueError) as e:
            logging.warning(f"[cluster {cluster_idx}] LLM 调用失败: {e}")
            # 隐私约定：登记只携带异常类型，不带消息
            self.proposal_errors[
                f"iter_{self.iteration}/cluster_{cluster_idx}"
            ] = type(e).__name__
            self._audit_factor_event("proposal", output_dir, **audit_context,
                                     status="error", error_type=type(e).__name__,
                                     response=None, factors=[])
            return []

        if output_dir and self.config.get("save_llm_conversation"):
            prompt_dir = output_dir / "llm_prompts"
            with open(
                prompt_dir
                / f"iter_{self.iteration}_cluster{cluster_idx}_proposal_response.txt",
                "w",
                encoding="utf-8",
            ) as f:
                f.write(response)

        factors = self.parse_llm_factor_response(response)
        proposal_id = self._audit_factor_event(
            "proposal", output_dir, **audit_context, status="ok", response=response, factors=factors,
            definition_hashes={f["name"]: sha256(f["description"].encode()).hexdigest() for f in factors},
        )
        for factor in factors:
            self._proposal_contexts.setdefault(factor["description"], {"examples": examples, "proposal_id": proposal_id})
        logging.info(f"  cluster {cluster_idx}: {len(factors)} 个因子")
        return factors

    def propose_factors_with_llm(
        self, output_dir: Optional[Path] = None, existing_factors: Set[str] = None
    ) -> List[Dict[str, str]]:
        """
        使用 LLM 提出新因子（Top-K cluster 并行提议）

        Args:
            output_dir: 输出目录（用于保存 prompt 和 response）
            existing_factors: 已存在的因子

        Returns:
            List of {"name": str, "description": str}，去重后
        """
        logging.info(f"\n[步骤 1] 获取候选因子")

        # 1. 选择 Top-K 高信息增益 cluster
        top_k = self.config.get("proposal_top_k", 3)
        clusters_indices = self.select_samples_via_clustering(top_k=top_k)

        if len(clusters_indices) == 1:
            all_factors = self._propose_single_cluster(
                0, clusters_indices[0], existing_factors, output_dir
            )
        else:
            # 并行调用
            all_factors = []
            with ThreadPoolExecutor(max_workers=len(clusters_indices)) as executor:
                futures = {
                    executor.submit(
                        self._propose_single_cluster,
                        i,
                        indices,
                        existing_factors,
                        output_dir,
                    ): i
                    for i, indices in enumerate(clusters_indices)
                }
                # Requests still run concurrently. New dedup consumes results
                # in cluster order, never in response-completion order.
                ordered_futures = futures if self.factor_dedup is not None else as_completed(futures)
                for future in ordered_futures:
                    cluster_i = futures[future]
                    try:
                        all_factors.extend(future.result())
                    except Exception as e:
                        logging.warning(f"  cluster proposal 异常: {e}")
                        self.proposal_errors[
                            f"iter_{self.iteration}/cluster_{cluster_i}"
                        ] = type(e).__name__

        # New path removes only identical proposals here. Semantic comparison
        # happens once, before annotation, against quality-gated measured factors.
        seen_names = []
        seen_descriptions = {}
        seen_definitions = {}
        deduped_factors = []
        for f in all_factors:
            if self.factor_dedup is not None:
                previous = seen_definitions.get(f["description"])
                is_overlapping = previous is not None
                overlapping_with, overlap_type = previous, "exact definition"
                if is_overlapping:
                    self._audit_factor_event(
                        "gate", output_dir, factor=f["name"], definition=f["description"],
                        gate="cross_cluster_exact_definition", decision="SKIP",
                        overlapping_with=previous)
            else:
                is_overlapping, overlapping_with, overlap_type = check_factor_overlapping(
                    f["name"], seen_names, seen_descriptions, f["description"],
                )
            if is_overlapping:
                logging.info(
                    f"  ⊘ 跨cluster去重 ({overlap_type}): {f['name']} ~ {overlapping_with}"
                )
                continue
            seen_names.append(f["name"])
            seen_descriptions[f["name"]] = f["description"]
            seen_definitions[f["description"]] = f["name"]
            deduped_factors.append(f)

        logging.info(
            f"✓ LLM 总计提议 {len(all_factors)} 个因子，去重后 {len(deduped_factors)} 个:"
        )
        for factor in deduped_factors:
            logging.info(f"  - {factor['name']}")

        self.proposed_factors_history.append(deduped_factors)

        return deduped_factors

    def prompt_factor_annotation(self, factor_state: str, text: str) -> str:
        """
        生成因子标注 prompt

        Args:
            factor_state: 因子定义（如 "Size\n1: large\n-1: small"）
            text: 待标注的文本

        Returns:
            格式化的 prompt
        """
        prompt = f"""

You are an expert data annotation assistant. Your task is to extract or calculate strictly numeric values from text.

# Data

```
{text.strip()}
```

# Annotation Task
Evaluate the value of this factor on the given data.

```
{factor_state.strip()}
```

Examples:
Good: "The value is: 3.2"
Good: "The value is: -1"
Good: "The value is: 0"
Bad: "The value is: €3.2"
Bad: "The value is: -1K"

"""
        return prompt

    def _factor_numerical_quality_gate(
        self, answers: np.ndarray, this_factor_name: str,
        factor_definition: Optional[str] = None,
    ) -> Tuple[np.ndarray, str, str]:
        """Use declared ternary labels even if this sample omits one category.

        The optional definition preserves legacy numeric-only callers; parsing
        is local and does not call a model or rewrite the factor rubric.
        """

        # NA gate
        NA_GATE_THRESHOLD = 0.6
        # A mention-gated factor is expected to be absent from part of the text;
        # requiring 60% mentions would reject legitimate attributes. Its own
        # threshold is applied below on the observed cells only.
        GATED_OBSERVED_THRESHOLD = 0.3
        mask_for_numeric = ~np.isnan(answers)
        spec = None
        if factor_definition is not None:
            from coatlearn.annotation.types import UnsupportedMeasurement
            from coatlearn.annotation.definitions import factor_definition_to_spec

            try:
                spec = factor_definition_to_spec(this_factor_name, factor_definition)
            except UnsupportedMeasurement:
                spec = None
        gated = spec is not None and spec.gate is not None
        if (not gated) and mask_for_numeric.mean() < NA_GATE_THRESHOLD:
            logging.warning(
                f"  ⚠ 因子 {this_factor_name} 数据质量差（有效率={mask_for_numeric.mean():.1%}, 使用零向量替代并标记"
            )
            return (
                np.zeros_like(answers),
                f"@ {this_factor_name}",
                f"Low valid annotation rate ({mask_for_numeric.mean():.1%})",
            )

        # Constant gate
        if mask_for_numeric.sum() > 0 and answers[mask_for_numeric].std() == 0:
            logging.warning(f"  ⚠ 因子 {this_factor_name} 为常数, 使用零向量替代并标记")
            return (
                np.zeros_like(answers),
                f"@ {this_factor_name}",
                "All annotated values are identical (constant)",
            )

        if (getattr(self, "config", None) or {}).get("factor_template") == "general" and (
            spec is None or spec.gate is None
        ):
            # General mode must not silently accept a ternary or free-form
            # definition; the candidate is rejected instead of zero-filled.
            return (np.zeros_like(answers), f"@ {this_factor_name}",
                    "General template definition could not be parsed")

        # A general-template factor carries a mention gate: its missing cells are
        # real missing, never zero-filled. Only the ternary choice keeps the old
        # nonzero-rate gate and the legacy numeric path keeps its value heuristic.
        is_triple_value_factor = (
            spec is not None
            and spec.jev_type == "choice"
            and tuple(label for label, _ in spec.criteria) == ("1", "0", "-1")
        ) or (
            spec is None
            and set(answers[mask_for_numeric]).issuperset({-1, 0, 1})
            and (len(set(answers[mask_for_numeric])) < 10)
        )

        if gated and not is_triple_value_factor:
            # General score/choice: retain missing as NaN; no clipping or fill.
            observed_ratio = float(mask_for_numeric.mean())
            if observed_ratio < GATED_OBSERVED_THRESHOLD:
                logging.warning(
                    f"  ⚠ 因子 {this_factor_name} 有效标注比例不足（{observed_ratio:.1%}），使用零向量替代并标记"
                )
                return (
                    np.zeros_like(answers),
                    f"@ {this_factor_name}",
                    f"Low observed annotation rate ({observed_ratio:.1%})",
                )
            return (answers, this_factor_name, "")

        if is_triple_value_factor:
            # A declared {-1, 0, 1} factor remains ternary when a sign is absent.
            # Missing labels will be zero-filled, not counted as nonzero evidence.
            nonzero_ratio = (mask_for_numeric & (answers != 0)).mean()
            if nonzero_ratio < NA_GATE_THRESHOLD:
                logging.warning(
                    f"  ⚠ 因子 {this_factor_name} 非零标注比例不足（{nonzero_ratio:.1%}），使用零向量替代并标记"
                )
                return (
                    np.zeros_like(answers),
                    f"@ {this_factor_name}",
                    f"Low valid annotation rate ({nonzero_ratio:.1%})",
                )

            # # Balance gate
            # imbalance = balance_check(answers[answers != 0])
            # if imbalance > 10:
            #     logging.warning(
            #         f"  ⚠ 因子 {this_factor_name} 数据极度不均衡, 使用零向量替代并标记"
            #     )
            #     return (
            #         np.zeros_like(answers),
            #         f"@ {this_factor_name}",
            #         "Extremely imbalanced annotation values",
            #     )

        else:
            # Extreme-Value gate

            q1 = np.percentile(answers[mask_for_numeric], 5)
            q3 = np.percentile(answers[mask_for_numeric], 95)

            iqr = q3 - q1
            multiplier = 1.5
            lower_bound = (
                answers[mask_for_numeric][
                    answers[mask_for_numeric] >= q1 - multiplier * iqr
                ]
            ).min()
            upper_bound = (
                answers[mask_for_numeric][
                    answers[mask_for_numeric] <= q3 + multiplier * iqr
                ]
            ).max()

            _answers = np.clip(answers, lower_bound, upper_bound)
            if (answers != _answers).any():
                logging.warning(
                    f"  ⚠ 因子 {this_factor_name} 含有极端值, 极端值已被剪裁"
                )
                answers = _answers

        # Fillout NA
        if np.isnan(answers).sum() > 0:
            logging.warning(
                f"  ⚠ 因子 {this_factor_name} 含有少量缺失值, 缺失值已被填充为0"
            )
            answers = np.nan_to_num(answers, 0)

        # Apply custom quality gates if provided
        if self.factor_quality_gate is not None:
            for gate_func in self.factor_quality_gate:
                try:
                    passed = gate_func(answers)
                    if not passed:
                        logging.warning(
                            f"  ⚠ 因子 {this_factor_name} 未通过自定义质量检查, 使用零向量替代并标记"
                        )
                        return (
                            np.zeros_like(answers),
                            f"@ {this_factor_name}",
                            "Failed custom quality gate",
                        )
                except Exception as e:
                    logging.warning(f"  ⚠ 自定义质量检查函数执行失败: {e}, 跳过该检查")

        return (answers, this_factor_name, "")

    def score_factor_quality(self, factor: str) -> int:
        """
        对单个因子打分（1-10），评估其作为因果因子的潜力。
        分数越高表示因子越有价值（定义清晰、因果相关性高、可区分）。

        Args:
            factor: 因子完整字符串（name + criteria）

        Returns:
            1-10 的整数分数，异常时返回 5（中位默认分）
        """
        prompt = f"""You are an expert evaluator for factor definitions. Score the following factor on a 1-10 scale based ONLY on the quality of its definition.

# Factor to evaluate:
{factor}

# Scoring criteria (1=worst, 10=best):
- **Clarity**: Are the criteria for values 1 and -1 sharp, specific, and unambiguous? Can a reasonable annotator consistently assign the same value given the same text?
- **Completeness**: Do both 1 and -1 have explicit, meaningful criteria (not placeholders)?
- **Mutual exclusivity**: Are the boundaries between 1, -1, and 0 clearly separated with no ambiguous overlap?
- **Specificity**: Do the criteria describe concrete, observable text features rather than vague abstractions?

Note: The value-0 criterion "Otherwise; or not mentioned" is a fixed template requirement, NOT a quality defect. Do NOT penalize it.

# Output (JSON only, no other text):
```json
{{"score": <integer 1-10>}}
```"""

        try:
            response = self.llm_proposer.chat(prompt, return_text=True)
            json_match = re.search(r"\{[\s\S]*\}", response)
            if json_match:
                result = json.loads(json_match.group())
                score = result.get("score", 5)
                return max(1, min(10, int(score)))
            return 5
        except Exception as e:
            logging.warning(f"  ⚠ 因子打分异常，返回默认分: {e}")
            return 5

    def score_factors_parallel(
        self, factor_list: List[str], num_workers: int = 10, num_trials: int = 3
    ) -> List[Tuple[str, float]]:
        """
        并行对所有因子打分（每因子 num_trials 次取平均），返回按分数降序排列的列表。

        Args:
            factor_list: 因子描述列表
            num_workers: 并行线程数
            num_trials: 每个因子独立打分次数，取平均

        Returns:
            按分数降序排列的 [(factor, avg_score), ...]
        """
        logging.info(
            f"\n[预评分] 并行评估 {len(factor_list)} 个因子质量 (每因子 {num_trials} 次取平均)..."
        )

        scores: List[Tuple[str, float]] = [("", 0.0)] * len(factor_list)

        def _score_one(idx):
            trial_scores = []
            for _ in range(num_trials):
                try:
                    s = self.score_factor_quality(factor_list[idx])
                    trial_scores.append(s)
                except Exception as e:
                    trial_scores.append(5)
                    self.judge_errors[f"scoring/factor_{idx}"] = type(e).__name__
            avg = sum(trial_scores) / len(trial_scores)
            scores[idx] = (factor_list[idx], avg)

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(_score_one, i) for i in range(len(factor_list))]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception:
                    pass

        sorted_factors = sorted(scores, key=lambda x: x[1], reverse=True)

        for factor_str, score in sorted_factors:
            fname = get_factor_name(factor_str)
            logging.info(f"  📊 {fname}: {score:.1f}/10")

        return sorted_factors

    def _llm_judge_once(
        self,
        new_factor: str,
        existing_factors: Dict[str, str],
    ) -> Tuple[bool, str, str, float]:
        """
        单次 LLM-as-judge 调用，返回 (should_reject, reason_category, detail, confidence)。
        """
        existing_block = ""
        if existing_factors:
            lines = []
            for fname, fdef in existing_factors.items():
                lines.append(f"- **{fname}**:\n{fdef}")
            existing_block = "\n\n".join(lines)
        else:
            existing_block = "(None)"

        prompt = f"""You are a strict quality judge for causal factor definitions. Evaluate the following NEW factor against EXISTING factors.

# EXISTING factors:
{existing_block}

# NEW factor to evaluate:
{new_factor}

# Factor definition template (for your reference)

Each factor follows this fixed template:
```
**Factor Name**
- 1: [Positive Criterion]
- 0: Otherwise; or not mentioned.
- -1: [Negative Criterion]
```
The value-0 criterion is **always** fixed as "Otherwise; or not mentioned" — this is a template requirement, NOT a quality defect. Do NOT penalize a factor for having a generic 0-value definition.

# Evaluation criteria:

## Criterion 1: Semantic Overlap
Two factors overlap ONLY if there is a **by-definition one-to-one map** between the NEW factor's values {1, 0, -1} and the EXISTING factor's values. That is, knowing one factor's annotation value deterministically determines the other's value, by virtue of their written criteria alone. If the mapping is imperfect, probabilistic, or only holds for a subset of texts, they do NOT overlap. Mere relatedness, sub-case, or generalization is NOT overlap.

## Criterion 2: Definition Quality
Check whether the NEW factor's **1** and **-1** criteria are concise, complete, and unambiguous:
- **Concise**: The criteria for values 1 and -1 should be clear and not overly verbose.
- **Complete**: Both 1 and -1 have explicit, meaningful criteria (not placeholder text like "reserved for future use" or "not applicable").
- **Unambiguous**: A reasonable annotator would assign the same value given the same text. The boundaries between 1, -1, and the catch-all 0 should be sharp.

# Output format (JSON only, no other text):

```json
{{
  "analysis": "Step-by-step reasoning: analyze the by-definition mapping between the NEW and each EXISTING factor's values {1, 0, -1}. Show whether a deterministic one-to-one map exists.",
  "is_overlapping": true/false,
  "overlapping_with": "name of the existing factor it overlaps with, or empty string",
  "overlap_reason": "brief explanation of why they overlap, or empty string",
  "is_well_defined": true/false,
  "quality_issues": "brief description of quality problems, or empty string if well-defined"
}}
```"""

        try:
            response = self.llm_proposer.chat(prompt, return_text=True)
        except (RuntimeError, ValueError) as e:
            logging.warning(f"  ⚠ LLM-judge 调用失败，默认通过: {e}")
            self.judge_errors[
                f"iter_{self.iteration}/{get_factor_name(new_factor)}"
            ] = f"llm call failed (fail-open): {type(e).__name__}"
            return False, "", ""

        try:
            json_match = re.search(r"\{[\s\S]*\}", response)
            if not json_match:
                logging.warning(
                    f"  ⚠ LLM-judge 响应无法解析为 JSON，默认通过: {response[:100]}"
                )
                self.judge_errors[
                    f"iter_{self.iteration}/{get_factor_name(new_factor)}"
                ] = "response not parseable as JSON (fail-open)"
                return False, "", ""

            result = json.loads(json_match.group())

            is_overlapping = result.get("is_overlapping", False)
            is_well_defined = result.get("is_well_defined", True)

            if is_overlapping:
                overlapping_with = result.get("overlapping_with", "unknown")
                overlap_reason = result.get("overlap_reason", "")
                detail = f"overlaps with '{overlapping_with}': {overlap_reason}"
                return True, "overlap", detail

            if not is_well_defined:
                quality_issues = result.get("quality_issues", "")
                detail = f"definition quality issues: {quality_issues}"
                return True, "quality", detail

            return False, "", ""

        except Exception as e:
            logging.warning(f"  ⚠ LLM-judge 解析异常，默认通过: {e}")
            self.judge_errors[
                f"iter_{self.iteration}/{get_factor_name(new_factor)}"
            ] = f"parse exception (fail-open): {type(e).__name__}"
            return False, "", ""

    def llm_judge_factor(
        self,
        new_factor: str,
        existing_factors: Dict[str, str],
        output_dir: Optional[Path] = None,
        num_judges: int = 3,
    ) -> Tuple[bool, str, str]:
        """
        用 LLM-as-judge 检验因子质量（并行 num_judges 次按 confidence 加权投票）：

        1. 新因子是否与已有因子语义重叠
        2. 因子定义是否简洁、完备、无歧义

        Args:
            new_factor: 新因子完整字符串（name + criteria）
            existing_factors: {factor_name: factor_full_string} 已有因子字典
            output_dir: 输出目录（用于保存 prompt/response）
            num_judges: 并行 judge 次数

        Returns:
            (should_reject, reason_category, detail)
            reason_category: "overlap" | "quality" | ""
        """
        verdicts: List[Tuple[bool, str, str, float]] = [None] * num_judges

        def _judge_once(idx):
            verdicts[idx] = self._llm_judge_once(new_factor, existing_factors)

        with ThreadPoolExecutor(max_workers=num_judges) as executor:
            futures = [executor.submit(_judge_once, i) for i in range(num_judges)]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception:
                    pass

        for i, v in enumerate(verdicts):
            if v is None:
                verdicts[i] = (False, "", "")

        if output_dir and self.config.get("save_llm_conversation"):
            judge_dir = output_dir / "llm_prompts"
            judge_dir.mkdir(parents=True, exist_ok=True)
            new_factor_name = get_factor_name(new_factor)
            judge_log_path = (
                judge_dir
                / f"iter_{self.iteration}_judge_{new_factor_name}_verdicts.txt"
            )
            with open(judge_log_path, "w", encoding="utf-8") as f:
                for i, (rej, cat, det) in enumerate(verdicts):
                    f.write(
                        f"Judge {i + 1}: reject={rej}, category={cat}, detail={det}\n"
                    )

        best_reject_reason = ""
        best_reject_detail = ""

        for should_reject, reason, detail in verdicts:
            if should_reject:
                best_reject_reason = reason
                best_reject_detail = detail
                break

        verdict_summary = "; ".join(
            f"{'REJECT' if r else 'PASS'}" for r, _, _ in verdicts
        )
        logging.info(f"  ⚖ LLM-judge 投票: {verdict_summary}")

        if best_reject_detail != "":
            return True, best_reject_reason, best_reject_detail
        return False, "", ""

    @staticmethod
    def _repair_prompt(failures) -> str:
        blocks = []
        for index, (definition, reason) in enumerate(failures, 1):
            blocks.append(f"## Block {index}\n{definition}\nError: {reason}")
        return f"""You repair factor definitions that failed format validation.

# Failed definitions (verbatim)

{chr(10).join(blocks)}

# Required format

Each corrected factor follows exactly one of:

```
**Factor Name**
- Type: score
- Values: <n comma-separated numbers, low to high>
- Mentioned when: <when the text explicitly mentions this attribute>
- 0: <description of level 0>
- 1: <description of level 1>
```

```
**Factor Name**
- Type: choice
- Values: <option=number, ...>
- Mentioned when: <when the text explicitly mentions this attribute>
- <option>: <description>
```

The number of level lines must equal the number of Values. Level labels are
0, 1, 2, ... in order, or exactly the Values themselves. Do not add factors.
Return only the corrected definitions, one after another.
"""

    def _repair_general_definitions(self, factor_list, output_dir=None):
        """One batched repair call for every malformed definition in this round."""
        from coatlearn.annotation.types import UnsupportedMeasurement
        from coatlearn.annotation.definitions import general_definition_to_spec

        failures, valid = [], []
        for definition in factor_list:
            try:
                general_definition_to_spec(get_factor_name(definition), definition)
            except UnsupportedMeasurement as exc:
                failures.append((definition, exc.reason))
            else:
                valid.append(definition)
        if not failures:
            return factor_list
        error_type = None
        try:
            response = self.llm_proposer.chat(self._repair_prompt(failures), return_text=True)
        except Exception as exc:  # repair is best-effort; failures are audited
            response, error_type = "", type(exc).__name__
        fixed = {}
        for item in self._parse_general_factors(response or ""):
            try:
                general_definition_to_spec(item["name"], item["description"])
            except UnsupportedMeasurement:
                continue
            fixed[" ".join(item["name"].casefold().split())] = item["description"]
        merged, definition_links = [], []
        for definition in factor_list:
            key = " ".join(get_factor_name(definition).casefold().split())
            repaired = fixed.get(key, definition)
            merged.append(repaired)
            if repaired != definition:
                context = self._proposal_contexts.get(definition)
                if context is not None:
                    self._proposal_contexts.setdefault(repaired, dict(context))
                definition_links.append({"original_definition": definition,
                                         "repaired_definition": repaired,
                                         "proposal_id": context.get("proposal_id") if context else None})
        still_invalid = []
        for definition in merged:
            try:
                general_definition_to_spec(get_factor_name(definition), definition)
            except UnsupportedMeasurement:
                still_invalid.append(get_factor_name(definition))
        self._audit_factor_event(
            "repair", output_dir, failed=len(failures),
            repaired=len(merged) - len(still_invalid) - len(valid),
            still_invalid=still_invalid, error_type=error_type, definition_links=definition_links)
        logging.info(
            f"  ⟳ general 修复: 失败 {len(failures)}，修复后仍不合法 {len(still_invalid)}"
        )
        return merged

    def _factor_passes_gates(self, this_factor, this_factor_name, output_dir,
                             factor_rolling_history, pending=None) -> bool:
        """Run the definition gates; return True when annotation may start."""
        if self.config.get("factor_template") == "general":
            # General mode never falls back to the ternary template: a
            # definition without Type/Values/Mentioned when is dropped.
            from coatlearn.annotation.types import UnsupportedMeasurement
            from coatlearn.annotation.definitions import general_definition_to_spec

            try:
                general_definition_to_spec(this_factor_name, this_factor)
            except UnsupportedMeasurement as exc:
                logging.info(
                    f"  ⊘ general 模板不匹配，丢弃: {this_factor_name} — {exc.reason}"
                )
                self._audit_factor_event(
                    "gate", output_dir, factor=this_factor_name, definition=this_factor,
                    gate="general_template_format", decision="SKIP", reason=exc.reason)
                return False

        if self.factor_dedup is not None:
            if self._deduplicate_candidate(this_factor, output_dir, pending=pending):
                return False
        else:
            # Legacy fuzzy rules are unchanged when concept dedup is off.
            is_overlapping, overlapping_factor, overlap_type = check_factor_overlapping(
                this_factor_name,
                list(self.annotation.keys()) + factor_rolling_history,
                self.factor_pools,
                this_factor,
            )
            if is_overlapping:
                logging.info(
                    f"  ⊘ 跳过 {overlap_type}: {this_factor_name} ~ {overlapping_factor}"
                )
                self._audit_factor_event("gate", output_dir, factor=this_factor_name,
                                         definition=this_factor, gate="rule_overlap",
                                         decision="SKIP", overlapping_with=overlapping_factor)
                return False

        if self.factor_judge is not None:
            should_reject, reason, detail = self._jev_judge_factor(this_factor, output_dir)
        else:
            valid_existing_names = {
                k for k in set(list(self.V) + factor_rolling_history)
                if not k.startswith("@ ")
            }
            existing_for_judge = {
                k: v for k, v in self.factor_pools.items() if k in valid_existing_names
            }
            # Batched measurements are deferred, but legacy judges must still
            # see accepted definitions from this batch without sharing vectors.
            if pending:
                existing_for_judge.update(dict(pending))
            # If the new comparator owns dedup, legacy judges still check
            # definition quality but receive no second overlap comparison set.
            if self.factor_dedup is not None:
                existing_for_judge = {}
            should_reject, reason, detail = self.llm_judge_factor(
                this_factor, existing_for_judge, output_dir
            )
        if should_reject:
            logging.info(
                f"  ⊘ definition judge 未放行 ({reason}): {this_factor_name} — {detail}"
            )
            return False
        return True

    def _finalize_factor_annotation(self, this_factor, this_factor_name, answers,
                                    errors_this_factor, output_dir):
        """Apply the numerical gate, then store, save and audit one factor."""
        sample_size = len(self.text_values)
        if errors_this_factor:
            for _idx, _reason in errors_this_factor.items():
                self.annotation_errors[(this_factor_name, _idx)] = _reason
            logging.warning(
                f"  ⚠ 因子 {this_factor_name} 有 "
                f"{sum(1 for k in errors_this_factor if k != -1)}/{sample_size} "
                "个样本标注失败（相应样本保持 NaN，明细见 self.annotation_errors）"
            )

        answers = np.array(answers)
        answers, this_factor_name, reject_reason = (
            self._factor_numerical_quality_gate(
                answers, this_factor_name, factor_definition=this_factor
            )
        )
        backend_archive_id = None
        if reject_reason:
            self.rejected_factors[this_factor_name] = reject_reason
            archive = getattr(self.llm_annotator, "archive_rejected_factor", None)
            if self.factor_dedup is not None and callable(archive):
                # Workers have joined. Preserve the rejected batch before
                # releasing its auto-prepared spec for a future proposal.
                backend_archive_id = archive(
                    get_factor_name(this_factor), this_factor, reject_reason)

        self.annotation[this_factor_name] = answers
        self.factor_pools[this_factor_name] = this_factor

        artifact = None
        if output_dir:
            factor_df = pd.DataFrame({this_factor_name: answers})
            factors_dir = Path(output_dir) / "factors"
            factors_dir.mkdir(parents=True, exist_ok=True)
            factor_path = factors_dir / f"iter_{self.iteration}_{this_factor_name}.csv"
            if self.factor_dedup is not None and factor_path.exists():
                from uuid import uuid4
                factor_path = factor_path.with_name(f"{factor_path.stem}__{uuid4().hex}.csv")
            factor_df.to_csv(factor_path, index=False)
            artifact = str(factor_path.relative_to(output_dir))
        if self.factor_dedup is not None:
            context = self._proposal_contexts.get(this_factor, {})
            self._audit_factor_event(
                "annotation", output_dir, factor=get_factor_name(this_factor),
                definition=this_factor, stored_name=this_factor_name,
                proposal_id=context.get("proposal_id"), rows=sample_size,
                rejection_reason=reject_reason, backend_archive_id=backend_archive_id,
                artifact=artifact)
        return this_factor_name, answers

    def _annotate_in_batches(self, items, output_dir):
        """Annotate accepted (name, definition) pairs with capped batched requests.

        One request carries one text and up to ``max_factors_per_request``
        factors instead of one factor per request. Factor order is fixed by
        default. Explicit shuffle_questions=True uses a row-derived seed to
        shuffle factors reproducibly; mention/value pairs remain together.
        """
        options = self.config.get("annotation_batching") or {}
        cap = max(1, int(options.get("max_factors_per_request", 8)))
        shuffle = bool(options.get("shuffle_questions", False))
        seed = int(self.config.get("seed", 42))
        sample_size = len(self.text_values)
        record = self.llm_annotator.annotate_record
        answers = {name: [np.nan] * sample_size for name, _ in items}
        errors = {name: {} for name, _ in items}

        for start in range(0, len(items), cap):
            chunk = items[start:start + cap]
            # Resolve every definition once, before the pool: a definition that
            # cannot be parsed must fail that factor only, never the whole batch.
            specs = {}
            for name, definition in chunk:
                try:
                    specs[name] = self.llm_annotator.prepare_factor(name, definition)
                except Exception as exc:  # noqa: BLE001 - reported per factor
                    for idx in range(sample_size):
                        errors[name][idx] = type(exc).__name__
                    logging.warning(
                        f"    ⚠ 规格解析失败，跳过 {name}: {type(exc).__name__}")
            if not specs:
                continue

            def process_row(idx, specs=specs):
                order = list(specs)
                if shuffle and len(order) > 1:
                    random.Random(seed * 1000003 + idx).shuffle(order)
                ordered = {name: specs[name] for name in order}
                try:
                    result = record(self.text_values[idx], ordered, sample_id=idx)
                except Exception as exc:  # noqa: BLE001 - recorded per factor below
                    for name in specs:
                        errors[name][idx] = type(exc).__name__
                    return
                for name in specs:
                    annotation = result.get(name)
                    value = getattr(annotation, "value", None)
                    if (isinstance(value, (int, float, np.integer, np.floating))
                            and not isinstance(value, bool)):
                        value = float(value)
                        if np.isfinite(value):
                            answers[name][idx] = value

            with ThreadPoolExecutor(max_workers=self.config["num_workers"]) as executor:
                futures = [executor.submit(process_row, idx) for idx in range(sample_size)]
                for future in tqdm(as_completed(futures), total=sample_size,
                                   desc=f"    批 {start // cap + 1}",
                                   file=sys.stderr, ncols=80):
                    try:
                        future.result()
                    except Exception as exc:  # noqa: BLE001
                        logging.error(
                            f"    ✗ 批标注线程池非预期失败: {type(exc).__name__}")
            logging.info(
                f"  ✓ 批 {start // cap + 1}: {len(specs)} 因子 × {sample_size} 样本")
        return answers, errors

    def _annotate_factors_batched(self, sorted_factor_list, output_dir):
        """Two-phase annotation: gate every definition first, then batch annotate."""
        accepted = []
        factor_rolling_history = []
        annotated_results = {}

        def flush():
            if not accepted:
                return
            answers, errors = self._annotate_in_batches(accepted, output_dir)
            for name, definition in accepted:
                stored_name, values = self._finalize_factor_annotation(
                    definition, name, answers[name], errors[name], output_dir)
                annotated_results[stored_name] = values
            accepted.clear()

        for factor_idx, this_factor in enumerate(sorted_factor_list):
            this_factor_name = get_factor_name(this_factor)
            # A repeated name may have a changed rubric after numerical
            # rejection. Finalize/archive before deciding on its next version;
            # otherwise a pending duplicate could hide a legitimate reproposal.
            from coatlearn.annotation.dedup import normalize_factor_name
            if any(normalize_factor_name(name) == normalize_factor_name(this_factor_name)
                   for name, _ in accepted):
                flush()
            if not self._factor_passes_gates(
                    this_factor, this_factor_name, output_dir,
                    factor_rolling_history, pending=accepted):
                continue
            factor_rolling_history.append(this_factor_name)
            accepted.append((this_factor_name, this_factor))
            logging.info(
                f"  [{factor_idx + 1}/{len(sorted_factor_list)}] 待标注: {this_factor_name}")

        flush()
        logging.info(f"✓ 成功标注 {len(annotated_results)} 个因子")
        return annotated_results

    def annotate_factors_parallel(
        self, factor_list: List[str], output_dir: Optional[Path] = None
    ) -> Dict[str, np.ndarray]:
        """
        并行标注因子

        若 llm_annotator 提供 prepare_factor(factor_name, factor_definition) /
        annotate_factor(factor_name, factor_definition, text, *, sample_id=None)，
        则分别在线程池前调用一次 / 逐样本优先调用；否则沿用 .chat + 数值解析路线。
        真实故障（API/adapter 异常、annotate_factor 返回非数值类型）按
        (factor_name, sample_idx) 记录到 self.annotation_errors，不会被静默
        当作成功；普通缺失（.chat 路线解析不到数值、NaN/Inf 标注值）不登记，
        样本保持 NaN 交由质量门控处理（通过门控后缺失填 0，未通过则零向量
        替代并标记）。

        Args:
            factor_list: 因子描述列表
            output_dir: 输出目录

        Returns:
            Dict of {factor_name: annotation_values}
        """
        logging.info(f"\n[步骤 2] 并行标注 {len(factor_list)} 个因子")

        if self.factor_judge is not None:
            # The approved clarity/format judge replaces BOTH legacy quality
            # layers, including their score-based ordering and extra criteria.
            sorted_factor_list = list(factor_list)
        else:
            # Legacy default: pre-score and order by quality before judging.
            scored = self.score_factors_parallel(
                factor_list, num_workers=min(len(factor_list), 10)
            )
            score_threshold = self.config.get("score_threshold", 5)
            skipped = [(f, s) for f, s in scored if s < score_threshold]
            if skipped:
                for f_str, s in skipped:
                    fname = get_factor_name(f_str)
                    logging.info(
                        f"  ⊘ 跳过低分因子: {fname} (score={s:.1f} < {score_threshold})"
                    )
            sorted_factor_list = [f for f, s in scored if s >= score_threshold]
            if self.factor_dedup is not None:
                # Preserve quality rejection, but choose representatives in the
                # same stable proposal order as the Jev-quality-judge path.
                passed = set(sorted_factor_list)
                sorted_factor_list = [f for f in factor_list if f in passed]

        if self.config.get("factor_template") == "general" and self.config.get(
            "factor_template_repair", True
        ):
            sorted_factor_list = self._repair_general_definitions(
                sorted_factor_list, output_dir
            )

        annotated_results = {}
        factor_rolling_history = []

        batching = self.config.get("annotation_batching") or {}
        if (batching.get("enabled", bool(batching))
                and callable(getattr(self.llm_annotator, "annotate_record", None))):
            logging.info(
                f"  批处理标注：每批最多 {batching.get('max_factors_per_request', 8)} 个因子，"
                f"question 顺序{'随机打乱' if batching.get('shuffle_questions', False) else '固定'}")
            return self._annotate_factors_batched(sorted_factor_list, output_dir)

        for factor_idx, this_factor in enumerate(sorted_factor_list):
            this_factor_name = get_factor_name(this_factor)

            if not self._factor_passes_gates(
                    this_factor, this_factor_name, output_dir, factor_rolling_history):
                continue

            factor_rolling_history.append(this_factor_name)
            logging.info(
                f"  [{factor_idx + 1}/{len(factor_list)}] 标注: {this_factor_name}"
            )

            sample_size = len(self.text_values)
            answers = [np.nan] * sample_size
            errors_this_factor = {}

            # Jev 标注钩子（可选，见 SOURCE_MAP.md）：
            # 1) prepare_factor(factor_name, factor_definition)：每个因子的线程池启动前调用一次；
            # 2) annotate_factor(factor_name, factor_definition, text, *, sample_id=None)：
            #    存在时单样本标注优先走它并直接取返回的 float；不存在则保持原
            #    .chat + numeric parser 路线不变。
            prepare_factor = getattr(self.llm_annotator, "prepare_factor", None)
            if callable(prepare_factor):
                prepare_factor(this_factor_name, this_factor)
            annotate_factor = getattr(self.llm_annotator, "annotate_factor", None)
            use_annotate_factor = callable(annotate_factor)

            # 定义标注函数
            def process_annotation(idx):
                try:
                    if use_annotate_factor:
                        this_value = annotate_factor(
                            this_factor_name,
                            this_factor,
                            self.text_values[idx],
                            sample_id=idx,
                        )
                        if isinstance(
                            this_value, (int, float, np.integer, np.floating)
                        ) and not isinstance(this_value, bool):
                            this_value = float(this_value)
                        else:
                            errors_this_factor[idx] = (
                                "annotate_factor returned non-numeric "
                                f"{type(this_value).__name__}"
                            )
                            this_value = np.nan
                    else:
                        this_prompt = self.prompt_factor_annotation(
                            this_factor, self.text_values[idx]
                        )

                        this_response = self.llm_annotator.chat(
                            this_prompt, return_text=True
                        )

                        if output_dir and self.config.get("save_llm_conversation"):
                            factor_log_dir = (
                                output_dir
                                / "llm_prompts"
                                / f"iter_{self.iteration}_{this_factor_name}"
                            )
                            factor_log_dir.mkdir(parents=True, exist_ok=True)
                            prompt_file = factor_log_dir / f"idx_{idx:04d}_prompt.txt"
                            response_file = (
                                factor_log_dir / f"idx_{idx:04d}_response.txt"
                            )
                            with open(prompt_file, "w", encoding="utf-8") as f:
                                f.write(this_prompt)
                            with open(response_file, "w", encoding="utf-8") as f:
                                f.write(this_response)

                        this_value = get_value_from_responce(this_response)

                    if (
                        isinstance(this_value, (int, float, np.integer, np.floating))
                        and not isinstance(this_value, bool)
                    ):
                        this_value = float(this_value)
                        if np.isfinite(this_value):
                            answers[idx] = this_value
                        else:
                            # NaN/Inf 是普通缺失标注：保持 NaN 交由既有质量门控
                            # 处理（通过门控后缺失填 0，未通过则零向量替代并标记），
                            # 不登记 annotation_errors。
                            logging.debug(
                                f"    ⚠ 非finite标注值 idx={idx}: 保持 NaN 交由质量门控"
                            )
                    # else: 普通缺失（.chat 路线解析不到数值，或协议路线返回的
                    # NaN）——保持 NaN 交由质量门控，不登记 annotation_errors；
                    # adapter 返回非数值类型的契约违约已在上方登记。

                    time.sleep(0.1)
                except RuntimeError as e:
                    logging.error(f"    ✗ 标注 API 彻底失败 idx={idx}: {e}")
                    # 隐私约定：登记只携带异常类型，不带消息
                    errors_this_factor[idx] = "RuntimeError"
                except ValueError as e:
                    logging.warning(
                        f"    ⚠ 标注值或内容不可用 idx={idx}: {type(e).__name__}"
                    )
                    errors_this_factor[idx] = "ValueError"
                except Exception as e:
                    logging.warning(f"    ⚠ 标注失败 idx={idx}: {e}")
                    errors_this_factor[idx] = type(e).__name__

            # 并行标注
            with ThreadPoolExecutor(max_workers=self.config["num_workers"]) as executor:
                futures = [
                    executor.submit(process_annotation, idx)
                    for idx in range(sample_size)
                ]

                # 使用 tqdm 显示进度
                for future in tqdm(
                    as_completed(futures),
                    total=sample_size,
                    desc=f"    {this_factor_name}",
                    file=sys.stderr,
                    ncols=80,
                ):
                    try:
                        future.result()
                    except Exception as e:
                        # process_annotation 内部已捕获异常，这里属于非预期失败，
                        # 不允许静默吞掉后按全成功处理
                        logging.error(
                            f"    ✗ 标注线程池非预期失败 ({this_factor_name}): {e}"
                        )
                        errors_this_factor.setdefault(
                            -1, f"unexpected pool failure: {type(e).__name__}"
                        )

            this_factor_name, answers = self._finalize_factor_annotation(
                this_factor, this_factor_name, answers, errors_this_factor, output_dir)
            annotated_results[this_factor_name] = answers

        logging.info(f"✓ 成功标注 {len(annotated_results)} 个因子")
        return annotated_results

    def _select_mb_on_subsample(
        self,
        annoted_values: np.ndarray,
        annotated_name: List[str],
        vif_threshold: float,
        alpha: float,
        test_method: str,
    ) -> Set[str]:
        """
        在给定数据子集上执行 VIF pre-cleanup + in/out phase

        Args:
            annoted_values: 数据矩阵（已子采样行）
            annotated_name: 列名列表
            vif_threshold: VIF 阈值
            alpha: 显著性水平
            test_method: 独立性测试方法

        Returns:
            选择的因子名称集合（不含 target_name）
        """
        if test_method not in ["mv_fisherz", "chisq", "gsq"]:
            annoted_values = np.nan_to_num(annoted_values, nan=0.0)

        protected_indices = set(
            [0]
            + [
                i
                for i, name in enumerate(annotated_name)
                if name in self.protected_factors
            ]
        )
        candidate_indices = [
            i for i in range(1, len(annotated_name)) if i not in protected_indices
        ]

        while len(candidate_indices) > 1:
            all_check_indices = sorted(protected_indices) + candidate_indices
            X_temp = annoted_values[:, all_check_indices]
            if np.isfinite(X_temp).all(axis=1).sum() <= X_temp.shape[1] + 1:
                break  # not enough complete rows to judge collinearity
            vif_vals = {
                col_idx: vif_or_inf(X_temp, len(protected_indices) + rel_i)
                for rel_i, col_idx in enumerate(candidate_indices)
            }
            max_vif = max(vif_vals.values())
            if max_vif <= vif_threshold:
                break
            worst = max(vif_vals.items(), key=lambda x: x[1])[0]
            candidate_indices.remove(worst)

        kept_indices = sorted(protected_indices) + sorted(candidate_indices)
        sub_annotated_name = [annotated_name[i] for i in kept_indices]
        sub_annoted_values = annoted_values[:, kept_indices]
        if test_method not in ["mv_fisherz", "chisq", "gsq"]:
            sub_annoted_values = np.nan_to_num(sub_annoted_values, nan=0.0)

        ci_test = self._make_ci_test(sub_annoted_values, test_method)

        factor_indeces = []
        while True:
            candidate_stats = {}
            for node_index in range(1, len(sub_annotated_name)):
                if node_index not in factor_indeces:
                    test_indices = [0, node_index] + factor_indeces
                    this_vif = vif_or_inf(sub_annoted_values[:, test_indices], 1)
                    if this_vif <= vif_threshold:
                        p_value = ci_test(0, node_index, factor_indeces)
                    else:
                        p_value = 1.0
                    if p_value < alpha:
                        candidate_stats[node_index] = {
                            "p_value": p_value,
                            "vif": this_vif,
                        }
            if not candidate_stats:
                break
            min_p_node = min(candidate_stats.items(), key=lambda x: x[1]["p_value"])
            factor_indeces.append(min_p_node[0])

        node_to_delete = []
        remaining_nodes = list(enumerate(factor_indeces))
        protected_node_indices = [
            sub_annotated_name.index(f) for f in self.protected_factors
        ]

        while remaining_nodes:
            node_stats = {}
            for _, node_index in remaining_nodes:
                if node_index in protected_node_indices:
                    continue
                this_other_nodes = [
                    f
                    for f in factor_indeces
                    if (f != node_index) and (f not in node_to_delete)
                ]
                p_value = ci_test(0, node_index, this_other_nodes)
                if p_value >= alpha:
                    node_stats[node_index] = p_value
            if not node_stats:
                break
            max_p_node = max(node_stats.items(), key=lambda x: x[1])
            node_to_delete.append(max_p_node[0])
            remaining_nodes = [
                (i, n) for i, n in remaining_nodes if n not in node_to_delete
            ]

        factor_indeces = [0] + [f for f in factor_indeces if f not in node_to_delete]
        selected = set([sub_annotated_name[f] for f in factor_indeces])
        return selected - {self.target_name}

    def discover_mb_with_stability_selection(
        self,
        new_factors: Set[str],
        vif_threshold: Optional[float] = None,
        alpha: Optional[float] = None,
        test_method: Optional[str] = None,
        B: Optional[int] = None,
        pi: Optional[float] = None,
    ) -> Set[str]:
        """
        使用 Stability Selection 发现 MB

        每次采样一半数据，并行 B 次 MB 选择，
        选择出现频率 > pi 的变量。

        Args:
            new_factors: 新标注的因子集合
            vif_threshold: VIF 阈值
            alpha: 显著性水平
            test_method: 独立性测试方法
            B: 子采样次数（默认使用 config["stability_B"]）
            pi: 频率阈值（默认使用 config["stability_pi"]）

        Returns:
            稳定因子 + protected_factors + {target_name} 组成的集合
        """
        if vif_threshold is None:
            vif_threshold = self.config["vif_threshold"]
        if alpha is None:
            alpha = self.config["alpha"]
        if test_method is None:
            test_method = self.config["test_method"]
        if B is None:
            B = self.config["stability_B"]
        if pi is None:
            pi = self.config["stability_pi"]

        logging.info(f"\n[步骤 3] Stability Selection MB 发现 (B={B}, pi={pi})")
        logging.info(f"  当前 unstable factors: {self.unstable_factors}")

        # 将 unstable factors 加入到 selected_factors 中
        selected_factors = self._filter_prediction_candidates(
            self.V.union(new_factors).union(self.unstable_factors))
        logging.info(f"  包含 unstable 后的候选因子: {selected_factors}")
        annotated_name = (
            [self.target_name]
            + list(self.protected_factors - {self.target_name})
            + list(selected_factors.difference(self.protected_factors))
        )

        data_interface = pd.DataFrame(self.annotation)
        annoted_values = data_interface[annotated_name].values

        n_samples = annoted_values.shape[0]
        half_n = n_samples // 2

        all_selected: List[Optional[Set[str]]] = [None] * B

        def _run_one(b_idx):
            row_indices = np.random.choice(n_samples, size=half_n, replace=False)
            sub_values = annoted_values[row_indices, :]
            all_selected[b_idx] = self._select_mb_on_subsample(
                sub_values, annotated_name, vif_threshold, alpha, test_method
            )

        with ThreadPoolExecutor(max_workers=B) as executor:
            futures = [executor.submit(_run_one, b) for b in range(B)]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logging.warning(f"  ⚠ Stability selection 子任务异常: {e}")

        factor_counts: Dict[str, int] = {}
        for selected_set in all_selected:
            if selected_set is None:
                continue
            for f in selected_set:
                factor_counts[f] = factor_counts.get(f, 0) + 1

        total_runs = sum(1 for s in all_selected if s is not None)
        freq = {f: c / total_runs for f, c in factor_counts.items()}

        stable_factors = {f for f, fr in freq.items() if fr > pi}
        # 更新 unstable factors：出现过但未达到阈值的因子
        self.unstable_factors = {f for f, fr in freq.items() if 0 < fr <= pi}

        new_V = stable_factors | self.protected_factors | {self.target_name}

        logging.info(f"\n✓ Stability Selection 完成 (B={B}, pi={pi})")
        logging.info(f"  因子频率:")
        for f, fr in sorted(freq.items(), key=lambda x: x[1], reverse=True):
            mark = "✓" if fr > pi else ("~" if fr > 0 else "✗")
            logging.info(f"    {mark} {f}: {fr:.2f}")
        logging.info(f"  稳定因子 (>{pi}): {stable_factors}")
        logging.info(f"  不稳定因子 (0<freq≤{pi}): {self.unstable_factors}")
        logging.info(f"  当前 MB: {new_V}")

        return new_V

    def discover_mb_with_ci_test(
        self,
        new_factors: Set[str],
        vif_threshold: Optional[float] = None,
        alpha: Optional[float] = None,
        test_method: Optional[str] = None,
    ) -> Set[str]:
        """
        使用条件独立性测试发现 MB

        Args:
            new_factors: 新标注的因子集合
            vif_threshold: VIF 阈值（默认使用配置）
            alpha: 显著性水平（默认使用配置）
            test_method: 独立性测试方法（默认使用配置）

        Returns:
            发现的 MB 集合
        """
        if vif_threshold is None:
            vif_threshold = self.config["vif_threshold"]
        if alpha is None:
            alpha = self.config["alpha"]
        if test_method is None:
            test_method = self.config["test_method"]

        logging.info(f"\n[步骤 3] MB 发现（条件独立性测试）")

        # 准备数据
        selected_factors = self._filter_prediction_candidates(self.V.union(new_factors))
        annotated_name = (
            [self.target_name]
            + list(self.protected_factors - {self.target_name})
            + list(selected_factors.difference(self.protected_factors))
        )

        data_interface = pd.DataFrame(self.annotation)
        annoted_values = data_interface[annotated_name].values

        # 处理 NaN
        if test_method not in ["mv_fisherz", "chisq", "gsq"]:
            annoted_values = np.nan_to_num(annoted_values, nan=0.0)

        # VIF pre-cleanup: 在 'in' phase 前移除高度共线性的候选因子
        protected_indices = set(
            [0]
            + [
                i
                for i, name in enumerate(annotated_name)
                if name in self.protected_factors
            ]
        )
        candidate_indices = [
            i for i in range(1, len(annotated_name)) if i not in protected_indices
        ]

        while len(candidate_indices) > 1:
            all_check_indices = sorted(protected_indices) + candidate_indices
            X_temp = annoted_values[:, all_check_indices]
            if np.isfinite(X_temp).all(axis=1).sum() <= X_temp.shape[1] + 1:
                break  # not enough complete rows to judge collinearity
            vif_vals = {
                col_idx: vif_or_inf(X_temp, len(protected_indices) + rel_i)
                for rel_i, col_idx in enumerate(candidate_indices)
            }
            max_vif = max(vif_vals.values())
            if max_vif <= vif_threshold:
                break
            worst = max(vif_vals.items(), key=lambda x: x[1])[0]
            worst_name = annotated_name[worst]
            candidate_indices.remove(worst)
            logging.info(f"  [VIF pre-clean] 移除 {worst_name} (VIF={max_vif:.2f})")

        kept_indices = sorted(protected_indices) + sorted(candidate_indices)
        annotated_name = [annotated_name[i] for i in kept_indices]
        annoted_values = annoted_values[:, kept_indices]
        if test_method not in ["mv_fisherz", "chisq", "gsq"]:
            annoted_values = np.nan_to_num(annoted_values, nan=0.0)

        logging.info(
            f"  VIF pre-clean 后剩余候选: {[annotated_name[i] for i in range(1, len(annotated_name)) if i not in [annotated_name.index(n) for n in self.protected_factors]]}"
        )

        # 初始化 CI 测试
        ci_test = self._make_ci_test(annoted_values, test_method)

        # MB 发现：'in' phase（逐步加入）
        factor_indeces = []

        iteration_count = 0
        while True:
            iteration_count += 1
            candidate_stats = {}

            for node_index in range(1, len(annotated_name)):
                if node_index not in factor_indeces:
                    # VIF 检查
                    test_indices = [0, node_index] + factor_indeces
                    this_vif = vif_or_inf(annoted_values[:, test_indices], 1)

                    if this_vif <= vif_threshold:
                        # CI 测试
                        p_value = ci_test(0, node_index, factor_indeces)
                    else:
                        p_value = 1.0

                    if p_value < alpha:
                        candidate_stats[node_index] = {
                            "p_value": p_value,
                            "vif": this_vif,
                        }

                    logging.info(
                        f"    [in phase] {annotated_name[node_index]}: p={p_value:.4f}, vif={this_vif:.2f}"
                    )

            # 如果没有更多候选因子，停止
            if not candidate_stats:
                break

            # 选择 p-value 最小的因子
            min_p_node = min(candidate_stats.items(), key=lambda x: x[1]["p_value"])
            factor_indeces.append(min_p_node[0])
            logging.info(
                f"  ✓ 加入 MB: {annotated_name[min_p_node[0]]} (p={min_p_node[1]['p_value']:.4f}, vif={min_p_node[1]['vif']:.2f})"
            )

        # MB 发现：'out' phase（逐步移除）
        node_to_delete = []
        remaining_nodes = list(enumerate(factor_indeces))
        protected_node_indices = [
            annotated_name.index(f) for f in self.protected_factors
        ]

        while remaining_nodes:
            node_stats = {}
            for percent_counter, node_index in remaining_nodes:
                if node_index in protected_node_indices:
                    continue
                this_other_nodes = [
                    f
                    for f in factor_indeces
                    if (f != node_index) and (f not in node_to_delete)
                ]
                p_value = ci_test(0, node_index, this_other_nodes)

                if p_value >= alpha:
                    node_stats[node_index] = p_value

                logging.info(
                    f"    [out phase] {annotated_name[node_index]}: {'keep' if p_value < alpha else 'drop'}, p={p_value:.4f}"
                )

            if not node_stats:
                break

            # 移除 p-value 最大的因子
            max_p_node = max(node_stats.items(), key=lambda x: x[1])
            node_to_delete.append(max_p_node[0])
            remaining_nodes = [
                (i, n) for i, n in remaining_nodes if n not in node_to_delete
            ]
            logging.info(
                f"  ✗ 移除: {annotated_name[max_p_node[0]]} (p={max_p_node[1]:.4f})"
            )

        # 构建最终的 MB
        factor_indeces = [0] + [f for f in factor_indeces if f not in node_to_delete]
        new_V = set([annotated_name[f] for f in factor_indeces])

        logging.info(f"\n✓ MB 发现完成: {len(new_V) - 1} 个因子（不包括 target）")
        logging.info(f"  当前 MB: {new_V}")

        return new_V

    def retrieve_factors_from_pools(
        self,
        init_factors: Optional[Set[str]] = None,
        vif_threshold: Optional[float] = None,
        alpha: Optional[float] = None,
        test_method: Optional[str] = None,
    ) -> Set[str]:
        """
        从因子池检索相关因子

        在 init_factors 基础上执行 in-phase（逐步加入池候选）和 out-phase（逐步移除冗余）。
        out-phase 不删除 init_factors 中的因子。

        Args:
            init_factors: 初始因子集合（不参与out-phase删除），默认为空集
            vif_threshold: VIF 阈值（默认使用配置）
            alpha: 显著性水平（默认使用配置）
            test_method: 独立性测试方法（默认使用配置）

        Returns:
            init_factors ∪ 从因子池中检索到的因子集合
        """
        if vif_threshold is None:
            vif_threshold = self.config["vif_threshold"]
        if alpha is None:
            alpha = self.config["alpha"]
        if test_method is None:
            test_method = self.config["test_method"]
        if init_factors is None:
            init_factors = set()

        init_factors = self._filter_prediction_candidates(init_factors)
        self.V = self._filter_prediction_candidates(self.V)
        logging.info(f"\n从因子池检索因子 (init_factors={len(init_factors)})")

        pool_factors = {
            name for name in self.factor_pools
            if name in self.annotation and not name.startswith("@ ")
        }
        pool_factors = self._filter_prediction_candidates(pool_factors)
        exclude = init_factors | self.V | self.protected_factors
        candidate_factors = pool_factors - exclude

        if len(candidate_factors) == 0:
            logging.info(f"  因子池中无新候选因子")
            return init_factors | self.protected_factors | {self.target_name}

        logging.info(f"  候选因子数: {len(candidate_factors)}")

        annotated_name = (
            [self.target_name]
            + list(self.protected_factors - {self.target_name})
            + list(init_factors.difference(self.protected_factors))
            + list(candidate_factors)
        )

        data_interface = pd.DataFrame(self.annotation)
        annoted_values = data_interface[annotated_name].values

        if test_method not in ["mv_fisherz", "chisq", "gsq"]:
            annoted_values = np.nan_to_num(annoted_values, nan=0.0)

        init_node_indices = set(
            i
            for i, name in enumerate(annotated_name)
            if name in (init_factors | self.protected_factors | {self.target_name})
        )
        candidate_start = len(annotated_name) - len(candidate_factors)

        ci_test = self._make_ci_test(annoted_values, test_method)

        # in-phase：只检查池候选（索引 >= candidate_start），条件集包含 init 因子
        factor_indeces = []

        while True:
            candidate_stats = {}

            for node_index in range(candidate_start, len(annotated_name)):
                if node_index in init_node_indices or node_index in factor_indeces:
                    continue
                cond = sorted(init_node_indices - {0, node_index}) + factor_indeces
                test_indices = [0, node_index] + cond
                this_vif = vif_or_inf(annoted_values[:, test_indices], 1)

                if this_vif <= vif_threshold:
                    p_value = ci_test(0, node_index, cond)
                else:
                    p_value = 1.0

                if p_value < alpha:
                    candidate_stats[node_index] = {
                        "p_value": p_value,
                        "vif": this_vif,
                    }

                logging.info(
                    f"    [pool in-phase] {annotated_name[node_index]}: p={p_value:.4f}, vif={this_vif:.2f}"
                )

            if not candidate_stats:
                break

            min_p_node = min(candidate_stats.items(), key=lambda x: x[1]["p_value"])
            factor_indeces.append(min_p_node[0])
            logging.info(
                f"  ✓ 从因子池加入: {annotated_name[min_p_node[0]]} (p={min_p_node[1]['p_value']:.4f}, vif={min_p_node[1]['vif']:.2f})"
            )

        # out-phase：只删除非 init 的因子
        node_to_delete = []
        all_selected = factor_indeces
        remaining_nodes = list(enumerate(all_selected))

        while remaining_nodes:
            node_stats = {}
            for _, node_index in remaining_nodes:
                if node_index in init_node_indices:
                    continue
                this_other = [
                    f
                    for f in all_selected
                    if (f != node_index) and (f not in node_to_delete)
                ] + sorted(init_node_indices - {0})
                p_value = ci_test(0, node_index, this_other)

                if p_value >= alpha:
                    node_stats[node_index] = p_value

                logging.info(
                    f"    [pool out-phase] {annotated_name[node_index]}: "
                    f"{'keep' if p_value < alpha else 'drop'}, p={p_value:.4f}"
                )

            if not node_stats:
                break

            max_p_node = max(node_stats.items(), key=lambda x: x[1])
            node_to_delete.append(max_p_node[0])
            remaining_nodes = [
                (i, n) for i, n in remaining_nodes if n not in node_to_delete
            ]
            logging.info(
                f"    ✗ 移除: {annotated_name[max_p_node[0]]} (p={max_p_node[1]:.4f})"
            )

        kept = sorted(init_node_indices) + [
            f for f in all_selected if f not in node_to_delete
        ]
        result = set(annotated_name[f] for f in kept)

        retrieved_from_pool = (
            result - init_factors - self.protected_factors - {self.target_name}
        )
        logging.info(f"\n✓ 因子池检索完成: 新检索 {len(retrieved_from_pool)} 个因子")
        logging.info(f"  检索到的因子: {retrieved_from_pool}")
        logging.info(f"  最终结果: {result}")

        return result

    def single_round(
        self,
        output_dir: Optional[Path] = None,
        alpha: Optional[float] = None,
        test_method: Optional[str] = None,
        vif_threshold: Optional[float] = None,
        enforce_vif_check_names: Optional[List[str]] = None,
        just_from_pools: bool = False,
    ) -> Set[str]:
        """
        执行单轮 COAT 迭代

        Args:
            output_dir: 输出目录
            alpha: 显著性水平
            test_method: 独立性测试方法
            vif_threshold: VIF 阈值
            enforce_vif_check_names: VIF 检查的变量列表
            just_from_pools: 是否仅从因子池检索（不调用 LLM）

        Returns:
            更新后的 Markov Blanket (self.V)
        """
        self.iteration += 1
        logging.info(f"\n{'=' * 60}")
        logging.info(f"Iteration {self.iteration}")
        logging.info(f"{'=' * 60}")

        # 准备参数
        if vif_threshold is None:
            vif_threshold = self.config["vif_threshold"]
        if alpha is None:
            alpha = self.config["alpha"]
        if test_method is None:
            test_method = self.config["test_method"]

        # Excluded vectors stay in the pool for audit, never in CI/proposal feedback.
        self.V = self._filter_prediction_candidates(self.V)
        # 保存迭代开始时的 V
        old_V = deepcopy(self.V)

        # =====================================================================
        # 步骤 1: 从因子池前置检索（丰富 V，为 LLM propose 提供背景因子）
        # =====================================================================
        if (len(self.V) == 0) or (just_from_pools):
            pre_pool_result = self.retrieve_factors_from_pools(
                init_factors=self.V,
                vif_threshold=vif_threshold,
                alpha=alpha,
                test_method=test_method,
            )
            new_pool_factors = (
                pre_pool_result - self.protected_factors - {self.target_name}
            )
            if new_pool_factors:
                logging.info(f"\n[步骤 1] 前置因子池检索: {new_pool_factors}")
                self.V = self.V.union(new_pool_factors)
            else:
                logging.info(f"\n[步骤 1] 前置因子池检索: 无新因子")
        else:
            logging.info(f"\n[步骤 1] 前置因子池检索: 已有 V 不为空，跳过前置检索")

        # =====================================================================
        # 步骤 2: LLM 提议新因子
        # =====================================================================
        if not just_from_pools:
            proposed_factors_dict = self.propose_factors_with_llm(
                output_dir, existing_factors=self.V
            )
            proposed_factor_descriptions = [
                f["description"] for f in proposed_factors_dict
            ]

            if len(proposed_factor_descriptions) == 0:
                logging.info("\n⚠ LLM 没有提议新因子")
                proposed_factor_descriptions = []
        else:
            proposed_factor_descriptions = []

        # =====================================================================
        # 步骤 3: 并行标注新因子
        # =====================================================================
        if not just_from_pools and len(proposed_factor_descriptions) > 0:
            annotated_results = self.annotate_factors_parallel(
                proposed_factor_descriptions, output_dir
            )
            new_factors = set(annotated_results.keys())
        else:
            new_factors = set()

        # =====================================================================
        # 步骤 4: 相对于累计因子池重新估计 MB（论文 Algorithm 1, 2e）
        # =====================================================================
        # 去重仅省去重复标注，不能剥夺旧候选的统计检验机会。复用其原始
        # 名称/定义/向量，不把启发式判为相似的新定义冒充成旧定义。
        # 数值门控拒绝的 @ 因子、尚未标注的定义不作为有效池候选。
        pool_candidates = {
            name for name in self.factor_pools
            if name in self.annotation and not name.startswith("@ ")
        }
        candidates = pool_candidates | {
            name for name in new_factors if not name.startswith("@ ")
        }
        candidates = self._filter_prediction_candidates(candidates)
        if self.factor_prediction_filter is not None:
            report = self.factor_prediction_filter.report()
            self.config["factor_prediction_gate_policy"] = report["policy"]
            self._audit_factor_event("prediction_gate", output_dir, **report)
        logging.info(f"  累计池候选: {sorted(candidates)}")
        stability_pi = self.config.get("stability_pi", None)
        if stability_pi is not None:
            new_V = self.discover_mb_with_stability_selection(
                new_factors=candidates,
                alpha=alpha,
                test_method=test_method,
                vif_threshold=vif_threshold,
            )
        else:
            new_V = self.discover_mb_with_ci_test(
                new_factors=candidates, alpha=alpha, test_method=test_method,
                vif_threshold=vif_threshold,
            )

        self._audit_factor_event(
            "mb_update", output_dir,
            previous=sorted(old_V),
            candidates=sorted(self.V | candidates | self.protected_factors),
            selected=sorted(new_V),
            just_from_pools=just_from_pools,
        )
        # 覆盖当前 MB，而非历史并集；旧因子仍可被 VIF/CI 剔除。
        self.V = new_V
        logging.info(f"✓ V 已更新: {old_V} → {self.V}")

        return self.V
