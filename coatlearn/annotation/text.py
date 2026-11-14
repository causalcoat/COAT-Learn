"""Local parsing and name-overlap rules used by the discovery algorithms."""
import re
import numpy as np
from nltk.corpus import wordnet
from coatlearn.utils.constants import STOP_WORDS

def _ensure_wordnet_available():
    """
    显式检查 NLTK WordNet 语料是否可用。

    缺失时抛出带修复指引的 RuntimeError：库内绝不自动下载语料，
    也不允许把“词表缺失”静默当作相似度 0（那会把所有因子判为不重叠，
    使查重统计失去意义）。个别词不在词表中的情形仍按原逻辑返回 0.0。
    """
    try:
        wordnet.synsets("coat_wordnet_probe")
    except LookupError as exc:
        raise RuntimeError(
            "NLTK WordNet corpus is not available in this environment. "
            "Install it once with `python -m nltk.downloader wordnet` "
            "(the library never downloads corpora on its own). "
            "Refusing to silently treat a missing lexicon as zero similarity."
        ) from exc


def get_wordnet_similarity(word1, word2):
    """
    计算两个词的 WordNet 相似度
    """
    _ensure_wordnet_available()
    synsets1 = wordnet.synsets(word1)
    synsets2 = wordnet.synsets(word2)

    if not synsets1 or not synsets2:
        return 0.0

    max_sim = 0.0
    for syn1 in synsets1:
        for syn2 in synsets2:
            try:
                sim = syn1.path_similarity(syn2)
                if sim and sim > max_sim:
                    max_sim = sim
            except:
                continue
    return max_sim


def check_factor_overlapping(
    new_factor_name,
    existing_factor_names,
    factor_description=None,
    new_factor_description=None,
):
    """
    检查新因子名称是否与已有因子名称存在重叠或相似

    参数:
    new_factor_name (str): 新因子名称
    existing_factor_names (list): 已有因子名称列表

    返回:
    tuple: (是否有重叠, 重叠的因子名称, 重叠类型描述)
    """
    import re

    def name_preprocess(name):
        name = name.lower().strip()  # 转换为小写进行比较
        for connector in ["_", "-"]:  # 去除连接符
            name = name.replace(connector, " ")
        return name

    # 预处理新因子名称
    new_factor_name = name_preprocess(new_factor_name)

    # 1. 检查完全匹配
    for existing_name in existing_factor_names:
        existing_name = name_preprocess(existing_name)
        if new_factor_name == existing_name:
            return True, existing_name, "exact match"

    # 2. 检查子字符串关系
    for existing_name in existing_factor_names:
        existing_name = name_preprocess(existing_name)
        if len(new_factor_name) > 8 and len(existing_name) > 8:  # 避免短字符串误判
            if (new_factor_name in existing_name) or (existing_name in new_factor_name):
                return True, existing_name, "substring overlap"

    # 3. 检查单词重叠
    for existing_name in existing_factor_names:
        existing_name = name_preprocess(existing_name)
        words1 = set(re.findall(r"\b\w+\b", new_factor_name))
        words2 = set(re.findall(r"\b\w+\b", existing_name))
        words1 = words1 - STOP_WORDS
        words2 = words2 - STOP_WORDS

        if len(words1) > 0 and len(words2) > 0:
            intersection = words1.intersection(words2)

            if len(intersection) > 0:
                overlap_ratio = len(intersection) / min(len(words1), len(words2))
                if overlap_ratio > 0.8:  # 超过80%的有意义单词重叠
                    return True, existing_name, f"word overlap ({overlap_ratio:.2f})"

    # 4. 检查近义词相似度
    for existing_name in existing_factor_names:
        existing_name = name_preprocess(existing_name)
        words1 = set(re.findall(r"\b\w+\b", new_factor_name))
        words2 = set(re.findall(r"\b\w+\b", existing_name))

        words1 = words1 - STOP_WORDS
        words2 = words2 - STOP_WORDS

        if len(words1) > 0 and len(words2) > 0:
            # 计算词对的相似度
            similar_pairs = []
            for w1 in words1:
                for w2 in words2:
                    sim = get_wordnet_similarity(w1, w2)
                    if sim > 0.5:  # 设置相似度阈值
                        similar_pairs.append((w1, w2, sim))

            ratio_1 = len(set([pair[0] for pair in similar_pairs])) / len(words1)
            ratio_2 = len(set([pair[1] for pair in similar_pairs])) / len(words2)

            if max(ratio_1, ratio_2) > 0.8:
                return (
                    True,
                    existing_name,
                    f"semantic similarity ({max(ratio_1, ratio_2):.2f})",
                )

    # 5. 检查描述性相似度
    if (factor_description is not None) and (new_factor_description is not None):
        for existing_name in existing_factor_names:
            if existing_name not in factor_description.keys():
                continue
            _this_existing_description = factor_description[existing_name]

            words1 = set(re.findall(r"\b\w+\b", new_factor_description))
            words2 = set(re.findall(r"\b\w+\b", _this_existing_description))
            words1 = words1 - STOP_WORDS
            words2 = words2 - STOP_WORDS

            if len(words1) > 5 and len(words2) > 5:  # 避免短字符串误判
                # 计算词对的相似度
                similar_pairs = []
                for w1 in words1:
                    for w2 in words2:
                        sim = get_wordnet_similarity(w1, w2)
                        if sim > 0.5:  # 设置相似度阈值
                            similar_pairs.append((w1, w2, sim))

                ratio_1 = len(set([pair[0] for pair in similar_pairs])) / len(words1)
                ratio_2 = len(set([pair[1] for pair in similar_pairs])) / len(words2)

                if max(ratio_1, ratio_2) >= 0.8:
                    return (
                        True,
                        existing_name,
                        f"semantic similarity in description ({max(ratio_1, ratio_2):.2f})",
                    )

    return False, "", ""


def get_value_from_responce(responce):
    if responce is None:
        return "None"
    if "he value is:" not in responce:
        return "?"

    last_part = responce.lower().split("the value is")[-1]

    if "nan" in last_part:
        return np.nan

    for _ in ["*", ":", "$"]:
        last_part = last_part.replace(_, "")

    if "[" in last_part:
        last_part = last_part.split("[")[-1]
    if "]" in last_part:
        last_part = last_part.split("]")[0]

    try:
        last_part = float(last_part)
        return last_part
    except:
        pass

    return np.nan


def get_factor_name(this_factor):
    # 获取第一行并转为小写
    this_factor = this_factor.split("\n")[0].lower()

    # 移除 "factor X:" 格式的内容
    import re

    this_factor = re.sub(r"factor\s*(\d+|[a-zA-Z]+)\s*:", "", this_factor).strip()

    # 移除数字编号（如 "1.", "2."等）
    for i in range(20):
        if f"{i}." in this_factor:
            this_factor = this_factor.replace(f"{i}", "").strip()

    # 移除冒号
    for s in ":":
        if s in this_factor:
            this_factor = this_factor.replace(s, "").strip()

    # 移除括号及其内容
    while "(" in this_factor and ")" in this_factor:
        start = this_factor.find("(")
        end = this_factor.find(")")
        if start < end:
            this_factor = (
                this_factor[:start].strip() + " " + this_factor[end + 1 :].strip()
            )
        else:
            break

    # 移除方括号及其内容
    while "[" in this_factor and "]" in this_factor:
        start = this_factor.find("[")
        end = this_factor.find("]")
        if start < end:
            this_factor = (
                this_factor[:start].strip() + " " + this_factor[end + 1 :].strip()
            )
        else:
            break

    # 移除大括号及其内容
    while "{" in this_factor and "}" in this_factor:
        start = this_factor.find("{")
        end = this_factor.find("}")
        if start < end:
            this_factor = (
                this_factor[:start].strip() + " " + this_factor[end + 1 :].strip()
            )
        else:
            break

    # 移除额外的标点符号
    for punct in [
        ".",
        ",",
        ";",
        "!",
        "?",
        '"',
        "'",
        "`",
        "*",
        "#",
        "&",
        "%",
        "$",
        "@",
        "|",
        "\\",
        "/",
        "<",
        ">",
    ]:
        this_factor = this_factor.replace(punct, "").strip()

    # 替换多个空格为单个空格
    this_factor = re.sub(r"\s+", " ", this_factor).strip()

    return this_factor


def check_factor_list(factor_list):
    factor_list = [
        f.replace("#", "").replace("**", "").replace("```", "").strip()
        for f in factor_list
    ]
    factor_list = [f for f in factor_list if len(f) > 10]
    factor_list = [f for f in factor_list if len(f.split("\n")) >= 2]
    factor_list = [f for f in factor_list if ("desc" in f.lower())]
    factor_list = [
        f for f in factor_list if ("sentiment intensity" not in f.lower())
    ]  # this is an example factor in prompt.

    return factor_list

