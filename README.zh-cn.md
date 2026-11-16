# COAT-Learn（`coatlearn`）

[English](<README.md>) · [使用文档](<docs/zh-cn/README.md>) · [项目网站](https://causalcoat.github.io/) · [论文](https://www.jmlr.org/papers/v27/25-2137.html)

coatlearn 是 COAT（Causal Representation Assistant，因果表征助手）研究框架的 Python 实现，用于从非结构化数据中发现、定义并测量有用的因子，再研究它们的因果结构与干预效应。它面向已有评分、诊断或行为记录，却还没有完整分析变量表的数据分析人员。

例如，你有顾客评分和评论，但不知道还应测量哪些产品属性。COAT 让语言模型提出“甜度”“香气”等候选因子，为每条观测赋值，再用统计分析的结果指导下一轮发现。语言模型负责理解资料和定义变量，因果关系由统计方法在相应假设下分析。

论文的非结构化数据包括文本、图像，以及需要借助工具测量的资料。本包的现成发现接口读取文本列；以下示例使用评论和帖子。论文中的图像与外部工具案例，不是本包直接读取图片或数据库的接口。

## 一分钟了解 COAT

[![播放一分钟 COAT 介绍：从评论到变量，再到统计反馈。](https://causalcoat.github.io/static/coat-journal/media/coat-readme-zh.webp)](https://causalcoat.github.io/index-zh.html#intro-video)

**[观看中文版](https://causalcoat.github.io/index-zh.html#intro-video)** · [Watch in English](https://causalcoat.github.io/#intro-video)

## 从资料到分析变量

**因子**是有明确含义和测量规则的属性。比如，把评论描述的甜度定义为“不甜、微甜、很甜”，并分别编码为 0、1、2。**因子标注**就是按同一规则给每条观测赋值，形成一列数值。只有名称而没有逐条取值，统计方法就无法判断这个属性是否提供了额外信息。

如果你已经知道要测量哪些属性，可以直接[标注自己的因子](<docs/zh-cn/annotation.md>)，得到供筛选、汇总或建模使用的表格。如果还不知道缺少哪些变量，可以让 COAT 结合统计反馈反复提出并筛选因子，这就是**因子发现**。

## 从变量到研究问题

三个方法对应逐步深入的分析目标，不要求你一开始就理解所有算法。

| 你想了解什么 | 方法与结果 | 结果有什么用 |
| --- | --- | --- |
| 哪些属性共同提供了关于评分的信息？ | COAT-MB 寻找**马尔可夫毯**：给定这组变量后，其他候选变量不再提供关于目标的额外信息。 | 得到用于后续预测和调查的变量集合。集合中不只有原因，也可能有结果等变量。 |
| 这些变量之间的因果结构是什么？ | COAT-PAG 补充因子并构建**部分祖先图**：节点代表变量，边端点表达可确定的因果信息与仍未确定的方向。 | 分析变量关系，避免把信息不足的方向画成确定箭头。 |
| 一个行动会使结果平均改变多少？ | COAT-ADJ 寻找**调整集**，即在图与假设下，可用于消除非因果路径影响的一组变量，再进行**效应估计**。 | 例如研究发送邮件对访问率的影响，而不把顾客原有差异当作邮件效果。 |

因果图可以来自 COAT、其他发现方法，或研究者明确提出的假设。若已有测量表和图，可以直接[寻找调整集与估计效应](<docs/zh-cn/causality.md>)，不必重新发现因子。概念之间的关系见[理解 COAT](<docs/zh-cn/concepts.md>)。

## 安装

使用 Python 3.11 或更新版本：

```bash
python -m pip install coatlearn
python -m coatlearn --help
```

安装的是 Python 库与命令行工具，不会下载模型权重、启动服务或调用模型。虚拟环境、可选依赖与配置准备见[安装与准备分析](<docs/zh-cn/installation.md>)。

## 先查看一个示例

按[示例下载步骤](<docs/zh-cn/installation.md#3-准备示例工作目录>)获取三份配置和两份 CSV，不需要克隆仓库。在下载后的 `coat-example` 目录执行：

```bash
python -m coatlearn mb --config config/mb.toml --decision-model jev
```

命令只读取本地数据和配置，显示方法、行数、使用的列与模型连接，不读取密钥、不调用模型。

任务设置保存在 TOML 文件中。TOML 是使用 `[task]` 等区段保存参数的文本格式，便于把一次分析的数据、模型和统计设置放在一起。

| 示例 | 输入与目标 | 保存的设置 |
| --- | --- | --- |
| [MB](<examples/mb.sh>) | 1000 条合成苹果评论；从 `Review` 中发现关于 `Score` 的因子。 | [配置](<config/mb.toml>)：5 轮，Fisher-Z 条件独立性检验，阈值 0.001。 |
| [PAG](<examples/pag.sh>) | 同一份苹果数据；从已有 `Taste`、`Score` 出发，发现更多因子以细化关系图。 | [配置](<config/pag.toml>)：2 轮，Fisher-Z，阈值 0.001，图搜索深度 2。 |
| [ADJ](<examples/adj.sh>) | 2000 条合成顾客帖子及邮件、访问记录；寻找发送邮件到网站访问的调整集并估计效应。 | [配置](<config/adj.toml>)：2 轮，支持缺失值的 Fisher-Z，阈值 0.001，并声明研究假设和筛选规则。 |

MB 与 PAG 共用一份 CSV，ADJ 使用另一份。[数据说明](<docs/zh-cn/data.md>)解释来源和编码；[配置说明](<docs/zh-cn/configuration.md>)解释参数。ADJ 数据不是原始随机化 Hillstrom 实验，不能直接沿用随机分组的解释。

## 执行第一次发现

发现需要一个提出因子的聊天模型，以及一个按规则赋值的标注模型。三份配置默认使用 DeepSeek Flash 提出因子、本地 Kev-9B 完成标注、定义审核和概念去重。

**Jev** 是 TypeSafe 的托管决策模型，回答分类、评分和是非问题。**Kev** 是可在本机运行的独立决策模型，使用兼容的请求格式。**vLLM** 则是部署聊天模型的推理服务软件，不是另一个因子发现方法，也不是启动 Kev 的必需工具。

第一次尝试可以选择托管 Jev，省去本地模型部署。先按[模型连接说明](<docs/zh-cn/models.md#连接托管模型>)准备 `DEEPSEEK_API_KEY` 与 `TYPESAFE_API_KEY`，再运行：

```bash
python -m coatlearn mb --config config/mb.toml --decision-model jev --run --output-dir runs/apple-mb
```

`--decision-model jev` 将标注、审核和去重一起切换为 Jev；提出因子的模型不变。`--run` 才会调用模型。托管调用会发送所用文本，并可能产生费用。若使用默认 Kev，先按[本地部署指南](<docs/zh-cn/kev.md>)启动服务。

执行后，打开输出目录中的 `result.json` 查看结果。MB 的 `diagnostics.final_blanket` 列出选中的变量名称，其中也包含目标 `Score`。`factors` 保存可用的因子定义；`provenance` 保存模型与分析设置。完整步骤和读取结果的代码见[第一个分析](<docs/zh-cn/quickstart.md>)。

## 在 Python 中使用

Python 和命令行可以使用同一份配置。下面先做本地检查，再明确执行；执行前同样需要准备模型凭据。

```python
from coatlearn.configuration import load_task
from coatlearn.execution import check_task, execute_task

plan = load_task("config/mb.toml", decision_model="jev")
checked = check_task(plan)
print(checked.summary["rows"])

result, directory = execute_task(checked, output_dir="runs/python-mb")
print(result["diagnostics"]["final_blanket"])
```

使用自己的 CSV 或 pandas DataFrame 时，可直接调用 `run_mb`、`run_pag`、`run_adj`。参见[准备数据](<docs/zh-cn/data.md>)和[发现接口](<docs/zh-cn/discovery.md>)。

## 继续阅读

- [安装与准备](<docs/zh-cn/installation.md>)：从 PyPI 安装、下载示例并选择任务配置。
- [理解 COAT](<docs/zh-cn/concepts.md>)：为什么要发现变量，三个方法分别回答什么问题。
- [第一个分析](<docs/zh-cn/quickstart.md>)：从安装到查看一份 MB 结果。
- [标注因子](<docs/zh-cn/annotation.md>)：把已有定义用于自己的资料。
- [模型连接](<docs/zh-cn/models.md>)与[配置参考](<docs/zh-cn/configuration.md>)：选择服务和调整参数。
- [结果参考](<docs/zh-cn/results.md>)：读取因子、图、调整集和估计。
- [通过代理使用](<docs/zh-cn/agents.md>)：让代理依据公开 [skill](<skills/coatlearn/SKILL.md>)协助分析。

研究依据见[论文](https://www.jmlr.org/papers/v27/25-2137.html)，引用信息见 [CITATION.cff](<CITATION.cff>)，参与开发见[贡献指南](<CONTRIBUTING.md>)。因果结论依赖测量质量、数据和方法假设，程序执行完成不代表这些假设已被证实。

## 许可

软件采用 [Apache-2.0](<LICENSE>)。两份示例 CSV 单独采用 [CC BY 4.0](<data/LICENSE>)，来源与署名方式见[数据说明](<data/README.md>)。作者与代码来源见 [NOTICE](<NOTICE>) 和 [SOURCE_MAP.md](<SOURCE_MAP.md>)。
