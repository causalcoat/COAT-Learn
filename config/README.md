# Task configurations

Install with `python -m pip install coatlearn` and prepare the working directory with the [installation guide](<../docs/installation.md>). Run the commands below from that directory.

The three TOML files describe complete example tasks. MB and PAG share the apple graph1 CSV; ADJ uses synthetic Hillstrom posts. Paths inside each configuration are relative to that file.

| Task | Configuration | Local summary |
| --- | --- | --- |
| Target factors | [MB](<mb.toml>) | `python -m coatlearn mb --config config/mb.toml` |
| Relationship graph | [PAG](<pag.toml>) | `python -m coatlearn pag --config config/pag.toml` |
| Adjustment and effects | [ADJ](<adj.toml>) | `python -m coatlearn adj --config config/adj.toml` |

Each configuration defines the main model and three decision-model choices. The roles select Kev-9B by default. `--decision-model jev` or `--decision-model kev4b` changes annotation and enabled review/comparison roles together, leaving analysis parameters unchanged.

The main model omits the temperature parameter. TOML has no null literal, so these task files express that choice with `omit_temperature = true`. Generic model configurations still default to temperature zero unless the caller makes this choice explicitly.

Inspect a task locally before adding `--run`. Export the named key variables or pass an explicit `--env-file` during execution. Model requests may send text and incur charges.

The [configuration guide](<../docs/configuration.md>) defines the analysis settings and the supplied ADJ assumptions. Review those assumptions for a new study rather than adopting them automatically.

## 中文

三份配置各自描述一个完整任务。MB 与 PAG 共用苹果 graph1，ADJ 使用合成 Hillstrom 帖子。默认使用 Kev-9B；`--decision-model` 一起切换已启用的决策角色，不改变统计参数。主模型用 `omit_temperature = true` 明确省略温度，不改变普通模型配置的默认值。

先按[安装指南](<../docs/zh-cn/installation.md>)准备工作目录，再运行上述命令查看本地摘要；确认模型和数据发送范围后再加 `--run`。参数与研究假设见[中文配置指南](<../docs/zh-cn/configuration.md>)。
