# COAT-Learn (`coatlearn`)

[中文](<README.zh-cn.md>) · [Documentation](<docs/README.md>) · [Website](https://causalcoat.github.io/) · [Paper](https://www.jmlr.org/papers/v27/25-2137.html)

coatlearn is a Python implementation of COAT (Causal Representation Assistant), a research framework for discovering, defining and measuring useful factors in unstructured data, then studying their causal structure and intervention effects. It is for analysts who have ratings, diagnoses or behavior records but do not yet have a complete table of variables to analyze.

For example, you have customer ratings and reviews, but do not know which product attributes to measure. COAT asks a language model to propose factors such as sweetness and aroma, assigns values to each observation, and uses statistical feedback to guide further discovery. The language model interprets the source material and defines variables. Statistical methods analyze causal relationships under their assumptions.

The paper considers text, images and material that requires external measurement tools. This package's ready-to-use discovery interfaces take a text column; the examples below use reviews and posts. The paper's image and external-tool studies are not direct image or database input interfaces in this package.

## From source material to variables

A **factor** is an attribute with a clear meaning and measurement rule. For example, define the sweetness described in a review as “not sweet,” “mildly sweet” or “very sweet,” encoded as 0, 1 and 2. **Factor annotation** applies that same rule to each observation to produce a numeric column. A name alone cannot tell a statistical method whether the attribute provides additional information.

If you know which attributes to measure, [annotate your own factors](<docs/annotation.md>) to build a table for filtering, summaries or modeling. If you do not know which variables are missing, COAT can repeatedly propose and select factors using statistical feedback. This is **factor discovery**.

## From variables to a research question

The three methods address progressively deeper analysis goals. You do not need to understand every algorithm before starting.

| What you want to understand | Method and result | What the result is useful for |
| --- | --- | --- |
| Which attributes jointly provide information about a rating? | COAT-MB seeks a **Markov blanket**: a set of variables that makes the remaining candidates provide no additional information about the target once that set is given. | A variable set for later prediction and investigation. It can include effects and other variables, not only causes. |
| What causal structure connects these variables? | COAT-PAG adds factors and builds a **partial ancestral graph**: nodes represent variables; edge marks record determined causal information and unresolved directions. | Investigate relationships without drawing a definite arrow where information is insufficient. |
| How much would an action change an outcome on average? | COAT-ADJ seeks **adjustment sets**, groups of variables that can remove the influence of non-causal paths under the graph and assumptions, then performs **effect estimation**. | For example, study an email's effect on visits without mistaking existing customer differences for the email's effect. |

A causal graph can come from COAT, another discovery method or explicit researcher assumptions. With a measured table and a graph, you can [find adjustment sets and estimate effects](<docs/causality.md>) without discovering factors again. [Understand COAT](<docs/concepts.md>) explains how these ideas fit together.

## Inspect an example first

Use Python 3.11 or newer. Run these commands from the project directory, which contains the [project configuration](<pyproject.toml>):

```bash
git clone https://github.com/causalcoat/COAT-Learn.git
cd COAT-Learn
python3 -m pip install -e .
bash examples/mb.sh
```

The second command reads local data and configuration only. It displays the method, row count, selected columns and model connections. It neither reads credentials nor calls a model.

Example scripts load settings from TOML files. TOML is a text format with sections such as `[task]`, so the data, models and statistical settings for an analysis can be kept together.

| Example | Input and goal | Saved settings |
| --- | --- | --- |
| [MB](<examples/mb.sh>) | 1,000 synthetic apple reviews; discover factors in `Review` that provide information about `Score`. | [Configuration](<config/mb.toml>): 5 rounds, Fisher-Z conditional-independence tests, threshold 0.001. |
| [PAG](<examples/pag.sh>) | The same apple data; start with measured `Taste` and `Score`, then discover factors to refine their graph. | [Configuration](<config/pag.toml>): 2 rounds, Fisher-Z, threshold 0.001, graph-search depth 2. |
| [ADJ](<examples/adj.sh>) | 2,000 synthetic customer posts with email and visit records; seek adjustment sets and estimate the effect of sending an email on visits. | [Configuration](<config/adj.toml>): 2 rounds, missing-value Fisher-Z, threshold 0.001, with explicit research assumptions and screening rules. |

MB and PAG share one CSV; ADJ uses the other. [Data](<docs/data.md>) explains their origins and encoding; [configuration](<docs/configuration.md>) explains the parameters. The ADJ data is not the original randomized Hillstrom experiment and must not be interpreted as randomly assigned treatment.

## Run your first discovery

Discovery needs a chat model to propose factors and an annotation model to assign values by rule. All three configurations default to DeepSeek Flash for proposal and local Kev-9B for annotation, definition review and concept deduplication.

**Jev** is TypeSafe's hosted decision model for classification, rating and yes/no questions. **Kev** is an independent decision model that you can run locally with a compatible request format. **vLLM** is software for serving chat models, not another discovery method or a requirement for running Kev.

For a first attempt, hosted Jev avoids local model deployment. Follow [hosted model setup](<docs/models.md#connect-hosted-models>) to prepare `DEEPSEEK_API_KEY` and `TYPESAFE_API_KEY`, then run:

```bash
bash examples/mb.sh --decision-model jev --run --output-dir runs/apple-mb
```

`--decision-model jev` switches annotation, review and deduplication together; the proposer stays unchanged. Only `--run` calls the models. Hosted calls send the selected text and may incur charges. To use the default Kev instead, first start a server with the [local deployment guide](<docs/kev.md>).

After execution, open `result.json` in the output directory. For MB, `diagnostics.final_blanket` lists the selected names, including the target `Score`. `factors` holds available definitions; `provenance` records model and analysis settings. [Your first analysis](<docs/quickstart.md>) includes the full steps and code for reading the result.

## Use Python

Python and the command line can use the same configuration. The following code checks local input, then explicitly executes the task. Prepare model credentials before execution.

```python
from coatlearn.configuration import load_task
from coatlearn.execution import check_task, execute_task

plan = load_task("config/mb.toml", decision_model="jev")
checked = check_task(plan)
print(checked.summary["rows"])

result, directory = execute_task(checked, output_dir="runs/python-mb")
print(result["diagnostics"]["final_blanket"])
```

For your own CSV or pandas DataFrame, call `run_mb`, `run_pag` or `run_adj` directly. See [prepare data](<docs/data.md>) and [discovery interfaces](<docs/discovery.md>).

## Read next

- [Understand COAT](<docs/concepts.md>): why discover variables, and what the three methods answer.
- [Your first analysis](<docs/quickstart.md>): install, run MB and inspect one result.
- [Annotate factors](<docs/annotation.md>): apply existing definitions to your material.
- [Model connections](<docs/models.md>) and [configuration reference](<docs/configuration.md>): choose services and adjust settings.
- [Result reference](<docs/results.md>): read factors, graphs, adjustment sets and estimates.
- [Agent use](<docs/agents.md>): let an agent assist through the public [skill](<skills/coatlearn/SKILL.md>).

See the [paper](https://www.jmlr.org/papers/v27/25-2137.html) for the research, [CITATION.cff](<CITATION.cff>) for citation details and the [contribution guide](<CONTRIBUTING.md>) for development. Causal conclusions depend on measurement quality, data and method assumptions. A completed run does not establish those assumptions.

## License

The software is licensed under [Apache-2.0](<LICENSE>). The two example CSV files are licensed separately under [CC BY 4.0](<data/LICENSE>); see their [source and attribution details](<data/README.md>). Authors and source provenance are recorded in [NOTICE](<NOTICE>) and [SOURCE_MAP.md](<SOURCE_MAP.md>).
