# Source provenance

The discovery procedures originate in the authors' `coat-discovery` research tree at commit `96d0967`. This publication imports the selected software from the authors' COAT-JMLR research workspace at source revision `57f97d4`, without importing its Git history. The maintained implementation is organized by function and published as `coatlearn` in [COAT-Learn](https://github.com/causalcoat/COAT-Learn).

The publication changes package names, source-root detection, packaging metadata, documentation and licensing. It preserves the MB, PAG and ADJ procedures, their statistical settings, factor-value mappings and effect-estimation policy. Author attribution is retained in [NOTICE](<NOTICE>) and [CITATION.cff](<CITATION.cff>).

| Current source | Origin and responsibility |
| --- | --- |
| [MB discovery](<coatlearn/discovery/mb.py>), [PAG discovery](<coatlearn/discovery/pag.py>), [ADJ discovery](<coatlearn/discovery/adj.py>) | Maintained journal algorithms, including factor proposal, measurement scheduling and statistical feedback |
| [Graph operations](<coatlearn/causality/graph.py>), [adjustment](<coatlearn/causality/adjustment.py>), [statistics](<coatlearn/causality/statistics.py>) | Graph and numeric helpers separated from the research utility modules |
| [Effect estimation](<coatlearn/causality/effects.py>), [postprocessing](<coatlearn/causality/postprocessing.py>) | Existing complete-case IPW and explicit graph-deletion hypotheses, separated from discovery |
| [Text helpers](<coatlearn/annotation/text.py>), [quality rules](<coatlearn/annotation/quality.py>), [logging](<coatlearn/utils/logging.py>) | Maintained helpers extracted from the original utility and base modules |
| [Models](<coatlearn/models/__init__.py>), [measurement](<coatlearn/annotation/measurement.py>) | Package adapters: OpenAI-compatible calls and official-format System One HTTP requests; factor interpretation is separate from transport |
| [Named causal interface](<coatlearn/causality/interface.py>), [batch annotation](<coatlearn/annotation/batch.py>) | Independent public use of the existing methods |
| [Task configuration](<coatlearn/configuration.py>), [execution](<coatlearn/execution.py>), [CLI](<coatlearn/cli/__init__.py>) | Shared task loading, explicit input mappings and execution artifacts |

This reorganization keeps the three algorithms, measurement mappings and effect-estimation policy. The old unused `TextCOAT` script implementation and dataset-specific loaders are not part of the public package. Historical experiment snapshots remain records of their original runs.
