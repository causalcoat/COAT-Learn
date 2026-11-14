# Contributing

Contributions should keep each module useful on its own. Model calls belong in `models`, fixed-definition measurement in `annotation`, graph and effect calculations in `causality`, and the three iterative procedures in `discovery`.

Discuss changes to research methods separately from interface or directory changes. Factor values follow their definitions. Model confidence is measurement metadata, and user-provided causal assumptions remain labelled as assumptions.

Use the Python interface from the [getting-started guide](<docs/quickstart.md>) or inspect commands with `python3 -m coatlearn --help`. Keep examples and documentation aligned with the public interfaces.

Keep credentials, private text and local model artifacts out of contributions. Configuration files contain environment-variable names, not keys. Dataset contributions need an explicit origin and redistribution permission.
