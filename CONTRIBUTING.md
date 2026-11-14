# Contributing

## Developer installation

For normal analysis, install the published package with `python -m pip install coatlearn`. To change the source, clone this repository into a separate development environment:

```bash
git clone https://github.com/causalcoat/COAT-Learn.git
cd COAT-Learn
python -m pip install -e .
```

## Module responsibilities

Contributions should keep each module useful on its own. Model calls belong in `models`, fixed-definition measurement in `annotation`, graph and effect calculations in `causality`, and the three iterative procedures in `discovery`.

Discuss changes to research methods separately from interface or directory changes. Factor values follow their definitions. Model confidence is measurement metadata, and user-provided causal assumptions remain labelled as assumptions.

Use the Python interface from the [getting-started guide](<docs/quickstart.md>) or inspect commands with `python3 -m coatlearn --help`. Keep examples and documentation aligned with the public interfaces.

Keep credentials, private text and local model artifacts out of contributions. Configuration files contain environment-variable names, not keys. Dataset contributions need an explicit origin and redistribution permission.

After editing the root README, regenerate the PyPI description with `python3 tools/prepare_pypi_readme.py`. Software releases use Git tags and the manual trusted-publishing workflow; see the [maintainer publishing guide](<maintainer/publishing.md>).
