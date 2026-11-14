# Publishing COAT-Learn

This is a maintainer procedure, not an instruction to upload a package now. The repository's publishing workflow defaults to **build-only**. Pushing a commit or creating a GitHub release does not automatically publish to PyPI.

## Release identity

The distribution, import namespace and CLI are `coatlearn`. The first release, [0.1.0](https://pypi.org/project/coatlearn/0.1.0/), is published through the approved `v0.1.0` tag. The published wheel and source archive are available on PyPI.

Versions come from Git tags through setuptools-scm. Do not add a version constant or edit a version field. The approved `v0.1.0` tag produces package version `0.1.0`. Each subsequent release number needs a maintainer decision. Untagged or modified checkouts generate development versions with local suffixes; those are useful locally but the publication workflow refuses to upload them.

Keep prior releases on PyPI. An uploaded filename cannot be overwritten; publish a new Git-tagged version for a change. The user-facing documentation describes the current package, not manually copied historical implementations.

## One-time account setup

1. Create or sign in to a [PyPI account](https://pypi.org/account/register/), verify its email and enable [two-factor authentication](https://pypi.org/help/#twofa). Keep recovery codes private.
2. In [PyPI Publishing](https://pypi.org/manage/account/publishing/), add a pending GitHub publisher with exactly these fields:

   | Field | Value |
   | --- | --- |
   | PyPI project name | `coatlearn` |
   | Owner | `causalcoat` |
   | Repository | `COAT-Learn` |
   | Workflow filename | `publish.yml` |
   | Environment | `pypi` |

3. In the GitHub repository settings, create environment `pypi`. Restrict deployment to release tags and require maintainer review where the repository's GitHub plan supports those protections. Keep the workflow's manual target choice even if environment review is unavailable.
4. If rehearsing on TestPyPI, create its separate account and publisher at [TestPyPI Publishing](https://test.pypi.org/manage/account/publishing/) with the same fields except environment `testpypi`. Create that GitHub environment too. TestPyPI accounts and projects are separate from PyPI.

Trusted Publishing exchanges the approved workflow's identity for a short-lived credential. No PyPI password or long-lived API token belongs in this repository or its GitHub secrets. A pending publisher does not reserve the package name; the project is created on the first successful upload.

## Prepare locally

From the repository root:

```bash
python3 -m pip install --upgrade build twine
python3 tools/prepare_pypi_readme.py
python3 -m build --outdir build/release-candidate
python3 -m twine check --strict build/release-candidate/*
python3 tools/check_distribution.py build/release-candidate
```

Use a fresh output directory for each attempt. The generated [PyPI README](<../README-PyPI.md>) has absolute links; edit [README.md](<../README.md>) and regenerate it rather than editing two descriptions. Include the generated result in the reviewed commit.

The source distribution contains the two approved example CSVs, documentation and public skill. The wheel installs the Python library and CLI. It does **not** install repository-root configuration, examples or datasets into the user's working directory. The default installation is `python -m pip install coatlearn`. The installation guide downloads only the example configurations, data and attribution into a working directory; users do not need a repository clone. Library users can also provide their own task configuration and data.

The SPDX metadata describes the source archive's combined contents: Apache-2.0 software, CC BY 4.0 example data and MIT documentation assets. Each keeps its own scope and notices. The data license does not replace the software license.

## Publish only after approval

1. Publish the reviewed source to GitHub when authorized. Confirm website source links and paper metadata.
2. Select **Build and publish Python package** in GitHub Actions, target `build-only`, and review the two build artifacts. This path never requests a PyPI upload identity.
3. Agree on the release version. Create and push its `vX.Y.Z` Git tag on a reviewed commit reachable from `main`. The workflow fetches full Git history so setuptools-scm can read the tag.
4. Optionally select the tagged ref and target `testpypi` to rehearse. Install its wheel without resolving dependencies from the test index; the test index is not a replacement dependency source.
5. After explicit production approval, manually select that exact tag and target `pypi`. The build must match the tag, be a final version, contain no local/development suffix and pass metadata checks. The separate publishing job uses environment `pypi` and `id-token: write` only for that job.
6. Inspect the live project page, verify a fresh `pip install coatlearn`, then update public installation text and any PyPI badges. Verify the actual published version rather than presenting a prepared workflow as a completed release.

## Official references

- [Python Packaging User Guide](https://packaging.python.org/en/latest/tutorials/packaging-projects/)
- [Create a project through a pending trusted publisher](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/)
- [Use a trusted publisher](https://docs.pypi.org/trusted-publishers/using-a-publisher/)
- [setuptools-scm configuration](https://setuptools-scm.readthedocs.io/en/stable/config/)
