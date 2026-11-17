"""Prepare and run the optional local Kev service in its own environment.

Nothing is installed or downloaded unless ``prepare(..., execute=True)`` is
called. Existing assets may be supplied as explicit directory paths.
"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
from urllib.parse import urlsplit

_MODELS = {
    "kev-9b-v1.0": ("jaredpalmer/kev-9b", "Qwen/Qwen3.5-9B-Base"),
    "kev-4b-v1.0": ("jaredpalmer/kev-4b", "Qwen/Qwen3.5-4B-Base"),
}

# Executed inside the dedicated environment. Hugging Face local-dir metadata
# records the commit for each file, including each shard of the base model.
_ASSETS = r'''
import json, sys
from pathlib import Path
from huggingface_hub import snapshot_download
from kev.checkpoint import Checkpoint
adapter, base = map(Path, sys.argv[1:3])
repo, expected_base, download = sys.argv[3], sys.argv[4], sys.argv[5] == "1"
if download and not all((adapter / name).is_file() for name in ("head.pt", "adapter_config.json", "adapter_model.safetensors")):
    snapshot_download(repo, revision="v1.0", local_dir=adapter)
if not all((adapter / name).is_file() for name in ("head.pt", "adapter_config.json", "adapter_model.safetensors")):
    raise ValueError("Adapter snapshot incomplete")
ck = Checkpoint(str(adapter))
if ck.meta.base != expected_base or not ck.meta.base_revision:
    raise ValueError("Adapter does not match selected model")
if download:
    snapshot_download(ck.meta.base, revision=ck.meta.base_revision, local_dir=base)
required = ["config.json"]
indexes = list(base.glob("*.safetensors.index.json"))
if indexes:
    required += sorted(set(json.loads(indexes[0].read_text())["weight_map"].values()))
    required.append(indexes[0].name)
else:
    required += [p.name for p in base.glob("*.safetensors")]
    if len(required) == 1:
        raise ValueError("Base model weights missing")
for name in required:
    f = base / name
    meta = base / ".cache" / "huggingface" / "download" / (name + ".metadata")
    if not f.is_file() or not meta.is_file() or meta.read_text().splitlines()[0] != ck.meta.base_revision:
        raise ValueError("Base snapshot incomplete or revision not verified")
'''

_SERVE = r'''
import sys
from coatlearn.models.kev_server import serve
serve(checkpoint=sys.argv[1], base_model=sys.argv[2], source_dir=sys.argv[3],
      model=sys.argv[4], port=int(sys.argv[5]))
'''


def _settings(model, address, work_dir, venv_dir, source_dir, adapter_dir, base_dir):
    if model not in _MODELS:
        raise ValueError("model must be kev-9b-v1.0 or kev-4b-v1.0")
    if address is None:
        from .catalog import get_catalog
        address = get_catalog()['decision_models']['kev']['addresses'][model]
    try:
        url = urlsplit(address)
        port = url.port
    except ValueError:
        raise ValueError("address must be http://127.0.0.1:<port>") from None
    if (url.scheme != "http" or url.hostname != "127.0.0.1" or not port
            or url.username is not None or url.password is not None
            or url.path not in ("", "/") or url.query or url.fragment):
        raise ValueError("address must be http://127.0.0.1:<port>")
    work = Path(work_dir or Path.home() / "coatlearn-kev" / model).expanduser().resolve()
    def path(value, default):
        return Path(value).expanduser().resolve() if value is not None else work / default
    return {"model": model, "address": f"http://127.0.0.1:{port}", "port": port,
            "work": work, "venv": path(venv_dir, ".venv"),
            "source": path(source_dir, "source"), "adapter": path(adapter_dir, "adapter"),
            "base": path(base_dir, "base"), "cache": work / "cache"}


def _run(command, *, env):
    try:
        result = subprocess.run(command, env=env, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL, check=False)
    except OSError:
        raise RuntimeError("Could not run local Kev setup command") from None
    if result.returncode:
        raise RuntimeError("Local Kev setup command failed; check dependencies, paths and network")


def prepare(model, address=None, *, work_dir=None,
            venv_dir=None, source_dir=None, adapter_dir=None, base_dir=None,
            execute=False):
    """Return a JSON-ready plan; execute it only when explicitly requested.

    Existing source and environment are reused. Existing weights must be a
    verifiable Hugging Face local-dir snapshot at the adapter's base revision.
    """
    s = _settings(model, address, work_dir, venv_dir, source_dir, adapter_dir, base_dir)
    python = s["venv"] / "bin" / "python"
    repo, base_repo = _MODELS[model]
    steps = []
    if not python.is_file():
        steps.append(["python3.12", "-m", "venv", str(s["venv"])])
    if not (s["source"] / "kev" / "checkpoint.py").is_file():
        if s["source"].exists():
            raise ValueError("Existing source directory is not a Kev checkout")
        steps.append(["git", "clone", "https://github.com/jaredpalmer/kev.git", str(s["source"])])
    checkout = Path(__file__).resolve().parent.parent
    if (checkout / "pyproject.toml").is_file():
        distribution = str(checkout) + "[kev]"
    else:
        from importlib.metadata import version
        distribution = "coatlearn[kev]==" + version("coatlearn")
    steps.append([str(python), "-m", "pip", "install", distribution, str(s["source"])])
    steps.append([str(python), "-c", _ASSETS, str(s["adapter"]), str(s["base"]), repo, base_repo, "1"])
    plan = {"model": model, "address": s["address"], "work_dir": str(s["work"]),
            "paths": {k: str(s[k]) for k in ("venv", "source", "adapter", "base", "cache")},
            "steps": steps, "executed": bool(execute)}
    if execute:
        s["work"].mkdir(parents=True, exist_ok=True)
        env = {**os.environ, "HF_HOME": str(s["cache"])}
        for command in steps:
            _run(command, env=env)
    return plan


def start(model, address=None, *, work_dir=None,
          venv_dir=None, source_dir=None, adapter_dir=None, base_dir=None):
    """Validate local assets and run the server in the foreground."""
    s = _settings(model, address, work_dir, venv_dir, source_dir, adapter_dir, base_dir)
    python = s["venv"] / "bin" / "python"
    if not python.is_file() or not (s["source"] / "kev" / "checkpoint.py").is_file():
        raise ValueError("Kev environment or source missing; run prepare first")
    env = {**os.environ, "HF_HOME": str(s["cache"]), "HF_HUB_OFFLINE": "1",
           "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1"}
    _run([str(python), "-c", _ASSETS, str(s["adapter"]), str(s["base"]),
          *_MODELS[model], "0"], env=env)
    try:
        result = subprocess.run([str(python), "-c", _SERVE, str(s["adapter"]),
                                 str(s["base"]), str(s["source"]), model, str(s["port"])],
                                env=env, stderr=subprocess.DEVNULL, check=False)
    except OSError:
        raise RuntimeError("Could not start local Kev server") from None
    if result.returncode:
        raise RuntimeError("Local Kev server failed to start or exited with an error")
