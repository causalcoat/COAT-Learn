"""Small, offline catalog for all task-planning interfaces."""
from copy import deepcopy
from pathlib import Path
import tomllib

_CATALOG = {
    "providers": {
        "deepseek": {"backend": "openai", "protocol": "chat", "address": "https://api.deepseek.com", "api_key_env": "DEEPSEEK_API_KEY", "models": ["deepseek-flash", "deepseek-v4-pro"], "efforts": ["none", "low", "high", "max"]},
        "openrouter": {"backend": "openai", "protocol": "chat", "address": "https://openrouter.ai/api/v1", "api_key_env": "OPENROUTER_API_KEY", "models": ["openai/gpt-4o-mini"], "efforts": []},
        "opencode": {"backend": "openai", "protocol": "chat", "address": "https://opencode.ai/inference/openai/v1", "api_key_env": "OPENCODE_API_KEY", "models": ["deepseek-v4.1-flash", "glm-5.3", "kimi-k3"], "efforts": []},
        "vllm": {"backend": "openai", "protocol": "chat", "address": "http://localhost:8000/v1", "api_key_env": "VLLM_API_KEY", "models": [], "efforts": []},
    },
    "decision_models": {
        "jev": {"backend": "jev", "models": ["jev-1.14.0"], "address": "https://api.typesafe.ai", "api_key_env": "TYPESAFE_API_KEY"},
        "kev": {"backend": "kev", "models": ["kev-9b-v1.0", "kev-4b-v1.0"], "address": "http://127.0.0.1:8019", "addresses": {"kev-9b-v1.0": "http://127.0.0.1:8019", "kev-4b-v1.0": "http://127.0.0.1:8014"}, "api_key_env": None},
    },
    "methods": {"mb": {"rounds": 2, "example_rounds": 5}, "pag": {"rounds": 2, "example_rounds": 2}, "adj": {"rounds": 2, "example_rounds": 2}},
}


def get_catalog(project=None) -> dict:
    """Return independent JSON-compatible catalog data without network access."""
    catalog = deepcopy(_CATALOG)
    root = Path(project) if project is not None else Path(__file__).resolve().parent.parent
    for method, settings in catalog['methods'].items():
        config = root / 'config' / (method + '.toml')
        if config.is_file():
            with config.open('rb') as handle:
                document = tomllib.load(handle)
            settings['example_rounds'] = document['analysis']['rounds']
            settings['example'] = document['task']
    return catalog
