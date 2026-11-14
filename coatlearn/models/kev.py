"""Local Kev Adapter at the existing typed-measurement seam.

The separate Python GPU process owns one checkpoint. This module reuses Jev's
measurement parsing, gates and metadata, but never reads a provider credential,
pretends to be Jev, or silently falls back to a hosted model.
"""
from dataclasses import asdict, dataclass
from urllib.parse import urlsplit

from .systemone import JevClient, JevConfig


@dataclass(frozen=True)
class KevConfig(JevConfig):
    """Loopback-only local endpoint; start one pinned checkpoint separately."""

    model: str = "kev-9b-v1.0"
    base_url: str = "http://127.0.0.1:8019"
    api_key_env: str = "CAUSALCOAT_LOCAL_NO_KEY"
    max_retries: int = 0
    max_concurrency: int = 3

    def __post_init__(self):
        super().__post_init__()
        parts = urlsplit(self.base_url)
        if parts.scheme != "http" or parts.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("Kev requires a loopback HTTP endpoint")
        if self.model not in {"kev-9b-v1.0", "kev-4b-v1.0"}:
            raise ValueError("Select kev-9b-v1.0 or kev-4b-v1.0")
        if self.max_retries != 0:
            raise ValueError("Local Kev requests do not retry automatically")

    def redacted(self):
        fields = asdict(self)
        fields.pop("api_key_env")
        return {"backend": "kev", **fields}


class KevClient(JevClient):
    """Fresh per-role measurement state, shared GPU via the local endpoint."""

    def __init__(self, config=None):
        super().__init__(config or KevConfig())

    def _ensure_client(self):
        with self._client_lock:
            if self._client is None:
                import httpx
                self._client = httpx.Client(base_url=self.config.base_url,
                                            timeout=self.config.timeout, trust_env=False,
                                            headers={"Content-Type": "application/json"})
            return self._client


def decision_config(fields):
    """Validate role fields without IO or client creation."""
    fields = dict(fields)
    backend = fields.pop("backend", "jev")
    if backend == "kev":
        return KevConfig(**fields)
    if backend == "jev":
        return JevConfig(**fields)
    raise ValueError("Typed decision backend must be jev or kev")


