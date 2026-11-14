"""Native System One model calls shared by hosted Jev and local Kev."""
from __future__ import annotations

import math
import os
import random
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlsplit

SYSTEM_ONE_PATH = "/v1/systemone"
_RETRYABLE_STATUS = (429, 529)
_MAX_RETRY_AFTER_WAIT = 120.0

@dataclass(frozen=True)
class JevConfig:
    """Jev 标注后端配置；密钥仅从环境变量 ``api_key_env`` 读取。"""

    model: str
    base_url: str = "https://api.typesafe.ai"
    api_key_env: str = "TYPESAFE_API_KEY"
    timeout: float = 60.0
    max_retries: int = 3
    max_concurrency: int = 16
    requests_per_minute: Optional[float] = None

    def __post_init__(self) -> None:
        """构造即校验；无效输入显式报错，不静默 coerce。"""
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("JevConfig.model 必须是非空字符串")
        if not isinstance(self.api_key_env, str) or not self.api_key_env.strip():
            raise ValueError("JevConfig.api_key_env 必须是非空字符串")
        if not isinstance(self.base_url, str) or not self.base_url.strip():
            raise ValueError("JevConfig.base_url 必须是非空字符串")
        _validate_base_url(self.base_url, "JevConfig.base_url")
        if not (_is_finite_number(self.timeout) and self.timeout > 0):
            raise ValueError("JevConfig.timeout 必须为正的有限数值")
        if not _is_nonnegative_int(self.max_retries):
            raise ValueError("JevConfig.max_retries 必须为非负整数（0 表示不重试）")
        if not (_is_nonnegative_int(self.max_concurrency) and self.max_concurrency >= 1):
            raise ValueError("JevConfig.max_concurrency 必须为 >=1 的整数")
        if self.requests_per_minute is not None and not (
            _is_finite_number(self.requests_per_minute) and self.requests_per_minute > 0
        ):
            raise ValueError("JevConfig.requests_per_minute 必须为正的有限数值或 None")

    def redacted(self) -> Dict[str, Any]:
        """无密钥配置快照，可安全写入实验日志。"""
        return {
            "model": self.model,
            "base_url": self.base_url,
            "api_key_env": self.api_key_env,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "max_concurrency": self.max_concurrency,
            "requests_per_minute": self.requests_per_minute,
        }


@dataclass(frozen=True)
class JevStats:
    """累计计数的安全只读快照（语义与 ModelStats 相同）。"""

    request_count: int = 0
    retry_count: int = 0
    failure_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


class JevAPIError(RuntimeError):
    """经过脱敏的 Jev 调用失败：不含 state、密钥或响应体原文。"""


class SystemOneClient:
    """A lazy model client. It accepts native questions, not COAT factors."""

    def __init__(self, config: JevConfig):
        if not isinstance(config, JevConfig):
            raise TypeError("config must be JevConfig")
        self.config = config
        self._client = None
        self._client_lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self._request_count = self._retry_count = self._failure_count = 0
        self._usage = {"input_tokens": 0, "output_tokens": 0}
        self._inflight = threading.BoundedSemaphore(config.max_concurrency)
        self._rate_lock = threading.Lock()
        self._min_interval = 60.0 / config.requests_per_minute if config.requests_per_minute else 0.0
        self._next_send_monotonic = 0.0

    def system_one(self, state, questions):
        """Return native answers for explicit questions about one state."""
        return self._system_one({"model": self.config.model, "state": state, "questions": questions})

    def stats(self) -> JevStats:
        """安全只读快照，便于结果诊断；可在任意线程调用。"""
        with self._stats_lock:
            return JevStats(
                request_count=self._request_count,
                retry_count=self._retry_count,
                failure_count=self._failure_count,
                input_tokens=self._usage["input_tokens"],
                output_tokens=self._usage["output_tokens"],
            )


    @property
    def request_count(self) -> int:
        return self.stats().request_count


    @property
    def failure_count(self) -> int:
        return self.stats().failure_count


    def _system_one(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        client = self._ensure_client()
        with self._inflight:
            # RPM 节流在 _post_with_retries 内按每次实际 HTTP 尝试执行
            return self._post_with_retries(client, payload)


    def _ensure_client(self) -> Any:
        client = self._client
        if client is not None:
            return client
        with self._client_lock:
            if self._client is None:
                try:
                    import httpx
                except ImportError:
                    raise JevAPIError(
                        "Jev 后端需要 httpx（openai SDK 依赖自带）；当前环境未安装"
                    ) from None
                api_key = os.environ.get(self.config.api_key_env)
                if not api_key:
                    raise JevAPIError(
                        f"环境变量 {self.config.api_key_env} 未设置，无法初始化 Jev 客户端"
                    )
                # httpx 默认 trust_env，保留系统代理设置；不打印任何头内容
                self._client = httpx.Client(
                    base_url=self.config.base_url,
                    timeout=self.config.timeout,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                )
            return self._client


    def _post_with_retries(self, client: Any, payload: Dict[str, Any]) -> Dict[str, Any]:
        import httpx

        max_attempts = self.config.max_retries + 1  # max_retries=0 恰好一次尝试
        for attempt in range(1, max_attempts + 1):
            self._rate_gate()  # 每次实际 HTTP 尝试（含重试）都过配额闸门
            self._bump(request=1)
            try:
                response = client.post(SYSTEM_ONE_PATH, json=payload)
            except httpx.TransportError as exc:
                if attempt >= max_attempts:
                    self._bump(failure=1)
                    raise JevAPIError(
                        f"Jev 请求失败（attempt={attempt}, error_type={type(exc).__name__}）；原始错误详情已省略"
                    ) from None
                self._bump(retry=1, failure=1)
                time.sleep(self._retry_delay(attempt, None))
                continue
            status = response.status_code
            if status in _RETRYABLE_STATUS:  # 429 限流 / 529 过载：退避后重试
                delay = None if attempt >= max_attempts else self._retry_delay(attempt, response)
                if delay is None:
                    self._bump(failure=1)
                    reason = (
                        f"已在 {attempt} 次尝试后放弃"
                        if attempt >= max_attempts
                        else f"服务端 Retry-After 超过 {_MAX_RETRY_AFTER_WAIT:.0f}s 上限，已明确停止重试"
                    )
                    raise JevAPIError(
                        f"Jev 持续限流或过载（status={status}, attempt={attempt}，{reason}）；"
                        "配额与限流以服务端信号为准"
                    )
                self._bump(retry=1, failure=1)
                time.sleep(delay)
                continue
            if status >= 400:  # 400/401/413/415/422 等：不重试，响应体不落日志
                self._bump(failure=1)
                raise JevAPIError(f"Jev API 拒绝请求（status={status}）；响应体未记录以避免泄漏 prompt")
            try:
                data = response.json()
            except ValueError:
                self._bump(failure=1)
                raise JevAPIError(f"Jev 响应不是合法 JSON（status={status}）") from None
            if not isinstance(data, dict):
                self._bump(failure=1)
                raise JevAPIError(f"Jev 响应顶层不是 JSON 对象（status={status}）")
            self._accumulate_usage(data.get("usage"))
            return data
        raise JevAPIError("Jev 重试循环异常退出")  # pragma: no cover - 逻辑上不可达


    def _retry_delay(self, attempt: int, response: Any) -> Optional[float]:
        """返回下次尝试前的等待秒数；None 表示应明确停止重试。

        Retry-After 被完整尊重（不截断），只加正 ε，绝不早于服务端要求的
        间隔发起重试。
        """
        retry_after = _extract_retry_after_seconds(self._extract_retry_after_raw(response))
        if retry_after is not None:
            if retry_after > _MAX_RETRY_AFTER_WAIT:
                return None
            return retry_after + random.uniform(0.0, 0.25)
        base = min(2.0 ** (attempt - 1), 30.0)
        return base * random.uniform(0.75, 1.25)


    @staticmethod
    def _extract_retry_after_raw(response: Any) -> Optional[Any]:
        """取出原始 Retry-After 值；response 为 None（传输错误）时返回 None。"""
        if response is None:
            return None
        headers = getattr(response, "headers", None)
        if headers is not None:
            try:
                value = headers.get("retry-after")
            except Exception:
                value = None
            if value is not None:
                return value
        return None


    def _rate_gate(self) -> None:
        """requests_per_minute > 0 时按单调时钟节流（线程安全）。"""
        if self._min_interval <= 0.0:
            return
        while True:
            with self._rate_lock:
                now = time.monotonic()
                wait = self._next_send_monotonic - now
                if wait <= 0.0:
                    self._next_send_monotonic = max(now, self._next_send_monotonic) + self._min_interval
                    return
            time.sleep(min(wait, 0.5))


    def _accumulate_usage(self, usage: Any) -> None:
        if not isinstance(usage, dict):
            return
        input_tokens = usage.get("input_tokens", usage.get("prompt_tokens", 0))
        output_tokens = usage.get("output_tokens", usage.get("completion_tokens", 0))
        if not _is_finite_number(input_tokens) or not _is_finite_number(output_tokens):
            return
        with self._stats_lock:
            self._usage["input_tokens"] += int(input_tokens)
            self._usage["output_tokens"] += int(output_tokens)


    def _bump(self, *, request: int = 0, retry: int = 0, failure: int = 0) -> None:
        with self._stats_lock:
            self._request_count += request
            self._retry_count += retry
            self._failure_count += failure


# Both hosted Jev and local Kev use this single transport implementation.
JevClient = SystemOneClient


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _is_nonnegative_int(value: Any) -> bool:
    """严格非负整数判断（bool 与浮点均拒绝，不做静默 coerce）。"""
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _validate_base_url(url: Any, field: str) -> None:
    """base_url 校验：仅 http/https，禁止内嵌 userinfo 凭据与 query。

    密钥只能通过 api_key_env 环境变量提供；错误信息不回显 URL 原文，
    以防 URL 路径段中携带敏感串被写入日志。
    """
    if not isinstance(url, str) or not url.strip():
        raise ValueError(f"{field} 必须是非空字符串")
    parts = urlsplit(url.strip())
    if parts.scheme not in ("http", "https"):
        raise ValueError(f"{field} 仅支持 http/https URL")
    if parts.username is not None or parts.password is not None:
        raise ValueError(f"{field} 不允许内嵌 userinfo 凭据；密钥请改用 api_key_env 环境变量")
    if parts.query:
        raise ValueError(f"{field} 不允许携带 query（可能携带密钥）；请改用 api_key_env 环境变量")


def _extract_retry_after_seconds(raw: Any) -> Optional[float]:
    """Retry-After 值（秒数字符串/数值或 HTTP-date）-> 非负秒数。

    不可解析或 NaN/inf 返回 None（交由指数退避兜底），绝不返回负数。
    """
    if raw is None:
        return None
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return None
        try:
            seconds = float(text)
        except ValueError:
            return _http_date_delay_seconds(text)
        return max(0.0, seconds) if _is_finite_number(seconds) else None
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        if not _is_finite_number(raw):
            return None
        return max(0.0, float(raw))
    return None


def _http_date_delay_seconds(text: str) -> Optional[float]:
    """可解析 HTTP-date 的 Retry-After -> 距此刻的非负秒数。"""
    try:
        when = parsedate_to_datetime(text)
    except (TypeError, ValueError, IndexError):
        return None
    if when is None:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return max(0.0, (when - datetime.now(timezone.utc)).total_seconds())


def _as_confidence(value: Any) -> Optional[float]:
    if not _is_finite_number(value):
        return None
    return float(value)


def _normalize_probabilities(value: Any) -> Optional[Dict[str, float]]:
    """严格归整 probabilities 的形状：非法内容抛 ValueError，绝不静默丢弃条目。

    数值合法性（负值/未知级别/不归一）由 MeasurementSpec.resolve_score_factor_value
    进一步校验；这里只保证"能被当作分布读取"。
    """
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("probabilities 不是 JSON 对象")
    normalized: Dict[str, float] = {}
    for key, probability in value.items():
        if not _is_finite_number(probability):
            raise ValueError(f"probabilities 的级别 {key!r} 不是有限数值")
        normalized[str(key)] = float(probability)
    return normalized


def _optional_float(value: Any) -> Optional[float]:
    if not _is_finite_number(value):
        return None
    return float(value)


