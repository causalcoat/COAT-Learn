"""
传统 LLM API 后端：OpenAI 兼容 chat 模型的最小同步封装。

设计约定（与 core 的并行开发契约一致）：
- 构造函数不发请求、不读取密钥、不触发任何第三方 import；首次显式调用
  :meth:`OpenAIChatModel.chat` 时才按环境变量 ``config.api_key_env`` 建立
  SDK client（懒加载 + 双检锁，线程安全），此后全部调用共享同一个 client。
- SDK 自带重试关闭（max_retries=0），重试/限流/并发统一由本模块控制：
  外层有界信号量（max_concurrency）+ 每次 HTTP 尝试（含重试）都执行的
  RPM 节流 + 指数退避。Retry-After（秒数或 HTTP-date）被完整尊重，只加
  正 ε，绝不提早重试；服务端要求的等待超过 ``_MAX_RETRY_AFTER_WAIT``
  上限时明确停止重试而不是提前重试。
- 配置在构造 :class:`ModelConfig` 时即校验（model/api_key_env 非空、
  数值类型与范围严格检查、base_url 拒绝内嵌 userinfo 凭据和任何 query，
  密钥只能走 api_key_env 环境变量）；max_retries=0 精确表示不重试；
  无效输入显式报错，绝不静默 coerce。
- 不读取 secret.yaml、不写任何配置或外网文件、不改写代理相关环境变量；
  异常信息只含模型名/状态码/异常类型，绝不打印 prompt、密钥或 provider
  原始错误文本。
- 主模型（knowledge）与标注模型（annotation）各用独立 ModelConfig 实例化，
  互不影响；并发能力来自调用方已有的 ThreadPoolExecutor，本模块不启动
  任何额外服务，也不提供默认模型（model 必填）。
"""

from __future__ import annotations

import math
import os
import random
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Dict, List, Optional, Protocol, Union, runtime_checkable
from urllib.parse import urlsplit

__all__ = [
    "ModelConfig",
    "ChatResult",
    "ModelStats",
    "ChatModel",
    "ModelRuntimeError",
    "OpenAIChatModel",
    "create_model",
]

# 服务端 Retry-After 超过该上限（秒）时明确停止重试，绝不提早重试
_MAX_RETRY_AFTER_WAIT = 120.0


def _is_finite_number(value: Any) -> bool:
    """有限数值判断（bool 与 NaN/Inf 均排除）。"""
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
        raise ValueError(f"{field} 必须是非空字符串或 None")
    parts = urlsplit(url.strip())
    if parts.scheme not in ("http", "https"):
        raise ValueError(f"{field} 仅支持 http/https URL")
    if parts.username is not None or parts.password is not None:
        raise ValueError(
            f"{field} 不允许内嵌 userinfo 凭据；密钥请改用 api_key_env 环境变量"
        )
    if parts.query:
        raise ValueError(
            f"{field} 不允许携带 query（可能携带密钥）；请改用 api_key_env 环境变量"
        )


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


@dataclass(frozen=True)
class ModelConfig:
    """单个 LLM 角色的连接与采样配置。密钥只能来自环境变量 ``api_key_env``。

    字段是 core 侧并行开发的公共契约：如需改动必须先通知所有相关方。
    """

    model: str
    base_url: Optional[str] = None
    api_key_env: str = "OPENAI_API_KEY"
    temperature: Optional[float] = 0.0
    max_tokens: Optional[int] = None
    timeout: float = 60.0
    max_retries: int = 3
    max_concurrency: int = 16
    requests_per_minute: Optional[float] = None
    reasoning_effort: Optional[str] = None

    def __post_init__(self) -> None:
        """构造即校验；输出上限和 reasoning 参数为 None 时不发送该字段。"""
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("ModelConfig.model 必须是非空字符串")
        if not isinstance(self.api_key_env, str) or not self.api_key_env.strip():
            raise ValueError("ModelConfig.api_key_env 必须是非空字符串")
        if self.base_url is not None:
            _validate_base_url(self.base_url, "ModelConfig.base_url")
        if self.reasoning_effort is not None and not (
            isinstance(self.reasoning_effort, str) and self.reasoning_effort.strip()
        ):
            raise ValueError("ModelConfig.reasoning_effort 必须为非空字符串或 None")
        if self.temperature is not None and not _is_finite_number(self.temperature):
            raise ValueError("ModelConfig.temperature 必须为有限数值或 None")
        if self.max_tokens is not None and not (
            _is_nonnegative_int(self.max_tokens) and self.max_tokens > 0
        ):
            raise ValueError("ModelConfig.max_tokens 必须为正整数或 None")
        if not (_is_finite_number(self.timeout) and self.timeout > 0):
            raise ValueError("ModelConfig.timeout 必须为正的有限数值")
        if not _is_nonnegative_int(self.max_retries):
            raise ValueError("ModelConfig.max_retries 必须为非负整数（0 表示不重试）")
        if not (_is_nonnegative_int(self.max_concurrency) and self.max_concurrency >= 1):
            raise ValueError("ModelConfig.max_concurrency 必须为 >=1 的整数")
        if self.requests_per_minute is not None and not (
            _is_finite_number(self.requests_per_minute) and self.requests_per_minute > 0
        ):
            raise ValueError(
                "ModelConfig.requests_per_minute 必须为正的有限数值或 None"
            )

    def redacted(self) -> Dict[str, Any]:
        """无密钥配置快照，可安全写入实验日志。"""
        return {
            "model": self.model,
            "base_url": self.base_url,
            "api_key_env": self.api_key_env,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "max_concurrency": self.max_concurrency,
            "requests_per_minute": self.requests_per_minute,
            "reasoning_effort": self.reasoning_effort,
        }


@dataclass(frozen=True)
class ChatResult:
    """一次成功 chat 的结构化结果；``return_text=True`` 时调用方只取 ``text``。"""

    text: str
    message: Dict[str, Any]
    finish_reason: Optional[str]
    usage: Dict[str, int]
    model: Optional[str]


@dataclass(frozen=True)
class ModelStats:
    """累计计数的安全只读快照。

    request_count 按每次 HTTP 尝试计；retry_count 是其中的重试次数；
    failure_count 是未成功（最终抛错或触发重试）的尝试数。
    """

    request_count: int = 0
    retry_count: int = 0
    failure_count: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ModelRuntimeError(RuntimeError):
    """经过脱敏的模型调用失败：不含 prompt、密钥或 provider 原始错误文本。"""


@runtime_checkable
class ChatModel(Protocol):
    """core 与各算法依赖的最小模型接口；鸭子类型即可满足，不强制继承。"""

    config: ModelConfig

    def chat(
        self,
        messages: Union[str, Sequence[Mapping[str, Any]]],
        return_text: bool = False,
        **kwargs: Any,
    ) -> Union["ChatResult", str]:
        ...


class OpenAIChatModel:
    """OpenAI 兼容 chat 模型的同步薄封装（懒加载、共享 client、可控并发）。

    不兼容旧版 openai<1.0 的 ``ChatCompletion`` 路径；需要 ``openai>=1.x``。
    """

    def __init__(self, config: ModelConfig) -> None:
        if not isinstance(config, ModelConfig):
            raise TypeError("config 必须是 ModelConfig")
        self.config = config
        self._client: Optional[Any] = None
        self._client_lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self._request_count = 0
        self._retry_count = 0
        self._failure_count = 0
        self._usage: Dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        self._inflight = threading.BoundedSemaphore(int(config.max_concurrency))
        self._rate_lock = threading.Lock()
        rpm = config.requests_per_minute
        self._min_interval = (60.0 / rpm) if rpm is not None else 0.0
        self._next_send_monotonic = 0.0

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: Union[str, Sequence[Mapping[str, Any]]],
        return_text: bool = False,
        **kwargs: Any,
    ) -> Union[ChatResult, str]:
        """同步发送一轮 chat。

        messages 传 str 时按单条 user 消息处理；kwargs 透传给
        ``client.chat.completions.create``（如 response_format），但不允许
        覆盖 model/messages。默认返回 :class:`ChatResult`；
        ``return_text=True`` 为兼容旧代码返回 str。
        """
        payload_messages = self._normalize_messages(messages)
        params: Dict[str, Any] = {"model": self.config.model, "messages": payload_messages}
        if self.config.temperature is not None:
            params["temperature"] = self.config.temperature
        if self.config.max_tokens is not None:
            params["max_tokens"] = self.config.max_tokens
        if self.config.reasoning_effort is not None:
            params["reasoning_effort"] = self.config.reasoning_effort
        for forbidden in ("model", "messages"):
            if forbidden in kwargs:
                raise ValueError(f"chat 的 kwargs 不允许覆盖 {forbidden!r}，请修改 ModelConfig")
        params.update(kwargs)

        client = self._ensure_client()
        with self._inflight:
            # RPM 节流在 _create_with_retries 内按每次实际 HTTP 尝试执行
            response = self._create_with_retries(client, params)

        usage = self._extract_usage(response)
        self._accumulate_usage(usage)
        choices = list(getattr(response, "choices", None) or [])
        if not choices:
            self._bump(failure=1)
            raise ModelRuntimeError(f"chat 返回缺少 choices（model={self.config.model!r}）")
        choice = choices[0]
        message_obj = getattr(choice, "message", None)
        content = getattr(message_obj, "content", None)
        result = ChatResult(
            text=content if isinstance(content, str) else "",
            message={
                "role": getattr(message_obj, "role", "assistant"),
                "content": content,
            },
            finish_reason=getattr(choice, "finish_reason", None),
            usage=usage,
            model=getattr(response, "model", None),
        )
        if return_text:
            return result.text
        return result

    def stats(self) -> ModelStats:
        """安全只读快照，便于结果诊断；可在任意线程调用。"""
        with self._stats_lock:
            return ModelStats(
                request_count=self._request_count,
                retry_count=self._retry_count,
                failure_count=self._failure_count,
                prompt_tokens=self._usage["prompt_tokens"],
                completion_tokens=self._usage["completion_tokens"],
                total_tokens=self._usage["total_tokens"],
            )

    @property
    def request_count(self) -> int:
        return self.stats().request_count

    @property
    def failure_count(self) -> int:
        return self.stats().failure_count

    def __repr__(self) -> str:  # config 本身不含密钥，可直接展示
        return f"OpenAIChatModel(config={self.config!r})"

    # ------------------------------------------------------------------
    # client 懒加载（双检锁；构造函数绝不进入这里）
    # ------------------------------------------------------------------

    def _ensure_client(self) -> Any:
        client = self._client
        if client is not None:
            return client
        with self._client_lock:
            if self._client is None:
                try:
                    import openai
                except ImportError:
                    raise ModelRuntimeError(
                        "chat 需要 openai>=1.x SDK；当前环境未安装，无法初始化客户端"
                    ) from None
                api_key = os.environ.get(self.config.api_key_env)
                if not api_key:
                    raise ModelRuntimeError(
                        f"环境变量 {self.config.api_key_env} 未设置，"
                        f"无法为模型 {self.config.model!r} 初始化客户端"
                    )
                # 不触碰代理相关环境变量；SDK 的重试关闭，外层统一控制
                self._client = openai.OpenAI(
                    api_key=api_key,
                    base_url=self.config.base_url,
                    timeout=self.config.timeout,
                    max_retries=0,
                )
            return self._client

    # ------------------------------------------------------------------
    # 请求、重试与限流
    # ------------------------------------------------------------------

    def _create_with_retries(self, client: Any, params: Dict[str, Any]) -> Any:
        import openai

        max_attempts = self.config.max_retries + 1  # max_retries=0 恰好一次尝试
        last_error: Optional[BaseException] = None
        for attempt in range(1, max_attempts + 1):
            self._rate_gate()  # 每次实际 HTTP 尝试（含重试）都过配额闸门
            self._bump(request=1)
            try:
                return client.chat.completions.create(**params)
            except Exception as exc:
                if not self._is_retryable_openai_error(openai, exc):
                    self._bump(failure=1)
                    raise ModelRuntimeError(
                        self._sanitize_failure(attempt, exc)
                    ) from None
                last_error = exc
                if attempt >= max_attempts:
                    break
                delay = self._retry_delay(attempt, exc)
                if delay is None:  # 服务端要求的等待超过上限：明确停止，不提早重试
                    self._bump(failure=1)
                    raise ModelRuntimeError(
                        self._sanitize_failure(attempt, exc)
                        + f"；服务端 Retry-After 超过 {_MAX_RETRY_AFTER_WAIT:.0f}s 上限，已明确停止重试"
                    ) from None
                self._bump(retry=1, failure=1)
                time.sleep(delay)
        self._bump(failure=1)
        raise ModelRuntimeError(
            self._sanitize_failure(max_attempts, last_error)
            + f"；已在 {max_attempts} 次尝试后退避重试完毕"
        )

    @staticmethod
    def _is_retryable_openai_error(openai: Any, exc: BaseException) -> bool:
        if isinstance(exc, openai.RateLimitError):
            return True
        if isinstance(exc, openai.APIConnectionError):  # 含 APITimeoutError
            return True
        if isinstance(exc, openai.APIStatusError):
            status = getattr(exc, "status_code", 0) or 0
            return status >= 500
        return False

    def _retry_delay(self, attempt: int, exc: BaseException) -> Optional[float]:
        """返回下次尝试前的等待秒数；None 表示应明确停止重试。

        Retry-After 被完整尊重（不截断），只加正 ε，绝不早于服务端要求的
        间隔发起重试。
        """
        retry_after = _extract_retry_after_seconds(self._extract_retry_after_raw(exc))
        if retry_after is not None:
            if retry_after > _MAX_RETRY_AFTER_WAIT:
                return None
            return retry_after + random.uniform(0.0, 0.25)
        base = min(2.0 ** (attempt - 1), 30.0)
        return base * random.uniform(0.75, 1.25)

    @staticmethod
    def _extract_retry_after_raw(exc: Optional[BaseException]) -> Optional[Any]:
        """取出原始 Retry-After 值（头字段优先，其次 SDK 提供的属性）。"""
        if exc is None:
            return None
        headers = getattr(getattr(exc, "response", None), "headers", None)
        if headers is not None:
            try:
                value = headers.get("retry-after")
            except Exception:
                value = None
            if value is not None:
                return value
        return getattr(exc, "retry_after", None)

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

    # ------------------------------------------------------------------
    # 输入/输出整理与统计
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_messages(
        messages: Union[str, Sequence[Mapping[str, Any]]],
    ) -> List[Dict[str, Any]]:
        if isinstance(messages, str):
            return [{"role": "user", "content": messages}]
        if isinstance(messages, (list, tuple)):
            if not messages:
                raise ValueError("messages 不能为空")
            normalized: List[Dict[str, Any]] = []
            for index, item in enumerate(messages):
                if not isinstance(item, Mapping):
                    raise TypeError(f"messages[{index}] 必须是含 role/content 的映射")
                role = item.get("role")
                content = item.get("content")
                if not isinstance(role, str) or not role:
                    raise TypeError(f"messages[{index}] 缺少有效的 role")
                if not isinstance(content, (str, list)):
                    raise TypeError(f"messages[{index}] 的 content 必须是字符串或多模态内容列表")
                normalized.append({"role": role, "content": content})
            return normalized
        raise TypeError("messages 必须是 str 或 list")

    @staticmethod
    def _extract_usage(response: Any) -> Dict[str, int]:
        usage_obj = getattr(response, "usage", None)

        def _int(field: str) -> int:
            return int(getattr(usage_obj, field, 0) or 0)

        return {
            "prompt_tokens": _int("prompt_tokens"),
            "completion_tokens": _int("completion_tokens"),
            "total_tokens": _int("total_tokens"),
        }

    def _accumulate_usage(self, usage: Dict[str, int]) -> None:
        with self._stats_lock:
            self._usage["prompt_tokens"] += usage["prompt_tokens"]
            self._usage["completion_tokens"] += usage["completion_tokens"]
            self._usage["total_tokens"] += usage["total_tokens"]

    def _bump(self, *, request: int = 0, retry: int = 0, failure: int = 0) -> None:
        with self._stats_lock:
            self._request_count += request
            self._retry_count += retry
            self._failure_count += failure

    def _sanitize_failure(self, attempt: int, exc: Optional[BaseException]) -> str:
        status = getattr(exc, "status_code", None)
        parts = [
            f"model={self.config.model!r}",
            f"attempt={attempt}",
            f"error_type={type(exc).__name__}",
        ]
        if status:
            parts.append(f"status={status}")
        return "chat 调用失败（" + ", ".join(parts) + "）；prompt、密钥与 provider 原始错误已省略"


def create_model(config: ModelConfig) -> OpenAIChatModel:
    """按配置构建 OpenAI 兼容模型（不做连通性检查、不选择默认模型）。"""
    return OpenAIChatModel(config)
