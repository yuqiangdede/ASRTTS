from __future__ import annotations

import json
import logging
import time
from typing import Any
from urllib import error, parse, request


logger = logging.getLogger(__name__)

_LOG_TEXT_LIMIT = 500


class AsrCorrectionError(RuntimeError):
    pass


def _clip_log_text(text: str, *, limit: int = _LOG_TEXT_LIMIT) -> str:
    value = str(text or "")
    if len(value) <= limit:
        return value
    return f"{value[:limit]}...<truncated {len(value) - limit} chars>"


class AsrCorrectionClient:
    def __init__(self, config: dict[str, Any]) -> None:
        self.enabled = bool(config.get("enabled", False))
        self.api_url = str(config.get("api_url", "http://localhost:1234/v1/chat/completions") or "").strip()
        self.model = str(config.get("model", "qwen3.5-2b") or "").strip()
        self.api_style = str(config.get("api_style", "auto") or "auto").strip().lower()
        self.system_prompt = str(
            config.get(
                "system_prompt",
                (
                    "你是 ASR 文本纠错助手。"
                    "任务：只修正语音识别导致的同音字、近音词、错别字和明显不通顺的词。"
                    "保持原意，不扩写，不解释，不总结。"
                    "如果原文已经正确，原样输出。"
                    "只输出纠正后的最终文本。"
                ),
            )
            or ""
        ).strip()
        token = str(config.get("token", "") or "").strip()
        api_key = str(config.get("api_key", "") or "").strip()
        self.api_key = api_key or token or None
        self.temperature = float(config.get("temperature", 0.2) or 0.2)
        self.connect_timeout_s = float(config.get("connect_timeout_s", 3.0) or 3.0)
        self.read_timeout_s = float(config.get("read_timeout_s", 30.0) or 30.0)
        configured_total_timeout = config.get("total_timeout_s", None)
        if configured_total_timeout in (None, ""):
            self.total_timeout_s = self.connect_timeout_s + self.read_timeout_s
        else:
            self.total_timeout_s = float(configured_total_timeout or 0.0)
        # urllib 这里只接受一个 timeout 参数，避免 total_timeout 比读取超时更短导致提前超时。
        self.total_timeout_s = max(self.total_timeout_s, self.connect_timeout_s, self.read_timeout_s)
        self.max_retries = int(config.get("max_retries", 1) or 1)
        self.backoff_s = float(config.get("backoff_s", 0.4) or 0.4)
        self.use_system_proxy = bool(config.get("use_system_proxy", False))

    def _build_system_prompt(
        self,
        *,
        domain: str | None = None,
        prompt_terms: list[str] | None = None,
        phrase_rule_hints: list[str] | None = None,
    ) -> str:
        parts = [self.system_prompt]
        if domain:
            parts.append(f"当前业务域：{domain}。")
        terms = [str(item).strip() for item in (prompt_terms or []) if str(item).strip()]
        if terms:
            parts.append(f"优先保留或修正为以下业务词汇：{'，'.join(terms)}。")
        hints = [str(item).strip() for item in (phrase_rule_hints or []) if str(item).strip()]
        if hints:
            parts.append(f"常见近音纠正规则参考：{'；'.join(hints)}。")
        return "\n".join(part for part in parts if part)

    def _extract_text(self, data: Any) -> str:
        if isinstance(data, str):
            return data.strip()
        if not isinstance(data, dict):
            return ""

        direct_keys = ("output", "response", "content", "text", "answer")
        for key in direct_keys:
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

        message = data.get("message")
        if isinstance(message, dict):
            value = message.get("content")
            if isinstance(value, str) and value.strip():
                return value.strip()

        choices = data.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                message = first.get("message")
                if isinstance(message, dict):
                    value = message.get("content")
                    if isinstance(value, str) and value.strip():
                        return value.strip()
                value = first.get("text")
                if isinstance(value, str) and value.strip():
                    return value.strip()

        return ""

    def _derive_legacy_url(self, url: str) -> str | None:
        if not url:
            return None
        parsed = parse.urlsplit(url)
        path = parsed.path or ""
        if path.endswith("/v1/chat/completions"):
            new_path = f"{path[:-len('/v1/chat/completions')]}/api/v1/chat"
            return parse.urlunsplit((parsed.scheme, parsed.netloc, new_path, parsed.query, parsed.fragment))
        return None

    def _derive_openai_url(self, url: str) -> str | None:
        if not url:
            return None
        parsed = parse.urlsplit(url)
        path = parsed.path or ""
        if path.endswith("/api/v1/chat"):
            new_path = f"{path[:-len('/api/v1/chat')]}/v1/chat/completions"
            return parse.urlunsplit((parsed.scheme, parsed.netloc, new_path, parsed.query, parsed.fragment))
        return None

    def _build_candidates(self) -> list[tuple[str, str]]:
        candidates: list[tuple[str, str]] = []
        url = self.api_url
        if not url:
            return candidates

        if self.api_style == "openai":
            return [("openai", url)]
        if self.api_style == "legacy":
            return [("legacy", url)]

        if url.endswith("/v1/chat/completions"):
            candidates.append(("openai", url))
            legacy_url = self._derive_legacy_url(url)
            if legacy_url:
                candidates.append(("legacy", legacy_url))
        elif url.endswith("/api/v1/chat"):
            candidates.append(("legacy", url))
            openai_url = self._derive_openai_url(url)
            if openai_url:
                candidates.append(("openai", openai_url))
        else:
            candidates.append(("openai", url))
            legacy_url = self._derive_legacy_url(url)
            if legacy_url:
                candidates.append(("legacy", legacy_url))

        deduped: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for item in candidates:
            if item in seen:
                continue
            seen.add(item)
            deduped.append(item)
        return deduped

    def _build_payload(self, *, style: str, system_prompt: str, content: str) -> dict[str, Any]:
        if style == "legacy":
            return {
                "model": self.model,
                "system_prompt": system_prompt,
                "input": content,
                "temperature": self.temperature,
            }
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            "temperature": self.temperature,
            "stream": False,
        }

    def _post_json(self, *, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        req = request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        opener = request.build_opener() if self.use_system_proxy else request.build_opener(request.ProxyHandler({}))
        with opener.open(req, timeout=self.total_timeout_s) as response:
            status_code = getattr(response, "status", None) or response.getcode()
            response_text = response.read().decode("utf-8", errors="replace")

        if status_code >= 400:
            raise AsrCorrectionError(f"纠错接口返回异常状态码：{status_code}")

        try:
            return json.loads(response_text)
        except json.JSONDecodeError as exc:
            raise AsrCorrectionError(f"纠错接口返回了无效 JSON：{exc}") from exc

    def correct(
        self,
        text: str,
        *,
        domain: str | None = None,
        prompt_terms: list[str] | None = None,
        phrase_rule_hints: list[str] | None = None,
    ) -> str:
        content = str(text or "").strip()
        if not content or not self.enabled:
            return content

        full_terms = [str(item).strip() for item in (prompt_terms or []) if str(item).strip()]
        full_phrase_rule_hints = [str(item).strip() for item in (phrase_rule_hints or []) if str(item).strip()]
        system_prompt = self._build_system_prompt(
            domain=domain,
            prompt_terms=full_terms,
            phrase_rule_hints=full_phrase_rule_hints,
        )
        candidates = self._build_candidates()
        if not candidates:
            raise AsrCorrectionError("未配置有效的纠错接口地址。")

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            for style, url in candidates:
                payload = self._build_payload(style=style, system_prompt=system_prompt, content=content)
                logger.info(
                    "ASR LLM correction request: style=%s url=%s model=%s temperature=%s attempt=%s domain=%s prompt_terms=%s phrase_rule_hints=%s",
                    style,
                    url,
                    self.model,
                    self.temperature,
                    attempt + 1,
                    domain or "",
                    ",".join(full_terms),
                    " | ".join(full_phrase_rule_hints),
                )
                logger.info("ASR LLM correction proxy mode: use_system_proxy=%s", self.use_system_proxy)
                logger.info("ASR LLM correction prompt(system): %s", _clip_log_text(system_prompt))
                logger.info("ASR LLM correction prompt(user): %s", _clip_log_text(content))

                try:
                    data = self._post_json(url=url, payload=payload)
                    corrected = self._extract_text(data)
                    if not corrected:
                        raise AsrCorrectionError(f"纠错接口返回空结果：style={style} url={url}")
                    logger.info(
                        "ASR LLM correction response: style=%s url=%s changed=%s output=%s",
                        style,
                        url,
                        corrected.strip() != content,
                        _clip_log_text(corrected),
                    )
                    return corrected
                except error.HTTPError as exc:
                    error_body = ""
                    try:
                        error_body = exc.read().decode("utf-8", errors="replace")
                    except Exception:
                        error_body = ""
                    last_error = AsrCorrectionError(f"纠错接口返回 HTTP {exc.code}")
                    logger.warning(
                        "ASR LLM correction HTTP error: style=%s url=%s code=%s body=%s",
                        style,
                        url,
                        exc.code,
                        _clip_log_text(error_body),
                    )
                except error.URLError as exc:
                    last_error = AsrCorrectionError(f"纠错接口连接失败：{exc.reason}")
                    logger.warning("ASR LLM correction URL error: style=%s url=%s reason=%s", style, url, exc.reason)
                except AsrCorrectionError as exc:
                    last_error = exc
                    logger.warning("ASR LLM correction protocol error: style=%s url=%s error=%s", style, url, exc)
                except Exception as exc:  # noqa: BLE001
                    last_error = AsrCorrectionError(f"纠错接口调用失败：{exc}")
                    logger.exception("ASR LLM correction unexpected error: style=%s url=%s", style, url)

            if attempt < self.max_retries:
                sleep_s = self.backoff_s * (2**attempt)
                logger.warning(
                    "ASR LLM correction retry: attempt=%s/%s sleep=%.2fs last_error=%s",
                    attempt + 1,
                    self.max_retries + 1,
                    sleep_s,
                    last_error,
                )
                time.sleep(sleep_s)

        raise AsrCorrectionError(
            f"ASR 语义纠错请求失败（已重试 {self.max_retries} 次）：{last_error}"
        )
