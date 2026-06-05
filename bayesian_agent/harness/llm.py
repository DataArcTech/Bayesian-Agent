"""OpenAI-compatible client for the first-party Bayesian-Agent harness."""

from __future__ import annotations

import json
import http.client
import re
import ssl
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence


DEFAULT_BASE_URL = "https://api.deepseek.com"


@dataclass
class OpenAIChatClient:
    """Tiny stdlib OpenAI-compatible chat-completions client."""

    api_key: str
    model: str = "deepseek-v4-flash"
    base_url: str = DEFAULT_BASE_URL
    max_tokens: int = 4096
    temperature: float = 0.0
    timeout_seconds: int = 180
    verify_ssl: bool = True
    host_header: str = ""
    max_retries: int = 4

    def chat(self, messages: Sequence[Mapping[str, Any]], tools: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": list(messages),
            "tools": list(tools),
            "tool_choice": "auto",
            "temperature": float(self.temperature),
            "max_tokens": int(self.max_tokens),
        }
        data = self._post_json(_auto_make_url(self.base_url, "chat/completions"), payload)
        choice = (data.get("choices") or [{}])[0]
        message = dict(choice.get("message") or {})
        tool_calls = []
        for raw_call in message.get("tool_calls") or []:
            function = dict(raw_call.get("function") or {})
            tool_calls.append(
                {
                    "id": str(raw_call.get("id") or ""),
                    "name": str(function.get("name") or ""),
                    "arguments": _parse_arguments(function.get("arguments")),
                }
            )
        return {
            "content": str(message.get("content") or ""),
            "tool_calls": tool_calls,
            "usage": normalize_usage(data.get("usage") or {}),
            "finish_reason": str(choice.get("finish_reason") or ""),
            "raw": data,
        }

    def _post_json(self, url: str, payload: Mapping[str, Any]) -> Dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.host_header:
            headers["Host"] = self.host_header
        context = None if self.verify_ssl else ssl._create_unverified_context()
        last_error: Optional[Exception] = None
        for attempt in range(int(self.max_retries or 0) + 1):
            request = urllib.request.Request(url, data=body, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(request, timeout=int(self.timeout_seconds), context=context) as response:
                    raw = response.read().decode("utf-8", errors="replace")
                return json.loads(raw)
            except (urllib.error.URLError, TimeoutError, http.client.RemoteDisconnected, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt >= int(self.max_retries or 0):
                    break
                time.sleep(min(2 ** attempt, 4))
        raise RuntimeError(f"Chat completion request failed: {last_error}")


def normalize_usage(raw: Mapping[str, Any]) -> Dict[str, int]:
    prompt = int(raw.get("prompt_tokens") or raw.get("input_tokens") or 0)
    completion = int(raw.get("completion_tokens") or raw.get("output_tokens") or 0)
    total = int(raw.get("total_tokens") or prompt + completion)
    return {"input_tokens": prompt, "output_tokens": completion, "total_tokens": total}


def _parse_arguments(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return dict(raw)
    if not raw:
        return {}
    try:
        parsed = json.loads(str(raw))
    except json.JSONDecodeError:
        return {"_raw": str(raw)}
    return dict(parsed) if isinstance(parsed, dict) else {"value": parsed}


def _auto_make_url(base: str, path: str) -> str:
    base = base.rstrip("/")
    path = path.strip("/")
    if base.endswith("$"):
        return base[:-1].rstrip("/")
    if base.endswith(path):
        return base
    if re.search(r"/v\d+(/|$)", base):
        return f"{base}/{path}"
    return f"{base}/v1/{path}"
