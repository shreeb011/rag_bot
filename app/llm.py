"""Chat-completion client for any OpenAI-compatible endpoint (OpenAI, Ollama, Groq, vLLM, LM Studio, ...)."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from typing import Protocol

import httpx

from app.config import Settings
from app.errors import UpstreamModelError

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


class LLM(Protocol):
    @property
    def name(self) -> str: ...

    def complete(self, system: str, user: str) -> str: ...

    def stream(self, system: str, user: str) -> Iterator[str]: ...


def strip_thinking(text: str) -> str:
    return _THINK_RE.sub("", text).strip()


class OpenAICompatibleLLM:
    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str,
        temperature: float = 0.0,
        max_tokens: int = 350,
        timeout_s: float = 180.0,
        keep_alive: str | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.keep_alive = keep_alive
        self._client = client or httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout_s,
        )

    @property
    def name(self) -> str:
        return self.model

    def ping(self) -> None:
        """Cheap reachability check for readiness probes (lists models, doesn't run one)."""
        self._client.get("/models", timeout=3.0).raise_for_status()

    def _payload(self, system: str, user: str, stream: bool, max_tokens: int | None = None) -> dict:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
            "stream": stream,
        }
        if self.keep_alive:
            payload["keep_alive"] = self.keep_alive  # Ollama extension: keep the model loaded between requests
        return payload

    def complete(self, system: str, user: str, max_tokens: int | None = None) -> str:
        try:
            response = self._client.post("/chat/completions", json=self._payload(system, user, False, max_tokens))
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
        except httpx.HTTPError as exc:
            raise UpstreamModelError(f"LLM request failed: {exc}") from exc
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise UpstreamModelError(f"Malformed LLM response: {exc}") from exc
        if not isinstance(content, str):
            raise UpstreamModelError("LLM returned no text content")
        return strip_thinking(content)

    def stream(self, system: str, user: str) -> Iterator[str]:
        """Yield text deltas as they are generated (server-sent events)."""
        try:
            with self._client.stream("POST", "/chat/completions", json=self._payload(system, user, True)) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    choices = json.loads(data).get("choices") or []
                    delta = (choices[0].get("delta") or {}).get("content") if choices else None
                    if delta:
                        yield delta
        except httpx.HTTPError as exc:
            raise UpstreamModelError(f"LLM request failed: {exc}") from exc
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise UpstreamModelError(f"Malformed LLM stream: {exc}") from exc


def build_llm(settings: Settings) -> LLM:
    return OpenAICompatibleLLM(
        model=settings.llm_model,
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key.get_secret_value(),
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        timeout_s=settings.llm_timeout_s,
        keep_alive=settings.keep_alive or None,
    )
