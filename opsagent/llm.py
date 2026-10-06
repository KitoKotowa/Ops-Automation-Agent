"""LLM providers behind one interface: structured output parsed into a Pydantic model.

Each provider turns on when its credentials are in the environment (see .env.example).
Model IDs can be overridden per provider with *_MODEL env vars, or per call.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)
ROOT = Path(__file__).resolve().parent.parent


def load_dotenv(path: Path = ROOT / ".env") -> None:
    """Minimal .env loader. Real environment variables win over the file."""
    if not path.exists():
        return
    # utf-8-sig: Windows editors may add a BOM (Python on Windows defaults to cp1252).
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        key, sep, value = line.strip().partition("=")
        if sep and not key.startswith("#") and value.strip():
            os.environ.setdefault(key.strip(), value.strip().strip('"'))


class LLMError(RuntimeError):
    pass


class NotConfigured(LLMError):
    pass


class Provider:
    key: str
    label: str
    default_model: str
    model_env: str

    @property
    def model(self) -> str:
        return os.environ.get(self.model_env) or self.default_model

    def available(self) -> bool:
        raise NotImplementedError

    def parse(self, system: str, user: str, schema: type[T], model: Optional[str] = None) -> T:
        raise NotImplementedError

    def info(self) -> dict:
        return {"key": self.key, "label": self.label, "model": self.model, "available": self.available()}


class AnthropicProvider(Provider):
    key, label, default_model, model_env = "anthropic", "Claude", "claude-opus-5-5", "ANTHROPIC_MODEL"
    _client = None

    def available(self) -> bool:
        return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))

    def parse(self, system, user, schema, model=None):
        import anthropic

        if self._client is None:
            self._client = anthropic.Anthropic(max_retries=3, timeout=120)
        response = self._client.beta.messages.parse(
            model=model or self.model,
            max_tokens=16000,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_format=schema,
            output_config={"effort": "medium"},
            # On a safety-classifier refusal, the API re-runs the request on a fallback model server-side.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        if response.stop_reason == "refusal":
            raise LLMError("LLM declined to answer")
        if response.stop_reason == "max_tokens":
            raise LLMError("LLM output was truncated")
        if response.parsed_output is None:
            raise LLMError("LLM output did not match the schema")
        return response.parsed_output


class OpenAIProvider(Provider):
    key, label, default_model, model_env = "openai", "OpenAI", "gpt-6-sol", "OPENAI_MODEL"
    _client = None

    def available(self) -> bool:
        return bool(os.environ.get("OPENAI_API_KEY"))

    def parse(self, system, user, schema, model=None):
        import openai

        if self._client is None:
            self._client = openai.OpenAI(max_retries=3, timeout=120)
        response = self._client.responses.parse(
            model=model or self.model, instructions=system, input=user, text_format=schema,
        )
        if response.output_parsed is None:
            raise LLMError("OpenAI returned no parsed output (possibly a refusal)")
        return response.output_parsed


class GeminiProvider(Provider):
    key, label, default_model, model_env = "gemini", "Gemini", "gemini-3.8-flash", "GEMINI_MODEL"
    _client = None

    def available(self) -> bool:
        return bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))

    def parse(self, system, user, schema, model=None):
        from google import genai
        from google.genai import types

        if self._client is None:
            # Reads GEMINI_API_KEY / GOOGLE_API_KEY; GEMINI_BASE_URL is optional (proxies, gateways).
            base_url = os.environ.get("GEMINI_BASE_URL")
            self._client = genai.Client(http_options=types.HttpOptions(base_url=base_url) if base_url else None)
        response = self._client.models.generate_content(
            model=model or self.model,
            contents=user,
            config=types.GenerateContentConfig(
                system_instruction=system, response_mime_type="application/json", response_schema=schema,
            ),
        )
        if isinstance(response.parsed, schema):
            return response.parsed
        if response.text:
            return _validate(schema, response.text, self.label)
        raise LLMError("Gemini returned no output (possibly blocked)")


class OpenAICompatProvider(Provider):
    """Any server speaking the OpenAI Chat Completions API: Ollama, LM Studio, OpenRouter, Groq, vLLM..."""

    def __init__(self, key: str, label: str, prefix: str, default_base_url: str = "", default_model: str = ""):
        self.key, self.label, self.prefix = key, label, prefix
        self.default_base_url, self.default_model = default_base_url, default_model
        self.model_env = f"{prefix}_MODEL"
        self._client = None

    @property
    def base_url(self) -> str:
        return os.environ.get(f"{self.prefix}_BASE_URL") or self.default_base_url

    def available(self) -> bool:
        return bool(self.base_url and self.model)

    def info(self) -> dict:
        label = os.environ.get(f"{self.prefix}_LABEL") or self.label
        return super().info() | {"label": label}

    def parse(self, system, user, schema, model=None):
        import openai

        if self._client is None:
            # Local servers accept any key; hosted ones (OpenRouter, Groq) need the real one.
            self._client = openai.OpenAI(
                base_url=self.base_url, api_key=os.environ.get(f"{self.prefix}_API_KEY") or "not-needed",
                max_retries=2, timeout=180,
            )
        model = model or self.model
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        try:
            completion = self._client.chat.completions.parse(model=model, messages=messages, response_format=schema)
            parsed = completion.choices[0].message.parsed
            if parsed is not None:
                return parsed
        except openai.BadRequestError:
            pass  # server doesn't support json_schema response_format; fall back to plain JSON mode
        messages[0]["content"] = (
            f"{system}\n\nRespond with ONLY a JSON object matching this schema:\n"
            f"{json.dumps(schema.model_json_schema())}"
        )
        completion = self._client.chat.completions.create(
            model=model, messages=messages, response_format={"type": "json_object"},
        )
        return _validate(schema, completion.choices[0].message.content or "", self.info()["label"])


def _validate(schema: type[T], text: str, label: str) -> T:
    try:
        return schema.model_validate_json(text)
    except ValidationError as e:
        raise LLMError(f"{label} returned JSON that doesn't match the schema: {e.errors()[0]['msg']}") from e


PROVIDERS: dict[str, Provider] = {
    p.key: p
    for p in [
        AnthropicProvider(),
        OpenAIProvider(),
        GeminiProvider(),
        OpenAICompatProvider("ollama", "Ollama (local)", "OLLAMA", "http://localhost:11434/v1"),
        OpenAICompatProvider("compat", "OpenAI-compatible", "OPENAI_COMPAT"),
    ]
}


def available() -> list[Provider]:
    return [p for p in PROVIDERS.values() if p.available()]


def default_provider() -> Provider | None:
    preferred = PROVIDERS.get(os.environ.get("LLM_PROVIDER", ""))
    if preferred and preferred.available():
        return preferred
    return next(iter(available()), None)


def get(key: str | None) -> Provider | None:
    """Resolve a requested provider key; None/"" means the default, "mock" means no LLM."""
    if key == "mock":
        return None
    if not key:
        return default_provider()
    p = PROVIDERS.get(key)
    if p is None:
        raise NotConfigured(f"Unknown provider '{key}'. Options: mock, {', '.join(PROVIDERS)}")
    if not p.available():
        raise NotConfigured(f"Provider '{key}' is not configured (missing key/model in .env).")
    return p
