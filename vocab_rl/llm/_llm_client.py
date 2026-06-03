"""Unified LLM client wrapper supporting both Anthropic and OpenAI.

Lets us swap providers in one place. Both providers' chat APIs are very
similar; this module hides the differences (system-prompt format, parameter
names, structured-output syntax).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from llm._env import load_env  # noqa: E402


load_env()


def _has_openai_key() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY"))


def _has_anthropic_key() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


# Provider/model defaults — prefer OpenAI now that Anthropic credits ran out.
DEFAULT_PROVIDER = "openai" if _has_openai_key() else "anthropic"

DEFAULT_TUTOR_MODEL = {
    "openai":    "gpt-4o",        # quality matters here
    "anthropic": "claude-sonnet-4-6",
}
DEFAULT_JUDGE_MODEL = {
    "openai":    "gpt-4o-mini",   # cheap, lots of calls
    "anthropic": "claude-haiku-4-5",
}
DEFAULT_POLICY_MODEL = {
    "openai":    "gpt-4o-mini",   # cheap
    "anthropic": "claude-haiku-4-5",
}


class LLMClient:
    """Thin wrapper over Anthropic OR OpenAI chat completions, exposing one
    shape: chat(system, user, max_tokens, json_schema=None) -> {"text", "usage"}."""

    def __init__(self, provider: str | None = None,
                 model: str | None = None) -> None:
        self.provider = provider or DEFAULT_PROVIDER
        if self.provider == "openai":
            from openai import OpenAI
            self.client = OpenAI()
            self.model = model
        elif self.provider == "anthropic":
            import anthropic
            self.client = anthropic.Anthropic()
            self.model = model
        else:
            raise ValueError(f"unknown provider {self.provider!r}")

    # ---- single-shot chat with optional system prompt and JSON schema ----

    def chat(self, system: str, user: str, *,
             max_tokens: int = 400,
             json_schema: dict | None = None,
             cache_system: bool = True) -> dict:
        if self.provider == "openai":
            return self._openai_chat(system, user, max_tokens, json_schema)
        return self._anthropic_chat(system, user, max_tokens, json_schema, cache_system)

    # ---- multi-turn chat (history) — used by Tutor for stateful sessions -

    def multiturn_chat(self, system: str, messages: list[dict], *,
                       max_tokens: int = 400,
                       cache_system: bool = True) -> dict:
        if self.provider == "openai":
            return self._openai_multiturn(system, messages, max_tokens)
        return self._anthropic_multiturn(system, messages, max_tokens, cache_system)

    # ---- backend-specific implementations -------------------------------

    def _openai_chat(self, system, user, max_tokens, json_schema) -> dict:
        kwargs = self._openai_kwargs(max_tokens)
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": user}
        ]
        if json_schema is not None:
            r = self.client.chat.completions.create(
                model=self.model, messages=messages,
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": "out", "schema": json_schema, "strict": True},
                },
                **kwargs,
            )
        else:
            r = self.client.chat.completions.create(
                model=self.model, messages=messages, **kwargs)
        text = r.choices[0].message.content or ""
        return {
            "text": text,
            "usage": {
                "input_tokens": r.usage.prompt_tokens,
                "output_tokens": r.usage.completion_tokens,
                "cache_read_input_tokens": getattr(r.usage, "prompt_tokens_details", None).cached_tokens
                    if getattr(r.usage, "prompt_tokens_details", None) else 0,
                "cache_creation_input_tokens": 0,
            },
        }

    def _openai_multiturn(self, system, messages, max_tokens) -> dict:
        kwargs = self._openai_kwargs(max_tokens)
        full = ([{"role": "system", "content": system}] if system else []) + messages
        r = self.client.chat.completions.create(
            model=self.model, messages=full, **kwargs)
        text = r.choices[0].message.content or ""
        return {
            "text": text,
            "usage": {
                "input_tokens": r.usage.prompt_tokens,
                "output_tokens": r.usage.completion_tokens,
                "cache_read_input_tokens": getattr(r.usage, "prompt_tokens_details", None).cached_tokens
                    if getattr(r.usage, "prompt_tokens_details", None) else 0,
                "cache_creation_input_tokens": 0,
            },
        }

    def _openai_kwargs(self, max_tokens: int) -> dict:
        # gpt-5 family uses max_completion_tokens; 4-series uses max_tokens.
        if self.model and self.model.startswith("gpt-5"):
            return {"max_completion_tokens": max_tokens}
        return {"max_tokens": max_tokens}

    def _anthropic_chat(self, system, user, max_tokens, json_schema, cache_system) -> dict:
        sys_param = self._anthropic_system(system, cache_system)
        kwargs = {"model": self.model, "max_tokens": max_tokens,
                  "system": sys_param,
                  "messages": [{"role": "user", "content": user}]}
        if json_schema is not None:
            kwargs["output_config"] = {"format": {"type": "json_schema", "schema": json_schema}}
        r = self.client.messages.create(**kwargs)
        text = "".join(b.text for b in r.content if b.type == "text")
        return {
            "text": text,
            "usage": {
                "input_tokens": r.usage.input_tokens,
                "output_tokens": r.usage.output_tokens,
                "cache_read_input_tokens": r.usage.cache_read_input_tokens or 0,
                "cache_creation_input_tokens": r.usage.cache_creation_input_tokens or 0,
            },
        }

    def _anthropic_multiturn(self, system, messages, max_tokens, cache_system) -> dict:
        r = self.client.messages.create(
            model=self.model, max_tokens=max_tokens,
            system=self._anthropic_system(system, cache_system),
            messages=messages,
        )
        text = "".join(b.text for b in r.content if b.type == "text")
        return {
            "text": text,
            "usage": {
                "input_tokens": r.usage.input_tokens,
                "output_tokens": r.usage.output_tokens,
                "cache_read_input_tokens": r.usage.cache_read_input_tokens or 0,
                "cache_creation_input_tokens": r.usage.cache_creation_input_tokens or 0,
            },
        }

    def _anthropic_system(self, system: str, cache: bool):
        if not system:
            return ""
        if cache:
            return [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
        return system
