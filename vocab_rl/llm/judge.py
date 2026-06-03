"""LLM-as-judge: scores naturalness of a generated tutoring turn on a 1-5 scale.

Supports either Claude (Anthropic) or OpenAI as backend. Cross-provider
judging (e.g., Claude tutor + OpenAI judge) eliminates same-family
self-preference bias, which we use in the final eval.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from llm._llm_client import LLMClient, DEFAULT_JUDGE_MODEL, DEFAULT_PROVIDER  # noqa: E402


SYSTEM_PROMPT = """You evaluate the naturalness of a short Japanese tutoring conversation turn.

Score on a 1-5 scale:
  5 — Native-sounding. Target vocabulary woven into the conversation organically. Pedagogical glosses (readings, English brackets) feel natural and unobtrusive.
  4 — Mostly natural with minor awkwardness, or slightly forced inclusion of one target word.
  3 — Comprehensible but stilted. Vocabulary feels listed or shoehorned rather than woven in.
  2 — Significantly forced. Reads more like a vocabulary drill than a conversation.
  1 — Unnatural, ungrammatical, or word-salad.

You are scoring CONVERSATIONAL NATURALNESS, not coverage. Do NOT penalize the tutor for which words appear or how many — the word selection is decided by an external scheduler, not by the tutor. Score the dialogue quality given the words it received.

Respond as a JSON object with keys "score" (integer 1-5) and "reason" (one short sentence)."""


SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
        "reason": {"type": "string"},
    },
    "required": ["score", "reason"],
    "additionalProperties": False,
}


class Judge:
    # Judge defaults to OpenAI (cross-family with Claude tutor — eliminates
    # same-family self-preference bias for the headline naturalness numbers).
    DEFAULT_PROVIDER = "openai"

    def __init__(self, provider: str | None = None,
                 model: str | None = None) -> None:
        provider = provider or self.DEFAULT_PROVIDER
        model = model or DEFAULT_JUDGE_MODEL[provider]
        self.client = LLMClient(provider=provider, model=model)
        self.provider = provider
        self.model = model

    def score(self, turn_text: str,
              prior_turn_text: str | None = None) -> tuple[int, str]:
        ctx = (f"Prior turn (for context only — do not score):\n{prior_turn_text}\n\n"
               if prior_turn_text else "")
        user_message = (f"{ctx}Turn to evaluate:\n{turn_text}\n\nScore this turn.")

        try:
            out = self.client.chat(
                system=SYSTEM_PROMPT,
                user=user_message,
                max_tokens=200,
                json_schema=SCHEMA,
            )
            parsed = json.loads(out["text"])
            return int(parsed["score"]), str(parsed["reason"])
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            # Degrade gracefully; eval loop logs and treats as missing.
            print(f"[Judge] parse failed ({e}); returning 3 as neutral. "
                  f"Raw: {out.get('text', '')[:120]!r}")
            return 3, "PARSE_ERROR"
