"""LLM-as-judge: scores naturalness of a generated tutoring turn on a 1-5 scale.

Uses claude-haiku-4-5 (cheap, lots of calls). Rubric is in the system prompt
and cached. Output is constrained to a small JSON schema so parsing is reliable.
"""

from __future__ import annotations

import json

import anthropic


JUDGE_MODEL = "claude-haiku-4-5"


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
    def __init__(self, model: str = JUDGE_MODEL) -> None:
        self.client = anthropic.Anthropic()
        self.model = model

    def score(self, turn_text: str,
              prior_turn_text: str | None = None) -> tuple[int, str]:
        ctx = (f"Prior turn (for context only — do not score):\n{prior_turn_text}\n\n"
               if prior_turn_text else "")
        user_message = (f"{ctx}Turn to evaluate:\n{turn_text}\n\n"
                        "Score this turn.")

        response = self.client.messages.create(
            model=self.model,
            max_tokens=200,
            system=[{
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }],
            messages=[{"role": "user", "content": user_message}],
            output_config={"format": {"type": "json_schema", "schema": SCHEMA}},
        )

        text = "".join(b.text for b in response.content if b.type == "text")
        parsed = json.loads(text)
        return int(parsed["score"]), str(parsed["reason"])
