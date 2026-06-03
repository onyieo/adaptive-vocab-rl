"""LLM tutor: turns a vocabulary directive into a Japanese tutoring conversation turn.

Supports either Claude (Anthropic) or OpenAI as backend via the LLMClient
wrapper. System prompt is held stable across turns within a session so
provider-level prompt caching kicks in.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simulator import VOCAB                                          # noqa: E402
from llm._llm_client import LLMClient, DEFAULT_TUTOR_MODEL, DEFAULT_PROVIDER  # noqa: E402


MAX_TUTOR_TOKENS = 500


def _build_vocab_block() -> str:
    lines = ["Full vocabulary inventory (the learner may have seen any subset):"]
    for i, (surface, reading, gloss, jlpt) in enumerate(VOCAB):
        lines.append(f"  [{i:2d}] {surface}  ({reading})  — {gloss}  [{jlpt}]")
    return "\n".join(lines)


SYSTEM_PROMPT = f"""You are Hana, a warm and patient Japanese language tutor. You are having a casual everyday conversation with a learner who is studying JLPT-level vocabulary (N5 easiest through N3 hardest).

Each turn, an offline scheduler will tell you which target words to incorporate. Your job is to weave those words into ONE short, natural-sounding conversational turn — not a vocabulary drill.

Style rules:
- 1-3 short Japanese sentences per turn.
- For words new to this session, include the reading in parentheses on first use, plus a brief English gloss in square brackets. Example: 経済 (けいざい) [economy].
- For words already established this session, just use them naturally, no re-gloss.
- Always emphasize the EMPHASIZED word naturally (make it the topical focus or the answer to a question), but do not announce that you are emphasizing it.
- End with a short question or conversational hook so the learner can respond.
- Stay in the everyday-conversation register. Avoid textbook-style explanations.

Output ONLY Hana's spoken turn. No labels, no preface, no meta-commentary, no translations of the whole turn.

{_build_vocab_block()}"""


class Tutor:
    """Stateful tutor — keeps the conversation history within a session."""

    # Tutor defaults to Claude Sonnet — best Japanese quality available.
    DEFAULT_PROVIDER = "anthropic"

    def __init__(self, provider: str | None = None,
                 model: str | None = None) -> None:
        provider = provider or self.DEFAULT_PROVIDER
        model = model or DEFAULT_TUTOR_MODEL[provider]
        self.client = LLMClient(provider=provider, model=model)
        self.history: list[dict] = []

    def reset(self) -> None:
        self.history = []

    def generate_turn(self, word_ids: list[int],
                      emphasized_id: int | None = None) -> dict:
        directive_lines = ["Target words for this turn:"]
        for wid in word_ids:
            surface, reading, gloss, jlpt = VOCAB[wid]
            mark = "  <-- EMPHASIZE" if wid == emphasized_id else ""
            directive_lines.append(
                f"  [{wid}] {surface} ({reading}) — {gloss} [{jlpt}]{mark}"
            )
        directive_lines.append("\nProduce Hana's next turn.")
        user_message = "\n".join(directive_lines)

        self.history.append({"role": "user", "content": user_message})
        out = self.client.multiturn_chat(
            system=SYSTEM_PROMPT,
            messages=self.history,
            max_tokens=MAX_TUTOR_TOKENS,
        )
        text = out["text"].strip()
        self.history.append({"role": "assistant", "content": text})
        return {"text": text, "usage": out["usage"]}
