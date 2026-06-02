"""LLM-prompted policy: baseline #4 from the proposal.

Prompts a Claude model with the current learner state at each turn and asks it
to output a structured directive (counts of new/active/review words to target).
The directive is then mapped to the same VocabEnv action index used by every
other policy, so comparisons are apples-to-apples.

Uses Haiku for cost — this policy runs ~thousands of times in a full eval
sweep. System prompt (instructions + action-space description) is cached.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import anthropic

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from env import ACTION_TABLE, WORDS_PER_TURN  # noqa: E402
from simulator import VOCAB                   # noqa: E402
from llm._env import require_api_key          # noqa: E402


LLM_POLICY_MODEL = "claude-haiku-4-5"


def _action_table_text() -> str:
    lines = ["Available actions (action_index → (n_new, n_active, n_review)):"]
    for i, (a, b, c) in enumerate(ACTION_TABLE):
        lines.append(f"  {i:2d} → ({a}, {b}, {c})")
    return "\n".join(lines)


SYSTEM_PROMPT = f"""You are an expert spaced-repetition scheduler. Each turn you choose ONE directive specifying how many new, actively-learning, and review words to target. The total must equal {WORDS_PER_TURN}.

Definitions:
- "new"     : word not yet introduced (stability == 0)
- "active"  : word seen but still being learned (stability in (0, 2.0))
- "review"  : word seen and well-learned (stability >= 2.0)

Your goal is to maximize the learner's total retained vocabulary across many sessions. Balance introducing new words (so the inventory grows) against reviewing old ones (so they don't decay below recall threshold). Pace introductions so the learner isn't overwhelmed by too many low-stability words at once.

{_action_table_text()}

Respond with a single JSON object: {{"action_index": <0-20>, "reason": "<one short sentence>"}}."""


SCHEMA = {
    "type": "object",
    "properties": {
        "action_index": {"type": "integer", "enum": list(range(len(ACTION_TABLE)))},
        "reason": {"type": "string"},
    },
    "required": ["action_index", "reason"],
    "additionalProperties": False,
}


def _state_summary(snap: list[dict], turn: int, turns_per_session: int,
                   sessions_done: int) -> str:
    """Condense the full state into a prompt-sized summary."""
    unseen = [w for w in snap if not w["seen"]]
    active = [w for w in snap if w["seen"] and w["stability"] < 2.0]
    review = [w for w in snap if w["seen"] and w["stability"] >= 2.0]

    def fmt(words, key, n=8):
        words_sorted = sorted(words, key=key)[:n]
        return ", ".join(
            f"[{w['word_id']}]{VOCAB[w['word_id']][0]}(R={w['retrievability']:.2f},"
            f"S={w['stability']:.1f})"
            for w in words_sorted
        ) or "(none)"

    return (
        f"Turn {turn % turns_per_session + 1}/{turns_per_session} of session "
        f"{sessions_done + 1}. Counts: unseen={len(unseen)}, "
        f"active={len(active)}, review={len(review)}.\n"
        f"Most due active (lowest R): {fmt(active, lambda w: w['retrievability'])}\n"
        f"Most due review (lowest R): {fmt(review, lambda w: w['retrievability'])}\n"
        f"Easiest unseen (lowest difficulty): "
        f"{fmt(unseen, lambda w: VOCAB[w['word_id']][3])}"
    )


class LLMPolicy:
    """Stateless across turns — each call is an independent decision based on
    the current state summary."""

    name = "LLMPrompted"

    def __init__(self, model: str = LLM_POLICY_MODEL,
                 turns_per_session: int = 20) -> None:
        require_api_key()
        self.client = anthropic.Anthropic()
        self.model = model
        self.turns_per_session = turns_per_session

    def bind_to_env(self, env):
        """Return a (state, rng) -> action closure that reads the live env
        snapshot. Use this so the policy fits the standard eval signature."""
        def fn(_state, rng):
            return self._act(env)
        return fn

    # Fallback action used when JSON parsing fails — matches FSRS+New
    # heuristic so the baseline degrades gracefully rather than crashing.
    _FALLBACK_ACTION = ACTION_TABLE.index((1, 0, 4))

    def _act(self, env) -> int:
        snap = env.sim.snapshot()
        sessions_done = env.turn // self.turns_per_session
        summary = _state_summary(snap, env.turn, self.turns_per_session,
                                 sessions_done)

        response = self.client.messages.create(
            model=self.model,
            max_tokens=200,
            system=[{
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }],
            messages=[{"role": "user",
                       "content": f"Current learner state:\n{summary}\n\nChoose your action."}],
            output_config={"format": {"type": "json_schema", "schema": SCHEMA}},
        )

        text = "".join(b.text for b in response.content if b.type == "text")
        try:
            parsed = json.loads(text)
            a = int(parsed["action_index"])
            if 0 <= a < len(ACTION_TABLE):
                return a
            print(f"[LLMPolicy] out-of-range action {a}; falling back")
            return self._FALLBACK_ACTION
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            # Structured outputs occasionally returns malformed JSON despite
            # the schema. Fall back rather than crash the eval.
            print(f"[LLMPolicy] JSON parse failed ({e}); falling back. "
                  f"Raw text (first 200 chars): {text[:200]!r}")
            return self._FALLBACK_ACTION


