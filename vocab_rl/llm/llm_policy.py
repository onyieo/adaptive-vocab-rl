"""LLM-prompted policy: baseline #4 from the proposal.

Prompts an LLM with the current learner state at each turn and asks it to
output a structured directive (counts of new/active/review words). The
directive maps to the same VocabEnv action index used by every other policy.

Defaults to OpenAI gpt-4o-mini (cheap, lots of calls). System prompt is
cached on the provider side.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from env import ACTION_TABLE, WORDS_PER_TURN                              # noqa: E402
from simulator import VOCAB                                               # noqa: E402
from llm._llm_client import LLMClient, DEFAULT_POLICY_MODEL, DEFAULT_PROVIDER  # noqa: E402


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
    """Stateless per turn — each call is independent based on current state."""

    name = "LLMPrompted"
    _FALLBACK_ACTION = ACTION_TABLE.index((1, 0, 4))

    def __init__(self, provider: str | None = None,
                 model: str | None = None,
                 turns_per_session: int = 20) -> None:
        provider = provider or DEFAULT_PROVIDER
        model = model or DEFAULT_POLICY_MODEL[provider]
        self.client = LLMClient(provider=provider, model=model)
        self.turns_per_session = turns_per_session

    def bind_to_env(self, env):
        def fn(_state, _rng):
            return self._act(env)
        return fn

    def _act(self, env) -> int:
        snap = env.sim.snapshot()
        sessions_done = env.turn // self.turns_per_session
        summary = _state_summary(snap, env.turn, self.turns_per_session,
                                 sessions_done)
        try:
            out = self.client.chat(
                system=SYSTEM_PROMPT,
                user=f"Current learner state:\n{summary}\n\nChoose your action.",
                max_tokens=200,
                json_schema=SCHEMA,
            )
            parsed = json.loads(out["text"])
            a = int(parsed["action_index"])
            if 0 <= a < len(ACTION_TABLE):
                return a
            print(f"[LLMPolicy] out-of-range action {a}; falling back")
            return self._FALLBACK_ACTION
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"[LLMPolicy] JSON parse failed ({e}); falling back. "
                  f"Raw text: {out.get('text', '')[:200]!r}")
            return self._FALLBACK_ACTION
        except Exception as e:
            print(f"[LLMPolicy] API call failed ({type(e).__name__}: {e}); falling back.")
            return self._FALLBACK_ACTION
