"""Student simulator for Japanese vocabulary learning.

Models memory using an FSRS-style forgetting curve:
    R(t) = (1 + t / (9 * S))^(-1)
where t is elapsed time (in days) since last exposure and S is stability.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


def _load_vocab_from_json() -> list[tuple[str, str, str, str]] | None:
    """Load the generated vocab.json if present. Returns None if missing
    or malformed (in which case we fall back to the embedded 51-word default)."""
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "vocab.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            entries = json.load(f)
        out = []
        for e in entries:
            out.append((e["surface"], e["reading"], e["gloss"], e["jlpt"]))
        return out
    except Exception as exc:
        print(f"[simulator] vocab.json present but unreadable ({exc}); "
              f"falling back to embedded default")
        return None


# Embedded fallback (51 words) — used when vocab.json is absent.
# Run generate_vocab.py to produce the larger ~500-word list described in
# the proposal.
_EMBEDDED_VOCAB: list[tuple[str, str, str, str]] = [
    # N5 (easiest)
    ("水", "みず", "water", "N5"),
    ("火", "ひ", "fire", "N5"),
    ("人", "ひと", "person", "N5"),
    ("本", "ほん", "book", "N5"),
    ("車", "くるま", "car", "N5"),
    ("学校", "がっこう", "school", "N5"),
    ("先生", "せんせい", "teacher", "N5"),
    ("食べる", "たべる", "to eat", "N5"),
    ("飲む", "のむ", "to drink", "N5"),
    ("行く", "いく", "to go", "N5"),
    ("見る", "みる", "to see", "N5"),
    ("聞く", "きく", "to hear/ask", "N5"),
    ("大きい", "おおきい", "big", "N5"),
    ("小さい", "ちいさい", "small", "N5"),
    ("新しい", "あたらしい", "new", "N5"),
    ("古い", "ふるい", "old", "N5"),
    ("今日", "きょう", "today", "N5"),
    ("明日", "あした", "tomorrow", "N5"),
    # N4
    ("経済", "けいざい", "economy", "N4"),
    ("政治", "せいじ", "politics", "N4"),
    ("文化", "ぶんか", "culture", "N4"),
    ("社会", "しゃかい", "society", "N4"),
    ("世界", "せかい", "world", "N4"),
    ("自由", "じゆう", "freedom", "N4"),
    ("意見", "いけん", "opinion", "N4"),
    ("理由", "りゆう", "reason", "N4"),
    ("経験", "けいけん", "experience", "N4"),
    ("説明", "せつめい", "explanation", "N4"),
    ("約束", "やくそく", "promise", "N4"),
    ("興味", "きょうみ", "interest", "N4"),
    ("試験", "しけん", "exam", "N4"),
    ("研究", "けんきゅう", "research", "N4"),
    ("予定", "よてい", "schedule", "N4"),
    ("注意", "ちゅうい", "caution", "N4"),
    ("案内", "あんない", "guidance", "N4"),
    # N3 (hardest)
    ("環境", "かんきょう", "environment", "N3"),
    ("影響", "えいきょう", "influence", "N3"),
    ("意識", "いしき", "consciousness", "N3"),
    ("感謝", "かんしゃ", "gratitude", "N3"),
    ("検討", "けんとう", "examination", "N3"),
    ("貢献", "こうけん", "contribution", "N3"),
    ("傾向", "けいこう", "tendency", "N3"),
    ("詳細", "しょうさい", "details", "N3"),
    ("迅速", "じんそく", "promptness", "N3"),
    ("把握", "はあく", "grasp", "N3"),
    ("矛盾", "むじゅん", "contradiction", "N3"),
    ("曖昧", "あいまい", "ambiguous", "N3"),
    ("妥協", "だきょう", "compromise", "N3"),
    ("葛藤", "かっとう", "conflict", "N3"),
    ("懸念", "けねん", "concern", "N3"),
    ("緻密", "ちみつ", "elaborate", "N3"),
]


# Public symbol used elsewhere. Prefer the JSON file when available.
VOCAB: list[tuple[str, str, str, str]] = _load_vocab_from_json() or _EMBEDDED_VOCAB


LEVEL_DIFFICULTY = {"N5": 2.5, "N4": 5.0, "N3": 7.5}


@dataclass
class WordState:
    word_id: int
    surface: str
    reading: str
    gloss: str
    jlpt: str
    difficulty: float                  # 0-10
    forgetting_constant: float = 9.0   # FSRS curve constant
    stability: float = 0.0             # 0 = unseen
    last_seen_time: float = 0.0        # in "days"
    times_seen: int = 0

    def retrievability(self, now: float) -> float:
        """FSRS-style recall probability at time `now` (days)."""
        if self.stability <= 0.0:
            return 0.0
        t = max(0.0, now - self.last_seen_time)
        return (1.0 + t / (self.forgetting_constant * self.stability)) ** -1.0


@dataclass
class StepResult:
    word_id: int
    recalled: bool
    pre_retrievability: float
    post_stability: float


@dataclass
class StudentSimulator:
    """Simulates a learner studying vocabulary across turns and sessions."""

    seed: int = 0
    initial_stability: float = 0.5      # stability granted on first SUCCESSFUL exposure
    success_growth_base: float = 1.8    # multiplier on success (modulated by difficulty)
    failure_reset: float = 0.3          # stability after a failed recall (on a seen word)
    forgetting_constant: float = 9.0    # FSRS forgetting-curve constant
    difficulty_scale: float = 1.0       # multiplier on the JLPT-level difficulty baselines
    first_exposure_failure_prob: float = 0.2
        # Probability the learner "doesn't catch" a word on first exposure.
        # Real learners sometimes miss words on introduction; without this the
        # simulator is unrealistically optimistic about acquisition rates.
        # On failure: stability is set to a low value (initial_stability/4) so
        # the word is still tracked as seen but barely retained.
    rng: np.random.Generator = field(init=False)
    words: list[WordState] = field(init=False)
    now: float = 0.0  # current sim time in days

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.words = []
        for i, (surface, reading, gloss, jlpt) in enumerate(VOCAB):
            # Per-word difficulty: level baseline (scaled) + small noise, clipped to [0, 10].
            base = LEVEL_DIFFICULTY[jlpt] * self.difficulty_scale
            noise = self.rng.normal(0.0, 0.6)
            diff = float(np.clip(base + noise, 0.0, 10.0))
            self.words.append(
                WordState(
                    word_id=i,
                    surface=surface,
                    reading=reading,
                    gloss=gloss,
                    jlpt=jlpt,
                    difficulty=diff,
                    forgetting_constant=self.forgetting_constant,
                )
            )

    # ---- introspection helpers used by policies ---------------------------

    def num_words(self) -> int:
        return len(self.words)

    def snapshot(self) -> list[dict]:
        """Return a lightweight view of state for policies (read-only)."""
        return [
            {
                "word_id": w.word_id,
                "jlpt": w.jlpt,
                "difficulty": w.difficulty,
                "stability": w.stability,
                "times_seen": w.times_seen,
                "retrievability": w.retrievability(self.now),
                "seen": w.stability > 0.0,
            }
            for w in self.words
        ]

    # ---- core dynamics ----------------------------------------------------

    def step(self, directive: Sequence[int]) -> list[StepResult]:
        """Expose the learner to a set of word IDs (one turn). Returns per-word results."""
        results: list[StepResult] = []
        for wid in directive:
            w = self.words[wid]
            r_pre = w.retrievability(self.now)

            if w.stability <= 0.0:
                # First exposure: probabilistic encoding. With probability
                # first_exposure_failure_prob the learner "misses" it
                # (stability set to a quarter of initial). Otherwise success.
                if self.rng.random() < self.first_exposure_failure_prob:
                    w.stability = self.initial_stability / 4.0
                    recalled = False
                else:
                    w.stability = self.initial_stability
                    recalled = True   # successful introduction
                r_pre = 0.0
            else:
                recalled = bool(self.rng.random() < r_pre)
                if recalled:
                    # Easier words grow faster on success; harder words grow slowly.
                    growth = self.success_growth_base * (1.0 - 0.05 * w.difficulty)
                    growth = max(1.1, growth)
                    w.stability = w.stability * growth
                else:
                    w.stability = self.failure_reset

            w.last_seen_time = self.now
            w.times_seen += 1
            results.append(
                StepResult(
                    word_id=wid,
                    recalled=recalled,
                    pre_retrievability=r_pre,
                    post_stability=w.stability,
                )
            )

        # Advance time slightly within a session (each turn ~ a couple minutes).
        self.now += 2.0 / (60.0 * 24.0)  # ~2 minutes in days
        return results

    def advance_time(self, days: float) -> None:
        """Skip forward in time (e.g., between sessions)."""
        self.now += days

    # ---- aggregate metrics ------------------------------------------------

    def metrics(self) -> dict:
        seen = [w for w in self.words if w.stability > 0.0]
        retr = [w.retrievability(self.now) for w in seen]
        acquired = sum(1 for r in retr if r > 0.9)
        avg_r = float(np.mean(retr)) if retr else 0.0
        return {
            "acquired": acquired,
            "avg_retrievability": avg_r,
            "n_seen": len(seen),
        }
