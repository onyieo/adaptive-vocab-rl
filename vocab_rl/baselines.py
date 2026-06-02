"""Baseline vocabulary-selection policies.

Each policy implements `select(snapshot, rng) -> list[int]` returning up to
WORDS_PER_TURN word IDs to expose this turn.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


WORDS_PER_TURN = 5


def _seen(snapshot: Sequence[dict]) -> list[dict]:
    return [w for w in snapshot if w["seen"]]


def _unseen(snapshot: Sequence[dict]) -> list[dict]:
    return [w for w in snapshot if not w["seen"]]


class RandomPolicy:
    name = "Random"

    def select(self, snapshot, rng: np.random.Generator) -> list[int]:
        n = len(snapshot)
        k = min(WORDS_PER_TURN, n)
        return list(rng.choice(n, size=k, replace=False))


class FSRSPolicy:
    """Pick the WORDS_PER_TURN seen words with lowest retrievability.

    If fewer than WORDS_PER_TURN are due, introduce one new word to fill in.
    """

    name = "FSRS"

    def select(self, snapshot, rng: np.random.Generator) -> list[int]:
        seen = _seen(snapshot)
        seen_sorted = sorted(seen, key=lambda w: w["retrievability"])
        picks = [w["word_id"] for w in seen_sorted[:WORDS_PER_TURN]]
        if len(picks) < WORDS_PER_TURN:
            unseen = _unseen(snapshot)
            if unseen:
                # Introduce one new word (easiest first).
                unseen_sorted = sorted(unseen, key=lambda w: w["difficulty"])
                picks.append(unseen_sorted[0]["word_id"])
        return picks


class FSRSPlusNewPolicy:
    """Always introduce 1 new word + the 4 most-due review words."""

    name = "FSRS+New"

    def select(self, snapshot, rng: np.random.Generator) -> list[int]:
        picks: list[int] = []
        unseen = sorted(_unseen(snapshot), key=lambda w: w["difficulty"])
        if unseen:
            picks.append(unseen[0]["word_id"])
        seen_sorted = sorted(_seen(snapshot), key=lambda w: w["retrievability"])
        for w in seen_sorted:
            if len(picks) >= WORDS_PER_TURN:
                break
            picks.append(w["word_id"])
        # If nothing is seen yet, fill remainder with more new words.
        i = 1
        while len(picks) < WORDS_PER_TURN and i < len(unseen):
            picks.append(unseen[i]["word_id"])
            i += 1
        return picks


class RuleBasedPolicy:
    """95% of turns: review most-due words. 5% of turns: introduce new word(s)."""

    name = "RuleBased"
    new_word_prob = 0.05

    def select(self, snapshot, rng: np.random.Generator) -> list[int]:
        introduce_new = rng.random() < self.new_word_prob
        seen_sorted = sorted(_seen(snapshot), key=lambda w: w["retrievability"])
        unseen_sorted = sorted(_unseen(snapshot), key=lambda w: w["difficulty"])

        picks: list[int] = []
        if introduce_new and unseen_sorted:
            picks.append(unseen_sorted[0]["word_id"])

        for w in seen_sorted:
            if len(picks) >= WORDS_PER_TURN:
                break
            picks.append(w["word_id"])

        # Cold start: no seen words yet -> introduce new ones.
        i = 0
        while len(picks) < WORDS_PER_TURN and i < len(unseen_sorted):
            wid = unseen_sorted[i]["word_id"]
            if wid not in picks:
                picks.append(wid)
            i += 1
        return picks


ALL_POLICIES = [RandomPolicy, FSRSPolicy, FSRSPlusNewPolicy, RuleBasedPolicy]
