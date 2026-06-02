"""Baseline policies expressed over the VocabEnv discrete action space.

Each policy maps the env state (and an RNG) to an action index in
[0, NUM_ACTIONS). These are used to collect the offline dataset that IQL/CQL
will train on, so the action distribution must cover a useful range.
"""

from __future__ import annotations

import numpy as np

from env import ACTION_TABLE, NUM_ACTIONS, WORDS_PER_TURN


def _action_index(n_new: int, n_active: int, n_review: int) -> int:
    return ACTION_TABLE.index((n_new, n_active, n_review))


# Precompute frequently-used indices.
A_ALL_REVIEW = _action_index(0, 0, 5)        # FSRS
A_ONE_NEW = _action_index(1, 0, 4)            # FSRS+New, RuleBased's "introduce" turn


class RandomAction:
    name = "RandomAction"

    def __call__(self, state, rng: np.random.Generator) -> int:
        return int(rng.integers(NUM_ACTIONS))


class FSRSAction:
    """Always 'review only' — the env's directive translator handles fallback
    to new words when there aren't 5 seen words yet."""
    name = "FSRSAction"

    def __call__(self, state, rng: np.random.Generator) -> int:
        return A_ALL_REVIEW


class FSRSPlusNewAction:
    name = "FSRSPlusNewAction"

    def __call__(self, state, rng: np.random.Generator) -> int:
        return A_ONE_NEW


class RuleBasedAction:
    """95% review-only, 5% introduce one new word."""
    name = "RuleBasedAction"
    new_word_prob = 0.05

    def __call__(self, state, rng: np.random.Generator) -> int:
        if rng.random() < self.new_word_prob:
            return A_ONE_NEW
        return A_ALL_REVIEW


class MixedExploreAction:
    """Stochastic mixture over a small set of 'reasonable' directives, to
    inject diversity into the offline buffer. Pure FSRS and FSRS+New on their
    own collapse to one action; IQL benefits from broader coverage."""
    name = "MixedExploreAction"
    # candidate (n_new, n_active, n_review) tuples we want to see frequently
    CANDIDATES = [(0, 0, 5), (1, 0, 4), (1, 1, 3), (2, 0, 3), (0, 1, 4),
                  (2, 1, 2), (3, 0, 2)]

    def __init__(self) -> None:
        self._indices = [_action_index(*c) for c in self.CANDIDATES]

    def __call__(self, state, rng: np.random.Generator) -> int:
        return int(rng.choice(self._indices))


class AggressiveIntroAction:
    """Heavy on new-word introductions. Samples uniformly from directives
    that have n_new >= 2. Designed to compensate for the existing baselines
    being review-heavy (FSRS, RuleBased, FSRS+New have n_new <= 1)."""
    name = "AggressiveIntro"

    def __init__(self) -> None:
        self._indices = [_action_index(a, b, WORDS_PER_TURN - a - b)
                         for a in range(2, WORDS_PER_TURN + 1)
                         for b in range(0, WORDS_PER_TURN - a + 1)]

    def __call__(self, state, rng: np.random.Generator) -> int:
        return int(rng.choice(self._indices))


ALL_BEHAVIOR_POLICIES = [
    RandomAction, FSRSAction, FSRSPlusNewAction, RuleBasedAction,
    MixedExploreAction,
]

# "Rebalanced" mix — drops FSRS (always 0,0,5 — clearly suboptimal at scale),
# replaces RuleBased (95% review-only) with MixedExplore + AggressiveIntro
# for more breadth. Yields a buffer biased toward higher-quality scheduling.
REBALANCED_BEHAVIOR_POLICIES = [
    RandomAction, FSRSPlusNewAction, MixedExploreAction,
    AggressiveIntroAction, AggressiveIntroAction,  # double weight on aggressive intro
]
