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


class SmartFSRSAction:
    """A genuinely smart hand-coded scheduler. The 'fair' baseline a real
    tutoring app would actually ship — not a strawman like Random or FSRS.

    Heuristic: maintain a 'cognitive load' bound. If too many actively-learning
    words are below the recall threshold, the learner is overwhelmed — pure
    review until the active set stabilizes. Otherwise, introduce 1 new word
    per turn (like FSRS+New) but only when there's headroom.

    This captures what a sophisticated spaced-repetition system should do:
    don't dump new content on a learner who's still struggling with what
    they have."""

    name = "SmartFSRS"
    LOAD_THRESHOLD = 10   # max # of low-stability seen words before pausing intro
    DUE_R = 0.7           # words below this R are "due" for review

    def __call__(self, state, rng: np.random.Generator) -> int:
        # NOTE: state vector doesn't expose snapshot cleanly. SmartFSRS reads
        # the env via the standard API; we re-implement by inferring from
        # the structured directive translator's bucket logic.
        # For training-data collection, we use a slightly degraded heuristic
        # that only uses information available in the state vector.
        # Sample a directive: (1,0,4) most of the time, (0,0,5) when overload
        # is likely. Since we can't measure load from state vec alone,
        # use a simple stochastic mix that approximates the policy.
        if rng.random() < 0.7:
            return _action_index(1, 0, 4)
        return _action_index(0, 0, 5)


class SmartFSRSEnvAction:
    """Env-aware version of SmartFSRS — reads the simulator snapshot directly
    via bind_to_env. Used in evaluation, not in training-data collection."""

    name = "SmartFSRS"
    LOAD_THRESHOLD = 10
    DUE_R = 0.7

    def bind_to_env(self, env):
        def fn(_state, _rng):
            snap = env.sim.snapshot()
            active_due = sum(1 for w in snap
                             if w["seen"] and 0 < w["stability"] < 2.0
                             and w["retrievability"] < self.DUE_R)
            if active_due >= self.LOAD_THRESHOLD:
                # Overloaded — pure review, no new intro
                return _action_index(0, 0, 5)
            unseen = any(not w["seen"] for w in snap)
            if unseen:
                return _action_index(1, 0, 4)
            return _action_index(0, 0, 5)
        return fn


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
