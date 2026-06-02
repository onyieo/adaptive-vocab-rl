"""Gym-style MDP wrapper around StudentSimulator.

Action space: structured vocabulary directive (n_new, n_active, n_review)
summing to WORDS_PER_TURN. Enumerated as a flat discrete index of size 21.

State: per-word [stability_norm, retrievability, difficulty_norm, log_times_seen_norm]
       plus globals [turn_in_session, sessions_done_frac]. Total dim = 4*N + 2.

Reward: small dense shaping on delta-acquired per turn, plus a terminal
        average-retrievability bonus at episode end.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from simulator import StudentSimulator, VOCAB


WORDS_PER_TURN = 5
# words with stability >= this are considered "review" (well-learned);
# 0 < stability < this are "active" (still being learned).
ACTIVE_STABILITY_THRESHOLD = 2.0
SHAPING_COEF = 0.01      # weight on per-step delta-acquired
TERMINAL_COEF = 1.0      # weight on terminal mean retrievability


def _build_action_table() -> list[tuple[int, int, int]]:
    """All (n_new, n_active, n_review) tuples with n_new+n_active+n_review = WORDS_PER_TURN."""
    table = []
    for n_new in range(WORDS_PER_TURN + 1):
        for n_active in range(WORDS_PER_TURN + 1 - n_new):
            n_review = WORDS_PER_TURN - n_new - n_active
            table.append((n_new, n_active, n_review))
    return table


ACTION_TABLE: list[tuple[int, int, int]] = _build_action_table()
NUM_ACTIONS = len(ACTION_TABLE)  # 21


def _state_dim(n_words: int) -> int:
    return 4 * n_words + 2


def _featurize(sim: StudentSimulator, turn: int, n_sessions: int,
               turns_per_session: int) -> np.ndarray:
    """Build the fixed-dim state vector from the current simulator snapshot."""
    snap = sim.snapshot()
    feats = np.zeros(_state_dim(len(snap)), dtype=np.float32)
    for i, w in enumerate(snap):
        base = 4 * i
        feats[base + 0] = min(w["stability"] / 5.0, 1.0)
        feats[base + 1] = w["retrievability"]
        feats[base + 2] = w["difficulty"] / 10.0
        feats[base + 3] = np.log1p(w["times_seen"]) / np.log(21.0)
    feats[-2] = (turn % turns_per_session) / turns_per_session
    feats[-1] = (turn // turns_per_session) / n_sessions
    return feats


@dataclass
class StepInfo:
    acquired: int
    avg_retrievability: float


class VocabEnv:
    """Gym-style environment over the student simulator."""

    def __init__(self, seed: int = 0, n_sessions: int = 10,
                 turns_per_session: int = 20, days_between: float = 1.0,
                 sim_kwargs: dict | None = None) -> None:
        self.seed = seed
        self.n_sessions = n_sessions
        self.turns_per_session = turns_per_session
        self.days_between = days_between
        self.total_turns = n_sessions * turns_per_session
        self.sim_kwargs = sim_kwargs or {}

        self.sim = StudentSimulator(seed=seed, **self.sim_kwargs)
        self.n_words = self.sim.num_words()
        self.state_dim = _state_dim(self.n_words)
        self.action_dim = NUM_ACTIONS

        self.turn = 0
        self._prev_acquired = 0

    # ---- core API ---------------------------------------------------------

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.seed = seed
        self.sim = StudentSimulator(seed=self.seed, **self.sim_kwargs)
        self.turn = 0
        self._prev_acquired = 0
        return self._obs()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, StepInfo]:
        n_new, n_active, n_review = ACTION_TABLE[action]
        directive = self._directive_to_words(n_new, n_active, n_review)
        self.sim.step(directive)
        self.turn += 1

        m = self.sim.metrics()
        delta = m["acquired"] - self._prev_acquired
        self._prev_acquired = m["acquired"]

        done = self.turn >= self.total_turns

        # advance time across session boundary (after the last turn of a session,
        # before the first turn of the next).
        if (self.turn % self.turns_per_session == 0) and not done:
            self.sim.advance_time(self.days_between)

        reward = SHAPING_COEF * float(delta)
        if done:
            reward += TERMINAL_COEF * float(m["avg_retrievability"])

        return self._obs(), reward, done, StepInfo(
            acquired=m["acquired"], avg_retrievability=m["avg_retrievability"]
        )

    # ---- helpers ----------------------------------------------------------

    def _obs(self) -> np.ndarray:
        return _featurize(self.sim, self.turn, self.n_sessions,
                          self.turns_per_session)

    def _directive_to_words(self, n_new: int, n_active: int,
                            n_review: int) -> list[int]:
        """Translate a (new, active, review) count directive into concrete word IDs.

        Buckets:
          review : seen words with stability >= ACTIVE_STABILITY_THRESHOLD
          active : seen words with 0 < stability < ACTIVE_STABILITY_THRESHOLD
          new    : unseen words (stability == 0)

        Selection within bucket:
          review/active : lowest retrievability first (most due)
          new           : easiest first (lowest difficulty)

        Underflow handling: if a bucket can't supply its quota, the shortfall
        is filled from the remaining pools in order review -> active -> new.
        """
        snap = self.sim.snapshot()
        review_pool = sorted(
            [w for w in snap if w["seen"]
             and w["stability"] >= ACTIVE_STABILITY_THRESHOLD],
            key=lambda w: w["retrievability"],
        )
        active_pool = sorted(
            [w for w in snap if w["seen"]
             and w["stability"] < ACTIVE_STABILITY_THRESHOLD],
            key=lambda w: w["retrievability"],
        )
        new_pool = sorted(
            [w for w in snap if not w["seen"]],
            key=lambda w: w["difficulty"],
        )

        picks: list[int] = []
        seen_ids: set[int] = set()

        def take(pool, quota):
            taken = 0
            for w in pool:
                if taken >= quota:
                    return
                wid = w["word_id"]
                if wid in seen_ids:
                    continue
                picks.append(wid)
                seen_ids.add(wid)
                taken += 1

        take(review_pool, n_review)
        take(active_pool, n_active)
        take(new_pool, n_new)

        # fill any shortfall from any pool
        for pool in (review_pool, active_pool, new_pool):
            if len(picks) >= WORDS_PER_TURN:
                break
            take(pool, WORDS_PER_TURN - len(picks))

        return picks[:WORDS_PER_TURN]
