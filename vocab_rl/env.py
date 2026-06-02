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

# === Reward design (v3) ===
#
# Three-component reward that better matches the actual goal of vocabulary
# tutoring than the v1 "coverage growth" reward (which was trivially won by
# Random because it just rewards exposure breadth).
#
# Components:
#   (a) COVERAGE  — per-step change in total retrievability across the FULL
#       vocabulary. Same as v1; the dense within-session learning signal.
#       coverage_t = (sum_R_after - sum_R_before) / N_WORDS
#
#   (b) NATURALNESS PENALTY — per-step penalty proportional to the squared
#       excess of new words above 1. Encodes the empirical finding from our
#       LLM-judge eval: turns introducing more than ~1 new word at a time
#       score lower on conversational naturalness (FSRS+New @ 1 new/turn
#       scored 3.30 vs Random/IQL with broader directives scoring 2.25-2.65).
#       naturalness_t = -NATURALNESS_ALPHA * max(0, n_new - 1)^2
#       This biases the policy toward steady pacing without needing the LLM
#       in the training loop.
#
#   (c) RETENTION BONUS at session boundaries — at each session end, bonus
#       proportional to the fraction of words STILL above the retention
#       threshold (R > 0.5) after the inter-session gap. Captures the proposal's
#       intended "long-term retention" reward: words that survive the gap are
#       what we actually want. This term explicitly punishes "introduce a
#       word once and never review" (those words decay below threshold during
#       the gap and don't earn the bonus).
#       retention_t = RETENTION_LAMBDA * (# words with R > RETENTION_THRESHOLD) / N_WORDS
#                     at session boundaries only, 0 otherwise.
#
# Set NATURALNESS_ALPHA=0 to ablate (b); RETENTION_LAMBDA=0 to ablate (c).
# Defaults reproduce the v3 reward; set both to 0 to recover v1.

REWARD_NORMALIZER = 1.0
# Tuned empirically so that (2,0,3) beats (1,0,4) (2.7 vs 1.9 return) but
# (3,0,2) and (5,0,0) get pushed below FSRS+New. Lets the policy adapt
# aggression mildly without devolving to "spray new words everywhere."
NATURALNESS_ALPHA_DEFAULT = 0.01
RETENTION_LAMBDA_DEFAULT = 1.0
RETENTION_THRESHOLD = 0.5


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
                 sim_kwargs: dict | None = None,
                 naturalness_alpha: float = NATURALNESS_ALPHA_DEFAULT,
                 retention_lambda: float = RETENTION_LAMBDA_DEFAULT) -> None:
        self.seed = seed
        self.n_sessions = n_sessions
        self.turns_per_session = turns_per_session
        self.days_between = days_between
        self.sim_kwargs = sim_kwargs or {}
        self.naturalness_alpha = naturalness_alpha
        self.retention_lambda = retention_lambda
        self.total_turns = n_sessions * turns_per_session

        self.sim = StudentSimulator(seed=seed, **self.sim_kwargs)
        self.n_words = self.sim.num_words()
        self.state_dim = _state_dim(self.n_words)
        self.action_dim = NUM_ACTIONS

        self.turn = 0
        self._prev_total_R = 0.0

    # ---- core API ---------------------------------------------------------

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.seed = seed
        self.sim = StudentSimulator(seed=self.seed, **self.sim_kwargs)
        self.turn = 0
        self._prev_total_R = self._total_retrievability()
        return self._obs()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, StepInfo]:
        n_new, n_active, n_review = ACTION_TABLE[action]
        directive = self._directive_to_words(n_new, n_active, n_review)
        self.sim.step(directive)
        self.turn += 1

        # --- (a) COVERAGE: per-step delta total retrievability ------------
        new_total_R = self._total_retrievability()
        coverage = REWARD_NORMALIZER * (new_total_R - self._prev_total_R) / self.n_words
        self._prev_total_R = new_total_R

        # --- (b) NATURALNESS PENALTY: squared excess of new words above 1 -
        naturalness = -self.naturalness_alpha * max(0, n_new - 1) ** 2

        # --- (c) RETENTION BONUS at session boundaries --------------------
        done = self.turn >= self.total_turns
        is_session_end = (self.turn % self.turns_per_session == 0)
        retention_bonus = 0.0
        if is_session_end:
            if not done:
                # Mid-session boundary: advance time, then measure post-gap
                # retention. Re-baseline _prev_total_R so the next turn isn't
                # charged for the inter-session decay.
                self.sim.advance_time(self.days_between)
                post_R = [w.retrievability(self.sim.now) for w in self.sim.words]
                self._prev_total_R = sum(post_R)
            else:
                # Final step: measure right-now retention (no time advance —
                # the post-rollout eval handles the 7-day-after measurement).
                post_R = [w.retrievability(self.sim.now) for w in self.sim.words]
            retention_bonus = (self.retention_lambda
                               * sum(1 for r in post_R if r > RETENTION_THRESHOLD)
                               / self.n_words)

        reward = coverage + naturalness + retention_bonus

        m = self.sim.metrics()
        return self._obs(), reward, done, StepInfo(
            acquired=m["acquired"], avg_retrievability=m["avg_retrievability"]
        )

    def _total_retrievability(self) -> float:
        return sum(w.retrievability(self.sim.now) for w in self.sim.words)

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
