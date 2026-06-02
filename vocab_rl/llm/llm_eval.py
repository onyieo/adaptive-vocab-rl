"""Run each policy for 1 session (20 turns) with real LLM-generated conversation
turns, score each turn with the LLM judge, save transcripts, and plot mean
naturalness per policy.

Cost note: roughly 5 policies x 20 turns x 2 API calls = 200 calls total.
Tutor system prompt is cached after the first call within a session.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Make vocab_rl/ importable when this file is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from baseline_actions import (FSRSAction, FSRSPlusNewAction, RandomAction,    # noqa: E402
                              RuleBasedAction)
from env import ACTION_TABLE, NUM_ACTIONS, VocabEnv                           # noqa: E402
from iql import IQLConfig, IQLTrainer                                         # noqa: E402
from simulator import VOCAB                                                   # noqa: E402

from llm.judge import Judge                                                   # noqa: E402
from llm.tutor import Tutor                                                   # noqa: E402


N_TURNS = 20


def load_iql(policy_path: str):
    """Return a (state, rng)->action callable, or None if no checkpoint exists."""
    if not os.path.exists(policy_path):
        return None
    env = VocabEnv()
    trainer = IQLTrainer(state_dim=env.state_dim, num_actions=NUM_ACTIONS,
                         cfg=IQLConfig())
    trainer.load(policy_path)

    def fn(state, _rng):
        return trainer.act(state, deterministic=True)
    return fn


def _pick_emphasized(env: VocabEnv, word_ids: list[int]) -> int | None:
    """Emphasize the first newly-introduced word if any, else the first picked."""
    if not word_ids:
        return None
    snap = env.sim.snapshot()
    for wid in word_ids:
        if not snap[wid]["seen"]:
            return wid
    return word_ids[0]


def run_policy_with_llm(name: str, policy_fn, seed: int = 0) -> list[dict]:
    env = VocabEnv(seed=seed)
    rng = np.random.default_rng(seed + 10_000)
    tutor = Tutor()
    judge = Judge()

    state = env.reset()
    transcript: list[dict] = []
    prior_text: str | None = None

    for t in range(N_TURNS):
        action = int(policy_fn(state, rng))
        n_new, n_active, n_review = ACTION_TABLE[action]
        word_ids = env._directive_to_words(n_new, n_active, n_review)
        emphasized = _pick_emphasized(env, word_ids)

        tutor_result = tutor.generate_turn(word_ids, emphasized_id=emphasized)
        text = tutor_result["text"]
        score, reason = judge.score(text, prior_turn_text=prior_text)

        transcript.append({
            "turn": t,
            "action_index": action,
            "directive_counts": {"new": n_new, "active": n_active, "review": n_review},
            "word_ids": [int(w) for w in word_ids],
            "words": [
                {"surface": VOCAB[w][0], "reading": VOCAB[w][1],
                 "gloss": VOCAB[w][2], "jlpt": VOCAB[w][3]}
                for w in word_ids
            ],
            "emphasized": int(emphasized) if emphasized is not None else None,
            "text": text,
            "naturalness": score,
            "naturalness_reason": reason,
            "tutor_usage": tutor_result["usage"],
        })
        print(f"  turn {t:2d}: score={score}  | {text[:80]}...")

        prior_text = text
        state, _r, _done, _info = env.step(action)

    return transcript


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--iql-policy", default="iql_policy.pt")
    p.add_argument("--outdir", default="llm_eval_results")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    here = Path(__file__).resolve().parent.parent  # vocab_rl/
    iql_path = (args.iql_policy if os.path.isabs(args.iql_policy)
                else str(here / args.iql_policy))
    outdir = args.outdir if os.path.isabs(args.outdir) else str(here / args.outdir)
    os.makedirs(outdir, exist_ok=True)

    policies: list[tuple[str, object]] = [
        ("Random",    RandomAction()),
        ("FSRS",      FSRSAction()),
        ("FSRS+New",  FSRSPlusNewAction()),
        ("RuleBased", RuleBasedAction()),
    ]
    iql_fn = load_iql(iql_path)
    if iql_fn is not None:
        policies.append(("IQL", iql_fn))
        print(f"loaded IQL policy from {iql_path}")
    else:
        print(f"(skipping IQL — no policy found at {iql_path})")

    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    for name, fn in policies:
        print(f"\n=== {name} ===")
        transcript = run_policy_with_llm(name, fn, seed=args.seed)
        safe = name.replace("+", "plus")
        out_path = os.path.join(outdir, f"transcript_{safe}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"policy": name, "seed": args.seed, "turns": transcript},
                      f, indent=2, ensure_ascii=False)
        scores = np.array([t["naturalness"] for t in transcript], dtype=float)
        means[name] = float(scores.mean())
        stds[name]  = float(scores.std())
        print(f"  mean naturalness: {means[name]:.2f} +- {stds[name]:.2f}")
        print(f"  saved -> {out_path}")

    names = list(means.keys())
    mvals = [means[n] for n in names]
    svals = [stds[n] for n in names]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, mvals, yerr=svals, capsize=5,
                  color="steelblue", edgecolor="black")
    for bar, m in zip(bars, mvals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.08,
                f"{m:.2f}", ha="center", fontsize=10)
    ax.set_ylim(0, 5.5)
    ax.set_ylabel("Mean naturalness (LLM judge, 1-5)")
    ax.set_title(f"Conversation naturalness by policy ({N_TURNS} turns, seed={args.seed})")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    chart_path = os.path.join(outdir, "naturalness.png")
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"\nsaved chart -> {chart_path}")

    summary = {n: {"mean": means[n], "std": stds[n]} for n in names}
    with open(os.path.join(outdir, "summary.json"), "w") as f:
        json.dump({"seed": args.seed, "n_turns": N_TURNS, "naturalness": summary},
                  f, indent=2)


if __name__ == "__main__":
    main()
