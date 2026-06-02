"""Evaluate IQL policy alongside baselines in VocabEnv. Produces comparison plots."""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from baseline_actions import (FSRSAction, FSRSPlusNewAction, RandomAction,
                              RuleBasedAction)
from env import ACTION_TABLE, NUM_ACTIONS, VocabEnv
from iql import IQLConfig, IQLTrainer


N_SEEDS = 5


def run_policy(policy_fn, env_seed: int, policy_rng_seed: int
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (acquired_per_turn, avg_r_per_turn, action_per_turn)."""
    env = VocabEnv(seed=env_seed)
    rng = np.random.default_rng(policy_rng_seed)
    s = env.reset()
    acquired = np.zeros(env.total_turns)
    avg_r = np.zeros(env.total_turns)
    actions = np.zeros(env.total_turns, dtype=np.int64)
    for t in range(env.total_turns):
        a = policy_fn(s, rng)
        actions[t] = int(a)
        s, _r, _done, info = env.step(int(a))
        acquired[t] = info.acquired
        avg_r[t] = info.avg_retrievability
    return acquired, avg_r, actions


def evaluate(name: str, policy_fn) -> dict:
    acqs, rrs, acts = [], [], []
    for seed in range(N_SEEDS):
        a, r, ac = run_policy(policy_fn, env_seed=seed,
                              policy_rng_seed=seed + 10_000)
        acqs.append(a); rrs.append(r); acts.append(ac)
    acqs = np.stack(acqs); rrs = np.stack(rrs); acts = np.stack(acts)
    return {
        "name": name,
        "acq_mean": acqs.mean(axis=0), "acq_std": acqs.std(axis=0),
        "r_mean": rrs.mean(axis=0),    "r_std": rrs.std(axis=0),
        "actions": acts,  # (n_seeds, total_turns)
    }


def plot_metric(results, key_mean, key_std, ylabel, title, outpath,
                total_turns, turns_per_session, n_sessions):
    plt.figure(figsize=(9, 5.5))
    x = np.arange(1, total_turns + 1)
    for res in results:
        m = res[key_mean]; sd = res[key_std]
        line, = plt.plot(x, m, label=res["name"], linewidth=2)
        plt.fill_between(x, m - sd, m + sd, alpha=0.18, color=line.get_color())
    for s in range(1, n_sessions):
        plt.axvline(s * turns_per_session, color="gray", alpha=0.25,
                    linestyle="--", linewidth=0.8)
    plt.xlabel("Turn (across sessions)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc="best", frameon=True)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()
    print(f"  wrote {outpath}")


def plot_action_distribution(results, outdir: str, turns_per_session: int) -> None:
    """For each policy, show what action it chose at each turn-in-session.

    Aggregates across all sessions and all seeds: P(action | turn_in_session).
    Reveals whether the policy adapts pacing (e.g., introduces new words early
    in a session and reviews late) -- exactly the qualitative claim the
    proposal predicts for the learned policy.
    """
    n_policies = len(results)
    fig, axes = plt.subplots(1, n_policies, figsize=(3.6 * n_policies, 4.5),
                             sharey=True)
    if n_policies == 1:
        axes = [axes]

    # Group actions by "new word count" for a compact visual --- 6 categories
    # (n_new = 0..5) instead of 21 raw actions.
    new_counts = np.array([t[0] for t in ACTION_TABLE])
    n_buckets = 6
    cmap = plt.cm.viridis(np.linspace(0.1, 0.9, n_buckets))

    for ax, res in zip(axes, results):
        acts = res["actions"]  # (n_seeds, total_turns)
        # Reshape to (n_seeds, n_sessions, turns_per_session)
        n_seeds, total = acts.shape
        n_sessions = total // turns_per_session
        acts = acts[:, :n_sessions * turns_per_session].reshape(
            n_seeds, n_sessions, turns_per_session)
        # For each turn_in_session, fraction of (seed, session) pairs that
        # used each n_new bucket
        fracs = np.zeros((turns_per_session, n_buckets))
        flat = acts.reshape(-1, turns_per_session)  # (n_seeds*n_sessions, T)
        denom = flat.shape[0]
        for t in range(turns_per_session):
            buckets = new_counts[flat[:, t]]
            for b in range(n_buckets):
                fracs[t, b] = (buckets == b).sum() / denom

        # Stacked bar (one stack per turn-in-session)
        bottom = np.zeros(turns_per_session)
        x = np.arange(1, turns_per_session + 1)
        for b in range(n_buckets):
            ax.bar(x, fracs[:, b], bottom=bottom, color=cmap[b],
                   label=f"{b} new" if ax is axes[0] else None,
                   width=0.95, edgecolor="none")
            bottom += fracs[:, b]
        ax.set_title(res["name"])
        ax.set_xlabel("Turn within session")
        ax.set_xlim(0.5, turns_per_session + 0.5)
        ax.set_ylim(0.0, 1.0)
    axes[0].set_ylabel("Fraction of actions")
    axes[0].legend(loc="lower left", title="New words\nin directive",
                   fontsize=8, framealpha=0.85)
    fig.suptitle("Action mix by turn-in-session "
                 "(stack = directives grouped by # new words)")
    plt.tight_layout()
    out = os.path.join(outdir, "action_distribution.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="iql_policy.pt")
    parser.add_argument("--outdir", default=".")
    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    policy_path = (args.policy if os.path.isabs(args.policy)
                   else os.path.join(here, args.policy))
    outdir = args.outdir if os.path.isabs(args.outdir) else os.path.join(here, args.outdir)

    # Load IQL policy
    env_probe = VocabEnv()
    trainer = IQLTrainer(state_dim=env_probe.state_dim,
                         num_actions=NUM_ACTIONS, cfg=IQLConfig())
    trainer.load(policy_path)
    print(f"Loaded IQL policy from {policy_path}")

    def iql_fn(state, rng):
        return trainer.act(state, deterministic=True)

    behaviors = [
        ("Random", RandomAction()),
        ("FSRS", FSRSAction()),
        ("FSRS+New", FSRSPlusNewAction()),
        ("RuleBased", RuleBasedAction()),
        ("IQL (ours)", iql_fn),
    ]

    print(f"Evaluating {len(behaviors)} policies x {N_SEEDS} seeds...")
    results = []
    for name, fn in behaviors:
        print(f"  policy: {name}")
        results.append(evaluate(name, fn))

    print("\nFinal-turn metrics (mean over seeds):")
    for res in results:
        print(f"  {res['name']:12s}  acquired={res['acq_mean'][-1]:5.2f}  "
              f"avg_R={res['r_mean'][-1]:.3f}")

    env = VocabEnv()
    plot_metric(results, "acq_mean", "acq_std",
                ylabel="Acquired words (retrievability > 0.9)",
                title="Vocabulary Acquisition: IQL vs Baselines",
                outpath=os.path.join(outdir, "iql_acquisition.png"),
                total_turns=env.total_turns,
                turns_per_session=env.turns_per_session,
                n_sessions=env.n_sessions)
    plot_metric(results, "r_mean", "r_std",
                ylabel="Mean retrievability of seen words",
                title="Average Retrievability: IQL vs Baselines",
                outpath=os.path.join(outdir, "iql_retrievability.png"),
                total_turns=env.total_turns,
                turns_per_session=env.turns_per_session,
                n_sessions=env.n_sessions)
    plot_action_distribution(results, outdir, env.turns_per_session)


if __name__ == "__main__":
    main()
