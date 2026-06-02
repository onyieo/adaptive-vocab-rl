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


N_SEEDS = 10
BOOTSTRAP_RESAMPLES = 1000
CI_ALPHA = 0.05  # 95% CI


def _bootstrap_ci(samples: np.ndarray, axis: int = 0,
                  n_resamples: int = BOOTSTRAP_RESAMPLES,
                  alpha: float = CI_ALPHA,
                  rng: np.random.Generator | None = None
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Percentile bootstrap CI on the mean along `axis`.

    Returns (lo, hi) — the alpha/2 and 1-alpha/2 quantiles of the resampled
    means. Works for both 1-D (returns scalars) and 2-D (returns arrays).
    """
    if rng is None:
        rng = np.random.default_rng(0)
    samples = np.asarray(samples)
    n = samples.shape[axis]
    idx = rng.integers(0, n, size=(n_resamples, n))
    # Move resample axis to front, then take means
    resampled = np.take(samples, idx, axis=axis).mean(axis=axis + 1)
    lo = np.quantile(resampled, alpha / 2, axis=0)
    hi = np.quantile(resampled, 1 - alpha / 2, axis=0)
    return lo, hi


POST_GAP_DAYS = 7.0          # measure retention 7 days after the final session
POST_GAP_THRESHOLD = 0.5     # R > 0.5 = the learner can still recall it


def run_policy(policy_fn, env_seed: int, policy_rng_seed: int
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Returns (acquired_per_turn, avg_r_per_turn, action_per_turn, post_gap).

    `post_gap` is a dict of retention metrics measured POST_GAP_DAYS days after
    the final turn: this is the proposal's intended "real" retention measure
    that avoids the freshly-exposed-words-have-R≈1 issue of in-session metrics.

    If policy_fn has a `.bind_to_env(env)` method it's an env-aware policy
    (e.g. the LLM-prompted baseline) and we adapt it per-env.
    """
    env = VocabEnv(seed=env_seed)
    rng = np.random.default_rng(policy_rng_seed)
    s = env.reset()
    fn = policy_fn.bind_to_env(env) if hasattr(policy_fn, "bind_to_env") else policy_fn
    acquired = np.zeros(env.total_turns)
    avg_r = np.zeros(env.total_turns)
    actions = np.zeros(env.total_turns, dtype=np.int64)
    for t in range(env.total_turns):
        a = fn(s, rng)
        actions[t] = int(a)
        s, _r, _done, info = env.step(int(a))
        acquired[t] = info.acquired
        avg_r[t] = info.avg_retrievability

    # Advance time and re-measure: post-gap retention.
    env.sim.advance_time(POST_GAP_DAYS)
    post_R = [w.retrievability(env.sim.now) for w in env.sim.words]
    seen_post_R = [r for r, w in zip(post_R, env.sim.words) if w.stability > 0]
    post_gap = {
        "retained": int(sum(1 for r in post_R if r > POST_GAP_THRESHOLD)),
        "n_seen": len(seen_post_R),
        "avg_R_all": float(np.mean(post_R)),
        "avg_R_seen": float(np.mean(seen_post_R)) if seen_post_R else 0.0,
    }
    return acquired, avg_r, actions, post_gap


def evaluate(name: str, policy_fn) -> dict:
    acqs, rrs, acts = [], [], []
    post_gaps: list[dict] = []
    for seed in range(N_SEEDS):
        a, r, ac, pg = run_policy(policy_fn, env_seed=seed,
                                  policy_rng_seed=seed + 10_000)
        acqs.append(a); rrs.append(r); acts.append(ac)
        post_gaps.append(pg)
    acqs = np.stack(acqs); rrs = np.stack(rrs); acts = np.stack(acts)
    pg_retained = np.array([p["retained"] for p in post_gaps])
    pg_avg_seen = np.array([p["avg_R_seen"] for p in post_gaps])

    acq_lo, acq_hi = _bootstrap_ci(acqs)
    r_lo, r_hi = _bootstrap_ci(rrs)
    pg_lo, pg_hi = _bootstrap_ci(pg_retained)
    pg_r_lo, pg_r_hi = _bootstrap_ci(pg_avg_seen)

    return {
        "name": name,
        "acq_mean": acqs.mean(axis=0), "acq_lo": acq_lo, "acq_hi": acq_hi,
        "r_mean":   rrs.mean(axis=0),  "r_lo":   r_lo,   "r_hi":   r_hi,
        "actions": acts,
        "post_gap_retained_mean": float(pg_retained.mean()),
        "post_gap_retained_ci": (float(pg_lo), float(pg_hi)),
        "post_gap_R_seen_mean": float(pg_avg_seen.mean()),
        "post_gap_R_seen_ci": (float(pg_r_lo), float(pg_r_hi)),
    }


def plot_metric(results, key_mean, key_lo, key_hi, ylabel, title, outpath,
                total_turns, turns_per_session, n_sessions):
    plt.figure(figsize=(9, 5.5))
    x = np.arange(1, total_turns + 1)
    for res in results:
        m = res[key_mean]; lo = res[key_lo]; hi = res[key_hi]
        line, = plt.plot(x, m, label=res["name"], linewidth=2)
        plt.fill_between(x, lo, hi, alpha=0.18, color=line.get_color())
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
    parser.add_argument("--cql-policy", default="cql_policy.pt",
                        help="Optional CQL checkpoint; included if it exists")
    parser.add_argument("--with-llm-policy", action="store_true",
                        help="Include the LLM-prompted baseline (costs API calls)")
    parser.add_argument("--outdir", default=".")
    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    policy_path = (args.policy if os.path.isabs(args.policy)
                   else os.path.join(here, args.policy))
    cql_path = (args.cql_policy if os.path.isabs(args.cql_policy)
                else os.path.join(here, args.cql_policy))
    outdir = args.outdir if os.path.isabs(args.outdir) else os.path.join(here, args.outdir)

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

    if os.path.exists(cql_path):
        from cql import CQLConfig, CQLTrainer
        cql_trainer = CQLTrainer(state_dim=env_probe.state_dim,
                                 num_actions=NUM_ACTIONS, cfg=CQLConfig())
        cql_trainer.load(cql_path)
        print(f"Loaded CQL policy from {cql_path}")

        def cql_fn(state, rng):
            return cql_trainer.act(state, deterministic=True)

        behaviors.append(("CQL (ours)", cql_fn))

    if args.with_llm_policy:
        from llm.llm_policy import LLMPolicy
        behaviors.append(("LLMPrompted", LLMPolicy(turns_per_session=env_probe.turns_per_session)))
        print("Including LLM-prompted baseline (API calls follow)")

    print(f"Evaluating {len(behaviors)} policies x {N_SEEDS} seeds...")
    results = []
    for name, fn in behaviors:
        print(f"  policy: {name}")
        results.append(evaluate(name, fn))

    print("\nFinal-turn metrics (mean over seeds):")
    for res in results:
        print(f"  {res['name']:12s}  acquired={res['acq_mean'][-1]:5.2f}  "
              f"avg_R={res['r_mean'][-1]:.3f}")

    print(f"\nPost-gap retention ({POST_GAP_DAYS:.0f} days after final session, "
          f"R > {POST_GAP_THRESHOLD}):")
    for res in results:
        lo, hi = res["post_gap_retained_ci"]
        print(f"  {res['name']:12s}  retained={res['post_gap_retained_mean']:5.2f} "
              f"[{lo:5.2f}, {hi:5.2f}]  "
              f"avg_R_seen={res['post_gap_R_seen_mean']:.3f}")

    env = VocabEnv()
    plot_metric(results, "acq_mean", "acq_lo", "acq_hi",
                ylabel="Acquired words (retrievability > 0.9)",
                title=f"Vocabulary Acquisition: IQL vs Baselines (n={N_SEEDS} seeds, 95% bootstrap CI)",
                outpath=os.path.join(outdir, "iql_acquisition.png"),
                total_turns=env.total_turns,
                turns_per_session=env.turns_per_session,
                n_sessions=env.n_sessions)
    plot_metric(results, "r_mean", "r_lo", "r_hi",
                ylabel="Mean retrievability of seen words",
                title=f"Average Retrievability: IQL vs Baselines (n={N_SEEDS} seeds, 95% bootstrap CI)",
                outpath=os.path.join(outdir, "iql_retrievability.png"),
                total_turns=env.total_turns,
                turns_per_session=env.turns_per_session,
                n_sessions=env.n_sessions)
    plot_action_distribution(results, outdir, env.turns_per_session)
    plot_post_gap_retention(results, outdir)


def plot_post_gap_retention(results, outdir: str) -> None:
    """Bar chart of post-gap retention (R > threshold after a multi-day gap).

    The proposal's intended retention metric — avoids the freshly-exposed-words
    have-R≈1 issue of in-session metrics. Y-axis: # words still retrievable.
    """
    names = [r["name"] for r in results]
    means = [r["post_gap_retained_mean"] for r in results]
    los = [m - r["post_gap_retained_ci"][0] for m, r in zip(means, results)]
    his = [r["post_gap_retained_ci"][1] - m for m, r in zip(means, results)]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(names, means, yerr=[los, his], capsize=5,
                  color="steelblue", edgecolor="black")
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{m:.1f}", ha="center", fontsize=10)
    ax.set_ylabel(f"Words retained (R > {POST_GAP_THRESHOLD})")
    ax.set_title(
        f"Post-gap retention: {POST_GAP_DAYS:.0f} days after final session "
        f"(n={N_SEEDS} seeds, 95% bootstrap CI)"
    )
    ax.grid(True, alpha=0.3, axis="y")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    out = os.path.join(outdir, "post_gap_retention.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
