"""Evaluate a trained IQL/CQL policy on held-out simulator parameterizations.

Tests the proposal's central robustness claim: if the policy learned
generalizable scheduling structure (rather than overfitting to the training
simulator's exact dynamics), it should degrade gracefully---not collapse---
when forgetting rates and difficulty distributions are perturbed.

Eval grid:
  - default              (forgetting=9.0, difficulty_scale=1.0)   <- training distribution
  - fast_forgetting      (forgetting=5.0)
  - slow_forgetting      (forgetting=15.0)
  - harder_words         (difficulty_scale=1.3)
  - easier_words         (difficulty_scale=0.7)

For each variant we report final-turn acquired and mean retrievability,
averaged over seeds, for the learned policy vs. the strongest baseline
(FSRS+New in our setup).
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from baseline_actions import FSRSAction, FSRSPlusNewAction, RandomAction
from env import NUM_ACTIONS, VocabEnv
from iql import IQLConfig, IQLTrainer


N_SEEDS = 5

VARIANTS: list[tuple[str, dict]] = [
    ("default",        {}),                                          # in-distribution baseline
    ("fast_forget",    {"forgetting_constant": 5.0}),                # extrapolates below train range (7-12)
    ("slow_forget",    {"forgetting_constant": 15.0}),               # extrapolates above train range
    ("harder_words",   {"difficulty_scale": 1.3}),                   # extrapolates above (train: 0.85-1.15)
    ("easier_words",   {"difficulty_scale": 0.7}),                   # extrapolates below
    ("combo_ood",      {"forgetting_constant": 5.0,
                        "difficulty_scale": 1.3}),                   # joint OOD — worst case
]


def run_one(policy_fn, env_seed: int, policy_rng_seed: int,
            sim_kwargs: dict) -> tuple[float, float]:
    env = VocabEnv(seed=env_seed, sim_kwargs=sim_kwargs)
    rng = np.random.default_rng(policy_rng_seed)
    s = env.reset()
    final_acq = 0; final_r = 0.0
    for _ in range(env.total_turns):
        a = policy_fn(s, rng)
        s, _r, _d, info = env.step(int(a))
        final_acq = info.acquired
        final_r = info.avg_retrievability
    return float(final_acq), float(final_r)


def evaluate(policy_fn, sim_kwargs: dict) -> tuple[np.ndarray, np.ndarray]:
    accs, rrs = [], []
    for seed in range(N_SEEDS):
        a, r = run_one(policy_fn, env_seed=seed,
                       policy_rng_seed=seed + 10_000, sim_kwargs=sim_kwargs)
        accs.append(a); rrs.append(r)
    return np.array(accs), np.array(rrs)


def bar_plot(rows: list[dict], variant_names: list[str], outpath: str) -> None:
    n_var = len(variant_names)
    n_pol = len(rows)
    x = np.arange(n_var)
    width = 0.8 / n_pol

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for i, row in enumerate(rows):
        offset = (i - (n_pol - 1) / 2) * width
        axes[0].bar(x + offset, row["acq_mean"], width, yerr=row["acq_std"],
                    label=row["name"], capsize=3)
        axes[1].bar(x + offset, row["r_mean"], width, yerr=row["r_std"],
                    label=row["name"], capsize=3)

    for ax, ylabel, title in [
        (axes[0], "Acquired words (R>0.9)", "Final acquisition across simulator variants"),
        (axes[1], "Mean retrievability", "Final retrievability across simulator variants"),
    ]:
        ax.set_xticks(x)
        ax.set_xticklabels(variant_names, rotation=20, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis="y")
        ax.legend(loc="best", frameon=True)

    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()
    print(f"  wrote {outpath}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--policy", default="iql_policy.pt")
    p.add_argument("--out", default="robustness.png")
    args = p.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    policy_path = args.policy if os.path.isabs(args.policy) else os.path.join(here, args.policy)
    out_path = args.out if os.path.isabs(args.out) else os.path.join(here, args.out)

    env_probe = VocabEnv()
    # Try CQL first (newer/better), fall back to IQL on TypeError/KeyError
    # from a state_dict mismatch. Label the policy by the filename stem.
    policy_label = os.path.basename(policy_path).split(".")[0]
    if "cql" in policy_label.lower():
        from cql import CQLConfig, CQLTrainer
        trainer = CQLTrainer(state_dim=env_probe.state_dim,
                             num_actions=NUM_ACTIONS, cfg=CQLConfig())
    elif "ppo" in policy_label.lower():
        trainer = None  # PPO has different load path
    else:
        trainer = IQLTrainer(state_dim=env_probe.state_dim,
                             num_actions=NUM_ACTIONS, cfg=IQLConfig())

    if trainer is not None:
        trainer.load(policy_path)
        def learned_fn(state, rng):
            return trainer.act(state, deterministic=True)
    else:
        # PPO branch
        import torch
        from ppo import load as load_ppo, act_deterministic, PPOConfig  # noqa: F401
        ppo_model = load_ppo(policy_path)
        ppo_device = torch.device("cpu")
        def learned_fn(state, rng):
            return act_deterministic(ppo_model, state, ppo_device)

    print(f"Loaded policy from {policy_path} (label: {policy_label})")

    policies = [
        (policy_label, learned_fn),
        ("FSRS+New", FSRSPlusNewAction()),
        ("FSRS", FSRSAction()),
        ("Random", RandomAction()),
    ]

    print(f"Evaluating across {len(VARIANTS)} simulator variants x "
          f"{len(policies)} policies x {N_SEEDS} seeds...\n")

    print(f"{'variant':<15}{'policy':<14}{'acq mean±sd':>16}{'R mean±sd':>16}")
    print("-" * 61)
    rows: list[dict] = []
    for name, fn in policies:
        row = {"name": name, "acq_mean": [], "acq_std": [],
               "r_mean": [], "r_std": []}
        for vname, kw in VARIANTS:
            acc, rr = evaluate(fn, kw)
            row["acq_mean"].append(acc.mean())
            row["acq_std"].append(acc.std())
            row["r_mean"].append(rr.mean())
            row["r_std"].append(rr.std())
            print(f"{vname:<15}{name:<14}"
                  f"{acc.mean():7.2f} ± {acc.std():4.2f}    "
                  f"{rr.mean():.3f} ± {rr.std():.3f}")
        rows.append({k: (np.array(v) if isinstance(v, list)
                         and not isinstance(v[0], str) else v)
                     for k, v in row.items()})
        print()

    bar_plot(rows, [v[0] for v in VARIANTS], out_path)


if __name__ == "__main__":
    main()
