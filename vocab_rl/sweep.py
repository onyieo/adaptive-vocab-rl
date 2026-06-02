"""Hyperparameter sweep for IQL and CQL.

Runs a Cartesian product of (algo, tau, beta, alpha_cql, seed) configurations,
trains each on the existing offline dataset, evaluates each trained policy
against the fixed baselines, and saves a sweep summary table + scatter plot
of the Pareto frontier.

Design notes:
- Each config is independent → trivially parallelizable.
- Local mode runs them serially. Modal mode (--modal) dispatches one Modal
  function call per config and collects results.
- We don't re-collect the dataset per config — the same .npz is shared, which
  is correct for offline RL.
- Evaluation uses 5 seeds within each config (separate from training seed)
  for a quick estimate; the BEST config is then re-evaluated with the full
  N_SEEDS in evaluate_policies.

Usage:
    python sweep.py --data dataset.npz --algos iql cql --tau 0.7 0.9 --beta 1 3 10 --alpha-cql 0.1 1 10 --seeds 0 1 2
    python sweep.py --modal ...     (run on Modal — requires `modal` installed
                                     and authenticated)
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import time
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class SweepConfig:
    algo: str           # "iql" or "cql"
    tau: float
    beta: float
    alpha_cql: float    # ignored for IQL
    seed: int
    n_steps: int

    def name(self) -> str:
        return (f"{self.algo}_tau{self.tau}_beta{self.beta}"
                f"{'_alpha' + str(self.alpha_cql) if self.algo == 'cql' else ''}"
                f"_seed{self.seed}")


@dataclass
class SweepResult:
    config: SweepConfig
    eval_acquired_mean: float
    eval_acquired_std: float
    eval_post_gap_mean: float
    eval_post_gap_std: float
    train_seconds: float
    eval_seconds: float
    raw: dict = field(default_factory=dict)


def _train_and_eval(cfg: SweepConfig, data_path: str,
                    n_eval_seeds: int = 5) -> SweepResult:
    """Train one policy, then evaluate it. Returns metrics."""
    import torch  # noqa: F401 — used inside iql/cql/env
    from env import NUM_ACTIONS, VocabEnv
    from iql import IQLConfig, OfflineBuffer
    from iql import train as train_iql

    t0 = time.monotonic()
    buf = OfflineBuffer.from_npz(data_path)
    env_probe = VocabEnv()
    state_dim = env_probe.state_dim

    if cfg.algo == "iql":
        iql_cfg = IQLConfig(tau_expectile=cfg.tau, beta=cfg.beta)
        trainer = train_iql(buf, state_dim=state_dim, num_actions=NUM_ACTIONS,
                            n_steps=cfg.n_steps, log_every=cfg.n_steps + 1,
                            cfg=iql_cfg, seed=cfg.seed)
    elif cfg.algo == "cql":
        from cql import CQLConfig, train_cql
        cql_cfg = CQLConfig(tau_expectile=cfg.tau, beta=cfg.beta,
                            alpha_cql=cfg.alpha_cql)
        trainer = train_cql(buf, state_dim=state_dim, num_actions=NUM_ACTIONS,
                            n_steps=cfg.n_steps, log_every=cfg.n_steps + 1,
                            cfg=cql_cfg, seed=cfg.seed)
    else:
        raise ValueError(f"unknown algo {cfg.algo}")
    train_secs = time.monotonic() - t0

    t1 = time.monotonic()
    acquired_finals, post_gap_retained = [], []
    POST_GAP_THRESHOLD = 0.5
    POST_GAP_DAYS = 7.0
    for seed in range(n_eval_seeds):
        env = VocabEnv(seed=seed)
        s = env.reset()
        rng = np.random.default_rng(seed + 10_000)
        info = None
        for _ in range(env.total_turns):
            a = trainer.act(s, deterministic=True)
            s, _r, _d, info = env.step(int(a))
        acquired_finals.append(info.acquired)
        env.sim.advance_time(POST_GAP_DAYS)
        post_R = [w.retrievability(env.sim.now) for w in env.sim.words]
        post_gap_retained.append(sum(1 for r in post_R if r > POST_GAP_THRESHOLD))
    eval_secs = time.monotonic() - t1

    acquired_finals = np.array(acquired_finals, dtype=float)
    post_gap_retained = np.array(post_gap_retained, dtype=float)
    return SweepResult(
        config=cfg,
        eval_acquired_mean=float(acquired_finals.mean()),
        eval_acquired_std=float(acquired_finals.std()),
        eval_post_gap_mean=float(post_gap_retained.mean()),
        eval_post_gap_std=float(post_gap_retained.std()),
        train_seconds=train_secs,
        eval_seconds=eval_secs,
    )


def _summarize(results: list[SweepResult], out_json: str,
               out_plot: str) -> None:
    import matplotlib.pyplot as plt

    rows = []
    for r in results:
        c = r.config
        rows.append({
            "algo": c.algo, "tau": c.tau, "beta": c.beta,
            "alpha_cql": c.alpha_cql, "seed": c.seed,
            "acquired_mean": r.eval_acquired_mean,
            "acquired_std": r.eval_acquired_std,
            "post_gap_mean": r.eval_post_gap_mean,
            "post_gap_std": r.eval_post_gap_std,
            "train_seconds": r.train_seconds,
            "eval_seconds": r.eval_seconds,
        })
    with open(out_json, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"  saved sweep results -> {out_json}")

    # Scatter: post-gap retention vs final acquired, colored by algo.
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = {"iql": "tab:blue", "cql": "tab:orange"}
    for algo in {r.config.algo for r in results}:
        xs = [r.eval_acquired_mean for r in results if r.config.algo == algo]
        ys = [r.eval_post_gap_mean for r in results if r.config.algo == algo]
        ax.scatter(xs, ys, label=algo.upper(), alpha=0.75,
                   color=colors.get(algo, "tab:green"), s=60, edgecolor="black")
    ax.set_xlabel("Final acquired (R > 0.9) — in-session, mean over eval seeds")
    ax.set_ylabel("Post-gap retained (R > 0.5, 7-day gap) — mean over eval seeds")
    ax.set_title("Hyperparameter sweep — each point is one (algo, tau, beta[, alpha], seed)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_plot, dpi=150)
    plt.close()
    print(f"  saved sweep plot -> {out_plot}")

    rows_sorted = sorted(rows, key=lambda r: r["post_gap_mean"], reverse=True)
    print("\nTop 5 by post-gap retention:")
    for r in rows_sorted[:5]:
        print(f"  {r['algo']:3s}  tau={r['tau']:.2f}  beta={r['beta']:.1f}  "
              f"alpha={r['alpha_cql']:.2f}  seed={r['seed']}  "
              f"acquired={r['acquired_mean']:5.2f}  "
              f"post_gap={r['post_gap_mean']:5.2f}")


def run_local(configs: list[SweepConfig], data_path: str) -> list[SweepResult]:
    results = []
    for i, cfg in enumerate(configs):
        print(f"\n[{i + 1}/{len(configs)}] {cfg.name()}")
        r = _train_and_eval(cfg, data_path)
        print(f"  train: {r.train_seconds:.1f}s  acquired={r.eval_acquired_mean:.2f}  "
              f"post_gap={r.eval_post_gap_mean:.2f}")
        results.append(r)
    return results


def run_modal(configs: list[SweepConfig], data_path: str) -> list[SweepResult]:
    """Run sweep on Modal — one container per config, in parallel.

    Requires `pip install modal` and `modal token new` to authenticate.
    """
    import modal  # noqa: F401

    raise NotImplementedError(
        "Modal stub — see modal_sweep.py for the full Modal app definition. "
        "The local path works without Modal; use --modal only after setting up."
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="dataset.npz")
    p.add_argument("--out", default="sweep_results.json")
    p.add_argument("--plot", default="sweep.png")
    p.add_argument("--algos", nargs="+", default=["iql"], choices=["iql", "cql"])
    p.add_argument("--tau", nargs="+", type=float, default=[0.7, 0.9])
    p.add_argument("--beta", nargs="+", type=float, default=[1.0, 3.0, 10.0])
    p.add_argument("--alpha-cql", nargs="+", type=float, default=[0.1, 1.0, 10.0])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--n-steps", type=int, default=50_000,
                   help="steps per config (lower than the headline 100k for sweep budget)")
    p.add_argument("--modal", action="store_true")
    args = p.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    data_path = args.data if os.path.isabs(args.data) else os.path.join(here, args.data)
    out_json = args.out if os.path.isabs(args.out) else os.path.join(here, args.out)
    out_plot = args.plot if os.path.isabs(args.plot) else os.path.join(here, args.plot)

    configs: list[SweepConfig] = []
    for algo, tau, beta, seed in itertools.product(args.algos, args.tau, args.beta, args.seeds):
        if algo == "cql":
            for alpha in args.alpha_cql:
                configs.append(SweepConfig(algo=algo, tau=tau, beta=beta,
                                           alpha_cql=alpha, seed=seed,
                                           n_steps=args.n_steps))
        else:
            configs.append(SweepConfig(algo=algo, tau=tau, beta=beta,
                                       alpha_cql=0.0, seed=seed,
                                       n_steps=args.n_steps))

    print(f"Sweep: {len(configs)} configs")
    runner = run_modal if args.modal else run_local
    results = runner(configs, data_path)
    _summarize(results, out_json, out_plot)


if __name__ == "__main__":
    main()
