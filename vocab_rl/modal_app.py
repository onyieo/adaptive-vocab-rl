"""Modal app: parallel hyperparameter sweep for the full v3 pipeline.

Each Modal function call runs ONE config end-to-end:
  collect -> distill (if PPO available) -> train -> evaluate
and logs metrics to WandB.

The local driver dispatches all configs in parallel and exits — Modal jobs
keep running on Modal's infra even if the laptop disconnects. WandB stores
all results in the cloud.

Setup (one-time):
    pip install modal wandb
    modal token new
    wandb login
    # Secrets used: 'wandb' (existing) and 'llm-keys' (created by sweep launch)

Run:
    python modal_app.py            # local entrypoint dispatches the sweep
    # OR
    modal run modal_app.py::sweep

Inspect results: https://wandb.ai/<your-handle>/adaptive-vocab-rl
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import modal


HERE = Path(__file__).resolve().parent

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "matplotlib", "torch", "wandb",
                 "anthropic", "openai", "python-dotenv")
    .add_local_dir(str(HERE), "/vocab_rl")
)

app = modal.App("adaptive-vocab-rl", image=image)


@app.function(
    timeout=60 * 60,
    cpu=4,
    memory=8192,
    secrets=[modal.Secret.from_name("wandb"),
             modal.Secret.from_name("llm-keys")],
)
def train_and_eval_one(config: dict) -> dict:
    """Train ONE config and return eval metrics.

    `config` is a dict with keys:
      algo            'iql' | 'cql'
      seed            int
      n_steps         int
      reward_scale    float
      beta            float (IQL only)
      tau             float (IQL only)
      alpha_cql       float (CQL only)
      naturalness_alpha   float (env reward knob)
      retention_lambda    float (env reward knob)
      multi_sim       bool
      rebalanced      bool
      distill_ppo     bool
      n_traj          int
      wandb_project   str
      run_name        str
    """
    sys.path.insert(0, "/vocab_rl")
    import time
    import numpy as np
    import torch
    import wandb

    from baseline_actions import (FSRSAction, FSRSPlusNewAction,
                                  RandomAction, RuleBasedAction,
                                  SmartFSRSEnvAction)
    from collect_dataset import collect
    from env import ACTION_TABLE, NUM_ACTIONS, VocabEnv
    from iql import IQLConfig, OfflineBuffer, train as train_iql
    from cql import CQLConfig, train_cql

    wandb.init(
        project=config.get("wandb_project", "adaptive-vocab-rl"),
        name=config.get("run_name", f"{config['algo']}_seed{config['seed']}"),
        config=config,
    )

    # ---- 1. collect dataset (env uses v3 reward by default) -----------
    t0 = time.monotonic()
    data_path = f"/tmp/dataset_{config['algo']}_{config['seed']}.npz"
    collect(
        n_traj_per_policy=config["n_traj"],
        out_path=data_path,
        seed_start=config["seed"] * 1000,
        multi_sim=config["multi_sim"],
        rebalanced=config["rebalanced"],
    )
    wandb.log({"setup/collect_secs": time.monotonic() - t0})

    # ---- 2. (optional) distill from a pre-trained PPO ----------------
    # Skipped on Modal — PPO checkpoint isn't available remotely. The
    # rebalanced+multi-sim buffer alone should be enough; if distillation
    # turns out to matter we can add a volume-based PPO upload later.

    # ---- 3. train -----------------------------------------------------
    t0 = time.monotonic()
    buf = OfflineBuffer.from_npz(data_path, reward_scale=config["reward_scale"])
    env_probe = VocabEnv(
        naturalness_alpha=config.get("naturalness_alpha", 0.01),
        retention_lambda=config.get("retention_lambda", 1.0),
    )
    state_dim = env_probe.state_dim

    if config["algo"] == "iql":
        cfg = IQLConfig(tau_expectile=config["tau"], beta=config["beta"])
        trainer = train_iql(
            buf, state_dim=state_dim, num_actions=NUM_ACTIONS,
            n_steps=config["n_steps"], cfg=cfg, seed=config["seed"],
            log_every=config["n_steps"] // 20 or 1,
        )
    else:
        cfg = CQLConfig(
            tau_expectile=config.get("tau", 0.7),
            beta=config.get("beta", 3.0),
            alpha_cql=config["alpha_cql"],
        )
        trainer = train_cql(
            buf, state_dim=state_dim, num_actions=NUM_ACTIONS,
            n_steps=config["n_steps"], cfg=cfg, seed=config["seed"],
            log_every=config["n_steps"] // 20 or 1,
        )
    wandb.log({"setup/train_secs": time.monotonic() - t0})

    # ---- 4. evaluate (10 seeds) --------------------------------------
    t0 = time.monotonic()
    POST_GAP_DAYS = 7.0
    POST_GAP_THRESHOLD = 0.5
    N_EVAL_SEEDS = 10

    acquired_finals = []
    post_gap_retained = []
    for s in range(N_EVAL_SEEDS):
        env = VocabEnv(
            seed=s,
            naturalness_alpha=config.get("naturalness_alpha", 0.01),
            retention_lambda=config.get("retention_lambda", 1.0),
        )
        state = env.reset()
        info = None
        for _ in range(env.total_turns):
            a = trainer.act(state, deterministic=True)
            state, _r, _d, info = env.step(int(a))
        acquired_finals.append(info.acquired)
        env.sim.advance_time(POST_GAP_DAYS)
        post_R = [w.retrievability(env.sim.now) for w in env.sim.words]
        post_gap_retained.append(sum(1 for r in post_R if r > POST_GAP_THRESHOLD))

    acquired_finals = np.array(acquired_finals, dtype=float)
    post_gap_retained = np.array(post_gap_retained, dtype=float)
    metrics = {
        "eval/acquired_mean": float(acquired_finals.mean()),
        "eval/acquired_std":  float(acquired_finals.std()),
        "eval/retained_mean": float(post_gap_retained.mean()),
        "eval/retained_std":  float(post_gap_retained.std()),
        "setup/eval_secs":    time.monotonic() - t0,
    }
    wandb.log(metrics)
    wandb.finish()

    return {**config, **metrics,
            "raw_acquired": acquired_finals.tolist(),
            "raw_retained": post_gap_retained.tolist()}


# ---------------------------------------------------------------------------
# Local entrypoint — dispatches the sweep
# ---------------------------------------------------------------------------

import itertools
import json
import time


def _make_cfg(**kw) -> dict:
    """Common defaults; override via kw."""
    base = {
        "n_steps": 80_000,
        "reward_scale": 100.0,
        "tau": 0.7, "beta": 3.0,
        "alpha_cql": 1.0,
        "naturalness_alpha": 0.01,
        "retention_lambda": 1.0,
        "multi_sim": True,
        "rebalanced": True,
        "distill_ppo": False,
        "n_traj": 100,
        "wandb_project": "adaptive-vocab-rl",
    }
    base.update(kw)
    return base


def _build_sweep_configs() -> list[dict]:
    """Focused sweep addressing the Tier-1 concerns from the project audit.

    Three studies, sharing one set of seeds:
      A. Reward sensitivity: vary (naturalness_alpha, retention_lambda)
         at the default IQL/CQL hyperparams. 3 nat * 2 ret = 6 configs * 5 seeds = 30
      B. CQL alpha sensitivity: vary alpha_cql at default reward. 3 * 5 = 15
      C. IQL beta sensitivity: vary beta at default reward. 3 * 5 = 15
      D. Multi-seed at the v3 default config to tighten CIs: 10 extra seeds = 10

    Total = 70 configs. ~30-40 min wall-clock on Modal with 30-way parallelism.
    """
    configs = []
    sweep_seeds = list(range(5))     # 5 seeds for sweeps
    final_seeds = list(range(5, 15)) # 10 more seeds for the final-CI run

    # A. Reward sensitivity (CQL)
    for seed in sweep_seeds:
        for nat_a in [0.005, 0.01, 0.05]:
            for ret_l in [0.5, 2.0]:
                configs.append(_make_cfg(
                    algo="cql", seed=seed,
                    naturalness_alpha=nat_a, retention_lambda=ret_l,
                    run_name=f"A_reward_natA{nat_a}_retL{ret_l}_seed{seed}",
                ))

    # B. CQL alpha sensitivity
    for seed in sweep_seeds:
        for a_cql in [0.1, 5.0, 10.0]:   # 1.0 is the default (covered by D)
            configs.append(_make_cfg(
                algo="cql", seed=seed, alpha_cql=a_cql,
                run_name=f"B_alphaCQL_acql{a_cql}_seed{seed}",
            ))

    # C. IQL beta sensitivity
    for seed in sweep_seeds:
        for beta in [3.0, 10.0, 30.0]:
            configs.append(_make_cfg(
                algo="iql", seed=seed, beta=beta,
                run_name=f"C_IQL_beta{beta}_seed{seed}",
            ))

    # D. CQL default for final-CI (10 more seeds beyond the 5 in sweeps)
    for seed in final_seeds:
        configs.append(_make_cfg(
            algo="cql", seed=seed,
            run_name=f"D_final_cql_seed{seed}",
        ))
    for seed in final_seeds:
        configs.append(_make_cfg(
            algo="iql", seed=seed, beta=10.0,
            run_name=f"D_final_iql_seed{seed}",
        ))

    return configs


@app.local_entrypoint()
def sweep():
    """Dispatch the full sweep to Modal in parallel and exit.

    All results are logged to WandB throughout. The local process can be
    killed once dispatch completes — jobs continue running on Modal.
    """
    configs = _build_sweep_configs()
    print(f"Dispatching {len(configs)} configs to Modal in parallel...")
    print(f"  CQL sweep: alpha_cql x naturalness_alpha x retention_lambda x seed")
    print(f"  Watch live: https://wandb.ai/?project=adaptive-vocab-rl")
    print()

    t0 = time.time()
    results = list(train_and_eval_one.map(configs))
    elapsed = time.time() - t0
    print(f"\nAll {len(results)} configs done in {elapsed/60:.1f} min wall-clock.")

    # Save raw results locally for downstream analysis.
    out_path = HERE / "sweep_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved -> {out_path}")

    # Top 10 by retention
    by_ret = sorted(results, key=lambda r: r["eval/retained_mean"], reverse=True)
    print(f"\nTop 10 configs by post-gap retention:")
    for r in by_ret[:10]:
        print(f"  retain={r['eval/retained_mean']:6.2f}±{r['eval/retained_std']:4.2f}  "
              f"acq={r['eval/acquired_mean']:6.2f}  "
              f"alpha_cql={r['alpha_cql']}  natA={r['naturalness_alpha']}  "
              f"retL={r['retention_lambda']}  seed={r['seed']}")


if __name__ == "__main__":
    # Allow `python modal_app.py` as a shortcut for `modal run modal_app.py::sweep`
    # by re-execing through modal.
    import subprocess
    subprocess.run(["modal", "run", "--detach", f"{__file__}::sweep"], check=True)
