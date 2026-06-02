"""Modal app for parallel hyperparameter sweeps.

Each Modal function call trains one (algo, tau, beta, alpha, seed) config and
returns the eval metrics. Calls run in parallel across Modal containers — a
50-config sweep takes ~5 minutes wall-clock instead of ~5 hours serial.

Setup (one-time):
    pip install modal
    modal token new

Run:
    python modal_sweep.py --algos iql cql --tau 0.7 0.9 --beta 1 3 10 \\
        --alpha-cql 0.1 1 10 --seeds 0 1 2 3 4

Or from inside sweep.py: `python sweep.py --modal ...` (delegates here).

Notes:
- Uploads the local `vocab_rl/` directory and the chosen dataset.npz into the
  container image at build time. Re-collecting the dataset locally requires
  rebuilding the image, which Modal caches.
- GPU is optional. The networks are tiny — CPU is fine and avoids GPU queue
  latency on Modal. Flip `gpu=` if you want to test it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

try:
    import modal
except ImportError:
    print("modal not installed. Run: pip install modal && modal token new")
    sys.exit(1)


HERE = os.path.dirname(os.path.abspath(__file__))


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "matplotlib", "torch")
    .add_local_dir(HERE, "/vocab_rl")
)

app = modal.App("adaptive-vocab-rl-sweep", image=image)


@app.function(
    timeout=60 * 30,        # 30 min cap per config
    cpu=2,
    memory=4096,
)
def train_and_eval_remote(config_dict: dict, data_path_in_container: str) -> dict:
    """Train one config and return eval metrics. Runs inside Modal."""
    sys.path.insert(0, "/vocab_rl")

    from sweep import SweepConfig, _train_and_eval

    cfg = SweepConfig(**config_dict)
    result = _train_and_eval(cfg, data_path_in_container)
    return {
        "config": config_dict,
        "eval_acquired_mean": result.eval_acquired_mean,
        "eval_acquired_std": result.eval_acquired_std,
        "eval_post_gap_mean": result.eval_post_gap_mean,
        "eval_post_gap_std": result.eval_post_gap_std,
        "train_seconds": result.train_seconds,
        "eval_seconds": result.eval_seconds,
    }


@app.local_entrypoint()
def main(
    algos: str = "iql,cql",
    tau: str = "0.7,0.9",
    beta: str = "1.0,3.0,10.0",
    alpha_cql: str = "0.1,1.0,10.0",
    seeds: str = "0,1,2",
    n_steps: int = 50_000,
    out: str = "modal_sweep_results.json",
):
    import itertools

    algos_list = algos.split(",")
    tau_list = [float(x) for x in tau.split(",")]
    beta_list = [float(x) for x in beta.split(",")]
    alpha_list = [float(x) for x in alpha_cql.split(",")]
    seeds_list = [int(x) for x in seeds.split(",")]

    configs = []
    for algo, t, b, s in itertools.product(algos_list, tau_list, beta_list, seeds_list):
        if algo == "cql":
            for a in alpha_list:
                configs.append(dict(algo=algo, tau=t, beta=b, alpha_cql=a,
                                    seed=s, n_steps=n_steps))
        else:
            configs.append(dict(algo=algo, tau=t, beta=b, alpha_cql=0.0,
                                seed=s, n_steps=n_steps))

    print(f"Dispatching {len(configs)} configs to Modal...")
    args_for_map = [(cfg, "/vocab_rl/dataset.npz") for cfg in configs]
    results = list(train_and_eval_remote.starmap(args_for_map))
    print(f"Got {len(results)} results back.")

    out_path = out if os.path.isabs(out) else os.path.join(HERE, out)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved -> {out_path}")

    results.sort(key=lambda r: r["eval_post_gap_mean"], reverse=True)
    print("\nTop 5 configs by post-gap retention:")
    for r in results[:5]:
        c = r["config"]
        print(f"  {c['algo']:3s}  tau={c['tau']}  beta={c['beta']}  "
              f"alpha={c['alpha_cql']}  seed={c['seed']}  "
              f"acquired={r['eval_acquired_mean']:.2f}  "
              f"post_gap={r['eval_post_gap_mean']:.2f}")
