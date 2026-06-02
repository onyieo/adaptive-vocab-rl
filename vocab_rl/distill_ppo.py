"""PPO-distillation collector.

Samples N trajectories from a trained PPO policy and appends them to an
existing offline dataset. Yields a higher-quality buffer for the subsequent
offline RL retrain.

This is a deliberately mild form of "knowledge transfer" from online to
offline RL — we're not using PPO as a teacher to imitate, we're augmenting
the offline buffer with PPO's behavior. CQL and IQL then learn from the
combined data. The hope: more high-quality (high-return) trajectories pull
the offline policy toward PPO's region of the action space.

Usage:
    python distill_ppo.py --ppo ppo_policy.pt --in dataset.npz \\
                          --out dataset_with_ppo.npz --n 100
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from env import VocabEnv
from ppo import load as load_ppo, act_deterministic, PPOConfig  # noqa: F401


def collect_ppo_trajectories(ppo_path: str, n_traj: int,
                             seed_start: int = 9000,
                             multi_sim: bool = True) -> dict:
    from collect_dataset import TRAINING_SIM_VARIANTS
    model = load_ppo(ppo_path)
    device = torch.device("cpu")

    env_probe = VocabEnv()
    state_dim = env_probe.state_dim
    steps_per_traj = env_probe.total_turns
    total = n_traj * steps_per_traj

    states = np.zeros((total, state_dim), dtype=np.float32)
    actions = np.zeros(total, dtype=np.int64)
    rewards = np.zeros(total, dtype=np.float32)
    next_states = np.zeros((total, state_dim), dtype=np.float32)
    dones = np.zeros(total, dtype=bool)

    variants = TRAINING_SIM_VARIANTS if multi_sim else [{}]
    idx = 0
    for k in range(n_traj):
        kwargs = variants[k % len(variants)] if multi_sim else {}
        env = VocabEnv(seed=seed_start + k, sim_kwargs=kwargs)
        s = env.reset()
        for _ in range(steps_per_traj):
            a = act_deterministic(model, s, device)
            s2, r, done, _info = env.step(int(a))
            states[idx] = s
            actions[idx] = a
            rewards[idx] = r
            next_states[idx] = s2
            dones[idx] = done
            idx += 1
            s = s2
        if (k + 1) % 20 == 0:
            print(f"  {k + 1}/{n_traj} PPO trajectories collected")

    return {
        "states": states, "actions": actions, "rewards": rewards,
        "next_states": next_states, "dones": dones,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ppo", default="ppo_policy.pt")
    p.add_argument("--in", dest="in_path", default="dataset.npz")
    p.add_argument("--out", default="dataset_with_ppo.npz")
    p.add_argument("--n", type=int, default=100,
                   help="number of PPO trajectories to append")
    p.add_argument("--seed-start", type=int, default=9000)
    p.add_argument("--no-multi-sim", action="store_true")
    args = p.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    ppo_path = args.ppo if os.path.isabs(args.ppo) else os.path.join(here, args.ppo)
    in_path = args.in_path if os.path.isabs(args.in_path) else os.path.join(here, args.in_path)
    out_path = args.out if os.path.isabs(args.out) else os.path.join(here, args.out)

    print(f"Loading existing buffer from {in_path}")
    existing = dict(np.load(in_path, allow_pickle=True))
    print(f"  existing size: {len(existing['actions'])} transitions")

    print(f"Collecting {args.n} PPO trajectories from {ppo_path}")
    new = collect_ppo_trajectories(ppo_path, args.n,
                                   seed_start=args.seed_start,
                                   multi_sim=not args.no_multi_sim)

    merged = {}
    for k in ("states", "actions", "rewards", "next_states", "dones"):
        merged[k] = np.concatenate([existing[k], new[k]], axis=0)
    # Mark PPO trajectories with policy_id = 99 (separate from baselines).
    n_existing = len(existing["actions"])
    n_new = len(new["actions"])
    if "policy_id" in existing:
        new_pid = np.full(n_new, 99, dtype=np.int8)
        merged["policy_id"] = np.concatenate(
            [existing["policy_id"], new_pid], axis=0)
    if "sim_variant" in existing:
        # PPO traj sim_variant cycles same as new; rebuild
        sv = np.array([(i % 5) for i in range(n_new)], dtype=np.int8)
        merged["sim_variant"] = np.concatenate(
            [existing["sim_variant"], sv], axis=0)
    if "policy_names" in existing:
        names = list(existing["policy_names"])
        if "PPODistilled" not in names:
            names.append("PPODistilled")
        merged["policy_names"] = np.array(names)
    if "variants" in existing:
        merged["variants"] = existing["variants"]

    np.savez_compressed(out_path, **merged)
    print(f"\nsaved merged buffer ({len(merged['actions'])} transitions, "
          f"{os.path.getsize(out_path)/1e6:.1f} MB) -> {out_path}")


if __name__ == "__main__":
    main()
