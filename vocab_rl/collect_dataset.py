"""Roll out behavior policies in VocabEnv to build an offline dataset.

Saves a single .npz file with arrays:
  states      (N, state_dim)   float32
  actions     (N,)              int64
  rewards     (N,)              float32
  next_states (N, state_dim)   float32
  dones       (N,)              bool
  policy_id   (N,)              int8   (which behavior policy produced this transition)

A "transition" is a single env step. With 200 turns per trajectory, K trajectories
per policy, and P policies, N = K * 200 * P.
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from baseline_actions import ALL_BEHAVIOR_POLICIES
from env import VocabEnv


# Training-time simulator parameterizations. The default is included so the
# mixed buffer always contains some "canonical" trajectories; the rest are
# perturbations of forgetting rate and difficulty distribution. We deliberately
# hold a "harder_words" + "fast_forget" combination OUT of training so the
# robustness eval has a true out-of-distribution test.
TRAINING_SIM_VARIANTS: list[dict] = [
    {},  # default: forgetting=9.0, difficulty_scale=1.0
    {"forgetting_constant": 7.0},
    {"forgetting_constant": 12.0},
    {"difficulty_scale": 0.85},
    {"difficulty_scale": 1.15},
]


def collect(n_traj_per_policy: int, out_path: str, seed_start: int = 0,
            multi_sim: bool = False) -> None:
    policies = [cls() for cls in ALL_BEHAVIOR_POLICIES]
    n_policies = len(policies)

    env = VocabEnv()
    state_dim = env.state_dim
    steps_per_traj = env.total_turns
    total = n_policies * n_traj_per_policy * steps_per_traj

    states = np.zeros((total, state_dim), dtype=np.float32)
    actions = np.zeros(total, dtype=np.int64)
    rewards = np.zeros(total, dtype=np.float32)
    next_states = np.zeros((total, state_dim), dtype=np.float32)
    dones = np.zeros(total, dtype=bool)
    policy_id = np.zeros(total, dtype=np.int8)
    sim_variant = np.zeros(total, dtype=np.int8)

    variants = TRAINING_SIM_VARIANTS if multi_sim else [{}]
    if multi_sim:
        print(f"Multi-sim training: cycling through {len(variants)} variants "
              f"(default, fast/slow forgetting, harder/easier words). "
              f"Held-out test variants in robustness_eval.")

    idx = 0
    seed = seed_start
    for pid, policy in enumerate(policies):
        for k in range(n_traj_per_policy):
            variant_idx = k % len(variants) if multi_sim else 0
            kwargs = variants[variant_idx]
            env = VocabEnv(seed=seed, sim_kwargs=kwargs)
            rng = np.random.default_rng(seed + 1_000_000)
            s = env.reset()
            for _ in range(steps_per_traj):
                a = policy(s, rng)
                s2, r, done, _info = env.step(int(a))
                states[idx] = s
                actions[idx] = a
                rewards[idx] = r
                next_states[idx] = s2
                dones[idx] = done
                policy_id[idx] = pid
                sim_variant[idx] = variant_idx
                idx += 1
                s = s2
                if done:
                    break
            seed += 1
        print(f"  {policy.name}: {n_traj_per_policy} trajectories collected "
              f"(running total: {idx} transitions)")

    assert idx == total, f"expected {total} transitions, got {idx}"

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez_compressed(
        out_path,
        states=states, actions=actions, rewards=rewards,
        next_states=next_states, dones=dones, policy_id=policy_id,
        sim_variant=sim_variant,
        policy_names=np.array([p.name for p in policies]),
        variants=np.array([str(v) for v in variants]),
    )
    print(f"\nsaved {total} transitions to {out_path} "
          f"({os.path.getsize(out_path) / 1e6:.1f} MB)")
    print(f"  per-policy reward (terminal, mean): "
          f"{[float(rewards[(policy_id == p) & dones].mean()) if dones[policy_id == p].any() else 0.0 for p in range(n_policies)]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200,
                        help="trajectories per behavior policy")
    parser.add_argument("--out", default="dataset.npz")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--multi-sim", action="store_true",
                        help="Cycle through multiple simulator parameterizations "
                             "during training-data collection (Phase A robustness goal).")
    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    out = args.out if os.path.isabs(args.out) else os.path.join(here, args.out)
    print(f"Collecting {args.n} trajectories x {len(ALL_BEHAVIOR_POLICIES)} "
          f"policies (seed_start={args.seed}, multi_sim={args.multi_sim})")
    collect(args.n, out, seed_start=args.seed, multi_sim=args.multi_sim)


if __name__ == "__main__":
    main()
