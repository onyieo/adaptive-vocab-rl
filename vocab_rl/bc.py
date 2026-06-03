"""Behavior cloning of the PPO policy.

A key ablation: if a simple supervised model trained on PPO's actions matches
PPO's performance, then our offline RL methods (IQL/CQL) aren't really doing
anything an MLP imitating PPO couldn't do.

Workflow:
1. Roll out PPO for N trajectories on the v3-reward env.
2. Train an MLP with cross-entropy on (state, action) pairs.
3. Evaluate.

If BC ≈ PPO: offline RL didn't add much beyond imitation.
If BC << PPO: PPO has some non-trivial structure that BC can't capture.
If BC ≈ IQL/CQL > PPO: imitation is the bottleneck — RL methods at this
                       scale of buffer are essentially BC anyway.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from env import NUM_ACTIONS, VocabEnv
from ppo import load as load_ppo, act_deterministic, PPOConfig  # noqa: F401


def collect_ppo_trajectories(ppo_path: str, n_traj: int,
                             seed_start: int = 20000) -> dict:
    from collect_dataset import TRAINING_SIM_VARIANTS
    model = load_ppo(ppo_path)
    device = torch.device("cpu")

    env_probe = VocabEnv()
    state_dim = env_probe.state_dim
    steps_per_traj = env_probe.total_turns
    total = n_traj * steps_per_traj

    states = np.zeros((total, state_dim), dtype=np.float32)
    actions = np.zeros(total, dtype=np.int64)

    idx = 0
    for k in range(n_traj):
        kwargs = TRAINING_SIM_VARIANTS[k % len(TRAINING_SIM_VARIANTS)]
        env = VocabEnv(seed=seed_start + k, sim_kwargs=kwargs)
        s = env.reset()
        for _ in range(steps_per_traj):
            a = act_deterministic(model, s, device)
            states[idx] = s
            actions[idx] = a
            s, _r, _d, _info = env.step(int(a))
            idx += 1

    return {"states": states[:idx], "actions": actions[:idx]}


class BCPolicy(nn.Module):
    def __init__(self, state_dim: int, num_actions: int,
                 hidden: tuple[int, ...] = (256, 256)) -> None:
        super().__init__()
        layers = []
        last = state_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        layers.append(nn.Linear(last, num_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, s):
        return self.net(s)


def train_bc(data: dict, n_steps: int = 30_000, batch_size: int = 256,
             lr: float = 3e-4, seed: int = 0) -> BCPolicy:
    torch.manual_seed(seed)
    states = torch.from_numpy(data["states"])
    actions = torch.from_numpy(data["actions"])
    device = torch.device("cpu")
    model = BCPolicy(states.shape[1], NUM_ACTIONS).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    n = len(states)
    for step in range(1, n_steps + 1):
        idx = rng.integers(0, n, size=batch_size)
        s = states[idx].to(device)
        a = actions[idx].to(device)
        logits = model(s)
        loss = F.cross_entropy(logits, a)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 5000 == 0:
            print(f"  step {step:5d}  ce_loss={loss.item():.4f}")
    return model


@torch.no_grad()
def act_bc(model: BCPolicy, state: np.ndarray) -> int:
    s = torch.from_numpy(state.astype(np.float32)).unsqueeze(0)
    return int(model(s).argmax(dim=-1).item())


def save(model: BCPolicy, path: str, state_dim: int, num_actions: int) -> None:
    torch.save({"model": model.state_dict(),
                "state_dim": state_dim,
                "num_actions": num_actions}, path)


def load(path: str) -> BCPolicy:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = BCPolicy(ckpt["state_dim"], ckpt["num_actions"])
    model.load_state_dict(ckpt["model"])
    return model


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ppo", default="ppo_policy.pt")
    p.add_argument("--out", default="bc_policy.pt")
    p.add_argument("--n-traj", type=int, default=200,
                   help="PPO trajectories to collect for training")
    p.add_argument("--n-steps", type=int, default=30_000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    ppo_path = args.ppo if os.path.isabs(args.ppo) else os.path.join(here, args.ppo)
    out_path = args.out if os.path.isabs(args.out) else os.path.join(here, args.out)

    print(f"Collecting {args.n_traj} PPO trajectories from {ppo_path}...")
    data = collect_ppo_trajectories(ppo_path, args.n_traj)
    print(f"  collected {len(data['actions'])} (s, a) pairs")

    print(f"Training BC for {args.n_steps} steps...")
    model = train_bc(data, n_steps=args.n_steps, seed=args.seed)

    env_probe = VocabEnv()
    save(model, out_path, state_dim=env_probe.state_dim, num_actions=NUM_ACTIONS)
    print(f"\nsaved BC policy -> {out_path}")


if __name__ == "__main__":
    main()
