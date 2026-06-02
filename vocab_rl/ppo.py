"""PPO online baseline.

Trains a discrete-action policy by interacting with VocabEnv online, in
contrast to the offline IQL/CQL methods. Provides the methodological
counterfactual the proposal called for: 'is offline RL trained on
mixed-quality trajectories at least as good as online RL trained from
scratch in the simulator?'

Reference: Schulman et al. 2017, "Proximal Policy Optimization Algorithms".

Single-environment implementation (sufficient for our scale). Policy and
value networks share an MLP trunk; clipped surrogate objective; GAE for
advantages.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from env import NUM_ACTIONS, VocabEnv


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------

class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, num_actions: int,
                 hidden: tuple[int, ...] = (256, 256)) -> None:
        super().__init__()
        layers = []
        last = state_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.Tanh()]
            last = h
        self.trunk = nn.Sequential(*layers)
        self.policy_head = nn.Linear(last, num_actions)
        self.value_head = nn.Linear(last, 1)

    def forward(self, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.trunk(s)
        return self.policy_head(z), self.value_head(z).squeeze(-1)

    @torch.no_grad()
    def act(self, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, v = self(s)
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
        return a, dist.log_prob(a), v


# ---------------------------------------------------------------------------
# Rollout buffer + GAE
# ---------------------------------------------------------------------------

@dataclass
class RolloutBatch:
    states: torch.Tensor       # (T, state_dim)
    actions: torch.Tensor      # (T,)
    log_probs: torch.Tensor    # (T,)
    values: torch.Tensor       # (T,)
    rewards: torch.Tensor      # (T,)
    dones: torch.Tensor        # (T,)
    advantages: torch.Tensor   # (T,)
    returns: torch.Tensor      # (T,)


def collect_rollout(env: VocabEnv, model: ActorCritic, n_steps: int,
                    device: torch.device,
                    gamma: float, gae_lambda: float) -> RolloutBatch:
    states, actions, log_probs, values, rewards, dones = [], [], [], [], [], []
    s = env.reset()
    s_t = torch.from_numpy(s.astype(np.float32)).to(device)
    for _ in range(n_steps):
        with torch.no_grad():
            a, lp, v = model.act(s_t.unsqueeze(0))
        a_int = int(a.item())
        s_next, r, done, _info = env.step(a_int)

        states.append(s_t)
        actions.append(a)
        log_probs.append(lp)
        values.append(v)
        rewards.append(r)
        dones.append(float(done))

        if done:
            s_next = env.reset()
        s_t = torch.from_numpy(s_next.astype(np.float32)).to(device)

    # Bootstrap value for the last state.
    with torch.no_grad():
        _, last_v = model(s_t.unsqueeze(0))
    last_v = last_v.squeeze().item()

    # GAE
    advantages = np.zeros(n_steps, dtype=np.float32)
    gae = 0.0
    vals = [v.item() for v in values] + [last_v]
    for t in reversed(range(n_steps)):
        not_done = 1.0 - dones[t]
        delta = rewards[t] + gamma * vals[t + 1] * not_done - vals[t]
        gae = delta + gamma * gae_lambda * not_done * gae
        advantages[t] = gae
    returns = advantages + np.array(vals[:-1], dtype=np.float32)

    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return RolloutBatch(
        states=torch.stack(states),
        actions=torch.stack(actions).squeeze(-1),
        log_probs=torch.stack(log_probs).squeeze(-1),
        values=torch.stack(values),
        rewards=torch.tensor(rewards, dtype=torch.float32, device=device),
        dones=torch.tensor(dones, dtype=torch.float32, device=device),
        advantages=torch.from_numpy(advantages).to(device),
        returns=torch.from_numpy(returns.astype(np.float32)).to(device),
    )


# ---------------------------------------------------------------------------
# PPO update
# ---------------------------------------------------------------------------

@dataclass
class PPOConfig:
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    n_epochs: int = 8
    minibatch_size: int = 256
    rollout_steps: int = 2000
    total_env_steps: int = 200_000
    hidden: tuple[int, ...] = (256, 256)


def ppo_update(model: ActorCritic, optimizer: torch.optim.Optimizer,
               batch: RolloutBatch, cfg: PPOConfig) -> dict[str, float]:
    states = batch.states
    actions = batch.actions
    old_log_probs = batch.log_probs.detach()
    returns = batch.returns
    advantages = batch.advantages

    n = states.size(0)
    indices = torch.arange(n, device=states.device)

    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0
    n_minibatches = 0

    for _ in range(cfg.n_epochs):
        idx = indices[torch.randperm(n, device=states.device)]
        for start in range(0, n, cfg.minibatch_size):
            mb = idx[start:start + cfg.minibatch_size]
            logits, values = model(states[mb])
            dist = torch.distributions.Categorical(logits=logits)
            log_probs = dist.log_prob(actions[mb])
            entropy = dist.entropy().mean()

            ratio = torch.exp(log_probs - old_log_probs[mb])
            surr1 = ratio * advantages[mb]
            surr2 = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * advantages[mb]
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(values, returns[mb])

            loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()
            n_minibatches += 1

    return {
        "policy_loss": total_policy_loss / max(1, n_minibatches),
        "value_loss": total_value_loss / max(1, n_minibatches),
        "entropy": total_entropy / max(1, n_minibatches),
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(cfg: PPOConfig, seed: int = 0,
          device: torch.device | None = None) -> ActorCritic:
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    env = VocabEnv(seed=seed)
    model = ActorCritic(env.state_dim, NUM_ACTIONS, hidden=cfg.hidden).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    n_updates = cfg.total_env_steps // cfg.rollout_steps
    episode_returns: list[float] = []
    rolling_return = 0.0

    for u in range(1, n_updates + 1):
        batch = collect_rollout(env, model, cfg.rollout_steps, dev,
                                cfg.gamma, cfg.gae_lambda)
        info = ppo_update(model, opt, batch, cfg)

        # Episodic return: sum rewards between done flags.
        rewards = batch.rewards.cpu().numpy()
        dones = batch.dones.cpu().numpy()
        ep_returns = []
        run = 0.0
        for r, d in zip(rewards, dones):
            run += float(r)
            if d:
                ep_returns.append(run); run = 0.0
        if ep_returns:
            rolling_return = (0.9 * rolling_return + 0.1 * np.mean(ep_returns)
                              if episode_returns else np.mean(ep_returns))
            episode_returns.extend(ep_returns)

        if u % 5 == 0 or u == 1:
            print(f"  update {u:3d} ({u * cfg.rollout_steps} env steps): "
                  f"policy={info['policy_loss']:+.4f}  value={info['value_loss']:.4f}  "
                  f"entropy={info['entropy']:.2f}  ep_return={rolling_return:.3f}")

    return model


@torch.no_grad()
def act_deterministic(model: ActorCritic, state: np.ndarray,
                      device: torch.device) -> int:
    s = torch.from_numpy(state.astype(np.float32)).unsqueeze(0).to(device)
    logits, _ = model(s)
    return int(logits.argmax(dim=-1).item())


def save(model: ActorCritic, path: str, cfg: PPOConfig,
         state_dim: int, num_actions: int) -> None:
    # Save cfg as a plain dict so reloading doesn't depend on the PPOConfig
    # class being available in the loading module's __main__ namespace.
    torch.save({"model": model.state_dict(),
                "hidden": list(cfg.hidden),
                "state_dim": state_dim,
                "num_actions": num_actions}, path)


def load(path: str, device: torch.device | None = None) -> ActorCritic:
    dev = device or torch.device("cpu")
    ckpt = torch.load(path, map_location=dev, weights_only=False)
    hidden = tuple(ckpt.get("hidden", (256, 256)))
    model = ActorCritic(ckpt["state_dim"], ckpt["num_actions"],
                        hidden=hidden).to(dev)
    model.load_state_dict(ckpt["model"])
    return model


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="ppo_policy.pt")
    p.add_argument("--total-steps", type=int, default=200_000)
    p.add_argument("--rollout-steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--minibatch", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    cfg = PPOConfig(
        lr=args.lr, clip_eps=args.clip, n_epochs=args.epochs,
        minibatch_size=args.minibatch, rollout_steps=args.rollout_steps,
        total_env_steps=args.total_steps,
    )

    here = os.path.dirname(os.path.abspath(__file__))
    out = args.out if os.path.isabs(args.out) else os.path.join(here, args.out)

    print(f"Training PPO for {args.total_steps} env steps "
          f"(seed={args.seed}, device={args.device or 'auto'})")
    model = train(cfg, seed=args.seed,
                  device=torch.device(args.device) if args.device else None)

    env_probe = VocabEnv()
    save(model, out, cfg,
         state_dim=env_probe.state_dim, num_actions=NUM_ACTIONS)
    print(f"\nsaved PPO policy -> {out}")


if __name__ == "__main__":
    main()
