"""Implicit Q-Learning (IQL) for the discrete VocabEnv action space.

Reference: Kostrikov, Nair, Levine. "Offline Reinforcement Learning with
Implicit Q-Learning." ICLR 2022.

Three networks (all MLPs):
  V(s)    -> scalar
  Q(s)    -> (num_actions,)    [used as Q(s, a) via gather]
  pi(s)   -> (num_actions,)    [logits over discrete actions]
Plus a target Q network (Polyak-averaged copy of Q).

Three losses, optimized jointly each step:
  L_V  = expectile_loss( Q_target(s,a) - V(s) ; tau )
  L_Q  = MSE( Q(s,a) , r + gamma (1-d) V(s') )
  L_pi = - E[ exp(beta * (Q_target(s,a) - V(s))) * log pi(a|s) ]   (AWR)

The key idea: V is regressed on Q with an upper-expectile loss, so it
approximates max_a Q(s,a) WITHOUT ever evaluating Q at out-of-distribution
actions. The policy is then extracted by advantage-weighted regression
against the dataset's actions, which is also OOD-action-free.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------

def _mlp(in_dim: int, out_dim: int, hidden: tuple[int, ...] = (256, 256)) -> nn.Sequential:
    layers: list[nn.Module] = []
    last = in_dim
    for h in hidden:
        layers += [nn.Linear(last, h), nn.ReLU()]
        last = h
    layers.append(nn.Linear(last, out_dim))
    return nn.Sequential(*layers)


class VNet(nn.Module):
    def __init__(self, state_dim: int, hidden=(256, 256)) -> None:
        super().__init__()
        self.net = _mlp(state_dim, 1, hidden)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.net(s).squeeze(-1)


class QNet(nn.Module):
    """Q(s) -> (B, num_actions). Q(s, a) obtained via gather."""

    def __init__(self, state_dim: int, num_actions: int, hidden=(256, 256)) -> None:
        super().__init__()
        self.net = _mlp(state_dim, num_actions, hidden)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.net(s)


class PolicyNet(nn.Module):
    def __init__(self, state_dim: int, num_actions: int, hidden=(256, 256)) -> None:
        super().__init__()
        self.net = _mlp(state_dim, num_actions, hidden)

    def logits(self, s: torch.Tensor) -> torch.Tensor:
        return self.net(s)

    def log_prob(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return F.log_softmax(self.logits(s), dim=-1).gather(1, a.unsqueeze(-1)).squeeze(-1)

    @torch.no_grad()
    def act(self, s: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        logits = self.logits(s)
        if deterministic:
            return logits.argmax(dim=-1)
        return torch.distributions.Categorical(logits=logits).sample()


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

def expectile_loss(diff: torch.Tensor, tau: float) -> torch.Tensor:
    """L^tau(u) = |tau - 1(u<0)| * u^2.

    diff = target - V(s). Penalizes under-prediction (diff > 0) more heavily
    when tau > 0.5, pushing V toward an upper expectile of the Q distribution.
    """
    weight = torch.where(diff > 0, tau, 1.0 - tau)
    return (weight * diff.pow(2)).mean()


# ---------------------------------------------------------------------------
# Replay buffer / dataset
# ---------------------------------------------------------------------------

@dataclass
class OfflineBuffer:
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    dones: np.ndarray

    @classmethod
    def from_npz(cls, path: str) -> "OfflineBuffer":
        d = np.load(path, allow_pickle=True)
        return cls(
            states=d["states"].astype(np.float32),
            actions=d["actions"].astype(np.int64),
            rewards=d["rewards"].astype(np.float32),
            next_states=d["next_states"].astype(np.float32),
            dones=d["dones"].astype(bool),
        )

    def __len__(self) -> int:
        return len(self.actions)

    def sample(self, batch_size: int, rng: np.random.Generator,
               device: torch.device) -> dict[str, torch.Tensor]:
        idx = rng.integers(0, len(self), size=batch_size)
        return {
            "s":  torch.from_numpy(self.states[idx]).to(device),
            "a":  torch.from_numpy(self.actions[idx]).to(device),
            "r":  torch.from_numpy(self.rewards[idx]).to(device),
            "s2": torch.from_numpy(self.next_states[idx]).to(device),
            "d":  torch.from_numpy(self.dones[idx].astype(np.float32)).to(device),
        }


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

@dataclass
class IQLConfig:
    gamma: float = 0.99
    tau_expectile: float = 0.7      # upper expectile for V regression
    beta: float = 3.0               # AWR temperature
    adv_clip: float = 100.0         # clip exp(beta*A) to avoid blowup
    polyak: float = 0.005           # target Q EMA rate
    lr: float = 3e-4
    batch_size: int = 256
    hidden: tuple[int, ...] = (256, 256)


@dataclass
class IQLTrainer:
    state_dim: int
    num_actions: int
    cfg: IQLConfig = field(default_factory=IQLConfig)
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))

    def __post_init__(self) -> None:
        h = self.cfg.hidden
        self.v = VNet(self.state_dim, h).to(self.device)
        self.q = QNet(self.state_dim, self.num_actions, h).to(self.device)
        self.q_target = QNet(self.state_dim, self.num_actions, h).to(self.device)
        self.q_target.load_state_dict(self.q.state_dict())
        for p in self.q_target.parameters():
            p.requires_grad_(False)
        self.pi = PolicyNet(self.state_dim, self.num_actions, h).to(self.device)

        self.v_opt  = torch.optim.Adam(self.v.parameters(),  lr=self.cfg.lr)
        self.q_opt  = torch.optim.Adam(self.q.parameters(),  lr=self.cfg.lr)
        self.pi_opt = torch.optim.Adam(self.pi.parameters(), lr=self.cfg.lr)

    # ---- update steps -----------------------------------------------------

    def _q_at(self, q_net: nn.Module, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return q_net(s).gather(1, a.unsqueeze(-1)).squeeze(-1)

    def update(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        s, a, r, s2, d = batch["s"], batch["a"], batch["r"], batch["s2"], batch["d"]

        # --- V update: expectile regression on Q_target(s, a) - V(s) -------
        with torch.no_grad():
            q_t = self._q_at(self.q_target, s, a)
        v = self.v(s)
        v_loss = expectile_loss(q_t - v, self.cfg.tau_expectile)
        self.v_opt.zero_grad(); v_loss.backward(); self.v_opt.step()

        # --- Q update: MSE to r + gamma (1-d) V(s') -----------------------
        with torch.no_grad():
            v_next = self.v(s2)
            target = r + self.cfg.gamma * (1.0 - d) * v_next
        q_sa = self._q_at(self.q, s, a)
        q_loss = F.mse_loss(q_sa, target)
        self.q_opt.zero_grad(); q_loss.backward(); self.q_opt.step()

        # --- pi update: AWR with weights from Q_target - V -----------------
        with torch.no_grad():
            adv = self._q_at(self.q_target, s, a) - self.v(s)
            w = torch.exp(self.cfg.beta * adv).clamp(max=self.cfg.adv_clip)
        log_pi = self.pi.log_prob(s, a)
        pi_loss = -(w * log_pi).mean()
        self.pi_opt.zero_grad(); pi_loss.backward(); self.pi_opt.step()

        # --- Polyak update of target Q ------------------------------------
        with torch.no_grad():
            for p, pt in zip(self.q.parameters(), self.q_target.parameters()):
                pt.data.mul_(1.0 - self.cfg.polyak).add_(self.cfg.polyak * p.data)

        return {
            "v_loss": float(v_loss.item()),
            "q_loss": float(q_loss.item()),
            "pi_loss": float(pi_loss.item()),
            "adv_mean": float(adv.mean().item()),
            "weight_mean": float(w.mean().item()),
        }

    # ---- inference --------------------------------------------------------

    @torch.no_grad()
    def act(self, state: np.ndarray, deterministic: bool = True) -> int:
        s = torch.from_numpy(state.astype(np.float32)).unsqueeze(0).to(self.device)
        a = self.pi.act(s, deterministic=deterministic)
        return int(a.item())

    # ---- save/load --------------------------------------------------------

    def save(self, path: str) -> None:
        torch.save({
            "v": self.v.state_dict(),
            "q": self.q.state_dict(),
            "q_target": self.q_target.state_dict(),
            "pi": self.pi.state_dict(),
            "cfg": self.cfg,
            "state_dim": self.state_dim,
            "num_actions": self.num_actions,
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.v.load_state_dict(ckpt["v"])
        self.q.load_state_dict(ckpt["q"])
        self.q_target.load_state_dict(ckpt["q_target"])
        self.pi.load_state_dict(ckpt["pi"])


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(buffer: OfflineBuffer, state_dim: int, num_actions: int,
          n_steps: int = 100_000, log_every: int = 5_000,
          cfg: Optional[IQLConfig] = None, seed: int = 0,
          device: Optional[str] = None) -> IQLTrainer:
    cfg = cfg or IQLConfig()
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    trainer = IQLTrainer(state_dim=state_dim, num_actions=num_actions,
                         cfg=cfg, device=dev)
    rng = np.random.default_rng(seed)

    logs: list[dict[str, float]] = []
    for step in range(1, n_steps + 1):
        batch = buffer.sample(cfg.batch_size, rng, dev)
        info = trainer.update(batch)
        logs.append(info)
        if step % log_every == 0:
            recent = logs[-log_every:]
            mean = {k: float(np.mean([d[k] for d in recent])) for k in recent[0]}
            print(f"  step {step:6d} | "
                  f"v={mean['v_loss']:.4f}  q={mean['q_loss']:.4f}  "
                  f"pi={mean['pi_loss']:.4f}  "
                  f"adv={mean['adv_mean']:+.3f}  w={mean['weight_mean']:.2f}")
    return trainer
