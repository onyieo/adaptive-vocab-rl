"""Conservative Q-Learning (discrete-action variant) for VocabEnv.

Reference: Kumar, Zhou, Tucker, Levine. "Conservative Q-Learning for Offline
Reinforcement Learning." NeurIPS 2020.

For discrete actions the conservative penalty has a clean closed form:

  L_CQL = alpha * E_s[ logsumexp_a Q(s, a)  -  Q(s, a_dataset) ]

Intuition: the first term pushes Q values DOWN for all actions at state s;
the second term pulls Q values UP for the dataset action. Net effect: Q(s, a)
is pushed below Q(s, a_data) for out-of-distribution actions a. This prevents
the policy-extraction step from being seduced by inflated Q estimates on
actions never seen in the buffer.

We reuse the IQL value function and AWR policy extraction; only the Q loss
changes. (Pure CQL would also use Bellman backups via max over actions; we
use the IQL-style V-based backup here so the only difference vs. IQLTrainer
is the conservative penalty.)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from iql import IQLConfig, IQLTrainer


@dataclass
class CQLConfig(IQLConfig):
    alpha_cql: float = 1.0   # weight on the conservative penalty


class CQLTrainer(IQLTrainer):
    cfg: CQLConfig

    def update(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        s, a, r, s2, d = batch["s"], batch["a"], batch["r"], batch["s2"], batch["d"]

        # --- V update (identical to IQL) ----------------------------------
        from iql import expectile_loss
        with torch.no_grad():
            q_t = self._q_at(self.q_target, s, a)
        v = self.v(s)
        v_loss = expectile_loss(q_t - v, self.cfg.tau_expectile)
        self.v_opt.zero_grad(); v_loss.backward(); self.v_opt.step()

        # --- Q update with CQL conservative penalty -----------------------
        with torch.no_grad():
            v_next = self.v(s2)
            target = r + self.cfg.gamma * (1.0 - d) * v_next
        q_all = self.q(s)                                  # (B, A)
        q_sa = q_all.gather(1, a.unsqueeze(-1)).squeeze(-1)
        bellman_loss = F.mse_loss(q_sa, target)
        # Conservative term: pull down Q on all actions, push up on data action.
        cql_penalty = (torch.logsumexp(q_all, dim=-1) - q_sa).mean()
        q_loss = bellman_loss + self.cfg.alpha_cql * cql_penalty
        self.q_opt.zero_grad(); q_loss.backward(); self.q_opt.step()

        # --- pi update (AWR, identical to IQL) ----------------------------
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
            "bellman_loss": float(bellman_loss.item()),
            "cql_penalty": float(cql_penalty.item()),
            "pi_loss": float(pi_loss.item()),
            "adv_mean": float(adv.mean().item()),
        }


def train_cql(buffer, state_dim: int, num_actions: int,
              n_steps: int = 100_000, log_every: int = 5_000,
              cfg: CQLConfig | None = None, seed: int = 0,
              device: str | None = None) -> CQLTrainer:
    import numpy as np
    cfg = cfg or CQLConfig()
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    trainer = CQLTrainer(state_dim=state_dim, num_actions=num_actions,
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
                  f"v={mean['v_loss']:.4f}  bell={mean['bellman_loss']:.4f}  "
                  f"cql={mean['cql_penalty']:.3f}  pi={mean['pi_loss']:.4f}")
    return trainer
