"""Train a CQL policy on the same offline dataset as IQL."""

from __future__ import annotations

import argparse
import os

from cql import CQLConfig, train_cql
from env import NUM_ACTIONS, VocabEnv
from iql import OfflineBuffer


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="dataset.npz")
    p.add_argument("--out", default="cql_policy.pt")
    p.add_argument("--steps", type=int, default=100_000)
    p.add_argument("--alpha", type=float, default=1.0,
                   help="CQL conservative penalty weight")
    p.add_argument("--tau", type=float, default=0.7)
    p.add_argument("--beta", type=float, default=3.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--reward-scale", type=float, default=1.0,
                   help="Multiplier on rewards (try 100 at 500 words).")
    args = p.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    data_path = args.data if os.path.isabs(args.data) else os.path.join(here, args.data)
    out_path  = args.out  if os.path.isabs(args.out)  else os.path.join(here, args.out)

    print(f"Loading dataset from {data_path} (reward_scale={args.reward_scale})")
    buf = OfflineBuffer.from_npz(data_path, reward_scale=args.reward_scale)
    state_dim = buf.states.shape[1]
    env = VocabEnv()
    assert env.state_dim == state_dim

    cfg = CQLConfig(alpha_cql=args.alpha, tau_expectile=args.tau, beta=args.beta)
    print(f"Training CQL for {args.steps} steps "
          f"(alpha={cfg.alpha_cql}, tau={cfg.tau_expectile}, beta={cfg.beta})")
    trainer = train_cql(buf, state_dim=state_dim, num_actions=NUM_ACTIONS,
                        n_steps=args.steps, cfg=cfg, seed=args.seed,
                        device=args.device)
    trainer.save(out_path)
    print(f"\nsaved policy to {out_path}")


if __name__ == "__main__":
    main()
