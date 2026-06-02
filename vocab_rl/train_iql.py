"""Train an IQL policy on an offline dataset of VocabEnv trajectories."""

from __future__ import annotations

import argparse
import os

from env import NUM_ACTIONS, VocabEnv
from iql import IQLConfig, OfflineBuffer, train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="dataset.npz")
    parser.add_argument("--out", default="iql_policy.pt")
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--tau", type=float, default=0.7,
                        help="expectile for V regression")
    parser.add_argument("--beta", type=float, default=3.0,
                        help="AWR temperature")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None,
                        help="cpu / cuda / mps (default: auto)")
    parser.add_argument("--reward-scale", type=float, default=1.0,
                        help="Multiplier applied to all rewards on load. "
                             "Use ~100 at 500 words to escape AWR collapse.")
    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    data_path = args.data if os.path.isabs(args.data) else os.path.join(here, args.data)
    out_path = args.out if os.path.isabs(args.out) else os.path.join(here, args.out)

    print(f"Loading dataset from {data_path} (reward_scale={args.reward_scale})")
    buf = OfflineBuffer.from_npz(data_path, reward_scale=args.reward_scale)
    state_dim = buf.states.shape[1]
    print(f"  {len(buf)} transitions, state_dim={state_dim}, "
          f"num_actions={NUM_ACTIONS}")
    print(f"  reward stats: mean={buf.rewards.mean():.4f}, "
          f"std={buf.rewards.std():.4f}, "
          f"max={buf.rewards.max():.4f}")

    # Sanity: confirm env action dim matches dataset.
    env = VocabEnv()
    assert env.state_dim == state_dim, \
        f"state_dim mismatch: env={env.state_dim}, data={state_dim}"
    assert env.action_dim == NUM_ACTIONS

    cfg = IQLConfig(
        tau_expectile=args.tau, beta=args.beta,
        lr=args.lr, batch_size=args.batch,
    )
    print(f"Training IQL for {args.steps} steps "
          f"(tau={cfg.tau_expectile}, beta={cfg.beta}, lr={cfg.lr})")
    trainer = train(buf, state_dim=state_dim, num_actions=NUM_ACTIONS,
                    n_steps=args.steps, cfg=cfg, seed=args.seed,
                    device=args.device)
    trainer.save(out_path)
    print(f"\nsaved policy to {out_path}")


if __name__ == "__main__":
    main()
