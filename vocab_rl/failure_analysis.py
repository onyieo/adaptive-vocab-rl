"""Failure-mode analysis for a trained IQL/CQL policy.

For each step in a rollout, computes:
  - The action the policy chose
  - Q-values for ALL actions at that state
  - 'Regret' = max_a Q(s, a) - Q(s, a_policy)   (policy's gap from its own argmax)
  - Whether the chosen action matches the strongest baseline (FSRS+New)

Outputs:
  - failure_modes.png : (1) histogram of regret across all timesteps,
                         (2) regret-by-turn-in-session (do failures cluster late?),
                         (3) action-disagreement-with-FSRS+New per turn
  - failure_modes.json: top-K worst decisions with state context for the writeup.

Caveats this tool deliberately surfaces:
  - 'Regret' is wrt the policy's OWN Q-function, not ground-truth optimal Q.
    A regret of zero means the policy picked its highest-Q action; it does
    NOT mean the action was good. This is a calibration check, not a quality
    check.
  - Disagreement with FSRS+New is informative only insofar as FSRS+New is
    a reasonable point of comparison; differences could mean IQL learned
    something better OR worse. Combine with the eval metrics.
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from baseline_actions import FSRSPlusNewAction
from env import ACTION_TABLE, NUM_ACTIONS, VocabEnv
from iql import IQLConfig, IQLTrainer
from simulator import VOCAB


N_ROLLOUTS = 5
TOP_K_REGRET = 10


def _q_values(trainer: IQLTrainer, state: np.ndarray) -> np.ndarray:
    s = torch.from_numpy(state.astype(np.float32)).unsqueeze(0).to(trainer.device)
    with torch.no_grad():
        q = trainer.q(s).squeeze(0).cpu().numpy()
    return q


def _state_summary(env: VocabEnv) -> dict:
    snap = env.sim.snapshot()
    seen = [w for w in snap if w["seen"]]
    return {
        "turn": env.turn,
        "turn_in_session": env.turn % env.turns_per_session,
        "session": env.turn // env.turns_per_session,
        "n_seen": len(seen),
        "n_unseen": len(snap) - len(seen),
        "mean_R_seen": float(np.mean([w["retrievability"] for w in seen])) if seen else 0.0,
        "mean_stability_seen": float(np.mean([w["stability"] for w in seen])) if seen else 0.0,
        "n_due_under_0.5": sum(1 for w in seen if w["retrievability"] < 0.5),
    }


def analyze(policy_path: str, n_rollouts: int = N_ROLLOUTS) -> dict:
    env_probe = VocabEnv()
    trainer = IQLTrainer(state_dim=env_probe.state_dim,
                         num_actions=NUM_ACTIONS, cfg=IQLConfig())
    trainer.load(policy_path)

    fsrs_plus_new = FSRSPlusNewAction()
    rng = np.random.default_rng(0)

    records = []
    for seed in range(n_rollouts):
        env = VocabEnv(seed=seed)
        s = env.reset()
        for t in range(env.total_turns):
            q = _q_values(trainer, s)
            a = int(np.argmax(q))
            best_q = float(q.max())
            chosen_q = float(q[a])  # tautologically equal here, but kept for clarity
            regret = best_q - chosen_q  # always 0 for deterministic argmax IQL

            # More useful: how much worse is the SECOND-best action?
            second = float(np.sort(q)[-2])
            confidence = best_q - second  # bigger = more decisive

            baseline_a = fsrs_plus_new(s, rng)
            agrees_with_fsrs_plus_new = (a == baseline_a)

            records.append({
                "seed": seed,
                "step": t,
                "turn_in_session": t % env.turns_per_session,
                "session": t // env.turns_per_session,
                "action": a,
                "action_tuple": list(ACTION_TABLE[a]),
                "q_chosen": chosen_q,
                "q_best": best_q,
                "q_min": float(q.min()),
                "q_argmin": int(q.argmin()),
                "confidence": confidence,
                "regret_from_second": chosen_q - second,
                "agrees_fsrs_plus_new": bool(agrees_with_fsrs_plus_new),
                "fsrs_plus_new_action": int(baseline_a),
                "state_summary": _state_summary(env),
            })

            s, _r, _d, _info = env.step(a)

    return {"records": records}


def plot_failure_modes(analysis: dict, out_path: str,
                       turns_per_session: int = 20) -> None:
    recs = analysis["records"]
    confidence = np.array([r["confidence"] for r in recs])
    turn_in_sess = np.array([r["turn_in_session"] for r in recs])
    agrees = np.array([r["agrees_fsrs_plus_new"] for r in recs])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    axes[0].hist(confidence, bins=30, color="steelblue", edgecolor="black")
    axes[0].set_xlabel("Q(argmax) - Q(2nd best)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Policy decisiveness across all steps")
    axes[0].grid(True, alpha=0.3)

    by_turn_conf = [confidence[turn_in_sess == t].mean()
                    for t in range(turns_per_session)]
    axes[1].plot(range(1, turns_per_session + 1), by_turn_conf,
                 marker="o", linewidth=2, color="steelblue")
    axes[1].set_xlabel("Turn within session")
    axes[1].set_ylabel("Mean decisiveness")
    axes[1].set_title("Decisiveness by turn-in-session")
    axes[1].grid(True, alpha=0.3)

    by_turn_agree = [agrees[turn_in_sess == t].mean()
                     for t in range(turns_per_session)]
    axes[2].bar(range(1, turns_per_session + 1), by_turn_agree,
                color="tab:orange", edgecolor="black", width=0.85)
    axes[2].set_xlabel("Turn within session")
    axes[2].set_ylabel("Fraction agreeing with FSRS+New")
    axes[2].set_title("IQL vs FSRS+New action agreement")
    axes[2].set_ylim(0, 1)
    axes[2].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  wrote {out_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--policy", default="iql_policy.pt")
    p.add_argument("--rollouts", type=int, default=N_ROLLOUTS)
    p.add_argument("--out-plot", default="failure_modes.png")
    p.add_argument("--out-json", default="failure_modes.json")
    args = p.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    policy_path = (args.policy if os.path.isabs(args.policy)
                   else os.path.join(here, args.policy))
    out_plot = args.out_plot if os.path.isabs(args.out_plot) else os.path.join(here, args.out_plot)
    out_json = args.out_json if os.path.isabs(args.out_json) else os.path.join(here, args.out_json)

    print(f"Loaded policy from {policy_path}")
    analysis = analyze(policy_path, n_rollouts=args.rollouts)
    n = len(analysis["records"])
    print(f"  analyzed {n} steps across {args.rollouts} rollouts")

    # Sort by lowest decisiveness — these are the most uncertain decisions.
    recs = analysis["records"]
    least_decisive = sorted(recs, key=lambda r: r["confidence"])[:TOP_K_REGRET]
    print(f"\n{TOP_K_REGRET} least-decisive decisions (lowest Q-gap to 2nd best):")
    for r in least_decisive[:5]:
        s = r["state_summary"]
        print(f"  session {s['session']} turn {s['turn_in_session']:2d}: "
              f"chose {tuple(r['action_tuple'])}  "
              f"conf={r['confidence']:+.4f}  "
              f"seen={s['n_seen']:2d} due<0.5={s['n_due_under_0.5']:2d}  "
              f"agrees_FSRSplusNew={r['agrees_fsrs_plus_new']}")

    plot_failure_modes(analysis, out_plot)

    summary = {
        "n_steps": n,
        "rollouts": args.rollouts,
        "mean_confidence": float(np.mean([r["confidence"] for r in recs])),
        "fraction_agrees_FSRS_plus_New": float(np.mean(
            [r["agrees_fsrs_plus_new"] for r in recs])),
        "action_distribution": {str(tuple(ACTION_TABLE[i])):
                                int(sum(r["action"] == i for r in recs))
                                for i in range(NUM_ACTIONS)},
        "least_decisive_topk": least_decisive[:TOP_K_REGRET],
    }
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  saved -> {out_json}")


if __name__ == "__main__":
    main()
