"""Run baseline policies through the student simulator and plot results."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

from baselines import ALL_POLICIES
from simulator import StudentSimulator


N_SESSIONS = 10
TURNS_PER_SESSION = 20
DAYS_BETWEEN_SESSIONS = 1.0
N_SEEDS = 5

TOTAL_TURNS = N_SESSIONS * TURNS_PER_SESSION


def run_one(policy_cls, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run a single policy for one seed.

    Returns (acquired_per_turn, avg_r_per_turn, per_word_R[turn, word_id]).
    """
    sim = StudentSimulator(seed=seed)
    policy = policy_cls()
    rng = np.random.default_rng(seed + 10_000)

    n_words = sim.num_words()
    acquired = np.zeros(TOTAL_TURNS)
    avg_r = np.zeros(TOTAL_TURNS)
    per_word_R = np.zeros((TOTAL_TURNS, n_words))

    t = 0
    for s in range(N_SESSIONS):
        for _ in range(TURNS_PER_SESSION):
            snap = sim.snapshot()
            directive = policy.select(snap, rng)
            sim.step(directive)
            m = sim.metrics()
            acquired[t] = m["acquired"]
            avg_r[t] = m["avg_retrievability"]
            for w in sim.snapshot():
                per_word_R[t, w["word_id"]] = w["retrievability"]
            t += 1
        if s < N_SESSIONS - 1:
            sim.advance_time(DAYS_BETWEEN_SESSIONS)

    return acquired, avg_r, per_word_R


def run_policy(policy_cls) -> dict:
    acq_runs, r_runs, pw_runs = [], [], []
    for seed in range(N_SEEDS):
        a, r, pw = run_one(policy_cls, seed)
        acq_runs.append(a)
        r_runs.append(r)
        pw_runs.append(pw)
    acq = np.stack(acq_runs)
    rr = np.stack(r_runs)
    pw = np.stack(pw_runs).mean(axis=0)  # (turns, words) averaged over seeds
    return {
        "name": policy_cls.name,
        "acq_mean": acq.mean(axis=0),
        "acq_std": acq.std(axis=0),
        "r_mean": rr.mean(axis=0),
        "r_std": rr.std(axis=0),
        "per_word_R": pw,
    }


def plot_metric(results, key_mean, key_std, ylabel, title, outpath):
    plt.figure(figsize=(9, 5.5))
    x = np.arange(1, TOTAL_TURNS + 1)
    for res in results:
        mean = res[key_mean]
        std = res[key_std]
        line, = plt.plot(x, mean, label=res["name"], linewidth=2)
        plt.fill_between(x, mean - std, mean + std, alpha=0.18, color=line.get_color())

    # Mark session boundaries.
    for s in range(1, N_SESSIONS):
        plt.axvline(s * TURNS_PER_SESSION, color="gray", alpha=0.25,
                    linestyle="--", linewidth=0.8)

    plt.xlabel("Turn (across sessions)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc="best", frameon=True)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()
    print(f"  wrote {outpath}")


def plot_per_word_heatmaps(results, outpath):
    """Grid of per-word retention heatmaps (one panel per policy)."""
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 5.0), sharey=True)
    if n == 1:
        axes = [axes]
    im = None
    for ax, res in zip(axes, results):
        # transpose to (words, turns) so x=turn, y=word
        data = res["per_word_R"].T
        im = ax.imshow(data, aspect="auto", origin="lower",
                       vmin=0.0, vmax=1.0, cmap="viridis",
                       interpolation="nearest")
        ax.set_title(res["name"])
        ax.set_xlabel("Turn")
        for s in range(1, N_SESSIONS):
            ax.axvline(s * TURNS_PER_SESSION - 0.5,
                       color="white", alpha=0.4, linewidth=0.5)
    axes[0].set_ylabel("Word ID (0-17: N5,  18-34: N4,  35-50: N3)")
    fig.suptitle("Per-Word Retrievability Over Time (mean across seeds)")
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.012, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="Retrievability")
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  wrote {outpath}")


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    print(f"Running {len(ALL_POLICIES)} policies x {N_SEEDS} seeds "
          f"x {N_SESSIONS} sessions x {TURNS_PER_SESSION} turns...")

    results = []
    for policy_cls in ALL_POLICIES:
        print(f"  policy: {policy_cls.name}")
        results.append(run_policy(policy_cls))

    print("\nFinal metrics (mean over seeds, last turn):")
    for res in results:
        print(f"  {res['name']:10s}  acquired={res['acq_mean'][-1]:5.2f}"
              f"  avg_R={res['r_mean'][-1]:.3f}")

    plot_metric(
        results, "acq_mean", "acq_std",
        ylabel="Acquired words (retrievability > 0.9)",
        title="Vocabulary Acquisition Over Time",
        outpath=os.path.join(here, "acquisition.png"),
    )
    plot_metric(
        results, "r_mean", "r_std",
        ylabel="Mean retrievability of seen words",
        title="Average Retrievability Over Time",
        outpath=os.path.join(here, "retrievability.png"),
    )
    plot_per_word_heatmaps(results, os.path.join(here, "per_word_retention.png"))


if __name__ == "__main__":
    main()
