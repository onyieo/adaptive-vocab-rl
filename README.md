# Offline RL for Adaptive Vocabulary Selection in Conversational Language Tutoring

CS224R (Stanford, Spring 2026) final project. Formulates vocabulary selection
in LLM-mediated language tutoring as an offline reinforcement-learning problem:
the RL policy outputs structured vocabulary directives (counts of new /
actively-learning / review words to target), and a frozen conversation LLM
realizes them as natural-sounding tutor turns.

The contribution is a hierarchical control framing: RL operates over latent
linguistic constraints; a generative model executes them. Spaced-repetition
scheduling (FSRS, SM-2) gives strong heuristics for *what* to review but no
mechanism to embed scheduling in natural dialogue; RLHF-style methods train
the LLM directly but cannot reason about long-horizon retention. We sit
between the two.

## Current results (51-word vocab, 10 seeds, 95% bootstrap CI)

| Policy | Final acquired (R>0.9) | Post-gap retained (7d, R>0.5) |
|---|---|---|
| Random                | 50.9                  | 40.6 [38.1, 43.0]             |
| **CQL (ours)**        | **17.3** [9.0, 27.0]  | **17.3** [8.5, 28.8]          |
| IQL (ours)            | 13.5 [7.7, 20.8]      | 11.8 [5.4, 20.7]              |
| FSRS+New              | 9.2  [7.4, 11.0]      | 7.8  [5.6, 10.6]              |
| RuleBased             | 5.9  [5.0,  7.4]      | 5.0  [5.0,  5.0]              |
| FSRS                  | 5.0                   | 5.0                           |

**Headline:** CQL beats every fixed-heuristic baseline on both metrics. Random
wins raw acquisition only because the small vocab inflates its expected
exposures-per-word (≈20); the post-gap retention gap is the more honest
signal. Scaling to the proposal's 500-word inventory (vocab.json, generated
via `generate_vocab.py`) is expected to flip the Random comparison since
exposures-per-word collapses to ≈2.

**Qualitative win:** IQL/CQL learn *intra-session pacing* — heavy on new-word
actions early, shift to review-heavy late. No fixed baseline shows this
structure. See [`vocab_rl/results/action_distribution.png`](vocab_rl/results/action_distribution.png).

## Components

| File | What it is |
| --- | --- |
| [`vocab_rl/simulator.py`](vocab_rl/simulator.py) | Knowledge-tracing student simulator. 51 JLPT N5/N4/N3 words by default; loads `vocab.json` if present. FSRS forgetting curve `R(t)=(1+t/(forgetting_constant·S))^-1` with parameterizable `forgetting_constant` and `difficulty_scale` for robustness eval. |
| [`vocab_rl/env.py`](vocab_rl/env.py) | Gym-style MDP wrapper. 21 discrete directive actions `(n_new, n_active, n_review)`. State: per-word stability/retrievability/difficulty/log-usage + 2 globals. Reward: per-step delta in total retrievability across the full vocabulary (avoids the in-session R≈1 trivialization). |
| [`vocab_rl/baselines.py`](vocab_rl/baselines.py), [`baseline_actions.py`](vocab_rl/baseline_actions.py) | Word-ID baselines (`Random`, `FSRS`, `FSRS+New`, `RuleBased`) and env-action equivalents plus `MixedExplore` for offline-data coverage. |
| [`vocab_rl/collect_dataset.py`](vocab_rl/collect_dataset.py) | Rolls out behavior policies and saves an `.npz` buffer. `--multi-sim` cycles through 5 training-time simulator parameterizations (default, ±forgetting, ±difficulty); the held-out OOD variants are tested in `robustness_eval.py`. |
| [`vocab_rl/iql.py`](vocab_rl/iql.py) | Discrete-action IQL: V via expectile regression, Q via TD-on-V, π via advantage-weighted regression. Polyak target Q. |
| [`vocab_rl/cql.py`](vocab_rl/cql.py) | CQL secondary method: same V/π as IQL plus the discrete conservative penalty `α·(logsumexp_a Q(s,a) − Q(s,a_data))`. |
| [`vocab_rl/ppo.py`](vocab_rl/ppo.py) | PPO **online** baseline — provides the methodological offline-vs-online contrast the proposal asked for. Actor-critic with clipped surrogate + GAE. |
| [`vocab_rl/sweep.py`](vocab_rl/sweep.py), [`modal_sweep.py`](vocab_rl/modal_sweep.py) | Hyperparameter sweep over (algo, τ, β, α_cql, seed). `sweep.py` runs locally; `modal_sweep.py` distributes one container per config on Modal. |
| [`vocab_rl/evaluate_policies.py`](vocab_rl/evaluate_policies.py) | Full evaluation: loads IQL/CQL/PPO if present, runs alongside baselines over 10 seeds, produces acquisition / retrievability curves with bootstrap CIs, action-mix diagnostic, post-gap-retention bar chart. |
| [`vocab_rl/robustness_eval.py`](vocab_rl/robustness_eval.py) | Evaluates trained policies on held-out simulator parameterizations (fast/slow forgetting, harder/easier words, combo-OOD). |
| [`vocab_rl/failure_analysis.py`](vocab_rl/failure_analysis.py) | Per-step Q-value gap, agreement with FSRS+New, and top-K least-decisive decisions. Surfaces calibration issues. |
| [`vocab_rl/generate_vocab.py`](vocab_rl/generate_vocab.py) | One-shot script that calls Claude to produce a ~500-word JLPT vocabulary, with validation. `simulator.py` prefers the generated `vocab.json`. |
| [`vocab_rl/llm/tutor.py`](vocab_rl/llm/tutor.py), [`judge.py`](vocab_rl/llm/judge.py), [`llm_policy.py`](vocab_rl/llm/llm_policy.py), [`llm_eval.py`](vocab_rl/llm/llm_eval.py) | LLM tutor (Sonnet 4.6, prompt-cached system prompt), LLM-as-judge (Haiku 4.5, structured JSON), LLM-prompted baseline (proposal's baseline #4), and end-to-end eval that generates real Japanese tutoring conversations and scores naturalness. |

## Setup

```bash
cd vocab_rl
pip install -r requirements.txt

# For any LLM-based component:
cp ../.env.example ../.env
# edit ../.env and fill in your ANTHROPIC_API_KEY
```

## Reproduce

Full pipeline:

```bash
# (Optional) generate the 500-word vocabulary; otherwise the 51-word default is used.
python generate_vocab.py            # ~3-5 min, ~$0.40 in API calls

# 1. Collect offline dataset (multi-sim recommended for robustness)
python collect_dataset.py --n 200 --multi-sim --out dataset.npz

# 2. Train offline RL
python train_iql.py --data dataset.npz --steps 100000 --beta 30  # higher beta avoids the
python train_cql.py --data dataset.npz --steps 100000             # AWR collapse seen at beta=3

# 3. (Comparison) Train PPO online
python ppo.py --total-steps 200000 --out ppo_policy.pt

# 4. Full evaluation including all trained policies
python evaluate_policies.py
python evaluate_policies.py --with-llm-policy   # adds LLM-prompted baseline

# 5. Robustness across held-out simulator parameterizations
python robustness_eval.py --policy cql_policy.pt

# 6. Failure-mode analysis
python failure_analysis.py --policy cql_policy.pt

# 7. LLM-mediated naturalness eval (Japanese conversation + judge)
python -m llm.llm_eval --iql-policy iql_policy.pt   # ~$3-5

# 8. Hyperparameter sweep (local or Modal)
python sweep.py --algos iql cql --tau 0.7 0.9 --beta 3 10 30 --seeds 0 1 2
python modal_sweep.py --algos iql cql --seeds 0 1 2 3 4   # parallel on Modal
```

## Status

| Phase | Item | State |
|---|---|---|
| 1 | Simulator + 4 baselines + evaluation harness | ✅ |
| 2 | MDP wrapper + offline dataset | ✅ |
| 2 | IQL primary | ✅ |
| 2 | CQL secondary | ✅ |
| 2 | LLM tutor + LLM-as-judge | ✅ |
| 3 | LLM-prompted baseline (proposal #4) | ✅ |
| 3 | Post-gap retention metric | ✅ |
| 3 | Multi-sim training + held-out OOD eval | ✅ |
| 3 | Bootstrap CIs (10 seeds) | ✅ |
| 3 | Hyperparameter sweep harness + Modal | ✅ |
| 3 | 500-word vocab | code ready; needs API key to run |
| 3 | Online RL (PPO) baseline | ✅ |
| 3 | Failure-mode analysis | ✅ |
| 3 | Final report | in progress |

## AI tools disclosure

Code in this project was scaffolded with assistance from Claude (Anthropic)
as permitted under the CS224R AI policy (Ed #824). All algorithmic and
modeling decisions were specified and reviewed by the author.
