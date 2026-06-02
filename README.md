# Offline RL for Adaptive Vocabulary Selection in Conversational Language Tutoring

CS224R (Stanford, Spring 2026) final project. Formulates vocabulary selection in
LLM-mediated language tutoring as an offline reinforcement-learning problem:
the RL policy outputs structured vocabulary directives, and a frozen
conversation LLM realizes them as natural-sounding tutor turns.

## Components

| File | What it is |
| --- | --- |
| [`vocab_rl/simulator.py`](vocab_rl/simulator.py) | Knowledge-tracing student simulator over 51 Japanese vocabulary words (JLPT N5/N4/N3). FSRS-style forgetting curve `R(t) = (1 + t / (9·S))^-1` with parameterizable forgetting constant and difficulty scale. |
| [`vocab_rl/env.py`](vocab_rl/env.py) | Gym-style MDP wrapper. Discrete action space of 21 structured directives `(n_new, n_active, n_review)` summing to 5 words/turn. State = per-word stability/retrievability/difficulty/log-usage + session globals (206 dims). Reward = small delta-acquired shaping + terminal mean retrievability. |
| [`vocab_rl/baselines.py`](vocab_rl/baselines.py) | Word-ID baselines used in the milestone (`Random`, `FSRS`, `FSRS+New`, `RuleBased`). |
| [`vocab_rl/baseline_actions.py`](vocab_rl/baseline_actions.py) | Same baselines re-expressed over the env action space, plus a `MixedExplore` policy used to broaden offline-data coverage. |
| [`vocab_rl/collect_dataset.py`](vocab_rl/collect_dataset.py) | Rolls out all 5 behavior policies and saves a single `.npz` of `(s, a, r, s', done, policy_id)` transitions. |
| [`vocab_rl/iql.py`](vocab_rl/iql.py) | Discrete-action IQL. V via expectile regression on Q_target(s, a), Q via TD-on-V, π via advantage-weighted regression. Polyak target Q. |
| [`vocab_rl/cql.py`](vocab_rl/cql.py) | CQL secondary method. Same V/π updates as IQL plus the discrete conservative penalty `L_CQL = α · (logsumexp_a Q(s,a) - Q(s,a_data))`. |
| [`vocab_rl/train_iql.py`](vocab_rl/train_iql.py), [`train_cql.py`](vocab_rl/train_cql.py) | Training entrypoints. |
| [`vocab_rl/evaluate_policies.py`](vocab_rl/evaluate_policies.py) | Loads a trained policy and evaluates it alongside the baselines over 10 sessions × 20 turns × 5 seeds. Produces acquisition curves, retrievability curves, and an action-distribution diagnostic. |
| [`vocab_rl/robustness_eval.py`](vocab_rl/robustness_eval.py) | Evaluates the trained policy on held-out simulator parameterizations (default, fast/slow forgetting, harder/easier words) to test the proposal's robustness claim. |
| [`vocab_rl/run_experiment.py`](vocab_rl/run_experiment.py) | The original milestone-era experiment runner (no RL — just baselines + the per-word heatmap). |
| [`vocab_rl/llm/tutor.py`](vocab_rl/llm/tutor.py) | LLM tutor: Claude Sonnet 4.6 with prompt-cached system prompt (instructions + full vocab table). Turns a directive into one Japanese conversational turn. |
| [`vocab_rl/llm/judge.py`](vocab_rl/llm/judge.py) | LLM-as-judge: Claude Haiku 4.5 with structured JSON output, scoring naturalness 1–5. |
| [`vocab_rl/llm/llm_eval.py`](vocab_rl/llm/llm_eval.py) | Runs each policy for 1 session × 20 turns with real conversation generation; saves transcripts and produces a naturalness bar chart. |

## Reproduce

```bash
cd vocab_rl
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...   # only required for the LLM eval

# 1. Build offline dataset (~1 min)
python collect_dataset.py --n 200 --out dataset.npz

# 2. Train IQL (~10-20 min CPU) and CQL (~10-20 min CPU)
python train_iql.py --data dataset.npz --out iql_policy.pt --steps 100000
python train_cql.py --data dataset.npz --out cql_policy.pt --steps 100000

# 3. Evaluate against baselines (a few seconds)
python evaluate_policies.py --policy iql_policy.pt

# 4. Robustness across held-out simulator variants
python robustness_eval.py --policy iql_policy.pt

# 5. LLM-mediated naturalness eval (a few minutes, costs ~$3-5 in API calls)
python -m llm.llm_eval --iql-policy iql_policy.pt
```

## Status

Phase 1 (simulator + baselines) and Phase 2 (offline RL) are complete.
Phase 3 (full eval, ablations, final report) is in progress. See
[`report/milestone.tex`](report/milestone.tex) for the milestone report.

## AI tools disclosure

Code in this project was scaffolded with assistance from Claude (Anthropic) as
permitted under the CS224R AI policy (Ed #824). All algorithmic and modeling
decisions were specified and reviewed by the author.
