# explaining-quantum-networks

Code, cached input artifacts, and reference results for a workshop paper submission
*"When Routes Run Out: Adversarial Co-Learning and Explainable Robustness in
Quantum Repeater Networks"*.

The pipeline plays a two-player game on 50 structured repeater topologies:
Alice picks an end-to-end route for an E91 game turn, Eve picks a typed attack
surface (edge intercept–resend or repeater-memory degradation). Payoffs come
from cached [SeQUeNCe](https://github.com/sequence-toolbox/SeQUeNCe) E91
transcripts; both players learn with Exp3; results are compared against a
full-matrix minimax reference and explained with decision-tree models and
LLM prompt records.

## Repository layout

```
scripts/                        entry points (see "Reproducing" below)
sequence_game/                  first-party game/pipeline code
sequence_game/experiments/exp3_sequence/
  corpus.sqlite                 the 50 fixed graphs (graph seed 20260708)
  baselines_fidelity_0p98.sqlite  clean-route SeQUeNCe samples, 64/hop profile,
                                  memory-fidelity override 0.98, seed 98000
  attack_baselines.sqlite       attack-hit SeQUeNCe samples, 64/(kind,hop profile),
                                  seed 198000 (generated at fidelity override 1.0;
                                  hits break the CHSH violation regardless)
configs/physical/               device model profiles consumed when (re)generating
                                the SQLite caches (incl. the 0.544 effective
                                branching ratio taken from arXiv:2503.13898)
runs/exp3_dynamic_500k/         reference results (slimmed; see below)
paper/figures/                  the figures used in the paper (vector PDFs)
tests/                          pipeline/prompt/DT tests
sequence-toolbox-SeQUeNCe/      SeQUeNCe git submodule (pinned; see Setup)
```

## Setup

Requires Python 3.11+.

```bash
git clone <this-repo>
cd explaining-quantum-networks
git submodule update --init          # SeQUeNCe, pinned at 5911e969 (v1.0.0-23)
python -m venv .venv
.venv/bin/pip install -r requirements.txt   # installs SeQUeNCe editable from the submodule
```

The pinned submodule commit is the exact SeQUeNCe state used for every cached
sample and for the published run.

## Reproducing the published run

The three SQLite artifacts in `sequence_game/experiments/exp3_sequence/` are
the complete physics input: **the pipeline itself runs no live SeQUeNCe
simulation** when they are present. With them, one command regenerates the
entire run directory (payoff matrices, minimax oracle, 50×500k-turn Exp3
co-learning, decision trees, run figures; ~2.5 h on 12 cores, all seeds fixed):

```bash
.venv/bin/python scripts/exp3_sequence_results.py all \
  --out-dir runs/exp3_dynamic_500k \
  --workers 12 \
  --online-turns 500000 \
  --online-final-window 25000 \
  --online-step-record-stride 500 \
  --exp3-schedule-mode anytime \
  --baseline-cache-db-path sequence_game/experiments/exp3_sequence/baselines_fidelity_0p98.sqlite
```

All other parameters (seed 42, 50 graphs, 16 trials/cell, CHSH-only acceptance,
Exp3 schedule constants) are code defaults; the run's `config.json` records the
full resolved configuration.

LLM prompt records (pure post-processing of the run directory; no model, no
network):

```bash
.venv/bin/python scripts/package_exp3_sequence_llm_prompts.py \
  --run-dir runs/exp3_dynamic_500k --prompt-kind both
```

`scripts/run_exp3_sequence_llm_prompts.py` (stdlib-only) sends those prompts to
a local OpenAI/Ollama-style endpoint to collect model responses for the rubric
evaluation.

Paper figures (vector PDFs into `paper/figures/`; prebuilt copies of the
figures used in the paper are included):

```bash
.venv/bin/python scripts/make_paper_figures.py
```

## Reference results in `runs/exp3_dynamic_500k/`

The shipped run directory is the published run with three bulky, fully
regenerable outputs removed for repository size (they are recreated verbatim by
the pipeline command above, ~4 GB total):

* `exp3_summary.json` (~2.1 GB) — per-graph online summaries incl. step records
* `online_runs/` (~1.9 GB) — per-graph learning curves and step records
* `payoff_matrices/` (~61 MB) — per-graph payoff cell details

What remains — `config.json`, `run_summary.json`, `oracle_summary.json`,
`baseline_health.json`, `corpus.json`, `dt/`, `figures/`, `llm_prompts/` — is
sufficient to audit every number in the paper and to re-run the prompt
packaging. `scripts/make_paper_figures.py` additionally needs `online_runs/`,
so regenerate the run first if you want to rebuild the learning-dynamics
figures from scratch.

Provenance of the shipped run is recorded inside `run_summary.json` and
`config.json` (source git commit, clean tree, resolved configuration).

## Regenerating the SQLite caches from scratch

Only needed if you want to change the physics; this *does* run live SeQUeNCe
(hours). Each cache stores its own metadata (seed, config, git state).

```bash
# Clean-route baselines at memory-fidelity override 0.98 (seed 98000, 64/hop):
.venv/bin/python scripts/precompute_exp3_sequence_baselines.py \
  --db-path sequence_game/experiments/exp3_sequence/baselines_fidelity_0p98.sqlite \
  --sequence-memory-fidelity-override 0.98

# Attack-hit samples (seed 198000, 64 per attack kind x hop profile):
.venv/bin/python scripts/precompute_exp3_sequence_attacks.py

# Corpus route-length normalization (already applied to the shipped corpus.sqlite):
.venv/bin/python scripts/normalize_exp3_sequence_corpus.py
```

## Tests

```bash
.venv/bin/python -m pytest tests -q
```

## Scope

Payoffs are acceptance rates over cached simulator draws; the CHSH acceptance
rule is a finite-sample game mechanic. Memory fidelity 0.98 and efficiency
0.544 are controlled simulation parameters — the latter borrows one reported
number from arXiv:2503.13898 to motivate the scale of a high-efficiency
profile, without reproducing that architecture.
