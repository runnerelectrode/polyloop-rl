# polyloop-rl

A loop harness for LLM agents: production traces in, a retrain trigger, a dataset snapshot,
training on a Tinker-compatible endpoint, a held-out gate, promotion with rollback, and repeat.
The controller is code; the researcher is a pluggable agent that only proposes.

Status: scaffolding. First target is a coding-agent demo (Qwen3.5-9B + mini-swe-agent,
SWE-smith training, SWE-bench Verified evaluation) run as nightly cycles with error bars and
cost per cycle reported.

Layout (planned):

- `polyloop/` — loop spec, dispatcher, cycle runner, gate, receipts, backends
- `recipes/` — autoresearch-style recipes (`program.md` + `loop.yaml`) the agent may edit
- `docs/` — design notes and research
- `scripts/` — cluster bring-up and demo scripts

License: Apache-2.0.
