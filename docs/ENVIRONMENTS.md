# Environments

The cycle runner does not know how a task is run. It asks the loop's **environment** for tasks and
episodes and keeps what the controller owns: the held-out split, the thresholds, the budget, the
receipt and the lineage. This is the seam an external package uses to plug a different kind of
agent into the same loop (a voice agent scored by a simulator, a browser agent, a router).

```yaml
environment:
  kind: harbor-docker                         # registered name (default) ...
  # kind: polyvoice.envs.coval:CovalEnvironment   # ... or module:Class from any installed package
  options: {}                                 # constructor keyword arguments
```

An environment implements (see `polyloop/environment.py`):

| method | called by | contract |
|---|---|---|
| `load_tasks(dataset, limit, seed, names)` | snapshot, filter, evaluate, `polyloop eval` | returns objects with `.task_name`; the pool and the holdout are two datasets |
| `run_rollouts(label, tasks, policy, k, temperature, out, on_result)` | filter, evaluate, `polyloop eval` | K episodes per task under `policy` (`{"id", "sampler_path", "state_path"}`); returns `TaskResult`s and calls `on_result` as each lands; the runner writes `results.jsonl`, the environment may write more under `out` |
| `preflight()` | preflight | list of problems; non-empty aborts before anything is spent |
| `session_hints()` | train (opsd) | per proxy session id, hindsight text appended to every row of that session (a verdict, a judge's explanation); a session's last turn becomes a row only when such a hint exists |
| `excluded_sessions()` | train (opsd) | proxy sessions that must never be trained on: held-out evaluations, probes |
| `name`, `needs_engine_warm` | preflight, `polyloop eval` | label in the preflight receipt; whether the sampler engines must be warmed before rollouts |

`BaseEnvironment` supplies the optional parts. Registered names come from `polyloop/environment.py`
and from the `polyloop.environments` entry-point group, so a package registers one in its own
`pyproject.toml`:

```toml
[project.entry-points."polyloop.environments"]
coval = "polyvoice.envs.coval:CovalEnvironment"
```

Built in: `harbor-docker` (Harbor tasks, tinker-cookbook bash-tool loop, local Docker sandboxes,
verifier reward), used by every recipe in this repo. `tests/fake_env.py` is a scripted environment
that drives the whole runner without a trainer, Docker or a GPU.
