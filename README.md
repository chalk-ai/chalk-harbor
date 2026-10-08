# chalk-harbor

A [Harbor](https://www.harborframework.com/) environment provider that runs each trial in a
[Chalk](https://chalk.ai) sandbox. Point Harbor at it with `--env`, and a benchmark's task
containers run on Chalk compute instead of a local Docker daemon.

```bash
harbor run -p <tasks> -a <agent> -e chalk_harbor:ChalkSandboxEnvironment
```

## Install

Install it into the same environment as Harbor:

```bash
uv tool install harbor --with git+https://github.com/chalk-ai/chalk-harbor
```

Authenticate the way any `chalkcompute` client does: `chalk login`, `CHALK_CLIENT_ID`/`CHALK_CLIENT_SECRET`,
or `CHALK_WEB_IDENTITY_TOKEN_FILE`, with `CHALK_ENVIRONMENT_ID` choosing the environment.

## Try it

`examples/smoke` holds three tiny tasks with reference solutions, so Harbor's `oracle` agent solves
them without an LLM:

```bash
harbor run -p examples/smoke -a oracle -e chalk_harbor:ChalkSandboxEnvironment -n 3 --yes
```

Each task gets its own sandbox, and the run reports a mean reward of 1.0. `copy-context` checks
that a Dockerfile `COPY`, followed by a `RUN` that reads the copied file, builds correctly.

## How it works

- **Image:** a task's `docker_image` is used as-is. A single-stage `environment/Dockerfile` is
  translated into a chalkcompute `Image`: `FROM` becomes the base and every other instruction is
  replayed in order. `COPY`/`ADD` build context is embedded as a base64 tarball inside a `RUN`
  step (up to 4 MiB compressed per `COPY`), so later `RUN` steps can see the files.
- **Exec:** commands run through `bash` when the image has it and `sh` otherwise, as the requested
  user.
- **File transfer:** directories move as one tar archive in each direction, which preserves
  modes and symlinks.
- **Resources:** CPU and memory follow the task's `[environment]` settings.
- **Network:** `public` grants all IPv4 egress, `allowlist` maps hostnames and IPv4 CIDRs, and
  `no-network` grants none.

## Traces

Harbor emits no telemetry, and the agent runs inside the trial's sandbox, so nothing it does
reaches the tracer of the process that launched the trial. `chalk_harbor.tracing` turns the
trial's own record into OpenInference spans, with the original timestamps, under whatever span
is current:

- a `harbor.trial` span (AGENT) for the whole trial, with the instruction, final message and
  reward, and one child per phase: environment setup, agent setup, agent execution, verifier;
- an LLM span per agent turn, with model, token counts and cost;
- a TOOL span per tool call, with its arguments and observation.

**Live, while the trial runs.** Wrap the `harbor run` in `stream_trial_spans`, pointed at the
job directory it writes. Harbor rewrites `agent/trajectory.json` after every step, so each
turn's LLM span is exported as soon as the turn is recorded, and its tool spans once the next
step starts. The trial and agent-execution spans end, and arrive, when the trial does; a trace
viewer that hangs spans with a missing parent under a synthetic root shows the agent's
progress as it happens.

```python
from chalk_harbor import stream_trial_spans

with stream_trial_spans(f"jobs/{job}", instruction=open("tasks/<task>/instruction.md").read()):
    subprocess.run(["harbor", "run", ..., "-o", "jobs", "--job-name", job])
```

**Afterwards.** `emit_trial_spans("jobs/<job>/<trial>")` replays a finished trial in one go.

Called from inside a Chalk evaluation task, the spans land in that row's session, so the row's
trace shows the trial. Every span also carries `chalk.evaluation.id` and
`chalk.evaluation.run_id` from the call's evaluation metadata (`evaluation_attributes()`), so a
run's trials can be found by run id while the run is still going. Per-turn spans need an agent
that writes an ATIF trajectory (`agent/trajectory.json`), such as codex, claude-code or
terminus. The `oracle` agent writes none, so its trials show only the phases.

## Post-training on an evaluation

`chalk_harbor.post_training` trains a model on a Harbor-based Chalk evaluation, with the
evaluation's scorers as the reward: one on-policy GRPO step of a LoRA adapter (PEFT +
transformers) per iteration. A post-training workflow runs it as a Chalk training run with
`CHALK_TRAINING_MODULE=chalk_harbor.post_training.train_policy`. The training run's `config`
holds:

| Key | Meaning |
| --- | --- |
| `post_training_id`, `iteration` | which post-training and which iteration k |
| `base_model`, `lora_rank`, `learning_rate` | the Hugging Face base model and the LoRA step |
| `adapter_in`, `adapter_out` | the adapter to start from (`""` at k = 0) and the one to write |
| `adapter_dir` | where the adapter volume is mounted in the training run |
| `policy_server_url`, `policy_adapter_dir` | the vLLM server, and where it mounts the same volume |
| `result_dataset_revision_ids` | one result revision per rollout run of iteration k |
| `group_columns` | dataset columns whose values identify a row; equal values form one group |
| `output_column` | the task output column, whose JSON points at the trial (`volume_path`) |
| `reward_weights` | scorer id to weight (negative for a scorer to minimize) |
| `scorer_columns` | scorer id to `{column, field}`; defaults to `<scorer>_value` / `score` |
| `method` | `grpo` (default) or `sft`, which trains on the rollouts the scorers rate best, each weighted 1, so the step is cross-entropy on them |
| `sft_min_reward` | with `sft`, keep every rollout at or above this weighted reward; unset keeps each row's best |

Optional trainer settings (`max_seq_len`, `logprob_chunk_size`, `trajectory_volume`, and more)
are listed in `chalk_harbor/post_training/config.py`. The trainer reads each sample's ATIF
trajectory from the `harbor-traces` volume at `<volume_path>/agent/trajectory.json`. A
`trajectory` object inline in the output also works.

**The trainer image.** `docker/trainer.Dockerfile` installs CUDA torch, transformers, peft,
chalkcompute, chalkpy and this package. Build it with `docker build -f docker/trainer.Dockerfile .`
and push it to a registry the cluster pulls from. Alternatively, run
`scripts/build_trainer_image.py [--ref <commit>]`: it builds the same image with Chalk's image
builder and prints the URI to pass as `trainer_image`.

Tests run on CPU with a tiny random Qwen3: `uv run --extra post-training pytest`.

## Environment kwargs

Pass with `--ek key=value`:

| Kwarg | Meaning |
| --- | --- |
| `image_map` | `src=dst[,...]`: rewrite a `FROM` or `docker_image` reference, e.g. to a prebuilt image. |
| `entrypoint` | JSON list run as the sandbox's main process instead of `sleep infinity`. |
| `ready_command` | Shell command polled after start until it exits 0, e.g. to wait for entrypoint services. |
| `volumes` | `name:/path[,...]`: existing Chalk volumes to mount, for data too large to bake into an image. |
| `lifetime` | Maximum sandbox lifetime (default `3600s`), so a crashed run doesn't leak sandboxes. |

## Differences from Docker

- **The image's `ENTRYPOINT`/`CMD` does not run.** A sandbox's main process is `sleep infinity`
  unless `entrypoint` says otherwise. Tasks whose image starts a service need `entrypoint`, and
  usually `ready_command`.
- **Build-time writes under `/workspace` are discarded.** `COPY` into `/workspace` is staged under
  `/opt/.harbor-workspace` and restored when the sandbox starts. `RUN` steps that write there are
  still lost, so move that work elsewhere.
- **Memory is raised to 2 GiB per CPU** when a task asks for less, the sandbox service's minimum.
- **Only single-stage Dockerfiles work, with no docker-compose sidecars.** The network policy is
  fixed when the sandbox is created, so it can't change between the agent and verifier phases.

## License

Apache-2.0
