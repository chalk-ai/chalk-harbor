"""One GRPO + LoRA iteration of evaluation post-training, as a Chalk training function.

A post-training workflow runs the evaluation ``samples_per_row`` times with the current policy,
then starts a training run whose entrypoint (``python -m chalkcompute.training.entrypoint``)
calls ``train_policy(df, config)`` with the contract in ``config.TrainerConfig``. This:

1. reads every rollout's result dataset revision and computes each row's weighted reward and
   its Dr-GRPO advantage within its group (``rewards``);
2. loads the trajectory each informative row's output points at and rebuilds the agent's chat
   (``trajectory``), then tokenizes it with the base model's chat template and masks it to
   the tokens the policy generated (``masking``);
3. loads the base model with ``adapter_in`` (or a fresh LoRA) and takes one policy-gradient
   step (``grpo``);
4. saves the adapter to ``<adapter_dir>/<adapter_out>``, commits the volume, records a
   checkpoint and metrics, and asks the policy server to load the adapter as ``adapter_out``.

Everything that touches the network or the filesystem goes through ``TrainerIO`` so the
step runs end to end in tests with a tiny model and synthetic data.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from chalk_harbor.post_training.config import TrainerConfig
from chalk_harbor.post_training.masking import TemplateMismatchError, tokenize_chat
from chalk_harbor.post_training.rewards import (
    Sample,
    group_advantages,
    group_stats,
    score_rows,
)
from chalk_harbor.post_training.trajectory import (
    TrajectoryError,
    atif_to_chat,
    trajectory_ref,
)

# Optimizer state carried between iterations, under the training run's checkpoint volume
# (shared by every iteration of one post-training, since they share a run name).
OPTIMIZER_STATE_FILE = "post_training_optimizer.pt"
# A missing trajectory is retried once after this long: the Larkspur task uploads the trial
# record after its row returns, so the newest rows can lag the run's completion.
TRAJECTORY_RETRY_DELAY_SECONDS = 30.0
TRAJECTORY_READ_THREADS = 8


def _log(message: str) -> None:
    print(f"[post-training] {message}", file=sys.stderr, flush=True)


@dataclass(frozen=True)
class TrainerIO:
    """The trainer's side effects; ``default_io`` wires the real ones."""

    read_revision: Callable[[str], list[dict[str, Any]]]
    # Returns None when the file does not exist.
    read_trajectory: Callable[[str, str], Mapping[str, Any] | None]
    load_tokenizer: Callable[[str], Any]
    load_base_model: Callable[[str, str], Any]
    commit_mount: Callable[[str], None]
    load_adapter: Callable[[str, str, str, float], None]
    log_metrics: Callable[[Mapping[str, float], Mapping[str, str]], None]
    checkpoint: Callable[[Any, list[str], dict[str, Any]], None]
    checkpoint_dir: str | None
    retry_delay_seconds: float


@dataclass(frozen=True)
class IterationSummary:
    post_training_id: str
    iteration: int
    adapter_out: str
    rows: int
    skipped_rows: int
    groups: int
    informative_groups: int
    trained_sequences: int
    missing_trajectories: int
    truncated_sequences: int
    mean_reward: float
    reward_std: float
    loss: float
    grad_norm: float
    generated_tokens: int
    mean_response_tokens: float
    mean_logprob: float


def train_policy(df: Any, config: Mapping[str, Any]) -> dict[str, Any]:
    """Training-run entrypoint: one post-training iteration (see the module docstring).

    ``df`` is the training run's own dataset (the first rollout's results), which the
    contract's ``result_dataset_revision_ids`` already include; it is used only when that
    list is empty.
    """
    summary = run_iteration(TrainerConfig.from_mapping(config), df, default_io())
    return asdict(summary)


def run_iteration(config: TrainerConfig, df: Any, io: TrainerIO) -> IterationSummary:
    import torch

    _log(
        f"{config.post_training_id} iteration {config.iteration}: "
        + f"{config.adapter_in or config.base_model} -> {config.adapter_out}"
    )
    rows = _load_rows(config, df, io)
    scored, skipped = score_rows(
        rows,
        group_columns=config.group_columns,
        output_column=config.output_column,
        weights=config.reward_weights,
        columns=config.scorer_columns,
    )
    samples = group_advantages(scored)
    stats = group_stats(scored, samples, skipped)
    _log(
        f"{stats.rows} scored rows ({stats.skipped_rows} skipped) in {stats.groups} groups, "
        + f"{stats.informative_groups} with a reward spread; mean reward "
        + f"{stats.mean_reward:.4f} (std {stats.reward_std:.4f})"
    )
    if not samples:
        raise RuntimeError(
            "no group has a reward spread to learn from; every rollout of each dataset row "
            + "scored the same"
        )

    tokenizer = io.load_tokenizer(config.base_model)
    examples, missing, truncated = _build_examples(config, samples, tokenizer, io)
    if not examples:
        raise RuntimeError("no trajectory could be loaded for any informative row")

    from chalk_harbor.post_training.grpo import accumulate_and_step

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_policy(config, io, device)
    optimizer = _optimizer(config, model, io)
    _log(f"training on {len(examples)} sequences ({missing} trajectories missing)")
    result = accumulate_and_step(
        model.get_base_model(),
        optimizer,
        examples,
        chunk_size=config.logprob_chunk_size,
        max_grad_norm=config.max_grad_norm,
        device=device,
        log=_log,
    )
    _log(
        f"loss {result.loss:.6f}, grad norm {result.grad_norm:.4f}, "
        + f"{result.generated_tokens} generated tokens"
    )

    adapter_path = _save_adapter(config, model, io)
    _save_optimizer(config, optimizer, io)
    summary = IterationSummary(
        post_training_id=config.post_training_id,
        iteration=config.iteration,
        adapter_out=config.adapter_out,
        rows=stats.rows,
        skipped_rows=stats.skipped_rows,
        groups=stats.groups,
        informative_groups=stats.informative_groups,
        trained_sequences=len(examples),
        missing_trajectories=missing,
        truncated_sequences=truncated,
        mean_reward=stats.mean_reward,
        reward_std=stats.reward_std,
        loss=result.loss,
        grad_norm=result.grad_norm,
        generated_tokens=result.generated_tokens,
        mean_response_tokens=result.generated_tokens / len(examples),
        mean_logprob=result.mean_logprob,
    )
    _report(config, model, adapter_path, summary, io)
    io.load_adapter(
        config.policy_server_url,
        config.adapter_out,
        f"{config.policy_adapter_dir.rstrip('/')}/{config.adapter_out}",
        config.load_timeout_seconds,
    )
    _log(f"policy server serves {config.adapter_out}")
    return summary


# -- data ----------------------------------------------------------------------------------


def _load_rows(
    config: TrainerConfig, df: Any, io: TrainerIO
) -> list[tuple[str, dict[str, Any]]]:
    rows: list[tuple[str, dict[str, Any]]] = []
    if config.result_dataset_revision_ids:
        for revision in config.result_dataset_revision_ids:
            revision_rows = io.read_revision(revision)
            _log(f"revision {revision}: {len(revision_rows)} rows")
            rows.extend((f"{revision}#{i}", row) for i, row in enumerate(revision_rows))
        return rows
    return [(f"data#{i}", row) for i, row in enumerate(_frame_rows(df))]


def _frame_rows(df: Any) -> list[dict[str, Any]]:
    if df is None:
        return []
    if hasattr(df, "collect"):  # polars LazyFrame, as the training entrypoint passes it
        df = df.collect()
    if hasattr(df, "to_dicts"):
        return df.to_dicts()
    if hasattr(df, "to_pylist"):
        return df.to_pylist()
    if hasattr(df, "to_dict"):
        return df.to_dict(orient="records")
    raise TypeError(f"cannot read rows from {type(df).__name__}")


def _build_examples(
    config: TrainerConfig, samples: Sequence[Sample], tokenizer: Any, io: TrainerIO
) -> tuple[list[Any], int, int]:
    from chalk_harbor.post_training.grpo import TrainingExample

    trajectories = _load_trajectories(config, samples, io)
    examples: list[TrainingExample] = []
    missing = 0
    truncated = 0
    for sample, trajectory in zip(samples, trajectories, strict=True):
        if trajectory is None:
            missing += 1
            continue
        try:
            chat = atif_to_chat(trajectory)
            tokens = tokenize_chat(tokenizer, chat, config.max_seq_len)
        except (TrajectoryError, TemplateMismatchError, KeyError, TypeError) as exc:
            _log(f"skipping {sample.row.source}: {type(exc).__name__}: {exc}")
            missing += 1
            continue
        if tokens.generated_tokens == 0:
            missing += 1
            continue
        truncated += int(tokens.truncated)
        examples.append(
            TrainingExample(
                input_ids=tokens.input_ids,
                loss_mask=tokens.loss_mask,
                advantage=sample.advantage,
            )
        )
    return examples, missing, truncated


def _load_trajectories(
    config: TrainerConfig, samples: Sequence[Sample], io: TrainerIO
) -> list[Mapping[str, Any] | None]:
    refs = [
        trajectory_ref(
            s.row.output,
            path_field=config.trajectory_path_field,
            file=config.trajectory_file,
        )
        for s in samples
    ]

    def read(index: int) -> Mapping[str, Any] | None:
        ref = refs[index]
        if ref is None:
            return None
        if ref.inline is not None:
            return ref.inline
        assert ref.volume_path is not None
        return io.read_trajectory(config.trajectory_volume, ref.volume_path)

    with ThreadPoolExecutor(TRAJECTORY_READ_THREADS) as pool:
        loaded = list(pool.map(read, range(len(samples))))
        retry = [i for i, t in enumerate(loaded) if t is None and refs[i] is not None]
        if retry:
            _log(
                f"{len(retry)} trajectories not found; retrying in "
                + f"{io.retry_delay_seconds:.0f}s"
            )
            time.sleep(io.retry_delay_seconds)
            for i, trajectory in zip(retry, pool.map(read, retry), strict=True):
                loaded[i] = trajectory
    return loaded


# -- model ---------------------------------------------------------------------------------


def _load_policy(config: TrainerConfig, io: TrainerIO, device: Any) -> Any:
    from peft import LoraConfig, PeftModel, get_peft_model

    base = io.load_base_model(config.base_model, config.torch_dtype)
    base.config.use_cache = False
    base.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    if config.adapter_in:
        path = Path(config.adapter_dir) / config.adapter_in
        model = PeftModel.from_pretrained(base, str(path), is_trainable=True)
        _log(f"loaded adapter {path}")
    else:
        model = get_peft_model(
            base,
            LoraConfig(
                r=config.lora_rank,
                lora_alpha=config.lora_alpha,
                lora_dropout=0.0,
                target_modules="all-linear",
                bias="none",
                task_type="CAUSAL_LM",
            ),
        )
    model.to(device)
    model.train()
    return model


def _optimizer(config: TrainerConfig, model: Any, io: TrainerIO) -> Any:
    import torch

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        weight_decay=config.weight_decay,
    )
    state_path = _optimizer_state_path(config, io)
    if state_path is None or not config.adapter_in or not state_path.exists():
        return optimizer
    saved = torch.load(state_path, map_location="cpu", weights_only=False)
    # Adam's moments only fit the adapter they were accumulated for.
    if saved.get("adapter") != config.adapter_in:
        _log(f"ignoring optimizer state for {saved.get('adapter')!r}")
        return optimizer
    optimizer.load_state_dict(saved["state"])
    for group in optimizer.param_groups:
        group["lr"] = config.learning_rate
    _log(f"resumed optimizer state from {state_path}")
    return optimizer


def _optimizer_state_path(config: TrainerConfig, io: TrainerIO) -> Path | None:
    if not io.checkpoint_dir:
        return None
    return Path(io.checkpoint_dir) / config.post_training_id / OPTIMIZER_STATE_FILE


def _save_optimizer(config: TrainerConfig, optimizer: Any, io: TrainerIO) -> None:
    import torch

    state_path = _optimizer_state_path(config, io)
    if state_path is None:
        return
    state_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"adapter": config.adapter_out, "state": optimizer.state_dict()}, state_path
    )
    io.commit_mount(str(io.checkpoint_dir))


def _save_adapter(config: TrainerConfig, model: Any, io: TrainerIO) -> Path:
    path = Path(config.adapter_dir) / config.adapter_out
    if path.exists():  # a retried run rewrites the adapter it already wrote
        shutil.rmtree(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(path))
    io.commit_mount(config.adapter_dir)
    _log(f"saved adapter to {path}")
    return path


def _report(
    config: TrainerConfig,
    model: Any,
    adapter_path: Path,
    summary: IterationSummary,
    io: TrainerIO,
) -> None:
    tags = {
        "post_training_id": config.post_training_id,
        "iteration": str(config.iteration),
    }
    metrics = {
        "reward": summary.mean_reward,
        "reward_std": summary.reward_std,
        "loss": summary.loss,
        "grad_norm": summary.grad_norm,
        "learning_rate": config.learning_rate,
        "response_length": summary.mean_response_tokens,
    }
    try:
        io.log_metrics(metrics, tags)
    except Exception as exc:  # noqa: BLE001 - metrics must not fail a finished step
        _log(f"log_metrics failed: {type(exc).__name__}: {exc}")
    files = sorted(str(p) for p in adapter_path.iterdir() if p.is_file())
    metadata = {
        "post_training_id": config.post_training_id,
        "iteration": config.iteration,
        "adapter_name": config.adapter_out,
        "base_model": config.base_model,
        "lora_rank": config.lora_rank,
        "mean_reward": summary.mean_reward,
    }
    try:
        io.checkpoint(_adapter_module(model), files, metadata)
    except Exception as exc:  # noqa: BLE001 - the adapter is saved and served regardless
        _log(f"checkpoint failed: {type(exc).__name__}: {exc}")


def _adapter_module(model: Any) -> Any:
    """A module holding only the LoRA weights, for ``chalk.ml.checkpoint``.

    The checkpoint serializes a torch module's state dict; handing it the PEFT model would
    upload the whole base model. The PEFT adapter files go alongside as additional files.
    """
    import torch
    from peft import get_peft_model_state_dict

    module = torch.nn.Module()
    for name, tensor in get_peft_model_state_dict(model).items():
        module.register_buffer(
            name.replace(".", "__"), tensor.detach().contiguous().cpu()
        )
    return module


# -- the real side effects -----------------------------------------------------------------


def default_io() -> TrainerIO:
    return TrainerIO(
        read_revision=_read_revision,
        read_trajectory=_volume_json_reader(),
        load_tokenizer=_load_tokenizer,
        load_base_model=_load_base_model,
        commit_mount=commit_mount,
        load_adapter=load_lora_adapter,
        log_metrics=_log_metrics,
        checkpoint=_checkpoint,
        checkpoint_dir=os.environ.get("CHALK_CHECKPOINT_DIR"),
        retry_delay_seconds=TRAJECTORY_RETRY_DELAY_SECONDS,
    )


def _read_revision(revision_id: str) -> list[dict[str, Any]]:
    import chalkcompute

    return chalkcompute.DatasetClient().read(revision_id).to_pylist()


def _volume_json_reader() -> Callable[[str, str], Mapping[str, Any] | None]:
    """Reads JSON files from Chalk volumes, reusing one handle per volume."""
    import chalkcompute

    volumes: dict[str, Any] = {}

    def read(volume_name: str, path: str) -> Mapping[str, Any] | None:
        volume = volumes.get(volume_name)
        if volume is None:
            volume = chalkcompute.Volume(volume_name, create_if_missing=False)
            volumes[volume_name] = volume
        try:
            return json.loads(volume.read_file(path))
        except chalkcompute.VolumeFileNotFoundError:
            return None
        except (ValueError, chalkcompute.VolumeError) as exc:
            _log(f"cannot read {volume_name}:{path}: {type(exc).__name__}: {exc}")
            return None

    return read


def _load_tokenizer(base_model: str) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(base_model)


def _load_base_model(base_model: str, dtype: str) -> Any:
    import torch
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(
        base_model, dtype=getattr(torch, dtype), attn_implementation="sdpa"
    )


def commit_mount(path: str) -> None:
    """Commit writes under a Chalk volume mount, so other mounts can reload and see them.

    Writes to a mounted volume stay local to the container until committed; the mount
    exposes a ``.chalk-volume/commit`` control. Outside a Chalk mount this is a no-op.
    """
    control = Path(path) / ".chalk-volume" / "commit"
    if not control.exists():
        _log(f"{path} is not a Chalk volume mount; nothing to commit")
        return
    subprocess.run([str(control)], check=True, timeout=600)
    _log(f"committed {path}")


def load_lora_adapter(
    server_url: str, name: str, path: str, timeout_seconds: float
) -> None:
    """Make a vLLM policy server serve a LoRA adapter, on every replica, before returning.

    A server started with vLLM's filesystem LoRA resolver over the adapter volume loads
    ``name`` from its own mount on the first request that names it, so any number of replicas
    serve a new adapter once their mounts (which reload on an interval) show its files. One
    without the resolver serves only what ``/v1/load_lora_adapter`` loaded, so that is asked
    first: it makes a single-replica server ready, and on a resolver server it just loads one
    replica early. Then requests naming the adapter must succeed ``_SERVE_PROBE_STREAK`` times
    in a row; the load balancer spreads them, so a replica whose mount still lags answers 404
    and resets the streak until it catches up.

    The bearer key, when vLLM was started with one, comes from ``POLICY_SERVER_API_KEY``.
    """
    deadline = time.monotonic() + timeout_seconds
    _post_until_ok(
        f"{server_url.rstrip('/')}/v1/load_lora_adapter",
        {"lora_name": name, "lora_path": path, "load_inplace": True},
        what=f"load {name} from {path}",
        deadline=deadline,
    )
    streak = 0
    delay = 2.0
    while streak < _SERVE_PROBE_STREAK:
        status, detail = _post_json(
            f"{server_url.rstrip('/')}/v1/chat/completions",
            {
                "model": name,
                "max_tokens": 1,
                "messages": [{"role": "user", "content": "hi"}],
            },
            timeout=120,
        )
        if status == 200:
            streak += 1
            continue
        if status in (401, 403):
            raise RuntimeError(f"policy server refused the key: {detail}")
        streak = 0
        if time.monotonic() + delay > deadline:
            raise RuntimeError(f"policy server does not serve {name}: {detail}")
        _log(f"{name} not served everywhere yet ({detail}); retrying in {delay:.0f}s")
        time.sleep(delay)
        delay = min(delay * 1.5, 15.0)
    _log(f"{name} answered {_SERVE_PROBE_STREAK} requests in a row")


# Consecutive successful requests naming a new adapter that count as every replica serving
# it. Requests are spread over the replicas, so this should exceed the replica count.
_SERVE_PROBE_STREAK = 8


def _post_json(url: str, payload: dict[str, Any], timeout: float) -> tuple[int, str]:
    """POST JSON; the HTTP status (0 when the server is unreachable) and a short detail."""
    headers = {"Content-Type": "application/json"}
    api_key = os.environ.get("POLICY_SERVER_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers=headers, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read()[:500].decode(errors="replace")
    except urllib.error.HTTPError as exc:
        return exc.code, f"HTTP {exc.code}: {exc.read()[:500].decode(errors='replace')}"
    except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
        return 0, f"{type(exc).__name__}: {exc}"


def _post_until_ok(
    url: str, payload: dict[str, Any], what: str, deadline: float
) -> None:
    """POST until 200, retrying while the server is restarting or its mount lags.

    vLLM answers 400/404 while the adapter's files are not yet visible in its mount of the
    adapter volume, and 5xx or a refused connection while it restarts.
    """
    delay = 5.0
    while True:
        status, detail = _post_json(url, payload, timeout=300)
        if status == 200:
            _log(f"{what}: {detail}")
            return
        if status in (401, 403):
            raise RuntimeError(f"policy server refused the key: {detail}")
        if time.monotonic() + delay > deadline:
            raise RuntimeError(f"policy server could not {what}: {detail}")
        _log(f"could not {what} yet ({detail}); retrying in {delay:.0f}s")
        time.sleep(delay)
        delay = min(delay * 1.5, 30.0)


def _log_metrics(metrics: Mapping[str, float], tags: Mapping[str, str]) -> None:
    from chalk.ml.chalk_train import log_metrics

    log_metrics(metrics, tags=tags)


def _checkpoint(module: Any, files: list[str], metadata: dict[str, Any]) -> None:
    from chalk.ml.chalk_train import checkpoint

    checkpoint(module, additional_files=files, metadata=metadata)
