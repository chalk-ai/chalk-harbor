"""The GRPO step and a whole trainer iteration, on CPU with a tiny random Qwen3."""

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
pytest.importorskip("transformers")

from chalk_harbor.post_training.config import TrainerConfig
from chalk_harbor.post_training.grpo import (
    TrainingExample,
    accumulate_and_step,
    token_logprobs,
)
from chalk_harbor.post_training.trainer import (
    TrainerIO,
    run_iteration,
)


def _tiny_qwen(vocab_size: int) -> Any:
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(0)
    config = Qwen3Config(
        vocab_size=vocab_size,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=4096,
        tie_word_embeddings=True,
    )
    return Qwen3ForCausalLM(config).float()


def _with_lora(model: Any) -> Any:
    from peft import LoraConfig, get_peft_model

    return get_peft_model(
        model,
        LoraConfig(
            r=4, lora_alpha=8, target_modules="all-linear", task_type="CAUSAL_LM"
        ),
    )


def test_chunked_logprobs_match_the_full_softmax() -> None:
    model = _tiny_qwen(64)
    input_ids = torch.randint(0, 64, (1, 37))

    with torch.no_grad():
        chunked = token_logprobs(model, input_ids, 8)
        logits = model(input_ids=input_ids).logits[0, :-1].float()
        full = logits.log_softmax(-1).gather(-1, input_ids[0, 1:, None]).squeeze(-1)

    torch.testing.assert_close(chunked, full, rtol=1e-5, atol=1e-5)


def test_a_step_raises_the_likelihood_of_positive_advantage_tokens() -> None:
    model = _with_lora(_tiny_qwen(64))
    causal_lm = model.get_base_model()
    causal_lm.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    good = TrainingExample(
        input_ids=list(range(1, 25)), loss_mask=[0] * 8 + [1] * 16, advantage=1.0
    )
    bad = TrainingExample(
        input_ids=list(range(30, 54)), loss_mask=[0] * 8 + [1] * 16, advantage=-1.0
    )

    def generated_logprob(example: TrainingExample) -> float:
        with torch.no_grad():
            logprobs = token_logprobs(causal_lm, torch.tensor([example.input_ids]), 5)
        return float((logprobs * torch.tensor(example.loss_mask[1:])).sum())

    before = (generated_logprob(good), generated_logprob(bad))
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-2
    )
    result = accumulate_and_step(
        causal_lm,
        optimizer,
        [good, bad],
        chunk_size=5,
        max_grad_norm=1.0,
        device=torch.device("cpu"),
        log=lambda _: None,
    )
    after = (generated_logprob(good), generated_logprob(bad))

    assert result.generated_tokens == 32
    assert result.grad_norm > 0
    assert after[0] > before[0]
    assert after[1] < before[1]


# -- a whole iteration ----------------------------------------------------------------------

TASKS = ["missed-delivery-first", "price-drop-in-window", "haul-away-missed"]


def _result_rows(revision: str) -> list[dict[str, Any]]:
    """An evaluation result revision: dataset columns, output, and scorer columns."""
    rows = []
    for t, task in enumerate(TASKS):
        sample = int(revision[-1])
        rows.append(
            {
                "task_name": task,
                "run_tag": "larkspur-x",
                "output": json.dumps(
                    {"volume_path": f"posttrain/{revision}/{task}/{task}__t{sample}"}
                ),
                "output_error": None,
                # The last task scores the same in every sample: no signal.
                "larkspur-policy-compliance_value": {
                    "score": 1.0 if t == 2 else (sample % 2) * 0.5 + 0.25,
                    "metadata": "{}",
                },
                "larkspur-customer-got-irate_value": {
                    "score": 0.0 if t == 2 else float(sample % 2 == 0),
                    "metadata": None,
                },
                "cost-of-service-usd_value": 12.5,
            }
        )
    return rows


def _trajectories(
    make_trajectory: Callable[[str], dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    trajectories = {}
    for revision in ("rev0", "rev1"):
        for task in TASKS:
            sample = revision[-1]
            path = (
                f"posttrain/{revision}/{task}/{task}__t{sample}/agent/trajectory.json"
            )
            trajectories[path] = make_trajectory(f"Sample {sample} for {task}.")
    # One upload never landed.
    del trajectories[
        "posttrain/rev1/price-drop-in-window/price-drop-in-window__t1/agent/trajectory.json"
    ]
    return trajectories


class FakeIO:
    def __init__(
        self,
        tokenizer: Any,
        trajectories: Mapping[str, dict[str, Any]],
        checkpoint_dir: Path,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.trajectories = trajectories
        self.checkpoint_dir = checkpoint_dir
        self.trajectory_reads: list[tuple[str, str]] = []
        self.commits: list[str] = []
        self.loads: list[tuple[str, str, str]] = []
        self.metrics: list[tuple[dict[str, float], dict[str, str]]] = []
        self.checkpoints: list[tuple[list[str], dict[str, Any], list[str]]] = []

    def io(self) -> TrainerIO:
        return TrainerIO(
            read_revision=_result_rows,
            read_trajectory=self.read_trajectory,
            load_tokenizer=lambda _: self.tokenizer,
            load_base_model=lambda _model, _dtype: _tiny_qwen(len(self.tokenizer) + 1),
            commit_mount=self.commits.append,
            load_adapter=self.load_adapter,
            log_metrics=lambda m, t: self.metrics.append((dict(m), dict(t))),
            checkpoint=self.checkpoint,
            checkpoint_dir=str(self.checkpoint_dir),
            retry_delay_seconds=0.0,
        )

    def read_trajectory(self, volume: str, path: str) -> dict[str, Any] | None:
        self.trajectory_reads.append((volume, path))
        return self.trajectories.get(path)

    def load_adapter(self, url: str, name: str, path: str, timeout: float) -> None:
        self.loads.append((url, name, path))

    def checkpoint(
        self, module: Any, files: list[str], metadata: dict[str, Any]
    ) -> None:
        self.checkpoints.append(
            (sorted(Path(f).name for f in files), metadata, list(module.state_dict()))
        )


def _raw_config(adapter_dir: Path, iteration: int) -> dict[str, Any]:
    return {
        "post_training_id": "ptr1",
        "iteration": float(iteration),
        "base_model": "Qwen/Qwen3-4B-Instruct-2507",
        "lora_rank": 4.0,
        "learning_rate": 1e-3,
        "adapter_in": "" if iteration == 0 else f"adapter-ptr1-{iteration}",
        "adapter_out": f"adapter-ptr1-{iteration + 1}",
        "adapter_dir": str(adapter_dir),
        "policy_server_url": "http://policy:8000",
        "policy_adapter_dir": "/adapters",
        "result_dataset_revision_ids": ["rev0", "rev1"],
        "group_columns": ["task_name", "run_tag"],
        "output_column": "output",
        "reward_weights": {
            "larkspur-policy-compliance": 1.0,
            "larkspur-customer-got-irate": -0.25,
        },
        "scorer_columns": {
            "larkspur-policy-compliance": {
                "column": "larkspur-policy-compliance_value",
                "field": "score",
            },
            "larkspur-customer-got-irate": {
                "column": "larkspur-customer-got-irate_value",
                "field": "score",
            },
        },
        "torch_dtype": "float32",
        "logprob_chunk_size": 64,
    }


def _config(adapter_dir: Path, iteration: int) -> TrainerConfig:
    return TrainerConfig.from_mapping(_raw_config(adapter_dir, iteration))


def test_an_iteration_trains_saves_and_serves_the_adapter(
    tmp_path: Path,
    tiny_tokenizer: Any,
    make_trajectory: Callable[[str], dict[str, Any]],
) -> None:
    from safetensors.torch import load_file

    adapters = tmp_path / "adapters"
    fake = FakeIO(tiny_tokenizer, _trajectories(make_trajectory), tmp_path / "ckpt")

    summary = run_iteration(_config(adapters, 0), None, fake.io())

    # 2 samples x 3 tasks; the third task's rewards are equal, so 2 groups train, and one
    # of their 4 trajectories is missing.
    assert (summary.rows, summary.groups, summary.informative_groups) == (6, 3, 2)
    assert (summary.trained_sequences, summary.missing_trajectories) == (3, 1)
    assert all(volume == "harbor-traces" for volume, _ in fake.trajectory_reads)
    assert not any("haul-away-missed" in path for _, path in fake.trajectory_reads)
    assert summary.mean_reward == pytest.approx(
        ((0.25 - 0.25) * 2 + 0.75 * 2 + 1.0 * 2) / 6
    )
    assert summary.grad_norm > 0

    saved = adapters / "adapter-ptr1-1"
    config = json.loads((saved / "adapter_config.json").read_text())
    assert (config["r"], config["lora_alpha"]) == (4, 8)
    weights = load_file(str(saved / "adapter_model.safetensors"))
    # A fresh LoRA starts with B = 0, so a non-zero B is the step's update.
    assert any(
        "lora_B" in name and tensor.abs().sum() > 0 for name, tensor in weights.items()
    )
    assert fake.commits == [str(adapters), str(tmp_path / "ckpt")]
    assert fake.loads == [
        ("http://policy:8000", "adapter-ptr1-1", "/adapters/adapter-ptr1-1")
    ]
    metrics, tags = fake.metrics[0]
    assert set(metrics) == {
        "reward",
        "reward_std",
        "loss",
        "grad_norm",
        "learning_rate",
        "response_length",
    }
    assert tags == {"post_training_id": "ptr1", "iteration": "0"}
    files, metadata, buffers = fake.checkpoints[0]
    assert files == ["README.md", "adapter_config.json", "adapter_model.safetensors"]
    assert metadata["adapter_name"] == "adapter-ptr1-1"
    assert buffers and all("lora" in name for name in buffers)

    # The next iteration starts from that adapter and its optimizer state.
    summary = run_iteration(_config(adapters, 1), None, fake.io())

    assert summary.adapter_out == "adapter-ptr1-2"
    assert (adapters / "adapter-ptr1-2" / "adapter_model.safetensors").exists()
    assert fake.loads[-1][1:] == ("adapter-ptr1-2", "/adapters/adapter-ptr1-2")
    state = torch.load(
        tmp_path / "ckpt" / "ptr1" / "post_training_optimizer.pt", weights_only=False
    )
    assert state["adapter"] == "adapter-ptr1-2"
    assert all(s["step"] == 2 for s in state["state"]["state"].values())


def test_an_iteration_without_reward_spread_fails_loudly(
    tmp_path: Path,
    tiny_tokenizer: Any,
    make_trajectory: Callable[[str], dict[str, Any]],
) -> None:
    fake = FakeIO(tiny_tokenizer, _trajectories(make_trajectory), tmp_path / "ckpt")
    config = TrainerConfig.from_mapping(
        {
            **_raw_config(tmp_path / "adapters", 0),
            # Only the cost column, which is equal everywhere.
            "reward_weights": {"cost-of-service-usd": -0.01},
            "scorer_columns": {
                "cost-of-service-usd": {
                    "column": "cost-of-service-usd_value",
                    "field": "",
                }
            },
        }
    )

    with pytest.raises(RuntimeError, match="reward spread"):
        run_iteration(config, None, fake.io())
    assert fake.loads == []
