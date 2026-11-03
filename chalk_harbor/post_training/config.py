"""The training run ``config`` a post-training workflow hands the trainer.

The workflow passes it as a ``google.protobuf.Struct``, so every number arrives as a float
and every list as a list; ``TrainerConfig.from_mapping`` validates the keys and restores the
types. The contract keys are required. The trainer knobs after them are optional, so the
workflow can stay on the contract while a run tunes the trainer.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


class TrainerConfigError(ValueError):
    """The training run's config is missing a key or holds a value of the wrong type."""


@dataclass(frozen=True)
class ScorerColumn:
    """Where one scorer's score sits in an evaluation result row.

    ``column`` is the result column (``<scorer>_value`` for a function scorer). ``field`` is
    the key inside it when the column holds a struct or a JSON object, such as ``score``; an
    empty ``field`` means the column holds the number itself.
    """

    column: str
    field: str


# Trainer knobs that are not part of the workflow contract, and their values when absent.
OPTIONAL_KEYS: dict[str, Any] = {
    # LoRA scale is alpha / rank.
    "lora_alpha": None,  # None: twice the rank
    "max_seq_len": 32768,
    # Rows of logits materialized at once when scoring a sequence; the vocabulary is ~152k
    # wide, so a full 32k-token sequence of fp32 logits would need ~20 GB.
    "logprob_chunk_size": 2048,
    "max_grad_norm": 1.0,
    "adam_beta1": 0.9,
    "adam_beta2": 0.99,
    "weight_decay": 0.0,
    # Where a row's trajectory is: `<trajectory_path_field>` in the output JSON names a
    # directory in `trajectory_volume`, holding `trajectory_file`.
    "trajectory_volume": "harbor-traces",
    "trajectory_path_field": "volume_path",
    "trajectory_file": "agent/trajectory.json",
    # How long to keep retrying /v1/load_lora_adapter while the policy server's mount of the
    # adapter volume catches up with the commit.
    "load_timeout_seconds": 900.0,
    "torch_dtype": "bfloat16",
}


@dataclass(frozen=True)
class TrainerConfig:
    post_training_id: str
    iteration: int
    base_model: str
    lora_rank: int
    learning_rate: float
    adapter_in: str
    adapter_out: str
    adapter_dir: str
    policy_server_url: str
    policy_adapter_dir: str
    result_dataset_revision_ids: tuple[str, ...]
    group_columns: tuple[str, ...]
    output_column: str
    reward_weights: dict[str, float]
    scorer_columns: dict[str, ScorerColumn]

    lora_alpha: int
    max_seq_len: int
    logprob_chunk_size: int
    max_grad_norm: float
    adam_beta1: float
    adam_beta2: float
    weight_decay: float
    trajectory_volume: str
    trajectory_path_field: str
    trajectory_file: str
    load_timeout_seconds: float
    torch_dtype: str

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> TrainerConfig:
        if not isinstance(raw, Mapping):
            raise TrainerConfigError(
                f"config must be a mapping, got {type(raw).__name__}"
            )
        reward_weights = {
            str(k): _float(v, f"reward_weights[{k}]")
            for k, v in _mapping(raw, "reward_weights").items()
        }
        if not reward_weights:
            raise TrainerConfigError("reward_weights is empty")
        scorer_columns = {
            scorer: _scorer_column(scorer, _mapping(raw, "scorer_columns").get(scorer))
            for scorer in reward_weights
        }
        lora_rank = _int(_required(raw, "lora_rank"), "lora_rank")
        lora_alpha = _optional(raw, "lora_alpha")
        return cls(
            post_training_id=_str(raw, "post_training_id"),
            iteration=_int(_required(raw, "iteration"), "iteration"),
            base_model=_str(raw, "base_model"),
            lora_rank=lora_rank,
            learning_rate=_float(_required(raw, "learning_rate"), "learning_rate"),
            adapter_in=_str_or_empty(raw, "adapter_in"),
            adapter_out=_str(raw, "adapter_out"),
            adapter_dir=_str(raw, "adapter_dir"),
            policy_server_url=_str(raw, "policy_server_url").rstrip("/"),
            policy_adapter_dir=_str(raw, "policy_adapter_dir"),
            result_dataset_revision_ids=_strings(raw, "result_dataset_revision_ids"),
            group_columns=_strings(raw, "group_columns"),
            output_column=_str(raw, "output_column"),
            reward_weights=reward_weights,
            scorer_columns=scorer_columns,
            lora_alpha=2 * lora_rank
            if lora_alpha is None
            else _int(lora_alpha, "lora_alpha"),
            max_seq_len=_int(_optional(raw, "max_seq_len"), "max_seq_len"),
            logprob_chunk_size=_int(
                _optional(raw, "logprob_chunk_size"), "logprob_chunk_size"
            ),
            max_grad_norm=_float(_optional(raw, "max_grad_norm"), "max_grad_norm"),
            adam_beta1=_float(_optional(raw, "adam_beta1"), "adam_beta1"),
            adam_beta2=_float(_optional(raw, "adam_beta2"), "adam_beta2"),
            weight_decay=_float(_optional(raw, "weight_decay"), "weight_decay"),
            trajectory_volume=str(_optional(raw, "trajectory_volume")),
            trajectory_path_field=str(_optional(raw, "trajectory_path_field")),
            trajectory_file=str(_optional(raw, "trajectory_file")),
            load_timeout_seconds=_float(
                _optional(raw, "load_timeout_seconds"), "load_timeout_seconds"
            ),
            torch_dtype=str(_optional(raw, "torch_dtype")),
        )


def _required(raw: Mapping[str, Any], key: str) -> Any:
    if key not in raw or raw[key] is None:
        raise TrainerConfigError(f"config is missing {key!r}")
    return raw[key]


def _optional(raw: Mapping[str, Any], key: str) -> Any:
    value = raw.get(key)
    return OPTIONAL_KEYS[key] if value is None else value


def _str(raw: Mapping[str, Any], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise TrainerConfigError(f"{key!r} must be a non-empty string, got {value!r}")
    return value


def _str_or_empty(raw: Mapping[str, Any], key: str) -> str:
    value = raw.get(key)
    if value is None:
        return ""
    if not isinstance(value, str):
        raise TrainerConfigError(f"{key!r} must be a string, got {value!r}")
    return value


def _int(value: Any, key: str) -> int:
    # Struct carries every number as a double.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TrainerConfigError(f"{key!r} must be an integer, got {value!r}")
    if int(value) != value:
        raise TrainerConfigError(f"{key!r} must be an integer, got {value!r}")
    return int(value)


def _float(value: Any, key: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TrainerConfigError(f"{key!r} must be a number, got {value!r}")
    return float(value)


def _strings(raw: Mapping[str, Any], key: str) -> tuple[str, ...]:
    value = _required(raw, key)
    if not isinstance(value, (list, tuple)) or not all(
        isinstance(v, str) and v for v in value
    ):
        raise TrainerConfigError(f"{key!r} must be a list of strings, got {value!r}")
    return tuple(value)


def _mapping(raw: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = _required(raw, key)
    if not isinstance(value, Mapping):
        raise TrainerConfigError(f"{key!r} must be a mapping, got {value!r}")
    return value


def _scorer_column(scorer: str, spec: Any) -> ScorerColumn:
    # A weighted scorer without an explicit location uses the evaluation result's naming:
    # `<scorer>_value`, a {score, metadata} struct.
    if spec is None:
        return ScorerColumn(column=f"{scorer}_value", field="score")
    if not isinstance(spec, Mapping) or not isinstance(spec.get("column"), str):
        raise TrainerConfigError(
            f"scorer_columns[{scorer!r}] must be {{column, field}}, got {spec!r}"
        )
    field = spec.get("field") or ""
    if not isinstance(field, str):
        raise TrainerConfigError(
            f"scorer_columns[{scorer!r}].field must be a string, got {field!r}"
        )
    return ScorerColumn(column=spec["column"], field=field)
