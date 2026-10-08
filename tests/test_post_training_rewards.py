import json
from typing import Any

import pytest

from chalk_harbor.post_training.config import (
    ScorerColumn,
    TrainerConfig,
    TrainerConfigError,
)
from chalk_harbor.post_training.rewards import (
    ScoredRow,
    best_rollouts,
    extract_score,
    group_advantages,
    group_stats,
    score_rows,
    weighted_reward,
)

COLUMNS = {
    "policy": ScorerColumn(column="policy_value", field="score"),
    "irate": ScorerColumn(column="irate_value", field="score"),
    "usd": ScorerColumn(column="usd_value", field=""),
}


def _row(task: str, policy: float | None, irate: float, usd: float) -> dict[str, Any]:
    return {
        "task_name": task,
        "output": json.dumps({"volume_path": f"tag/{task}/trial"}),
        "policy_value": None if policy is None else {"score": policy, "metadata": "{}"},
        "irate_value": {"score": irate, "metadata": None},
        "usd_value": usd,
    }


@pytest.mark.parametrize(
    ("value", "field", "expected"),
    [
        ({"score": 0.5, "metadata": None}, "score", 0.5),
        ('{"score": 0.25}', "score", 0.25),
        (0.75, "", 0.75),
        (None, "score", None),
        ({"score": None}, "score", None),
        ({"score": float("nan")}, "score", None),
        ("not json", "score", None),
        (True, "", None),
    ],
)
def test_extract_score(value: Any, field: str, expected: float | None) -> None:
    assert extract_score(value, field) == expected


def test_weighted_reward_sums_weight_times_score() -> None:
    row = _row("t", policy=0.5, irate=1.0, usd=0.0)
    weights = {"policy": 2.0, "irate": -1.0}

    assert weighted_reward(row, weights, COLUMNS) == pytest.approx(0.0)


def test_a_missing_weighted_score_skips_the_row() -> None:
    rows = [("r#0", _row("t", None, 0.0, 1.0)), ("r#1", _row("t", 1.0, 0.0, 1.0))]

    scored, skipped = score_rows(
        rows,
        group_columns=["task_name"],
        output_column="output",
        weights={"policy": 1.0, "irate": -0.5},
        columns=COLUMNS,
    )

    assert skipped == 1
    assert [r.source for r in scored] == ["r#1"]


def test_advantages_are_reward_minus_group_mean_without_std_scaling() -> None:
    rows = [
        ("a", _row("t1", 1.0, 0.0, 0.0)),
        ("b", _row("t1", 0.0, 0.0, 0.0)),
        ("c", _row("t1", 0.5, 0.0, 0.0)),
        ("d", _row("t2", 0.2, 0.0, 0.0)),
        ("e", _row("t2", 0.2, 0.0, 0.0)),  # equal rewards: no signal
        ("f", _row("t3", 0.9, 0.0, 0.0)),  # a group of one
    ]
    scored, skipped = score_rows(
        rows,
        group_columns=["task_name"],
        output_column="output",
        weights={"policy": 1.0},
        columns=COLUMNS,
    )

    samples = group_advantages(scored)

    assert {s.row.source: s.advantage for s in samples} == pytest.approx(
        {"a": 0.5, "b": -0.5, "c": 0.0}
    )
    stats = group_stats(scored, samples, skipped)
    assert (stats.rows, stats.groups, stats.informative_groups) == (6, 3, 1)
    assert stats.mean_reward == pytest.approx((1.0 + 0.0 + 0.5 + 0.2 + 0.2 + 0.9) / 6)


def test_groups_compare_every_group_column_by_value() -> None:
    first = {**_row("t", 1.0, 0, 0), "difficulty": ["hard"]}
    second = {**_row("t", 0.0, 0, 0), "difficulty": ["hard"]}
    other = {**_row("t", 0.5, 0, 0), "difficulty": ["easy"]}

    scored, _ = score_rows(
        [("a", first), ("b", second), ("c", other)],
        group_columns=["task_name", "difficulty"],
        output_column="output",
        weights={"policy": 1.0},
        columns=COLUMNS,
    )

    assert [s.row.source for s in group_advantages(scored)] == ["a", "b"]


def _contract() -> dict[str, Any]:
    # As the workflow sends it: a protobuf Struct, so every number is a float.
    return {
        "post_training_id": "ptr1",
        "iteration": 1.0,
        "base_model": "Qwen/Qwen3-4B-Instruct-2507",
        "lora_rank": 16.0,
        "learning_rate": 1e-5,
        "adapter_in": "adapter-ptr1-1",
        "adapter_out": "adapter-ptr1-2",
        "adapter_dir": "/chalk/adapters",
        "policy_server_url": "http://policy:8000/",
        "policy_adapter_dir": "/adapters",
        "result_dataset_revision_ids": ["r1", "r2"],
        "group_columns": ["task_name"],
        "output_column": "output",
        "reward_weights": {
            "larkspur-policy-compliance": 1.0,
            "cost-of-service-usd": -0.01,
        },
        "scorer_columns": {
            "cost-of-service-usd": {"column": "cost-of-service-usd_value", "field": ""}
        },
    }


def test_config_restores_types_and_defaults_scorer_columns() -> None:
    config = TrainerConfig.from_mapping(_contract())

    assert (config.iteration, config.lora_rank, config.lora_alpha) == (1, 16, 32)
    assert config.policy_server_url == "http://policy:8000"
    assert config.result_dataset_revision_ids == ("r1", "r2")
    assert config.scorer_columns == {
        "larkspur-policy-compliance": ScorerColumn(
            column="larkspur-policy-compliance_value", field="score"
        ),
        "cost-of-service-usd": ScorerColumn(
            column="cost-of-service-usd_value", field=""
        ),
    }
    assert config.max_seq_len == 32768


def test_config_rejects_a_missing_contract_key() -> None:
    raw = _contract()
    del raw["adapter_out"]

    with pytest.raises(TrainerConfigError, match="adapter_out"):
        TrainerConfig.from_mapping(raw)


def test_config_takes_an_empty_adapter_in_at_the_first_iteration() -> None:
    config = TrainerConfig.from_mapping({**_contract(), "adapter_in": ""})

    assert config.adapter_in == ""


def test_best_rollouts_keeps_each_groups_best_or_those_above_a_threshold() -> None:
    rows = [
        ScoredRow(group_key=("a",), reward=0.2, output="a0", source="r0#0"),
        ScoredRow(group_key=("a",), reward=0.9, output="a1", source="r1#0"),
        ScoredRow(group_key=("b",), reward=0.4, output="b0", source="r0#1"),
        ScoredRow(group_key=("b",), reward=0.4, output="b1", source="r1#1"),
    ]

    # Without a threshold every group contributes its best rollouts, ties included.
    best = best_rollouts(rows, None)
    assert [(s.row.output, s.advantage) for s in best] == [
        ("a1", 1.0),
        ("b0", 1.0),
        ("b1", 1.0),
    ]
    # With one, only rollouts at or above it, whatever their group.
    assert [s.row.output for s in best_rollouts(rows, 0.4)] == ["a1", "b0", "b1"]
    assert [s.row.output for s in best_rollouts(rows, 0.5)] == ["a1"]
    assert best_rollouts(rows, 2.0) == []
