"""Rewards and GRPO advantages from evaluation result rows.

A row's reward is the weighted sum of its scorers' scores. Rows that ran the same dataset row
(equal values in every group column) form one GRPO group, and a row's advantage is its reward
minus its group's mean (Dr-GRPO: no division by the group's standard deviation, which would
up-weight groups whose rewards barely differ). Groups whose rewards are all equal carry no
signal and are dropped.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from chalk_harbor.post_training.config import ScorerColumn

# Rewards closer than this are treated as equal when deciding a group carries no signal.
ZERO_VARIANCE_EPSILON = 1e-8


@dataclass(frozen=True)
class ScoredRow:
    """One rollout: which group it belongs to, its reward, and the task output it produced."""

    group_key: tuple[str, ...]
    reward: float
    output: Any
    source: str  # "<revision id>#<row index>", for logs


@dataclass(frozen=True)
class Sample:
    row: ScoredRow
    advantage: float


@dataclass(frozen=True)
class GroupStats:
    rows: int
    groups: int
    # Groups with a reward spread, whose rows train.
    informative_groups: int
    skipped_rows: int
    mean_reward: float
    reward_std: float


def extract_score(value: Any, field: str) -> float | None:
    """A scorer's score from its result column, or None when the scorer produced none.

    The column holds a number, a struct (a dict once read through Arrow), or a JSON object
    string; ``field`` names the score inside the latter two.
    """
    if value is None:
        return None
    if isinstance(value, str) and field:
        try:
            value = json.loads(value)
        except ValueError:
            return None
    if field:
        if not isinstance(value, Mapping):
            return None
        value = value.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    score = float(value)
    return score if math.isfinite(score) else None


def weighted_reward(
    row: Mapping[str, Any],
    weights: Mapping[str, float],
    columns: Mapping[str, ScorerColumn],
) -> float | None:
    """Sum of weight x score over the weighted scorers; None if any of them has no score."""
    total = 0.0
    for scorer, weight in weights.items():
        spec = columns[scorer]
        score = extract_score(row.get(spec.column), spec.field)
        if score is None:
            return None
        total += weight * score
    return total


def group_key(row: Mapping[str, Any], group_columns: Sequence[str]) -> tuple[str, ...]:
    # JSON-encode so that values of any type (lists, structs, None) compare by content.
    return tuple(
        json.dumps(row.get(column), sort_keys=True, default=str)
        for column in group_columns
    )


def score_rows(
    rows: Iterable[tuple[str, Mapping[str, Any]]],
    *,
    group_columns: Sequence[str],
    output_column: str,
    weights: Mapping[str, float],
    columns: Mapping[str, ScorerColumn],
) -> tuple[list[ScoredRow], int]:
    """Score each ``(source, row)``; returns the scored rows and how many were skipped."""
    scored: list[ScoredRow] = []
    skipped = 0
    for source, row in rows:
        reward = weighted_reward(row, weights, columns)
        if reward is None or row.get(output_column) is None:
            skipped += 1
            continue
        scored.append(
            ScoredRow(
                group_key=group_key(row, group_columns),
                reward=reward,
                output=row[output_column],
                source=source,
            )
        )
    return scored, skipped


def group_advantages(rows: Sequence[ScoredRow]) -> list[Sample]:
    """Dr-GRPO advantages: reward minus the group mean, for groups whose rewards differ.

    Single-row groups and groups with equal rewards are dropped: their advantages are all
    zero, so they would cost a forward and backward pass for no gradient.
    """
    groups: dict[tuple[str, ...], list[ScoredRow]] = {}
    for row in rows:
        groups.setdefault(row.group_key, []).append(row)
    samples: list[Sample] = []
    for members in groups.values():
        if len(members) < 2:
            continue
        mean = sum(r.reward for r in members) / len(members)
        if max(abs(r.reward - mean) for r in members) <= ZERO_VARIANCE_EPSILON:
            continue
        samples.extend(Sample(row=r, advantage=r.reward - mean) for r in members)
    return samples


def best_rollouts(rows: Sequence[ScoredRow], min_reward: float | None) -> list[Sample]:
    """Rejection sampling for SFT: the rollouts worth imitating, each with advantage 1.

    With ``min_reward`` set, every rollout scoring at least that much is kept. Without it,
    each group keeps its best-scoring rollouts (all of them when tied), which needs no
    knowledge of the reward's scale. An advantage of 1 makes the policy-gradient loss the
    plain mean negative log-likelihood of the kept tokens, so the GRPO step trains SFT.
    """
    if min_reward is not None:
        return [Sample(row=r, advantage=1.0) for r in rows if r.reward >= min_reward]
    groups: dict[tuple[str, ...], list[ScoredRow]] = {}
    for row in rows:
        groups.setdefault(row.group_key, []).append(row)
    samples: list[Sample] = []
    for members in groups.values():
        best = max(r.reward for r in members)
        samples.extend(
            Sample(row=r, advantage=1.0)
            for r in members
            if best - r.reward <= ZERO_VARIANCE_EPSILON
        )
    return samples


def group_stats(
    rows: Sequence[ScoredRow], samples: Sequence[Sample], skipped: int
) -> GroupStats:
    rewards = [r.reward for r in rows]
    mean = sum(rewards) / len(rewards) if rewards else 0.0
    variance = sum((r - mean) ** 2 for r in rewards) / len(rewards) if rewards else 0.0
    return GroupStats(
        rows=len(rows),
        groups=len({r.group_key for r in rows}),
        informative_groups=len({s.row.group_key for s in samples}),
        skipped_rows=skipped,
        mean_reward=mean,
        reward_std=math.sqrt(variance),
    )
