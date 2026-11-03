"""The policy-gradient step: token log-probabilities and the GRPO loss.

The rollouts were sampled from the policy being trained (on-policy, one optimizer step per
batch), so the importance ratio is 1 and the clipped GRPO objective reduces to

    loss = -sum_i A_i * sum_{t generated in i} log pi(t) / N

with ``N`` the number of generated tokens in the whole batch. Dividing by one batch-wide
constant, rather than by each sequence's own length, keeps Dr-GRPO's property that a long
response is not down-weighted per token. Sequences are scored one at a time (a micro-batch
of one): multi-turn trajectories run to tens of thousands of tokens and differ widely in
length, so padding them together would waste most of the batch.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.checkpoint import checkpoint


@dataclass(frozen=True)
class TrainingExample:
    input_ids: list[int]
    loss_mask: list[int]
    advantage: float


@dataclass(frozen=True)
class StepResult:
    loss: float
    grad_norm: float
    generated_tokens: int
    mean_logprob: float


def _decoder(causal_lm: Any) -> Any:
    # The transformer below the LM head; its output is already final-normalized.
    return getattr(causal_lm, causal_lm.base_model_prefix)


def token_logprobs(
    causal_lm: Any, input_ids: torch.Tensor, chunk_size: int
) -> torch.Tensor:
    """log pi(input_ids[t] | input_ids[:t]) for t = 1..T-1, as a float32 tensor of T-1.

    The LM head runs over ``chunk_size`` positions at a time under activation
    checkpointing, so at most one chunk of vocabulary-wide logits is alive in the forward
    and the backward pass.
    """
    hidden = _decoder(causal_lm)(input_ids=input_ids, use_cache=False).last_hidden_state
    hidden = hidden[0, :-1]
    targets = input_ids[0, 1:]
    head = causal_lm.get_output_embeddings()

    def chunk_logprobs(h: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        logits = head(h).float()
        return torch.gather(logits.log_softmax(-1), -1, y.unsqueeze(-1)).squeeze(-1)

    pieces = []
    for start in range(0, hidden.shape[0], chunk_size):
        h = hidden[start : start + chunk_size]
        y = targets[start : start + chunk_size]
        if torch.is_grad_enabled():
            pieces.append(checkpoint(chunk_logprobs, h, y, use_reentrant=False))
        else:
            pieces.append(chunk_logprobs(h, y))
    return torch.cat(pieces)


def policy_gradient_loss(
    logprobs: torch.Tensor, target_mask: torch.Tensor, advantage: float, normalizer: int
) -> torch.Tensor:
    """One sequence's share of the batch loss; ``target_mask`` is aligned with ``logprobs``."""
    return -(advantage * (logprobs * target_mask).sum()) / normalizer


def accumulate_and_step(
    causal_lm: Any,
    optimizer: torch.optim.Optimizer,
    examples: Sequence[TrainingExample],
    *,
    chunk_size: int,
    max_grad_norm: float,
    device: torch.device,
    log: Callable[[str], None],
) -> StepResult:
    """Accumulate the loss gradient over every example, then take one optimizer step."""
    normalizer = sum(sum(e.loss_mask[1:]) for e in examples)
    if normalizer == 0:
        raise ValueError("no generated tokens to train on")
    trainable = [p for p in causal_lm.parameters() if p.requires_grad]
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    logprob_sum = 0.0
    for index, example in enumerate(examples):
        input_ids = torch.tensor([example.input_ids], dtype=torch.long, device=device)
        target_mask = torch.tensor(
            example.loss_mask[1:], dtype=torch.float32, device=device
        )
        if target_mask.sum() == 0:
            continue
        logprobs = token_logprobs(causal_lm, input_ids, chunk_size)
        loss = policy_gradient_loss(
            logprobs, target_mask, example.advantage, normalizer
        )
        loss.backward()
        total_loss += float(loss.detach())
        logprob_sum += float((logprobs.detach() * target_mask).sum())
        log(
            f"  sequence {index + 1}/{len(examples)}: {len(example.input_ids)} tokens, "
            + f"{int(target_mask.sum())} generated, advantage {example.advantage:+.4f}"
        )
    grad_norm = torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return StepResult(
        loss=total_loss,
        grad_norm=float(grad_norm),
        generated_tokens=normalizer,
        mean_logprob=logprob_sum / normalizer,
    )
