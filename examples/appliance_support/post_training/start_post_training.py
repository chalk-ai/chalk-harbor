#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = ["chalkcompute>=2.13"]
# ///
"""Start post-training a small model on the Larkspur evaluation.

    ./start_post_training.py --evaluation-id <id> --trainer-image <uri> \\
        --policy-server-url <url from serve_policy.py> [--dry-run]
    ./start_post_training.py --status <post-training id>

Calls ``EvaluationService.StartEvaluationPostTraining``. Each of ``--iterations`` iterations
runs the evaluation ``--samples-per-row`` times with the current policy, then a training run
takes one GRPO step of a LoRA adapter on the scored trajectories and loads the new adapter
into the policy server. See the README's "Post-training" section.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from typing import Any

SERVICE = "chalk.evaluation.v1.EvaluationService"

# Scorer id -> reward weight. Policy compliance is what the business needs; cost of service
# keeps "compliant" from meaning "refund everything"; the LLM judges and the survey shape the
# conversation but are noisier, so they weigh less; an irate customer is a penalty. The raw
# dollar figure (`cost-of-service-usd`) is unbounded, so it stays out: `larkspur-cost-of-service`
# already maps it into [0, 1].
REWARD_WEIGHTS = {
    "larkspur-policy-compliance": 1.0,
    "larkspur-cost-of-service": 0.5,
    "larkspur-agent-claims-accurate": 0.2,
    "larkspur-customer-satisfied": 0.15,
    "larkspur-csat-survey": 0.1,
    "larkspur-customer-got-irate": -0.3,
}


def connect_call(method: str, body: dict[str, Any]) -> dict[str, Any]:
    """POST one Connect RPC as JSON to the Chalk API, with this machine's Chalk login."""
    from chalkcompute import ConnectClient

    client = ConnectClient()
    client._ensure_auth()
    request = urllib.request.Request(
        f"{client._api_server.rstrip('/')}/{SERVICE}/{method}",
        data=json.dumps(body).encode(),
        headers={**client._headers, "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read() or b"{}")
    except urllib.error.HTTPError as exc:
        raise SystemExit(
            f"{method}: HTTP {exc.code}: {exc.read().decode(errors='replace')}"
        ) from exc


def start_request(args: argparse.Namespace) -> dict[str, Any]:
    """StartEvaluationPostTrainingRequest, in the proto's JSON form."""
    config: dict[str, Any] = {
        "baseModel": args.base_model,
        "iterations": args.iterations,
        "samplesPerRow": args.samples_per_row,
        "rewardWeights": json.loads(args.reward_weights)
        if args.reward_weights
        else REWARD_WEIGHTS,
        "learningRate": args.learning_rate,
        "loraRank": args.lora_rank,
        "trainerImage": args.trainer_image,
        "trainerGpu": args.trainer_gpu,
        "policyRouterPrefix": args.prefix,
        "policyServerUrl": args.policy_server_url,
        "adapterVolume": args.adapter_volume,
    }
    if args.model_registry_name:
        config["modelRegistryName"] = args.model_registry_name
    return {"evaluationId": args.evaluation_id, "config": config}


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--status", metavar="POST_TRAINING_ID")
    parser.add_argument("--evaluation-id")
    parser.add_argument("--trainer-image")
    parser.add_argument("--policy-server-url")
    parser.add_argument("--base-model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--samples-per-row", type=int, default=8)
    parser.add_argument(
        "--reward-weights",
        help="JSON object of scorer id -> weight (default: see source)",
    )
    # One optimizer step per iteration: LoRA wants roughly 10x full fine-tuning's learning
    # rate, and ten steps leave little room for a cautious one.
    parser.add_argument("--learning-rate", type=float, default=4e-5)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--trainer-gpu", default="nvidia-l40s:1")
    parser.add_argument("--prefix", default="posttrain")
    parser.add_argument("--adapter-volume", default="post-training-adapters")
    parser.add_argument("--model-registry-name")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if args.status:
        response = connect_call(
            "GetEvaluationPostTraining", {"postTrainingId": args.status}
        )
        print(json.dumps(response, indent=2))
        return 0
    missing = [
        flag
        for flag, value in [
            ("--evaluation-id", args.evaluation_id),
            ("--trainer-image", args.trainer_image),
            ("--policy-server-url", args.policy_server_url),
        ]
        if not value
    ]
    if missing:
        parser.error(f"required: {', '.join(missing)}")
    request = start_request(args)
    if args.dry_run:
        print(json.dumps(request, indent=2))
        return 0
    response = connect_call("StartEvaluationPostTraining", request)
    post_training = response.get("postTraining", {})
    print(json.dumps(post_training, indent=2))
    print(f"follow it with: {sys.argv[0]} --status {post_training.get('id')}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
