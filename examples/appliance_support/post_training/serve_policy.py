#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = ["chalkcompute>=2.13.3"]
# ///
"""Serve the post-training policy with vLLM, and route `<prefix>/<model>` to it.

    ./serve_policy.py --dry-run            # print what would be created
    ./serve_policy.py                      # create both (needs a GPU node pool)

Run once per environment. It creates:

1. the adapter volume (``--adapter-volume``) the training runs write LoRA adapters to;
2. a scaling group of ``--replicas`` servers running ``vllm/vllm-openai`` with the base model,
   the hermes tool-call parser the Larkspur agent's function calling needs, and the adapter
   volume mounted at ``/chalk/adapters``. vLLM's filesystem LoRA resolver points at that mount,
   so every replica loads ``adapter-<id>-<k>`` from it the first time a request names it;
   ``/v1/load_lora_adapter`` also works (``VLLM_ALLOW_RUNTIME_LORA_UPDATING``);
3. a Model Gateway provider connection of kind ``vllm`` with prefix ``--prefix``, so the
   evaluation's task reaches the policy through Chalk's AI router as
   ``<prefix>/Qwen/Qwen3-4B-Instruct-2507`` and, once trained, ``<prefix>/adapter-<id>-<k>``.

The volume mount in each replica sees new commits only after a reload, so the entrypoint
reloads it every 15 s in the background; the trainer retries until requests naming its adapter
succeed repeatedly, i.e. until every replica's mount shows it.

``--host`` runs on host-class GPUs (a hypervisor host pool, e.g. external A100s): it adds
Chalk workload identity (volume mounts there need it), Hugging Face egress, loopback gloo
(the host's hostname is longer than ``HOST_NAME_MAX``), the driver's libcuda for triton, and
eager mode unless ``--cuda-graphs``, which captures CUDA graphs without torch.compile (compile
adds minutes of startup under gVisor for no throughput gain there).

Prints the policy server URL to give StartEvaluationPostTraining as ``policy_server_url``.

Auth: ``--api-key-env NAME`` passes the value of local env var NAME to vLLM as
``VLLM_API_KEY`` (stored as a Chalk secret) and to the provider connection. Without it the
endpoint is open to anyone with its URL. With it, training runs need the same key as
``POLICY_SERVER_API_KEY`` to load adapters.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
import time
import urllib.error
import urllib.request
from typing import Any

BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
VLLM_IMAGE = "vllm/vllm-openai:v0.11.0"
ADAPTER_MOUNT = "/chalk/adapters"
PORT = 8000
RELOAD_SECONDS = 15


def vllm_command(args: argparse.Namespace) -> list[str]:
    """The server's entrypoint: a volume-reload loop beside vLLM's OpenAI server."""
    serve = [
        "python3", "-m", "vllm.entrypoints.openai.api_server",
        "--model", args.base_model, "--served-model-name", args.base_model,
        "--port", str(PORT), "--dtype", "bfloat16",
        "--max-model-len", str(args.max_model_len),
        "--gpu-memory-utilization", "0.90",
        *((["--compilation-config", '{"level": 0, "cudagraph_mode": "FULL_DECODE_ONLY"}'] if args.cuda_graphs else ["--enforce-eager"]) if args.host else []), "--enable-lora", "--max-lora-rank", str(args.max_lora_rank),
        "--max-loras", str(args.max_loras),
        "--enable-auto-tool-choice", "--tool-call-parser", "hermes",
    ]  # fmt: skip
    reload = f"{ADAPTER_MOUNT}/.chalk-volume/reload"
    script = " ".join(
        [
            f"(while true; do [ -x {reload} ] && {reload} >/dev/null 2>&1;",
            f"sleep {RELOAD_SECONDS}; done) &",
            "exec",
            shlex.join(serve),
        ]
    )
    return ["bash", "-c", script]


def _allow_hf_egress(hosts: list[str]) -> None:
    """ScalingGroup takes no network_policy; host containers get no egress without one."""
    import chalkcompute._scaling_group as sgmod
    from chalkcompute import NetworkPolicy

    base = sgmod.ContainerSpec

    class _Spec(base):  # type: ignore[misc,valid-type]
        def __init__(self, *a: Any, **kw: Any) -> None:
            kw.setdefault("network_policy", NetworkPolicy(allowed_hosts=hosts))
            super().__init__(*a, **kw)

    sgmod.ContainerSpec = _Spec  # type: ignore[assignment]


def scaling_group(args: argparse.Namespace) -> Any:
    from chalkcompute import ComputeClass, ReadinessProbe, ScalingGroup, Secret

    env: dict[str, Any] = {
        "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "True",
        # Every replica loads an adapter from its own mount of the adapter volume the first time a
        # request names it, so new adapters need no per-replica /v1/load_lora_adapter call.
        "VLLM_PLUGINS": "lora_filesystem_resolver",
        "VLLM_LORA_RESOLVER_CACHE_DIR": ADAPTER_MOUNT,
        **(
            {
                # Host containers have a hostname longer than HOST_NAME_MAX, which breaks
                # gloo's hostname lookup; and triton misses the injected libcuda.
                "GLOO_SOCKET_IFNAME": "lo",
                "VLLM_HOST_IP": "127.0.0.1",
                "TRITON_LIBCUDA_PATH": "/usr/lib/x86_64-linux-gnu",
            }
            if args.host
            else {}
        ),
        "HF_HOME": "/tmp/hf",
        # GKE mounts the driver libraries under /usr/local/nvidia; the vLLM image's own
        # LD_LIBRARY_PATH omits them, so vLLM would find no GPU there.
        "LD_LIBRARY_PATH": "/usr/local/nvidia/lib64:/usr/local/cuda/lib64",
    }
    if args.api_key_env:
        env["VLLM_API_KEY"] = Secret.from_local_env(args.api_key_env)
    if args.host:
        _allow_hf_egress(["huggingface.co", "*.huggingface.co", "hf.co", "*.hf.co"])
    return ScalingGroup(
        image=args.vllm_image,
        name=args.name,
        gpu=args.gpu,
        cpu="8",
        memory=args.memory,
        port=PORT,
        env=env,
        volumes=[(args.adapter_volume, ADAPTER_MOUNT)],
        entrypoint=vllm_command(args),
        readiness_probe=ReadinessProbe.http("/health"),
        # The filesystem LoRA resolver makes any replica serve an adapter from the volume.
        min_replicas=args.replicas,
        max_replicas=args.replicas,
        compute_class=ComputeClass.HOST if args.host else None,
        # An external host mounts versioned volumes with the container's own Chalk identity.
        chalk_identity=args.host,
    )


def provider_connection_request(
    args: argparse.Namespace, server_url: str
) -> dict[str, Any]:
    """CreateProviderConnection, in the proto's JSON form."""
    request: dict[str, Any] = {
        "name": args.connection_name,
        "providerKind": "vllm",
        "baseUrl": f"{server_url.rstrip('/')}/v1",
        # The router stores a prefix as one segment with its trailing slash.
        "prefix": f"{args.prefix.rstrip('/')}/",
        "exposure": "EXPOSURE_POLICY_DYNAMIC",
        "routingEnabled": True,
    }
    if args.api_key_env:
        request["apiKey"] = os.environ[args.api_key_env]
    return request


def connect_call(method: str, body: dict[str, Any]) -> dict[str, Any]:
    """POST one Connect RPC as JSON to the Chalk API, with this machine's Chalk login."""
    from chalkcompute import ConnectClient

    client = ConnectClient()
    client._ensure_auth()
    request = urllib.request.Request(
        f"{client._api_server.rstrip('/')}/{method}",
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


def wait_healthy(url: str, timeout: float) -> None:
    # Ready means the container started; vLLM still downloads and loads the weights.
    deadline = time.monotonic() + timeout
    while True:
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=10) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            pass
        if time.monotonic() > deadline:
            raise SystemExit(f"{url}/health not ready after {timeout:.0f}s")
        time.sleep(10)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--name", default="larkspur-policy")
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--vllm-image", default=VLLM_IMAGE)
    parser.add_argument("--gpu", default="nvidia-l40s:1")
    parser.add_argument("--adapter-volume", default="post-training-adapters")
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument(
        "--max-lora-rank", type=int, default=64, help="at least the lora_rank you train"
    )
    parser.add_argument("--max-loras", type=int, default=4)
    parser.add_argument("--prefix", default="posttrain")
    parser.add_argument("--connection-name", default="larkspur-policy")
    parser.add_argument("--api-key-env", default=None)
    parser.add_argument(
        "--host", action="store_true", help="run on a host-class (hypervisor) GPU host"
    )
    parser.add_argument("--memory", default="48Gi")
    parser.add_argument(
        "--replicas", type=int, default=1, help="each replica takes one --gpu"
    )
    parser.add_argument(
        "--cuda-graphs",
        action="store_true",
        help="host class: CUDA graphs without torch.compile",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if args.dry_run:
        print(json.dumps({"entrypoint": vllm_command(args)}, indent=2))
        print(
            json.dumps(
                provider_connection_request(args, "<scaling group url>"), indent=2
            )
        )
        return 0

    import chalkcompute

    with chalkcompute.VolumeClient.from_env() as volumes:
        if not volumes.exists(args.adapter_volume):
            volumes.create(args.adapter_volume)
            print(f"created volume {args.adapter_volume}", flush=True)
    group = scaling_group(args).deploy(build_timeout=1200, ready_timeout=1800)
    url = (group.web_url or "").rstrip("/")
    print(f"scaling group {group.id}: {url}", flush=True)
    wait_healthy(url, 1800)
    response = connect_call(
        "chalk.router.v1.ProviderConnectionService/CreateProviderConnection",
        provider_connection_request(args, url),
    )
    connection = response.get("connection", {})
    print(f"provider connection {connection.get('id')} prefix {args.prefix}/")
    print(f"policy_server_url: {url}")
    print(f"policy model at iteration 0: {args.prefix}/{args.base_model}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
