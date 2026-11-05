"""Inactive CPU checkpoint primitive; not called by the LoRA trainer.

A COMPLETE.json marker establishes only local filesystem completeness. It is not a
ChalkFS snapshot/commit, operation claim, cancellation fence, publication or readiness
acknowledgment. Callers must supply an authenticated expected identity and a trusted
manifest digest when recovering across a trust boundary. SHA256 is integrity, not auth.

Only ordinary CPU state_dict-based models/optimizers/schedulers are supported. Save
at a quiescent step boundary (no concurrent mutation); custom generators, dataloader
positions, in-flight gradients/gradient accumulation, AMP and external state must be
handled by a future trainer contract. Zero gradients before the next step. Accelerator
availability is refused because this primitive does not capture device RNG state.
Restore into fresh disposable objects: validation precedes loading, but arbitrary
user load_state_dict hooks can fail and are not transactionally rolled back.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import random
import stat
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


class CheckpointError(ValueError):
    """Incomplete, conflicting, unsupported or invalid local checkpoint."""


@dataclass(frozen=True)
class CheckpointIdentity:
    environment_id: str
    post_training_id: str
    iteration: int
    operation_id: str
    request_sha256: str
    fencing_token: int
    training_run_id: str
    weight_stream_id: str
    parent_version: int
    input_checkpoint_volume: str
    input_checkpoint_snapshot: int
    input_checkpoint_manifest_sha256: str
    input_trainer_step: int
    trainer_step: int
    trainer_config_sha256: str
    rollout_revision_ids: tuple[str, ...]
    architecture_sha256: str

    def document(self) -> dict[str, Any]:
        result = asdict(self)
        result["rollout_revision_ids"] = list(self.rollout_revision_ids)
        for key, value in result.items():
            if key in {
                "iteration",
                "fencing_token",
                "parent_version",
                "input_checkpoint_snapshot",
                "input_trainer_step",
                "trainer_step",
            }:
                if type(value) is not int or value < 0:
                    raise CheckpointError(f"{key} must be a nonnegative integer")
            elif key.endswith("sha256"):
                if (
                    not isinstance(value, str)
                    or len(value) != 64
                    or any(c not in "0123456789abcdef" for c in value)
                ):
                    raise CheckpointError(f"{key} must be a lowercase SHA256")
            elif key == "rollout_revision_ids":
                if not value or any(not isinstance(v, str) or not v for v in value):
                    raise CheckpointError("rollout revisions must be nonempty")
            elif not isinstance(value, str) or not value:
                raise CheckpointError(f"{key} must be nonempty")
        if self.trainer_step < self.input_trainer_step:
            raise CheckpointError("trainer_step precedes input_trainer_step")
        return result


def _json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _cpu_only() -> None:
    import torch

    if (
        (hasattr(torch, "accelerator") and torch.accelerator.is_available())
        or torch.cuda.is_available()
        or (hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
        or (hasattr(torch, "xpu") and torch.xpu.is_available())
    ):
        raise CheckpointError("accelerator RNG is unsupported; use a CPU-only process")


def _class_name(obj: Any) -> str:
    return f"{type(obj).__module__}.{type(obj).__qualname__}"


def _topology(model: Any, optimizer: Any, scheduler: Any) -> dict[str, Any]:
    if scheduler.optimizer is not optimizer:
        raise CheckpointError("scheduler must belong to the supplied optimizer")
    names = {id(p): name for name, p in model.named_parameters()}
    try:
        groups = [[names[id(p)] for p in g["params"]] for g in optimizer.param_groups]
    except KeyError as exc:
        raise CheckpointError(
            "optimizer contains a parameter outside the model"
        ) from exc
    return {
        "model": _class_name(model),
        "optimizer": _class_name(optimizer),
        "scheduler": _class_name(scheduler),
        "parameter_groups": groups,
        "module_names": [name for name, _ in model.named_modules()],
    }


def _rng() -> dict[str, Any]:
    import numpy as np
    import torch

    algorithm, keys, position, has_gauss, cached_gauss = np.random.get_state()
    return {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        "numpy": (algorithm, keys.tolist(), position, has_gauss, cached_gauss),
    }


def _validate_rng(rng: Any) -> None:
    import numpy as np
    import torch

    if not isinstance(rng, dict) or set(rng) != {"python", "numpy", "torch"}:
        raise CheckpointError("invalid RNG components")
    # Validate on isolated generators; do not change process RNG before all checks pass.
    random.Random().setstate(rng["python"])
    algorithm, keys, position, has_gauss, cached_gauss = rng["numpy"]
    np.random.RandomState().set_state(
        (algorithm, np.array(keys, dtype=np.uint32), position, has_gauss, cached_gauss)
    )
    torch.Generator(device="cpu").set_state(rng["torch"])


def _restore_rng(rng: dict[str, Any]) -> None:
    import numpy as np
    import torch

    random.setstate(rng["python"])
    algorithm, keys, position, has_gauss, cached_gauss = rng["numpy"]
    np.random.set_state(
        (algorithm, np.array(keys, dtype=np.uint32), position, has_gauss, cached_gauss)
    )
    torch.set_rng_state(rng["torch"])


def _check_cpu(value: Any) -> None:
    import torch

    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu":
            raise CheckpointError("only CPU tensors are supported")
    elif isinstance(value, dict):
        for item in value.values():
            _check_cpu(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _check_cpu(item)


def _check_target_cpu(model: Any, optimizer: Any, scheduler: Any) -> None:
    # Check actual destination tensors, including nonpersistent buffers and optimizer
    # parameters, as well as state_dicts. map_location only constrains saved payloads;
    # CPU load_state_dict into meta targets can warn and silently leave them meta.
    _check_cpu(list(model.parameters()))
    _check_cpu(list(model.buffers()))
    _check_cpu(optimizer.param_groups)
    _check_cpu(model.state_dict())
    _check_cpu(optimizer.state_dict())
    _check_cpu(scheduler.state_dict())


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write(path: Path, data: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _read_regular(path: Path) -> bytes:
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise CheckpointError("checkpoint component must be a regular file")
        with os.fdopen(fd, "rb", closefd=False) as stream:
            return stream.read()
    finally:
        os.close(fd)


def save_checkpoint(
    path: Path,
    *,
    identity: CheckpointIdentity,
    model: Any,
    optimizer: Any,
    scheduler: Any,
) -> str:
    """Exclusively create a new local checkpoint; existing paths always fail closed.

    No automatic retry/rewrite, including incomplete directories. Recover the original
    completed checkpoint with restore_checkpoint and its digest, or choose a new path
    under a future durable operation authority. Failed writes retain diagnostic files.
    """
    import torch

    _cpu_only()
    identity_doc = identity.document()
    _check_target_cpu(model, optimizer, scheduler)
    topology = _topology(model, optimizer, scheduler)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "rng": _rng(),
        "module_training": {
            name: module.training for name, module in model.named_modules()
        },
        "requires_grad": {
            name: param.requires_grad for name, param in model.named_parameters()
        },
    }
    _check_cpu(payload)
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    data = buffer.getvalue()
    # Refuse unsupported custom pickle state before claiming a checkpoint directory.
    # Restore never enables arbitrary pickle globals merely to load such a file.
    torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    manifest = _json(
        {
            "schema": 1,
            "identity": identity_doc,
            "topology": topology,
            "torch_version": str(torch.__version__),
            "state": {"file": "state.pt", "size": len(data), "sha256": _sha(data)},
        }
    )
    # mkdir is the exclusive writer claim. Never replace another attempt's directory.
    path.mkdir()
    _fsync_dir(path.parent)
    _write(path / "state.pt", data)
    _write(path / "COMPLETE.pending", manifest)
    os.rename(path / "COMPLETE.pending", path / "COMPLETE.json")
    _fsync_dir(path)
    return _sha(manifest)


def restore_checkpoint(
    path: Path,
    *,
    expected: CheckpointIdentity,
    manifest_sha256: str,
    model: Any,
    optimizer: Any,
    scheduler: Any,
) -> int:
    """Verify exact identity and all bytes before loading fresh supplied CPU objects."""
    import torch

    _cpu_only()
    expected_doc = expected.document()
    _check_target_cpu(model, optimizer, scheduler)
    if path.is_symlink() or not path.is_dir():
        raise CheckpointError("checkpoint must be a regular directory")
    if {p.name for p in path.iterdir()} != {"state.pt", "COMPLETE.json"}:
        raise CheckpointError("checkpoint is incomplete or has unexpected components")
    manifest_bytes = _read_regular(path / "COMPLETE.json")
    if _sha(manifest_bytes) != manifest_sha256:
        raise CheckpointError("manifest SHA256 mismatch")
    manifest = json.loads(manifest_bytes)
    if (
        _json(manifest) != manifest_bytes
        or set(manifest)
        != {
            "schema",
            "identity",
            "topology",
            "torch_version",
            "state",
        }
        or type(manifest["schema"]) is not int
        or manifest["schema"] != 1
    ):
        raise CheckpointError("invalid checkpoint manifest")
    if _json(manifest["identity"]) != _json(expected_doc):
        raise CheckpointError("checkpoint identity conflict")
    if manifest["topology"] != _topology(model, optimizer, scheduler):
        raise CheckpointError("checkpoint object topology conflict")
    if manifest["torch_version"] != str(torch.__version__):
        raise CheckpointError("checkpoint torch version conflict")
    state = manifest["state"]
    if (
        not isinstance(state, dict)
        or set(state) != {"file", "size", "sha256"}
        or state["file"] != "state.pt"
    ):
        raise CheckpointError("invalid checkpoint state descriptor")
    data = _read_regular(path / "state.pt")
    if (
        type(state["size"]) is not int
        or len(data) != state["size"]
        or _sha(data) != state["sha256"]
    ):
        raise CheckpointError("state SHA256/size mismatch")
    payload = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or set(payload) != {
        "model",
        "optimizer",
        "scheduler",
        "rng",
        "module_training",
        "requires_grad",
    }:
        raise CheckpointError("invalid checkpoint state components")
    _check_cpu(payload)
    _validate_rng(payload["rng"])
    modes = payload["module_training"]
    if (
        not isinstance(modes, dict)
        or set(modes) != dict(model.named_modules()).keys()
        or any(type(value) is not bool for value in modes.values())
    ):
        raise CheckpointError("invalid module training state")
    requires_grad = payload["requires_grad"]
    if (
        not isinstance(requires_grad, dict)
        or set(requires_grad) != dict(model.named_parameters()).keys()
        or any(type(value) is not bool for value in requires_grad.values())
    ):
        raise CheckpointError("invalid parameter requires_grad state")
    # Check model keys/shapes/dtypes before loading any live state.
    current = model.state_dict()
    if current.keys() != payload["model"].keys():
        raise CheckpointError("model state keys conflict")
    for name, value in current.items():
        saved = payload["model"][name]
        if isinstance(value, torch.Tensor) and (
            not isinstance(saved, torch.Tensor)
            or value.shape != saved.shape
            or value.dtype != saved.dtype
        ):
            raise CheckpointError(f"model state shape/dtype conflict: {name}")
    # File/manifest/RNG checks above are complete. Arbitrary state_dict hooks are a
    # separate trust boundary; callers must discard all objects on a loading error.
    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    for name, module in model.named_modules():
        module.training = modes[name]
    for name, param in model.named_parameters():
        param.requires_grad_(requires_grad[name])
    _restore_rng(payload["rng"])
    return expected.trainer_step
