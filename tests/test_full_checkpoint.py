"""Real CPU state/next-step tests, without importing optional Harbor integrations.

This workstation has MPS available. The cpu_only fixture emulates an accelerator-free
host capability response; all model/optimizer/RNG math still runs on actual CPU Torch.
The refusal regression separately exercises available accelerator handling.
"""

import functools
import hashlib
import importlib.util
import io
import json
import random
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

SOURCE = Path(__file__).parents[1] / "chalk_harbor/post_training/full_checkpoint.py"
spec = importlib.util.spec_from_file_location("full_checkpoint_test_subject", SOURCE)
checkpoint = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = checkpoint
spec.loader.exec_module(checkpoint)


@pytest.fixture
def cpu_only(monkeypatch):
    if hasattr(torch, "accelerator"):
        monkeypatch.setattr(torch.accelerator, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        torch.backends.mps,
        "is_available",
        functools.wraps(torch.backends.mps.is_available)(lambda: False),
    )
    if hasattr(torch, "xpu"):
        monkeypatch.setattr(torch.xpu, "is_available", lambda: False)


@pytest.fixture
def identity():
    return checkpoint.CheckpointIdentity(
        environment_id="environment",
        post_training_id="posttraining",
        iteration=0,
        operation_id="operation",
        request_sha256="1" * 64,
        fencing_token=0,
        training_run_id="run",
        weight_stream_id="stream",
        parent_version=0,
        input_checkpoint_volume="input",
        input_checkpoint_snapshot=0,
        input_checkpoint_manifest_sha256="2" * 64,
        input_trainer_step=0,
        trainer_step=1,
        trainer_config_sha256="3" * 64,
        rollout_revision_ids=("rollout#0",),
        architecture_sha256="4" * 64,
    )


def objects(width=3):
    model = torch.nn.Sequential(
        torch.nn.Linear(width, 5), torch.nn.Dropout(0.4), torch.nn.Linear(5, 1)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    return model, optimizer, scheduler


def step(model, optimizer, scheduler):
    optimizer.zero_grad()
    x = torch.randn(4, 3) * (random.random() + float(np.random.random()))
    loss = model(x).square().mean()
    loss.backward()
    optimizer.step()
    scheduler.step()
    return loss.detach().clone()


def save(path, identity, objs):
    return checkpoint.save_checkpoint(
        path, identity=identity, model=objs[0], optimizer=objs[1], scheduler=objs[2]
    )


def restore(path, identity, digest, objs):
    return checkpoint.restore_checkpoint(
        path,
        expected=identity,
        manifest_sha256=digest,
        model=objs[0],
        optimizer=objs[1],
        scheduler=objs[2],
    )


def equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            equal(a, b)
    else:
        assert left == right


def test_restore_reproduces_next_cpu_step_exactly(tmp_path, identity, cpu_only):
    random.seed(19)
    np.random.seed(20)
    torch.manual_seed(21)
    original = objects()
    step(*original)
    original[0][2].weight.requires_grad_(False)
    original[0][2].eval()  # Preserve individually configured module modes as well.
    path = tmp_path / "checkpoint"
    digest = save(path, identity, original)
    expected_loss = step(*original)
    expected_rng = checkpoint._rng()
    recovered = objects()  # Initialization deliberately consumes the global RNG.
    assert restore(path, identity, digest, recovered) == 1
    assert recovered[0][0].training is True
    assert recovered[0][2].training is False
    assert recovered[0][2].weight.requires_grad is False
    assert torch.equal(step(*recovered), expected_loss)
    for a, b in zip(original, recovered, strict=True):
        equal(a.state_dict(), b.state_dict())
    equal(checkpoint._rng(), expected_rng)


@pytest.mark.parametrize(
    "change",
    [
        {"environment_id": "other"},
        {"post_training_id": "other"},
        {"iteration": 1},
        {"operation_id": "other"},
        {"request_sha256": "5" * 64},
        {"fencing_token": 1},
        {"training_run_id": "other"},
        {"weight_stream_id": "other"},
        {"parent_version": 1},
        {"input_checkpoint_volume": "other"},
        {"input_checkpoint_snapshot": 1},
        {"input_checkpoint_manifest_sha256": "5" * 64},
        {"input_trainer_step": 1},
        {"trainer_step": 2},
        {"trainer_config_sha256": "5" * 64},
        {"rollout_revision_ids": ("different",)},
        {"architecture_sha256": "5" * 64},
    ],
)
def test_every_identity_field_is_checked_before_mutation(
    tmp_path, identity, cpu_only, change
):
    path = tmp_path / "checkpoint"
    digest = save(path, identity, objects())
    fresh = objects()
    before_model = {k: v.clone() for k, v in fresh[0].state_dict().items()}
    before_rng = checkpoint._rng()
    with pytest.raises(checkpoint.CheckpointError, match="identity conflict"):
        restore(path, replace(identity, **change), digest, fresh)
    equal(before_model, fresh[0].state_dict())
    equal(before_rng, checkpoint._rng())
    assert not fresh[1].state


@pytest.mark.parametrize(
    "change",
    [
        {"parent_version": None},
        {"input_checkpoint_snapshot": False},
        {"iteration": -1},
        {"request_sha256": "bad"},
        {"rollout_revision_ids": ()},
        {"trainer_step": -1},
    ],
)
def test_invalid_identity_fails_without_creating_directory(
    tmp_path, identity, cpu_only, change
):
    path = tmp_path / "checkpoint"
    with pytest.raises(checkpoint.CheckpointError):
        save(path, replace(identity, **change), objects())
    assert not path.exists()


def test_zero_presence_is_preserved(tmp_path, identity, cpu_only):
    path = tmp_path / "checkpoint"
    save(path, identity, objects())
    record = json.loads((path / "COMPLETE.json").read_bytes())["identity"]
    for key in (
        "iteration",
        "fencing_token",
        "parent_version",
        "input_checkpoint_snapshot",
        "input_trainer_step",
    ):
        assert type(record[key]) is int and record[key] == 0


@pytest.mark.parametrize("component", ["state.pt", "COMPLETE.json"])
def test_corrupt_or_missing_file_fails_before_mutation(
    tmp_path, identity, cpu_only, component
):
    path = tmp_path / "checkpoint"
    digest = save(path, identity, objects())
    fresh = objects()
    before = checkpoint._rng()
    (path / component).write_bytes(b"corrupted")
    with pytest.raises(checkpoint.CheckpointError, match="mismatch"):
        restore(path, identity, digest, fresh)
    equal(before, checkpoint._rng())
    assert not fresh[1].state
    (path / component).unlink()
    with pytest.raises(checkpoint.CheckpointError, match="incomplete"):
        restore(path, identity, digest, fresh)


def test_immutable_retry_never_overwrites_complete_or_incomplete(
    tmp_path, identity, cpu_only
):
    path = tmp_path / "checkpoint"
    digest = save(path, identity, objects())
    original = {p.name: p.read_bytes() for p in path.iterdir()}
    for wanted in (identity, replace(identity, operation_id="conflict")):
        with pytest.raises(FileExistsError):
            save(path, wanted, objects())
        assert original == {p.name: p.read_bytes() for p in path.iterdir()}
    restore(path, identity, digest, objects())
    partial = tmp_path / "incomplete"
    partial.mkdir()
    with pytest.raises(FileExistsError):
        save(partial, identity, objects())
    assert not list(partial.iterdir())


@pytest.mark.parametrize(
    "boundary,complete",
    [("state.pt", False), ("COMPLETE.pending", False), ("reply", True)],
)
def test_actual_process_exit_at_write_boundaries(
    tmp_path, identity, cpu_only, boundary, complete
):
    path = tmp_path / "checkpoint"
    # Exit after fsynced state, after fsynced pending manifest, or after completion
    # before returning the digest. Child is bounded and subprocess.run reaps it.
    code = """
import importlib.util,sys,os,json
from pathlib import Path
spec=importlib.util.spec_from_file_location("subject",sys.argv[1])
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
import torch,functools
if hasattr(torch,"accelerator"): torch.accelerator.is_available=lambda:False
torch.cuda.is_available=lambda:False
torch.backends.mps.is_available=functools.wraps(torch.backends.mps.is_available)(lambda:False)
if hasattr(torch,"xpu"): torch.xpu.is_available=lambda:False
identity=m.CheckpointIdentity(**json.loads(sys.argv[3]))
model=torch.nn.Linear(3,1);optimizer=torch.optim.AdamW(model.parameters());scheduler=torch.optim.lr_scheduler.StepLR(optimizer,1)
write=m._write
def crash(p,data):
 write(p,data)
 if p.name==sys.argv[4]: os._exit(73)
m._write=crash
m.save_checkpoint(Path(sys.argv[2]),identity=identity,model=model,optimizer=optimizer,scheduler=scheduler)
os._exit(73)
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            str(SOURCE),
            str(path),
            json.dumps(identity.document()),
            boundary,
        ],
        timeout=30,
        check=False,
    )
    assert result.returncode == 73
    if complete:
        manifest = (path / "COMPLETE.json").read_bytes()
        model = torch.nn.Linear(3, 1)
        optimizer = torch.optim.AdamW(model.parameters())
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1)
        restore(
            path,
            identity,
            hashlib.sha256(manifest).hexdigest(),
            (model, optimizer, scheduler),
        )
    else:
        with pytest.raises(checkpoint.CheckpointError, match="incomplete"):
            restore(path, identity, "0" * 64, objects())
    with pytest.raises(FileExistsError):
        save(path, identity, objects())


@pytest.mark.parametrize("capability", ["cuda", "mps", "xpu", "accelerator"])
def test_available_accelerator_refused(
    tmp_path, identity, cpu_only, monkeypatch, capability
):
    backend = (
        torch.backends.mps if capability == "mps" else getattr(torch, capability, None)
    )
    if backend is None:
        pytest.skip(f"Torch has no {capability} API")
    monkeypatch.setattr(backend, "is_available", lambda: True)
    with pytest.raises(checkpoint.CheckpointError, match="accelerator RNG"):
        save(tmp_path / "checkpoint", identity, objects())
    assert not (tmp_path / "checkpoint").exists()


def test_topology_and_model_shapes_checked_before_mutation(
    tmp_path, identity, cpu_only
):
    path = tmp_path / "checkpoint"
    digest = save(path, identity, objects())
    bad = objects(width=4)
    before = checkpoint._rng()
    with pytest.raises(checkpoint.CheckpointError, match="shape/dtype"):
        restore(path, identity, digest, bad)
    equal(before, checkpoint._rng())
    reversed_optimizer = torch.optim.AdamW(list(reversed(list(bad[0].parameters()))))
    reversed_scheduler = torch.optim.lr_scheduler.StepLR(reversed_optimizer, 1)
    with pytest.raises(checkpoint.CheckpointError, match="topology"):
        restore(
            path, identity, digest, (bad[0], reversed_optimizer, reversed_scheduler)
        )


def test_invalid_rng_with_valid_hash_is_rejected_before_live_state(
    tmp_path, identity, cpu_only
):
    path = tmp_path / "checkpoint"
    save(path, identity, objects())
    payload = torch.load(path / "state.pt", weights_only=True)
    payload["rng"]["python"] = (0, (), None)
    data = io.BytesIO()
    torch.save(payload, data)
    (path / "state.pt").write_bytes(data.getvalue())
    manifest = json.loads((path / "COMPLETE.json").read_bytes())
    manifest["state"].update(
        size=len(data.getvalue()), sha256=hashlib.sha256(data.getvalue()).hexdigest()
    )
    encoded = checkpoint._json(manifest)
    (path / "COMPLETE.json").write_bytes(encoded)
    fresh = objects()
    before_model = {k: v.clone() for k, v in fresh[0].state_dict().items()}
    before_rng = checkpoint._rng()
    with pytest.raises(ValueError):
        restore(path, identity, hashlib.sha256(encoded).hexdigest(), fresh)
    equal(before_model, fresh[0].state_dict())
    equal(before_rng, checkpoint._rng())


def test_component_symlink_and_unexpected_file_refused(tmp_path, identity, cpu_only):
    path = tmp_path / "checkpoint"
    digest = save(path, identity, objects())
    data = (path / "state.pt").read_bytes()
    (path / "state.pt").unlink()
    external = tmp_path / "external"
    external.write_bytes(data)
    (path / "state.pt").symlink_to(external)
    with pytest.raises(OSError):
        restore(path, identity, digest, objects())
    (path / "state.pt").unlink()
    (path / "state.pt").write_bytes(data)
    (path / "extra").touch()
    with pytest.raises(checkpoint.CheckpointError, match="unexpected"):
        restore(path, identity, digest, objects())


def test_concurrent_writers_have_one_immutable_winner(tmp_path, identity, cpu_only):
    from concurrent.futures import ThreadPoolExecutor

    path = tmp_path / "checkpoint"
    contenders = [(replace(identity, operation_id=str(i)), objects()) for i in range(2)]

    def attempt(candidate):
        wanted, objs = candidate
        try:
            return wanted, save(path, wanted, objs)
        except FileExistsError:
            return None

    with ThreadPoolExecutor(max_workers=2) as workers:
        results = list(workers.map(attempt, contenders))
    winners = [result for result in results if result is not None]
    assert len(winners) == 1
    wanted, digest = winners[0]
    restore(path, wanted, digest, objects())


def test_nonregular_state_is_refused_without_blocking(tmp_path, identity, cpu_only):
    import os

    path = tmp_path / "checkpoint"
    digest = save(path, identity, objects())
    (path / "state.pt").unlink()
    os.mkfifo(path / "state.pt")
    with pytest.raises(checkpoint.CheckpointError, match="regular file"):
        restore(path, identity, digest, objects())
