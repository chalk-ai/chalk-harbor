# Inactive CPU trainer checkpoint primitive

`chalk_harbor/post_training/full_checkpoint.py` is uncalled and unexported. The current
LoRA trainer/configuration is unchanged. This module cannot yet be used by a GPU
full-weight trainer: it explicitly refuses available CUDA, MPS, XPU or generic Torch
accelerators, and non-CPU tensors. It captures only the supported CPU state described
below. FULL_WEIGHTS admission must remain disabled.

`save_checkpoint(path, identity=..., model=..., optimizer=..., scheduler=...)` writes a
new exclusive directory and returns its canonical manifest SHA256. The payload holds
the complete model state_dict (parameters and buffers), optimizer state_dict, scheduler
state_dict, per-module training flags, parameter requires_grad flags and global Python,
NumPy and Torch CPU RNG states. The manifest binds the exact required operation/input,
configuration, provenance and trainer-step identity, object classes/parameter ordering,
Torch version and state-file size/SHA256. Integer zero is preserved for iterations,
steps, fences, parent versions and input snapshots.

Call only at a quiescent completed-step boundary. There is no in-flight gradient/AMP
or gradient-accumulation state, data-loader/sampler position, custom Generator capture,
external state or custom backend support. Clear gradients before the resumed next step.
The caller must construct the same architecture and hyperparameters from the bound
configuration. This prototype serializes the entire state in memory; it makes no
full-model memory-fit or performance claim. No cross-platform bitwise reproducibility
is promised.

The directory creation admits exactly one local writer. State bytes are fsynced before
a pending manifest is fsynced and renamed to `COMPLETE.json`; the directory is fsynced
before returning. Interrupted/incomplete and already-complete paths are never rewritten,
even for identical identities. A retry must restore the original completed bytes using
the original trusted digest or fail closed. It must not advance training merely because
the save reply was lost. Failed directories retain diagnostic files for the caller's
explicit handling. Local fsync behavior depends on the filesystem; the marker proves
only local completeness, not a durable ChalkFS snapshot or operation result.

`restore_checkpoint(..., expected=identity, manifest_sha256=digest, ...)` requires the
exact expected identity and trusted manifest digest. It checks all filenames, regular-file
status, canonical manifest bytes, payload hash/size, topology, Torch version, CPU payload and destination state,
RNG and model keys/shapes/dtypes before calling any state loading hook or changing global
RNG. It uses `torch.load(weights_only=True)`. Restore into fresh disposable objects:
arbitrary application `load_state_dict` hooks can still fail; object loading is not an
in-memory transaction. Discard all objects after a loading exception. Hashes provide
integrity, not authentication; callers must authenticate expected identity/digest, and
must retain exclusive filesystem ownership during save/restore.

This module supplies no operation claim, checkpoint Volume commit/snapshot, stable
commit-id recovery, publication, cancellation fence, metadata recording, serving version
acknowledgment or exactly-once guarantee. Those boundaries need separate reviewed APIs
and actual crash/race evidence before any trainer integration or FULL_WEIGHTS activation.

## Validation

Use an existing environment with Torch, NumPy, pytest and Ruff; the script never installs:

```sh
python scripts/check_full_checkpoint.py --python /absolute/path/to/existing/python
```

The isolated tests load the source module directly so they do not require unrelated
Harbor/PEFT/Transformers integrations. On an MPS workstation they explicitly mock only
accelerator-unavailability capability responses; the model/dropout, AdamW step,
scheduler and RNG operations are real CPU Torch. Separate tests prove accelerator
availability fails closed. Tests compare a resumed next step exactly, verify every
identity field, zero presence, corrupt/missing/nonregular files, concurrent exclusive
writers, immutable retries and actual child-process exit after state/pending-manifest
writes and after completion before the response. Each child is bounded and reaped. Actual meta-device target regressions cover model
parameters, nonpersistent buffers, optimizer and scheduler state: refusal calls no load
hook and preserves all supplied object state and global RNG.

Local evidence: 44 tests passed using existing Python 3.11.14 / Torch 2.15.0. Ruff lint
and formatting pass. The project requires Python >=3.12,<3.15; this is not a
supported-Python-version test result. No existing supported-version Torch environment
was available. Existing LoRA integration tests require missing Harbor/PEFT/Transformers
packages and were not run; no installation, image/model downloads, external calls or
compute allocations were made for this primitive.
