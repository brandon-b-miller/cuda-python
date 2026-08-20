# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""On-disk materialization of JIT source, so debuggers can display it.

NVRTC stamps the program name it is handed into the DWARF line table as the
source file's path. A program compiled from an in-memory string has no file
behind that path, so cuda-gdb resolves kernel frames and line numbers but
cannot show source text -- it reports ``No such file or directory`` for a path
like ``$CWD/default_program`` (gh-2385). Writing the source out and naming
*that* path is what closes the gap; the name alone never did.

Scope: this targets the reported case, where cuda-gdb launches the Python
process and breaks in a kernel while that process is still alive. Sources
therefore live in one process-scoped temporary directory and are removed when
the interpreter exits. Two consequences are deliberate and not yet handled:

* A cubin served from :class:`~cuda.core.utils.FileStreamProgramCache` in a
  later process skips compilation, so nothing recreates its source, and that
  path is unresolvable again.
* GPU core dumps are inspected after the process is gone, so the source will
  not be there either.

Both would need a store whose lifetime is at least the cubin's -- a persistent
content-addressed directory alongside the program cache. That is a larger
change and is left for later.

Entries are named by a digest of the source, which makes them immutable within
the directory: the filename determines the contents, so re-compiling identical
source reuses one file instead of accumulating copies.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import tempfile
import threading
import warnings
from pathlib import Path

# Enough digest to make collisions irrelevant while keeping the path readable
# in debugger output, where it is shown on every frame.
_DIGEST_CHARS = 32

DISABLE_ENV_VAR = "CUDA_CORE_JIT_SOURCE_CACHE"
_FALSEY = frozenset({"0", "false", "off", "no"})

# Held for the life of the interpreter on purpose. TemporaryDirectory removes
# its tree from a finalizer, so keeping the object in a module global is what
# decouples a source file's lifetime from the Program that compiled it -- a
# Program is often collectable immediately after `.compile()`, long before the
# kernel it produced is launched, let alone debugged.
_source_dir_lock = threading.Lock()
_source_dir: tempfile.TemporaryDirectory | None = None


def is_enabled() -> bool:
    """False when the user has opted out of writing source to disk."""
    value = os.environ.get(DISABLE_ENV_VAR)
    return value is None or value.strip().lower() not in _FALSEY


def source_dir() -> Path:
    """The process-scoped directory holding materialized JIT source."""
    global _source_dir
    with _source_dir_lock:
        if _source_dir is None:
            _source_dir = tempfile.TemporaryDirectory(prefix="cuda-core-jit-")
        return Path(_source_dir.name)


def materialize(code: bytes, suffix: str = ".cu") -> str | None:
    """Persist ``code`` and return its absolute path, or None on refusal.

    Returning None is not an error: the caller keeps whatever program name it
    already had, which reproduces the pre-existing behavior rather than failing
    a compile that would otherwise have succeeded. Writing source is a
    debuggability improvement, never a precondition for compiling.
    """
    if not is_enabled():
        return None

    try:
        target = source_dir() / f"{hashlib.sha256(code).hexdigest()[:_DIGEST_CHARS]}{suffix}"
        if not target.exists():
            _write_atomically(target, code)
        return os.fspath(target)
    except OSError:
        # A read-only or full filesystem, or a sandbox that forbids the temp
        # directory -- none of which should break a compile.
        return None


def reconcile_named_source(name: str, code: bytes) -> None:
    """Make a caller-supplied program name usable as a DWARF source path.

    NVRTC resolves the name relative to the process cwd, which is also what it
    records in the DWARF, so that absolute path is what a debugger will open.

    The policy is create-but-never-clobber. When nothing is there the source is
    written, because the caller named a path and asked for debug info and there
    is nothing to lose. When something *is* there it is left completely alone,
    even if the contents disagree -- it may be a file the caller is editing.
    A disagreement is warned about instead, because it is the one failure mode
    that is otherwise silent: a debugger will happily display whatever that
    file says, attributing the wrong lines to the compiled code.

    Unlike :func:`materialize`, anything written here persists: it is in a
    location the caller chose, so removing it later is not ours to do.
    """
    if not is_enabled():
        return

    target = Path(os.path.abspath(name))
    try:
        if not target.exists():
            _write_atomically(target, code)
            return
        on_disk = target.read_bytes()
    except OSError:
        return

    if on_disk.splitlines() != code.splitlines():
        warnings.warn(
            f"ProgramOptions.name={name!r} resolves to {os.fspath(target)!r}, which already exists "
            f"with different contents than the compiled source. NVRTC records that path in the "
            f"debug info, so a debugger will show that file's lines for this program. It was left "
            f"untouched; leave name unset to let cuda.core manage the copy instead.",
            UserWarning,
            stacklevel=4,
        )


def _write_atomically(target: Path, data: bytes) -> None:
    """Publish ``data`` at ``target`` without exposing a partial file.

    Staging alongside the target keeps the rename within one filesystem, so it
    is atomic and a debugger never observes a half-written source file.
    ``mkstemp`` creates the staged file owner-only and ``os.replace`` preserves
    that, so kernel source is not world-readable.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, staged = tempfile.mkstemp(dir=target.parent, suffix=".part")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.replace(staged, target)
        except PermissionError:
            # Windows refuses the rename while another process holds the
            # target open. Content addressing means that file already holds
            # the bytes we wanted, so there is nothing to retry for.
            if not target.exists():
                raise
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(staged)
        raise
