"""Shared pytest configuration.

* puts the project root on ``sys.path`` so ``import src...`` works no matter
  where pytest is invoked from;
* redirects pytest's temporary directory into the workspace (the sandboxed
  Windows temp dir is not writable, which otherwise breaks ``tmp_path``);
* registers *test-only* shims for optional GPU/CPU dependencies that cannot be
  installed on a Windows CPU box (``flash_attn``, ``scikit-learn``) -- only when
  the real package is missing, so the server always exercises the real code.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SHIMS = HERE / "_shims"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# The sandbox denies writes under the platform temp dir; pytest's tmp_path uses
# ``tempfile.gettempdir()`` lazily, so pointing it at the workspace fixes it.
_WORKSPACE_TMP = ROOT / ".pytest_work"
_WORKSPACE_TMP.mkdir(parents=True, exist_ok=True)
tempfile.tempdir = str(_WORKSPACE_TMP)


def _ensure(module_name: str, shim_dir: Path) -> None:
    try:
        __import__(module_name)
    except Exception:
        if shim_dir.is_dir() and str(shim_dir) not in sys.path:
            sys.path.insert(0, str(shim_dir))


# flash-attn requires CUDA to build; only needed to *import* rinalmo locally.
_ensure("flash_attn", SHIMS / "flash_attn_env")
# rinalmo.utils.sec_struct imports sklearn.metrics for the official SS metrics.
_ensure("sklearn", SHIMS / "sklearn_env")

# WinError 5 when pytest walks its own basetemp at session finish on this
# Windows box/AV setup; the directories are disposable, so skip the cleanup.
try:  # pragma: no cover - environment workaround, not test logic
    import _pytest.tmpdir as _pytest_tmpdir

    _pytest_tmpdir.cleanup_dead_symlinks = lambda *args, **kwargs: None
except Exception:
    pass

# ---------------------------------------------------------------------------
# Workspace-local tmp_path
# ---------------------------------------------------------------------------
# pytest's own tmpdir machinery creates/deletes directories on this sandboxed
# Windows drive and then cannot scan them again (WinError 5).  A plain fixture
# that creates unique directories with default permissions avoids that entirely.
import shutil
import uuid

import pytest


@pytest.fixture
def tmp_path():
    base = ROOT / ".pytest_work" / "cases"
    base.mkdir(parents=True, exist_ok=True)
    path = base / f"case_{uuid.uuid4().hex[:12]}"
    path.mkdir(parents=False, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


PROJECT_ROOT = ROOT
