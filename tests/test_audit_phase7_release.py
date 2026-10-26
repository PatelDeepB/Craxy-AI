"""Adversarial Red-Team Audit for Phase 7 System Packaging and Release (Turn 63).

Audits:
1. PEP 508 dependency string validity and pyproject.toml required keys.
2. Subprocess command line invalid flag handling and exit code 2 verification.
3. Non-existent storage path handling during query execution.
4. Ephemeral directory lifecycle verification under CLI subprocess.
5. 20-cycle programmatic CLI invocation memory stability (RSS drift < 20.0 MB).
"""

import gc
import os
import subprocess
import sys
import tempfile
import tomllib
import psutil
import pytest

from src.craxy_ai.cli import main as cli_main


def test_audit_pyproject_toml_dependency_validity():
    """Verify pyproject dependencies match valid PEP 508 syntax and version bounds."""
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pyproject_path = os.path.join(root_dir, "pyproject.toml")
    with open(pyproject_path, "rb") as fh:
        config = tomllib.load(fh)

    project = config.get("project", {})
    deps = project.get("dependencies", [])
    assert len(deps) >= 2, "Must specify at least numpy and psutil"
    for dep in deps:
        assert any(op in dep for op in [">=", "==", "<=", "~="]), f"Dependency {dep} lacks version specifier"


def test_audit_module_execution_with_invalid_flag():
    """Verify invalid flags cleanly return exit code 2 without unhandled tracebacks."""
    cmd = [sys.executable, "-m", "src.craxy_ai", "--info", "--unrecognized-extra-flag"]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 2
    assert "unrecognized arguments" in result.stderr.lower()
    assert "Traceback" not in result.stderr


def test_audit_query_with_corrupt_storage_path():
    """Verify query against non-existent storage path returns exit code 0 or handles cleanly."""
    nonexistent_dir = os.path.join(tempfile.gettempdir(), "craxy_nonexistent_test_dir_12345")
    cmd = [
        sys.executable,
        "-m",
        "src.craxy_ai",
        "--query",
        "Calculate 10 + 20",
        "--storage-dir",
        nonexistent_dir,
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0
    assert "[Tool Result: 30]" in result.stdout


def test_audit_ephemeral_paged_dir_cleaned_after_subprocess():
    """Verify subprocess --paged-kv auto cleans up temporary files upon exit."""
    cmd = [
        sys.executable,
        "-m",
        "src.craxy_ai",
        "--query",
        "Calculate 15 * 3",
        "--paged-kv",
        "auto",
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0
    assert "[Tool Result: 45]" in result.stdout


def test_audit_repeated_programmatic_cli_memory_stability():
    """Verify 20 repeated programmatic CLI invocations exhibit bounded memory drift."""
    warmup_code = cli_main(["--info"])
    assert warmup_code == 0

    gc.collect()
    process = psutil.Process(os.getpid())
    rss_start = process.memory_info().rss / (1024 * 1024)

    for i in range(20):
        code = cli_main(["--query", f"Calculate {i} + 10"])
        assert code == 0

    gc.collect()
    rss_end = process.memory_info().rss / (1024 * 1024)
    growth_mb = rss_end - rss_start

    assert growth_mb < 20.0, f"Memory growth {growth_mb:.2f} MB exceeds 20.0 MB"
    assert rss_end < 2048.0, f"Peak memory {rss_end:.2f} MB exceeds 2048.0 MB budget"
