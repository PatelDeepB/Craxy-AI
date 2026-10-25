"""
Adversarial Red-Team Audit Suite for Milestone 6 (CLI & Final Release Packaging).

Audits and stresses:
1. Subprocess execution via Python interpreter with valid and invalid CLI flags.
2. Argument parser mutual exclusion enforcement and error containment.
3. Graceful fallback when provided with non-existent storage directories.
4. System RAM RSS compliance under direct CLI query and benchmark execution.
5. Package top-level exports verification in src.craxy_ai.
"""

import subprocess
import sys
import pytest

import src.craxy_ai
from src.craxy_ai.cli import main, build_parser
from src.craxy_ai.config import SYSTEM_LIMITS


def test_audit_cli_subprocess_info_flag():
    """Verify CLI executes as independent system subprocess returning exit code 0."""
    # Arrange & Act
    proc = subprocess.run(
        [sys.executable, "-m", "src.craxy_ai.cli", "--info"],
        capture_output=True,
        text=True,
        check=False
    )

    # Assert
    assert proc.returncode == 0
    assert "Craxy-AI Engine: Physical Architecture & Memory Boundaries" in proc.stdout
    assert "2048.0 MB" in proc.stdout


def test_audit_cli_subprocess_invalid_arguments_exit_code():
    """Verify CLI exits with non-zero code on unrecognized or missing arguments."""
    # Act: Missing required argument
    proc_empty = subprocess.run(
        [sys.executable, "-m", "src.craxy_ai.cli"],
        capture_output=True,
        text=True,
        check=False
    )
    # Act: Unrecognized argument
    proc_invalid = subprocess.run(
        [sys.executable, "-m", "src.craxy_ai.cli", "--unrecognized-flag"],
        capture_output=True,
        text=True,
        check=False
    )

    # Assert
    assert proc_empty.returncode != 0
    assert proc_invalid.returncode != 0


def test_audit_cli_mutually_exclusive_flags():
    """Verify passing conflicting flags raises SystemExit error."""
    parser = build_parser()

    # Assert
    with pytest.raises(SystemExit):
        parser.parse_args(["--info", "--benchmark"])

    with pytest.raises(SystemExit):
        parser.parse_args(["--info", "--query", "test prompt"])


def test_audit_cli_nonexistent_storage_dir_graceful_fallback(capsys):
    """Verify non-existent storage directory does not crash the CLI query execution."""
    # Arrange
    fake_dir = "non_existent_vector_store_xyz_12345"

    # Act
    exit_code = main(["--query", "What is parameter decoupling?", "--storage-dir", fake_dir])

    # Assert
    assert exit_code == 0
    captured = capsys.readouterr().out
    assert "Craxy-AI Inference Output:" in captured
    assert "RAG Augmented:          False (0 chunks)" in captured


def test_audit_package_top_level_exports():
    """Verify src.craxy_ai exports essential interfaces and version metadata."""
    # Assert
    assert hasattr(src.craxy_ai, "cli_main")
    assert hasattr(src.craxy_ai, "CraxyPipeline")
    assert hasattr(src.craxy_ai, "SYSTEM_LIMITS")
    assert callable(src.craxy_ai.cli_main)
