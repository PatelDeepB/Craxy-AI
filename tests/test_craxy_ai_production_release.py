"""End-to-End System Packaging and Release Verification Test Suite (Turn 62).

Validates:
1. pyproject.toml PEP 621 packaging metadata, scripts, and dependencies.
2. python -m src.craxy_ai standalone module entrypoint execution.
3. python -m src.craxy_ai --chat terminal interactive loop with stdin stream.
4. README.md documentation integrity and system architecture completeness.
5. Package exports, semantic versioning, and physical DRAM budget compliance.
"""

import os
import subprocess
import sys
import tomllib
import psutil
import pytest

import src.craxy_ai as craxy_pkg


def test_release_pyproject_toml_configuration():
    """Verify pyproject.toml defines required PEP 621 metadata and scripts."""
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pyproject_path = os.path.join(root_dir, "pyproject.toml")
    assert os.path.exists(pyproject_path), "pyproject.toml must exist at repo root"

    with open(pyproject_path, "rb") as fh:
        config = tomllib.load(fh)

    project = config.get("project", {})
    assert project.get("name") == "craxy-ai"
    assert project.get("version") == "0.1.0"
    assert "numpy" in str(project.get("dependencies", []))
    assert "psutil" in str(project.get("dependencies", []))

    scripts = project.get("scripts", {})
    assert "craxy-ai" in scripts
    assert scripts["craxy-ai"] == "src.craxy_ai.cli:main"


def test_release_module_entrypoint_info():
    """Verify python -m src.craxy_ai --info executes successfully."""
    cmd = [sys.executable, "-m", "src.craxy_ai", "--info"]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0
    stdout = result.stdout
    assert "Max RAM RSS Budget:" in stdout
    assert "2048.0 MB" in stdout
    assert "NVMe Paged KV Cache" in stdout
    assert "Autonomous Code Agent" in stdout


def test_release_module_entrypoint_chat():
    """Verify python -m src.craxy_ai --chat executes interactively."""
    cmd = [sys.executable, "-m", "src.craxy_ai", "--chat"]
    input_text = "/help\n/exit\n"
    result = subprocess.run(
        cmd,
        input=input_text,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0
    stdout = result.stdout
    assert "Interactive Terminal Chat" in stdout
    assert "Available commands:" in stdout
    assert "Session terminated. Goodbye!" in stdout


def test_release_readme_documentation_completeness():
    """Verify README.md contains critical architecture and command references."""
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    readme_path = os.path.join(root_dir, "README.md")
    assert os.path.exists(readme_path), "README.md must exist at repo root"

    with open(readme_path, "r", encoding="utf-8") as fh:
        content = fh.read()

    assert "2,048 MB physical DRAM budget" in content
    assert "CraxyPagedKVCache" in content
    assert "AgenticCodeEngine" in content
    assert "CraxyGGUFEngine" in content
    assert "craxy-ai --chat" in content
    assert "craxy-ai --benchmark" in content
    assert "—" not in content, "Disallowed em dash found in README.md"


def test_release_package_exports_and_version():
    """Verify package exposes version and key core components."""
    assert hasattr(craxy_pkg, "__version__")
    assert craxy_pkg.__version__ == "0.1.0"
    assert hasattr(craxy_pkg, "CraxyPipeline")
    assert hasattr(craxy_pkg, "SYSTEM_LIMITS")
    assert craxy_pkg.SYSTEM_LIMITS.max_ram_rss_mb == 2048.0


def test_release_system_ram_budget_guarantee():
    """Verify active system RSS stays strictly within the 2,048 MB physical budget."""
    process = psutil.Process(os.getpid())
    current_rss_mb = process.memory_info().rss / (1024 * 1024)
    budget_mb = craxy_pkg.SYSTEM_LIMITS.max_ram_rss_mb

    assert current_rss_mb < budget_mb, f"RSS {current_rss_mb:.1f} MB exceeds {budget_mb} MB"
    headroom_mb = budget_mb - current_rss_mb
    assert headroom_mb > 500.0, f"Headroom {headroom_mb:.1f} MB is insufficient (< 500 MB)"
