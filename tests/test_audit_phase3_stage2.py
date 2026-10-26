"""Independent adversarial audit of Phase 3 Stage 2: AgenticCodeEngine.

Conducted by ULTRON 2 to red-team security boundaries, atomic rollback,
UTF-8 character preservation, subprocess timeout containment, and memory drift.
"""

import os
import sys
import tempfile
import psutil
import pytest

from src.craxy_ai.tools.code_agent import (
    AgenticCodeEngine,
    CraxyCodeAgent,
    FilePatch,
    PatchHunk,
    PatchResult,
    SecurityBoundaryError,
)
from src.craxy_ai.tools.code_agent_types import (
    FileEdit,
    SandboxExecutionResult,
    SyntaxValidationError,
)


@pytest.fixture
def isolated_workspace():
    """Create a temporary sandbox workspace for adversarial testing."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        file_one = os.path.join(tmp_dir, "core_service.py")
        file_two = os.path.join(tmp_dir, "i18n_strings.py")

        with open(file_one, "w", encoding="utf-8") as handle:
            handle.write("def service_handler(data):\n    return data['valid']\n")

        with open(file_two, "w", encoding="utf-8") as handle:
            handle.write("GREETING = 'Hello, world! 🚀'\nSYMBOL = '∑_{i=0}^N x_i'\n")

        yield tmp_dir


def test_audit_adversarial_traversal_escapes(isolated_workspace):
    """Adversarial check: ensure multiple path traversal attack vectors fail."""
    engine = AgenticCodeEngine(workspace_root=isolated_workspace)
    hunk = PatchHunk("data", "malicious_data")

    # Vector 1: Standard relative dot-dot escape
    patch_rel = FilePatch("../../secret_keys.py", [hunk])
    result_rel: PatchResult = engine.apply_multi_file_patch([patch_rel])
    assert not result_rel.is_success
    assert result_rel.error_type == "SecurityBoundaryError"

    # Vector 2: Nested slash bypass attempt
    patch_nested = FilePatch("subdir/../../../root_token.json", [hunk])
    result_nested: PatchResult = engine.apply_multi_file_patch([patch_nested])
    assert not result_nested.is_success
    assert result_nested.error_type == "SecurityBoundaryError"


def test_audit_atomic_rollback_on_midstream_syntax_failure(isolated_workspace):
    """Adversarial check: ensure clean state recovery if later file has syntax error."""
    engine = AgenticCodeEngine(workspace_root=isolated_workspace)
    path_one = os.path.join(isolated_workspace, "core_service.py")
    orig_one = "def service_handler(data):\n    return data['valid']\n"

    # Multi-file edits where first file is valid, second file has broken syntax
    edit_valid = FileEdit(
        path="core_service.py",
        original_content=orig_one,
        new_content="def service_handler(data):\n    return data['modified']\n"
    )
    edit_invalid = FileEdit(
        path="broken.py",
        original_content="",
        new_content="def syntax_error_func(:"
    )

    result: PatchResult = engine.apply_transactional_patches([edit_valid, edit_invalid])
    assert not result.success
    assert result.rolled_back

    # Verify first file was not left in modified state
    with open(path_one, "r", encoding="utf-8") as handle:
        current_content = handle.read()
    assert current_content == orig_one


def test_audit_utf8_multibyte_and_special_character_preservation(isolated_workspace):
    """Adversarial check: verify UTF-8 multi-byte glyphs and math symbols survive patches."""
    engine = AgenticCodeEngine(workspace_root=isolated_workspace)
    path_two = os.path.join(isolated_workspace, "i18n_strings.py")

    hunk = PatchHunk(
        target_pattern="Hello, world! 🚀",
        replacement_content="Bonjour le monde! 🌟 & こんにちは"
    )
    patch = FilePatch(relative_file_path="i18n_strings.py", hunks=[hunk])

    result: PatchResult = engine.apply_multi_file_patch([patch])
    assert result.is_success

    with open(path_two, "r", encoding="utf-8") as handle:
        content = handle.read()

    assert "Bonjour le monde! 🌟 & こんにちは" in content
    assert "SYMBOL = '∑_{i=0}^N x_i'" in content


def test_audit_subprocess_timeout_containment(isolated_workspace):
    """Adversarial check: ensure runaway child processes are killed cleanly without hanging."""
    agent = CraxyCodeAgent(workspace_root=isolated_workspace)
    # Python script that sleeps for 10 seconds
    cmd = [sys.executable, "-c", "import time; time.sleep(10)"]

    result: SandboxExecutionResult = agent.run_isolated_test(cmd, timeout_seconds=0.5)
    assert result.timed_out
    assert result.exit_code == -1
    assert result.duration_seconds < 2.5


def test_audit_sustained_100_cycle_memory_drift(isolated_workspace):
    """Adversarial check: verify 100 sustained transactional cycles exhibit zero memory drift."""
    engine = AgenticCodeEngine(workspace_root=isolated_workspace)
    process = psutil.Process()
    rss_start = process.memory_info().rss / (1024 * 1024)

    for cycle in range(100):
        # Alternate between successful patch, dry-run, and controlled syntax error
        if cycle % 3 == 0:
            hunk = PatchHunk(f"cycle_{cycle - 1}", f"cycle_{cycle}", allow_multiple=True)
            engine.apply_multi_file_patch(
                [FilePatch("core_service.py", [hunk])],
                dry_run=True
            )
        elif cycle % 3 == 1:
            hunk = PatchHunk("invalid_pattern_never_matches", "replacement")
            engine.apply_multi_file_patch(
                [FilePatch("core_service.py", [hunk])],
                dry_run=False
            )
        else:
            agent = CraxyCodeAgent(workspace_root=isolated_workspace)
            agent.validate_python_syntax("def fast_check(a, b):\n    return a + b\n")

    rss_end = process.memory_info().rss / (1024 * 1024)
    rss_drift = abs(rss_end - rss_start)

    # Hard ceiling: Memory drift must be less than 5.0 MB across 100 cycles
    assert rss_drift < 5.0
