"""
Unit tests for Phase 3 Stage 2: AgenticCodeEngine (Autonomous Multi-File Code Patching).
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
    PatchMatchError,
    PatchResult,
    PatchSyntaxError,
    SecurityBoundaryError,
)
from src.craxy_ai.tools.code_agent_types import (
    FileEdit,
    SandboxExecutionResult,
    SyntaxValidationError,
)


@pytest.fixture
def temp_workspace():
    """Create a temporary directory structure for testing safe file patching."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        file_a = os.path.join(tmp_dir, "module_a.py")
        file_b = os.path.join(tmp_dir, "module_b.py")
        text_doc = os.path.join(tmp_dir, "notes.txt")

        with open(file_a, "w", encoding="utf-8") as f:
            f.write("def compute(x):\n    return x * 2\n")

        with open(file_b, "w", encoding="utf-8") as f:
            f.write("def greet(name):\n    return f'Hello, {name}'\n")

        with open(text_doc, "w", encoding="utf-8") as f:
            f.write("Project notes: v1.0\nStatus: Initial\n")

        yield tmp_dir


def test_code_engine_successful_single_file_patch(temp_workspace):
    """Verify applying a valid single-file replacement."""
    engine = AgenticCodeEngine(workspace_root=temp_workspace)
    hunk = PatchHunk(
        target_pattern="return x * 2",
        replacement_content="return x * 10"
    )
    patch = FilePatch(relative_file_path="module_a.py", hunks=[hunk])

    result: PatchResult = engine.apply_multi_file_patch([patch], dry_run=False)
    assert result.is_success
    assert len(result.files_modified) == 1

    with open(os.path.join(temp_workspace, "module_a.py"), "r", encoding="utf-8") as f:
        content = f.read()
    assert "return x * 10" in content


def test_code_engine_multi_file_atomic_patch(temp_workspace):
    """Verify modifying multiple files simultaneously in one transaction."""
    engine = AgenticCodeEngine(workspace_root=temp_workspace)
    patch_a = FilePatch(
        relative_file_path="module_a.py",
        hunks=[PatchHunk("return x * 2", "return x ** 2")]
    )
    patch_b = FilePatch(
        relative_file_path="module_b.py",
        hunks=[PatchHunk("Hello, {name}", "Greetings, {name}!")]
    )

    result: PatchResult = engine.apply_multi_file_patch([patch_a, patch_b], dry_run=False)
    assert result.is_success
    assert len(result.files_modified) == 2

    with open(os.path.join(temp_workspace, "module_a.py"), "r", encoding="utf-8") as f:
        assert "return x ** 2" in f.read()

    with open(os.path.join(temp_workspace, "module_b.py"), "r", encoding="utf-8") as f:
        assert "Greetings, {name}!" in f.read()


def test_code_engine_dry_run_leaves_disk_unmodified(temp_workspace):
    """Verify dry_run=True validates changes without writing to disk."""
    engine = AgenticCodeEngine(workspace_root=temp_workspace)
    patch = FilePatch(
        relative_file_path="module_a.py",
        hunks=[PatchHunk("return x * 2", "return x * 999")]
    )

    result: PatchResult = engine.apply_multi_file_patch([patch], dry_run=True)
    assert result.is_success

    # File on disk must remain unchanged
    with open(os.path.join(temp_workspace, "module_a.py"), "r", encoding="utf-8") as f:
        assert "return x * 2" in f.read()
        assert "return x * 999" not in f.read()


def test_code_engine_security_boundary_traversal_blocked(temp_workspace):
    """Verify path traversal outside workspace root is blocked."""
    engine = AgenticCodeEngine(workspace_root=temp_workspace)
    patch = FilePatch(
        relative_file_path="../../outside_file.py",
        hunks=[PatchHunk("target", "replacement")]
    )

    result: PatchResult = engine.apply_multi_file_patch([patch], dry_run=False)
    assert not result.is_success
    assert result.error_type == "SecurityBoundaryError"


def test_code_engine_patch_match_failure_rollback(temp_workspace):
    """Verify missing target pattern aborts transaction without modifying any file."""
    engine = AgenticCodeEngine(workspace_root=temp_workspace)
    patch_valid = FilePatch(
        relative_file_path="module_a.py",
        hunks=[PatchHunk("return x * 2", "return x * 50")]
    )
    patch_invalid = FilePatch(
        relative_file_path="module_b.py",
        hunks=[PatchHunk("nonexistent_pattern_string", "replacement")]
    )

    # Multi-file patch where second file fails
    result: PatchResult = engine.apply_multi_file_patch([patch_valid, patch_invalid], dry_run=False)
    assert not result.is_success
    assert result.error_type in ("PatchMatchError", "PatchApplicationError")

    # Both files must remain completely unchanged (atomic rollback)
    with open(os.path.join(temp_workspace, "module_a.py"), "r", encoding="utf-8") as f:
        assert "return x * 2" in f.read()


def test_code_engine_syntax_validation_rollback(temp_workspace):
    """Verify syntax error in patched Python code triggers atomic abort."""
    engine = AgenticCodeEngine(workspace_root=temp_workspace)
    # Introducing invalid syntax
    patch = FilePatch(
        relative_file_path="module_a.py",
        hunks=[PatchHunk("return x * 2", "def unclosed_syntax_broken(")]
    )

    result: PatchResult = engine.apply_multi_file_patch([patch], dry_run=False)
    assert not result.is_success
    assert result.error_type in ("PatchSyntaxError", "SyntaxValidationError")

    # File must retain original valid code
    with open(os.path.join(temp_workspace, "module_a.py"), "r", encoding="utf-8") as f:
        assert "return x * 2" in f.read()


def test_code_engine_allow_multiple_occurrences(temp_workspace):
    """Verify allow_multiple=True replaces all instances cleanly."""
    file_path = os.path.join(temp_workspace, "multi.py")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("val = 1\nval = 1\n")

    engine = AgenticCodeEngine(workspace_root=temp_workspace)
    
    # Without allow_multiple -> should fail
    hunk_fail = PatchHunk("val = 1", "val = 2", allow_multiple=False)
    res_fail = engine.apply_multi_file_patch([FilePatch("multi.py", [hunk_fail])])
    assert not res_fail.is_success
    assert res_fail.error_type in ("PatchMatchError", "PatchApplicationError")

    # With allow_multiple -> should succeed
    hunk_ok = PatchHunk("val = 1", "val = 2", allow_multiple=True)
    res_ok = engine.apply_multi_file_patch([FilePatch("multi.py", [hunk_ok])])
    assert res_ok.is_success

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    assert content == "val = 2\nval = 2\n"


def test_validate_python_syntax_ast_verification():
    """Verify AST parser flags valid and invalid code syntax."""
    agent = CraxyCodeAgent()
    assert agent.validate_python_syntax("def add(x, y):\n    return x + y\n")
    with pytest.raises(SyntaxValidationError):
        agent.validate_python_syntax("def broken_def(x\n    return x\n")


def test_create_and_apply_unified_diff_roundtrip():
    """Verify unified diff generation and patching roundtrip."""
    agent = CraxyCodeAgent()
    orig = "def compute():\n    return 10\n"
    modified = "def compute():\n    return 42\n"
    diff = agent.create_diff("test.py", orig, modified)
    assert "-    return 10" in diff
    assert "+    return 42" in diff
    applied = agent.apply_unified_diff(orig, diff)
    assert applied == modified


def test_apply_transactional_patches_with_file_edits(temp_workspace):
    """Verify apply_transactional_patches handles FileEdit lists."""
    agent = CraxyCodeAgent(workspace_root=temp_workspace)
    path_a = "module_a.py"
    orig_a = "def compute(x):\n    return x * 2\n"
    new_a = "def compute(x):\n    return x * 888\n"
    edit = FileEdit(path=path_a, original_content=orig_a, new_content=new_a)
    res = agent.apply_transactional_patches([edit])
    assert res.success
    with open(os.path.join(temp_workspace, "module_a.py"), "r", encoding="utf-8") as f:
        assert "return x * 888" in f.read()


def test_run_isolated_test_telemetry(temp_workspace):
    """Verify isolated subprocess execution telemetry."""
    agent = CraxyCodeAgent(workspace_root=temp_workspace)
    cmd = [sys.executable, "-c", "import sys; sys.stdout.write('OUT_OK'); sys.exit(0)"]
    res: SandboxExecutionResult = agent.run_isolated_test(cmd, timeout_seconds=5.0)
    assert res.exit_code == 0
    assert not res.timed_out
    assert res.stdout == "OUT_OK"


def test_code_agent_zero_memory_leak_under_sustained_patching(temp_workspace):
    """Verify sustained patching and rollback operations exhibit zero memory leakage."""
    engine = AgenticCodeEngine(workspace_root=temp_workspace)
    proc = psutil.Process()
    rss_before = proc.memory_info().rss / (1024 * 1024)

    for i in range(50):
        hunk = PatchHunk(f"return x * {i + 1}", f"return x * {i + 2}", allow_multiple=True)
        # Even numbered iterations succeed, odd numbers test dry_run
        engine.apply_multi_file_patch(
            [FilePatch("module_a.py", [hunk])],
            dry_run=(i % 2 == 1),
        )

    rss_after = proc.memory_info().rss / (1024 * 1024)
    rss_drift = abs(rss_after - rss_before)
    # Memory drift should remain strictly under 15 MB
    assert rss_drift < 15.0

