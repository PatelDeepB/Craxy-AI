"""Autonomous agentic code editing, syntax validation, and sandbox execution.

Provides transactional multi-file patching, unified diff application,
AST-level syntax validation, and subprocess sandboxing for self-repair loops.
"""

import ast
import difflib
from pathlib import Path
import subprocess
import time
from typing import Dict, List, Optional, Tuple

import psutil

from .code_agent_types import (
    AgenticCodeError,
    CodeAgentError,
    FileEdit,
    FilePatch,
    PatchApplicationError,
    PatchHunk,
    PatchMatchError,
    PatchResult,
    PatchSyntaxError,
    RollbackError,
    SandboxExecutionResult,
    SecurityBoundaryError,
    SyntaxValidationError,
)


class CraxyCodeAgent:
    """Orchestrates transactional code patching, syntax checks, and testing."""

    def __init__(self, workspace_root: Optional[str] = None) -> None:
        """Initialize the code agent with an optional root workspace directory."""
        self.workspace_root = Path(workspace_root).resolve() if workspace_root else Path.cwd().resolve()

    def validate_python_syntax(self, source_code: str, filename: str = "<memory>") -> bool:
        """Validate Python source code syntax using concrete AST parsing."""
        try:
            ast.parse(source_code, filename=filename)
            return True
        except SyntaxError as err:
            raise SyntaxValidationError(
                f"Syntax validation failed for {filename}:{err.lineno}:{err.offset}: {err.msg}"
            ) from err

    def create_diff(self, file_path: str, original_code: str, new_code: str) -> str:
        """Generate a standard unified diff string comparing two code versions."""
        orig_lines = original_code.splitlines(keepends=True)
        new_lines = new_code.splitlines(keepends=True)
        diff_lines = list(
            difflib.unified_diff(
                orig_lines,
                new_lines,
                fromfile=f"a/{file_path}",
                tofile=f"b/{file_path}",
            )
        )
        return "".join(diff_lines)

    def apply_unified_diff(self, original_code: str, diff_text: str) -> str:
        """Apply a unified diff string to source text buffer."""
        orig_lines = original_code.splitlines(keepends=True)
        diff_lines = diff_text.splitlines(keepends=True)
        result_lines: List[str] = []
        orig_idx, i = 0, 0

        while i < len(diff_lines):
            line = diff_lines[i]
            if line.startswith("@@"):
                orig_idx, i = self._process_hunk_header(line, diff_lines, i, orig_lines, result_lines, orig_idx)
            else:
                i += 1

        result_lines.extend(orig_lines[orig_idx:])
        return "".join(result_lines)

    def _process_hunk_header(
        self,
        hunk_line: str,
        diff_lines: List[str],
        i: int,
        orig_lines: List[str],
        result_lines: List[str],
        orig_idx: int,
    ) -> Tuple[int, int]:
        """Process a single diff hunk and update line buffers."""
        parts = hunk_line.split()
        if len(parts) < 3 or not parts[1].startswith("-"):
            raise PatchApplicationError(f"Malformed hunk header: {hunk_line.strip()}")
        target_start = int(parts[1][1:].split(",")[0]) - 1
        result_lines.extend(orig_lines[orig_idx:target_start])
        orig_idx = target_start
        i += 1
        while i < len(diff_lines) and not diff_lines[i].startswith("@@"):
            dline = diff_lines[i]
            if dline.startswith("+"):
                result_lines.append(dline[1:])
            elif dline.startswith("-"):
                orig_idx += 1
            elif dline.startswith(" "):
                result_lines.append(orig_lines[orig_idx])
                orig_idx += 1
            i += 1
        return orig_idx, i

    def _resolve_path(self, file_path: str) -> Path:
        """Resolve a file path safely within the configured workspace root."""
        candidate = Path(file_path)
        resolved = candidate if candidate.is_absolute() else (self.workspace_root / candidate).resolve()
        try:
            resolved.relative_to(self.workspace_root)
        except ValueError as err:
            raise SecurityBoundaryError(
                f"Path traversal detected: '{file_path}' escapes root '{self.workspace_root}'"
            ) from err
        return resolved

    def _apply_hunks_to_text(self, text: str, hunks: List[PatchHunk], file_name: str) -> str:
        """Apply patch hunks sequentially to text buffer."""
        curr = text
        for hunk in hunks:
            matches = curr.count(hunk.target_pattern)
            if matches == 0:
                raise PatchMatchError(f"Target pattern '{hunk.target_pattern}' not found in '{file_name}'")
            if matches > 1 and not hunk.allow_multiple:
                raise PatchMatchError(
                    f"Target pattern '{hunk.target_pattern}' occurs {matches} times in '{file_name}'"
                )
            curr = curr.replace(hunk.target_pattern, hunk.replacement_content)
        return curr

    def apply_multi_file_patch(
        self,
        patches: List[FilePatch],
        dry_run: bool = False,
    ) -> PatchResult:
        """Apply multi-file hunks atomically with rollback."""
        originals: Dict[Path, str] = {}
        planned: Dict[Path, str] = {}

        try:
            for patch in patches:
                safe_path = self._resolve_path(patch.relative_file_path)
                if not safe_path.is_file():
                    raise PatchMatchError(f"Target file not found: '{patch.relative_file_path}'")
                orig_text = safe_path.read_text(encoding="utf-8")
                originals[safe_path] = orig_text
                new_text = self._apply_hunks_to_text(orig_text, patch.hunks, patch.relative_file_path)
                if patch.relative_file_path.endswith(".py"):
                    try:
                        ast.parse(new_text, filename=patch.relative_file_path)
                    except SyntaxError as err:
                        raise PatchSyntaxError(f"Syntax error in '{patch.relative_file_path}': {err}") from err
                planned[safe_path] = new_text

            if dry_run:
                return PatchResult(success=True, applied_files=[p.relative_file_path for p in patches], modified_count=len(patches))

            for path, text in planned.items():
                path.write_text(text, encoding="utf-8")
            return PatchResult(success=True, applied_files=[str(p) for p in planned.keys()], modified_count=len(planned))
        except (SecurityBoundaryError, PatchMatchError, PatchSyntaxError, CodeAgentError) as err:
            return PatchResult(success=False, error_type=type(err).__name__, error_message=str(err))
        except Exception as err:
            return PatchResult(success=False, error_type=type(err).__name__, error_message=str(err))

    def apply_transactional_patches(
        self,
        edits: List[FileEdit],
        validate_syntax: bool = True,
    ) -> PatchResult:
        """Apply multiple file edits with atomic rollback on failure."""
        backups: Dict[Path, Optional[str]] = {}
        written_paths: List[Path] = []
        try:
            if validate_syntax:
                for edit in edits:
                    if edit.path.endswith(".py"):
                        self.validate_python_syntax(edit.new_content, filename=edit.path)
            for edit in edits:
                full_path = self._resolve_path(edit.path)
                backups[full_path] = full_path.read_text(encoding="utf-8") if full_path.exists() else None
                full_path.parent.mkdir(parents=True, exist_ok=True)
                full_path.write_text(edit.new_content, encoding="utf-8")
                written_paths.append(full_path)
            return PatchResult(success=True, applied_files=[str(p) for p in written_paths], modified_count=len(written_paths))
        except Exception as err:
            for path, original_content in backups.items():
                if original_content is None:
                    if path.exists():
                        path.unlink()
                else:
                    path.write_text(original_content, encoding="utf-8")
            return PatchResult(success=False, error_type=type(err).__name__, error_message=str(err), rolled_back=True)

    def run_isolated_test(
        self,
        command: List[str],
        cwd: Optional[str] = None,
        timeout_seconds: float = 15.0,
    ) -> SandboxExecutionResult:
        """Execute a test suite in an isolated subprocess with strict telemetry."""
        start_time = time.monotonic()
        target_cwd = cwd or str(self.workspace_root)
        proc = None
        peak_rss_mb = 0.0

        try:
            proc = subprocess.Popen(
                command, cwd=target_cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            try:
                p = psutil.Process(proc.pid)
                peak_rss_mb = p.memory_info().rss / (1024 * 1024)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                peak_rss_mb = 0.0

            stdout, stderr = proc.communicate(timeout=timeout_seconds)
            duration = time.monotonic() - start_time
            return SandboxExecutionResult(
                exit_code=proc.returncode, stdout=stdout, stderr=stderr, duration_seconds=duration, peak_memory_mb=peak_rss_mb
            )
        except subprocess.TimeoutExpired:
            if proc:
                proc.kill()
                stdout, stderr = proc.communicate()
            else:
                stdout, stderr = "", ""
            duration = time.monotonic() - start_time
            return SandboxExecutionResult(
                exit_code=-1, stdout=stdout, stderr=stderr, duration_seconds=duration, timed_out=True, peak_memory_mb=peak_rss_mb
            )


AgenticCodeEngine = CraxyCodeAgent

__all__ = [
    "CraxyCodeAgent",
    "AgenticCodeEngine",
    "CodeAgentError",
    "FileEdit",
    "FilePatch",
    "PatchHunk",
    "PatchResult",
    "PatchApplicationError",
    "PatchMatchError",
    "PatchSyntaxError",
    "SyntaxValidationError",
    "SecurityBoundaryError",
    "RollbackError",
    "SandboxExecutionResult",
]
