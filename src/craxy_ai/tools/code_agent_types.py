"""Data structures and typed error hierarchies for agentic code modification.

Defines patch descriptors, validation records, execution sandbox results,
and domain-specific exception types for multi-file safe code refactoring.
"""

from dataclasses import dataclass, field
from typing import List, Optional


class CodeAgentError(Exception):
    """Base exception for all code agent operations."""


class PatchSyntaxError(CodeAgentError):
    """Raised when a unified diff or patch string cannot be parsed."""


class PatchApplicationError(CodeAgentError):
    """Raised when a patch hunk fails to match the target source buffer."""


class PatchMatchError(CodeAgentError):
    """Raised when target pattern does not match or occurs multiple times."""


class SyntaxValidationError(CodeAgentError):
    """Raised when modified code fails AST syntax or grammar validation."""


class ExecutionSandboxError(CodeAgentError):
    """Raised when an isolated subprocess execution fails critically."""


class RollbackError(CodeAgentError):
    """Raised when restoring an original file state during rollback fails."""


class SecurityBoundaryError(CodeAgentError):
    """Raised when target file path attempts to escape workspace boundary."""


# Alternative exception aliases
AgenticCodeError = CodeAgentError


@dataclass(frozen=True)
class FileEdit:
    """Represents a planned atomic modification to a single source file.

    Attributes:
        path: Path to the target source file.
        original_content: Content of the file prior to editing.
        new_content: Proposed new content after applying edits.
        diff: Unified diff representation of the change.
    """

    path: str
    original_content: str
    new_content: str
    diff: str = ""


@dataclass(frozen=True)
class PatchHunk:
    """Represents a discrete pattern replacement hunk."""

    target_pattern: str
    replacement_content: str
    allow_multiple: bool = False


@dataclass(frozen=True)
class FilePatch:
    """Represents a patch with relative file path and hunks."""

    relative_file_path: str
    hunks: List[PatchHunk]


@dataclass(frozen=True)
class PatchResult:
    """Represents the outcome of an atomic multi-file patch transaction.

    Attributes:
        success: Whether all patches applied and validated cleanly.
        applied_files: List of file paths successfully written.
        modified_count: Total number of files altered.
        error_message: Explanatory error details if failure occurred.
        rolled_back: Whether an automatic rollback was executed.
        is_success: Alias property for compatibility.
        files_modified: Alias property for compatibility.
        error_type: Name of error class for compatibility.
    """

    success: bool
    applied_files: List[str] = field(default_factory=list)
    modified_count: int = 0
    error_message: Optional[str] = None
    rolled_back: bool = False
    error_type: Optional[str] = None

    @property
    def is_success(self) -> bool:
        """Compatibility property for is_success."""
        return self.success

    @property
    def files_modified(self) -> List[str]:
        """Compatibility property for files_modified."""
        return self.applied_files


@dataclass(frozen=True)
class SandboxExecutionResult:
    """Telemetry captured from an isolated subprocess test run.

    Attributes:
        exit_code: Process return code (0 indicates success).
        stdout: Standard output captured from the process.
        stderr: Standard error captured from the process.
        duration_seconds: Wall-clock execution time in seconds.
        timed_out: Whether execution was aborted due to timeout.
        peak_memory_mb: Approximate memory overhead during execution.
    """

    exit_code: int
    stdout: str
    stderr: str
    duration_seconds: float
    timed_out: bool = False
    peak_memory_mb: float = 0.0
