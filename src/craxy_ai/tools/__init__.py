"""
In-process microsecond tool augmentation subsystem for Craxy-AI.
"""

from src.craxy_ai.tools.logic_evaluator import SafeLogicEvaluator
from src.craxy_ai.tools.logic_types import (
    LogicEvaluationResult,
    SecurityViolationError,
    StepLimitExceededError,
)
from src.craxy_ai.tools.math_evaluator import EvaluationResult, SafeMathEvaluator
from src.craxy_ai.tools.code_agent_types import (
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
from src.craxy_ai.tools.code_agent import CraxyCodeAgent, AgenticCodeEngine

__all__ = [
    "SafeMathEvaluator",
    "EvaluationResult",
    "SafeLogicEvaluator",
    "LogicEvaluationResult",
    "SecurityViolationError",
    "StepLimitExceededError",
    "CraxyCodeAgent",
    "AgenticCodeEngine",
    "FileEdit",
    "FilePatch",
    "PatchHunk",
    "PatchResult",
    "SandboxExecutionResult",
    "CodeAgentError",
    "AgenticCodeError",
    "PatchSyntaxError",
    "PatchApplicationError",
    "PatchMatchError",
    "SyntaxValidationError",
    "RollbackError",
    "SecurityBoundaryError",
]
