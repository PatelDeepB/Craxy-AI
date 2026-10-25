"""
Type definitions, signals, and security constants for SafeLogicEvaluator.
"""

import ast
import operator
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Set


class StepLimitExceededError(RuntimeError):
    """Raised when script execution exceeds permitted maximum steps."""


class SecurityViolationError(PermissionError):
    """Raised when script violates AST security sandboxing rules."""


class ReturnSignal(Exception):
    """Internal signal for propagating return values across AST execution."""
    def __init__(self, value: Any) -> None:
        self.value = value


class BreakSignal(Exception):
    """Internal signal for loop break."""


class ContinueSignal(Exception):
    """Internal signal for loop continue."""


@dataclass(frozen=True)
class LogicEvaluationResult:
    """Stores the typed result of an in-process algorithmic logic evaluation."""
    is_success: bool
    return_value: Any
    stdout: str
    steps_executed: int
    execution_time_us: float
    error_type: Optional[str] = None
    error_message: Optional[str] = None


BINARY_OPS: Dict[type, Callable[[Any, Any], Any]] = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
}

COMPARE_OPS: Dict[type, Callable[[Any, Any], bool]] = {
    ast.Eq: operator.eq, ast.NotEq: operator.ne,
    ast.Lt: operator.lt, ast.LtE: operator.le,
    ast.Gt: operator.gt, ast.GtE: operator.ge,
    ast.In: lambda a, b: a in b,
    ast.NotIn: lambda a, b: a not in b,
    ast.Is: operator.is_, ast.IsNot: operator.is_not,
}

DISALLOWED_NODES: Set[type] = {
    ast.Import, ast.ImportFrom, ast.Try, ast.With, ast.AsyncWith,
    ast.AsyncFor, ast.AsyncFunctionDef, ast.ClassDef, ast.Delete,
    ast.Global, ast.Nonlocal, ast.Yield, ast.YieldFrom,
}

DISALLOWED_NAMES: Set[str] = {
    "eval", "exec", "open", "compile", "__import__", "globals", "locals",
    "getattr", "setattr", "delattr", "hasattr", "breakpoint", "exit",
    "quit", "input", "os", "sys", "subprocess", "builtins", "__builtins__",
}

SAFE_METHODS: Set[str] = {
    "append", "extend", "pop", "insert", "remove", "index", "count",
    "reverse", "sort", "get", "keys", "values", "items", "update",
    "split", "join", "strip", "lower", "upper", "startswith", "endswith",
    "replace", "find", "add", "discard",
}
