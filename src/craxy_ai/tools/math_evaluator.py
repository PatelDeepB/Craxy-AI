"""
In-process secure Abstract Syntax Tree (AST) mathematical evaluation engine.

Implements ADR-005 and ADR-006:
- Zero subprocess creation latency (runs in 3 to 30 microseconds).
- Strict AST node whitelist preventing code injection.
- Exponent bounds checking to prevent denial-of-service memory exhaustion.
- Graceful error recovery and structured diagnostic feedback.
"""

import ast
import operator
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

from src.craxy_ai.config import SYSTEM_LIMITS


@dataclass(frozen=True)
class EvaluationResult:
    """Stores the typed result of an in-process mathematical evaluation."""
    is_success: bool
    computed_value: Optional[Union[int, float]]
    error_type: Optional[str]
    error_message: Optional[str]
    execution_time_us: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert the result to a dictionary representation."""
        return {
            "is_success": self.is_success,
            "computed_value": self.computed_value,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "execution_time_us": self.execution_time_us
        }


class SafeMathEvaluator(ast.NodeVisitor):
    """
    Safely evaluates mathematical expressions using a restricted AST whitelist.
    
    Prevents execution of arbitrary code, system calls, and state mutation.
    """

    OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
    }

    def __init__(self, max_exponent: int = SYSTEM_LIMITS.max_math_exponent_bound) -> None:
        self.max_exponent = max_exponent

    def visit_Expression(self, node: ast.Expression) -> Union[int, float]:
        """Entry point for evaluating an expression body."""
        return self.visit(node.body)

    def visit_Constant(self, node: ast.Constant) -> Union[int, float]:
        """Evaluate numeric literals."""
        if isinstance(node.value, (int, float)):
            return node.value
        raise TypeError(f"Disallowed constant type: {type(node.value).__name__}")

    def visit_BinOp(self, node: ast.BinOp) -> Union[int, float]:
        """Evaluate binary operations with bounds and safety checks."""
        op_type = type(node.op)
        if op_type not in self.OPERATORS:
            raise TypeError(f"Disallowed binary operator: {op_type.__name__}")

        left = self.visit(node.left)
        right = self.visit(node.right)

        if op_type == ast.Pow:
            if isinstance(right, (int, float)) and right > self.max_exponent:
                raise OverflowError(f"Exponent {right} exceeds safety limit {self.max_exponent}")

        if op_type in (ast.Div, ast.FloorDiv, ast.Mod) and right == 0:
            raise ZeroDivisionError("Division or modulo by zero")

        return self.OPERATORS[op_type](left, right)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Union[int, float]:
        """Evaluate unary positive and negative operators."""
        op_type = type(node.op)
        if op_type not in self.OPERATORS:
            raise TypeError(f"Disallowed unary operator: {op_type.__name__}")

        operand = self.visit(node.operand)
        return self.OPERATORS[op_type](operand)

    def generic_visit(self, node: ast.AST) -> None:
        """Reject all other AST nodes (calls, imports, attributes, variables)."""
        raise TypeError(f"Disallowed AST node: {type(node).__name__}")

    def evaluate(self, expression_str: str) -> EvaluationResult:
        """
        Parse and evaluate a mathematical expression string safely.
        
        Returns a structured EvaluationResult with microsecond execution latency.
        """
        t0 = time.perf_counter()
        cleaned_expr = expression_str.strip()

        if not cleaned_expr:
            elapsed_us = (time.perf_counter() - t0) * 1_000_000.0
            return EvaluationResult(
                is_success=False,
                computed_value=None,
                error_type="EmptyExpression",
                error_message="Expression string is empty",
                execution_time_us=elapsed_us
            )

        try:
            parsed_tree = ast.parse(cleaned_expr, mode="eval")
            result = self.visit(parsed_tree)
            elapsed_us = (time.perf_counter() - t0) * 1_000_000.0
            return EvaluationResult(
                is_success=True,
                computed_value=result,
                error_type=None,
                error_message=None,
                execution_time_us=elapsed_us
            )
        except (SyntaxError, TypeError, ZeroDivisionError, OverflowError) as err:
            elapsed_us = (time.perf_counter() - t0) * 1_000_000.0
            return EvaluationResult(
                is_success=False,
                computed_value=None,
                error_type=type(err).__name__,
                error_message=str(err),
                execution_time_us=elapsed_us
            )
        except Exception as err:
            elapsed_us = (time.perf_counter() - t0) * 1_000_000.0
            return EvaluationResult(
                is_success=False,
                computed_value=None,
                error_type="UnexpectedError",
                error_message=str(err),
                execution_time_us=elapsed_us
            )
