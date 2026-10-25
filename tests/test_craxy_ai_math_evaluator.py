"""
Unit tests for the production Craxy-AI SafeMathEvaluator tool.

Verifies arithmetic correctness, security isolation against code injection,
exponent overflow prevention, division by zero handling, and execution latency.
"""

import pytest
from src.craxy_ai.tools.math_evaluator import SafeMathEvaluator


@pytest.fixture
def evaluator() -> SafeMathEvaluator:
    """Fixture providing a fresh evaluator instance."""
    return SafeMathEvaluator(max_exponent=1000)


def test_standard_arithmetic_operations(evaluator: SafeMathEvaluator) -> None:
    """Should accurately compute standard multi-operation expressions."""
    # Arrange
    expression = "((250 * 12) + (150 * 8)) / 2"
    
    # Act
    result = evaluator.evaluate(expression)
    
    # Assert
    assert result.is_success is True
    assert result.computed_value == 2100.0
    assert result.error_type is None
    assert result.execution_time_us < 1000.0  # Under 1 millisecond


def test_complex_exponentiation_within_bounds(evaluator: SafeMathEvaluator) -> None:
    """Should correctly evaluate exponentiation when within configured limit."""
    # Arrange
    expression = "2 ** 10"
    
    # Act
    result = evaluator.evaluate(expression)
    
    # Assert
    assert result.is_success is True
    assert result.computed_value == 1024


def test_exponentiation_exceeding_bounds_is_rejected(evaluator: SafeMathEvaluator) -> None:
    """Should block excessive exponents to prevent CPU denial-of-service."""
    # Arrange: Exponent 5000 exceeds evaluator's max_exponent 1000
    expression = "2 ** 5000"
    
    # Act
    result = evaluator.evaluate(expression)
    
    # Assert
    assert result.is_success is False
    assert result.error_type == "OverflowError"
    assert "exceeds safety limit" in result.error_message


def test_division_by_zero_handling(evaluator: SafeMathEvaluator) -> None:
    """Should gracefully trap division by zero and return typed failure."""
    # Arrange
    expression = "500 / 0"
    
    # Act
    result = evaluator.evaluate(expression)
    
    # Assert
    assert result.is_success is False
    assert result.error_type == "ZeroDivisionError"
    assert result.computed_value is None


def test_security_rejection_of_arbitrary_function_calls(evaluator: SafeMathEvaluator) -> None:
    """Should strictly reject function calls to prevent code execution."""
    # Arrange: Attempting to call system function
    expression = "__import__('os').system('dir')"
    
    # Act
    result = evaluator.evaluate(expression)
    
    # Assert
    assert result.is_success is False
    assert result.error_type == "TypeError"
    assert "Disallowed AST node" in result.error_message


def test_security_rejection_of_variable_names(evaluator: SafeMathEvaluator) -> None:
    """Should disallow free variable names that could access local scope."""
    # Arrange
    expression = "x + 10"
    
    # Act
    result = evaluator.evaluate(expression)
    
    # Assert
    assert result.is_success is False
    assert result.error_type == "TypeError"
    assert "Name" in result.error_message


def test_empty_expression_returns_failure(evaluator: SafeMathEvaluator) -> None:
    """Should return structured error for empty expression strings."""
    # Arrange
    expression = "   "
    
    # Act
    result = evaluator.evaluate(expression)
    
    # Assert
    assert result.is_success is False
    assert result.error_type == "EmptyExpression"
