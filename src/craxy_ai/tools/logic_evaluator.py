"""
Secure Abstract Syntax Tree (AST) algorithmic logic execution engine for Craxy-AI.

Implements Phase 2 Stage 8:
- AST sandboxing rejecting dangerous constructs (imports, exec, eval, private attributes).
- Deterministic execution step bounding (max 10,000 steps) preventing infinite loops.
- Recursion depth bounding preventing stack overflow.
- Buffered stdout collection and structured typed results.
"""

import ast
import time
from typing import Any, Dict, List, Optional

from src.craxy_ai.tools.logic_types import (
    BINARY_OPS,
    COMPARE_OPS,
    DISALLOWED_NAMES,
    DISALLOWED_NODES,
    SAFE_METHODS,
    BreakSignal,
    ContinueSignal,
    LogicEvaluationResult,
    ReturnSignal,
    SecurityViolationError,
    StepLimitExceededError,
)


class SafeLogicEvaluator:
    """Safely evaluates algorithmic Python code via an AST-level interpreter."""

    def __init__(self, max_steps: int = 10_000, max_call_depth: int = 50) -> None:
        self.max_steps = max_steps
        self.max_call_depth = max_call_depth
        self.step_counter = 0
        self.call_depth = 0
        self.output_buffer: List[str] = []

    def _tick(self) -> None:
        self.step_counter += 1
        if self.step_counter > self.max_steps:
            raise StepLimitExceededError(f"Step limit of {self.max_steps} exceeded")

    def _validate_ast(self, tree: ast.AST) -> None:
        for node in ast.walk(tree):
            if type(node) in DISALLOWED_NODES:
                raise SecurityViolationError(f"Disallowed construct: {type(node).__name__}")
            if isinstance(node, ast.Name) and node.id in DISALLOWED_NAMES:
                raise SecurityViolationError(f"Disallowed name access: {node.id}")
            if isinstance(node, ast.Attribute):
                if node.attr.startswith("_") or node.attr not in SAFE_METHODS:
                    raise SecurityViolationError(f"Disallowed attribute access: {node.attr}")

    def evaluate(
        self,
        code_str: str,
        entry_point: Optional[str] = None,
        args: Optional[List[Any]] = None
    ) -> LogicEvaluationResult:
        t0 = time.perf_counter()
        self.step_counter = 0
        self.call_depth = 0
        self.output_buffer = []

        try:
            tree = ast.parse(code_str, mode="exec")
            self._validate_ast(tree)
            scope = self._create_builtins()

            ret_val = None
            for stmt in tree.body:
                self._tick()
                ret_val = self._exec_stmt(stmt, scope)

            if entry_point:
                if entry_point not in scope or not callable(scope[entry_point]):
                    raise KeyError(f"Entry point '{entry_point}' not found in executable scope")
                func_args = args or []
                ret_val = scope[entry_point](*func_args)

            elapsed_us = (time.perf_counter() - t0) * 1_000_000.0
            return LogicEvaluationResult(
                is_success=True,
                return_value=ret_val,
                stdout="".join(self.output_buffer),
                steps_executed=self.step_counter,
                execution_time_us=elapsed_us
            )
        except Exception as err:
            elapsed_us = (time.perf_counter() - t0) * 1_000_000.0
            return LogicEvaluationResult(
                is_success=False,
                return_value=None,
                stdout="".join(self.output_buffer),
                steps_executed=self.step_counter,
                execution_time_us=elapsed_us,
                error_type=type(err).__name__,
                error_message=str(err)
            )

    def _create_builtins(self) -> Dict[str, Any]:
        return {
            "range": range, "len": len, "min": min, "max": max, "sum": sum,
            "abs": abs, "round": round, "int": int, "float": float, "str": str,
            "bool": bool, "list": list, "dict": dict, "set": set, "tuple": tuple,
            "sorted": sorted, "reversed": lambda x: list(reversed(x)),
            "enumerate": enumerate, "zip": zip, "all": all, "any": any,
            "pow": pow, "divmod": divmod,
            "print": lambda *a, **k: self.output_buffer.append(" ".join(map(str, a)) + "\n"),
        }

    def _exec_stmt(self, node: ast.stmt, scope: Dict[str, Any]) -> Any:
        self._tick()
        if isinstance(node, ast.Assign):
            val = self._eval_expr(node.value, scope)
            for target in node.targets:
                self._assign_target(target, val, scope)
            return val
        elif isinstance(node, ast.AugAssign):
            val = self._eval_expr(node.value, scope)
            current = self._eval_expr(node.target, scope)
            if type(node.op) == ast.Mult:
                if isinstance(current, (str, list, tuple)) and isinstance(val, int) and (len(current) * val > 1_000_000):
                    raise MemoryError(f"Sequence multiplication exceeds safe limit of 1,000,000 items")
            op_func = BINARY_OPS.get(type(node.op))
            if not op_func:
                raise TypeError(f"Unsupported augassign op: {type(node.op).__name__}")
            new_val = op_func(current, val)
            self._assign_target(node.target, new_val, scope)
            return new_val
        elif isinstance(node, ast.For):
            iterable = self._eval_expr(node.iter, scope)
            for item in iterable:
                self._tick()
                self._assign_target(node.target, item, scope)
                try:
                    for s in node.body:
                        self._exec_stmt(s, scope)
                except BreakSignal:
                    break
                except ContinueSignal:
                    continue
            return None
        elif isinstance(node, ast.While):
            while self._eval_expr(node.test, scope):
                self._tick()
                try:
                    for s in node.body:
                        self._exec_stmt(s, scope)
                except BreakSignal:
                    break
                except ContinueSignal:
                    continue
            return None
        elif isinstance(node, ast.If):
            branch = node.body if self._eval_expr(node.test, scope) else node.orelse
            for s in branch:
                self._exec_stmt(s, scope)
            return None
        elif isinstance(node, ast.FunctionDef):
            def user_func(*args: Any) -> Any:
                self.call_depth += 1
                if self.call_depth > self.max_call_depth:
                    raise RecursionError(f"Call depth {self.max_call_depth} exceeded")
                local_scope = dict(scope)
                for arg_name, arg_val in zip([a.arg for a in node.args.args], args):
                    local_scope[arg_name] = arg_val
                try:
                    for s in node.body:
                        self._exec_stmt(s, local_scope)
                    return None
                except ReturnSignal as sig:
                    return sig.value
                finally:
                    self.call_depth -= 1
            scope[node.name] = user_func
            return None
        elif isinstance(node, ast.Return):
            val = self._eval_expr(node.value, scope) if node.value else None
            raise ReturnSignal(val)
        elif isinstance(node, ast.Break):
            raise BreakSignal()
        elif isinstance(node, ast.Continue):
            raise ContinueSignal()
        elif isinstance(node, ast.Expr):
            return self._eval_expr(node.value, scope)
        elif isinstance(node, ast.Pass):
            return None
        raise TypeError(f"Unsupported statement type: {type(node).__name__}")

    def _assign_target(self, target: ast.AST, val: Any, scope: Dict[str, Any]) -> None:
        if isinstance(target, ast.Name):
            scope[target.id] = val
        elif isinstance(target, ast.Subscript):
            container = self._eval_expr(target.value, scope)
            key = self._eval_slice_or_index(target.slice, scope)
            container[key] = val
        elif isinstance(target, (ast.Tuple, ast.List)):
            for sub_target, sub_val in zip(target.elts, val):
                self._assign_target(sub_target, sub_val, scope)
        else:
            raise TypeError(f"Unsupported assignment target: {type(target).__name__}")

    def _eval_expr(self, node: ast.AST, scope: Dict[str, Any]) -> Any:
        self._tick()
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Name):
            if node.id in scope:
                return scope[node.id]
            raise NameError(f"Name '{node.id}' is not defined")
        elif isinstance(node, ast.BinOp):
            left = self._eval_expr(node.left, scope)
            right = self._eval_expr(node.right, scope)
            if type(node.op) == ast.Pow and isinstance(right, (int, float)) and right > 1000:
                raise OverflowError(f"Exponent {right} exceeds safety limit")
            if type(node.op) == ast.Mult:
                if isinstance(left, (str, list, tuple)) and isinstance(right, int) and (len(left) * right > 1_000_000):
                    raise MemoryError(f"Sequence multiplication exceeds safe limit of 1,000,000 items")
                if isinstance(right, (str, list, tuple)) and isinstance(left, int) and (len(right) * left > 1_000_000):
                    raise MemoryError(f"Sequence multiplication exceeds safe limit of 1,000,000 items")
            op_fn = BINARY_OPS.get(type(node.op))
            if not op_fn:
                raise TypeError(f"Unsupported binary operator: {type(node.op).__name__}")
            return op_fn(left, right)
        elif isinstance(node, ast.UnaryOp):
            operand = self._eval_expr(node.operand, scope)
            if isinstance(node.op, ast.USub):
                return -operand
            elif isinstance(node.op, ast.UAdd):
                return +operand
            elif isinstance(node.op, ast.Not):
                return not operand
            raise TypeError(f"Unsupported unary operator: {type(node.op).__name__}")
        elif isinstance(node, ast.BoolOp):
            if isinstance(node.op, ast.And):
                for val_node in node.values:
                    v = self._eval_expr(val_node, scope)
                    if not v:
                        return v
                return v
            elif isinstance(node.op, ast.Or):
                for val_node in node.values:
                    v = self._eval_expr(val_node, scope)
                    if v:
                        return v
                return v
        elif isinstance(node, ast.Compare):
            left = self._eval_expr(node.left, scope)
            for op, comparator in zip(node.ops, node.comparators):
                right = self._eval_expr(comparator, scope)
                cmp_fn = COMPARE_OPS.get(type(op))
                if not cmp_fn or not cmp_fn(left, right):
                    return False
                left = right
            return True
        elif isinstance(node, ast.Call):
            func = self._eval_expr(node.func, scope)
            args = [self._eval_expr(a, scope) for a in node.args]
            return func(*args)
        elif isinstance(node, ast.Attribute):
            obj = self._eval_expr(node.value, scope)
            return getattr(obj, node.attr)
        elif isinstance(node, ast.Subscript):
            container = self._eval_expr(node.value, scope)
            key = self._eval_slice_or_index(node.slice, scope)
            return container[key]
        elif isinstance(node, ast.List):
            return [self._eval_expr(e, scope) for e in node.elts]
        elif isinstance(node, ast.Tuple):
            return tuple(self._eval_expr(e, scope) for e in node.elts)
        elif isinstance(node, ast.Set):
            return {self._eval_expr(e, scope) for e in node.elts}
        elif isinstance(node, ast.Dict):
            return {self._eval_expr(k, scope): self._eval_expr(v, scope) for k, v in zip(node.keys, node.values)}
        elif isinstance(node, ast.ListComp):
            comp_scope = dict(scope)
            results = []
            gen = node.generators[0]
            iterable = self._eval_expr(gen.iter, comp_scope)
            for item in iterable:
                self._tick()
                self._assign_target(gen.target, item, comp_scope)
                if all(self._eval_expr(cond, comp_scope) for cond in gen.ifs):
                    results.append(self._eval_expr(node.elt, comp_scope))
            return results
        raise TypeError(f"Unsupported expression node: {type(node).__name__}")

    def _eval_slice_or_index(self, node: ast.AST, scope: Dict[str, Any]) -> Any:
        if isinstance(node, ast.Slice):
            lower = self._eval_expr(node.lower, scope) if node.lower else None
            upper = self._eval_expr(node.upper, scope) if node.upper else None
            step = self._eval_expr(node.step, scope) if node.step else None
            return slice(lower, upper, step)
        return self._eval_expr(node, scope)
