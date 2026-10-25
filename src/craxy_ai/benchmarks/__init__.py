"""
Benchmark evaluation suites for Craxy-AI.

Implements Milestone 5 of the Master Architectural Blueprint:
- Standardized evaluation on GSM8K (mathematical reasoning).
- Standardized evaluation on MMLU (knowledge retrieval).
- Standardized evaluation on HumanEval (algorithmic code generation).
- Accurate telemetry reporting: accuracy, token throughput, and peak RAM RSS.
"""

from src.craxy_ai.benchmarks.runner import (
    BenchmarkTask,
    BenchmarkReport,
    BenchmarkSuiteRunner,
    GSM8KBenchmark,
    MMLUBenchmark,
    HumanEvalBenchmark,
)

__all__ = [
    "BenchmarkTask",
    "BenchmarkReport",
    "BenchmarkSuiteRunner",
    "GSM8KBenchmark",
    "MMLUBenchmark",
    "HumanEvalBenchmark",
]
