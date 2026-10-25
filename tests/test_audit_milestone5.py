"""
Adversarial Red-Team Audit Suite for Milestone 5 (Benchmark Evaluation Harness).

Audits and stresses:
1. Empty task suite resilience (guarding against ZeroDivisionError).
2. Verifier rejection of incorrect/hallucinated answers.
3. Telemetry numerical integrity (latency, throughput, accuracy).
4. Memory allocation stability across 10 complete benchmark cycles.
5. System RAM ceiling compliance under sustained benchmark evaluation.
"""

import gc
import os
import shutil
import tempfile
from typing import Generator, Tuple
import numpy as np
import pytest

from src.craxy_ai.benchmarks.runner import (
    BenchmarkSuiteRunner,
    BenchmarkTask,
    BenchmarkReport,
    GSM8KBenchmark,
    MMLUBenchmark,
    HumanEvalBenchmark,
)
from src.craxy_ai.config import SYSTEM_LIMITS
from src.craxy_ai.memory.disk_store import OnDiskDocumentStore
from src.craxy_ai.memory.retriever import MemoryMappedRetriever
from src.craxy_ai.pipeline import CraxyPipeline


@pytest.fixture
def audit_benchmark_pipeline() -> Generator[Tuple[CraxyPipeline, str], None, None]:
    """Provide a CraxyPipeline backed by on-disk vector memory."""
    temp_dir = tempfile.mkdtemp(prefix="audit_bench_pipe_")
    store = OnDiskDocumentStore(storage_dir=temp_dir, embedding_dim=64)

    test_docs = [
        "FlashAttention sequence chunking limits activation surge.",
        "INT8 Key-Value cache keeps 2k context memory at 112 MB.",
        "AST tool sandboxing provides microsecond zero-error math."
    ]

    for idx, doc in enumerate(test_docs):
        vec = np.zeros(64, dtype=np.float32)
        vec[idx] = 1.0
        store.add_chunk(text=doc, embedding_vector=vec)

    store.build_store()
    retriever = MemoryMappedRetriever(storage_dir=temp_dir)
    pipeline = CraxyPipeline(retriever=retriever)

    yield pipeline, temp_dir

    shutil.rmtree(temp_dir, ignore_errors=True)


def test_audit_empty_benchmark_tasks_resilience():
    """Verify BenchmarkSuiteRunner handles empty task lists without division by zero."""
    # Arrange
    pipeline = CraxyPipeline()
    runner = BenchmarkSuiteRunner(pipeline)

    # Act
    report: BenchmarkReport = runner.run_suite(
        name="EmptySuite",
        tasks=[],
        verifier=GSM8KBenchmark.verify_answer
    )

    # Assert
    assert report.total_tasks == 0
    assert report.passed_tasks == 0
    assert report.accuracy_pct == 0.0
    assert report.avg_latency_ms == 0.0
    assert report.tokens_per_sec == 0.0
    assert report.fits_in_2gb_budget is True


def test_audit_benchmark_verifiers_reject_wrong_answers():
    """Verify that benchmark verifiers strictly reject mismatched answers."""
    # Arrange & Act & Assert
    # GSM8K
    assert GSM8KBenchmark.verify_answer("Computed answer: 548", "548") is True
    assert GSM8KBenchmark.verify_answer("Computed answer: 548", "999") is False

    # MMLU
    assert MMLUBenchmark.verify_answer("Document specifies FlashAttention mechanism.", "FlashAttention") is True
    assert MMLUBenchmark.verify_answer("Document discusses CPU scheduling.", "FlashAttention") is False

    # HumanEval
    assert HumanEvalBenchmark.verify_answer("Analyzed binary search algorithm.", "binary search") is True
    assert HumanEvalBenchmark.verify_answer("Analyzed quicksort routine.", "binary search") is False


def test_audit_adversarial_incorrect_math_task():
    """Verify benchmark accurately records zero accuracy when model fails adversarial task."""
    # Arrange
    pipeline = CraxyPipeline()
    runner = BenchmarkSuiteRunner(pipeline)
    
    # Task expects an impossible answer
    impossible_task = BenchmarkTask(
        task_id="adv_01",
        category="math",
        prompt="Calculate 10 + 10",
        expected_answer="999999"
    )

    # Act
    report: BenchmarkReport = runner.run_suite(
        name="AdversarialMath",
        tasks=[impossible_task],
        verifier=GSM8KBenchmark.verify_answer
    )

    # Assert
    assert report.total_tasks == 1
    assert report.passed_tasks == 0
    assert report.accuracy_pct == 0.0
    assert report.avg_latency_ms > 0.0
    assert report.fits_in_2gb_budget is True


def test_audit_sustained_benchmark_execution_memory_stability(audit_benchmark_pipeline):
    """Verify 10 repeated benchmark suite runs maintain allocator stability under RAM ceiling."""
    # Arrange
    pipeline, _ = audit_benchmark_pipeline
    runner = BenchmarkSuiteRunner(pipeline)
    gc.collect()
    rss_initial = pipeline.get_memory_summary()["current_rss_mb"]

    # Act: Run all suites 10 times consecutively
    for _ in range(10):
        reports = runner.run_all(embedding_dim=64)
        for report in reports.values():
            assert report.fits_in_2gb_budget is True
            assert report.peak_rss_mb <= SYSTEM_LIMITS.max_ram_rss_mb

    gc.collect()
    rss_final = pipeline.get_memory_summary()["current_rss_mb"]
    memory_drift = rss_final - rss_initial

    # Assert
    assert memory_drift < 35.0
    assert rss_final <= SYSTEM_LIMITS.max_ram_rss_mb
