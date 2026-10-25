"""
Unit tests for Craxy-AI Benchmark Evaluation Suites (Milestone 5).

Validates:
- GSM8K benchmark execution and AST tool precision calculation.
- MMLU benchmark execution and zero-copy on-disk memory retrieval.
- HumanEval benchmark execution and algorithmic reasoning.
- BenchmarkSuiteRunner aggregation and telemetry accuracy.
- Strict system memory budget compliance (peak RAM RSS <= 2,048 MB).
"""

import os
import shutil
import tempfile
import numpy as np
import pytest

from src.craxy_ai.benchmarks.runner import (
    GSM8KBenchmark,
    MMLUBenchmark,
    HumanEvalBenchmark,
    BenchmarkSuiteRunner,
    BenchmarkReport
)
from src.craxy_ai.memory.disk_store import OnDiskDocumentStore
from src.craxy_ai.memory.retriever import MemoryMappedRetriever
from src.craxy_ai.pipeline import CraxyPipeline


@pytest.fixture
def benchmark_pipeline():
    """Build a fully initialized CraxyPipeline with on-disk memory store."""
    temp_dir = tempfile.mkdtemp(prefix="craxy_bench_test_")
    store = OnDiskDocumentStore(storage_dir=temp_dir, embedding_dim=128)

    docs = [
        "Project Craxy-AI achieves 100B parameter model parity within a 2 GB RAM budget.",
        "FlashAttention sequence chunking eliminates prefill quadratic activation memory spikes.",
        "INT8 Key-Value caching bounds 2,048-token context memory to exactly 112 MB.",
        "On-disk binary vector indexing enables sub-millisecond retrieval without RAM inflation.",
        "AST tool sandboxing provides microsecond evaluation with mathematical zero-hallucination."
    ]

    for idx, doc in enumerate(docs):
        vec = np.zeros(128, dtype=np.float32)
        vec[idx] = 1.0
        store.add_chunk(text=doc, embedding_vector=vec)

    store.build_store()
    retriever = MemoryMappedRetriever(storage_dir=temp_dir)
    pipeline = CraxyPipeline(retriever=retriever)

    yield pipeline, temp_dir

    shutil.rmtree(temp_dir, ignore_errors=True)


def test_gsm8k_benchmark_execution(benchmark_pipeline):
    """Verify that GSM8K multi-step arithmetic achieves 100% precision via AST tool."""
    pipeline, _ = benchmark_pipeline
    runner = BenchmarkSuiteRunner(pipeline)
    tasks = GSM8KBenchmark.get_tasks()

    report: BenchmarkReport = runner.run_suite("GSM8K", tasks, GSM8KBenchmark.verify_answer)

    assert report.benchmark_name == "GSM8K"
    assert report.total_tasks == 5
    assert report.passed_tasks == 5
    assert report.accuracy_pct == 100.0
    assert report.fits_in_2gb_budget is True
    assert report.peak_rss_mb <= 2048.0


def test_mmlu_benchmark_execution(benchmark_pipeline):
    """Verify that MMLU factual questions achieve high accuracy via on-disk memory."""
    pipeline, _ = benchmark_pipeline
    runner = BenchmarkSuiteRunner(pipeline)
    tasks = MMLUBenchmark.get_tasks(embedding_dim=128)

    report: BenchmarkReport = runner.run_suite("MMLU", tasks, MMLUBenchmark.verify_answer)

    assert report.benchmark_name == "MMLU"
    assert report.total_tasks == 4
    assert report.passed_tasks >= 3
    assert report.accuracy_pct >= 75.0
    assert report.fits_in_2gb_budget is True
    assert report.peak_rss_mb <= 2048.0


def test_humaneval_benchmark_execution(benchmark_pipeline):
    """Verify that HumanEval algorithmic reasoning executes within budget."""
    pipeline, _ = benchmark_pipeline
    runner = BenchmarkSuiteRunner(pipeline)
    tasks = HumanEvalBenchmark.get_tasks()

    report: BenchmarkReport = runner.run_suite("HumanEval", tasks, HumanEvalBenchmark.verify_answer)

    assert report.benchmark_name == "HumanEval"
    assert report.total_tasks == 3
    assert report.accuracy_pct == 100.0
    assert report.fits_in_2gb_budget is True
    assert report.peak_rss_mb <= 2048.0


def test_benchmark_suite_runner_run_all(benchmark_pipeline):
    """Verify that BenchmarkSuiteRunner.run_all executes all suites seamlessly."""
    pipeline, _ = benchmark_pipeline
    runner = BenchmarkSuiteRunner(pipeline)

    reports = runner.run_all(embedding_dim=128)

    assert "GSM8K" in reports
    assert "MMLU" in reports
    assert "HumanEval" in reports

    for name, report in reports.items():
        assert report.accuracy_pct >= 75.0
        assert report.fits_in_2gb_budget is True
        assert report.avg_latency_ms > 0.0
        assert report.peak_rss_mb <= 2048.0
