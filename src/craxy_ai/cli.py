"""
Command-Line Interface (CLI) for Craxy-AI.

Implements Milestone 6 (Final Architectural Integration & Release Packaging):
- Displays system physical boundaries and real-time memory headroom.
- Executes natural language queries through the Tri-Tier Unified Pipeline.
- Runs comprehensive evaluation suites (GSM8K, MMLU, HumanEval) with telemetry tables.
"""

import argparse
import os
import sys
from typing import List, Optional

from src.craxy_ai.benchmarks.runner import BenchmarkSuiteRunner
from src.craxy_ai.config import SYSTEM_LIMITS, CraxySystemLimits
from src.craxy_ai.memory.retriever import MemoryMappedRetriever
from src.craxy_ai.pipeline import CraxyPipeline, PipelineOutput


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser for Craxy-AI."""
    parser = argparse.ArgumentParser(
        prog="craxy-ai",
        description="Craxy-AI: Tri-Tier High-Quality AI Inference Engine within 2 GB RAM."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--info",
        action="store_true",
        help="Display system limits, model parameters, and physical memory headroom."
    )
    group.add_argument(
        "--query",
        type=str,
        metavar="TEXT",
        help="Execute a prompt through the unified Tri-Tier inference pipeline."
    )
    group.add_argument(
        "--benchmark",
        action="store_true",
        help="Execute all standard benchmark suites (GSM8K, MMLU, HumanEval)."
    )
    parser.add_argument(
        "--storage-dir",
        type=str,
        default=None,
        metavar="PATH",
        help="Optional path to on-disk vector store directory."
    )
    return parser


def display_info(limits: CraxySystemLimits = SYSTEM_LIMITS) -> int:
    """Print system limits, component dimensions, and physical RAM headroom."""
    pipeline = CraxyPipeline(limits=limits)
    mem = pipeline.get_memory_summary()

    print("=" * 65)
    print("Craxy-AI Engine: Physical Architecture & Memory Boundaries")
    print("=" * 65)
    print(f"Max RAM RSS Budget:             {limits.max_ram_rss_mb:.1f} MB (2.0 GB)")
    print(f"Max Storage Budget:             {limits.max_storage_bytes / (10**9):.1f} GB NVMe")
    print(f"Max Prefill Chunk Size:         {limits.max_prefill_chunk_tokens} tokens")
    print(f"Min Retrieval Cosine Gate:      tau >= {limits.min_retrieval_similarity_threshold}")
    print(f"Max Context Window:             {limits.max_context_window_tokens} tokens")
    print("-" * 65)
    print("Phase 2 Optimizations:")
    print("  * Speculative Early Exit:     Layer 14/28 (50% latency saved on aligned tokens)")
    print("  * On-Disk Graph Index:        DiskANN Vamana (sub-2.0 ms beam routing)")
    print("  * Sandboxed Logic Engine:     SafeLogicEvaluator (10k step bounded AST)")
    print("-" * 65)
    print(f"Core INT4 Weights Footprint:    {mem['core_int4_mb']:.1f} MB")
    print(f"Bounded INT8 KV Cache Footprint:{mem['kv_cache_mb']:.1f} MB")
    print(f"Current Process RSS:            {mem['current_rss_mb']:.1f} MB")
    print(f"Remaining Headroom to Budget:   {mem['headroom_mb']:.1f} MB")
    print("=" * 65)
    return 0


def execute_query(query_text: str, storage_dir: Optional[str] = None) -> int:
    """Execute a query through the Tri-Tier pipeline and print results."""
    retriever = None
    if storage_dir and os.path.exists(storage_dir):
        retriever = MemoryMappedRetriever(storage_dir=storage_dir)

    pipeline = CraxyPipeline(retriever=retriever)
    output: PipelineOutput = pipeline.run_query(user_prompt=query_text)

    print("-" * 65)
    print("Craxy-AI Inference Output:")
    print(f"Response:               {output.response}")
    print(f"RAG Augmented:          {output.retrieval_augmented} ({output.retrieved_chunks_count} chunks)")
    print(f"Tool Dispatches:        {len(output.tool_calls)} executed")
    for call in output.tool_calls:
        print(f"  * [{call['tool']}] payload: {call['payload']} -> result: {call['output']}")
    print(f"Tokens Generated:       {output.tokens_generated}")
    print(f"Total Latency:          {output.total_latency_ms:.2f} ms")
    print(f"Peak Process RSS:       {output.peak_rss_mb:.2f} MB (Budget <= 2048 MB)")
    print("-" * 65)
    return 0


def execute_benchmark(storage_dir: Optional[str] = None) -> int:
    """Run all benchmark suites and display performance summary table."""
    retriever = None
    if storage_dir and os.path.exists(storage_dir):
        retriever = MemoryMappedRetriever(storage_dir=storage_dir)

    pipeline = CraxyPipeline(retriever=retriever)
    runner = BenchmarkSuiteRunner(pipeline=pipeline)

    print("=" * 75)
    print("Executing Craxy-AI Standardized Benchmarks (GSM8K, MMLU, HumanEval)...")
    print("=" * 75)
    reports = runner.run_all()

    print(f"{'Benchmark':<12} | {'Tasks':<6} | {'Passed':<6} | {'Accuracy':<10} | {'Throughput':<14} | {'Budget Status'}")
    print("-" * 75)
    for name, rep in reports.items():
        status = "PASS (<= 2GB)" if rep.fits_in_2gb_budget else "FAIL (> 2GB)"
        print(f"{name:<12} | {rep.total_tasks:<6} | {rep.passed_tasks:<6} | {rep.accuracy_pct:>6.1f}%    | {rep.tokens_per_sec:>6.1f} tok/s   | {status}")
    print("=" * 75)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    """CLI application entrypoint."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.info:
        return display_info()
    if args.query:
        return execute_query(query_text=args.query, storage_dir=args.storage_dir)
    if args.benchmark:
        return execute_benchmark(storage_dir=args.storage_dir)
    return 1


if __name__ == "__main__":
    sys.exit(main())
