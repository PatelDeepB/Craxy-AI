"""Command-Line Interface (CLI) for Craxy-AI.

Implements Milestone 6, Phase 5, and Phase 6 Frontier CLI Exposure:
- Displays system physical boundaries, component dimensions, and real-time memory headroom.
- Executes queries through the Tri-Tier Unified Pipeline with optional GGUF or Paged KV.
- Validates and sandboxes code patches via embedded CraxyCodeAgent.
- Runs comprehensive evaluation suites (GSM8K, MMLU, HumanEval) with telemetry tables.
- Launches interactive terminal chat session (--chat) with 128K context paged memory.
"""

import argparse
import os
import shutil
import sys
import tempfile
from typing import Any, List, Optional, Tuple

from src.craxy_ai.benchmarks.runner import BenchmarkSuiteRunner
from src.craxy_ai.cli_helpers import (
    display_info,
    print_benchmark_table,
    print_query_output,
)
from src.craxy_ai.memory.retriever import MemoryMappedRetriever
from src.craxy_ai.pipeline import CraxyPipeline, PipelineOutput
from src.craxy_ai.tools.code_agent import CraxyCodeAgent


def _add_primary_modes(parser: argparse.ArgumentParser) -> None:
    """Register mutually exclusive primary operation modes."""
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
        "--chat",
        action="store_true",
        help="Launch interactive terminal multi-turn chat session."
    )
    group.add_argument(
        "--benchmark",
        action="store_true",
        help="Execute all standard benchmark suites (GSM8K, MMLU, HumanEval)."
    )
    group.add_argument(
        "--code-patch",
        type=str,
        metavar="CODE",
        help="Validate and test Python code via CraxyCodeAgent."
    )
    group.add_argument(
        "--eval",
        "--dashboard",
        dest="eval_mode",
        action="store_true",
        help="Run comprehensive Frontier Evaluation Suite & Telemetry Dashboard."
    )



def _add_gguf_flags(parser: argparse.ArgumentParser) -> None:
    """Register GGUF engine configuration flags."""
    parser.add_argument(
        "--use-gguf",
        "--gguf",
        dest="use_gguf",
        nargs="?",
        const=True,
        default=False,
        metavar="PATH",
        help="Route execution to quantized GGUF neural engine."
    )
    parser.add_argument(
        "--gguf-path",
        type=str,
        default=None,
        metavar="PATH",
        help="Explicit path to quantized GGUF model file."
    )


def _add_storage_flags(parser: argparse.ArgumentParser) -> None:
    """Register memory and storage backing flags."""
    parser.add_argument(
        "--storage-dir",
        type=str,
        default=None,
        metavar="PATH",
        help="Optional path to on-disk vector store directory."
    )
    parser.add_argument(
        "--paged-kv",
        dest="use_paged_kv",
        nargs="?",
        const=True,
        default=False,
        metavar="DIR",
        help="Enable NVMe-paged 128K context KV cache."
    )
    parser.add_argument(
        "--paged-kv-dir",
        type=str,
        default=None,
        metavar="DIR",
        help="Explicit storage directory for paged KV cache."
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser for Craxy-AI."""
    parser = argparse.ArgumentParser(
        prog="craxy-ai",
        description="Craxy-AI: Tri-Tier High-Quality AI Inference Engine within 2 GB RAM."
    )
    _add_primary_modes(parser)
    _add_gguf_flags(parser)
    _add_storage_flags(parser)
    return parser


def _init_query_pipeline(
    storage_dir: Optional[str],
    use_gguf: bool,
    use_paged_kv: bool,
    gguf_path: Optional[str],
    paged_kv_dir: Optional[str],
) -> Tuple[CraxyPipeline, Any, Any, Optional[str]]:
    """Initialize Tri-Tier pipeline and components with temporary resource tracking."""
    retriever = MemoryMappedRetriever(storage_dir=storage_dir) if storage_dir and os.path.exists(storage_dir) else None
    gguf_engine = None
    if use_gguf or gguf_path:
        from src.craxy_ai.core.gguf_engine import CraxyGGUFEngine
        from src.craxy_ai.core.gguf_types import GGUFEngineConfig
        target_path = gguf_path if gguf_path and isinstance(gguf_path, str) else None
        gguf_engine = CraxyGGUFEngine(GGUFEngineConfig(model_path=target_path))

    paged_cache = None
    temp_dir = None
    if use_paged_kv or paged_kv_dir:
        from src.craxy_ai.core.paged_kv import CraxyPagedKVCache
        if not paged_kv_dir or paged_kv_dir == "auto":
            temp_dir = tempfile.mkdtemp(prefix="craxy_cli_paged_")
            paged_kv_dir = temp_dir
        paged_cache = CraxyPagedKVCache(backing_dir=paged_kv_dir)

    pipeline = CraxyPipeline(retriever=retriever, gguf_engine=gguf_engine, paged_kv=paged_cache)
    return pipeline, paged_cache, gguf_engine, temp_dir


def execute_query(
    query_text: str,
    storage_dir: Optional[str] = None,
    use_gguf: bool = False,
    use_paged_kv: bool = False,
    gguf_path: Optional[str] = None,
    paged_kv_dir: Optional[str] = None,
) -> int:
    """Execute a query through the Tri-Tier pipeline and print results."""
    pipeline, paged_cache, gguf_engine, temp_dir = _init_query_pipeline(
        storage_dir, use_gguf, use_paged_kv, gguf_path, paged_kv_dir
    )
    try:
        output: PipelineOutput = pipeline.run_query(user_prompt=query_text, use_gguf=bool(use_gguf or gguf_path))
        print_query_output(output)
        return 0
    except Exception as err:
        print("-" * 65)
        print("Craxy-AI Query Execution Error:")
        print(f"Error:                  {err}")
        print("-" * 65)
        return 1
    finally:
        if paged_cache:
            paged_cache.close()
        if gguf_engine:
            gguf_engine.close()
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)


def execute_code_patch(code_text: str) -> int:
    """Validate Python code syntax using CraxyCodeAgent."""
    agent = CraxyCodeAgent()
    try:
        agent.validate_python_syntax(code_text)
        print("-" * 65)
        print("CraxyCodeAgent Syntax Validation:")
        print("Syntax Valid:           True")
        print(f"Workspace Root:         {agent.workspace_root}")
        print("-" * 65)
        return 0
    except Exception as err:
        print("-" * 65)
        print("CraxyCodeAgent Syntax Validation:")
        print("Syntax Valid:           False")
        print(f"Error:                  {err}")
        print("-" * 65)
        return 1


def execute_benchmark(storage_dir: Optional[str] = None) -> int:
    """Run all benchmark suites and display performance summary table."""
    retriever = None
    if storage_dir and os.path.exists(storage_dir):
        retriever = MemoryMappedRetriever(storage_dir=storage_dir)

    pipeline = CraxyPipeline(retriever=retriever)
    runner = BenchmarkSuiteRunner(pipeline=pipeline)

    print("=" * 85)
    print("Executing Craxy-AI Standardized Benchmarks (GSM8K, MMLU, HumanEval)...")
    print("=" * 85)
    reports = runner.run_all()
    print_benchmark_table(reports)
    return 0


def _handle_chat(args: argparse.Namespace) -> int:
    """Launch interactive terminal chat session with parsed configuration."""
    from src.craxy_ai.chat import run_interactive_chat
    from src.craxy_ai.chat_types import ChatConfig
    cfg = ChatConfig(
        use_gguf=bool(args.use_gguf),
        gguf_path=args.use_gguf if isinstance(args.use_gguf, str) else args.gguf_path,
        use_paged_kv=bool(args.use_paged_kv),
        paged_kv_dir=args.use_paged_kv if isinstance(args.use_paged_kv, str) else args.paged_kv_dir,
        storage_dir=args.storage_dir,
    )
    return run_interactive_chat(config=cfg)


def execute_eval() -> int:
    """Execute Frontier Evaluation Suite and render Telemetry Dashboard."""
    from src.craxy_ai.benchmarks.frontier_eval import FrontierEvaluationSuite, render_dashboard
    suite = FrontierEvaluationSuite()
    result = suite.run_full_evaluation()
    print(render_dashboard(result))
    return 0 if result.all_passed else 1


def main(argv: Optional[List[str]] = None) -> int:
    """CLI application entrypoint."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.info:
        return display_info()
    if args.chat:
        return _handle_chat(args)
    if args.query:
        is_gguf = bool(args.use_gguf)
        gguf_p = args.use_gguf if isinstance(args.use_gguf, str) else args.gguf_path
        is_paged = bool(args.use_paged_kv)
        paged_d = args.use_paged_kv if isinstance(args.use_paged_kv, str) else args.paged_kv_dir
        return execute_query(
            query_text=args.query,
            storage_dir=args.storage_dir,
            use_gguf=is_gguf,
            use_paged_kv=is_paged,
            gguf_path=gguf_p,
            paged_kv_dir=paged_d,
        )
    if args.code_patch:
        return execute_code_patch(code_text=args.code_patch)
    if args.benchmark:
        return execute_benchmark(storage_dir=args.storage_dir)
    if args.eval_mode:
        return execute_eval()
    return 1



if __name__ == "__main__":
    sys.exit(main())
