"""
Craxy-AI: Tri-Tier High-Quality AI Inference Engine.

Engineered to run 100B-grade AI models within a strict 2,048 MB system RAM budget
and 100 GB NVMe storage budget through decoupled parametric reasoning,
on-disk non-parametric memory retrieval, and in-process tool augmentation.
"""

__version__ = "0.1.0"

from src.craxy_ai.config import SYSTEM_LIMITS, CraxySystemLimits
from src.craxy_ai.pipeline import CraxyPipeline, PipelineState, PipelineOutput
from src.craxy_ai.cli import main as cli_main
from src.craxy_ai.core.model import CraxyNeuralCore
from src.craxy_ai.tools.math_evaluator import SafeMathEvaluator
from src.craxy_ai.tools.logic_evaluator import SafeLogicEvaluator
from src.craxy_ai.tools.code_agent import AgenticCodeEngine, CraxyCodeAgent
from src.craxy_ai.core.early_exit import EarlyExitClassifier
from src.craxy_ai.core.paged_kv import CraxyPagedKVCache, PagedKVConfig
from src.craxy_ai.core.gguf_engine import CraxyGGUFEngine, GGUFEngineConfig
from src.craxy_ai.memory.graph_index import VamanaGraphStore

__all__ = [
    "CraxyPipeline",
    "PipelineState",
    "PipelineOutput",
    "CraxyNeuralCore",
    "CraxyPagedKVCache",
    "PagedKVConfig",
    "CraxyGGUFEngine",
    "GGUFEngineConfig",
    "SafeMathEvaluator",
    "SafeLogicEvaluator",
    "AgenticCodeEngine",
    "CraxyCodeAgent",
    "EarlyExitClassifier",
    "VamanaGraphStore",
    "SYSTEM_LIMITS",
    "CraxySystemLimits",
    "cli_main",
    "__version__",
]

