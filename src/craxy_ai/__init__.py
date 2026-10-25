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

__all__ = [
    "CraxyPipeline",
    "PipelineState",
    "PipelineOutput",
    "SYSTEM_LIMITS",
    "CraxySystemLimits",
    "cli_main",
    "__version__",
]
