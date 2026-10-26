"""
System configuration and hard boundary constraints for Craxy-AI.

Implements architectural limits codified in ADR-001 through ADR-006:
- Maximum Resident Set Size: 2,048 MB
- Maximum Non-Volatile Storage: 100 GB
- Maximum Chunked Prefill Sequence Length: 512 tokens
- Minimum Retrieval Similarity Threshold: 0.65
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class CraxySystemLimits:
    """Hard physical constraints enforced by the Craxy-AI runtime."""
    max_ram_rss_mb: float = 2048.0
    max_storage_bytes: int = 100 * 10**9  # 100 GB
    max_prefill_chunk_tokens: int = 512
    max_context_window_tokens: int = 2048
    max_paged_context_window_tokens: int = 131072  # 128K tokens context
    min_retrieval_similarity_threshold: float = 0.65
    max_math_exponent_bound: int = 10000
    target_tokens_per_second: float = 2.0


@dataclass(frozen=True)
class ModelArchitectureConfig:
    """Configuration dimensions for the in-RAM compact generative backbone."""
    num_layers: int = 28
    hidden_dim: int = 2048
    num_query_heads: int = 16
    num_kv_heads: int = 8
    head_dim: int = 128
    weight_quant_bits: int = 4
    kv_quant_bits: int = 8


# Global default configuration instance
SYSTEM_LIMITS = CraxySystemLimits()
DEFAULT_MODEL_CONFIG = ModelArchitectureConfig()
