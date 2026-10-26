"""Data models and configuration for Craxy-AI Interactive Chat Session.

Provides:
- ChatMessage: immutable-style conversation turn dataclass.
- ChatConfig: session configuration and physical memory boundaries.
"""

from dataclasses import dataclass, field
import time
from typing import Any, Dict, List, Optional


@dataclass
class ChatMessage:
    """Represents a single conversational turn in Craxy-AI dialogue."""

    role: str
    content: str
    timestamp: float = field(default_factory=time.time)
    tool_calls: Optional[List[Dict[str, Any]]] = None

    def to_prompt_line(self) -> str:
        """Format message as formatted conversational dialogue line."""
        prefix = self.role.capitalize()
        return f"{prefix}: {self.content}"


@dataclass
class ChatConfig:
    """Configuration parameters for interactive chat sessions."""

    system_prompt: str = "You are Craxy-AI, an autonomous frontier-grade AI operating within 2 GB RAM."
    max_history_tokens: int = 131072
    max_visible_history_turns: int = 16
    use_gguf: bool = False
    gguf_path: Optional[str] = None
    use_paged_kv: bool = False
    paged_kv_dir: Optional[str] = None
    storage_dir: Optional[str] = None
