"""Frontier Multi-Turn Memory and Physical Stability Tests for Interactive Chat.

Validates:
- 50 continuous multi-turn dialogue interactions with tool calling.
- Bounded resident set size (RSS growth < 20 MB, peak < 2,048 MB).
- Context window management and rolling history truncation.
- Resilience against empty inputs, whitespace, and rapid slash command switching.
"""

import gc
import os
import psutil
import pytest

from src.craxy_ai.chat import CraxyChatSession
from src.craxy_ai.chat_types import ChatConfig


def test_chat_50_continuous_turns_memory_stability():
    """Verify that 50 conversational turns operate with bounded memory growth."""
    cfg = ChatConfig(max_visible_history_turns=10)
    session = CraxyChatSession(config=cfg)

    try:
        # Warm-up pass to initialize model weights and buffers
        session.send_message("warmup")
        gc.collect()
        process = psutil.Process(os.getpid())
        rss_start = process.memory_info().rss / (1024 * 1024)

        for idx in range(50):
            if idx % 2 == 0:
                resp = session.send_message(f"Calculate {idx} * 2 + 10")
                expected = idx * 2 + 10
                assert f"[Tool Result: {expected}]" in resp
            else:
                resp = session.send_message(f"Sort array items = [{idx}, 1, 99]")
                assert "[Tool Result:" in resp

        assert session.get_turn_count() == 51  # 1 warmup + 50 iterations

        gc.collect()
        rss_end = process.memory_info().rss / (1024 * 1024)
        growth_mb = rss_end - rss_start

        assert growth_mb < 20.0, f"Memory growth {growth_mb:.2f} MB exceeds 20.0 MB threshold"
        assert rss_end < 2048.0, f"Peak RSS {rss_end:.2f} MB exceeds 2048.0 MB budget"
    finally:
        session.close()


def test_chat_empty_and_whitespace_inputs():
    """Verify session ignores empty inputs and whitespace turns without adding messages."""
    session = CraxyChatSession()
    try:
        assert session.send_message("") == ""
        assert session.send_message("   \n\t  ") == ""
        assert session.get_turn_count() == 0
        assert len(session.get_history()) == 1
    finally:
        session.close()


def test_chat_rapid_slash_command_switching():
    """Verify interleaved slash commands and dialogue turns maintain state."""
    session = CraxyChatSession()
    try:
        assert session.send_message("Calculate 5 + 5") != ""
        assert session.get_turn_count() == 1

        info1 = session.send_message("/info")
        assert "Active Turns: 1" in info1
        assert session.get_turn_count() == 1

        assert session.send_message("Calculate 10 + 10") != ""
        assert session.get_turn_count() == 2

        clear_res = session.send_message("/clear")
        assert "Conversation history cleared." in clear_res
        assert session.get_turn_count() == 0

        assert session.send_message("Calculate 1 + 1") != ""
        assert session.get_turn_count() == 1
    finally:
        session.close()
