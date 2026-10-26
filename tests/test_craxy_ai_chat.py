"""Tests for Phase 6 Interactive Terminal Chat Mode & Multi-Turn Paged Memory.

Validates:
- CraxyChatSession lifecycle, turn tracking, and role-attributed history.
- Built-in slash commands (/help, /info, /clear, /exit, /quit).
- Autonomous tool execution (arithmetic, logic) within conversational turns.
- 128K NVMe-paged KV cache block ingestion and deterministic cleanup.
- Terminal stream I/O loop and CLI --chat entrypoint.
"""

import io
import os
from unittest.mock import MagicMock, patch
import pytest

from src.craxy_ai.chat import CraxyChatSession, run_interactive_chat
from src.craxy_ai.chat_types import ChatConfig, ChatMessage
from src.craxy_ai.cli import main
from src.craxy_ai.core.gguf_types import GGUFGenerationResult


def test_chat_session_initialization_and_system_prompt():
    """Verify session starts with system prompt and zero conversational turns."""
    custom_prompt = "You are a specialized code and math assistant."
    cfg = ChatConfig(system_prompt=custom_prompt)
    session = CraxyChatSession(config=cfg)

    try:
        assert session.is_active is True
        assert session.get_turn_count() == 0
        history = session.get_history()
        assert len(history) == 1
        assert history[0].role == "system"
        assert history[0].content == custom_prompt
    finally:
        session.close()


def test_chat_session_slash_commands():
    """Verify built-in slash meta-commands behave deterministically."""
    session = CraxyChatSession()
    try:
        # /help
        help_out = session.handle_slash_command("/help")
        assert "Available commands:" in help_out
        assert "/info" in help_out
        assert "/clear" in help_out

        # Add a mock message and test /info
        session.send_message("Calculate 15 + 15")
        assert session.get_turn_count() == 1
        info_out = session.handle_slash_command("/info")
        assert "Active Turns: 1" in info_out
        assert "Process RSS:" in info_out
        assert "Headroom:" in info_out

        # /clear
        clear_out = session.handle_slash_command("/clear")
        assert "Conversation history cleared." in clear_out
        assert session.get_turn_count() == 0
        assert len(session.get_history()) == 1

        # Unknown command
        unknown_out = session.handle_slash_command("/foobar")
        assert "Unknown command: /foobar" in unknown_out

        # /exit
        exit_out = session.handle_slash_command("/exit")
        assert "Session terminated. Goodbye!" in exit_out
        assert session.is_active is False
    finally:
        session.close()


def test_chat_session_multi_turn_dialogue_and_tool_calling():
    """Verify tool dispatching and history retention across multiple turns."""
    session = CraxyChatSession()
    try:
        # Turn 1: Math calculation
        res1 = session.send_message("What is 25 * 4 + 50?")
        assert "[Tool Result: 150]" in res1
        assert session.get_turn_count() == 1

        # Turn 2: Logic sorting
        res2 = session.send_message("Sort array nums = [8, 2, 6]")
        assert "[Tool Result: [2, 6, 8]]" in res2
        assert session.get_turn_count() == 2

        # Verify history structure
        history = session.get_history()
        assert len(history) == 5  # 1 system + 2 user + 2 assistant
        assert history[1].role == "user"
        assert history[2].role == "assistant"
        assert history[3].role == "user"
        assert history[4].role == "assistant"
    finally:
        session.close()


def test_chat_session_with_paged_kv_cache():
    """Verify conversational tokens stream to NVMe-paged cache and clean up."""
    cfg = ChatConfig(use_paged_kv=True, paged_kv_dir="auto")
    session = CraxyChatSession(config=cfg)

    temp_dir = session._temp_paged_dir
    assert temp_dir is not None
    assert os.path.exists(temp_dir)

    try:
        session.send_message("Calculate 120 / 3")
        session.send_message("Sort array items = [10, 5, 20]")

        stats = session.paged_cache.get_telemetry()
        assert stats["allocated_blocks"] >= 2
    finally:
        session.close()

    assert not os.path.exists(temp_dir), "Temporary paged KV directory was not cleaned up!"


def test_chat_session_with_gguf_mock():
    """Verify chat routes prompt through GGUF engine when configured."""
    mock_res = GGUFGenerationResult(
        generated_text="GGUF assistant conversational turn.",
        prompt_tokens=12,
        completion_tokens=6,
        total_tokens=18,
        tokens_per_second=140.0,
        peak_rss_mb=420.0,
        duration_seconds=0.04,
        finish_reason="stop",
    )
    with patch("src.craxy_ai.core.gguf_engine.CraxyGGUFEngine") as mock_engine_cls:
        instance = MagicMock()
        instance.generate.return_value = mock_res
        mock_engine_cls.return_value = instance

        cfg = ChatConfig(use_gguf=True, gguf_path="weights.gguf")
        session = CraxyChatSession(config=cfg)
        try:
            res = session.send_message("Explain quantum superposition")
            assert "GGUF assistant conversational turn." in res
            assert session.get_turn_count() == 1
        finally:
            session.close()


def test_run_interactive_chat_stream_io():
    """Verify terminal stream loop processes inputs and terminates cleanly."""
    in_stream = io.StringIO("Calculate 200 * 2\n/info\n/quit\n")
    out_stream = io.StringIO()

    ret_code = run_interactive_chat(in_stream=in_stream, out_stream=out_stream)
    output = out_stream.getvalue()

    assert ret_code == 0
    assert "Craxy-AI Interactive Terminal Chat" in output
    assert "[Tool Result: 400]" in output
    assert "Active Turns: 1" in output
    assert "Session terminated. Goodbye!" in output


def test_cli_chat_flag_dispatch():
    """Verify that CLI --chat flag invokes interactive session cleanly."""
    mock_input = io.StringIO("/exit\n")
    mock_output = io.StringIO()

    with patch("sys.stdin", mock_input), patch("sys.stdout", mock_output):
        ret_code = main(["--chat"])

    captured = mock_output.getvalue()
    assert ret_code == 0
    assert "Craxy-AI Interactive Terminal Chat" in captured
    assert "Session terminated. Goodbye!" in captured
