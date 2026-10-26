"""Adversarial Red-Team Audit for Phase 6 Interactive Chat Mode (Turn 61).

Audits:
1. Slash command edge cases, case-insensitivity, and arguments tolerance.
2. Oversized input stress (50k characters) and context preservation.
3. Multi-instance session isolation and independent paged KV disk lifecycles.
4. Pipeline execution error containment and history consistency.
5. 100-cycle continuous multi-turn dialogue memory stability (RSS drift < 20.0 MB).
"""

import gc
import os
from unittest.mock import patch
import psutil
import pytest

from src.craxy_ai.chat import CraxyChatSession
from src.craxy_ai.chat_types import ChatConfig


def test_audit_chat_slash_command_robustness():
    """Verify slash command parser withstands uppercase, whitespace, and extra arguments."""
    session = CraxyChatSession()
    try:
        # Uppercase and extra arguments
        help_res = session.handle_slash_command("  /HELP please  ")
        assert "Available commands:" in help_res

        # Add message then clear with uppercase and arguments
        session.send_message("Calculate 10 + 20")
        assert session.get_turn_count() == 1

        info_res = session.handle_slash_command("/INFO verbose")
        assert "Active Turns: 1" in info_res

        clear_res = session.handle_slash_command("  /CLEAR all history  ")
        assert "Conversation history cleared." in clear_res
        assert session.get_turn_count() == 0

        # Mixed-case quit
        quit_res = session.handle_slash_command("/QuIt")
        assert "Session terminated. Goodbye!" in quit_res
        assert session.is_active is False
    finally:
        session.close()


def test_audit_chat_oversized_payload_handling():
    """Verify session handles 50,000-character input without process abort."""
    session = CraxyChatSession()
    try:
        huge_text = "What is 10 + 20? " + ("A" * 50000)
        res = session.send_message(huge_text)
        assert res != ""
        assert session.get_turn_count() == 1
    finally:
        session.close()


def test_audit_chat_concurrent_instance_isolation():
    """Verify multiple chat instances maintain independent histories and storage files."""
    cfg1 = ChatConfig(use_paged_kv=True, paged_kv_dir="auto")
    cfg2 = ChatConfig(use_paged_kv=True, paged_kv_dir="auto")

    session1 = CraxyChatSession(config=cfg1)
    session2 = CraxyChatSession(config=cfg2)

    dir1 = session1._temp_paged_dir
    dir2 = session2._temp_paged_dir

    assert dir1 != dir2
    assert os.path.exists(dir1)
    assert os.path.exists(dir2)

    try:
        session1.send_message("Calculate 50 * 2")
        session2.send_message("Sort array items = [5, 1, 3]")

        # Check turn isolation
        assert session1.get_turn_count() == 1
        assert session2.get_turn_count() == 1
        assert "[Tool Result: 100]" in session1.get_history()[2].content
        assert "[Tool Result: [1, 3, 5]]" in session2.get_history()[2].content

        # Close session 1; session 2 must remain fully operational
        session1.close()
        assert not os.path.exists(dir1)
        assert os.path.exists(dir2)

        res2_more = session2.send_message("Calculate 7 * 7")
        assert "[Tool Result: 49]" in res2_more
        assert session2.get_turn_count() == 2
    finally:
        session2.close()

    assert not os.path.exists(dir2)


def test_audit_chat_pipeline_failure_containment():
    """Verify pipeline execution errors are safely trapped and recorded in history."""
    session = CraxyChatSession()
    try:
        with patch.object(session.pipeline, "run_query", side_effect=RuntimeError("Simulated generation crash")):
            res = session.send_message("Trigger failure")
            assert "error" in res.lower()
            assert "Simulated generation crash" in res

        # History must contain user query and assistant error response
        history = session.get_history()
        assert len(history) == 3
        assert history[1].role == "user"
        assert history[2].role == "assistant"
        assert "Simulated generation crash" in history[2].content
    finally:
        session.close()


def test_audit_chat_100_cycle_memory_stability():
    """Verify 100 consecutive chat turns show bounded memory growth (< 20.0 MB)."""
    cfg = ChatConfig(max_visible_history_turns=10)
    session = CraxyChatSession(config=cfg)

    try:
        session.send_message("warmup")
        gc.collect()
        process = psutil.Process(os.getpid())
        rss_start = process.memory_info().rss / (1024 * 1024)

        for cycle in range(100):
            if cycle % 2 == 0:
                resp = session.send_message(f"Calculate {cycle} + 10")
                assert f"[Tool Result: {cycle + 10}]" in resp
            else:
                resp = session.send_message(f"Sort array nums = [{cycle}, 0, 42]")
                assert "[Tool Result:" in resp

        assert session.get_turn_count() == 101

        gc.collect()
        rss_end = process.memory_info().rss / (1024 * 1024)
        growth_mb = rss_end - rss_start

        assert growth_mb < 20.0, f"Memory growth {growth_mb:.2f} MB exceeds 20.0 MB"
        assert rss_end < 2048.0, f"Peak memory {rss_end:.2f} MB exceeds 2048.0 MB budget"
    finally:
        session.close()
