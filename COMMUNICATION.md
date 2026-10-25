# ULTRON Communication and Turn State

> **Manual handoff only.** This file records which agent is authorized by the user. It is not a real mutex, does not watch files, and cannot automatically start or stop a Codex session. Never activate both agents at the same time.

## Current state

- **STATE:** `ULTRON_2_ACTIVE`
- **TURN:** `8`
- **CURRENT OWNER:** `ULTRON 2`
- **LAST COMPLETED AGENT:** `ULTRON 1`
- **LAST COMPLETED COMMIT:** `afc79d7`
- **NEXT AGENT:** `ULTRON 1`
- **LAST HANDOFF SUMMARY:** ULTRON 1 completed Turn 7 / Milestone 1. ULTRON 2 claims Turn 8 to audit `src/craxy_ai/config.py` and `src/craxy_ai/tools/math_evaluator.py`, perform fuzzing and security tests on `SafeMathEvaluator`, and approve Milestone 2.
- **NEXT TASK:** Turn 8 audit: Verification of Milestone 1 production files, red-team fuzz testing of SafeMathEvaluator, and approval for Milestone 2.
- **UPDATED AT (UTC):** `2026-10-25T12:06:00Z`

## Allowed state transitions

1. `READY_FOR_ULTRON_1` → `ULTRON_1_ACTIVE` → `READY_FOR_ULTRON_2`
2. `READY_FOR_ULTRON_2` → `ULTRON_2_ACTIVE` → `READY_FOR_ULTRON_1`
3. Either active state → `PAUSED_FOR_HUMAN` if there is a conflict, unexpected dirty work, failed commit, unclear state, or other blocker.
4. Only the human may resolve `PAUSED_FOR_HUMAN` after checking the working tree and last commit.

## Instructions for whichever agent is prompted

- Start only if the state is exactly your matching `READY_FOR_...` state.
- If the state names the other agent as active, or is unclear/unexpected, do not modify any files. Report that you are blocked.
- Claim the turn by setting the matching `..._ACTIVE` state, recording a UTC timestamp, turn number, current `HEAD`, task, and owner, then commit this claim before beginning research.
- At completion, append the report to `research/ULTRON_RESEARCH_LOG.md`, record any experiments/decisions, and commit the completed work.
- In the final handoff commit, change the state to the other agent's `READY_FOR_...` state, set owner to `NONE`, and record the commit hash/summary/next task. Then stop.
- Do not leave the state as ready if your commit did not succeed. Change it to `PAUSED_FOR_HUMAN` if safe handoff is impossible.

## Turn history

| Turn | Agent | Start UTC | End UTC | Completion commit | Status | Summary |
|---:|---|---|---|---|---|---|
| 0 | NONE | - | - | b510f6f | INITIALIZED | Repository initialized with LICENSE and README. |
| 1 | ULTRON 1 | 2026-10-25T11:10:40Z | 2026-10-25T11:20:00Z | 71d9c24 | COMPLETED | Established charter, plan, ADRs, EXP-001 benchmark; handed off to ULTRON 2. |
| 2 | ULTRON 2 | 2026-10-25T11:28:00Z | 2026-10-25T11:38:00Z | d585eb7 | COMPLETED | EXP-001 repro; EXP-002 activation surge & MoE backbone proof; ADR-004 chunked prefill. |
| 3 | ULTRON 1 | 2026-10-25T11:37:43Z | 2026-10-25T11:45:00Z | 830e389 | COMPLETED | Implemented EXP-003 on-disk retrieval engine (0.776 ms latency, 10 MB RAM delta, 12/12 tests pass); handed off to ULTRON 2. |
| 4 | ULTRON 2 | 2026-10-25T11:46:00Z | 2026-10-25T11:50:00Z | faa14f8 | COMPLETED | EXP-004 concurrent RAM audit (1.14 GB peak); DiskANN scaling proof; ADR-005 AST math sandbox. |
| 5 | ULTRON 1 | 2026-10-25T11:50:00Z | 2026-10-25T11:55:00Z | 585a015 | COMPLETED | Formulated definitive Craxy-AI Architectural Master Blueprint (ARCHITECTURE_BLUEPRINT.md); handed off to ULTRON 2. |
| 6 | ULTRON 2 | 2026-10-25T11:53:00Z | 2026-10-25T11:58:00Z | d1dd88c | COMPLETED | Audited Master Blueprint (EXP-005); adopted ADR-006; verified relevance gate & context ceiling; handed off to ULTRON 1. |
| 7 | ULTRON 1 | 2026-10-25T12:00:19Z | 2026-10-25T12:05:00Z | 5ede113 | COMPLETED | Implemented Milestone 1 package, config.py, and SafeMathEvaluator (28/28 tests pass); handed off to ULTRON 2. |
| 8 | ULTRON 2 | 2026-10-25T12:06:00Z | IN_PROGRESS | - | ACTIVE | Turn claimed: independent audit and red-team test of Milestone 1 package. |

## Human coordinator checklist

- [ ] Confirm the previous Codex turn has fully returned its final response.
- [ ] Confirm the previous agent's final commit exists and the working tree has no unexpected uncommitted changes.
- [ ] Confirm `STATE` is `READY_FOR_...` for the agent you intend to start.
- [ ] Start only that agent and paste its prompt from `prompts/FIRST_PROMPTS.md`.
- [ ] Do not send another prompt to the other agent until the current turn has ended.
