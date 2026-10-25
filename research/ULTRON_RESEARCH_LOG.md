# ULTRON Research Log

## Turn 1: Research Mapping, Memory Accounting, & Sequential Streaming Feasibility

- **Agent:** ULTRON 1 (Architect & Experimental Builder)
- **Turn Number:** 1
- **Date/Time (UTC):** 2026-10-25T11:15:52Z
- **Task:** Map research problem and prior art, define operational benchmark and research charter, produce staged research plan, and execute initial memory streaming benchmark (EXP-001).
- **Commit Hash:** `285b2bb` (work units: `e7e4b48`, `120df2b`, `285b2bb`)

---

### 1. Executive Result
In this foundational turn, ULTRON 1 established the formal `RESEARCH_CHARTER.md`, operational quality metrics (parity against Llama-3-70B and Qwen-2.5-72B across MMLU, GSM8K, and HumanEval), and a 5-stage research roadmap in `research/PLAN.md`. We executed Experiment EXP-001, measuring PyTorch runtime RSS (190.19 MB), quantifying KV cache scaling (256 MB at 2k context vs 1,024 MB at 8k context in FP16), and demonstrating that dynamic sequential layer streaming over 32 layers runs at a peak RSS of only 288.39 MB with zero memory drift (0.01 MB). Crucially, the measured disk I/O latency (1.627 seconds for a 2.0 GB layer pass at 1,259 MB/s) empirically refutes naive full-model dense streaming for 100B models, demonstrating that the architecture must pivot to either Sparse MoE expert paging or Non-Parametric Retrieval Augmentation.

---

### 2. Prompt Analysis & Discrepancy Reconciliation
As requested by the prompt ("analyse everything first and if you start work make sure that other agent will know"):
1. **Missing Charter:** `RESEARCH_CHARTER.md` was referenced in `AGENTS.md` and the prompt, but did not exist in the repository. ULTRON 1 formally created `RESEARCH_CHARTER.md` based on the 2 GB RAM and 100 GB storage goals.
2. **Role File Path Discrepancy:** The prompt and `AGENTS.md` referenced `prompts/ULTRON_1.md` and `prompts/ULTRON_2.md`, but the files were located in root. ULTRON 1 created `prompts/ULTRON_1.md` and `prompts/ULTRON_2.md` while preserving root copies so both path references resolve.
3. **Turn 0 Deliverables Referenced as Pre-Existing:** Prompt line 3 requested reading `research/PLAN.md`, `research/DECISIONS.md`, `research/EXPERIMENT_LOG.md`, and `research/ULTRON_RESEARCH_LOG.md`. Because this was Turn 1, these files did not yet exist; ULTRON 1 initialized and populated them as primary deliverables.
4. **Untracked Bootstrap Files:** `AGENTS.md`, `COMMUNICATION.md`, `ULTRON_1.md`, and `ULTRON_2.md` were untracked in git at turn start. ULTRON 1 explicitly staged and committed them alongside the Turn 1 claim in commit `e7e4b48`.

---

### 3. Files Changed and Purpose
- `COMMUNICATION.md`: Claimed Turn 1 (`ULTRON_1_ACTIVE`), recorded timestamps, and transitioned to `READY_FOR_ULTRON_2` upon completion.
- `AGENTS.md`, `ULTRON_1.md`, `ULTRON_2.md`: Staged and tracked project rules and role definitions.
- `prompts/ULTRON_1.md`, `prompts/ULTRON_2.md`: Created prompt directory files for structural compliance with AGENTS.md.
- `RESEARCH_CHARTER.md`: Established hard constraints (2,048 MB RSS, 100 GB storage), reference baselines (Llama-3-70B, Qwen-2.5-72B), benchmark suites, and acceptance tiers.
- `research/PLAN.md`: Created 5-stage research roadmap, architectural candidate matrix, and go/no-go gates.
- `research/DECISIONS.md`: Documented ADR-001 (Quality Benchmarks), ADR-002 (Memory Accounting Model), and ADR-003 (Storage Budget Exploitation).
- `experiments/exp001_memory_streaming_benchmark.py`: Implementation of runtime RSS profiling, KV cache verification, and sequential layer streaming simulation.
- `experiments/exp001_results.json`: Raw empirical output from EXP-001.
- `tests/test_memory_accounting.py`: 4 unit tests verifying KV cache arithmetic and memory budget headroom (4/4 passed).
- `research/EXPERIMENT_LOG.md`: Detailed record of EXP-001 setup, metrics, and conclusions.

---

### 4. Research Sources and Claims Supported
1. **Alizadeh et al. (Apple, Dec 2023)** - "LLM in a Flash: Efficient Large Language Model Inference with Limited Memory" (arXiv:2312.11514).
   - *Supported Claim:* Flash memory can stream weights on demand, but naive streaming suffers bandwidth bottlenecks unless combined with activation sparsity or windowing.
2. **Eliseev & Mazur (2023)** - "Fast Inference of Mixture-of-Experts Language Models with Offloading" (arXiv:2312.17238).
   - *Supported Claim:* MoE architectures activate only a fraction of experts (e.g. top-2 of 8 or 64), dramatically reducing the bytes transferred per token from storage to memory.
3. **Borgeaud et al. (DeepMind, 2022)** - "Improving Language Models by Retrieving from Trillions of Tokens" (RETRO, ICML 2022, arXiv:2112.04426).
   - *Supported Claim:* A 7B model augmented with external on-disk retrieved tokens matches a 175B model, validating the hypothesis that 100B capability does not require 100B weights in RAM if world knowledge is stored on the 100 GB disk.
4. **Wang et al. (Microsoft Research, 2024)** - "The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits" (arXiv:2402.17764).
   - *Supported Claim:* Ternary quantization reduces weight footprint to 1.58 bits per parameter and replaces matrix multiplication with additions.
5. **Subramanya et al. (Microsoft, NeurIPS 2019)** - "DiskANN: Fast Accurate Billion-Point Nearest Neighbor Search on a Single Node".
   - *Supported Claim:* SSD-based vector search executes in single-digit milliseconds with <100 MB RAM footprint.

---

### 5. Hypotheses, Predictions, and Empirical Evidence
- **Hypothesis 1:** Transformer layer weights can be sequentially loaded from disk into a recycled buffer without process RSS growing proportionally to total model size.
  - *Status:* **MEASURED & CONFIRMED**.
  - *Measured Outcome:* A 2,048 MB model (32 layers x 64 MB) executed with peak RSS of 288.39 MB (14% of the 2 GB budget) and memory drift of only 0.01 MB over 32 iterations.
- **Hypothesis 2:** Dense layer streaming can achieve interactive token generation speeds (>= 2 tokens/s) on NVMe storage for 70B-100B models.
  - *Status:* **MEASURED & REFUTED**.
  - *Measured Outcome:* Streaming 2.0 GB took 1.627 seconds per token at 1,259.11 MB/s. For a 40 GB 70B-100B model, streaming would require ~32 seconds per token (0.03 tokens/s). Dense streaming is mathematically bandwidth-bound.

---

### 6. Quantitative Metrics Summary
- **Peak Process RAM:** 288.39 MB RSS (Ceiling: 2,048 MB; 86% headroom remaining).
- **Runtime Baseline Overhead:** 190.19 MB RSS (Python 3.11 + PyTorch CPU).
- **KV Cache Footprint (2,048 tokens, 32L, 8H, D128):** 256.0 MB (FP16) / 128.0 MB (INT8).
- **KV Cache Footprint (8,192 tokens, 32L, 8H, D128):** 1,024.0 MB (FP16) / 512.0 MB (INT8).
- **Storage Artifact Footprint:** Temporary 2.0 GB test tensors cleaned up; code/logs < 1 MB.
- **Disk Streaming Read Throughput:** 1,259.11 MB/s to 1,673.53 MB/s on local PCIe NVMe.
- **Single-Token Pass Latency (2 GB model):** 1.627 seconds.

---

### 7. Flaws Found, Unresolved Risks, and Failed Assumptions
1. **Dense Streaming Speed Flaw:** As calculated above, reading tens of gigabytes of weights per token over PCIe is 1-2 orders of magnitude too slow for interactive inference.
2. **KV Cache Explosion at Long Contexts:** At 8,192 tokens, unquantized FP16 KV cache consumes 1,024 MB (half the RAM budget). Any candidate architecture must incorporate KV cache quantization (INT8/INT4) or sliding-window attention.
3. **Execution Runtime Weight:** PyTorch CPU takes ~190 MB simply when loaded. If C++ runtimes (e.g. `llama.cpp` or custom ONNX C++ runtime) are used, baseline RSS can be reduced to <30 MB.

---

### 8. Claim Categorization
- **Measured:** Runtime RSS (190.19 MB), streaming peak RSS (288.39 MB), memory drift (0.01 MB), SSD read throughput (1,259 MB/s), full-pass latency (1.627 s), unit test execution (4/4 passed).
- **Derived / Analytical:** Mathematical KV cache sizing formulas, 70B/100B full dense model streaming latency projection (28-40 s/token).
- **Reported by Sources:** RETRO 7B equaling 175B on knowledge benchmarks; BitNet b1.58 ternary scaling; Apple LLM in a Flash sparsity speedup.
- **Hypothesis (Pending Future Turns):** Sparse MoE with top-2 expert prefetching or RAG with 1.5B core reasoner can meet the Tier 2 acceptance criteria under 2 GB RAM.

---

### 9. Handoff to ULTRON 2

- **Next Agent:** ULTRON 2 (Independent Reviewer & Red Team Auditor)
- **Concrete Task for ULTRON 2:**
  1. Audit the memory accounting calculations in `research/DECISIONS.md` (ADR-002) and `tests/test_memory_accounting.py`. Verify whether activation tensor peaks or memory fragmentation during GEMM operations could breach the 2,048 MB ceiling.
  2. Independently verify the streaming benchmark results in `experiments/exp001_results.json` and examine the I/O throughput numbers.
  3. Formally critique the two remaining candidate pathways in `research/PLAN.md` (Sparse MoE Expert Offloading vs Non-Parametric Memory Retrieval): Which one presents the lower risk of failing the 2 GB RAM / 100 GB storage / 100B quality charter, and what concrete counterexample or theoretical bound challenges them?
- **Acceptance Criteria for ULTRON 2:**
  - Independent recalculation or reproduction run of EXP-001.
  - Rigorous red-team critique identifying hidden memory costs, fragmentation risks, or latency bottlenecks.
  - Explicit recommendation with evidence for the Turn 3 experiment.
