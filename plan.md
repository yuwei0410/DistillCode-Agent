# DistillCode-Agent: Project Roadmap & Execution Plan

## 1. Project Overview & Objectives
DistillCode-Agent is an end-to-end autonomous code-repair pipeline built to address context bloat, small-model function-calling degradation, and high inference costs. 

The system couples a deterministic, stateful repair agent with an automated trajectory-harvesting flywheel that distills multi-turn debugging behavior from a frontier teacher model into an open-source 7B model (Qwen 2.5 Coder 7B) using 4-bit QLoRA.

### Key Milestones
1. Build an isolated Docker execution sandbox with deterministic AST traceback compaction.
2. Construct a stateful self-correction agent using LangGraph.
3. Harvest and prune 25-30 successful multi-turn trajectories across real Python issue instances.
4. Fine-tune Qwen 2.5 Coder 7B on free cloud GPU compute (Kaggle/Colab T4 via Unsloth).
5. Benchmark Baseline 7B vs. Distilled 7B vs. Teacher across Pass@1, schema syntax accuracy, and token efficiency.

---

## 2. 3-Week Timeline & Phase Breakdown

### Phase 1: Environment Sandboxing & Core Agent Architecture (Days 1–7)
- Day 1-2: Environment Setup & Docker Sandboxing
  * Configure WSL2 (Ubuntu) with Docker Desktop integration.
  * Create Dockerfile.sandbox based on python:3.11-slim hosting a target repository (e.g., psf/requests or pallets/flask).
  * Build docker_runner.py using the Docker Python SDK with strict command execution timeouts (e.g., 30s max per run).
- Day 3-4: Deterministic Tool Schemas
  * Implement read_file_bounded: Reads lines within a specified start/end range to prevent full-file context dumps.
  * Implement find_symbol_ast: Locates function/class definitions across repository files without loading entire files into memory.
  * Implement apply_unified_patch: Validates and applies git diffs or direct replacements.
  * Implement run_pytest_isolated: Executes targeted tests inside the container and returns exit code, stdout, and stderr.
- Day 5-7: LangGraph State Machine & Traceback Compactor
  * Implement traceback_cleaner.py to extract only failing files, assertion errors, and local frame lines from raw pytest outputs.
  * Assemble LangGraph graph: Agent Node -> Docker Executor Node -> Test Verifier Node -> Reflection Loop (Max 4 iterations).
  * Wire in teacher LLM (Google AI Studio Gemini 2.5 Flash free endpoint via OpenAI-compatible SDK).
  * Validation Checkpoint: Manually inject 2 bug scenarios and confirm the loop self-corrects without context window bloat.

### Phase 2: Trajectory Collection & Data Curation Flywheel (Days 8–12)
- Day 8-9: Trajectory Harvesting Runs
  * Curate 30 seed tasks from closed issues/PRs of the target repository.
  * Execute the Phase 1 agent across all 30 tasks, logging full state history (Thought, Tool Call, Observation, Reflection).
  * Filter runs: Retain only trajectories that terminated with a passing test suite (exit_code == 0).
- Day 10-12: Pruning, Compaction & SFT Formatting
  * Build prune_trajectories.py: Strip dead-end exploratory loops and duplicate tool invocations to preserve high training signal.
  * Format pruned trajectories into standard OpenAI/ChatML multi-turn tool-calling JSONL schema.
  * Create data splits: Train set (20-25 trajectories) and Held-out Evaluation set (5-10 tasks).
  * Validation Checkpoint: Verify JSONL schema validity against Hugging Face / Unsloth dataset loaders.

### Phase 3: Zero-Cost Fine-Tuning & Quantization (Days 13–16)
- Day 13-14: QLoRA Fine-Tuning on Free GPU Compute
  * Launch a free Kaggle Notebook or Google Colab session with an NVIDIA T4 GPU (16 GB VRAM).
  * Load Qwen/Qwen2.5-Coder-7B-Instruct with 4-bit quantization using Unsloth.
  * Configure LoRA hyperparameters: rank = 16, alpha = 16, target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"].
  * Train for 2-3 epochs, monitoring validation loss for convergence without catastrophic forgetting.
- Day 15-16: GGUF Export & Local Serving
  * Export merged adapter weights to GGUF format (Q4_K_M quantization, ~4.5 GB).
  * Download the GGUF file to the local Windows/WSL2 environment.
  * Create a Modelfile and load the model into local Ollama or llama.cpp serving an OpenAI-compatible endpoint at localhost:11434.
  * Validation Checkpoint: Run a test prompt against local Ollama to verify native tool call JSON formatting.

### Phase 4: Benchmarking, Tracing & Artifact Finalization (Days 17–21)
- Day 17-18: Comparative Evaluation Harness
  * Execute held-out evaluation tasks across three setups:
    1. Base Qwen 2.5 Coder 7B
    2. Distilled Qwen 2.5 Coder 7B (Our Model)
    3. Teacher Model (Gemini 2.5 Flash)
  * Record hard comparative metrics: Pass@1 resolution rate, Tool syntax/argument parse error rate, and average token consumption per task.
- Day 19-20: Observability & Logging
  * Integrate Langfuse (free cloud tier) or self-hosted Arize Phoenix to capture visual execution traces.
  * Record side-by-side run logs demonstrating failure recovery vs. uncompacted loops.
- Day 21: Documentation & Portfolio Release
  * Finalize GitHub repository structure, setup instructions, architecture diagrams, and benchmark evaluation tables.

---

## 3. Modular Decoupling & Architecture Fallbacks

The pipeline enforces strict separation between orchestration, tool execution, and the underlying inference models using the OpenAI-compatible API standard.

- Swapping the Teacher Model:
  If the free-tier teacher underperforms on complex tasks, the client in src/agent/graph.py can be swapped to Anthropic Claude 3.5/3.7 Sonnet, OpenAI GPT-4o, or DeepSeek-V3 via OpenRouter by updating the client initialization. All graph logic, state variables, and tool schemas remain unchanged.
- Swapping the Student Model:
  The trajectory dataset uses standard role-based schemas (system, user, assistant, tool). The base model in the Unsloth training script can be switched from Qwen 2.5 Coder 7B to Llama 3.1 8B or Mistral 7B with zero data conversion.
- Swapping the Target Repository:
  To change the evaluation target from requests to another library (e.g., flask, click), only the Dockerfile and task seed issue list need updating. LangGraph nodes and execution bridges require no modifications.

---

## 4. Benchmark Tracking Table Template

| Model / Configuration | Pass@1 Resolution (%) | Tool Syntax Error Rate (%) | Avg Tokens / Task | Cost / 100 Tasks ($) |
|:---|:---:|:---:|:---:|:---:|
| Teacher Model (Gemini 2.5 Flash) | -- | -- | -- | $0.00 (Free Tier) |
| Base Student (Qwen 2.5 Coder 7B) | -- | -- | -- | $0.00 (Local GGUF) |
| Distilled Student (DistillCode 7B) | -- | -- | -- | $0.00 (Local GGUF) |