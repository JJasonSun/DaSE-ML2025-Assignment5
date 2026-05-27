# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project Overview

LLM Needle-in-a-Haystack (NIAH) evaluation framework — an automated, pluggable platform for testing LLM long-context retrieval and complex reasoning. Built as an ECNU Machine Learning course assignment.

## Common Commands

```bash
# Install dependencies
uv pip install -r requirements.txt

# Run evaluation (default: 20 cases sampled from all_en, multi-needle, LLM judge)
uv run python run.py --agent agents.agent_plus:AdvancedRetrievalAgent

# Sample more cases
uv run python run.py --agent agents.agent_plus:AdvancedRetrievalAgent --num_samples 50

# Single-needle grid search (context_length x depth)
uv run python run.py --agent agents.agent_plus:AdvancedRetrievalAgent --test_mode single

# Enable extended thinking for better reasoning
uv run python run.py --agent agents.agent_plus:AdvancedRetrievalAgent --enable_thinking True

# Regenerate HTML report from latest local structured data
uv run python generate_report.py --input results/latest_evaluation_data.json --output results/evaluation_report.html

# Skip health check for faster iteration
uv run python run.py --agent agents.agent_plus:AdvancedRetrievalAgent --skip_model_test True

```

## Architecture

### Plugin-based Agent system
Agents are loaded dynamically via `core/agent_factory.py` using `importlib`. The `--agent` flag takes `module.path:ClassName` format (e.g., `agents.agent_plus:AdvancedRetrievalAgent`). All agents must extend `agents.base_agent.ModelProvider`. The base class provides default implementations for `generate_prompt`, `encode_text_to_tokens`, `decode_tokens`, `_create_chat_completion`, and answer post-processing. Subclasses only need to implement:
- `async evaluate_model(prompt: Dict) -> str` — main entry point

### Test modes
- **multi**: Multiple needles inserted into different files at random depths, repeated `num_tests` times per test case
- **single**: Grid search over context_length (1K–100K) and depth_percent (0–100%) with one needle

### Agent hierarchy
```
ModelProvider (base_agent.py)         — abstract base + answer post-processing
├── ExampleAgent (agent_template.py)  — random baseline
├── SyncRetrievalAgent (sync_agent.py)— keyword retrieval + sentence extraction
└── AdvancedRetrievalAgent (agent_plus.py) — BM25 + dense embedding + rerank hybrid
    └── ScenarioAwareAgent (scenario_agent.py) — intent classification + dynamic prompts
```

### Retrieval pipeline (AdvancedRetrievalAgent)
Token-level chunking (500 tokens, 100 overlap) → BM25 keyword retrieval (top 30) → Dense vector retrieval via `ecnu-embedding-small` (top 20) → Merge & deduplicate → Rerank via `ecnu-rerank` (top 8) → Dynamic assembly with score threshold + neighbor enrichment. Falls back to full-context mode when total tokens < 64K.

### Evaluators
- `LLMEvaluator`: Uses `ecnu-plus` as judge, scores 0–10 (semantic accuracy)
- `StringMatchEvaluator`: Binary 0/1 exact match

### Reporting
Each evaluation writes one local structured data snapshot to `results/latest_evaluation_data.json`, overwriting the previous snapshot. Report generation is plugin-based: `core/reporter_factory.py` loads `reporters.deepseek_html_reporter:DeepSeekHtmlReporter` by default. The plugin renders `results/evaluation_report.html` from the structured data and uses `deepseek-v4-pro` with thinking enabled only for the product-analysis text. `generate_report.py` can regenerate the HTML report from local data without rerunning evaluation.

### Model configuration
All ECNU model names are centralized in `core/ecnu_constants.py`. Evaluation environment variables: `ECNU_API_KEY`, `ECNU_BASE_URL`, `MODEL_NAME`. Report analysis environment variables: `DS_API_KEY`, `DS_BASE_URL`, `DS_MODEL_NAME`.

## Adding a New Agent

1. Create `agents/your_agent.py`
2. Subclass `ModelProvider` (or `AdvancedRetrievalAgent` for hybrid retrieval)
3. Implement `async evaluate_model(prompt: Dict) -> str`
4. Run with `--agent agents.your_agent:YourAgentClass`
5. See `agents/agent_template.py` for a minimal reference implementation

## Integrity Constraints (Assignment Rules)
- Must NOT read original `PaulGrahamEssays/*.txt` files directly — context is provided at runtime
- Must NOT read test case JSON files — questions are passed via the prompt dict
- Must NOT hardcode answers
