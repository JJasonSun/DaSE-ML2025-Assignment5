# AGENTS.md

This file provides guidance to Codex when working with code in this repository.

## Constraints & Tone

1. 语言：始终使用中文。
2. 态度：保持客观与真实。如果用户前提有误，直接指出。
3. 工具：遇到不懂的概念或时效性信息，必须使用联网搜索。

## Project Overview

LLM Needle-in-a-Haystack evaluation framework for long-context retrieval, multi-document evidence aggregation, exact reasoning, automated scoring, structured result capture, and HTML report generation.

## Common Commands

```bash
# Install dependencies
uv pip install -r requirements.txt

# Recommended evaluation: tool-augmented agent, 20 balanced samples, multi mode, LLM judge
uv run python run.py --agent agents.tool_augmented_agent:ToolAugmentedAgent

# Smoke test: same multi-mode flow, 5 cases, 1 run each
uv run python run.py --agent agents.tool_augmented_agent:ToolAugmentedAgent --num_samples 5 --num_tests 1

# Larger sample
uv run python run.py --agent agents.tool_augmented_agent:ToolAugmentedAgent --num_samples 50

# Hybrid retrieval baseline for ablation
uv run python run.py --agent agents.hybrid_retrieval_agent:HybridRetrievalAgent

# Single-needle grid search
uv run python run.py --agent agents.tool_augmented_agent:ToolAugmentedAgent --test_mode single

# Enable model thinking
uv run python run.py --agent agents.tool_augmented_agent:ToolAugmentedAgent --enable_thinking True

# Regenerate HTML report from latest structured data
uv run python generate_report.py --input results/latest_ToolAugmentedAgent_evaluation_data.json
```

## Architecture

Agents are dynamically loaded by `core/agent_factory.py` using the `module.path:ClassName` format. Public agents are intentionally limited:

```text
ModelProvider
├── BaselineAgent
└── HybridRetrievalAgent
    └── ToolAugmentedAgent
```

- `BaselineAgent`: minimal control group.
- `HybridRetrievalAgent`: BM25 + dense embedding + rerank + neighbor chunk retrieval.
- `ToolAugmentedAgent`: recommended path; adds scenario routing, structured extraction, deterministic Python tools, answer formatting, and tool-chain trace.

## Test Modes

- `multi`: multiple needles inserted into different files at random depths, repeated `num_tests` times per test case.
- `single`: grid search over `context_length x depth_percent` with one needle.

## Reporting

Each evaluation writes the latest structured data and HTML report per agent, for example `results/latest_ToolAugmentedAgent_evaluation_data.json` and `results/ToolAugmentedAgent_evaluation_report.html`. The default reporter uses `deepseek-v4-pro` only for the product-analysis text, and its analysis prompt includes the tested agent's profile so attribution and suggestions match that agent's design.

## Model Configuration

ECNU model names are centralized in `core/ecnu_constants.py`. Evaluation uses `ECNU_API_KEY`, `ECNU_BASE_URL`, and `MODEL_NAME`. Report analysis uses `DS_API_KEY`, `DS_BASE_URL`, and `DS_MODEL_NAME`.

## Adding a New Agent

1. Create `agents/your_agent.py`.
2. Subclass `ModelProvider` or `HybridRetrievalAgent`.
3. Implement `async evaluate_model(prompt: Dict) -> str`.
4. Run with `--agent agents.your_agent:YourAgentClass`.

## Integrity Constraints

- Must not read original `PaulGrahamEssays/*.txt` files directly.
- Must not read test case JSON files.
- Must not hardcode answers.
