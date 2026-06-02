# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

LLM Needle-in-a-Haystack evaluation framework for long-context retrieval, multi-document evidence aggregation, exact reasoning, automated scoring, structured result capture, and HTML report generation.

## Common Commands

```bash
uv pip install -r requirements.txt
uv run python run.py --agent agents.tool_augmented_agent:ToolAugmentedAgent
uv run python run.py --agent agents.tool_augmented_agent:ToolAugmentedAgent --num_samples 5 --num_tests 1
uv run python run.py --agent agents.hybrid_retrieval_agent:HybridRetrievalAgent
uv run python run.py --agent agents.tool_augmented_agent:ToolAugmentedAgent --test_mode single
uv run python generate_report.py --input results/latest_ToolAugmentedAgent_evaluation_data.json
```

## Agent Architecture

Agents are dynamically loaded via `core/agent_factory.py` with `module.path:ClassName`.

```text
ModelProvider
├── BaselineAgent
└── HybridRetrievalAgent
    └── ToolAugmentedAgent
```

- `BaselineAgent`: minimal control group and development reference.
- `HybridRetrievalAgent`: BM25 + dense embedding + rerank retrieval baseline.
- `ToolAugmentedAgent`: recommended agent, combining retrieval, structured extraction, deterministic Python tools, fallback handling, and trace metadata.

## Reporting

Evaluations write the latest structured data and HTML report per agent, for example `results/latest_ToolAugmentedAgent_evaluation_data.json` and `results/ToolAugmentedAgent_evaluation_report.html`. Report text analysis uses DeepSeek with the tested agent's profile in the analysis prompt, while evaluation and retrieval use ECNU APIs.

## Integrity Constraints

- Do not let agents read `PaulGrahamEssays/*.txt` directly.
- Do not let agents read test case JSON files.
- Do not hardcode answers.
