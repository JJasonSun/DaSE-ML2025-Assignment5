# Project Handoff Summary

生成日期：2026-06-06  
原始项目路径：`D:\Workspace\Study\ML-Assignment-5`  
交接目的：迁移项目到另一台电脑后，保留本轮协作的需求背景、设计决策、代码状态、评测结果、运行方式和后续优化方向。

## 1. 项目当前定位

本项目是一个面向 LLM / RAG / Agent 的长上下文评测与诊断系统。它基于 Needle-in-a-Haystack 范式，评估模型在长上下文、多文档、多 needle、复杂推理场景下的表现，并沉淀结构化评测数据、执行 trace、Bad Case 归因和 HTML 报告。

围绕 AI 产品经理求职叙事，项目定位已经从“算法/工程实现”调整为：

- AI Agent 能力评估体系。
- RAG 与工具增强 Agent 的架构对照实验。
- Bad Case 归因与产品迭代闭环。
- 模型能力边界、Prompt 优化、检索策略调整、工具能力建设和模型选型的数据依据。

它和另外两个简历项目的分工如下：

| 项目 | 简历定位 | 主要展示能力 |
| :--- | :--- | :--- |
| DeepCast 深度调研 AI 播客生成 Agent | AI 内容产品 / 多 Agent 生成链路产品化 | 端到端 Agent Demo、任务拆解、SSE 进度体验、内容生成稳定性 |
| 多模态 RAG 图文问答系统 | 企业知识库 / 多模态文档问答落地 | PDF 解析、向量检索、模型问答、批量评测 |
| 本项目 | Agent Benchmark / AI Agent 评测与诊断 | 评测体系、能力边界、架构对照、Bad Case 归因、数据驱动迭代 |

如果投 AI 产品经理、AI Agent、AI Coding、模型评测、开发者工具、MaaS 或企业 AI 产品岗位，本项目可以作为主项目或靠前项目。

## 2. 本轮对话与决策时间线

### 2.1 初始目标：修复 ToolAugmentedAgent 工具执行回归

用户最初要求基于 `results/` 文件夹下的分析报告和未提交改动，修复 `ToolAugmentedAgent` 的工具执行回归，重点让它至少能正确执行：

- `abs(hash.count(...))`
- 字符串反转切片，例如 `confirmation_code[::-1]`
- Caesar / Julius shift 解码

当时的目标是降低 `tool_execution_failed`，并跑最小复现或冒烟验证，报告分数和失败数下降情况。

随后进一步明确：

- 编码类任务应优先走确定性工具。
- 工具执行失败不应无条件 fallback 到 Hybrid 猜答案。
- trace 中要记录明确失败原因。
- 支持 Base64、URL-safe Base64、Hex / Base16、Caesar、reverse、count 等通用 operation。

### 2.2 报告生成逻辑：每个 Agent 保留自己的最新数据和报告

用户指出报告不应只存一份，而是每个 Agent 都要保留最新结构化数据和 HTML 报告，并且文件名带 Agent 名称。

当前实现：

- `results/latest_<AgentName>_evaluation_data.json`
- `results/<AgentName>_evaluation_report.html`

例如：

- `results/latest_ToolAugmentedAgent_evaluation_data.json`
- `results/ToolAugmentedAgent_evaluation_report.html`
- `results/latest_HybridRetrievalAgent_evaluation_data.json`
- `results/HybridRetrievalAgent_evaluation_report.html`

路径逻辑由 `core/report_paths.py` 管理，运行评测和单独生成报告都会使用这套命名规则。

### 2.3 报告分析 prompt：应了解被测 Agent 的情况

用户希望不同 Agent 写评测报告时，提示词里要告知该 Agent 的相关情况，报告建议才能有针对性。

后来进一步收敛为：每个 Agent 类在代码里定义自己的 `AGENT_PROFILE`，报告分析时只引用各自 profile。

当前实现：

- `agents/baseline_agent.py` 有 `BaselineAgent.AGENT_PROFILE`。
- `agents/hybrid_retrieval_agent.py` 有 `HybridRetrievalAgent.AGENT_PROFILE`。
- `agents/tool_augmented_agent.py` 有 `ToolAugmentedAgent.AGENT_PROFILE`。
- `core/agent_profiles.py` 通过 `module.path:ClassName` 动态 import Agent 类，读取其 `AGENT_PROFILE`。
- `reporters/deepseek_html_reporter.py` 在 AI 分析 prompt 中注入：

```text
Agent context:
{agent_profile}
```

如果 Agent 没有 profile 或 import 失败，会使用保守的默认 profile，避免报告虚构实现细节。

### 2.4 是否继续优化 HybridRetrievalAgent

用户多次询问：`ToolAugmentedAgent` 和 `HybridRetrievalAgent` 是否都需要优化，还是只优化 Tool。

核心判断：

- `HybridRetrievalAgent` 本质是检索增强基线，不应改成半个 ToolAugmentedAgent。
- 它可以小幅增强检索质量和 trace，但不要加计算器、解码器、日期工具。
- `ToolAugmentedAgent` 才是主力优化对象，因为它展示的是“LLM 规划 + 工具执行 + 诊断闭环”的 Agent 产品形态。

后来用户要求按照“小幅增强 Hybrid”的建议实现，并明确调用 ECNU 模型能力。

当前 Hybrid 已做的小幅增强：

- 通用 query expansion：
  - quoted strings
  - 方括号 ID
  - 全大写下划线 ID
  - 含数字/短横线 identifier
  - capitalized noun phrases
  - 操作相关 query，如 MD5、decode、count 等关键词
- 动态证据窗口：
  - 多证据任务扩大 `target_evidence_blocks`
  - neighbor radius 可从 1 扩到 2
- retrieval trace：
  - `selected_queries`
  - `evidence_block_count`
  - `retrieved_files`
  - `rerank_scores`
  - `context_chars`
  - `neighbor_radius`
  - `target_evidence_blocks`
  - `embedding_model`
  - `rerank_model`
- prompt 强化：
  - 要求保留精确数字、大小写和字符。
  - 证据不足时返回 `Unknown`。

Hybrid 仍保持“无工具检索增强基线”的角色。

### 2.5 ToolAugmentedAgent 去硬编码化

用户在更大样本评测后指出：有些 case 的判断可以充分利用 LLM 能力，不一定都在代码里硬编码，否则代码会过于冗长。

最终策略调整为：

- 不继续为单个 case 堆正则、标签别名或项目名 alias。
- 采用三段式：
  - LLM Planner：理解 question + retrieved context，输出结构化 operation plan。
  - Evidence Validator：检查 plan 中字段是否齐全、是否可执行。
  - Deterministic Executor：只执行通用确定性工具。
- 工具失败时进行一次 LLM 二次规划，不直接让 Hybrid 猜答案。
- 二次规划仍失败则返回 `Unknown`，并保留失败 trace。

当前 `ToolAugmentedAgent` 已经是 planner-first 架构。

关键 trace 字段：

- `planner_attempts`
- `operation_plan`
- `validation_result`
- `tool_failure_reason`
- `execution_mode`
- `path`
- `task_type`
- `tool_answer`
- `context_chars`

支持的失败原因包括：

- `missing_required_evidence`
- `unsupported_operation`
- `invalid_operation_parse`
- `invalid_decoded_output`

支持的执行模式包括：

- `planned_tool`
- `repaired_planned_tool`
- `failed`
- `fallback_to_hybrid`，目前主路径不鼓励无条件 fallback

### 2.6 使用 ECNU API

用户明确要求调用 `ecnu-api` skill，充分利用 ECNU 的模型。

已读取并遵循 ECNU OpenAI-compatible API 约定：

- Base URL：`https://chat.ecnu.edu.cn/open/api/v1`
- 主模型：`ecnu-max`
- 辅助模型：`ecnu-plus`
- embedding：`ecnu-embedding-small`
- rerank：`ecnu-rerank`

项目中模型名集中在 `core/ecnu_constants.py`。

环境变量：

```env
ECNU_API_KEY=sk-xxxx
ECNU_BASE_URL=https://chat.ecnu.edu.cn/open/api/v1
MODEL_NAME=ecnu-max
```

DeepSeek 只用于报告分析文本：

```env
DS_API_KEY=sk-xxxx
DS_BASE_URL=https://api.deepseek.com
DS_MODEL_NAME=deepseek-v4-pro
```

### 2.7 最新报告与是否适合放简历

用户多次要求查看最新 `ToolAugmentedAgent` 和 `HybridRetrievalAgent` 报告，判断项目是否足够放简历上。

结论：

- 可以放。
- 更适合写成 AI Agent 评测与诊断项目，而不是单纯 RAG 或算法工程项目。
- 重点强调：评测体系、能力边界、Bad Case 归因、Agent 架构对照、数据驱动迭代。
- 不要过度宣称生产可用，也不要把小样本 Tool 回归结果和大样本 Hybrid 结果写成严格同条件 A/B。

推荐项目名：

```text
长上下文 Agent 评测与诊断系统｜AI Agent 能力评估与产品迭代实践
```

最终简历表述建议：

```text
面向长上下文检索、多文档证据聚合与精确推理场景，搭建 Needle-in-a-Haystack 自动化评测体系，构建 1600 条测试集，覆盖计算、日期、编码、字符串分析等典型 Agent 失败场景，用于评估 LLM / RAG / Agent 在复杂任务中的能力边界。

设计 Baseline / Hybrid Retrieval / Tool-Augmented 三层 Agent 对照实验，区分“无检索”“检索增强”“工具增强”不同方案的效果差异，帮助判断哪些问题适合靠 Prompt / 检索优化解决，哪些必须引入规划与工具执行。

建立结构化评测报告与 Bad Case 归因机制，沉淀得分分布、任务类型表现、执行路径、失败原因、检索 trace、工具 trace 等指标，将模型问题归因为证据缺失、检索后推理失败、操作解析错误、输出格式偏差等可迭代方向。

设计面向产品迭代的双模式评测：用 multi 模式验证多文档 RAG 场景下的证据召回与聚合能力，用 single 模式扫描上下文长度和信息位置对回答稳定性的影响，辅助判断 Prompt、检索策略、工具增强和模型选型的优化优先级。

基于评测结果推动 Agent 方案迭代，将 ToolAugmentedAgent 优化为“LLM 规划 + 确定性工具执行”架构；最新回归 mean score 达 8.00，验证工具增强与诊断闭环对精确任务稳定性的价值。
```

其中最后一句曾进一步压缩为：

```text
基于评测结果推动 Agent 方案迭代，将 ToolAugmentedAgent 优化为“LLM 规划 + 确定性工具执行”架构；最新回归 mean score 达 8.00，验证工具增强与诊断闭环对精确任务稳定性的价值。
```

### 2.8 Prompt 语言与编码策略

早期曾讨论过 prompt 语言选择。用户一开始希望理解并优化项目 prompt，后来一度考虑把 prompt 改成中文。但项目在 Windows / PowerShell 环境下多次出现中文源码 mojibake，例如 `浣犳槸`、`锛`、`鈥` 等乱码，影响模型理解和代码可维护性。

最终决策：

- 发给模型的核心 prompt 尽量使用英文。
- DeepSeek 报告分析 prompt 使用英文指令，但要求输出简体中文分析正文。
- Python 源码中尽量避免直接写大段中文 prompt。
- 文档和简历表述可以使用中文，但需要用 UTF-8 保存和查看。
- 如果 PowerShell `Get-Content` 显示中文乱码，先用 UTF-8 方式读取确认，不要直接认定文件内容损坏。

### 2.9 CLI、测试集和默认运行方式收敛

用户希望项目更适合展示和复现，不要每次都手动指定大量参数。当前 CLI 已收敛为：

- 默认测试集：`test_cases/test_cases_all_en.json`
- 默认 `--test_mode multi`
- 默认 `--num_samples 20`
- 默认 `--num_tests 3`
- 默认 `--evaluator_type llm`
- 默认 `--generate_report True`
- 默认 `--enable_thinking False`

因此推荐主命令保持简洁：

```bash
uv run python run.py --agent agents.tool_augmented_agent:ToolAugmentedAgent
```

快速冒烟：

```bash
uv run python run.py --agent agents.tool_augmented_agent:ToolAugmentedAgent --num_samples 5 --num_tests 1
```

扩大样本：

```bash
uv run python run.py --agent agents.tool_augmented_agent:ToolAugmentedAgent --num_samples 50
```

### 2.10 健康检查强制化

用户曾希望删除健康检查开关，避免在模型、embedding 或 rerank 服务不可用时继续跑出误导性结果。

当前决策：

- 不保留 `--skip_model_test` 这类跳过健康检查的入口。
- 评测前必须执行健康检查。
- 健康检查覆盖主测模型、LLM judge、embedding 和 rerank。
- 健康检查失败应先排查 `.env`、ECNU API、模型名和上游服务状态。

这也是迁移到新电脑后优先跑冒烟测试的原因之一：先确认 API 和依赖都可用，再解释评测结果。

### 2.11 报告形态：HTML 看板替代图片可视化

项目早期有过 PNG 可视化 / `visualizations/` 之类的思路，用户后来明确希望：

- 不生成图片。
- 不引用图片。
- 不让模型读图。
- 评测结果直接生成 HTML 数据看板。
- DeepSeek 只写分析文字，不负责生成 HTML 页面结构。

当前实现边界：

- `core/evaluation_data.py` 负责结构化数据和 metrics。
- `reporters/deepseek_html_reporter.py` 负责 HTML 模板渲染和调用 DeepSeek 生成分析文字。
- `generate_report.py` 支持基于本地 JSON 单独重生成 HTML。
- DeepSeek 输出被要求为 Markdown 文本，报告插件会把它安全渲染进页面。

这个设计是为了让报告稳定、可复现，也避免 LLM 直接生成 HTML 时格式漂移。

### 2.12 Agent 收敛与旧入口清理

用户认为原来 `agents/` 里 Agent 太多，不利于 AI 产品经理简历叙事。最终收敛为三层公开 Agent：

- `BaselineAgent`
- `HybridRetrievalAgent`
- `ToolAugmentedAgent`

旧 Agent / 旧入口不再作为公开推荐路径：

- `ExampleAgent`
- `SyncRetrievalAgent`
- `AdvancedRetrievalAgent`
- `ScenarioAwareAgent`
- `agent_plus.py`
- `agent_template.py`
- `sync_agent.py`
- `scenario_agent.py`

当时明确不保留旧导入路径兼容 shim，因为项目当前目标是求职展示和架构清晰，而不是对外发布稳定 SDK。

### 2.13 JD 对齐过程

用户提供了 `docs/JD.md` 作为 AI 产品经理岗位参考。PowerShell 读取时显示过 mojibake，但能提取出的 JD 关键词包括：

- AI 产品经理。
- Vibe Coding。
- Agent 场景评测与梳理。
- 模型能力评估。
- 数据驱动迭代。
- Bad Case 分析。
- AI Coding / 开发者工具。
- MaaS / 商业化。
- PRD、产品方案、用户研究、跨团队协作。

因此本项目简历叙事从“我实现了一个评测框架”改为“我搭建了 Agent 能力评估和产品迭代闭环”。这是一次重要定位调整：降低算法工程味，增强 AI 产品经理岗位相关性。

## 3. 当前代码架构

### 3.1 公开 Agent 收敛

当前公开 Agent 保持三层结构：

```text
ModelProvider
├── BaselineAgent
└── HybridRetrievalAgent
    └── ToolAugmentedAgent
```

| Agent | 文件 | 定位 |
| :--- | :--- | :--- |
| `BaselineAgent` | `agents/baseline_agent.py` | 最小对照组 |
| `HybridRetrievalAgent` | `agents/hybrid_retrieval_agent.py` | 检索增强基线 |
| `ToolAugmentedAgent` | `agents/tool_augmented_agent.py` | 主力工具增强方案 |

不建议恢复旧 Agent 路径，也不建议继续扩很多 Agent。项目叙事上，三层对照已经足够清晰。

### 3.2 ToolAugmentedAgent 当前能力

`ToolAugmentedAgent` 已重构为 planner-first：

1. 选择上下文：
   - 小上下文直接用 full context。
   - 大上下文复用 Hybrid 检索。
2. LLM planner 生成结构化 plan。
3. Python validator 检查 plan。
4. deterministic executor 执行通用工具。
5. 工具失败时进行一次 replanning。
6. 仍失败则返回 `Unknown` 并记录 trace。

支持任务类型：

- `computation`
- `date_time`
- `string_analysis`
- `encoding`

支持工具能力：

- 大整数四则运算。
- 整数除法、差值、绝对值。
- 日期差、星期计算。
- 字符串 count。
- `abs(hash.count("F") - hash.count("b"))`。
- 字符串反转切片，例如 `confirmation_code[::-1]`。
- MD5 前缀。
- Base64。
- URL-safe Base64。
- Base32。
- Hex / Base16 / ASCII hex。
- Caesar / Julius shift / rotate / alphabet rotation。
- reverse / mirror / read backwards。

### 3.3 HybridRetrievalAgent 当前能力

`HybridRetrievalAgent` 是检索增强基线：

- BM25。
- ECNU dense embedding。
- ECNU rerank。
- neighbor chunk expansion。
- 通用 query expansion。
- full-context shortcut。
- retrieval trace。

它故意不加工具：

- 不加 calculator。
- 不加 decoder。
- 不加 date tool。
- 不把 Hybrid 改成半个 ToolAugmentedAgent。

报告分析时应把 Hybrid 的失败区分为：

- 检索没召回证据。
- 已召回证据但 LLM 后检索推理失败。
- 答案格式不稳定。

### 3.4 Agent Profile 机制

每个 Agent 类都应该定义 `AGENT_PROFILE`。报告分析只引用被测 Agent 的 profile，不再在报告 prompt 里手写固定 Agent 背景。

当前 profile 字段：

- `positioning`
- `expected_strengths`
- `expected_limits`
- `analysis_focus`

入口：

- `core/agent_profiles.py`

报告调用：

- `reporters/deepseek_html_reporter.py`

测试：

- `tests/test_agent_profiles.py`

## 4. 评测与报告体系

### 4.1 测试集

默认测试集：

```text
test_cases/test_cases_all_en.json
```

包含 1600 条英文测试用例，覆盖：

- `computation`
- `date_time`
- `encoding`
- `string_analysis`

注意诚信约束：

- Agent 不得直接读取 `PaulGrahamEssays/*.txt` 原始文件。
- Agent 不得读取测试 case JSON。
- Agent 不得硬编码答案。

### 4.2 multi 模式

默认推荐模式。

特点：

- 多文档。
- 多 needle。
- needle 随机插入不同文件、不同深度。
- 每个 case 可重复运行 `num_tests` 次。
- 更接近真实 RAG 场景下的信息召回、聚合和推理。

适合评估：

- 多文档检索。
- 多证据聚合。
- 跨片段信息组合。
- 精确任务稳定性。

### 4.3 single 模式

长上下文扫描模式。

特点：

- 单 needle。
- 扫描 `context_length x depth_percent` 网格。
- 适合看上下文长度和插入位置对模型稳定性的影响。

适合评估：

- 长上下文退化。
- needle 深度敏感性。
- Prompt / 模型选型在不同上下文长度下的表现。

### 4.4 评分体系

当前支持：

- LLM-as-a-Judge：`evaluators/llm_evaluator.py`
- String Match：`evaluators/string_match_evaluator.py`

默认：

```text
--evaluator_type llm
```

LLM judge 使用 ECNU 模型，报告分析使用 DeepSeek。

### 4.5 报告输出

每次评测输出：

- 结构化 JSON。
- HTML 报告。

报告包含：

- 概览指标。
- 分数分布。
- 分类型表现。
- multi 模式统计。
- single 模式热力图。
- Bad Case 表格。
- Tool Diagnostics。
- AI Product Analysis。

AI 分析要求：

- 简体中文。
- Markdown 文本。
- 不输出 HTML。
- 不虚构未提供的数据。
- 必须结合 Agent profile、metrics、bad cases 和 tool diagnostics。

## 5. 最新评测结果

以下数据来自 `results/` 下当前 latest 文件。

### 5.1 ToolAugmentedAgent

文件：

```text
results/latest_ToolAugmentedAgent_evaluation_data.json
results/ToolAugmentedAgent_evaluation_report.html
```

生成时间：

```text
2026-06-02 22:26:07
```

配置：

- Agent：`agents.tool_augmented_agent:ToolAugmentedAgent`
- test mode：`multi`
- evaluator：`llm`
- samples：20
- tests per case：3
- total runs：60

指标：

- mean score：8.00
- max score：10.00
- min score：0.00
- Good：48
- Partial：0
- Fail：12

分类型表现：

| 类型 | runs | mean score | bad cases | good cases |
| :--- | ---: | ---: | ---: | ---: |
| computation | 15 | 6.67 | 5 | 10 |
| date_time | 15 | 8.67 | 2 | 13 |
| encoding | 15 | 9.33 | 1 | 14 |
| string_analysis | 15 | 7.33 | 4 | 11 |

Tool diagnostics：

- path counts：
  - `tool_augmented`: 52
  - `tool_augmented_failed`: 8
- fallback/failure counts：
  - `invalid_operation_parse`: 2
  - `missing_required_evidence`: 6
- task counts：
  - `computation`: 29
  - `date_time`: 12
  - `encoding`: 12
  - `string_analysis`: 7

解读：

- Tool 当前已经适合放进简历作为“工具增强 Agent + 诊断闭环”的证据。
- encoding、date_time 表现较好。
- computation 和 string_analysis 仍有继续优化空间。
- 这次是 20 cases x 3 runs 的小样本回归，不要和 100 cases x 3 runs 的 Hybrid 严格等价比较。

### 5.2 HybridRetrievalAgent

文件：

```text
results/latest_HybridRetrievalAgent_evaluation_data.json
results/HybridRetrievalAgent_evaluation_report.html
```

生成时间：

```text
2026-06-03 14:25:13
```

配置：

- Agent：`agents.hybrid_retrieval_agent:HybridRetrievalAgent`
- test mode：`multi`
- evaluator：`llm`
- samples：100
- tests per case：3
- total runs：300

指标：

- mean score：2.06
- max score：10.00
- min score：0.00
- Good：49
- Partial：14
- Fail：237

分类型表现：

| 类型 | runs | mean score | bad cases | good cases |
| :--- | ---: | ---: | ---: | ---: |
| computation | 75 | 0.60 | 72 | 3 |
| date_time | 75 | 4.17 | 44 | 31 |
| encoding | 75 | 3.15 | 48 | 13 |
| string_analysis | 75 | 0.31 | 73 | 2 |

Tool diagnostics：

- path counts：
  - `hybrid_retrieval`: 300
- task counts：
  - `unknown`: 300

解读：

- Hybrid 是有价值的弱基线。
- 它在 date_time 和 encoding 中偶尔能成功，但在 computation 和 string_analysis 中暴露出纯检索增强的结构性短板。
- 用户曾问“100 样本 Hybrid 分数反而降低”的原因，结论是：大样本暴露了后检索精确推理短板，不是小幅增强把它搞坏。
- 检索 trace 显示部分 bad case 已召回相关证据，但 LLM 不能可靠完成 MD5、计数、解码、大整数计算等精确任务。

### 5.3 BaselineAgent

文件：

```text
results/latest_BaselineAgent_evaluation_data.json
results/BaselineAgent_evaluation_report.html
```

生成时间：

```text
2026-06-02 15:18:23
```

配置：

- samples：20
- tests per case：3
- total runs：60

指标：

- mean score：0.67
- Good：4
- Fail：56

解读：

- Baseline 只用于证明随机上下文窗口几乎不可用。
- 它能强化三层 Agent 对照实验的叙事清晰度。

## 6. 当前测试状态

最近一次本地测试命令：

```bash
uv run python -m unittest discover -s tests -p "test_*.py"
```

结果：

```text
Ran 16 tests in 2.928s
OK
```

覆盖测试文件：

- `tests/test_tool_augmented_agent.py`
- `tests/test_hybrid_retrieval_agent.py`
- `tests/test_agent_profiles.py`

重点覆盖：

- `abs(hash.count("F") - hash.count("b"))`
- `confirmation_code[::-1]`
- Caesar/Julius shift decode
- Base64 / URL-safe Base64 / Hex
- 日期差与星期
- 大整数四则运算与整数除法
- 工具失败后二次规划 trace
- Hybrid query expansion
- Hybrid 动态 evidence window
- Agent profile 动态读取和 fallback

## 7. 当前关键文件

### 7.1 入口与配置

- `run.py`
  - 主入口。
  - 加载 CLI 参数。
  - 健康检查。
  - 执行评测。
  - 保存结构化数据。
  - 生成 HTML 报告。

- `generate_report.py`
  - 不重新跑评测。
  - 基于已有 JSON 生成 HTML。

- `core/config.py`
  - CLI 参数定义。

- `core/ecnu_constants.py`
  - ECNU 模型名集中管理。

- `core/report_paths.py`
  - 根据 Agent 名生成 latest JSON 和 HTML 报告路径。

### 7.2 Agent

- `agents/base_agent.py`
  - OpenAI-compatible chat 调用。
  - token 估算。
  - answer finalize。

- `agents/baseline_agent.py`
  - 最小对照组。
  - 已有 `AGENT_PROFILE`。

- `agents/hybrid_retrieval_agent.py`
  - BM25 + ECNU embedding + ECNU rerank + neighbor chunk。
  - 已有 query expansion 和 retrieval trace。
  - 已有 `AGENT_PROFILE`。

- `agents/tool_augmented_agent.py`
  - Planner-first tool-augmented agent。
  - 已有 `AGENT_PROFILE`。
  - 当前主力方案。

### 7.3 评测核心

- `core/llm_multi_needle_haystack_tester.py`
  - multi 模式。
  - 多文档、多 needle。
  - 记录 `agent_trace`。

- `core/llm_single_needle_haystack_tester.py`
  - single 模式。
  - 长上下文 x 插入深度扫描。
  - 记录 `agent_trace`。

- `core/runner.py`
  - 根据 `test_mode` 调用 tester。

- `core/test_case_loader.py`
  - 加载默认测试集。
  - 支持按类型均衡抽样。
  - single 模式过滤多 needle case。

### 7.4 报告

- `core/evaluation_data.py`
  - 生成结构化数据。
  - 计算 metrics。
  - 生成 tool diagnostics。

- `core/agent_profiles.py`
  - 动态读取 Agent profile。

- `reporters/deepseek_html_reporter.py`
  - 默认 HTML 报告插件。
  - DeepSeek 只生成分析文字。
  - HTML 由程序模板渲染。

### 7.5 测试

- `tests/test_tool_augmented_agent.py`
- `tests/test_hybrid_retrieval_agent.py`
- `tests/test_agent_profiles.py`

## 8. 环境变量与依赖

迁移到新电脑后需要重新创建 `.env`。不要把真实 key 提交到仓库。

```env
ECNU_API_KEY=sk-xxxx
ECNU_BASE_URL=https://chat.ecnu.edu.cn/open/api/v1
MODEL_NAME=ecnu-max

DS_API_KEY=sk-xxxx
DS_BASE_URL=https://api.deepseek.com
DS_MODEL_NAME=deepseek-v4-pro
```

依赖安装：

```bash
uv pip install -r requirements.txt
```

如果新电脑没有 `uv`，需要先安装 `uv`。

## 9. 推荐运行流程

### 9.1 查看帮助

```bash
uv run python run.py --help
```

### 9.2 跑本地单元测试

```bash
uv run python -m unittest discover -s tests -p "test_*.py"
```

### 9.3 Tool 冒烟评测

```bash
uv run python run.py --agent agents.tool_augmented_agent:ToolAugmentedAgent --num_samples 5 --num_tests 1
```

### 9.4 Tool 默认评测

```bash
uv run python run.py --agent agents.tool_augmented_agent:ToolAugmentedAgent
```

默认：

- `test_mode=multi`
- `num_samples=20`
- `num_tests=3`
- `evaluator_type=llm`
- `generate_report=True`
- `enable_thinking=False`

### 9.5 Hybrid 对照

```bash
uv run python run.py --agent agents.hybrid_retrieval_agent:HybridRetrievalAgent
```

如果要复现最新大样本 Hybrid：

```bash
uv run python run.py --agent agents.hybrid_retrieval_agent:HybridRetrievalAgent --num_samples 100 --num_tests 3
```

### 9.6 Baseline 对照

```bash
uv run python run.py --agent agents.baseline_agent:BaselineAgent
```

### 9.7 single 模式扫描

```bash
uv run python run.py --agent agents.tool_augmented_agent:ToolAugmentedAgent --test_mode single
```

如果抽样后单 needle case 太少，可以增大样本量：

```bash
uv run python run.py --agent agents.tool_augmented_agent:ToolAugmentedAgent --test_mode single --num_samples 100
```

### 9.8 只重新生成报告

```bash
uv run python generate_report.py --input results/latest_ToolAugmentedAgent_evaluation_data.json
```

## 10. 迁移清单

1. 复制完整项目目录。
2. 确认以下目录存在：
   - `agents/`
   - `core/`
   - `evaluators/`
   - `reporters/`
   - `tests/`
   - `test_cases/`
   - `PaulGrahamEssays/`
   - `results/`
   - `docs/`
3. 在新电脑安装 `uv`。
4. 重新创建 `.env`。
5. 安装依赖：

   ```bash
   uv pip install -r requirements.txt
   ```

6. 运行单元测试：

   ```bash
   uv run python -m unittest discover -s tests -p "test_*.py"
   ```

7. 运行 Tool 冒烟：

   ```bash
   uv run python run.py --agent agents.tool_augmented_agent:ToolAugmentedAgent --num_samples 5 --num_tests 1
   ```

8. 检查输出文件：

   ```text
   results/latest_ToolAugmentedAgent_evaluation_data.json
   results/ToolAugmentedAgent_evaluation_report.html
   ```

9. 如果报告生成失败但 JSON 已生成，单独重跑：

   ```bash
   uv run python generate_report.py --input results/latest_ToolAugmentedAgent_evaluation_data.json
   ```

## 11. 已知问题与注意事项

### 11.1 README 和 AGENTS 可能仍有乱码

当前 `README.md` 和部分 `AGENTS.md` 内容在 PowerShell 中显示 mojibake，例如 `闀挎枃`、`璇勬祴`、`鈹` 等。

这份 `docs/project_handoff_summary.md` 应作为迁移时更可信的项目说明。

后续建议重写 README，内容可从本文件提取。

### 11.2 不要过度优化 Hybrid

Hybrid 当前分数低是有意义的，它作为检索增强基线暴露了纯 RAG 在精确任务上的短板。

允许继续做：

- query expansion。
- rerank 输入优化。
- neighbor chunk 调整。
- retrieval trace 增强。
- final answer prompt 微调。

不建议做：

- 给 Hybrid 加计算器。
- 给 Hybrid 加解码器。
- 给 Hybrid 加日期工具。
- 把 Hybrid 改成 ToolAugmentedAgent。

### 11.3 Tool 仍有提升空间

Tool 最新小样本回归 mean score 8.00，但仍有失败：

- `missing_required_evidence`: 6
- `invalid_operation_parse`: 2

优先改进方向：

- evidence completeness。
- LLM planner 输出稳定性。
- operation parse 覆盖。
- string_analysis 中更复杂的字符/位置任务。
- computation 中更复杂公式解析。

原则：

- 不继续堆 case-specific alias。
- 只新增通用工具 primitive。
- 失败 trace 要能解释为什么失败。

### 11.4 小样本和大样本不要混淆

当前最新 Tool 是 20 cases x 3 runs，Hybrid 是 100 cases x 3 runs。

简历或报告中可以写：

- Tool 最新回归 mean score 8.00。
- Hybrid 大样本 mean score 2.06。
- 这些结果共同支持“工具增强比纯检索增强更适合精确任务”的判断。

但不要写成严格同条件 A/B 实验。

如果要做更严谨对照，建议重新跑：

```bash
uv run python run.py --agent agents.tool_augmented_agent:ToolAugmentedAgent --num_samples 100 --num_tests 3
uv run python run.py --agent agents.hybrid_retrieval_agent:HybridRetrievalAgent --num_samples 100 --num_tests 3
```

### 11.5 上游 API 可能不稳定

健康检查或评测中可能遇到：

- ECNU 服务临时 500。
- key 配置错误。
- base URL 错误。
- DeepSeek 报告生成失败。

判断方式：

- 如果评测开始前健康检查失败，先查 ECNU 配置。
- 如果评测已完成但报告失败，先查 DeepSeek 配置。
- 报告失败不代表评测数据无效，可以用 JSON 重新生成。

## 12. 后续优化优先级

### P0：迁移后先确认可运行

1. 配 `.env`。
2. 安装依赖。
3. 跑 16 个单元测试。
4. 跑 Tool 5 样本冒烟。

### P1：重写 README

当前 README 乱码，建议迁移后第一件文档工作就是重写 README。

README 应包含：

- 项目定位。
- 三层 Agent 架构。
- multi / single 模式。
- 默认命令。
- 环境变量。
- 报告输出。
- 最新指标摘要。
- 诚信约束。

### P2：跑同条件 Tool vs Hybrid 大样本对照

为了让简历和项目报告更严谨，可以跑同条件：

- Tool：100 samples x 3 runs。
- Hybrid：100 samples x 3 runs。

观察：

- Tool mean score 是否保持在 6.27 或更高。
- `tool_augmented_failed` 是否下降。
- `unsupported_task_type` 是否明显减少。
- encoding / computation / string_analysis 的提升是否稳定。

### P3：继续优化 Tool 失败项

优先看 latest Tool 中的 12 个 fail：

- 是否证据没召回。
- 是否 planner 没抽对字段。
- 是否 operation parse 失败。
- 是否工具执行后格式不符合答案。

只针对通用能力补齐，不为单个 case 写硬编码。

### P4：保留项目叙事

不要把这个项目继续扩成“大而全 Agent 平台”。它最适合的求职叙事是：

- AI Agent 评测。
- 能力边界诊断。
- Bad Case 归因。
- 数据驱动产品迭代。
- RAG 与工具增强方案对照。

## 13. 简历使用建议

### 13.1 推荐项目标题

```text
长上下文 Agent 评测与诊断系统｜AI Agent 能力评估与产品迭代实践
```

### 13.2 推荐精简版项目描述

```text
面向长上下文检索、多文档证据聚合与精确推理场景，搭建 Needle-in-a-Haystack 自动化评测体系，构建 1600 条测试集，覆盖计算、日期、编码、字符串分析等典型 Agent 失败场景，用于评估 LLM / RAG / Agent 在复杂任务中的能力边界。

设计 Baseline / Hybrid Retrieval / Tool-Augmented 三层 Agent 对照实验，区分“无检索”“检索增强”“工具增强”不同方案的效果差异，帮助判断哪些问题适合靠 Prompt / 检索优化解决，哪些必须引入规划与工具执行。

建立结构化评测报告与 Bad Case 归因机制，沉淀得分分布、任务类型表现、执行路径、失败原因、检索 trace、工具 trace 等指标，将模型问题归因为证据缺失、检索后推理失败、操作解析错误、输出格式偏差等可迭代方向。

设计面向产品迭代的双模式评测：用 multi 模式验证多文档 RAG 场景下的证据召回与聚合能力，用 single 模式扫描上下文长度和信息位置对回答稳定性的影响，辅助判断 Prompt、检索策略、工具增强和模型选型的优化优先级。

基于评测结果推动 Agent 方案迭代，将 ToolAugmentedAgent 优化为“LLM 规划 + 确定性工具执行”架构；最新回归 mean score 达 8.00，验证工具增强与诊断闭环对精确任务稳定性的价值。
```

### 13.3 与另外两个项目的排序

投 AI Agent / AI Coding / 模型评测 / 开发者工具：

1. 长上下文 Agent 评测与诊断系统
2. DeepCast 深度调研 AI 播客生成 Agent
3. 多模态 RAG 图文问答系统

投 AIGC 内容产品 / 创作者工具：

1. DeepCast 深度调研 AI 播客生成 Agent
2. 长上下文 Agent 评测与诊断系统
3. 多模态 RAG 图文问答系统

投企业知识库 / RAG / MaaS / B 端 AI：

1. 多模态 RAG 图文问答系统
2. 长上下文 Agent 评测与诊断系统
3. DeepCast 深度调研 AI 播客生成 Agent

### 13.4 不建议写法

不要写：

- 已达到生产级。
- 100% 解决长上下文推理。
- 完成多模型大规模横评。
- Tool 与 Hybrid 严格同条件大样本对比已经完成。

可以写：

- 搭建评测体系。
- 完成多轮 Agent 迭代。
- 通过结构化报告进行 Bad Case 归因。
- 用最新回归验证工具增强和诊断闭环价值。

## 14. 记忆核对后的补充说明

本节是和本地 Codex 记忆对照后补充的迁移提醒。它们不是新的需求，而是为了避免换电脑后丢掉一些容易被忽略的上下文。

### 14.1 低成本验证命令

除了单元测试和 5 样本冒烟，之前做过的低成本验证还包括：

```bash
uv run python -m compileall -q .
uv run python run.py --help
```

`compileall` 用来确认 Python 文件没有语法错误；`run.py --help` 用来确认 CLI 参数面没有因为重构而损坏。

### 14.2 ECNU API skill 的使用边界

项目曾参考 `ecnu-api` skill 检查 ECNU OpenAI-compatible API 调用方式。迁移后如果继续排查 ECNU 模型、Embedding 或 Rerank 调用，应优先确认：

- `core/ecnu_constants.py` 中模型名是否仍然正确。
- `ECNU_BASE_URL` 是否仍为 `https://chat.ecnu.edu.cn/open/api/v1`。
- 主测模型、评分模型、Embedding、Rerank 是否都能通过健康检查。
- 不要在没有 API 文档要求时额外设置 `temperature`、`top_p`、`max_tokens`、`thinking_budget_tokens` 等参数；项目当前倾向尽可能使用服务端默认参数。

### 14.3 指标使用边界

记忆中可用于简历或项目说明的量化锚点是：

- `ToolAugmentedAgent`：最新小样本回归 mean score `8.00`，范围是 `20 cases x 3 runs`。
- `HybridRetrievalAgent`：最新大样本回归 mean score `2.06`，范围是 `100 cases x 3 runs`。

这两个数字可以共同说明工具增强路线的价值，但不能写成严格同条件 A/B 对比。若要做严格对比，需要重新用相同样本量和相同运行次数跑 Tool 与 Hybrid。

### 14.4 不属于本项目的记忆

本地记忆中还包含 DeepCast、ecnu-api skill 开源发布、多模态 RAG 等其他项目的信息。它们只在“简历项目排序”和“项目叙事区分”中有参考价值，不应混入本项目的代码结构、评测指标或运行命令。

## 15. 当前最重要的上下文结论

- 项目核心方向已经清楚：AI Agent 评测与诊断，不是单纯 RAG Demo。
- 公开 Agent 保持三层：Baseline、Hybrid、Tool。
- Hybrid 保持检索增强基线，只做小幅检索增强。
- Tool 是主力，已经改成 planner-first。
- 每个 Agent 都应维护自己的 `AGENT_PROFILE`。
- 报告分析 prompt 会引用 Agent profile。
- 核心模型 prompt 尽量保持英文，报告分析输出简体中文。
- 评测前必须做健康检查，不建议恢复跳过健康检查开关。
- 报告采用结构化 JSON + HTML 看板，不再走 PNG 图片可视化。
- 每个 Agent 保留自己的 latest JSON 和 HTML report。
- 默认 CLI 已收敛：`multi`、20 samples、3 runs、LLM judge、自动生成报告。
- 最新 Tool 小样本回归 mean score 8.00。
- 最新 Hybrid 大样本 mean score 2.06。
- 单元测试当前 16 个通过。
- 迁移后优先确认 `.env`、依赖、单测、5 样本 Tool 冒烟。
- README/AGENTS 存在乱码，本文件是当前可信交接文档。
