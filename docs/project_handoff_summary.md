# Project Handoff Summary

生成日期：2026-06-06
原始项目路径：`D:\Workspace\Study\ML-Assignment-5`

这份文档用于迁移项目到另一台电脑时保留上下文。它总结了本轮协作中围绕项目定位、CLI、Agent 架构、评测报告、模型调用、提示词和已知问题做过的主要讨论与改造。

## 1. 项目定位

本项目最初是一个 LLM Needle-in-a-Haystack 评测框架，用于评估大模型在长上下文、多文档、多 needle 场景下的检索、信息聚合和复杂推理能力。

后续围绕 AI 产品经理岗位展示需求，项目叙事被调整为：

- 不只是“写了一个 Agent”，而是一个可评估、可诊断、可回归优化的 AI 系统评测平台。
- 核心能力展示点包括：RAG/Agent 评测设计、bad case 归因、工具增强推理、自动化 HTML 报告、结构化评测数据沉淀。
- 与另一个“自动播客生成 Agent”项目形成区分：播客项目偏 AI 产品应用，这个项目偏 AI 系统评测、产品化诊断和工程闭环。

## 2. 主要需求与决策时间线

### 2.1 Prompt 与语言策略

早期需求是理解项目并优化 prompt，同时把 prompt 改成英文。随后一度希望项目中的提示词改成中文，但由于 Windows/PowerShell 环境下多次出现中文源码乱码，最终决定：

- 所有会发给模型的 prompt 统一改为英文。
- DeepSeek 报告分析 prompt 也使用英文，但要求模型输出简体中文分析正文。
- Python 源码中尽量避免直接写中文 prompt，降低编码损坏风险。
- UI 文案和文档可以继续使用中文，但需要确保 UTF-8 保存。

这个决策的背景是：项目中曾出现大量类似 `浣犳槸...`、`锛`、`鈥` 的 mojibake 乱码，直接影响模型理解 prompt。

### 2.2 项目结构整理

用户希望检查并优化项目结构，减少零散脚本和冗余文件。主要调整方向：

- 将自动可视化/报告生成集成到主流程，不再依赖单独脚本手动运行。
- 删除单独 `visualizations/` 文件夹下只有一个可视化脚本的结构。
- 不再保留 PNG 图片可视化输出。
- 不再把每次评测结果都保存成多份 JSON，只保留最近一次结构化数据。
- 后续又调整为按 Agent 生成结构化数据和 HTML 报告，避免不同 Agent 的结果互相覆盖。

当前报告输出逻辑由 `core/report_paths.py` 管理：

- 默认结构化数据：`results/latest_<AgentName>_evaluation_data.json`
- 默认 HTML 报告：`results/<AgentName>_evaluation_report.html`

例如：

- `results/latest_ToolAugmentedAgent_evaluation_data.json`
- `results/ToolAugmentedAgent_evaluation_report.html`

### 2.3 测试用例与 CLI 简化

用户希望只使用 `test_cases/test_cases_all_en.json`，每次从中按类型均衡抽样，而不是命令行每次都指定测试 JSON。

当前行为：

- `--test_case_json` 可选。
- 默认使用 `test_cases/test_cases_all_en.json`。
- 默认 `--num_samples 20`。
- 默认 `--num_tests 3`。
- 默认 `--test_mode multi`。
- 默认 `--evaluator_type llm`。
- 默认 `--generate_report True`。
- 默认 `--enable_thinking False`。

快速冒烟测试建议：

```bash
uv run python run.py --agent agents.tool_augmented_agent:ToolAugmentedAgent --num_samples 5 --num_tests 1
```

正式默认评测建议：

```bash
uv run python run.py --agent agents.tool_augmented_agent:ToolAugmentedAgent
```

扩大样本量：

```bash
uv run python run.py --agent agents.tool_augmented_agent:ToolAugmentedAgent --num_samples 50
```

只重新生成报告：

```bash
uv run python generate_report.py --input results/latest_ToolAugmentedAgent_evaluation_data.json
```

### 2.4 `multi` 与 `single` 模式解释

用户曾不理解 `multi` 和 `single` 的区别，后来 README 中做过解释。当前理解如下：

#### `multi`

默认推荐模式。每条测试用例可能包含多个 needle，系统会把这些 needle 随机插入不同 haystack 文件的不同深度。

Agent 接收的是结构化 `context_data`，需要完成：

- 多文档检索
- 多 needle 聚合
- 跨片段证据组合
- 计算、日期、编码、字符串等复杂推理

它更接近真实 RAG/Agent 场景。

#### `single`

长上下文扫描模式。只支持单 needle case。它把 haystack 拼成一个长上下文字符串，将 needle 插入指定深度，然后扫描 `context_length x depth_percent` 网格。

它适合分析：

- 长上下文长度敏感区间
- needle 插入深度敏感区间
- 中间位置遗忘
- 上下文窗口退化

它通常比 `multi` 更耗时，不适合作为快速冒烟测试。

### 2.5 健康检查强制化

用户希望删除健康检查开关，所有评测必须先做健康检查。

当前行为：

- `--skip_model_test` 已删除。
- `run.py` 无条件调用 `core.health_check.check_models(...)`。
- 健康检查覆盖：
  - 主测模型
  - LLM 评分模型
  - Embedding 模型
  - Rerank 模型

如果健康检查失败，评测应直接中止，不进入测试循环。

### 2.6 HTML 报告替代图片可视化

项目曾经支持 PNG 可视化，后来用户明确希望：

- 不生成图片。
- 不引用图片。
- 不让模型读图。
- 直接生成一个 HTML 数据看板。

当前报告架构：

- 结构化数据由 `core/evaluation_data.py` 生成。
- 报告插件由 `core/reporter_factory.py` 加载。
- 默认报告插件是 `reporters.deepseek_html_reporter:DeepSeekHtmlReporter`。
- 报告输出为 HTML。
- DeepSeek 只负责生成分析文字，不负责生成 HTML 页面结构。
- HTML 页面由程序模板渲染，避免模型输出不可控 HTML。

报告中包含：

- 概览指标
- 分数分布
- 分类型表现
- 工具链诊断
- `multi` 模式逐次表现
- `single` 模式热力图
- Bad Case 表格
- AI 产品评测分析

DeepSeek 报告分析配置：

- `DS_API_KEY`
- `DS_BASE_URL`
- `DS_MODEL_NAME`

默认模型名：

```env
DS_MODEL_NAME=deepseek-v4-pro
```

### 2.7 DeepSeek 报告插件独立化

用户希望数据分析和写报告作为独立插件存在，可以在跑完评测后自动生成，也可以单独基于本地结构化数据生成。

当前实现：

- 评测结束后自动生成结构化数据和 HTML 报告。
- `generate_report.py` 可以单独读取本地结构化数据生成 HTML。
- 报告插件实现位于 `reporters/deepseek_html_reporter.py`。
- 报告插件基类位于 `reporters/base_reporter.py`。

### 2.8 Agent 收敛与重命名

用户觉得原来的 `agents/` 文件夹中 Agent 太多，不利于 AI 产品经理岗位叙事。最终决策是收敛为三类公开 Agent：

| Agent                    | 文件                                 | 定位                  |
| :----------------------- | :----------------------------------- | :-------------------- |
| `BaselineAgent`        | `agents/baseline_agent.py`         | 最小 baseline，对照组 |
| `HybridRetrievalAgent` | `agents/hybrid_retrieval_agent.py` | 混合检索基线          |
| `ToolAugmentedAgent`   | `agents/tool_augmented_agent.py`   | 推荐主力方案          |

旧 Agent 已不再作为公开入口：

- `ExampleAgent`
- `SyncRetrievalAgent`
- `AdvancedRetrievalAgent`
- `ScenarioAwareAgent`
- `agent_plus.py`
- `agent_template.py`
- `sync_agent.py`
- `scenario_agent.py`

当时明确决策是不保留旧导入路径兼容 shim，因为项目当前更重视求职展示和架构清晰度。

推荐主命令：

```bash
uv run python run.py --agent agents.tool_augmented_agent:ToolAugmentedAgent
```

检索基线对照：

```bash
uv run python run.py --agent agents.hybrid_retrieval_agent:HybridRetrievalAgent
```

最小 baseline：

```bash
uv run python run.py --agent agents.baseline_agent:BaselineAgent
```

### 2.9 Tool-Augmented Agent 的优化方向

DeepSeek 报告曾指出以下失败模式：

- computation 分数很低。
- encoding 经常返回编码前字符串，而不是解码结果。
- date_time 部分失败来自日期格式不支持。
- string_analysis 容易和 encoding 混淆。
- 工具链大量 `tool_execution_failed`，导致回退到 Hybrid。

基于最新报告分析，发现一个关键 case：

- case 574 的结构化抽取已经拿到所有数字和公式。
- 标准答案是 `14964689780889120`。
- 旧工具输出是 `1`。
- 根因不是检索，而是计算工具用关键词粗暴选择了 `numbers[0] // numbers[1]`，没有执行抽取出的公式。

最新优化已完成：

- 分类顺序调整，避免字符串统计题被误判为 encoding。
- 结构化抽取 prompt 改为英文。
- computation 优先执行抽取出的 Python-style `operation`。
- 用受限 AST 执行表达式，只允许基础算术。
- date_time 支持 `YYYY-MM-DD`、`YYYY-M-D`、`Month DD, YYYY`。
- encoding 支持 Base64、Base64URL、Base32、Hex、Caesar。
- encoding 会优先选择真正的 encoded payload，而不是协议名或 message id。
- string_analysis 支持十六进制数字求和等确定性规则。

本地非 API 验证曾通过：

- case 574 公式输出 `14964689780889120`
- Base64 `TEFNQkRBMzA5` 解码输出 `LAMBDA309`
- ISO 日期可计算 weekday
- 字符串十六进制数字求和可执行
- `sum of all hexadecimal digits ... encoded data` 不再被分类为 encoding

注意：这些是本地工具函数验证，不是完整真实评测。

## 3. 当前关键文件说明

### 3.1 入口与配置

- `run.py`

  - 命令行主入口。
  - 加载 CLI 参数。
  - 强制执行健康检查。
  - 加载并抽样测试用例。
  - 执行评测。
  - 保存结构化数据。
  - 调用报告插件生成 HTML。
- `generate_report.py`

  - 不跑评测，只基于本地结构化数据生成 HTML 报告。
- `core/config.py`

  - CLI 参数定义。
  - 注意：当前文件注释/文档字符串仍有乱码，但参数本身可用。
- `core/report_paths.py`

  - 根据 Agent 名称生成结构化数据和报告路径。

### 3.2 Agent

- `agents/base_agent.py`

  - Agent 基类。
  - 包含 OpenAI-compatible chat 调用、thinking 开关、token 编码、答案清洗。
- `agents/baseline_agent.py`

  - 随机选择上下文窗口。
  - 仅作为弱 baseline。
- `agents/hybrid_retrieval_agent.py`

  - 混合检索：
    - BM25
    - Embedding
    - Rerank
    - neighbor chunk 补充
  - 当上下文 token 数小于阈值时，会直接使用完整上下文。
- `agents/tool_augmented_agent.py`

  - 当前推荐主力 Agent。
  - 继承 `HybridRetrievalAgent`。
  - 增加任务分类、结构化抽取和 Python 工具推理。

### 3.3 评测核心

- `core/llm_multi_needle_haystack_tester.py`

  - 多文档、多 needle 测试。
  - 会把 needle 插入不同文件。
  - 结果中记录 `agent_trace`。
- `core/llm_single_needle_haystack_tester.py`

  - 单文档、单 needle 长上下文扫描。
  - 结果中记录 `agent_trace`。
- `core/runner.py`

  - 根据 `test_mode` 调用 single 或 multi tester。
  - 初始化 evaluator。
- `core/test_case_loader.py`

  - 加载测试用例。
  - 支持按类型均衡抽样。
  - single 模式会过滤多 needle case。

### 3.4 评测器

- `evaluators/llm_evaluator.py`

  - 使用 `ecnu-plus` 进行 LLM-as-a-Judge 评分。
  - prompt 已改成英文。
- `evaluators/string_match_evaluator.py`

  - 精确字符串匹配评测器。
  - 文档字符串仍有中文，但不影响模型调用。

### 3.5 报告

- `core/evaluation_data.py`

  - 把 `all_results` 和 summaries 转成结构化评测数据。
  - 包含工具链诊断：
    - path counts
    - fallback counts
    - task type counts
- `reporters/deepseek_html_reporter.py`

  - DeepSeek HTML 报告插件。
  - prompt 使用英文。
  - 要求 DeepSeek 输出简体中文分析正文。
  - 内置 HTML/CSS 模板。
  - 对模型生成的 Markdown 做安全转换。
  - 对模型误输出 HTML 的情况做清洗。

## 4. 当前模型与环境变量

迁移到新电脑后需要准备 `.env`。不要把真实 key 写入公开仓库。

```env
ECNU_API_KEY=sk-xxxx
ECNU_BASE_URL=https://chat.ecnu.edu.cn/open/api/v1
MODEL_NAME=ecnu-max

DS_API_KEY=sk-xxxx
DS_BASE_URL=https://api.deepseek.com
DS_MODEL_NAME=deepseek-v4-pro
```

ECNU 模型名集中在：

```text
core/ecnu_constants.py
```

DeepSeek 报告模型默认值在：

```text
reporters/deepseek_html_reporter.py
```

## 5. 当前推荐运行流程

### 5.1 安装依赖

```bash
uv pip install -r requirements.txt
```

### 5.2 快速冒烟测试

```bash
uv run python run.py --agent agents.tool_augmented_agent:ToolAugmentedAgent --num_samples 5 --num_tests 1
```

预期流程：

1. 读取 `.env`。
2. 执行健康检查。
3. 从 `test_cases/test_cases_all_en.json` 按类型均衡抽样 5 条。
4. 使用 `multi` 模式运行，每条重复 1 次。
5. 保存结构化数据。
6. 生成 HTML 报告。

### 5.3 正式默认评测

```bash
uv run python run.py --agent agents.tool_augmented_agent:ToolAugmentedAgent
```

默认是：

- `multi`
- `num_samples=20`
- `num_tests=3`
- `evaluator_type=llm`
- `generate_report=True`
- `enable_thinking=False`

### 5.4 对照实验

```bash
uv run python run.py --agent agents.hybrid_retrieval_agent:HybridRetrievalAgent
uv run python run.py --agent agents.baseline_agent:BaselineAgent
```

### 5.5 Single 模式扫描

```bash
uv run python run.py --agent agents.tool_augmented_agent:ToolAugmentedAgent --test_mode single
```

如果 single 模式采样后没有单 needle case，可以增大样本量：

```bash
uv run python run.py --agent agents.tool_augmented_agent:ToolAugmentedAgent --test_mode single --num_samples 100
```

### 5.6 只重生成报告

```bash
uv run python generate_report.py --input results/latest_ToolAugmentedAgent_evaluation_data.json
```

## 6. 已知问题与风险

### 6.1 README 当前存在乱码

当前 `README.md` 在读取时显示大量 mojibake 乱码，例如 `闀挎枃`、`璇勬祴` 等。它不应作为当前可信交接文档。

建议后续优先重写 README，或者从本文件提取内容重建 README。

### 6.2 部分 Python 注释/文档字符串仍有中文乱码

目前影响模型调用的 prompt 已重点清理为英文，但以下文件仍可能包含中文注释或 docstring，部分存在乱码：

- `core/config.py`
- `run.py`
- `core/test_case_loader.py`
- `core/runner.py`
- `core/llm_multi_needle_haystack_tester.py`
- `core/llm_single_needle_haystack_tester.py`
- `evaluators/string_match_evaluator.py`

这些主要是注释/文档字符串，不一定影响运行，但会影响代码可读性。迁移后如果继续整理，建议统一改成英文注释。

### 6.3 最新 Agent 优化尚未跑真实评测

最新一次对 `ToolAugmentedAgent` 的工具链修复只做了本地函数级验证，没有跑完整 ECNU/DeepSeek API 评测。

原因是完整评测会产生 API 成本。

迁移后建议先跑：

```bash
uv run python run.py --agent agents.tool_augmented_agent:ToolAugmentedAgent --num_samples 5 --num_tests 1
```

重点观察：

- computation 平均分是否明显提升。
- `tool_execution_failed` 是否下降。
- encoding 是否还返回编码原文。
- date_time 是否还因日期格式失败。
- HTML 报告中的 AI 分析是否仍出现 HTML 源码。

### 6.4 健康检查可能因上游服务不稳定失败

曾出现过 `openai.InternalServerError: Internal Server Error`。这类错误通常不是本地代码逻辑错误，而是上游模型服务、API key、base URL 或临时服务状态问题。

健康检查失败时，应该先确认：

- `.env` 是否正确。
- ECNU API 是否可用。
- DeepSeek 只在生成报告时需要，不影响 ECNU 健康检查。
- 是否是上游临时 500。

### 6.5 不要恢复旧 Agent 路径

旧命令如：

```bash
uv run python run.py --agent agents.agent_plus:AdvancedRetrievalAgent
```

已经不再推荐，也可能不可用。当前推荐使用：

```bash
uv run python run.py --agent agents.tool_augmented_agent:ToolAugmentedAgent
```

## 7. 简历与项目叙事建议

如果投递 AI 产品经理相关岗位，这个项目更适合写成“AI 系统评测平台/Agent 评测平台”，不要只写成“RAG Agent”。

推荐表达方向：

> 设计并实现面向长上下文检索与复杂推理的 LLM 自动化评测平台，支持多文档多 needle 场景、按任务类型均衡抽样、LLM-as-a-Judge 评分、结构化 bad case 归因与 HTML 数据看板报告。进一步构建 Baseline、Hybrid Retrieval、Tool-Augmented 三层 Agent 对照体系，将计算、日期、编码和字符串任务从模型心算迁移到 Python 工具链，提升评测结果的可解释性和可回归优化能力。

可以强调的能力：

- AI 产品评测方法论
- 指标体系设计
- bad case 归因
- Agent 架构收敛
- RAG 检索链路理解
- 工具增强推理
- 报告自动化
- 对比实验设计
- 工程化 CLI 和默认参数设计

避免夸大的点：

- 不要说已经做了大规模多模型对比，因为个人资源有限，最终决定不做。
- 不要说所有优化已被真实评测证明，因为最新工具链优化尚未跑完整评测。
- 不要把本地个人迭代写成公开社区合并成果。

## 8. 迁移清单

迁移到另一台电脑时建议按以下顺序检查：

1. 复制完整项目目录。
2. 确认 `.env` 已重新创建，不要依赖旧电脑环境变量。
3. 安装 `uv`。
4. 执行：

   ```bash
   uv pip install -r requirements.txt
   ```
5. 静态检查：

   ```bash
   uv run python -m compileall -q .
   uv run python run.py --help
   ```
6. 冒烟评测：

   ```bash
   uv run python run.py --agent agents.tool_augmented_agent:ToolAugmentedAgent --num_samples 5 --num_tests 1
   ```
7. 检查输出：

   ```text
   results/latest_ToolAugmentedAgent_evaluation_data.json
   results/ToolAugmentedAgent_evaluation_report.html
   ```
8. 如果报告生成失败但评测数据存在，单独重生成：

   ```bash
   uv run python generate_report.py --input results/latest_ToolAugmentedAgent_evaluation_data.json
   ```

## 9. 后续建议

优先级从高到低：

1. 重写乱码 README，让它与当前真实 CLI 和 Agent 架构一致。
2. 跑一次 5 样本冒烟测试，验证最新 `ToolAugmentedAgent` 修复是否有效。
3. 若冒烟结果明显提升，再跑默认 20 样本正式评测。
4. 对 latest structured data 做 bad case 复盘，继续优化工具链。
5. 把 `core/` 和 `evaluators/` 中不影响运行但乱码的注释/docstring 改成英文。
6. 如需简历展示，保留一份表现较好的 HTML 报告截图或部署版本，但仓库中不提交结果图片。

## 10. 当前最重要的上下文结论

这个项目当前的核心方向已经很清楚：

- 默认不要再扩展很多 Agent。
- 保持三层 Agent 对照体系。
- 默认推荐 `ToolAugmentedAgent`。
- 默认跑 `multi`。
- 所有评测必须健康检查。
- 输出结构化数据和 HTML 报告。
- 模型调用 prompt 统一英文。
- DeepSeek 只做报告文字分析，不生成 HTML。
- 最新工具链优化需要真实冒烟测试验证。
