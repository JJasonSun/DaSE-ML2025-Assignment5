# LLM 长文本与复杂推理自动化评测平台

本项目是一个面向大模型长上下文检索与复杂推理能力的 Needle-in-a-Haystack 评测框架。它把测试用例抽样、Agent 执行、LLM-as-a-Judge 评分、结构化数据落盘和 HTML 评测报告生成串成一条自动化链路，重点服务于 RAG/Agent 产品方案的能力验证与 bad case 归因。

## 核心设计

### 1. 三层 Agent 演进

项目不再保留多个职责重叠的 Agent，而是收敛成一条清晰的产品化演进链路：

| Agent | 定位 | 作用 |
| :--- | :--- | :--- |
| `BaselineAgent` | 最小对照组 | 随机抽取上下文窗口，用于验证评测框架和提供弱基线 |
| `HybridRetrievalAgent` | 检索增强基线 | BM25 + Dense Embedding + Rerank + 邻近 chunk 补充 |
| `ToolAugmentedAgent` | 推荐主力方案 | 混合检索 + 场景识别 + 结构化抽取 + Python 工具精确推理 |

`ToolAugmentedAgent` 的设计重点是减少模型“心算”：LLM 负责证据定位和结构化抽取，Python 工具负责计算、日期、字符串与编码处理，最终统一清洗为简洁答案。

### 2. 两种评测模式

#### `multi` 模式：多文档多 needle 检索

默认推荐模式。每条测试用例可以包含多个 needle，系统会把这些 needle 随机插入不同 haystack 文件的不同深度位置。Agent 接收结构化 `context_data`，需要跨文件检索、聚合信息并推理作答。

它更接近真实 RAG 场景，适合评估多文档召回、多 needle 聚合、复杂推理和答案格式稳定性。

#### `single` 模式：上下文长度 x 插入深度扫描

`single` 模式会把 haystack 拼接成一个长上下文字符串，把单条 needle 插入指定深度，并遍历 `context_length x depth_percent` 网格。

它适合观察模型在不同上下文长度、不同插入位置下的退化区间。该模式通常比 `multi` 更耗时，主要用于长上下文能力边界分析。

## 评测方法论

测试集 `test_cases/test_cases_all_en.json` 包含 1600 条英文测试用例，覆盖四类任务：

| 类型 | 评测重点 |
| :--- | :--- |
| `computation` | 大整数、差值、整除、比例等精确计算 |
| `date_time` | 日期、星期、偏移、截止时间推理 |
| `encoding` | Base64、Hex、Caesar 等编码识别与解码 |
| `string_analysis` | 字符串计数、位置、子串和模式分析 |

默认从题库中按类型均衡抽样 20 条，每条在 `multi` 模式下重复 3 次，以降低随机插入位置带来的偶然性。

报告会输出：

- `results/latest_evaluation_data.json`：最近一次结构化评测数据，只保留一份
- `results/evaluation_report.html`：纯 HTML 数据看板与 AI 分析报告

HTML 报告基于结构化数据展示总览指标、类型表现、分数分布、single 模式热力表、bad case 表和工具链诊断。报告分析文本由 `deepseek-v4-pro` 基于结构化数据生成，不依赖图片或多模态读图。

## 项目结构

```text
agents/
├── base_agent.py
├── baseline_agent.py
├── hybrid_retrieval_agent.py
└── tool_augmented_agent.py
core/
├── agent_factory.py
├── config.py
├── evaluation_data.py
├── health_check.py
├── reporter_factory.py
├── runner.py
└── test_case_loader.py
evaluators/
├── llm_evaluator.py
└── string_match_evaluator.py
reporters/
└── deepseek_html_reporter.py
run.py
generate_report.py
```

## 快速开始

### 1. 环境配置

创建 `.env`：

```env
ECNU_API_KEY=sk-xxxx
ECNU_BASE_URL=https://chat.ecnu.edu.cn/open/api/v1
MODEL_NAME=ecnu-max

DS_API_KEY=sk-xxxx
DS_BASE_URL=https://api.deepseek.com
DS_MODEL_NAME=deepseek-v4-pro
```

安装依赖：

```bash
uv pip install -r requirements.txt
```

所有评测都会先执行健康检查，确认主测模型、评分模型、Embedding 和 Rerank 服务可用。

### 2. 运行评测

#### 快速冒烟测试

使用和正式评测相同的 `multi` 流程，只是样本量更小、每条用例只重复 1 次。

```bash
uv run python run.py --agent agents.tool_augmented_agent:ToolAugmentedAgent --num_samples 5 --num_tests 1
```

#### 推荐：默认正式评测

默认抽样 20 条测试用例，每条用例重复 3 次，并生成 HTML 报告。

```bash
uv run python run.py --agent agents.tool_augmented_agent:ToolAugmentedAgent
```

#### 扩大样本量

```bash
uv run python run.py --agent agents.tool_augmented_agent:ToolAugmentedAgent --num_samples 50
```

#### 检索基线对照

```bash
uv run python run.py --agent agents.hybrid_retrieval_agent:HybridRetrievalAgent
```

#### 最小 baseline 对照

```bash
uv run python run.py --agent agents.baseline_agent:BaselineAgent
```

#### single 模式：上下文长度 x 插入深度扫描

```bash
uv run python run.py --agent agents.tool_augmented_agent:ToolAugmentedAgent --test_mode single
```

#### 开启思考模式

默认关闭。只在需要提升复杂推理质量时开启，因为会增加延迟和调用成本。

```bash
uv run python run.py --agent agents.tool_augmented_agent:ToolAugmentedAgent --enable_thinking True
```

#### 只重生成 HTML 报告

不重新跑评测，直接使用最近一次结构化数据生成报告。

```bash
uv run python generate_report.py --input results/latest_evaluation_data.json --output results/evaluation_report.html
```

#### 关闭报告生成

```bash
uv run python run.py --agent agents.tool_augmented_agent:ToolAugmentedAgent --generate_report False
```

### 3. 常用参数

| 参数 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `--agent` | 必填 | Agent 路径，格式为 `module.path:ClassName` |
| `--test_case_json` | `test_cases/test_cases_all_en.json` | 测试题库路径 |
| `--num_samples` | `20` | 按类型均衡抽样的用例数 |
| `--test_mode` | `multi` | `multi` 或 `single` |
| `--evaluator_type` | `llm` | `llm` 语义评分或 `string` 精确匹配 |
| `--num_tests` | `3` | `multi` 模式下每个用例的重复次数 |
| `--enable_thinking` | `False` | 是否开启主测模型 thinking |
| `--generate_report` | `True` | 是否生成 HTML 评测报告 |

## 扩展自定义 Agent

1. 在 `agents/` 下创建新文件
2. 继承 `ModelProvider`，或继承 `HybridRetrievalAgent` 复用检索管线
3. 实现 `async evaluate_model(prompt: Dict) -> str`
4. 运行时使用 `--agent agents.your_agent:YourAgent`

## 诚信约束

- Agent 不得直接读取 `PaulGrahamEssays/*.txt` 原始文件
- Agent 不得读取测试集 JSON
- Agent 不得硬编码答案
