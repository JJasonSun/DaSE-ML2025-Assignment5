# LLM 长文本与复杂推理自动化测评平台 (NIAH Evaluation Framework)

## 📌 项目定位

这是一个面向大语言模型（LLM）的**高自动化、可插拔的检索与复杂推理评估平台**。

在真实的 AI 产品落地中，大模型的长文本理解能力（Context Window）往往伴随着“中间迷失”与“幻觉”。本项目基于强化版的“大海捞针 (Needle in a Haystack)”范式，构建了一套完整的自动化评测管线。通过多维度的测试用例、自动化的评分引擎以及完善的数据分析面板，帮助技术与产品团队深挖模型能力边界、打磨 RAG（检索增强生成）管线，并为产品的模型选型提供数据支撑。

---

## 💡 核心架构与产品思考

本项目不只是一个测试脚本，而是一套沉淀了最佳实践的 AI 检索测试产品。

### 1. 接口化 Agent 架构（Pluggable Architecture）
* **实现**：`agents/base_agent.py` 定义了标准化的底层接口，从 baseline 的 `SyncRetrievalAgent` 到复杂的 `AdvancedRetrievalAgent` 均可插拔替换。
* **产品思考**：评测引擎的生命力在于“解耦”。将其设计为插件式，意味着当我们需要评估全新的大模型、或者测试一种最新的 Prompt 技术时，不需要修改任何评测循环的底层逻辑。研发与产品能以极低的成本（新增一个文件）快速开展对照实验。

### 2. 混合检索管线（Hybrid Retrieval Pipeline）
* **实现**：在 `AdvancedRetrievalAgent` 内封装了 **BM25关键词粗排 + Dense Embedding向量检索 + Rerank 精排把关** 的黄金三路召回。
* **产品思考**：为什么不只用纯向量检索？因为在实际业务中我们会发现，纯向量对于同义词很强，但对于“硬规则（如ID、数字、罕见编码）”极易漏召回。将混合检索流水线内置入 Agent，能够探索模型在不同信息信噪比下的鲁棒性，从而反哺真实产品中 RAG 策略的调优。

### 3. 场景分类与动态路由（Scenario Routing & Dynamic Prompting）
* **实现**：`ScenarioAwareAgent` 通过小模型（如 ecnu-plus）对用户意图进行前置判别并计算置信度（Confidence），进而动态挂载对应场景（日期推算、逻辑计算等）的专业 Prompt。
* **产品思考**：One prompt doesn't fit all. 业务落地中 API 成本与准确率是永恒的博弈。通过意图识别+动态路由分发处理流，能在不显著拉高响应时间的前提下，最大化复杂推算题的准召率，这是解决高复杂场景落地的标准架构决策。

### 4. 多维度自动化评分体系（Multi-dimensional Evaluation）
* **实现**：`evaluators/` 目录下同时提供了精确字符串匹配（`StringMatchEvaluator`）与大模型裁判（`LLMEvaluator`: LLM-as-a-Judge）。
* **产品思考**：死板的正则匹配很容易将“模型正确但表述不同的答案”误判为 False；而纯人工评估又无法应对上千次的网格化跑分。提供支持“部分正确与语义同构”容忍度的 LLM 裁判，能在高优测试中更精准地反馈模型的情商与逻辑，实现 ROI 的平衡。

---

## 📊 数据驱动的瓶颈与归因分析

本平台支持一键式的评测数据洞察，通过运行 `visualizations/visualize_results.py` 脚本，可将底层 JSON 数据直接凝炼为产品决策报表：

* **大海捞针热力图 (Heatmap)**：通过多维的 Context Length 和 Depth Percent 交叉，直观扫描模型的“记忆黑洞”究竟是在 30k 处还是 80% 深度处。
* **错误归因饼图 (Performance Pie)**：分析大文档下部分失败（Partial/Fail）的占比分布情况，驱动我们去调整：到底是该换模型，还是该优化 Rerank 分数阈值上限。

---

## 🛠️ 系统架构 (System Architecture)

经过深度解耦重构，项目遵循**中台化、组件化**的设计理念，确保了高可维护性与扩展性：

### 核心分层设计 (Decoupled Components)

*   **`core/` (业务中台层)**:
    *   `config.py`: 集中式的命令行参数管理与全局配置中心。
    *   `agent_factory.py`: 采用反射机制实现 Agent 动态加载，支持任意自定义 Agent 的即插即用。
    *   `health_check.py`: 引入**预检机制 (Health Check)**，在长周期实验开始前自动验证 API 连通性、Embedding 指标、Rerank 响应状态，实现快速失败与调试。
    *   `runner.py`: 高层测试抽象层，屏蔽了 `Single` 与 `Multi` 模式下的底层差异。
*   **`agents/` (策略执行层)**: 存放各种不同策略的 LLM 智能体。
    *   `base_agent.py`: 定义了 `ModelProvider` 抽象基类，规范了所有 Agent 的接口实现。
    *   `sync_agent.py`: **同步检索 Agent (SyncRetrievalAgent)**。移除了复杂的异步逻辑，通过增强关键词提取与句子级内容抽取，提供高稳定性的基础检索能力。
    *   `agent_plus.py`: **高级检索 Agent (AdvancedRetrievalAgent)**。集成了混合检索（BM25 + 向量检索）与 Rerank 精排管线，是针对复杂 RAG 场景的生产级实现。
    *   `scenario_agent.py`: **场景感知 Agent (ScenarioAwareAgent)**。引入意图识别逻辑，根据问题类型动态路由不同的特定场景处理流。
    *   `agent_template.py`: 开发者模板。供快速扩展自定义 Agent 的参考实现。
*   **`evaluators/` (质量打分层)**: 定义了多模态（字符串、LLM 评分）的评判逻辑。
*   **`visualizations/` (数据表现层)**: 负责将海量 JSON 测评指标转化为可视化的热力图与报告明细。

### 项目结构

```text
├── agents/             # Agent 策略插件系统 (不同检索/推理模型实现)
├── core/               # 系统核心中台 (配置、工厂、执行引擎与健康预检)
│   ├── agent_factory.py
│   ├── config.py
│   ├── health_check.py
│   ├── runner.py
│   └── test_case_loader.py
├── evaluators/         # 插件式评分引擎 (LLM Judge / String Match)
├── test_cases/         # 测试题库集
├── docs/               # 产品路线图、文档与API范例
├── visualizations/     # 测评数据可视化出图工具
├── run.py              # 统一命令行调度入口 (Minimal Entrypoint) 🚀
└── requirements.txt
```

---

## 🚀 快速开始

### 环境配置 `.env`

请确保你的 `.env` 只保留 ECNU 相关配置。主测模型可手动改成 `ecnu-max`，其余 helper / 评测场景固定使用 `ecnu-plus`：

```env
# ECNU-only runtime configuration
ECNU_API_KEY=sk-xxxx
ECNU_BASE_URL=https://chat.ecnu.edu.cn/open/api/v1
MODEL_NAME=ecnu-max
```

### 运行自动化评测

启动主入口 `run.py` 进行 NIAH 多用例评测（Windows 推荐使用单行命令）：

```bash
# 启动混合检索及深层推断 (AdvancedRetrievalAgent)
uv run python run.py --agent agents.agent_plus:AdvancedRetrievalAgent --test_case_json test_cases/test_cases_all_en.json --test_mode multi --evaluator_type llm --num_tests 1 --save_results True

# 启动场景路由感知策略 (ScenarioAwareAgent)
uv run python run.py --agent agents.scenario_agent:ScenarioAwareAgent --test_case_json test_cases/test_cases_all_en.json --test_mode multi --evaluator_type llm --num_tests 1 --save_results True
```

单独说明一下模式选择：single 模式走的是“纯上下文字符串”，更适合 `SyncRetrievalAgent`；`AdvancedRetrievalAgent` 这种吃 `context_data` 的，得用 multi 模式。

#### 参数详细说明 (Parameter Reference)

| 参数名 | 必填 | 默认值 | 说明 |
| :--- | :---: | :--- | :--- |
| `--agent` | 是 | - | 指定 Agent 的路径，格式为 `module.path:ClassName` |
| `--test_case_json` | 是 | - | 指向测评题库的 JSON 文件路径 |
| `--test_mode` | 否 | `multi` | 测评模式：`multi` (多文档多针) 或 `single` (纯上下文字符串) |
| `--evaluator_type` | 否 | `llm` | 评分器类型：`llm` (语义化 Judge) 或 `string` (全匹配) |
| `--num_tests` | 否 | `5` | 在 `multi` 模式下，针对每个题目循环执行的随机试验次数 |
| `--save_results` | 否 | `False` | 是否将测评数据保存至 `results/` 文件夹 |
| `--results_version` | 否 | `1` | 实验版本号，用于在 `results/` 中区分不同的实验批次 |
| `--skip_model_test` | 否 | `False` | 是否跳过开始前的网络与 API 健康检查 (建议调试时开启) |
| `--haystack_dir` | 否 | `PaulGrahamEssays` | 干扰库 (Haystack) 的文本文件目录 |

### 基础连通性测试 (Baseline Test)

使用精简版题库快速验证 Agent 的基础检索与评分逻辑：

```bash
# 使用同步测试 Agent 运行基础测试 (test_cases.json)
uv run python run.py --agent agents.sync_agent:SyncRetrievalAgent --test_case_json test_cases/test_cases.json --test_mode single --evaluator_type string --num_tests 1
```

_注：项目启动时会优先运行模型健康连通性检查（Health Check），确保网络及 API 接口调通后才会进入耗时的测试流程（调试期间可追加 `--skip_model_test True` 跳过该检查）。_