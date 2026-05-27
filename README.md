# LLM 长文本与复杂推理自动化测评平台 (NIAH Evaluation Framework)

## 📌 项目定位

面向大语言模型的**自动化、可插拔的长文本检索与复杂推理评测平台**。

在真实的 AI 产品落地中，大模型的长文本能力（Context Window）往往伴随着"中间迷失"与"幻觉"。本项目基于强化版的"大海捞针 (Needle in a Haystack)"范式，构建了一套完整的自动化评测管线——从测试用例设计、Agent 策略执行、自动化评分到数据可视化归因，帮助团队深挖模型能力边界、打磨 RAG 管线，并为模型选型提供数据支撑。

---

## 💡 核心设计

### 1. 插件式 Agent 架构

`agents/base_agent.py` 定义了统一接口，所有 Agent 可通过 `--agent` 参数即插即用。新增一个 Agent 只需一个文件，无需改动评测引擎。

### 2. 混合检索管线

`AdvancedRetrievalAgent` 内置 **BM25 粗排 → Dense Embedding 向量检索 → Rerank 精排** 三路召回。纯向量对同义词强但对 ID、数字等硬规则易漏召回，混合检索能更真实地反映 RAG 生产环境的表现。

### 3. 场景感知路由

`ScenarioAwareAgent` 先用小模型判别问题类型（日期/计算/编码/字符串），再动态挂载对应场景的专业 Prompt，最大化复杂推理的准确率。

### 4. 双轨评分体系

- **LLM-as-a-Judge**（`ecnu-plus`）：语义评分 0–10，容忍表述差异，适合深度分析
- **String Match**：精确匹配 0/1，适合快速测试

---

## 🧪 两种测试模式

平台提供两种测试模式，考察模型的不同能力维度：

### `multi` 模式 — 多文档检索（推荐）

```
PaulGrahamEssays/           测试用例
├── essay1.txt              needle_1 ──→ 随机插入 essay3.txt (深度 37%)
├── essay2.txt              needle_2 ──→ 随机插入 essay7.txt (深度 82%)
├── essay3.txt  [needle_1]  ...
├── essay4.txt              Agent 需要从所有文件中检索出全部 needle 并回答
├── essay5.txt
├── essay6.txt
├── essay7.txt  [needle_2]
└── ...
```

**做法**：将测试用例中的每条 needle 随机插入到 haystack 的不同文件、不同深度位置。Agent 接收所有文件内容（结构化的 `context_data`），需要跨文件检索并推理。同一用例重复 `num_tests` 次（默认 3），每次插入位置随机变化。

**考察**：多文档场景下的跨文件检索能力、信息聚合与推理能力。更贴近真实 RAG 生产环境。

**适用 Agent**：`AdvancedRetrievalAgent`（有混合检索管线）、`ScenarioAwareAgent`、`SyncRetrievalAgent`

### `single` 模式 — 上下文长度 × 深度扫描

```
context_length:  1K ──────────────────────── 100K  (10 个梯度)
depth_percent:   0% ──────────────────────── 100%  (10 个梯度)

                 ┌─────────────────────────────────┐
  1K × 0%        │ [needle] 无关文本...             │
  1K × 10%       │ 无关文本 [needle] ...            │
  ...            │ ...                              │
  100K × 100%    │ 无关文本...... [needle]           │
                 └─────────────────────────────────┘
                 共 100 组测试，自动生成热力图
```

**做法**：将所有 haystack 文本拼接成一个长字符串，截断到目标 `context_length`，再将 needle 插入到指定 `depth_percent` 位置。遍历所有 (长度, 深度) 组合，形成网格化测试。

**考察**：模型在不同上下文长度下的"记忆"能力、对"中间迷失"现象的抵抗力。输出热力图可直观看到模型在哪个长度/深度区间开始丢失信息。

**适用 Agent**：`SyncRetrievalAgent`（接收纯字符串 `context`）、`AdvancedRetrievalAgent`

### 模式对比

| 维度        | `multi`                               | `single`                       |
| :---------- | :-------------------------------------- | :------------------------------- |
| 上下文形式  | 多个独立文件（结构化 `context_data`） | 单个长字符串（`context`）      |
| Needle 数量 | 多条，分散在不同文件                    | 单条，固定位置                   |
| 测试维度    | 每个用例重复 N 次，位置随机             | context_length × depth 网格扫描 |
| 用例数量    | 从题库抽样（默认 20 条）                | 单条用例 × 100 组参数           |
| 输出可视化  | 柱状图 + 饼图 + Bad Case 归因           | 热力图 (长度 vs 深度)            |
| 核心场景    | RAG 多文档检索、信息聚合                | 长上下文记忆、位置敏感性         |
| 推荐用途    | 生产级评测、Agent 能力对比              | 模型能力边界探测、学术研究       |

---

## 📐 评测方法论

### 测试集设计

测试集 `test_cases_all_en.json` 包含 **1600 条**测试用例，按场景类型均衡划分为 4 类，每类 400 条：

| 类型                | 考察维度             | 示例                                                   |
| :------------------ | :------------------- | :----------------------------------------------------- |
| `date_time`       | 日期推算与时间差计算 | "项目 A 的部署日期是 2042-10-5，距 11-22 还有多少天？" |
| `computation`     | 数值计算与逻辑推理   | "根据文中提到的三个增长率，求复合增长后的最终值"       |
| `encoding`        | 编码识别与解码       | "文中的 Base64 编码字符串解码后是什么？"               |
| `string_analysis` | 字符串操作与文本分析 | "文中提到的密码字符串中，第 N 个字符是什么？"          |

每次运行时，平台从 1600 条中按类型均衡抽样（默认 20 条，每类 5 条），确保评测结果能全面反映模型在不同推理维度上的能力分布。

### Haystack 干扰机制

每条测试用例的 needle（关键信息）会被随机插入到 Paul Graham Essays 语料库的不同深度位置，模拟真实场景中关键信息被大量无关文本淹没的情况。在 multi 模式下，同一用例重复 `num_tests` 次（默认 3 次），每次插入位置随机变化，以消除位置偏差。

---

## 📊 数据驱动的瓶颈与归因分析

测试完成后自动生成 HTML 评测报告（输出到 `results/evaluation_report.html`）：

| 模块                    | 模式   | 说明                                                              |
| :---------------------- | :----- | :---------------------------------------------------------------- |
| **HTML 热力图**   | single | 用表格色块展示 context_length × depth_percent 的得分             |
| **得分分布看板**  | multi  | 用 HTML 条形图展示逐题得分与 Good/Partial/Fail 分布               |
| **Bad Case 归因** | both   | 按问题类型拆解得分，定位薄弱维度（哪类问题模型最弱）              |
| **AI 分析报告**   | both   | `deepseek-v4-pro` 开启 thinking，基于结构化指标生成产品评测解读 |

---

## 🛠️ 系统架构

```text
├── agents/                  # Agent 策略插件
│   ├── base_agent.py        #   ModelProvider 抽象基类
│   ├── sync_agent.py        #   SyncRetrievalAgent — 关键词检索
│   ├── agent_plus.py        #   AdvancedRetrievalAgent — BM25+向量+Rerank
│   ├── scenario_agent.py    #   ScenarioAwareAgent — 意图识别+动态路由
│   └── agent_template.py    #   ExampleAgent — 开发者模板
├── core/                    # 系统核心
│   ├── config.py            #   命令行参数配置
│   ├── agent_factory.py     #   Agent 动态加载
│   ├── runner.py            #   测试执行引擎
│   ├── test_case_loader.py  #   测试用例加载与均衡抽样
│   ├── health_check.py      #   API 连通性预检
│   ├── evaluation_data.py   #   生成最近一次结构化评测数据
│   ├── reporter_factory.py  #   报告插件动态加载
│   └── report.py            #   兼容默认 HTML 报告入口
├── reporters/               # 报告生成插件
│   ├── base_reporter.py
│   └── deepseek_html_reporter.py
├── evaluators/              # 评分引擎
│   ├── evaluator.py         #   Evaluator 基类
│   ├── llm_evaluator.py     #   LLM-as-a-Judge (0-10 语义评分)
│   └── string_match_evaluator.py  #   精确匹配 (0/1)
├── test_cases/              # 测试题库
├── generate_report.py       # 基于本地结构化数据单独重生成 HTML 报告
├── run.py                   # 统一命令行入口
└── requirements.txt
```

### Agent 继承关系

```
ModelProvider (base_agent.py)
├── ExampleAgent (agent_template.py)      — 随机 baseline
├── SyncRetrievalAgent (sync_agent.py)    — 关键词检索 + 句子级抽取
└── AdvancedRetrievalAgent (agent_plus.py) — BM25 + 向量 + Rerank 混合检索
    └── ScenarioAwareAgent (scenario_agent.py) — 意图识别 + 动态 Prompt 路由
```

---

## 🚀 快速开始

### 1. 环境配置

创建 `.env` 文件：

```env
ECNU_API_KEY=sk-xxxx
ECNU_BASE_URL=https://chat.ecnu.edu.cn/open/api/v1
MODEL_NAME=ecnu-max       # 主测模型，可改为 ecnu-plus

DS_API_KEY=sk-xxxx        # HTML 报告里的 AI 数据分析模型
DS_BASE_URL=https://api.deepseek.com
DS_MODEL_NAME=deepseek-v4-pro
```

平台评测链路使用 ECNU 提供的模型服务，HTML 报告的数据分析文本使用 DeepSeek API：

| 模型                     | 用途                                   | 底层模型           | 上下文 |
| :----------------------- | :------------------------------------- | :----------------- | :----- |
| `ecnu-max`             | 主测模型（可通过 `MODEL_NAME` 切换） | DeepSeek-V4-Flash  | 1M     |
| `ecnu-plus`            | 评分器 / 场景分类 / Helper             | Qwen3.6-27B        | 256K   |
| `ecnu-embedding-small` | Dense 向量检索（1024 维）              | bge-m3             | 8K     |
| `ecnu-rerank`          | Rerank 精排                            | bge-reranker-v2-m3 | 8K     |
| `deepseek-v4-pro`      | HTML 报告的 AI 产品评测分析            | DeepSeek V4 Pro    | 1M     |

安装依赖：

```bash
uv pip install -r requirements.txt
```

### 2. 运行评测

```bash
# 推荐：multi 模式 + AdvancedRetrievalAgent（默认抽样 20 条，每条跑 3 次）
uv run python run.py --agent agents.agent_plus:AdvancedRetrievalAgent

# 场景感知路由 Agent
uv run python run.py --agent agents.scenario_agent:ScenarioAwareAgent

# 抽样 50 条测试用例
uv run python run.py --agent agents.agent_plus:AdvancedRetrievalAgent --num_samples 50

# single 模式：扫描上下文长度 × 深度的热力图
uv run python run.py --agent agents.sync_agent:SyncRetrievalAgent --test_mode single

# 快速冒烟测试（跳过健康检查，5 条用例，精确匹配）
uv run python run.py --agent agents.sync_agent:SyncRetrievalAgent --num_samples 5 --test_mode single --evaluator_type string --num_tests 1 --skip_model_test True

# 开启思考模式（Extended Thinking），提升复杂推理准确率
uv run python run.py --agent agents.agent_plus:AdvancedRetrievalAgent --enable_thinking True

# 关闭 HTML 评测报告生成
uv run python run.py --agent agents.agent_plus:AdvancedRetrievalAgent --generate_report False

# 基于最近一次结构化数据单独重生成 HTML 报告
uv run python generate_report.py --input results/latest_evaluation_data.json --output results/evaluation_report.html
```

### 3. 参数说明

| 参数                  | 默认值                                | 说明                                                          |
| :-------------------- | :------------------------------------ | :------------------------------------------------------------ |
| `--agent`           | *(必填)*                            | Agent 路径，格式 `module.path:ClassName`                    |
| `--test_case_json`  | `test_cases/test_cases_all_en.json` | 测试题库 JSON 路径                                            |
| `--num_samples`     | `20`                                | 从题库中按类型均衡抽样的用例数                                |
| `--test_mode`       | `multi`                             | 测试模式：`multi`（多文档）或 `single`（网格扫描）        |
| `--evaluator_type`  | `llm`                               | 评分器：`llm`（语义评分）或 `string`（精确匹配）          |
| `--num_tests`       | `3`                                 | multi 模式下每个用例的重复试验次数                            |
| `--enable_thinking` | `False`                             | 开启模型思考模式（Extended Thinking），提升推理质量但增加延迟 |
| `--skip_model_test` | `False`                             | 跳过 API 健康检查（调试时使用）                               |
| `--haystack_dir`    | `PaulGrahamEssays`                  | 干扰库文本文件目录                                            |
| `--generate_report` | `True`                              | 测试完成后自动生成 HTML 评测报告                              |
| `--reporter`        | `reporters.deepseek_html_reporter:DeepSeekHtmlReporter` | 报告生成插件路径，格式同 Agent 插件                            |
| `--report_data_path` | `results/latest_evaluation_data.json` | 最近一次结构化评测数据，运行时覆盖旧文件，只保留一份              |
| `--report_output_path` | `results/evaluation_report.html` | HTML 报告输出路径                                             |

---

## 🔌 扩展：添加自定义 Agent

1. 在 `agents/` 下创建 `your_agent.py`
2. 继承 `ModelProvider`（或 `AdvancedRetrievalAgent` 以复用混合检索）
3. 实现 `async evaluate_model(prompt: Dict) -> str`
4. 运行时指定 `--agent agents.your_agent:YourAgent`

详见 `agents/agent_template.py` 的最小参考实现。
