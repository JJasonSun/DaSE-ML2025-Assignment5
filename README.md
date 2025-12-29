# DaSE-ML2025-Assignment5 说明

## 项目简介

本项目实现“Needle in a Haystack”长文本检索评测框架。系统会把若干关键信息（needle）随机插入多篇文章（haystack）中，并调用你实现的 Agent 来回答相关问题，最后使用评测器打分。你可以用该框架本地调试、提交并查看排行榜成绩。

## 🚀 核心检索策略 (Advanced Retrieval Strategy)

本项目实现了基于 **AdvancedRetrievalAgent**（`agents/agent_plus.py`）与 **ScenarioAwareAgent**（`agents/scenario_agent.py`）的高精度多文档检索系统，专门针对作业中的 Needle in a Haystack（大海捞针）场景进行了迭代优化：

### 1. Token 级分块与混合召回 (Token Chunking + Hybrid Retrieval)
* **Token 级滑动窗口分块**：按 token 切分文档（默认 `chunk_size_tokens=500`, `overlap=100`），在保证覆盖率的同时降低噪声。
* **BM25 关键词召回**：使用 `rank_bm25` 对增强关键词进行粗排（数字 / 日期 / 连字符 ID / 关键实体等），提升对“硬针”（ID、编码、数值）的命中率。
* **向量召回 (Dense Retrieval)**：使用 `ecnu-embedding-small` 计算 query / chunk 向量相似度；当 chunk 很多时，先以 BM25 pool 缩小候选范围，兼顾速度与召回。
* **多查询支持**：检索函数支持传入多条查询（Query Expansion 的输出），实现“不同表述/不同关键词”的覆盖。

### 2. 精排：Rerank 把关 (ecnu-rerank)
* **候选合并去重**：BM25 + 向量两路候选合并后去重，并控制候选上限（默认 60）以保持性能。
* **精排重排序**：使用 `ecnu-rerank` 对候选片段进行重排，锚点 query 始终使用原始问题，增强“针/干扰”区分。
* **动态证据拼接**：根据重排分数阈值组装证据片段，并控制证据总 token，避免上下文超限。

### 3. 生成与兜底：两阶段回答 (Thinking + ecnu-reasoner Fallback)
* **严格 JSON 输出**：生成阶段强制 `response_format={"type":"json_object"}`，并在提示词中要求仅输出 `{"answer": "..."}`。
* **第一阶段：主模型深度推理**：对主模型（`MODEL_NAME`，非 `ecnu-` 前缀）开启思考模式并提高 token 预算，鼓励“积极推导与精确运算”。
* **第二阶段：兜底推理**：若答案为空/Unknown，则切换到 `ecnu-reasoner` 再尝试一次；当上下文过长时，会启用 Query Expansion 并扩大精排召回（如 top_n 临时提升），提高最终命中率。
* **重要说明**：当前 API 调用不支持通用的“工具调用（tools/function calling）”，因此提示词不再引导模型调用外部工具，仅要求模型在思考模式下进行严密推导与精确计算。

---

## 🧠 场景感知策略 (ScenarioAwareAgent)

`ScenarioAwareAgent` 会先对问题进行场景分类，再切换到对应的专业提示词：

* 分类模型：`ecnu-max`
* 输出：JSON（`category` + `confidence`），仅当置信度足够高才采纳
* 场景：
   - `encoding`：编码/解码
   - `string_analysis`：字符串计数/位置/子串
   - `computation`：多步/大数计算
   - `date_time`：日期/星期/时长推算

各场景提示词均强调：只使用上下文 + 通用推理/计算能力；信息分散时要坚持检索与推导；最终仅输出 JSON。

---

## ✅ Agent 选择建议与运行示例

### 什么时候用 `ScenarioAwareAgent`

推荐在以下情况优先使用 `ScenarioAwareAgent`（`agents/scenario_agent.py`）：

* **题型分布杂**：同一批测试同时包含编码/字符串/计算/日期等混合问题。
* **希望更强的“题型对齐”**：先做场景分类，再套用对应的专业提示词，通常能减少提示词不匹配导致的失误。
* **更强调推导细节的稳定性**：尤其是 `date_time`（日期→星期/时长）与多步计算题。

注意：场景分类采用置信度门槛（代码中默认 `confidence>=0.7` 才启用场景提示词），低置信度会自动回退到通用提示词。

### 什么时候用 `AdvancedRetrievalAgent`

推荐在以下情况使用 `AdvancedRetrievalAgent`（`agents/agent_plus.py`）：

* **想要更“纯粹”的检索 + 推理基线**：不引入场景分类步骤，链路更短。
* **问题类型不明显或高度开放**：担心错误分类带来提示词偏置时，用通用提示词更稳。
* **只想验证检索链路**：例如只关注 BM25/向量召回/重排/证据拼接效果。

### 对应运行命令示例

使用 `ScenarioAwareAgent`：

```bash
# Bash / macOS / Linux
python run.py \
   --agent agents.scenario_agent:ScenarioAwareAgent \
   --test_case_json test_cases/test_cases_all_en.json \
   --test_mode multi \
   --evaluator_type llm \
   --num_tests 1
```

```powershell
# PowerShell (Windows)
python run.py `
   --agent agents.scenario_agent:ScenarioAwareAgent `
   --test_case_json test_cases/test_cases_all_en.json `
   --test_mode multi `
   --evaluator_type llm `
   --num_tests 1
```

使用 `AdvancedRetrievalAgent`：

```bash
# Bash / macOS / Linux
python run.py \
   --agent agents.agent_plus:AdvancedRetrievalAgent \
   --test_case_json test_cases/test_cases_all_en.json \
   --test_mode multi \
   --evaluator_type llm \
   --num_tests 1
```

```powershell
# PowerShell (Windows)
python run.py `
   --agent agents.agent_plus:AdvancedRetrievalAgent `
   --test_case_json test_cases/test_cases_all_en.json `
   --test_mode multi `
   --evaluator_type llm `
   --num_tests 1
```

## ⚙️ 环境配置 (.env)

请确保你的 `.env` 文件包含以下配置。**注意：我们对 API 获取逻辑进行了重要改动。**

### 1. 基础配置
```env
# --- 基础配置 ---
STUDENT_ID=学号
STUDENT_NAME=姓名
STUDENT_NICKNAME=昵称（排行榜显示）
MAIN_CONTRIBUTOR=ai

# --- 主模型配置 (用于 Agent 生成答案) ---
API_KEY=你的API密钥
BASE_URL=https://api.example.com/v1
MODEL_NAME=glm-4.5-plus

# --- ECNU 专用模型配置 (用于 Embedding 和 Rerank，必须配置) ---
ECNU_API_KEY=你的ECNU密钥
ECNU_BASE_URL=https://chat.ecnu.edu.cn/open/api/v1
```

补充说明：

- `MODEL_NAME`：主回答模型（建议使用你自己的主模型服务）。
- ECNU 侧模型在代码中用于检索增强：`ecnu-embedding-small`（向量检索）、`ecnu-rerank`（精排）、`ecnu-plus`（Query Expansion）、`ecnu-max`（场景分类）、`ecnu-reasoner`（兜底推理）。
- 部分 `ecnu-*` 模型不支持 `thinking` 参数；项目已在调用层自动规避，仅对非 `ecnu-` 主模型开启思考模式配置。

### 2. 评测器配置 (LLMEvaluator)
**⚠️ 重要改动：** `LLMEvaluator` 现在会**直接且优先**从以下环境变量中获取 API 信息，以实现评测环境与 Agent 环境的完全隔离：

```env
# --- 评测模型配置 (可选，独立于 Agent) ---
EVAL_API_KEY=评测密钥
EVAL_BASE_URL=https://open.bigmodel.cn/api/paas/v4/
EVAL_MODEL_NAME=glm-4.5
```

---

## 🛠️ 常用指令

## 目录速览与文件作用

- `run.py`：命令行入口，读取测试用例并调用对应的测试器与评测器。
- `llm_multi_needle_haystack_tester.py`：多文档场景测试器，将多个 needle 插入不同文件并评测。
- `llm_single_needle_haystack_tester.py`：单文档场景测试器，支持不同上下文长度与插入深度的组合。
- `model.py`：`ModelProvider` 抽象基类，定义 Agent 必须实现的接口。
- `agents/`：示例与参考 Agent：
   - `agent_template.py`：最小可运行基线示例。
   - `agent_plus.py`：`AdvancedRetrievalAgent`（混合检索 + Rerank + 两阶段兜底 + Query Expansion）。
   - `scenario_agent.py`：`ScenarioAwareAgent`（先分类场景，再套用专业提示词，底层复用 `AdvancedRetrievalAgent`）。
   - `sync_agent.py`：同步实现（便于在不方便 asyncio 的环境中使用）。
- `evaluators/`：评测器实现，`string_match_evaluator` 为精确匹配，`llm_evaluator` 使用大模型评分（支持 JSON 模式与思考模式禁用）。
- `test_case_loader.py`：读取与校验测试用例 JSON。
- `test_cases/`：示例或评测用的测试用例文件夹。
- `PaulGrahamEssays/`：默认的 haystack 文本集合。
- `submit.py` + `submit_core.*`：提交脚本与核心实现。
- `Assignment5.md`：作业原文说明与评分规则。
- `check_api.py`：快速验证 API 连通性/模型列表的小工具。
- `test_agent_logic.py`：本地快速验证 Agent 行为的轻量脚本（可选）。

## 工作流与运行方式

1. **编写 Agent**

   - 继承 `model.ModelProvider` 并实现四个接口：
     - `evaluate_model`：接收 prompt 字典（包含上下文与问题），返回回答字符串。
     - `generate_prompt`：组装传给 `evaluate_model` 的字典结构。
     - `encode_text_to_tokens` / `decode_tokens`：用于计算、截断与还原文本长度。
   - 可参考 `agents/agent_template.py`（基线随机策略）或 `agents/sync_agent.py`（同步、多策略检索）进行二次开发。
   - `ModelProvider` 是纯抽象基类，必须在自定义 Agent 中完整实现 `evaluate_model`、`generate_prompt`、`encode_text_to_tokens` 以及 `decode_tokens` 等接口，否则测试流程会因 `NotImplementedError` 或类型不匹配而失败。`agents/agent_template.py` 与 `agents/sync_agent.py` 展示了可运行的实现。
2. **准备测试用例**

   - 将自定义测试集写入 JSON，字段至少包含 `needle`、`question`、`ground_truth`，可选 `id`。
   - 通过 `test_case_loader.py` 自动判断单 needle 或多 needle（`needle` 为列表且长度>1 时视为多 needle）。
3. **本地运行评测**

   ```bash
   # Bash / macOS / Linux
   python run.py \
      --agent agents.agent_plus:AdvancedRetrievalAgent \
      --test_case_json test_cases/test_cases_all_en.json \
      --test_mode multi \
      --evaluator_type llm \
      --num_tests 1
   ```

   ```powershell
   # PowerShell (Windows)
   python run.py `
      --agent agents.agent_plus:AdvancedRetrievalAgent `
      --test_case_json test_cases/test_cases_all_en.json `
      --test_mode multi `
      --evaluator_type llm `
      --num_tests 1
   ```

   > 💡 **提示**：示例中默认使用 `--evaluator_type llm`，因为这是正式评测的标准。如果你在开发初期需要快速验证逻辑、节省 API 消耗，可以使用 `--evaluator_type string` 进行精确匹配测试。
   >

   - `--haystack_dir` 可指定其他文本目录。
   - 单文档模式可调节 `context_lengths_*` 与 `document_depth_percent_*` 控制长度与插入位置。

选择建议：若需快速批量验证用 `--evaluator_type string`；若想评估语义等价或更接近人工判断可选 `--evaluator_type llm`。不论使用哪种评测器，整个测试流程都需要调用 API，所以 `string` 也会有 API 成本。正式验证与比赛排序阶段应优先走 `--evaluator_type llm`。

- 运行结束后，若 `save_results=True`，结果写入 `results/`，上下文（可选）写入 `contexts/`。

### 参数说明：`test_mode` 与 `evaluator_type`

- `test_mode`（测试模式）

  - 可选：`multi` 或 `single`。
  - `multi`：多文档/多 needle 场景，使用 `LLMMultiNeedleHaystackTester`。适合评估模型在不同文档与不同插入位置上的鲁棒性，会对每个用例进行多次随机试验（受 `--num_tests` 控制），但 API 调用与运行时间较多。
  - `single`：单文档/单 needle 场景，使用 `LLMSingleNeedleHaystackTester`。适合做上下文长度（`context_lengths_*`）和插入深度（`document_depth_percent_*`）的网格扫描与可重复实验，便于分析模型对上下文规模的敏感性。
- `evaluator_type`（评测器类型）

  - 可选：`string` 或 `llm`。
  - `string`：使用 `StringMatchEvaluator` 进行精确字符串匹配（可忽略大小写/空白）。虽然比 `llm` 模式更直接，但评测本质仍需调用一次模型 API，主要省在评分逻辑复杂度与第二次调用上。适合 exact-match 场景与本地调试。
  - `llm`：使用 `LLMEvaluator` 通过另一个大模型对输出与 ground truth 做语义评分。能识别同义改写与部分正确，默认用于正式评测（与 `Assignment5.md` 中的默认一致），但会增加延迟与 API 成本，评分存在一定随机性。

选择建议：若需快速批量验证用 `--evaluator_type string`；若想评估语义等价或更接近人工判断可选 `--evaluator_type llm`。若评估多文档检索稳健性选 `--test_mode multi`，做可控变量扫描时选 `--test_mode single`。

## 高级检索 Agent 特性 (`AdvancedRetrievalAgent`)

本项目默认推荐使用 `AdvancedRetrievalAgent`（`agents/agent_plus.py`），它具备以下核心特性：

1. **混合检索 + 精排**：
   - **粗召回**：BM25（rank_bm25）+ Dense Embedding（`ecnu-embedding-small`）两路召回，覆盖“硬针 + 软线索”。
   - **精排把关**：`ecnu-rerank` 对候选片段重排序，强化 needle 定位。
2. **两阶段回答兜底**：
   - 首次回答失败（Unknown/空）时，自动切换到 `ecnu-reasoner` 进行二次推理。
   - 当上下文过长时，二次推理会触发 Query Expansion 并扩大精排召回，提高命中率。
3. **模型路由与兼容性**：
   - 检索增强（Embedding/Rerank/Expansion/Classification/Reasoner）走 ECNU 侧配置。
   - 主回答模型使用 `MODEL_NAME`；仅对非 `ecnu-` 前缀模型注入思考模式配置，避免接口不兼容。
4. **输出约束**：
   - 严格 JSON 输出（`{"answer":"..."}`）；若确无答案，仅输出 `{"answer":"Unknown"}`，不附加解释。

## 提交作业

   使用 `submit.py` 脚本提交你的 Agent 实现。请确保提供完整的模块路径：

   ```bash
   # 示例：提交 agents 目录下的 sync_agent.py 中的 SyncRetrievalAgent 类
   python submit.py --agent agents.sync_agent:SyncRetrievalAgent
   ```

   提交成功后，你的 Agent 将在服务器端进行正式评测。

## 查看与调试

   - 控制台会输出每个用例的得分、均值、最好/最差等统计。
   - 开启 `--print_ongoing_status` 可实时查看插入 needle、生成 prompt 与模型响应摘要。
5. **提交评测**

   ```bash
   python submit.py --agent your_module:YourAgentClass
   ```

   - 优先从命令行读取 `--api_key` / `--base_url`，否则回退到 `.env`。
   - 提交脚本会校验 `STUDENT_ID`、`STUDENT_NAME`、`STUDENT_NICKNAME`、`MAIN_CONTRIBUTOR` 等信息。

## 推荐工作流（开发 → 评测 → 提交）及示例命令

下面给出从本地开发到最终提交的推荐流程与示例命令：

1) 快速验证 API 与环境（轻量）

```bash
python check_api.py
```

2) 小规模验证 / smoke test（快速迭代，低成本）

```bash
# Bash / macOS / Linux
python run.py \
   --agent agents.sync_agent:SyncRetrievalAgent \
   --test_case_json test_cases/test_cases_all_en.json \
   --test_mode single \
   --evaluator_type llm \
   --num_tests 1
```

```powershell
# PowerShell（Windows）
python run.py `
   --agent agents.sync_agent:SyncRetrievalAgent `
   --test_case_json test_cases/test_cases_all_en.json `
   --test_mode single `
   --evaluator_type llm `
   --num_tests 1
```

> 💡 **提示**：此处使用 `--evaluator_type llm` 进行语义评测。在开发初期，若需快速验证逻辑，可临时切换为 `string` 模式。

3) 完整评测并保存结果（生成 `results/`，用于提交与统计）

```bash
# Bash / macOS / Linux
python run.py \
   --agent agents.agent_plus:AdvancedRetrievalAgent \
   --test_case_json test_cases/test_cases_all_en.json \
   --test_mode multi \
   --evaluator_type llm \
   --num_tests 1 \
   --save_results True
```

```powershell
# PowerShell (Windows)
python run.py `
   --agent agents.agent_plus:AdvancedRetrievalAgent `
   --test_case_json test_cases/test_cases_all_en.json `
   --test_mode multi `
   --evaluator_type llm `
   --num_tests 1 `
   --save_results True
```

4) 提交（确认 `results/` 与 `.env` 中的学生信息正确后）

```bash
python submit.py --agent agents.agent_plus:AdvancedRetrievalAgent
```

要点提示：

- 使用小规模验证（step 2）可以快速定位逻辑/prompt 问题，避免一次耗尽配额。
- 若选择 `--evaluator_type llm`，评估步骤会额外产生 API 调用与延迟，请预留配额并酌情减少 `num_tests`。
- 确保 `.env` 中 `API_KEY`、`BASE_URL`、学生信息字段已正确填写（`check_api.py` 可快速验证连通性）。

## 结果与输出

- `results/*.json`：保存模型回答、得分、运行时间、needle 信息等。
- `contexts/*.txt`（可选）：保存插入 needle 后的上下文，便于复现。
- 控制台日志：汇总各测试用例及全局统计。

## 开发建议

- 优化检索策略：结合关键词召回、向量检索或 RAG，定位 needle 所在段落。
- 控制上下文长度：利用 `encode_text_to_tokens` / `decode_tokens` 精确截断，确保请求不超限。
- 调试评测器：`string` 模式可快速验证逻辑，`llm` 模式能提供更细粒度语义评分。

如需更多背景与评分细节，请参阅 `Assignment5.md`。祝你取得好成绩！
