# DaSE-ML2025-Assignment5 说明

## 项目简介

本项目实现“Needle in a Haystack”长文本检索评测框架。系统会把若干关键信息（needle）随机插入多篇文章（haystack）中，并调用你实现的 Agent 来回答相关问题，最后使用评测器打分。你可以用该框架本地调试、提交并查看排行榜成绩。

## 环境与依赖

1. 安装依赖：`pip install -r requirements.txt`
2. 在项目根目录创建 `.env`，填写：
   ```env
   API_KEY=你的API密钥
   BASE_URL=https://chat.ecnu.edu.cn/open/api/v1  # 或你的兼容 OpenAI 的服务地址
   STUDENT_ID=学号
   STUDENT_NAME=姓名
   STUDENT_NICKNAME=昵称（排行榜显示）
   MAIN_CONTRIBUTOR=human   # 或 ai
   TEST_MODEL=用于 check_api 的模型名称（可选）
   ```
3. 可运行 `python check_api.py` 验证 API_KEY 与 BASE_URL 是否可用；若设置了 TEST_MODEL 会顺便做一次推理测试。

## 目录速览与文件作用

- `run.py`：命令行入口，读取测试用例并调用对应的测试器与评测器。
- `llm_multi_needle_haystack_tester.py`：多文档场景测试器，将多个 needle 插入不同文件并评测。
- `llm_single_needle_haystack_tester.py`：单文档场景测试器，支持不同上下文长度与插入深度的组合。
- `model.py`：`ModelProvider` 抽象基类，定义 Agent 必须实现的接口。
- `agents/`：示例与参考 Agent，`agent_template.py`（异步示例），`sync_agent.py`（同步示例）。
- `evaluators/`：评测器实现，`string_match_evaluator` 为精确匹配，`llm_evaluator` 使用大模型评分。
- `test_case_loader.py`：读取与校验测试用例 JSON。
- `test_cases/`：示例或评测用的测试用例文件夹。
- `PaulGrahamEssays/`：默认的 haystack 文本集合。
- `submit.py` + `submit_core.*`：提交脚本与核心实现。
- `Assignment5.md`：作业原文说明与评分规则。
- `check_api.py`：快速验证 API 连通性的小工具。

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
   # Bash / macOS / Linux（使用反斜杠作为续行符）
   python run.py \
      --agent agents.sync_agent:SyncRetrievalAgent \
      --test_case_json test_cases/test_cases_all_en.json \
      --test_mode multi \            # 可选 single/multi
      --evaluator_type string \      # 可选 string/llm
      --num_tests 3                  # 多文档模式下重复次数
   ```

   ```powershell
   # PowerShell（Windows）——使用反引号 ` 作为续行符
   python run.py `
      --agent agents.sync_agent:SyncRetrievalAgent `
      --test_case_json test_cases/test_cases_all_en.json `
      --test_mode multi `
      --evaluator_type string `
      --num_tests 3
   ```

   > ⚠️ 虽然示例中用了 `--evaluator_type string` 来快速获得确定性结果，但别忘了根据 `Assignment5.md`，**正式宜优先使用 `LLMEvaluator`（即 `--evaluator_type llm`）作为默认值**，因为整个评测流程最终是通过 LLM 评分的。`string` 模式只是为了快速验证、调试 forward，而本质上也会调用一次模型（更简洁的 API 结果比对），因此并没有“零成本”。
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

4. **查看与调试**

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
   --evaluator_type string \
   --num_tests 1
```

```powershell
# PowerShell（Windows）
python run.py `
   --agent agents.sync_agent:SyncRetrievalAgent `
   --test_case_json test_cases/test_cases_all_en.json `
   --test_mode single `
   --evaluator_type string `
   --num_tests 1
```

> ⚠️ sample 中为了让 smoke test 更快更可控所以用 `--evaluator_type string`，但正式的本地验证应**默认走 `--evaluator_type llm`**（与 `Assignment5.md` 中默认一致），`string` 模式只是为了低成本调试；无论哪种模式，整个流程都会发起 LLM API 调用。

3) 完整评测并保存结果（生成 `results/`，用于提交与统计）

```bash
# Bash / macOS / Linux
python run.py \
   --agent agents.sync_agent:SyncRetrievalAgent \
   --test_case_json test_cases/test_cases_all_en.json \
   --test_mode multi \
   --evaluator_type string \
   --num_tests 5 \
   --save_results True
```

```powershell
# PowerShell（Windows）
python run.py `
   --agent agents.sync_agent:SyncRetrievalAgent `
   --test_case_json test_cases/test_cases_all_en.json `
   --test_mode multi `
   --evaluator_type string `
   --num_tests 5 `
   --save_results True
```

4) 提交（确认 `results/` 与 `.env` 中的学生信息正确后）

```bash
python submit.py --agent your_module:YourAgentClass

# python submit.py --agent agents.sync_agent:SyncRetrievalAgent
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
