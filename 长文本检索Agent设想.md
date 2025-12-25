## 1) 总体架构：两阶段检索 + 证据合成回答（RAG on multi-doc）

 **核心思想** ：不要把所有文档直接塞给 LLM（即使 128K 也容易“注意力稀释”），而是做“检索 → 精读 → 汇总回答”的 pipeline。

### Stage A：候选片段召回（Recall）

对每个文档做切分（chunking），对 chunk 做向量化，然后对问题向量检索 top-K chunk。

* Embedding：用 `ecnu-embedding-small`（1024 维）做 chunk 向量
* Chunk 策略（推荐）
  * `chunk_size`: 350–600 tokens
  * `overlap`: 80–120 tokens
  * 保留 `doc_id, chunk_id, char/token offset` 方便回溯上下文
* 召回：topK 先取 30~80（取决于文档数量与长度）

### Stage B：重排序（Precision）

召回 chunk 里会有大量“语义相似但不发现 needle”的噪声，所以加 rerank：

* Rerank：`ecnu-rerank` 对 (question, chunk_text) 做相关性排序，取 topN（比如 8~15）
* 进一步去重：对高度重叠 chunk 做合并/只保留得分高的

### Stage C：证据约束的答案生成（Synthesis）

把 topN chunk（每段都带出处标记）喂给生成模型，让它 **只基于证据回答** ，并尽量输出结构化答案以讨好 LLMEvaluator（语义评分更稳定）。

* 生成模型选择
  * 默认：`ecnu-max`（快、推理能力强、128K、支持 function_call 但你未必需要）
  * 若你发现题目需要更强“跨段推理/多 needle 汇总”：可切换 `ecnu-reasoner`（思维链强，但慢一些，且不支持 function_call）
* 提示词要点
  * 明确“只能使用给定 evidence，缺失则回答 Unknown / Not found”
  * 要求输出：**直接答案 + 简短引用证据片段（可选）**
    * 注意：评分器不一定要求引用，但引用能显著降低幻觉，提高语义评分一致性。

---

## 2) Needle 特性与检索策略上的“对症优化”

“needle 随机插入”意味着它可能：

* 出现在任意主题段落里，和文章语义不强指向
* 可能是 **独特字符串** 、数字、日期、人名、短句
* 可能与问题是**关键词强匹配**而非语义相似

因此建议在 Stage A 加一个“ **混合检索** ”：

### 2.1 混合检索：BM25/关键词 + 向量召回

* 关键词检索对独特 needle（比如带数字/专有名词/格式串）非常有效
* 方案：chunk 同时建立 BM25（或 TF-IDF）倒排索引
* 召回时取：
  * 向量 topK_v
  * BM25 topK_b
  * 合并去重后进入 rerank
    这样能显著提升 needle 命中率，尤其是“语义不相关但关键词相关”的插入信息。

### 2.2 规则型强化（低成本增益）

对 question 做轻量特征分析：

* 若包含引号、冒号、长数字、邮箱/URL样式、ID样式：提高 BM25 权重与 topK_b
* 若是“列举/汇总多个点”：提高 topN 精读段落数量，并让生成阶段更强调“覆盖所有条目”

---

## 3) 上下文组织：让 LLM 更“确定”地抽取 needle

把 evidence 组织成统一格式，减少模型在长段里迷路：

**证据块格式示例**

* `[Doc {i} | Chunk {j} | Score {s}]`
  `...chunk text...`

并在 system/assistant 指令里写清：

* 先做“答案草稿”
* 再逐条核对是否在 evidence 中出现（自检）
* 若证据不足：输出 “Not found in provided documents.”

这会明显降低“看起来很合理但不对”的幻觉，从而提升 LLMEvaluator 分数（语义评分对幻觉通常很敏感）。

---

## 4) 性能与成本：缓存、并发与增量索引

测试会反复调用你的 Agent，多轮不同 needles 插入。你可以在不违规前提下做工程优化：

1. **缓存 embedding**

* 对同一批 `context_data`（每次测试插入 needle 后文本会变，但大部分 chunk 仍相同）
* 你可以对 chunk 文本做 hash（如 sha1），embedding 结果落盘或内存 LRU
* 注意：插入 needle 会改变 chunk 切分边界；为减少抖动，建议“按字符窗口切分”或“滑窗切分”更稳定

2. **异步并发**

* embedding / rerank / LLM 调用都可并发（受 rpm 限制）
* `ecnu-max` rpm 60，embedding/rerank rpm 600
* 典型策略：embedding 批量并发，rerank 批量并发，最终生成一次调用

3. **两段式生成（可选）**

* 先让 LLM 在 evidence 上“抽取候选 facts（结构化）”
* 再让 LLM 依据结构化 facts 输出最终答案
  这对多 needle 汇总题更稳，但会多一次生成调用（注意限流与耗时权衡）。

---

## 5) 结合本作业接口的落地映射

作业要求你实现的最关键接口是：

* `generate_prompt()`：构造 `{context_data, question, ...}` 给 `evaluate_model()`
* `evaluate_model(prompt)`：你在这里做  **chunk → 混合检索 → rerank → 拼 evidence → LLM 生成** ，最后返回字符串答案
* `encode_text_to_tokens / decode_tokens`：用于长度控制与切分（建议用你所调用模型对应 tokenizer；若框架已有工具就复用）

你可以把“索引构建”放在 Agent 初始化或第一次调用时进行（只基于 `prompt['context_data']`，不触碰原始目录文件，符合规则）。

---

## 6) 推荐的默认参数组合（可直接开干）

* Embedding：`ecnu-embedding-small`
* Rerank：`ecnu-rerank`
* Generate：`ecnu-max`（优先），难题 fallback `ecnu-reasoner`
* Chunk：450 tokens / overlap 100
* topK_v=40，topK_b=40，合并后 rerank 取 topN=10
* 生成上下文：topN chunk 全部带 doc/chunk 标头
* 输出：简洁直接，必要时列点；若不确定明确写 Not found/Unknown
