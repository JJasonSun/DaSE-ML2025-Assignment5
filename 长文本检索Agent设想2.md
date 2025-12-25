这是一个非常经典的 **RAG (Retrieval-Augmented Generation, 检索增强生成)** 任务，特别是针对“大海捞针”（Needle in a Haystack）场景，考验的是在长上下文中精准定位微小信息片段的能力。

结合 **作业要求** 和  **可用模型资源** ，我为你设计了一个高分实现方案。这个方案结合了**混合检索（Hybrid Search）**和**重排序（Reranking）**策略，旨在最大化检索的召回率（Recall）和准确率（Precision）。

---

### 核心设计思路：两阶段检索 + 混合排序

由于“针”（Needle）通常是一个具体的实体、数字或短语，单纯的向量检索（Dense Retrieval）可能会因为语义泛化而丢失精确匹配；而单纯的关键词检索（BM25）又缺乏语义理解。

**最佳策略是：**

1. **切分（Chunking）** ：将长文本切分成小块。
2. **初筛（Retrieval）** ：同时使用 **BM25（关键词匹配）** 和  **向量检索（语义匹配）** ，获取较多候选片段（比如 Top-50）。
3. **精排（Reranking）** ：使用 `ecnu-rerank` 模型对候选片段进行语义打分，选出最相关的 Top-N（比如 Top-5）。
4. **生成（Generation）** ：将 Top-N 片段作为 Context，喂给 `ecnu-max` 生成答案。

---

### 详细实现步骤

#### 1. 模型选择 (Model Selection)

根据 `ECNU模型说明文档.md`，我们选用以下组合：

* **生成模型 (Generator)** : `ecnu-max` (DeepSeek V3.2)
* *理由* ：通用推理最强，拥有 128K 上下文，处理复杂指令效果最好。
* **向量模型 (Embedding)** : `ecnu-embedding-small`
* *理由* ：用于语义检索，必须使用。
* **重排模型 (Reranker)** : `ecnu-rerank`
* *理由* ：**这是提分的关键。** 向量检索只能看大概，Rerank 能精准判断“这段话是否回答了这个问题”。

#### 2. 代码架构设计 (Code Structure)

你需要修改 `agent.py` (假设文件名) 并继承 `ModelProvider`。

##### Step A: 文本预处理与切分 (Preprocessing & Chunking)

由于 `evaluate_model` 每次接收的 context 可能不同（动态插入 needle），你需要在线处理。

* **策略** ：滑动窗口切分 (Sliding Window)。
* **参数建议** ：Chunk Size = 500 tokens, Overlap = 100 tokens。
* **库** ：可以使用 `langchain` 的 `RecursiveCharacterTextSplitter` 或简单的字符串切片。

##### Step B: 混合检索 (Hybrid Retrieval) - *创新点*

为了防止漏掉“针”，我们不能只信赖向量。

1. **BM25 检索** ：

* 使用 `rank_bm25` 库（需安装）。
* 将所有 Chunk 进行分词并建立 BM25 索引。
* 检索出 Top-K (例如 30 个)。
* *优势* ：对人名、日期、特定术语（Needle 常有的特征）极其敏感。

1. **向量检索 (Vector Search)** ：

* 调用 `ecnu-embedding-small` API 将所有 Chunk 转为向量。
* 计算 Query 向量与 Chunk 向量的余弦相似度。
* 检索出 Top-K (例如 30 个)。
* *注意* ：为了速度，可以用 `numpy` 做矩阵运算，或者使用 `faiss`（如果允许安装）。

1. **合并候选集** ：将 BM25 和 向量检索的结果取并集（去重），得到候选池（Pool）。

##### Step C: 重排序 (Reranking) - *核心提分点*

将 Step B 得到的候选池（可能约 40-60 个片段）发送给 `ecnu-rerank` 接口。

* **输入** ：Query + List[Chunks]
* **输出** ：每个 Chunk 的相关性得分。
* **筛选** ：取得分最高的 Top-5 或 Top-10（取决于总 Token 数限制）。

##### Step D: 生成回答 (Generation)

构建最终 Prompt：

**Plaintext**

```
请根据以下参考文档回答问题。如果文档中没有相关信息，请说明无法回答。

参考文档：
---
[片段1的内容]
---
[片段2的内容]
---
...

问题：{question}
```

调用 `ecnu-max` 生成答案。

---

### 代码逻辑参考 (Pseudo-Code)

以下是一个大致的 Python 类结构建议，你需要填充具体的 API 调用逻辑。

**Python**

```
import os
import requests
import json
import numpy as np
from typing import List, Dict, Optional
from model import ModelProvider # 假设这是基类
# 推荐安装 rank_bm25 用于关键词检索
# pip install rank_bm25
from rank_bm25 import BM25Okapi

class MyAgent(ModelProvider):
    def __init__(self, **kwargs):
        self.api_key = os.getenv("API_KEY")
        self.base_url = os.getenv("BASE_URL")
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    # --- 辅助函数：调用 Embedding API ---
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        # 实现调用 ecnu-embedding-small 的逻辑
        # 注意：如果有大量 chunk，需要分批调用 (batch processing)
        pass

    # --- 辅助函数：调用 Rerank API ---
    def rerank(self, query: str, docs: List[str]) -> List[Dict]:
        # 实现调用 ecnu-rerank 的逻辑
        # 返回排序后的文档列表和分数
        pass

    # --- 辅助函数：调用 Chat API ---
    def chat(self, messages: List[Dict]) -> str:
        # 实现调用 ecnu-max 的逻辑
        pass

    def encode_text_to_tokens(self, text: str) -> List[int]:
        # 简单估算或使用 tiktoken (如果允许)
        # 1中文约等于0.7 token，1英文单词约等于1.3 token
        # 简单实现：return [1] * len(text) 
        pass

    def decode_tokens(self, tokens: List[int], context_length: Optional[int] = None) -> str:
        pass

    def split_text(self, text: str, chunk_size=500, overlap=100) -> List[str]:
        # 实现简单的滑动窗口切分
        return chunks

    def evaluate_model(self, prompt: Dict) -> str:
        context_data = prompt['context_data'] # 这是一个文件名到内容的列表或字典
        question = prompt['question']

        # 1. 预处理：合并所有文本并切分
        all_text = "\n".join(context_data) # 或者遍历每个文件单独切分
        chunks = self.split_text(all_text)

        # --- 检索阶段 (创新点：混合检索) ---
      
        # A. BM25 检索
        tokenized_corpus = [chunk.split() for chunk in chunks] # 简单按空格分词，中文需jieba
        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_query = question.split()
        # 获取 BM25 top_n
        bm25_top_chunks = bm25.get_top_n(tokenized_query, chunks, n=20)

        # B. 向量检索
        # 考虑到性能，如果 chunks 太多，可以跳过向量检索，只用 BM25 + Rerank
        # 但为了高分，建议加上向量检索。
        # query_emb = self.get_embeddings([question])[0]
        # doc_embs = self.get_embeddings(chunks)
        # vector_top_chunks = ... # 计算余弦相似度取 Top 20

        # C. 合并候选集
        candidates = list(set(bm25_top_chunks)) # + vector_top_chunks

        # --- 重排序阶段 (关键步骤) ---
        reranked_results = self.rerank(question, candidates)
      
        # 取 Top 5-10 最相关的片段
        best_chunks = [res['document'] for res in reranked_results[:8]]
      
        # --- 生成阶段 ---
        context_str = "\n\n".join(best_chunks)
      
        system_prompt = "你是一个不仅智能而且诚实的助手。请根据提供的上下文片段回答用户的问题。"
        user_prompt = f"上下文：\n{context_str}\n\n问题：{question}\n请根据上下文提取关键信息回答。"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        return self.chat(messages)

    def generate_prompt(self, **kwargs) -> Dict:
        # 这个函数由测试框架调用，将数据打包传递给 evaluate_model
        return {
            "context_data": kwargs.get("context_data", []),
            "question": kwargs.get("question", "")
        }
```

---

### 提分技巧与注意事项

1. **处理 Token Limit** ：

* 虽然 `ecnu-max` 支持 128k，但 Rerank 模型通常只有 8k 或 16k 限制。所以**必须**先检索出少量候选集，再送给 Rerank，最后送给 LLM。不要把所有文本直接丢给 Rerank。

1. **鲁棒性 (Robustness)** ：

* Needle 可能是数字、ID 或无意义字符串。BM25 在这方面通常比 Embedding 表现好。**一定要保留 BM25。**
* 如果 `context_data` 特别小（例如总共才 5k token），可以直接跳过检索，全量塞给 `ecnu-max`，这样准确率最高。你可以在代码里加个判断：`if total_len < 30000: use_full_context() else: use_rag()`。这也算一种根据场景自适应的 **创新性** 。

1. **依赖安装** ：

* 记得在 `requirements.txt` 中添加你用到的库（如 `rank_bm25`, `requests`, `numpy` 等）。

1. **API 调用频率** ：

* 注意文档中的 RPM (每分钟请求数)。`embedding` 和 `rerank` 都是 600 RPM，比较充裕。`ecnu-max` 是 60 RPM。如果你的切片很多，计算 Embedding 时一定要使用 Batch（批量）请求，不要一个一个循环调，否则会超时或被限流。

1. **创新性描述** ：

* 如果需要提交报告或在代码注释中说明，强调你使用了 **"Hybrid Retrieval (BM25 + Dense) with Cross-Encoder Reranking"** 架构，这是目前业界解决长文本检索的标准SOTA方案。

按照这个方案实现，既能保证找到“大海捞针”中的针（依靠 BM25 和 Rerank），又能保证回答的流畅性和逻辑性（依靠 `ecnu-max`），在准确率和创新性上应该都能拿到高分。
