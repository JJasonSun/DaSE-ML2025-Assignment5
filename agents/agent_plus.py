import asyncio
import json
import math
import os
import re
from typing import Dict, List, Optional, Tuple, Union

import requests
import tiktoken
from dotenv import load_dotenv
from openai import OpenAI

from model import ModelProvider


class AdvancedRetrievalAgent(ModelProvider):
    """
    长文本检索 Agent（设想2版）：
    - Token 级切分：chunk_size=500，overlap=100
    - 混合召回：BM25(精确匹配) + Dense(语义匹配)
    - 精排：ecnu-rerank 选 Top-N（默认 8）
    - 生成：ecnu-max；当总上下文较小（<30000 tokens）直接全量喂给模型
    """

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        load_dotenv()

        api_key = api_key or os.getenv("API_KEY") or os.getenv("ECNU_API_KEY") or ""
        base_url = base_url or os.getenv("BASE_URL") or os.getenv("ECNU_BASE_URL") or ""
        super().__init__(api_key=api_key, base_url=base_url)

        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model_name = os.getenv("MODEL_NAME") or "ecnu-max"
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

        self.ecnu_api_key = os.getenv("ECNU_API_KEY") or self.api_key
        self.ecnu_base_url = (os.getenv("ECNU_BASE_URL") or self.base_url).rstrip("/")
        self.ecnu_client = OpenAI(api_key=self.ecnu_api_key, base_url=self.ecnu_base_url)

        self.embedding_model = "ecnu-embedding-small"
        self.rerank_model = "ecnu-rerank"

        self.chunk_size_tokens = 500
        self.chunk_overlap_tokens = 100
        self.top_k_bm25 = 30
        self.top_k_vector = 20
        self.rerank_top_n = 8

        self.full_context_threshold_tokens = 64000
        self.max_evidence_tokens = 32000

        self.tokenizer = tiktoken.encoding_for_model("gpt-4")
        self.prompts = self._load_prompts()

    def _load_prompts(self) -> Dict[str, str]:
        prompt_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts", "agent_prompts.json")
        try:
            if os.path.exists(prompt_path):
                with open(prompt_path, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception:
            pass

        return {
            "system_prompt": (
                "You are a high-precision retrieval agent.\n"
                "Use ONLY the provided context/evidence.\n"
                "If the answer is not present, reply \"Unknown\".\n"
                "Return ONLY a JSON object: {\"answer\": \"...\"}.\n"
                "Do NOT output your reasoning."
            ),
            "user_prompt_template": (
                "Context:\n{context}\n\nQuestion: {question}\n\n"
                "Return ONLY a JSON object: {{\"answer\": \"...\"}}."
            ),
        }

    # -------------------------- Chat helpers -------------------------- #
    async def _create_chat_completion(
        self,
        messages: List[Dict],
        model: Optional[str] = None,
        temperature: float = 0,
        max_tokens: int = 800,
        timeout: int = 60,
        response_format: Optional[Dict] = None,
        enable_thinking: bool = False,
        thinking_budget_tokens: int = 1024,
    ) -> str:
        model_to_use = model or self.model_name
        params: Dict = {
            "model": model_to_use,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "timeout": timeout,
        }
        if response_format:
            params["response_format"] = response_format

        extra_body: Dict = {}
        if self._supports_thinking(model_to_use):
            if enable_thinking:
                print("本次任务开启思考模式")
                extra_body["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget_tokens}
            else:
                extra_body["thinking"] = {"type": "disabled"}
        if extra_body:
            params["extra_body"] = extra_body

        # 根据模型名称自动选择 Client
        client_to_use = self.ecnu_client if model_to_use.startswith("ecnu-") else self.client

        def _sync_call():
            return client_to_use.chat.completions.create(**params)

        try:
            try:
                completion = await asyncio.to_thread(_sync_call)
            except AttributeError:
                loop = asyncio.get_running_loop()
                completion = await loop.run_in_executor(None, _sync_call)
            raw = completion.to_dict()
            print(f"[Debug] Raw response: {raw}")
            return self._extract_content_from_response(raw)
        except Exception as exc:
            return f"API error: {exc}" if exc else "API error"

    def _extract_content_from_response(self, result: dict) -> str:
        try:
            choice = result.get("choices", [{}])[0]
            message = choice.get("message", {})
            content = message.get("content", "")
            reasoning = message.get("reasoning_content", "")
            finish_reason = choice.get("finish_reason", "unknown")

            if isinstance(content, str) and content.strip():
                return content.strip()
            
            # 如果 content 为空但有 reasoning_content，可能是因为思维链过长导致截断
            # 尝试返回思维链内容，以便 _extract_answer 尝试从中解析答案
            if isinstance(reasoning, str) and reasoning.strip():
                print(f"[Debug] Content is empty, but reasoning_content found (finish_reason: {finish_reason})")
                return reasoning.strip()

            return f"Empty response (finish_reason: {finish_reason})"
        except Exception as e:
            return f"Response parsing error: {str(e)[:80]}"

    def _extract_answer(self, response_raw: str) -> str:
        try:
            clean_raw = response_raw.strip()
            if "```" in clean_raw:
                m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", clean_raw, re.DOTALL)
                if m:
                    clean_raw = m.group(1)
            data = json.loads(clean_raw)
            return str(data.get("answer", "")).strip()
        except Exception:
            m = re.search(r'"answer"\s*:\s*"([^"]*)"', response_raw)
            return (m.group(1).strip() if m else response_raw.strip())

    # -------------------------- ModelProvider API -------------------------- #
    async def evaluate_model(self, prompt: Dict) -> str:
        context_data = prompt.get("context_data", {}) or {}
        question = prompt.get("question", "") or ""
        if not question:
            return "Missing required input data"

        full_context = self._build_full_context(context_data)
        if full_context["total_tokens"] <= self.full_context_threshold_tokens:
            context_for_llm = full_context["text"]
        else:
            evidence = self._retrieve_with_hybrid(question, context_data)
            context_for_llm = evidence["evidence_text"]

        system_prompt = self.prompts.get("system_prompt", "")
        user_template = self.prompts.get("user_prompt_template", "{context}\n\n{question}")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_template.format(context=context_for_llm, question=question)},
        ]

        # 默认不启用思考模式（避免输出 reasoning_content、减少截断风险）
        response_raw = await self._create_chat_completion(
            messages=messages,
            temperature=0,
            max_tokens=800,
            timeout=90,
            response_format={"type": "json_object"},
            enable_thinking=False,
        )
        answer = self._extract_answer(response_raw)

        # 兜底：开启思考模式；ecnu-* 只有 ecnu-reasoner 支持 thinking
        if not answer or answer.lower() == "unknown" or answer.lower().startswith("api error") or answer.lower().startswith("empty response"):
            fallback_model = self._select_thinking_model(self.model_name)

            # 自适应策略：如果是因为检索不到，尝试扩大检索范围
            if full_context["total_tokens"] > self.full_context_threshold_tokens:
                # 临时增加召回数量进行重试
                orig_top_n = self.rerank_top_n
                self.rerank_top_n = 15
                evidence = self._retrieve_with_hybrid(question, context_data)
                context_for_llm_retry = evidence["evidence_text"]
                self.rerank_top_n = orig_top_n

                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_template.format(context=context_for_llm_retry, question=question)},
                ]

            response_raw = await self._create_chat_completion(
                messages=messages,
                model=fallback_model,
                temperature=0,
                max_tokens=1000,
                timeout=90,
                response_format={"type": "json_object"},
                enable_thinking=True,
                thinking_budget_tokens=1024,
            )
            answer = self._extract_answer(response_raw)

        # 最后兜底：仍然空/被截断，则关闭思考重试一次（避免 reasoning 截断导致 content=None）
        if not answer or answer.lower().startswith("empty response"):
            response_raw = await self._create_chat_completion(
                messages=messages,
                model=self._select_thinking_model(self.model_name),
                temperature=0,
                max_tokens=500,
                timeout=90,
                response_format={"type": "json_object"},
                enable_thinking=False,
            )
            answer = self._extract_answer(response_raw)

        return (answer or "Unknown").strip()

    def generate_prompt(self, **kwargs) -> Dict:
        return {"context_data": kwargs.get("context_data"), "question": kwargs.get("question")}

    def encode_text_to_tokens(self, text: str) -> List[int]:
        return self.tokenizer.encode(text or "")

    def decode_tokens(self, tokens: List[int], context_length: Optional[int] = None) -> str:
        if context_length is not None:
            tokens = tokens[:context_length]
        return self.tokenizer.decode(tokens)

    # -------------------------- Full-context mode -------------------------- #
    def _build_full_context(self, context_data: Dict) -> Dict[str, Union[str, int]]:
        files = context_data.get("files", []) or []
        parts: List[str] = []
        total_tokens = 0
        for i, f in enumerate(files):
            filename = f.get("filename", f"doc_{i}")
            text = f.get("modified_content", "") or ""
            parts.append(f"=== {filename} ===\n{text}")
            total_tokens += len(self.encode_text_to_tokens(text))
        return {"text": "\n\n".join(parts), "total_tokens": total_tokens}

    # -------------------------- Retrieval pipeline -------------------------- #
    def _retrieve_with_hybrid(self, question: str, context_data: Dict) -> Dict[str, str]:
        files = context_data.get("files", []) or []
        if not files:
            return {"evidence_text": "No relevant content found."}

        chunks: List[Dict] = []
        for doc_id, file_data in enumerate(files):
            content = file_data.get("modified_content", "") or ""
            filename = file_data.get("filename", f"doc_{doc_id}")
            chunks.extend(self._chunk_document(content, filename, doc_id))

        if not chunks:
            return {"evidence_text": "No relevant content found."}

        keywords = self._get_keywords(question)

        # 1) BM25 召回（精确匹配强）
        bm25_scores = self._score_bm25(chunks, keywords)
        bm25_top = sorted(bm25_scores.items(), key=lambda x: x[1], reverse=True)[: self.top_k_bm25]
        bm25_set = {idx for idx, _ in bm25_top}

        # 2) Dense 召回（语义匹配强）
        query_emb = self._get_embeddings(question)
        vector_scores: List[Tuple[int, float]] = []
        if query_emb:
            # 若 chunk 较少就全量向量召回，否则只对 BM25 高分的若干 chunk 做向量（避免太慢）
            if len(chunks) <= 300:
                candidate_for_vector = list(range(len(chunks)))
            else:
                dense_pool = min(400, len(chunks))
                bm25_pool = sorted(bm25_scores.items(), key=lambda x: x[1], reverse=True)[:dense_pool]
                candidate_for_vector = [idx for idx, _ in bm25_pool]

            texts = [chunks[i]["text"] for i in candidate_for_vector]
            doc_embs = self._get_embeddings_in_batches(texts, batch_size=64)
            for idx, emb in zip(candidate_for_vector, doc_embs):
                if emb:
                    sim = self._cosine_similarity(query_emb, emb)
                    vector_scores.append((idx, sim))
            vector_scores = sorted(vector_scores, key=lambda x: x[1], reverse=True)[: self.top_k_vector]

        vector_set = {idx for idx, _ in vector_scores}

        # 3) 合并候选并去重
        merged_indices = list(bm25_set | vector_set)
        merged_indices = merged_indices[: (self.top_k_bm25 + self.top_k_vector)]

        rerank_inputs: List[str] = []
        chunk_map: Dict[str, int] = {}
        seen = set()
        for idx in merged_indices:
            chunk = chunks[idx]
            key = (chunk["doc_id"], chunk["chunk_id"])
            if key in seen:
                continue
            seen.add(key)
            header = (
                f"[Doc {chunk['doc_id']} | File {chunk['filename']} | Chunk {chunk['chunk_id']} | "
                f"Offset {chunk['token_start']}-{chunk['token_end']}]"
            )
            text_to_rerank = f"{header}\n{chunk['text']}"
            rerank_inputs.append(text_to_rerank)
            chunk_map[text_to_rerank] = idx

        if not rerank_inputs:
            return {"evidence_text": "No relevant content found."}

        # 4) Rerank 精排
        reranked_results = self._rerank_documents(question, rerank_inputs, top_n=self.rerank_top_n * 2)

        # 5) 组装 evidence（动态 Top-N + 上下文补全）
        if not reranked_results:
            return {"evidence_text": "No relevant content found."}

        # 动态阈值：保留得分显著高于平均水平或前几名的
        scores = [r.get("relevance_score", 0) for r in reranked_results]
        max_score = max(scores) if scores else 0
        threshold = max_score * 0.2  # 阈值：最高分的 20%

        evidence_blocks: List[str] = []
        total_tokens = 0
        added_chunk_indices = set()

        for rank_idx, res in enumerate(reranked_results):
            score = res.get("relevance_score", 0)
            # 如果分数太低且已经有一定数量的证据，则停止
            if score < threshold and len(evidence_blocks) >= 3:
                break
            if len(evidence_blocks) >= self.rerank_top_n:
                break

            snippet = res["document"]
            orig_idx = chunk_map.get(snippet)

            if orig_idx is not None and orig_idx not in added_chunk_indices:
                block = f"[Rank {rank_idx + 1} | Score {score:.4f}]\n{snippet}"
                block_tokens = len(self.encode_text_to_tokens(block))
                if total_tokens + block_tokens <= self.max_evidence_tokens:
                    evidence_blocks.append(block)
                    total_tokens += block_tokens
                    added_chunk_indices.add(orig_idx)

                    # 上下文补全：对于前 3 名，尝试添加相邻 chunk 以保证语义完整
                    if rank_idx < 3:
                        for neighbor_idx in [orig_idx - 1, orig_idx + 1]:
                            if 0 <= neighbor_idx < len(chunks) and neighbor_idx not in added_chunk_indices:
                                neighbor = chunks[neighbor_idx]
                                if neighbor["doc_id"] == chunks[orig_idx]["doc_id"]:
                                    n_header = f"[Neighbor of Rank {rank_idx + 1} | Chunk {neighbor['chunk_id']}]"
                                    n_block = f"{n_header}\n{neighbor['text']}"
                                    n_tokens = len(self.encode_text_to_tokens(n_block))
                                    if total_tokens + n_tokens <= self.max_evidence_tokens:
                                        evidence_blocks.append(n_block)
                                        total_tokens += n_tokens
                                        added_chunk_indices.add(neighbor_idx)

        evidence_text = "\n\n---\n\n".join(evidence_blocks) if evidence_blocks else "No relevant content found."
        return {"evidence_text": evidence_text}

    def _chunk_document(self, content: str, filename: str, doc_id: int) -> List[Dict]:
        tokens = self.encode_text_to_tokens(content)
        if not tokens:
            return []

        chunks: List[Dict] = []
        step = max(1, self.chunk_size_tokens - self.chunk_overlap_tokens)
        chunk_id = 0
        for start in range(0, len(tokens), step):
            end = start + self.chunk_size_tokens
            window = tokens[start:end]
            if not window:
                continue
            text = self.decode_tokens(window).strip()
            if len(text) < 20:
                continue
            chunks.append(
                {
                    "doc_id": doc_id,
                    "filename": filename,
                    "chunk_id": chunk_id,
                    "token_start": start,
                    "token_end": min(end, len(tokens)),
                    "text": text,
                }
            )
            chunk_id += 1
        return chunks

    # -------------------------- Retrieval utils -------------------------- #
    def _score_bm25(self, chunks: List[Dict], keywords: List[str]) -> Dict[int, float]:
        if not keywords or not chunks:
            return {}

        tokenized_chunks: List[List[str]] = []
        df: Dict[str, int] = {}
        for chunk in chunks:
            # 保留形如 AF-PROJ-8876 / ARC-7F3B-92E4 的连字符编码
            words = re.findall(r"[a-zA-Z0-9]+(?:-[a-zA-Z0-9]+)*", chunk["text"].lower())
            tokenized_chunks.append(words)
            for w in set(words):
                df[w] = df.get(w, 0) + 1

        N = len(chunks)
        avgdl = sum(len(c) for c in tokenized_chunks) / max(N, 1)
        k1, b = 1.5, 0.75

        scores: Dict[int, float] = {}
        for idx, words in enumerate(tokenized_chunks):
            tf: Dict[str, int] = {}
            for w in words:
                tf[w] = tf.get(w, 0) + 1

            dl = len(words)
            score = 0.0
            for kw in keywords:
                term = kw.lower()
                if term not in tf:
                    continue
                df_t = df.get(term, 0)
                idf = math.log((N - df_t + 0.5) / (df_t + 0.5) + 1)
                numer = tf[term] * (k1 + 1)
                denom = tf[term] + k1 * (1 - b + b * dl / max(avgdl, 1e-6))
                score += idf * numer / max(denom, 1e-6)
            scores[idx] = score
        return scores

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        dot = 0.0
        na = 0.0
        nb = 0.0
        for x, y in zip(a, b):
            dot += x * y
            na += x * x
            nb += y * y
        denom = math.sqrt(na) * math.sqrt(nb)
        return dot / denom if denom > 0 else 0.0

    def _get_keywords(self, question: str) -> List[str]:
        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "can",
            "what",
            "when",
            "where",
            "why",
            "how",
            "who",
            "which",
            "that",
            "this",
            "these",
            "those",
        }

        q = question or ""
        keywords: List[str] = []

        # 1) 强特征：带连字符的编码 / ID
        codes = re.findall(r"\b[A-Z0-9]+-[A-Z0-9-]+[A-Z0-9]\b", q)
        for c in codes:
            c_l = c.lower()
            keywords.append(c_l)
            keywords.extend([p for p in c_l.split("-") if p])

        project_codes = re.findall(r"\b[A-Z]-\d+-[A-Za-z]+\b", q)
        for c in project_codes:
            c_l = c.lower()
            keywords.append(c_l)
            keywords.extend([p for p in c_l.split("-") if p])

        simple_codes = re.findall(r"\b[A-Z]{2,}-[A-Z0-9]+\b", q)
        for c in simple_codes:
            c_l = c.lower()
            keywords.append(c_l)
            keywords.extend([p for p in c_l.split("-") if p])

        # 1.1) JSON / Key-Value pairs: "key": "value" or key=value
        kv_pairs = re.findall(r'["\']?(\w+)["\']?\s*[:=]\s*["\']?([^"\'\s,{}]+)["\']?', q)
        for k, v in kv_pairs:
            keywords.append(k.lower())
            keywords.append(v.lower())

        # 1.2) Special symbol combinations: [[...]], <<...>>, ((...))
        special_content = re.findall(r"[\[<{(]([^\[<{(]+)[\]>})]", q)
        for sc in special_content:
            sc_l = sc.lower()
            keywords.append(sc_l)
            keywords.extend([p for p in re.split(r"[\s\-_]+", sc_l) if len(p) > 2])

        # 2) 数字 / 日期相关
        keywords.extend(re.findall(r"\b\d+\b", q))
        keywords.extend(re.findall(r"\b20[0-9]{2}\b", q))

        months = re.findall(
            r"\b(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)\b",
            q.lower(),
        )
        keywords.extend(months)

        days = re.findall(
            r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday|mon|tue|wed|thu|fri|sat|sun)\b",
            q.lower(),
        )
        keywords.extend(days)

        # 3) 常规单词（过滤停用词）
        words = re.findall(r"\b[a-zA-Z]+\b", q.lower())
        keywords.extend([w for w in words if w not in stop_words and len(w) > 2])

        # 控制数量，避免 BM25 被噪声拖累
        return list(dict.fromkeys(keywords))[:25]

    # -------------------------- External model calls -------------------------- #
    def _get_embeddings(self, input_data: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        try:
            if isinstance(input_data, str):
                response = self.ecnu_client.embeddings.create(model=self.embedding_model, input=input_data)
                return response.data[0].embedding
            resp = self.ecnu_client.embeddings.create(model=self.embedding_model, input=input_data)
            sorted_data = sorted(resp.data, key=lambda x: x.index)
            return [item.embedding for item in sorted_data]
        except Exception as e:
            print(f"Error getting embeddings: {e}")
            return [] if isinstance(input_data, str) else [[] for _ in input_data]

    def _get_embeddings_in_batches(self, texts: List[str], batch_size: int = 64) -> List[List[float]]:
        if not texts:
            return []
        results: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_embs = self._get_embeddings(batch)
            results.extend(batch_embs if isinstance(batch_embs, list) else [])
        return results

    def _rerank_documents(self, query: str, documents: List[str], top_n: int = 8) -> List[Dict]:
        if not documents:
            return []
        url = f"{self.ecnu_base_url}/rerank"
        headers = {"Authorization": f"Bearer {self.ecnu_api_key}", "Content-Type": "application/json"}
        payload = {"model": self.rerank_model, "query": query, "documents": documents, "top_n": top_n, "return_documents": True}
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
            return result.get("results", [])
        except Exception as e:
            print(f"Error during reranking: {e}")
            return [{"document": d, "relevance_score": 0.0} for d in documents[:top_n]]

    # -------------------------- Thinking mode rules -------------------------- #
    def _supports_thinking(self, model_name: str) -> bool:
        name = (model_name or "").lower()
        if name.startswith("ecnu-"):
            return name == "ecnu-reasoner"
        return True

    def _select_thinking_model(self, current_model: str) -> str:
        name = (current_model or "").lower()
        if name.startswith("ecnu-") and name != "ecnu-reasoner":
            return "ecnu-reasoner"
        return current_model

