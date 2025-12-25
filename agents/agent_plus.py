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
    符合长文本检索设想的高级检索 Agent：
    - Token 级切分（默认 350 tokens，overlap 100）
    - BM25 + 向量混合召回 (topK_b=40, topK_v=40)
    - ecnu-rerank 精排 (topN=10)
    - 证据块格式 + 仅基于证据回答，缺失则 Unknown
    """

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        load_dotenv()

        # 生成模型（优先 API_KEY -> ECNU_API_KEY）
        self.api_key = api_key or os.getenv("API_KEY") or os.getenv("ECNU_API_KEY")
        self.base_url = (base_url or os.getenv("BASE_URL") or os.getenv("ECNU_BASE_URL") or "").rstrip("/")
        self.model_name = os.getenv("MODEL_NAME") or "ecnu-max"
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

        # ECNU 专用模型（Embedding / Rerank）
        self.ecnu_api_key = os.getenv("ECNU_API_KEY") or self.api_key
        self.ecnu_base_url = (os.getenv("ECNU_BASE_URL") or self.base_url or "").rstrip("/")
        self.ecnu_client = OpenAI(api_key=self.ecnu_api_key, base_url=self.ecnu_base_url)
        self.embedding_model = "ecnu-embedding-small"
        self.rerank_model = "ecnu-rerank"

        # 检索参数
        self.chunk_size_tokens = 350
        self.chunk_overlap_tokens = 100
        self.top_k_bm25 = 40
        self.top_k_vector = 40
        self.rerank_top_n = 10
        self.max_tokens_per_request = 16000

        self.tokenizer = tiktoken.encoding_for_model("gpt-4")
        self.prompts = self._load_prompts()

    # -------------------------- Prompt -------------------------- #
    def _load_prompts(self) -> Dict[str, str]:
        prompt_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts", "agent_prompts.json")
        try:
            if os.path.exists(prompt_path):
                with open(prompt_path, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception:
            pass

        # 回退提示词
        return {
            "system_prompt": (
                "You are a grounded retrieval agent. Use ONLY provided evidence chunks.\n"
                "If the answer is not in evidence, reply \"Unknown\".\n"
                "Return a concise answer; add a brief citation if helpful."
            ),
            "user_prompt_template": (
                "Evidence:\n{context}\n\nQuestion: {question}\n\n"
                "Answer in English. If insufficient evidence, output \"Unknown\"."
            ),
        }

    # -------------------------- OpenAI helpers -------------------------- #
    async def _create_chat_completion(
        self,
        messages: List[Dict],
        model: Optional[str] = None,
        temperature: float = 0,
        max_tokens: int = 800,
        timeout: int = 60,
        response_format: Optional[Dict] = None,
        enable_thinking: bool = False,
    ) -> str:
        model_to_use = model or self.model_name
        params = {
            "model": model_to_use,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "timeout": timeout,
        }
        if response_format:
            params["response_format"] = response_format

        extra_body: Dict = {}
        if enable_thinking and self._supports_thinking(model_to_use):
            print("本次任务开启思考模式")
            # 控制思考长度，避免长 reasoning_content 被截断
            extra_body["thinking"] = {"type": "enabled", "budget_tokens": 256}
        elif not enable_thinking and self._supports_thinking(model_to_use):
            # 显式关闭，避免默认打开思考模式
            extra_body["thinking"] = {"type": "disabled"}
        if extra_body:
            params["extra_body"] = extra_body

        def _sync_call():
            return self.client.chat.completions.create(**params)

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
            if isinstance(content, str) and content.strip():
                return content.strip()
            reasoning = message.get("reasoning_content", "")
            if isinstance(reasoning, str) and reasoning.strip():
                return reasoning.strip()
            tool_calls = message.get("tool_calls")
            if tool_calls:
                return f"Tool calls detected: {tool_calls}"
            return f"Empty response (finish_reason: {choice.get('finish_reason', 'unknown')})"
        except Exception as e:
            return f"Response parsing error: {str(e)[:80]}"

    # -------------------------- Core API -------------------------- #
    async def evaluate_model(self, prompt: Dict) -> str:
        try:
            context_data = prompt.get("context_data", {})
            question = prompt.get("question", "")
            if not context_data or not question:
                return "Missing required input data"

            retrieval = self._retrieve_with_hybrid(question, context_data)
            evidence_text = retrieval["evidence_text"]

            system_prompt = (
                "You must answer ONLY using the provided evidence blocks.\n"
                "If evidence is insufficient, reply with \"Unknown\".\n"
                "Keep the answer concise; add minimal citation if helpful.\n"
                "If you run internal thinking, keep it short and focused."
            )

            user_template = (
                "Evidence blocks:\n{evidence}\n\n"
                "Question: {question}\n"
                "Answer directly. If unsure, reply \"Unknown\"."
            )
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_template.format(evidence=evidence_text, question=question)},
            ]

            response_raw = await self._create_chat_completion(
                messages=messages,
                max_tokens=800,
                temperature=0,
                response_format={"type": "json_object"},
                enable_thinking=False,
            )

            answer = self._extract_answer(response_raw)
            if not answer or answer.lower() == "unknown":
                # 兜底：开启思考模式；如当前为 ecnu-* 则切换 ecnu-reasoner
                fallback_model = self._select_thinking_model(self.model_name)
                backup_raw = await self._create_chat_completion(
                    messages=messages,
                    model=fallback_model,
                    max_tokens=500,  # 控制输出长度
                    temperature=0,
                    response_format={"type": "json_object"},
                    enable_thinking=True,
                )
                answer = self._extract_answer(backup_raw) or "Unknown"

                # 若思考模式导致 content 为空（reasoning 过长被截断），再尝试关闭思考重试
                if not answer or answer.lower().startswith("empty response"):
                    retry_raw = await self._create_chat_completion(
                        messages=messages,
                        model=fallback_model,
                        max_tokens=400,
                        temperature=0,
                        response_format={"type": "json_object"},
                        enable_thinking=False,
                    )
                    answer = self._extract_answer(retry_raw) or "Unknown"

            return answer.strip()
        except Exception as e:
            return f"Error: {str(e)[:80]}"

    def generate_prompt(self, **kwargs) -> Dict:
        return {"context_data": kwargs.get("context_data"), "question": kwargs.get("question")}

    def encode_text_to_tokens(self, text: str) -> List[int]:
        return self.tokenizer.encode(text)

    def decode_tokens(self, tokens: List[int], context_length: Optional[int] = None) -> str:
        if context_length:
            tokens = tokens[:context_length]
        return self.tokenizer.decode(tokens)

    # -------------------------- Retrieval -------------------------- #
    def _retrieve_with_hybrid(self, question: str, context_data: Dict) -> Dict:
        files = context_data.get("files", [])
        if not files:
            return {"evidence_text": "No relevant content found."}

        keywords = self._get_keywords(question)
        query_emb = self._get_embeddings(question)

        chunks: List[Dict] = []
        for doc_id, file_data in enumerate(files):
            content = file_data.get("modified_content", "")
            filename = file_data.get("filename", f"doc_{doc_id}")
            chunks.extend(self._chunk_document(content, filename, doc_id))

        if not chunks:
            return {"evidence_text": "No relevant content found."}

        # BM25 召回
        bm25_scores = self._score_bm25(chunks, keywords)
        bm25_top = sorted(bm25_scores.items(), key=lambda x: x[1], reverse=True)[: self.top_k_bm25]
        bm25_set = {idx for idx, _ in bm25_top}

        # 向量召回（对所有 chunk 取 embedding 可能开销大，这里对前 200 个关键词得分高的 chunk 计算）
        candidate_for_vector = [idx for idx, _ in bm25_top] if bm25_top else list(range(len(chunks)))
        candidate_for_vector = candidate_for_vector[:200]

        vector_scores: List[Tuple[int, float]] = []
        if query_emb:
            texts = [chunks[i]["text"] for i in candidate_for_vector]
            doc_embs = self._get_embeddings(texts)
            for idx, emb in zip(candidate_for_vector, doc_embs):
                if emb:
                    sim = self._dot(query_emb, emb)
                    vector_scores.append((idx, sim))
            vector_scores = sorted(vector_scores, key=lambda x: x[1], reverse=True)[: self.top_k_vector]

        vector_set = {idx for idx, _ in vector_scores}

        # 合并召回
        merged_indices = list(bm25_set | vector_set)
        merged_indices = merged_indices[: (self.top_k_bm25 + self.top_k_vector)]
        rerank_inputs = []
        for idx in merged_indices:
            chunk = chunks[idx]
            header = f"[Doc {chunk['doc_id']} | Chunk {chunk['chunk_id']} | Offset {chunk['token_start']}-{chunk['token_end']}]"
            rerank_inputs.append(f"{header}\n{chunk['text']}")

        # 精排
        reranked = self._rerank_documents(question, rerank_inputs, top_n=self.rerank_top_n)

        evidence_blocks = []
        for rank_idx, snippet in enumerate(reranked):
            score = max(0.0, 1.0 - rank_idx * 0.05)  # 简单衰减评分用于展示
            evidence_blocks.append(f"[Rank {rank_idx+1} | Score {score:.2f}]\n{snippet}")

        evidence_text = "\n\n---\n\n".join(evidence_blocks)
        return {"evidence_text": evidence_text}

    def _chunk_document(self, content: str, filename: str, doc_id: int) -> List[Dict]:
        tokens = self.encode_text_to_tokens(content)
        if not tokens:
            return []

        chunks = []
        step = max(1, self.chunk_size_tokens - self.chunk_overlap_tokens)
        chunk_id = 0
        for start in range(0, len(tokens), step):
            end = start + self.chunk_size_tokens
            window = tokens[start:end]
            if not window:
                continue
            text = self.decode_tokens(window)
            if len(text.strip()) < 20:
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

    # -------------------------- Scoring -------------------------- #
    def _score_bm25(self, chunks: List[Dict], keywords: List[str]) -> Dict[int, float]:
        """
        轻量 BM25 实现，使用词频 + 文档长度标准化。
        """
        if not keywords or not chunks:
            return {}

        tokenized_chunks: List[List[str]] = []
        df: Dict[str, int] = {}
        for chunk in chunks:
            words = re.findall(r"[a-zA-Z0-9]+", chunk["text"].lower())
            tokenized_chunks.append(words)
            unique_words = set(words)
            for w in unique_words:
                df[w] = df.get(w, 0) + 1

        N = len(chunks)
        avgdl = sum(len(c) for c in tokenized_chunks) / max(N, 1)
        k1, b = 1.5, 0.75

        scores: Dict[int, float] = {}
        for idx, words in enumerate(tokenized_chunks):
            tf: Dict[str, int] = {}
            for w in words:
                tf[w] = tf.get(w, 0) + 1
            score = 0.0
            dl = len(words)
            for kw in keywords:
                term = kw.lower()
                if term not in tf:
                    continue
                idf = math.log((N - df.get(term, 0) + 0.5) / (df.get(term, 0) + 0.5) + 1)
                numer = tf[term] * (k1 + 1)
                denom = tf[term] + k1 * (1 - b + b * dl / max(avgdl, 1e-6))
                score += idf * numer / max(denom, 1e-6)
            scores[idx] = score
        return scores

    def _dot(self, a: List[float], b: List[float]) -> float:
        return sum(x * y for x, y in zip(a, b))

    def _supports_thinking(self, model_name: str) -> bool:
        """
        判断模型是否支持 thinking 参数。
        约定：ecnu-* 只有 ecnu-reasoner 支持。
        """
        name = (model_name or "").lower()
        if name.startswith("ecnu-"):
            return name == "ecnu-reasoner"
        # 其他模型默认支持思考开关
        return True

    def _select_thinking_model(self, current_model: str) -> str:
        """
        若当前为 ecnu-* 且非 reasoner，则切换到 ecnu-reasoner，否则使用原模型。
        """
        name = (current_model or "").lower()
        if name.startswith("ecnu-") and name != "ecnu-reasoner":
            return "ecnu-reasoner"
        return current_model

    # -------------------------- Models -------------------------- #
    def _get_embeddings(self, input_data: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """
        使用 ecnu-embedding-small 获取向量，直接调用（API 无上限）。
        """
        try:
            if isinstance(input_data, str):
                response = self.ecnu_client.embeddings.create(model=self.embedding_model, input=input_data)
                return response.data[0].embedding

            resp = self.ecnu_client.embeddings.create(model=self.embedding_model, input=input_data)
            sorted_data = sorted(resp.data, key=lambda x: x.index)
            return [item.embedding for item in sorted_data]
        except Exception as e:
            print(f"Error getting embeddings: {e}")
            if isinstance(input_data, str):
                return []
            return [[] for _ in input_data]

    def _rerank_documents(self, query: str, documents: List[str], top_n: int = 10) -> List[str]:
        if not documents:
            return []
        url = f"{self.ecnu_base_url}/rerank"
        headers = {"Authorization": f"Bearer {self.ecnu_api_key}", "Content-Type": "application/json"}
        payload = {
            "model": self.rerank_model,
            "query": query,
            "documents": documents,
            "top_n": top_n,
            "return_documents": True,
        }
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
            results = result.get("results", [])
            if not results:
                return documents[:top_n]
            return [item["document"] for item in results]
        except Exception as e:
            print(f"Error during reranking: {e}")
            return documents[:top_n]

    # -------------------------- Helpers -------------------------- #
    def _extract_answer(self, response_raw: str) -> str:
        try:
            clean_raw = response_raw.strip()
            if "```" in clean_raw:
                m = re.search(r"```(?:json)?\\s*(\\{.*?\\})\\s*```", clean_raw, re.DOTALL)
                if m:
                    clean_raw = m.group(1)
            data = json.loads(clean_raw)
            answer = str(data.get("answer", "")).strip()
            return answer or clean_raw
        except Exception:
            match = re.search(r'"answer"\\s*:\\s*"([^"]+)"', response_raw)
            if match:
                return match.group(1).strip()
            return response_raw.strip()

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

        keywords: List[str] = []
        # 编码 / 数字
        codes = re.findall(r"\b[A-Z0-9]+-[A-Z0-9-]+[A-Z0-9]\b", question)
        keywords.extend([c.lower() for c in codes])
        project_codes = re.findall(r"\b[A-Z]-\d+-[A-Za-z]+\b", question)
        keywords.extend([c.lower() for c in project_codes])
        simple_codes = re.findall(r"\b[A-Z]{2,}-[A-Z0-9]+\b", question)
        keywords.extend([c.lower() for c in simple_codes])
        numbers = re.findall(r"\b\d+\b", question)
        keywords.extend(numbers)
        years = re.findall(r"\b20[2-9]\d\b", question)
        keywords.extend(years)
        months = re.findall(
            r"\b(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)\b",
            question.lower(),
        )
        keywords.extend(months)
        days = re.findall(r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday|mon|tue|wed|thu|fri|sat|sun)\b", question.lower())
        keywords.extend(days)
        words = re.findall(r"\b[a-zA-Z]+\b", question.lower())
        keywords.extend([w for w in words if w not in stop_words and len(w) > 2])
        return list(dict.fromkeys(keywords))[:25]
