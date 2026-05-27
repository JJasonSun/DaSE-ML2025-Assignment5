import math
import os
import re
from typing import Dict, List, Optional, Union, cast

import requests
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

from core.ecnu_constants import (
    DEFAULT_ECNU_BASE_URL,
    ECNU_EMBEDDING_MODEL_NAME,
    ECNU_MAIN_MODEL_NAME,
    ECNU_RERANK_MODEL_NAME,
)
from .base_agent import ModelProvider


class AdvancedRetrievalAgent(ModelProvider):
    """
    Hybrid retrieval agent: BM25 + Dense Embedding + Rerank.
    Falls back to full-context mode when total tokens < 64K.
    """

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        load_dotenv()

        api_key = api_key or os.getenv("ECNU_API_KEY") or ""
        base_url = base_url or os.getenv("ECNU_BASE_URL") or DEFAULT_ECNU_BASE_URL
        super().__init__(api_key=api_key, base_url=base_url)

        self.base_url = base_url.rstrip("/")
        self.model_name = os.getenv("MODEL_NAME") or ECNU_MAIN_MODEL_NAME

        self.embedding_model = ECNU_EMBEDDING_MODEL_NAME
        self.rerank_model = ECNU_RERANK_MODEL_NAME

        self.chunk_size_tokens = 500
        self.chunk_overlap_tokens = 100
        self.top_k_bm25 = 30
        self.top_k_vector = 20
        self.rerank_top_n = 8

        self.full_context_threshold_tokens = 64000
        self.max_evidence_tokens = 32000
        self.max_ecnu_retrieval_chars = 8192

        self.prompts = self._load_prompts()

    def _load_prompts(self) -> Dict[str, str]:
        return {
            "system_prompt": (
                "You are a meticulous retrieval and reasoning expert operating in a Needle-in-a-Haystack scenario: "
                "your task is to locate precise evidence within a vast context and derive the correct answer.\n\n"
                "## Principles\n"
                "1. Grounding: Use ONLY the provided context combined with general reasoning and arithmetic. "
                "Do NOT introduce external knowledge or assumptions.\n"
                "2. Internal reasoning: Decompose the problem, locate evidence, and verify — all internally. "
                "Never output your reasoning process, chain-of-thought, or intermediate steps.\n"
                "3. Active computation: When dates require weekday calculation or numbers require arithmetic, "
                "perform the computation internally and ensure accuracy.\n"
                "4. Output format: Output ONLY the final answer. No explanations, no restating evidence, "
                "no bullet points, no prefixes like \"The answer is\", no reasoning traces.\n"
                "5. Fallback: If after exhaustive retrieval and computation no answer can be determined, "
                "return exactly \"Unknown\".\n"
                "6. Resilience: Evidence may be fragmented, obscured, or scattered across passages. "
                "Stay patient, apply rigorous logic, and avoid premature abandonment.\n\n"
                "## Workflow\n"
                "Analyze the question → locate and align evidence → compute if necessary → cross-verify → output only the final answer."
            ),
            "user_prompt_template": (
                "Context:\n{context}\n\nQuestion: {question}\n\n"
                "Output only the final answer. No explanation."
            ),
        }

    # -------------------------- ModelProvider API -------------------------- #
    async def evaluate_model(self, prompt: Dict) -> str:
        context_data = prompt.get("context_data", {}) or {}
        context_str = prompt.get("context", "") or ""
        question = prompt.get("question", "") or ""
        if not question:
            return "Missing required input data"

        # single 模式：直接使用 context 字符串
        if not context_data and context_str:
            context_for_llm = context_str
        else:
            full_context = self._build_full_context(context_data)
            total_context_tokens = int(full_context["total_tokens"])
            if total_context_tokens <= self.full_context_threshold_tokens:
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

        response_raw = await self._create_chat_completion(
            messages=messages,
            enable_thinking=getattr(self, 'enable_thinking', False),
        )
        return self.finalize_answer(response_raw.strip())

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
    def _retrieve_with_hybrid(self, question: str, context_data: Dict, queries: Optional[List[str]] = None) -> Dict[str, str]:
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

        search_queries = queries if queries else [question]
        bm25_indices = set()
        vector_indices = set()

        for q in search_queries:
            keywords = self._get_keywords(q)
            bm25_scores = self._score_bm25(chunks, keywords)
            bm25_top = sorted(bm25_scores.items(), key=lambda x: x[1], reverse=True)[: self.top_k_bm25]
            bm25_indices.update([idx for idx, score in bm25_top if score > 0])

            query_emb = cast(List[float], self._get_embeddings(q))
            if query_emb:
                if len(chunks) <= 300:
                    candidate_for_vector = list(range(len(chunks)))
                else:
                    dense_pool = min(400, len(chunks))
                    bm25_pool = sorted(bm25_scores.items(), key=lambda x: x[1], reverse=True)[:dense_pool]
                    candidate_for_vector = [idx for idx, _ in bm25_pool]

                texts = [chunks[i]["text"] for i in candidate_for_vector]
                doc_embs = cast(List[List[float]], self._get_embeddings_in_batches(texts, batch_size=64))

                q_vector_scores = []
                for idx, emb in zip(candidate_for_vector, doc_embs):
                    if emb:
                        sim = self._cosine_similarity(query_emb, cast(List[float], emb))
                        q_vector_scores.append((idx, sim))
                q_vector_top = sorted(q_vector_scores, key=lambda x: x[1], reverse=True)[: self.top_k_vector]
                vector_indices.update([idx for idx, sim in q_vector_top])

        merged_indices = list(bm25_indices | vector_indices)[:60]

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
            text_to_rerank = self._trim_retrieval_text(f"{header}\n{chunk['text']}")
            rerank_inputs.append(text_to_rerank)
            chunk_map[text_to_rerank] = idx

        if not rerank_inputs:
            return {"evidence_text": "No relevant content found."}

        reranked_results = self._rerank_documents(question, rerank_inputs, top_n=self.rerank_top_n * 2)

        if not reranked_results:
            return {"evidence_text": "No relevant content found."}

        scores = [r.get("relevance_score", 0) for r in reranked_results]
        max_score = max(scores) if scores else 0
        threshold = max_score * 0.15

        evidence_blocks: List[str] = []
        total_tokens = 0
        added_chunk_indices = set()

        for rank_idx, res in enumerate(reranked_results):
            score = res.get("relevance_score", 0)
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

        tokenized_chunks = [self._tokenize(chunk["text"]) for chunk in chunks]
        bm25 = BM25Okapi(tokenized_chunks)

        scores: Dict[int, float] = {i: 0.0 for i in range(len(chunks))}

        for kw in keywords:
            term = kw.lower()
            weight = 2.0 if ("-" in term or ":" in term or any(c.isdigit() for c in term)) else 1.0

            tokenized_query = self._tokenize(term)
            if not tokenized_query:
                continue

            kw_scores = bm25.get_scores(tokenized_query)
            for i, score in enumerate(kw_scores):
                if score > 0:
                    scores[i] += score * weight

        return scores

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"[a-zA-Z0-9]+(?:-[a-zA-Z0-9]+)*", text.lower())

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
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "can", "what", "when", "where", "why", "how",
            "who", "which", "that", "this", "these", "those",
        }

        q = question or ""
        keywords: List[str] = []

        # Hyphenated codes / IDs
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

        # JSON / Key-Value pairs
        kv_pairs = re.findall(r'["\']?(\w+)["\']?\s*[:=]\s*["\']?([^"\'\s,{}]+)["\']?', q)
        for k, v in kv_pairs:
            keywords.append(k.lower())
            keywords.append(v.lower())

        # Special symbol combinations: [[...]], <<...>>, ((...))
        special_content = re.findall(r"[\[<{(]([^\[<{(]+)[\]>})]", q)
        for sc in special_content:
            sc_l = sc.lower()
            keywords.append(sc_l)
            keywords.extend([p for p in re.split(r"[\s\-_]+", sc_l) if len(p) > 2])

        # Numbers / dates
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

        # Content words (filtered)
        words = re.findall(r"\b[a-zA-Z]+\b", q.lower())
        keywords.extend([w for w in words if w not in stop_words and len(w) > 2])

        return list(dict.fromkeys(keywords))[:25]

    # -------------------------- External model calls -------------------------- #
    def _get_embeddings(self, input_data: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        try:
            if isinstance(input_data, str):
                response = self.client.embeddings.create(model=self.embedding_model, input=self._trim_retrieval_text(input_data))
                return response.data[0].embedding
            resp = self.client.embeddings.create(
                model=self.embedding_model,
                input=[self._trim_retrieval_text(text) for text in input_data],
            )
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
            if isinstance(batch_embs, list):
                results.extend(cast(List[List[float]], batch_embs))
        return results

    def _rerank_documents(self, query: str, documents: List[str], top_n: int = 8) -> List[Dict]:
        if not documents:
            return []
        url = f"{self.base_url}/rerank"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
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
            return result.get("results", [])
        except Exception as e:
            print(f"Error during reranking: {e}")
            return [{"document": d, "relevance_score": 0.0} for d in documents[:top_n]]

    def _trim_retrieval_text(self, text: str) -> str:
        text = text or ""
        if len(text) <= self.max_ecnu_retrieval_chars:
            return text
        return text[: self.max_ecnu_retrieval_chars]
