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


class HybridRetrievalAgent(ModelProvider):
    """Hybrid retrieval baseline: BM25 + dense embedding + rerank + neighbor chunks."""

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        load_dotenv()
        api_key = api_key or os.getenv("ECNU_API_KEY") or ""
        base_url = (base_url or os.getenv("ECNU_BASE_URL") or DEFAULT_ECNU_BASE_URL).rstrip("/")
        super().__init__(api_key=api_key, base_url=base_url)

        self.base_url = base_url
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
        self.last_trace: Dict = {}

        self.prompts = {
            "system_prompt": (
                "You are a rigorous evidence retrieval and reasoning agent for Needle-in-a-Haystack tasks. "
                "Use only the supplied context. Internally locate all relevant evidence, cross-check entity-value "
                "bindings, and perform any required reasoning before answering. "
                "Return only the final answer. Do not include explanations, prefixes, markdown, bullet points, "
                "citations, or reasoning traces. If the evidence is insufficient after careful search, return Unknown."
            ),
            "user_prompt_template": (
                "Context:\n{context}\n\n"
                "Question:\n{question}\n\n"
                "Return the final answer only. Preserve exact digits, capitalization, and spelling when relevant."
            ),
        }

    async def evaluate_model(self, prompt: Dict) -> str:
        question = prompt.get("question", "") or ""
        if not question:
            return "Missing required input data"

        context_for_llm = self._select_context(prompt, question)
        messages = [
            {"role": "system", "content": self.prompts["system_prompt"]},
            {
                "role": "user",
                "content": self.prompts["user_prompt_template"].format(context=context_for_llm, question=question),
            },
        ]
        response_raw = await self._create_chat_completion(
            messages=messages,
            enable_thinking=getattr(self, "enable_thinking", False),
        )
        self.last_trace = {"agent": self.__class__.__name__, "path": "hybrid_retrieval"}
        return self.finalize_answer(response_raw.strip())

    def _select_context(self, prompt: Dict, question: str) -> str:
        context_data = prompt.get("context_data", {}) or {}
        context_str = prompt.get("context", "") or ""
        if not context_data and context_str:
            return context_str

        full_context = self._build_full_context(context_data)
        if int(full_context["total_tokens"]) <= self.full_context_threshold_tokens:
            return str(full_context["text"])

        return self._retrieve_with_hybrid(question, context_data)["evidence_text"]

    def _build_full_context(self, context_data: Dict) -> Dict[str, Union[str, int]]:
        files = context_data.get("files", []) or []
        parts: List[str] = []
        total_tokens = 0
        for i, file_data in enumerate(files):
            filename = file_data.get("filename", f"doc_{i}")
            text = file_data.get("modified_content", "") or ""
            parts.append(f"=== {filename} ===\n{text}")
            total_tokens += len(self.encode_text_to_tokens(text))
        return {"text": "\n\n".join(parts), "total_tokens": total_tokens}

    def _retrieve_with_hybrid(self, question: str, context_data: Dict, queries: Optional[List[str]] = None) -> Dict[str, str]:
        files = context_data.get("files", []) or []
        chunks: List[Dict] = []
        for doc_id, file_data in enumerate(files):
            content = file_data.get("modified_content", "") or ""
            filename = file_data.get("filename", f"doc_{doc_id}")
            chunks.extend(self._chunk_document(content, filename, doc_id))
        if not chunks:
            return {"evidence_text": "No relevant content found."}

        search_queries = queries or [question]
        bm25_indices = set()
        vector_indices = set()
        for query in search_queries:
            keywords = self._get_keywords(query)
            bm25_scores = self._score_bm25(chunks, keywords)
            bm25_top = sorted(bm25_scores.items(), key=lambda x: x[1], reverse=True)[: self.top_k_bm25]
            bm25_indices.update(idx for idx, score in bm25_top if score > 0)

            query_emb = cast(List[float], self._get_embeddings(query))
            if query_emb:
                candidate_indices = list(range(len(chunks))) if len(chunks) <= 300 else [idx for idx, _ in bm25_top[:300]]
                texts = [chunks[i]["text"] for i in candidate_indices]
                doc_embs = cast(List[List[float]], self._get_embeddings_in_batches(texts, batch_size=64))
                scored = []
                for idx, emb in zip(candidate_indices, doc_embs):
                    if emb:
                        scored.append((idx, self._cosine_similarity(query_emb, cast(List[float], emb))))
                vector_indices.update(idx for idx, _ in sorted(scored, key=lambda x: x[1], reverse=True)[: self.top_k_vector])

        merged_indices = list(bm25_indices | vector_indices)[:80]
        rerank_inputs: List[str] = []
        chunk_map: Dict[str, int] = {}
        for idx in merged_indices:
            chunk = chunks[idx]
            text = self._trim_retrieval_text(
                f"[Doc {chunk['doc_id']} | File {chunk['filename']} | Chunk {chunk['chunk_id']}]\n{chunk['text']}"
            )
            rerank_inputs.append(text)
            chunk_map[text] = idx

        if not rerank_inputs:
            return {"evidence_text": "No relevant content found."}

        reranked = self._rerank_documents(question, rerank_inputs, top_n=self.rerank_top_n * 2)
        threshold = (max((r.get("relevance_score", 0) for r in reranked), default=0) or 0) * 0.12

        evidence_blocks: List[str] = []
        total_tokens = 0
        added = set()
        for rank_idx, result in enumerate(reranked):
            score = result.get("relevance_score", 0)
            if score < threshold and len(evidence_blocks) >= 3:
                break
            if len(evidence_blocks) >= self.rerank_top_n:
                break

            snippet = result.get("document", "")
            orig_idx = chunk_map.get(snippet)
            for idx in self._with_neighbors(orig_idx, chunks) if orig_idx is not None else []:
                if idx in added:
                    continue
                chunk = chunks[idx]
                prefix = f"[Rank {rank_idx + 1} | Score {score:.4f} | Chunk {chunk['chunk_id']}]"
                block = f"{prefix}\n{chunk['text']}"
                block_tokens = len(self.encode_text_to_tokens(block))
                if total_tokens + block_tokens > self.max_evidence_tokens:
                    continue
                evidence_blocks.append(block)
                total_tokens += block_tokens
                added.add(idx)

        return {"evidence_text": "\n\n---\n\n".join(evidence_blocks) if evidence_blocks else "No relevant content found."}

    def _with_neighbors(self, idx: int, chunks: List[Dict]) -> List[int]:
        result = [idx]
        for neighbor_idx in (idx - 1, idx + 1):
            if 0 <= neighbor_idx < len(chunks) and chunks[neighbor_idx]["doc_id"] == chunks[idx]["doc_id"]:
                result.append(neighbor_idx)
        return result

    def _chunk_document(self, content: str, filename: str, doc_id: int) -> List[Dict]:
        tokens = self.encode_text_to_tokens(content)
        chunks: List[Dict] = []
        step = max(1, self.chunk_size_tokens - self.chunk_overlap_tokens)
        for chunk_id, start in enumerate(range(0, len(tokens), step)):
            end = start + self.chunk_size_tokens
            text = self.decode_tokens(tokens[start:end]).strip()
            if len(text) >= 20:
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
        return chunks

    def _score_bm25(self, chunks: List[Dict], keywords: List[str]) -> Dict[int, float]:
        if not keywords or not chunks:
            return {}
        tokenized_chunks = [self._tokenize(chunk["text"]) for chunk in chunks]
        bm25 = BM25Okapi(tokenized_chunks)
        scores: Dict[int, float] = {i: 0.0 for i in range(len(chunks))}
        for keyword in keywords:
            weight = 2.0 if ("-" in keyword or any(c.isdigit() for c in keyword)) else 1.0
            for idx, score in enumerate(bm25.get_scores(self._tokenize(keyword))):
                if score > 0:
                    scores[idx] += score * weight
        return scores

    def _get_keywords(self, question: str) -> List[str]:
        stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by",
            "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
            "will", "would", "could", "should", "may", "might", "can", "what", "when", "where", "why", "how",
            "who", "which", "that", "this", "these", "those",
        }
        terms: List[str] = []
        terms.extend(re.findall(r"\b[A-Z0-9]+-[A-Z0-9-]+[A-Z0-9]\b", question))
        terms.extend(re.findall(r"\b\d+\b", question))
        terms.extend(re.findall(r"\b20[0-9]{2}\b", question))
        terms.extend(
            re.findall(
                r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b",
                question.lower(),
            )
        )
        terms.extend(w for w in re.findall(r"\b[a-zA-Z]+\b", question.lower()) if w not in stop_words and len(w) > 2)
        expanded = []
        for term in terms:
            expanded.append(term.lower())
            if "-" in term:
                expanded.extend(part.lower() for part in term.split("-") if part)
        return list(dict.fromkeys(expanded))[:30]

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"[a-zA-Z0-9]+(?:-[a-zA-Z0-9]+)*", text.lower())

    def _get_embeddings(self, input_data: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        try:
            if isinstance(input_data, str):
                response = self.client.embeddings.create(model=self.embedding_model, input=self._trim_retrieval_text(input_data))
                return response.data[0].embedding
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=[self._trim_retrieval_text(text) for text in input_data],
            )
            return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]
        except Exception as exc:
            print(f"Error getting embeddings: {exc}")
            return [] if isinstance(input_data, str) else [[] for _ in input_data]

    def _get_embeddings_in_batches(self, texts: List[str], batch_size: int = 64) -> List[List[float]]:
        results: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            batch_embeddings = self._get_embeddings(texts[i : i + batch_size])
            results.extend(cast(List[List[float]], batch_embeddings))
        return results

    def _rerank_documents(self, query: str, documents: List[str], top_n: int = 8) -> List[Dict]:
        if not documents:
            return []
        try:
            response = requests.post(
                f"{self.base_url}/rerank",
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json={
                    "model": self.rerank_model,
                    "query": query,
                    "documents": documents,
                    "top_n": top_n,
                    "return_documents": True,
                },
                timeout=30,
            )
            response.raise_for_status()
            return response.json().get("results", [])
        except Exception as exc:
            print(f"Error during reranking: {exc}")
            return [{"document": document, "relevance_score": 0.0} for document in documents[:top_n]]

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        return dot / (na * nb) if na > 0 and nb > 0 else 0.0

    def _trim_retrieval_text(self, text: str) -> str:
        return (text or "")[: self.max_ecnu_retrieval_chars]
