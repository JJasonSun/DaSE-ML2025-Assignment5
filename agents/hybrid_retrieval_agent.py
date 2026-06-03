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

    AGENT_PROFILE = {
        "positioning": "Retrieval-augmented baseline using BM25, ECNU dense embeddings, ECNU rerank, generic query expansion, dynamic neighbor chunk expansion, and retrieval trace metadata before final LLM answering.",
        "expected_strengths": "Designed for fairer multi-document evidence recall, better identifier/query coverage, and retrieval diagnostics without adding deterministic tools.",
        "expected_limits": "Still intentionally relies on the LLM for exact arithmetic, date reasoning, string operations, encoding, and final answer formatting; it should remain a retrieval baseline rather than a weak ToolAugmentedAgent.",
        "analysis_focus": "Separate retrieval failures from post-retrieval reasoning or formatting failures. Use retrieval trace fields such as selected_queries, retrieved_files, rerank_scores, evidence_block_count, and neighbor_radius. Suggestions should target query expansion, rerank, chunking, context selection, and final-answer prompting, not calculators or decoders.",
    }

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
        self._last_retrieval_trace: Dict = {}

        self.prompts = {
            "system_prompt": (
                "You are a rigorous evidence retrieval and reasoning agent for Needle-in-a-Haystack tasks. "
                "Use only the supplied context. Internally locate all relevant evidence, cross-check entity-value "
                "bindings, and perform any required reasoning before answering. For calculation, date, decoding, "
                "or string-count questions, first identify the exact source values in the context, preserve every "
                "digit and character, and only then derive the final answer. "
                "Return only the final answer. Do not include explanations, prefixes, markdown, bullet points, "
                "citations, or reasoning traces. If the evidence is insufficient after careful search, return Unknown."
            ),
            "user_prompt_template": (
                "Context:\n{context}\n\n"
                "Question:\n{question}\n\n"
                "Return the final answer only. Preserve exact digits, capitalization, and spelling when relevant. "
                "If multiple evidence values are required, use only values that are explicitly present in the context."
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
        self.last_trace = {
            "agent": self.__class__.__name__,
            "path": "hybrid_retrieval",
            **self._last_retrieval_trace,
        }
        return self.finalize_answer(response_raw.strip())

    def _select_context(self, prompt: Dict, question: str) -> str:
        context_data = prompt.get("context_data", {}) or {}
        context_str = prompt.get("context", "") or ""
        if not context_data and context_str:
            return context_str

        full_context = self._build_full_context(context_data)
        if int(full_context["total_tokens"]) <= self.full_context_threshold_tokens:
            self._last_retrieval_trace = {
                "retrieval_mode": "full_context",
            "context_chars": len(str(full_context["text"])),
            "selected_queries": [question],
            "evidence_block_count": len(context_data.get("files", []) or []),
            "retrieved_files": [file_data.get("filename", "") for file_data in context_data.get("files", []) or []],
            "embedding_model": self.embedding_model,
            "rerank_model": self.rerank_model,
        }
            return str(full_context["text"])

        retrieval = self._retrieve_with_hybrid(question, context_data, queries=self._build_retrieval_queries(question))
        self._last_retrieval_trace = retrieval.get("trace", {})
        return retrieval["evidence_text"]

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

    def _retrieve_with_hybrid(self, question: str, context_data: Dict, queries: Optional[List[str]] = None) -> Dict[str, Union[str, Dict]]:
        files = context_data.get("files", []) or []
        chunks: List[Dict] = []
        for doc_id, file_data in enumerate(files):
            content = file_data.get("modified_content", "") or ""
            filename = file_data.get("filename", f"doc_{doc_id}")
            chunks.extend(self._chunk_document(content, filename, doc_id))
        if not chunks:
            return {
                "evidence_text": "No relevant content found.",
                "trace": {
                    "retrieval_mode": "hybrid",
                    "selected_queries": queries or [question],
                    "evidence_block_count": 0,
                    "retrieved_files": [],
                    "rerank_scores": [],
                    "context_chars": 0,
                },
            }

        search_queries = queries or self._build_retrieval_queries(question)
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

        target_blocks = self._dynamic_rerank_top_n(question)
        neighbor_radius = self._dynamic_neighbor_radius(question)
        reranked = self._rerank_documents(question, rerank_inputs, top_n=target_blocks * 2)
        threshold = (max((r.get("relevance_score", 0) for r in reranked), default=0) or 0) * 0.12

        evidence_blocks: List[str] = []
        total_tokens = 0
        added = set()
        retrieved_files = []
        rerank_scores = []
        for rank_idx, result in enumerate(reranked):
            score = result.get("relevance_score", 0)
            if score < threshold and len(evidence_blocks) >= 3:
                break
            if len(evidence_blocks) >= target_blocks:
                break

            snippet = result.get("document", "")
            orig_idx = chunk_map.get(snippet)
            if orig_idx is not None:
                rerank_scores.append(round(float(score or 0), 4))
            for idx in self._with_neighbors(orig_idx, chunks, radius=neighbor_radius) if orig_idx is not None else []:
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
                retrieved_files.append(str(chunk["filename"]))

        evidence_text = "\n\n---\n\n".join(evidence_blocks) if evidence_blocks else "No relevant content found."
        return {
            "evidence_text": evidence_text,
            "trace": {
                "retrieval_mode": "hybrid",
            "selected_queries": search_queries,
            "evidence_block_count": len(evidence_blocks),
            "retrieved_files": list(dict.fromkeys(retrieved_files)),
            "rerank_scores": rerank_scores[:target_blocks],
            "context_chars": len(evidence_text),
            "neighbor_radius": neighbor_radius,
            "target_evidence_blocks": target_blocks,
            "embedding_model": self.embedding_model,
            "rerank_model": self.rerank_model,
        },
    }

    def _with_neighbors(self, idx: int, chunks: List[Dict], radius: int = 1) -> List[int]:
        result = [idx]
        for offset in range(1, radius + 1):
            for neighbor_idx in (idx - offset, idx + offset):
                if 0 <= neighbor_idx < len(chunks) and chunks[neighbor_idx]["doc_id"] == chunks[idx]["doc_id"]:
                    result.append(neighbor_idx)
        return result

    def _build_retrieval_queries(self, question: str) -> List[str]:
        queries = [question]
        queries.extend(re.findall(r"\[([A-Za-z0-9_-]{3,120})\]", question))
        queries.extend(re.findall(r"['\"]([^'\"]{3,100})['\"]", question))
        queries.extend(re.findall(r"\b[A-Z][A-Za-z0-9_]*(?:-[A-Za-z0-9_]+)+\b", question))
        queries.extend(re.findall(r"\b[A-Z][A-Z0-9_]{5,}\b", question))
        queries.extend(re.findall(r"\b[A-Za-z]+[A-Za-z0-9_-]*\d[A-Za-z0-9_-]*\b", question))
        queries.extend(self._capitalized_phrases(question))

        if self._looks_multi_evidence(question):
            queries.extend(self._operation_queries(question))

        ordered: List[str] = []
        seen = set()
        for query in queries:
            clean = query.strip(" .,:;")
            if clean and clean.lower() not in seen:
                ordered.append(clean)
                seen.add(clean.lower())
        return ordered[:8]

    def _capitalized_phrases(self, question: str) -> List[str]:
        phrases = re.findall(r"\b(?:[A-Z][a-zA-Z0-9_-]+(?:\s+|$)){2,5}", question)
        cleaned = []
        for phrase in phrases:
            phrase = re.sub(r"^(For|Using|Based|According|During|In|From)\s+", "", phrase.strip())
            if len(phrase) >= 6:
                cleaned.append(phrase)
        return cleaned

    def _operation_queries(self, question: str) -> List[str]:
        q = question.lower()
        queries = []
        if any(term in q for term in ("difference", "subtract", "minus")):
            queries.append("difference subtract minus")
        if any(term in q for term in ("multiply", "multiplier", "product")):
            queries.append("multiplier product")
        if any(term in q for term in ("divide", "division", "divisor", "ratio", "quotient")):
            queries.append("divisor quotient")
        if any(term in q for term in ("date", "days", "weekday", "launch", "deadline")):
            queries.append("date scheduled launch deadline")
        if any(term in q for term in ("encoded", "decode", "cipher", "base64", "hex")):
            queries.append("encoded string payload method")
        return queries

    def _looks_multi_evidence(self, question: str) -> bool:
        q = question.lower()
        connectors = len(re.findall(r"\b(and|then|finally|between|from|minus|divided by|multiplied by)\b", q))
        operations = len(re.findall(r"\b(subtract|minus|multiply|multiplied|divide|divided|difference|product|divisor)\b", q))
        quoted_or_ids = len(re.findall(r"['\"][^'\"]+['\"]|\b[A-Z0-9]+-[A-Z0-9-]+\b", question))
        return connectors >= 2 or operations >= 2 or quoted_or_ids >= 2

    def _dynamic_rerank_top_n(self, question: str) -> int:
        if self._looks_multi_evidence(question):
            return min(self.rerank_top_n + 4, 14)
        return self.rerank_top_n

    def _dynamic_neighbor_radius(self, question: str) -> int:
        return 2 if self._looks_multi_evidence(question) else 1

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
