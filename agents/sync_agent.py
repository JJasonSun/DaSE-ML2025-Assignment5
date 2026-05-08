import json
import os
import re
from typing import Dict, List, Optional

from dotenv import load_dotenv

from core.ecnu_constants import DEFAULT_ECNU_BASE_URL, ECNU_MAIN_MODEL_NAME
from .base_agent import ModelProvider


class SyncRetrievalAgent(ModelProvider):
    """
    Keyword-based retrieval agent with sentence-level content extraction.
    Supports both single-mode (plain context string) and multi-mode (structured context_data).
    """

    def __init__(self, api_key: str, base_url: str):
        load_dotenv()
        api_key = api_key or os.getenv("ECNU_API_KEY") or ""
        base_url = (base_url or os.getenv("ECNU_BASE_URL") or DEFAULT_ECNU_BASE_URL).rstrip("/")
        super().__init__(api_key=api_key, base_url=base_url)
        self.model_name = os.getenv("MODEL_NAME") or ECNU_MAIN_MODEL_NAME

        self.max_tokens_per_request = 512
        self.top_k_files = 5

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

    async def evaluate_model(self, prompt: Dict) -> str:
        try:
            context_data = prompt.get("context_data")
            context = prompt.get("context")
            question = prompt.get("question", "")

            if not question:
                return "Missing required input data"

            if context_data:
                selected_content = self._retrieve_content(context_data, question)
            elif isinstance(context, str) and context.strip():
                selected_content = self._truncate_text(context, self.max_tokens_per_request)
            else:
                return "Missing required input data"

            if not selected_content or selected_content == "No relevant content found.":
                return "Unknown"

            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a strict answer extractor. Return ONLY the final answer — no explanation, "
                        "no restating the question, no prefixes, no suffixes, no numbering, no bullet points, "
                        "no quotes, no code blocks, no extra punctuation.\n"
                        "1. If the question involves date calculation or weekday derivation, compute it from "
                        "the dates provided in the context.\n"
                        "2. Output dates/weekdays in English format (e.g., Thursday, December 25, 2031).\n"
                        "3. For numeric answers, output Arabic numerals directly.\n"
                        "4. If the answer is a word, phrase, or number, output only that content.\n"
                        "5. If the answer cannot be determined, output: Unknown\n"
                        "6. The final answer must be in English."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Answer the question based ONLY on the context below. Output only the answer, no explanation.\n\n"
                        f"Context:\n{selected_content}\n\n"
                        f"Question: {question}\n\n"
                        f"Answer:"
                    ),
                },
            ]

            response = await self._create_chat_completion(
                messages=messages,
                temperature=1,
                top_p=0.95,
                max_tokens=8000,
                timeout=180,
                enable_thinking=True,
                thinking_budget_tokens=4000,
            )

            if response and response.strip():
                return self.finalize_answer(self._extract_answer(response))
            else:
                return "Unknown"

        except Exception as e:
            return f"Error: {str(e)[:50]}"

    def _retrieve_content(self, context_data: Dict, question: str) -> str:
        files = context_data["files"]
        keywords = self._get_keywords(question)

        relevant_files = []
        for file_data in files:
            filename = file_data["filename"]
            content = file_data["modified_content"]
            content_lower = content.lower()

            keyword_matches = 0
            exact_matches = 0

            for keyword in keywords:
                kw_lower = keyword.lower()
                keyword_matches += content_lower.count(kw_lower)
                if re.search(r"\b" + re.escape(kw_lower) + r"\b", content_lower):
                    exact_matches += 1

            if keyword_matches > 0 or exact_matches > 0:
                content_length = len(content.split())
                density_score = keyword_matches / max(content_length, 1) * 1000
                exact_bonus = exact_matches * 100
                final_score = density_score + exact_bonus
                relevant_files.append((filename, final_score, file_data))

        relevant_files.sort(key=lambda x: x[1], reverse=True)

        content_parts = []
        total_tokens = 0

        for filename, score, file_data in relevant_files[: self.top_k_files]:
            if total_tokens >= self.max_tokens_per_request:
                break

            content = file_data["modified_content"]
            extracted_content = self._extract_relevant_content(content, keywords, question)

            content_tokens = len(self.encode_text_to_tokens(extracted_content))
            if total_tokens + content_tokens > self.max_tokens_per_request:
                remaining = self.max_tokens_per_request - total_tokens
                if remaining > 200:
                    extracted_content = self._truncate_text(extracted_content, remaining)
                    content_parts.append(f"=== {filename} ===\n{extracted_content}")
                break
            else:
                content_parts.append(f"=== {filename} ===\n{extracted_content}")
                total_tokens += content_tokens

        return "\n\n".join(content_parts) if content_parts else "No relevant content found."

    def _get_keywords(self, question: str) -> List[str]:
        stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by",
            "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
            "will", "would", "could", "should", "may", "might", "can", "what", "when", "where", "why", "how",
            "who", "which", "that", "this", "these", "those",
        }

        keywords = []

        codes = re.findall(r"\b[A-Z0-9]+-[A-Z0-9-]+[A-Z0-9]\b", question)
        keywords.extend([code.lower() for code in codes])

        project_codes = re.findall(r"\b[A-Z]-\d+-[A-Za-z]+\b", question)
        keywords.extend([code.lower() for code in project_codes])

        simple_codes = re.findall(r"\b[A-Z]{2,}-[A-Z0-9]+\b", question)
        keywords.extend([code.lower() for code in simple_codes])

        numbers = re.findall(r"\b\d+\b", question)
        keywords.extend(numbers)

        years = re.findall(r"\b20[2-9]\d\b", question)
        keywords.extend(years)

        months = re.findall(
            r"\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\b",
            question.lower(),
        )
        keywords.extend(months)

        words = re.findall(r"\b[a-zA-Z]+\b", question.lower())
        keywords.extend([w for w in words if w not in stop_words and len(w) > 2])

        return list(dict.fromkeys(keywords))[:15]

    def _extract_relevant_content(self, content: str, keywords: List[str], question: str) -> str:
        sentences = re.split(r"[.!?]+", content)
        scored_sentences = []

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:
                continue

            sentence_lower = sentence.lower()
            score = 0

            for keyword in keywords:
                kw_lower = keyword.lower()
                if kw_lower in sentence_lower:
                    if re.search(r"\b" + re.escape(kw_lower) + r"\b", sentence_lower):
                        score += 3
                    else:
                        score += 1

            if re.search(r"\b\d+\b", sentence):
                score += 1

            if re.search(r"\b(scheduled|date|day|week|days|between|from|to|on|in)\b", sentence_lower):
                score += 1

            if score > 0:
                scored_sentences.append((sentence, score))

        if not scored_sentences:
            return self._get_best_chunk(content, keywords)

        scored_sentences.sort(key=lambda x: x[1], reverse=True)

        result_sentences = []
        total_tokens = 0
        max_tokens = self.max_tokens_per_request // 6

        for sentence, score in scored_sentences[:10]:
            sentence_tokens = len(self.encode_text_to_tokens(sentence))
            if total_tokens + sentence_tokens > max_tokens:
                break
            result_sentences.append(sentence)
            total_tokens += sentence_tokens

        if result_sentences:
            result_with_context = []
            for target_sentence in result_sentences[:5]:
                sentence_pos = content.find(target_sentence)
                if sentence_pos != -1:
                    start = max(0, sentence_pos - 200)
                    end = min(len(content), sentence_pos + len(target_sentence) + 200)
                    context_chunk = content[start:end].strip()
                    result_with_context.append(context_chunk)
            return "\n\n".join(result_with_context)

        return "\n".join(result_sentences)

    def _get_best_chunk(self, content: str, keywords: List[str]) -> str:
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]

        if not paragraphs:
            return content[:1500]

        scored_paras = []
        for para in paragraphs:
            para_lower = para.lower()
            score = sum(para_lower.count(kw.lower()) for kw in keywords)
            if score > 0:
                scored_paras.append((para, score))

        if not scored_paras:
            return "\n\n".join(paragraphs[:2])

        scored_paras.sort(key=lambda x: x[1], reverse=True)

        result_paras = []
        total_tokens = 0
        max_tokens = self.max_tokens_per_request // 6

        for para, score in scored_paras:
            para_tokens = len(self.encode_text_to_tokens(para))
            if total_tokens + para_tokens > max_tokens:
                break
            result_paras.append(para)
            total_tokens += para_tokens

        return "\n\n".join(result_paras) if result_paras else scored_paras[0][0]

    def _truncate_text(self, text: str, max_tokens: int) -> str:
        tokens = self.encode_text_to_tokens(text)
        if len(tokens) <= max_tokens:
            return text
        return self.decode_tokens(tokens[:max_tokens])

    def generate_prompt(self, **kwargs) -> Dict:
        return {
            "context": kwargs.get("context"),
            "context_data": kwargs.get("context_data"),
            "question": kwargs.get("question"),
        }
