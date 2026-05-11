import asyncio
import json
import re
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import tiktoken
from openai import OpenAI


class ModelProvider(ABC):
    """Abstract base class for NIAH evaluation agents."""

    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = "custom-agent"
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.tokenizer = tiktoken.encoding_for_model("gpt-4")

    @abstractmethod
    async def evaluate_model(self, prompt: Dict) -> str:
        ...

    def generate_prompt(self, **kwargs) -> Dict:
        return {
            "context": kwargs.get("context"),
            "context_data": kwargs.get("context_data"),
            "question": kwargs.get("question"),
        }

    def encode_text_to_tokens(self, text: str) -> List[int]:
        return self.tokenizer.encode(text or "")

    def decode_tokens(self, tokens: List[int], context_length: Optional[int] = None) -> str:
        if context_length is not None:
            tokens = tokens[:context_length]
        return self.tokenizer.decode(tokens)

    # ---- Shared LLM call infrastructure ---- #

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
        top_p: float = 1.0,
    ) -> str:
        model_to_use = model or self.model_name
        params: Dict = {
            "model": model_to_use,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "timeout": timeout,
            "top_p": top_p,
        }
        if response_format:
            params["response_format"] = response_format

        extra_body: Dict = {}
        if enable_thinking:
            extra_body["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget_tokens}
        else:
            extra_body["thinking"] = {"type": "disabled"}

        if extra_body:
            params["extra_body"] = extra_body

        client_to_use = self.client
        max_retries = 3

        def _sync_call():
            return client_to_use.chat.completions.create(**params)

        for attempt in range(max_retries):
            try:
                try:
                    completion = await asyncio.to_thread(_sync_call)
                except AttributeError:
                    loop = asyncio.get_running_loop()
                    completion = await loop.run_in_executor(None, _sync_call)
                raw = completion.to_dict()
                return self._extract_content_from_response(raw)
            except Exception as exc:
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
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

            if isinstance(reasoning, str) and reasoning.strip():
                return reasoning.strip()

            return f"Empty response (finish_reason: {finish_reason})"
        except Exception as e:
            return f"Response parsing error: {str(e)[:80]}"

    def compress_final_answer(self, response: str) -> str:
        """
        压缩模型输出，只保留最终答案本身。

        适用于包含解释、推理过程、Markdown、JSON 包装等情况。
        """
        if not isinstance(response, str):
            response = "" if response is None else str(response)

        text = response.strip()
        if not text:
            return ""

        text = self._strip_code_fences(text)
        text = self._extract_json_answer(text)
        text = self._extract_labeled_answer(text)
        text = self._strip_explanatory_suffix(text)
        text = self._normalize_answer(text)
        return text.strip()

    def finalize_answer(self, response: str) -> str:
        """
        仅在输出看起来包含解释、格式化包装或多余内容时，才回退到答案压缩。

        对于已经足够干净的单行最终答案，尽量保留原始表达，仅做轻量归一化。
        """
        if not isinstance(response, str):
            response = "" if response is None else str(response)

        text = response.strip()
        if not text:
            return ""

        if self._looks_like_noisy_answer(text):
            return self.compress_final_answer(text)

        return self._normalize_answer(text).strip()

    def _strip_code_fences(self, text: str) -> str:
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return text

    def _extract_json_answer(self, text: str) -> str:
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                for key in ("answer", "final_answer", "result", "output", "content"):
                    value = data.get(key)
                    if isinstance(value, str) and value.strip():
                        return value.strip()
                for value in data.values():
                    if isinstance(value, str) and value.strip():
                        return value.strip()
        except Exception:
            pass

        match = re.search(
            r'"(?:answer|final_answer|result|output|content)"\s*:\s*"([^"]*?)"',
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if match:
            return match.group(1).strip()
        return text

    def _extract_labeled_answer(self, text: str) -> str:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return text

        label_patterns = (
            r"^(?:最终答案|答案|answer|final\s*answer|result)\s*[:：]\s*(.+)$",
            r"^(?:the\s*)?answer\s*(?:is)?\s*[:：]\s*(.+)$",
        )

        for line in lines:
            for pattern in label_patterns:
                match = re.match(pattern, line, re.IGNORECASE)
                if match:
                    candidate = match.group(1).strip()
                    if candidate:
                        return candidate

        return lines[0]

    def _strip_explanatory_suffix(self, text: str) -> str:
        separators = (
            "\n",
            " because ",
            " since ",
            " because:",
            " due to ",
            " reason:",
            " analysis:",
            " explanation:",
            " 推理",
            " 分析",
            " 因为",
            " 由于",
            " 解释",
        )

        lower_text = text.lower()
        for separator in separators:
            idx = lower_text.find(separator.lower())
            if idx > 0:
                return text[:idx].strip()
        return text

    def _normalize_answer(self, text: str) -> str:
        text = text.strip()
        text = re.sub(r"^(?:[\-\*•]+\s*|\d+[\.\)]\s+)", "", text)
        text = text.strip("\"'“”‘’`")
        text = text.rstrip(".,;:!?！？。")
        text = re.sub(r"\s+", " ", text)
        return text

    def _looks_like_noisy_answer(self, text: str) -> bool:
        lowered = text.lower()
        if "```" in text:
            return True
        if "\n" in text:
            return True
        if lowered.startswith("{") or lowered.startswith("["):
            return True
        if re.match(r"^(?:最终答案|答案|answer|final\s*answer|result)\s*[:：]", text, re.IGNORECASE):
            return True
        if re.match(r"^(?:the\s*)?answer\s*(?:is)?\s*[:：]", text, re.IGNORECASE):
            return True
        noisy_markers = (
            " because ", " since ", " due to ", "analysis", "explanation", "reason",
            "推理", "分析", "因为", "由于", "解释",
        )
        return any(marker in lowered for marker in noisy_markers)
