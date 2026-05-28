import base64
import binascii
import json
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from core.ecnu_constants import ECNU_PLUS_MODEL_NAME
from .hybrid_retrieval_agent import HybridRetrievalAgent


class ToolAugmentedAgent(HybridRetrievalAgent):
    """
    Productized agent: retrieval + structured extraction + deterministic tools.
    LLMs locate and structure evidence; Python handles exact arithmetic/date/string work.
    """

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        super().__init__(api_key=api_key, base_url=base_url)
        self.last_trace: Dict = {}

    async def evaluate_model(self, prompt: Dict) -> str:
        question = prompt.get("question", "") or ""
        if not question:
            return "Missing required input data"

        task_type = self._classify_task(question)
        if task_type == "general":
            return await self._fallback(prompt, "unsupported_task_type", task_type)

        context = self._select_context(prompt, question)
        extraction = await self._extract_structured_evidence(question, context, task_type)

        answer = self._solve_with_tools(question, context, task_type, extraction)
        if answer:
            self.last_trace = {
                "agent": self.__class__.__name__,
                "path": "tool_augmented",
                "task_type": task_type,
                "extraction": extraction,
                "tool_answer": answer,
                "fallback_reason": None,
            }
            return self.finalize_answer(answer)

        return await self._fallback(prompt, "tool_execution_failed", task_type, extraction)

    async def _fallback(
        self,
        prompt: Dict,
        reason: str,
        task_type: str,
        extraction: Optional[Dict] = None,
    ) -> str:
        response = await super().evaluate_model(prompt)
        self.last_trace = {
            "agent": self.__class__.__name__,
            "path": "fallback_to_hybrid",
            "task_type": task_type,
            "extraction": extraction or {},
            "fallback_reason": reason,
        }
        return response

    def _classify_task(self, question: str) -> str:
        q = question.lower()
        if any(term in q for term in ("base64", "hex", "caesar", "cipher", "decode", "encoded")):
            return "encoding"
        if any(term in q for term in ("day of the week", "weekday", "date", "deadline", "delivery", "launch")):
            return "date_time"
        if any(term in q for term in ("character", "substring", "occurrence", "count", "position", "index", "string")):
            return "string_analysis"
        if any(
            term in q
            for term in (
                "difference",
                "sum",
                "total",
                "product",
                "ratio",
                "divide",
                "full simulation",
                "how many",
                "magnitude",
                "cycles",
            )
        ):
            return "computation"
        return "general"

    async def _extract_structured_evidence(self, question: str, context: str, task_type: str) -> Dict:
        prompt = (
            "你是结构化证据抽取器。请只输出 JSON，不要解释。\n"
            "字段要求：task_type, evidence_items, operation, constraints。\n"
            "evidence_items 中每项包含 entity, value, unit_or_type, source_snippet。\n\n"
            f"任务类型：{task_type}\n"
            f"问题：{question}\n\n"
            f"上下文：\n{context[:24000]}\n\n"
            "JSON："
        )
        response = await self._create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            model=ECNU_PLUS_MODEL_NAME,
            enable_thinking=False,
            response_format={"type": "json_object"},
        )
        return self._parse_json_object(response)

    def _parse_json_object(self, text: str) -> Dict:
        if not text:
            return {}
        clean = text.strip()
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", clean, re.DOTALL)
        if match:
            clean = match.group(1)
        try:
            data = json.loads(clean)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _solve_with_tools(self, question: str, context: str, task_type: str, extraction: Dict) -> Optional[str]:
        if task_type == "computation":
            return self._solve_computation(question, context, extraction)
        if task_type == "date_time":
            return self._solve_date_time(question, context, extraction)
        if task_type == "string_analysis":
            return self._solve_string_analysis(question, context, extraction)
        if task_type == "encoding":
            return self._solve_encoding(question, context, extraction)
        return None

    def _evidence_values(self, extraction: Dict) -> List[str]:
        values = []
        for item in extraction.get("evidence_items", []) or []:
            if isinstance(item, dict) and item.get("value") is not None:
                values.append(str(item["value"]))
        return values

    def _numbers_from(self, question: str, context: str, extraction: Dict) -> List[int]:
        source = "\n".join(self._evidence_values(extraction)) + "\n" + context + "\n" + question
        return [int(n) for n in re.findall(r"(?<![A-Za-z])-?\d{2,}(?![A-Za-z])", source)]

    def _solve_computation(self, question: str, context: str, extraction: Dict) -> Optional[str]:
        q = question.lower()
        numbers = self._numbers_from(question, context, extraction)
        if len(numbers) < 2:
            return None

        if any(term in q for term in ("full simulation", "how many full", "can be completed", "divide", "per")):
            divisor = next((n for n in numbers[1:] if n != 0), None)
            return str(numbers[0] // divisor) if divisor else None
        if "difference" in q or "absolute" in q:
            return str(abs(numbers[0] - numbers[1]))
        if "sum" in q or ("total" in q and "per" not in q):
            return str(sum(numbers[:2]))
        if "product" in q or "multiply" in q:
            return str(numbers[0] * numbers[1])
        return None

    def _solve_date_time(self, question: str, context: str, extraction: Dict) -> Optional[str]:
        source = "\n".join(self._evidence_values(extraction)) + "\n" + context + "\n" + question
        date_match = re.search(
            r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),\s*(\d{4})\b",
            source,
            re.IGNORECASE,
        )
        if not date_match:
            return None

        date = datetime.strptime(" ".join(date_match.groups()), "%B %d %Y")
        q = question.lower()
        offset_match = re.search(r"(\d+)\s+days?\s+(before|after|prior to)", source, re.IGNORECASE)
        if offset_match and any(term in q for term in ("delivery", "deadline", "completed by", "before", "after")):
            days = int(offset_match.group(1))
            direction = offset_match.group(2).lower()
            date = date - timedelta(days=days) if direction in ("before", "prior to") else date + timedelta(days=days)

        if any(term in q for term in ("day of the week", "weekday", "what day")):
            return date.strftime("%A")
        return f"{date.strftime('%B')} {date.day}, {date.year}"

    def _solve_string_analysis(self, question: str, context: str, extraction: Dict) -> Optional[str]:
        q = question.lower()
        values = self._evidence_values(extraction)
        target = self._quoted_value(question) or self._quoted_value(context)
        haystack = values[0] if values else context
        if not target:
            return None

        if "count" in q or "occurrence" in q or "how many" in q:
            return str(haystack.count(target))
        if "position" in q or "index" in q:
            idx = haystack.find(target)
            return str(idx) if idx >= 0 else None
        if "length" in q:
            return str(len(target))
        return None

    def _solve_encoding(self, question: str, context: str, extraction: Dict) -> Optional[str]:
        q = question.lower()
        values = self._evidence_values(extraction)
        encoded = values[0] if values else self._quoted_value(question) or self._quoted_value(context)
        if not encoded:
            return None
        encoded = encoded.strip()

        try:
            if "base64" in q:
                return base64.b64decode(encoded).decode("utf-8").strip()
            if "hex" in q:
                return bytes.fromhex(encoded).decode("utf-8").strip()
            if "caesar" in q:
                shift_match = re.search(r"shift(?:ed)?\s*(?:by)?\s*(\d+)", q)
                shift = int(shift_match.group(1)) if shift_match else 3
                return self._caesar_decode(encoded, shift)
        except (binascii.Error, UnicodeDecodeError, ValueError):
            return None
        return None

    def _quoted_value(self, text: str) -> Optional[str]:
        match = re.search(r"['\"]([^'\"]{1,500})['\"]", text)
        return match.group(1) if match else None

    def _caesar_decode(self, text: str, shift: int) -> str:
        chars = []
        for char in text:
            if "a" <= char <= "z":
                chars.append(chr((ord(char) - ord("a") - shift) % 26 + ord("a")))
            elif "A" <= char <= "Z":
                chars.append(chr((ord(char) - ord("A") - shift) % 26 + ord("A")))
            else:
                chars.append(char)
        return "".join(chars).strip()
