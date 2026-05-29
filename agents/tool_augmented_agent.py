import ast
import base64
import binascii
import json
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

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
        if answer is not None and str(answer).strip():
            self.last_trace = {
                "agent": self.__class__.__name__,
                "path": "tool_augmented",
                "task_type": task_type,
                "extraction": extraction,
                "tool_answer": answer,
                "fallback_reason": None,
            }
            return self.finalize_answer(str(answer))

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

        string_markers = (
            "count",
            "occurrence",
            "position",
            "index",
            "substring",
            "character",
            "length",
            "reverse",
            "backwards",
            "read backwards",
            "confirmation code",
            "sum of all hexadecimal digits",
            "hexadecimal digits (0-9 only)",
        )
        if any(term in q for term in string_markers):
            return "string_analysis"

        date_markers = ("day of the week", "weekday", "date", "deadline", "delivery", "launch")
        if any(term in q for term in date_markers):
            return "date_time"

        encoding_markers = (
            "base64",
            "base32",
            "hex encoded",
            "caesar",
            "cipher",
            "decode",
            "decoded",
            "encoded signal",
            "original identifier",
        )
        if any(term in q for term in encoding_markers):
            return "encoding"

        computation_markers = (
            "difference",
            "sum",
            "total",
            "product",
            "ratio",
            "divide",
            "division",
            "multiply",
            "full simulation",
            "how many",
            "magnitude",
            "cycles",
            "integer division",
        )
        if any(term in q for term in computation_markers):
            return "computation"

        return "general"

    async def _extract_structured_evidence(self, question: str, context: str, task_type: str) -> Dict:
        prompt = (
            "You are a structured evidence extraction component for a Needle-in-a-Haystack evaluator.\n"
            "Return only valid JSON. Do not explain.\n\n"
            "Schema:\n"
            "{\n"
            '  "task_type": "computation|date_time|string_analysis|encoding",\n'
            '  "evidence_items": [\n'
            '    {"entity": "stable_machine_name", "value": "exact value", "unit_or_type": "unit/type", '
            '"source_snippet": "short verbatim evidence"}\n'
            "  ],\n"
            '  "operation": "a Python-style expression or concise operation description",\n'
            '  "constraints": ["format rules, date boundary rules, decoding method, or arithmetic rules"]\n'
            "}\n\n"
            "Extraction rules:\n"
            "- Include every number, encoded token, date, string, or shift value required to answer the question.\n"
            "- Use stable snake_case entity names when possible, for example base_mineral_reserves.\n"
            "- For computation tasks, make operation executable when possible using the entity names.\n"
            "- For encoding tasks, identify the encoded payload separately from protocol or method evidence.\n"
            "- For date tasks, preserve exact dates and offsets.\n\n"
            f"Task type: {task_type}\n"
            f"Question:\n{question}\n\n"
            f"Context:\n{context[:24000]}\n\n"
            "JSON:"
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
            data = json.loads(clean)  # type: ignore[name-defined]
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _solve_with_tools(self, question: str, context: str, task_type: str, extraction: Dict) -> Optional[str]:
        operation_answer = self._evaluate_extracted_string_operation(question, context, extraction)
        if operation_answer is not None:
            return operation_answer

        if task_type == "computation":
            return self._solve_computation(question, context, extraction)
        if task_type == "date_time":
            return self._solve_date_time(question, context, extraction)
        if task_type == "string_analysis":
            return self._solve_string_analysis(question, context, extraction)
        if task_type == "encoding":
            return self._solve_encoding(question, context, extraction)
        return None

    def _evidence_items(self, extraction: Dict) -> List[Dict]:
        items = extraction.get("evidence_items", []) or []
        return [item for item in items if isinstance(item, dict)]

    def _evidence_values(self, extraction: Dict) -> List[str]:
        values = []
        for item in self._evidence_items(extraction):
            if item.get("value") is not None:
                values.append(str(item["value"]))
        return values

    def _string_evidence(self, extraction: Dict, context: str) -> Dict[str, str]:
        variables: Dict[str, str] = {}
        for idx, item in enumerate(self._evidence_items(extraction)):
            value = item.get("value")
            if value is None or isinstance(value, bool):
                continue
            name = self._safe_identifier(str(item.get("entity") or f"value_{idx}"))
            if name:
                variables[name] = str(value)

        for name in list(variables):
            if "hash" in name:
                longer_hash = self._longest_hex_after_hash(context)
                if longer_hash and len(longer_hash) > len(variables[name]):
                    variables[name] = longer_hash

        return variables

    def _evaluate_extracted_string_operation(self, question: str, context: str, extraction: Dict) -> Optional[str]:
        operation = extraction.get("operation")
        if not isinstance(operation, str) or not operation.strip():
            return None

        variables = self._string_evidence(extraction, context)
        if not variables:
            return None

        expression = operation.strip()
        reverse_match = re.fullmatch(r"([A-Za-z_]\w*)\s*\[\s*::\s*-1\s*\]", expression)
        if reverse_match:
            value = variables.get(self._safe_identifier(reverse_match.group(1)))
            return value[::-1] if value is not None else None

        count_pattern = (
            r"([A-Za-z_]\w*)(\.lower\(\))?\.count\(\s*['\"]([^'\"]+)['\"]\s*\)"
            r"\s*-\s*"
            r"([A-Za-z_]\w*)(\.lower\(\))?\.count\(\s*['\"]([^'\"]+)['\"]\s*\)"
        )
        diff_match = re.fullmatch(rf"abs\(\s*{count_pattern}\s*\)", expression)
        if diff_match:
            force_case_sensitive = "lowercase" in question.lower()
            left_lower_call = None if force_case_sensitive else diff_match.group(2)
            right_lower_call = None if force_case_sensitive else diff_match.group(5)
            left = self._count_from_operation(variables, diff_match.group(1), left_lower_call, diff_match.group(3))
            right = self._count_from_operation(variables, diff_match.group(4), right_lower_call, diff_match.group(6))
            return str(abs(left - right)) if left is not None and right is not None else None

        single_count_match = re.fullmatch(
            r"([A-Za-z_]\w*)(\.lower\(\))?\.count\(\s*['\"]([^'\"]+)['\"]\s*\)",
            expression,
        )
        if single_count_match:
            lower_call = None if "lowercase" in question.lower() else single_count_match.group(2)
            count = self._count_from_operation(
                variables,
                single_count_match.group(1),
                lower_call,
                single_count_match.group(3),
            )
            return str(count) if count is not None else None

        caesar_match = re.fullmatch(
            r"(?:caesar_decode|decode_caesar)\(\s*([A-Za-z_]\w*)\s*,\s*(?:shift\s*=\s*)?([A-Za-z_]\w*|\d+)\s*\)",
            expression,
        )
        if caesar_match:
            encoded = variables.get(self._safe_identifier(caesar_match.group(1)))
            shift = self._resolve_shift(caesar_match.group(2), variables)
            if encoded is not None and shift is not None:
                return self._caesar_decode(encoded, shift)

        return None

    def _count_from_operation(
        self,
        variables: Dict[str, str],
        variable_name: str,
        lower_call: Optional[str],
        target: str,
    ) -> Optional[int]:
        value = variables.get(self._safe_identifier(variable_name))
        if value is None:
            return None
        if lower_call:
            value = value.lower()
            target = target.lower()
        return value.count(target)

    def _resolve_shift(self, shift_token: str, variables: Dict[str, str]) -> Optional[int]:
        if shift_token.isdigit():
            return int(shift_token)
        value = variables.get(self._safe_identifier(shift_token))
        if value is None:
            return None
        match = re.search(r"-?\d+", value)
        return int(match.group(0)) if match else None

    def _longest_hex_after_hash(self, context: str) -> Optional[str]:
        matches = re.findall(r"\bhash[^:\n]*:\s*([0-9A-Fa-f]{16,})", context)
        return max(matches, key=len) if matches else None

    def _numeric_evidence(self, extraction: Dict) -> List[Tuple[str, int]]:
        pairs: List[Tuple[str, int]] = []
        for idx, item in enumerate(self._evidence_items(extraction)):
            value = item.get("value")
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                pairs.append((str(item.get("entity") or f"value_{idx}"), int(value)))
                continue
            if isinstance(value, str):
                numbers = re.findall(r"(?<![A-Za-z])-?\d+(?![A-Za-z])", value.replace(",", ""))
                if len(numbers) == 1:
                    pairs.append((str(item.get("entity") or f"value_{idx}"), int(numbers[0])))
        return pairs

    def _numbers_from(self, question: str, context: str, extraction: Dict) -> List[int]:
        evidence_numbers = [value for _, value in self._numeric_evidence(extraction)]
        if evidence_numbers:
            return evidence_numbers
        source = "\n".join(self._evidence_values(extraction)) + "\n" + question
        return [int(n) for n in re.findall(r"(?<![A-Za-z])-?\d{1,}(?![A-Za-z])", source.replace(",", ""))]

    def _solve_computation(self, question: str, context: str, extraction: Dict) -> Optional[str]:
        answer = self._evaluate_extracted_operation(extraction)
        if answer is not None:
            return str(answer)

        q = question.lower()
        numbers = self._numbers_from(question, context, extraction)
        if len(numbers) < 2:
            return None

        if (
            "difference" in q
            and ("multiply" in q or "product" in q)
            and ("divide" in q or "integer division" in q)
            and len(numbers) >= 4
        ):
            divisor = numbers[3]
            return str(((numbers[0] - numbers[1]) * numbers[2]) // divisor) if divisor else None

        if any(term in q for term in ("full simulation", "how many full", "can be completed")):
            dividend = max(numbers)
            divisors = [n for n in numbers if 0 < n != dividend]
            return str(dividend // min(divisors)) if divisors else None

        if "divide" in q or "division" in q or "ratio" in q:
            dividend = max(numbers)
            divisors = [n for n in numbers if 0 < n != dividend]
            return str(dividend // min(divisors)) if divisors else None

        if "difference" in q or "absolute" in q:
            return str(abs(numbers[0] - numbers[1]))
        if "sum" in q or ("total" in q and "per" not in q):
            return str(sum(numbers))
        if "product" in q or "multiply" in q:
            product = 1
            for number in numbers:
                product *= number
            return str(product)
        return None

    def _evaluate_extracted_operation(self, extraction: Dict) -> Optional[int]:
        operation = extraction.get("operation")
        if not isinstance(operation, str) or not operation.strip():
            return None

        variables: Dict[str, int] = {}
        replacements: List[Tuple[str, str]] = []
        for idx, (entity, value) in enumerate(self._numeric_evidence(extraction)):
            name = self._safe_identifier(entity) or f"value_{idx}"
            unique_name = name
            suffix = 2
            while unique_name in variables:
                unique_name = f"{name}_{suffix}"
                suffix += 1
            variables[unique_name] = value
            replacements.append((entity, unique_name))
            replacements.append((entity.lower(), unique_name))

        if not variables:
            return None

        expression = operation.strip()
        for source, target in sorted(replacements, key=lambda x: len(x[0]), reverse=True):
            if source and source != target:
                expression = re.sub(rf"\b{re.escape(source)}\b", target, expression, flags=re.IGNORECASE)

        allowed_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_+-*/%() .")
        if any(char not in allowed_chars for char in expression):
            return None

        try:
            tree = ast.parse(expression, mode="eval")
            value = self._eval_ast(tree.body, variables)
            return int(value)
        except Exception:
            return None

    def _eval_ast(self, node: ast.AST, variables: Dict[str, int]) -> int:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return int(node.value)
        if isinstance(node, ast.Name) and node.id in variables:
            return variables[node.id]
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -self._eval_ast(node.operand, variables)
        if isinstance(node, ast.BinOp):
            left = self._eval_ast(node.left, variables)
            right = self._eval_ast(node.right, variables)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.FloorDiv):
                return left // right
            if isinstance(node.op, ast.Div):
                return left // right if left % right == 0 else int(left / right)
            if isinstance(node.op, ast.Mod):
                return left % right
        raise ValueError("Unsupported expression")

    def _safe_identifier(self, text: str) -> str:
        name = re.sub(r"[^0-9A-Za-z_]+", "_", text.strip().lower()).strip("_")
        if not name:
            return ""
        if name[0].isdigit():
            name = f"value_{name}"
        return name

    def _solve_date_time(self, question: str, context: str, extraction: Dict) -> Optional[str]:
        source = "\n".join(self._evidence_values(extraction)) + "\n" + question
        dates = self._extract_dates(source)
        if not dates:
            return None

        date = dates[0]
        offset_match = re.search(r"(\d+)\s+days?\s+(before|after|prior to)", source, re.IGNORECASE)
        if offset_match:
            days = int(offset_match.group(1))
            direction = offset_match.group(2).lower()
            date = date - timedelta(days=days) if direction in ("before", "prior to") else date + timedelta(days=days)

        q = question.lower()
        if any(term in q for term in ("day of the week", "weekday", "what day")):
            return date.strftime("%A")
        return f"{date.strftime('%B')} {date.day}, {date.year}"

    def _extract_dates(self, source: str) -> List[datetime]:
        dates: List[datetime] = []
        for match in re.finditer(r"\b(20\d{2})[-/](\d{1,2})[-/](\d{1,2})\b", source):
            year, month, day = map(int, match.groups())
            try:
                dates.append(datetime(year, month, day))
            except ValueError:
                pass

        month_pattern = (
            r"\b(January|February|March|April|May|June|July|August|September|October|November|December)"
            r"\s+(\d{1,2}),\s*(20\d{2})\b"
        )
        for match in re.finditer(month_pattern, source, re.IGNORECASE):
            try:
                dates.append(datetime.strptime(" ".join(match.groups()), "%B %d %Y"))
            except ValueError:
                pass
        return dates

    def _solve_string_analysis(self, question: str, context: str, extraction: Dict) -> Optional[str]:
        q = question.lower()
        values = self._evidence_values(extraction)
        candidates = values + [self._quoted_value(question) or "", self._quoted_value(context) or ""]

        if "sum of all hexadecimal digits" in q or "hexadecimal digits (0-9 only)" in q:
            haystack = self._longest_alnum(candidates)
            if haystack:
                return str(sum(int(ch) for ch in haystack if ch.isdigit()))

        target = self._quoted_value(question)
        haystack = self._best_string_haystack(values, target) or context
        if not target:
            target = self._target_after_phrase(question)
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
        encoded = self._select_encoded_value(question, context, extraction)
        if not encoded:
            return None
        encoded = encoded.strip().strip("\"'")

        q = (question + "\n" + "\n".join(self._evidence_values(extraction))).lower()
        try:
            if "base32" in q:
                return base64.b32decode(self._pad_base(encoded, block_size=8).upper()).decode("utf-8").strip()
            if "base64" in q or self._looks_like_base64(encoded):
                decoder = base64.urlsafe_b64decode if ("-" in encoded or "_" in encoded) else base64.b64decode
                return decoder(self._pad_base(encoded, block_size=4)).decode("utf-8").strip()
            if "hex" in q and re.fullmatch(r"[0-9A-Fa-f]+", encoded) and len(encoded) % 2 == 0:
                return bytes.fromhex(encoded).decode("utf-8").strip()
            if "caesar" in q or "rotate" in q or "shift" in q or "cipher" in q:
                shift = self._extract_shift(q) or 3
                return self._caesar_decode(encoded, shift)
        except (binascii.Error, UnicodeDecodeError, ValueError):
            return None
        return None

    def _select_encoded_value(self, question: str, context: str, extraction: Dict) -> Optional[str]:
        scored: List[Tuple[int, str]] = []
        for item in self._evidence_items(extraction):
            value = str(item.get("value") or "").strip()
            if not value:
                continue
            meta = f"{item.get('entity', '')} {item.get('unit_or_type', '')} {item.get('source_snippet', '')}".lower()
            score = 0
            if any(term in meta for term in ("encoded", "cipher", "token", "payload", "code", "identifier", "callsign")):
                score += 4
            if any(term in meta for term in ("protocol", "method", "standard", "section", "shift", "rotation")):
                score -= 2
            if re.fullmatch(r"[A-Za-z0-9+/=_-]{6,}", value):
                score += 2
            if re.fullmatch(r"[A-Z0-9-]{5,}", value):
                score += 1
            scored.append((score, value))

        if scored:
            return max(scored, key=lambda item: (item[0], len(item[1])))[1]

        return self._quoted_value(question) or self._quoted_value(context)

    def _quoted_value(self, text: str) -> Optional[str]:
        match = re.search(r"['\"]([^'\"]{1,500})['\"]", text)
        return match.group(1) if match else None

    def _target_after_phrase(self, question: str) -> Optional[str]:
        match = re.search(r"(?:character|string|substring)\s+([A-Za-z0-9_-]{1,50})", question, re.IGNORECASE)
        return match.group(1) if match else None

    def _best_string_haystack(self, values: List[str], target: Optional[str]) -> Optional[str]:
        if target:
            containing = [value for value in values if target in value]
            if containing:
                return max(containing, key=len)
        return max(values, key=len) if values else None

    def _longest_alnum(self, values: List[str]) -> Optional[str]:
        cleaned = [value for value in values if re.search(r"[A-Za-z0-9]", value)]
        return max(cleaned, key=len) if cleaned else None

    def _pad_base(self, text: str, block_size: int) -> str:
        return text + "=" * ((block_size - len(text) % block_size) % block_size)

    def _looks_like_base64(self, text: str) -> bool:
        if not re.fullmatch(r"[A-Za-z0-9+/=_-]{8,}", text):
            return False
        return len(text) % 4 in (0, 2, 3)

    def _extract_shift(self, text: str) -> Optional[int]:
        digit_match = re.search(r"(?:shift|rotate|rotation|positions?|by exactly|by)\D{0,20}(\d+)", text)
        if digit_match:
            return int(digit_match.group(1))
        words = {
            "one": 1,
            "two": 2,
            "three": 3,
            "four": 4,
            "five": 5,
            "six": 6,
            "seven": 7,
            "eight": 8,
            "nine": 9,
            "ten": 10,
            "eleven": 11,
            "twelve": 12,
            "thirteen": 13,
        }
        for word, value in words.items():
            if re.search(rf"\b{word}\s+positions?\b|\bby\s+{word}\b", text):
                return value
        return None

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
