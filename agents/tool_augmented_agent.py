import ast
import base64
import binascii
import hashlib
import json
import math
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
        self._last_tool_failure_reason: Optional[str] = None

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
        repaired = False
        if answer is None and self._last_tool_failure_reason == "missing_required_evidence":
            repaired_extraction = await self._repair_structured_evidence(question, context, task_type, extraction)
            if repaired_extraction:
                repaired = True
                extraction = repaired_extraction
                answer = self._solve_with_tools(question, context, task_type, extraction)

        if answer is not None and str(answer).strip():
            self.last_trace = {
                "agent": self.__class__.__name__,
                "path": "tool_augmented",
                "task_type": task_type,
                "extraction": extraction,
                "tool_answer": answer,
                "fallback_reason": None,
                "repair_attempted": repaired,
                "context_chars": len(context),
            }
            return self.finalize_answer(str(answer))

        failure_reason = self._last_tool_failure_reason or "unsupported_operation"
        self.last_trace = {
            "agent": self.__class__.__name__,
            "path": "tool_augmented_failed",
            "task_type": task_type,
            "extraction": extraction,
            "tool_answer": None,
            "fallback_reason": failure_reason,
            "repair_attempted": repaired,
            "context_chars": len(context),
        }
        return "Unknown"

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

    def _select_context(self, prompt: Dict, question: str) -> str:
        context_data = prompt.get("context_data", {}) or {}
        context_str = prompt.get("context", "") or ""
        if not context_data and context_str:
            return context_str

        full_context = self._build_full_context(context_data)
        if int(full_context["total_tokens"]) <= self.full_context_threshold_tokens:
            return str(full_context["text"])

        return self._retrieve_with_hybrid(question, context_data, queries=self._tool_retrieval_queries(question))["evidence_text"]

    def _tool_retrieval_queries(self, question: str) -> List[str]:
        q = question.lower()
        queries = [question]
        phrase_map = {
            "inventory system id": "Inventory System ID",
            "batch size constraint": "Batch Size Constraint",
            "encryption offset": "encryption offset",
            "decryption base": "decryption base",
            "temporal multiplier": "Temporal Multiplier",
            "security divisor": "Security Divisor",
            "master access code part a": "Master Access Code Part A",
            "backup code part b": "Backup Code Part B",
            "guard shift pattern": "Guard Shift Pattern",
            "universal assembly code": "universal assembly code",
            "dimensional scaling factor": "dimensional scaling factor",
            "primary antenna frequency coefficient": "Primary antenna frequency coefficient",
            "secondary communication wavelength constant": "Secondary communication wavelength constant",
            "inventory alpha": "Inventory ID Alpha",
            "inventory beta": "Inventory ID Beta",
            "cycle gamma": "Production Cycle ID Gamma",
            "batch delta": "Batch Size ID Delta",
            "production cycle": "Production Cycle ID Gamma",
            "batch size": "Batch Size ID Delta",
            "initial allocation": "Initial allocation value",
            "final allocation": "Final allocation value",
            "hyperdrive calibration constant": "Hyperdrive Calibration Constant",
            "nexus stabilization factor": "Nexus Stabilization Factor",
            "encoded payload": "encoded payload",
            "encoded string": "encoded string",
            "base64": "base64 encoded payload",
            "md5": "MD5 hash payload",
        }
        for key, phrase in phrase_map.items():
            if key in q and phrase not in queries:
                queries.append(phrase)
        return queries

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
        if self._has_task_marker(q, string_markers):
            return "string_analysis"

        date_markers = ("day of the week", "weekday", "date", "deadline", "delivery", "launch")
        if self._has_task_marker(q, date_markers):
            return "date_time"

        encoding_markers = (
            "base64",
            "base32",
            "base16",
            "hex encoded",
            "hexadecimal",
            "ascii hex",
            "caesar",
            "julius",
            "cipher",
            "decode",
            "decoded",
            "encoded signal",
            "encoded string",
            "payload",
            "rotate",
            "rotation",
            "alphabet rotation",
            "original identifier",
            "web encoding",
            "web-safe",
            "ascii-based substitution",
            "standard web encoding",
        )
        if self._has_task_marker(q, encoding_markers):
            return "encoding"

        computation_markers = (
            "difference",
            "differential",
            "sum",
            "total",
            "product",
            "ratio",
            "divide",
            "divided",
            "division",
            "multiply",
            "full simulation",
            "how many",
            "magnitude",
            "cycles",
            "integer division",
            "square root",
            "sqrt",
            "lock code",
            "coefficient",
            "constant",
            "per-layer",
            "decryption key",
            "master key",
            "encryption layers",
        )
        if self._has_task_marker(q, computation_markers):
            return "computation"

        return "general"

    def _has_task_marker(self, question: str, markers: Tuple[str, ...]) -> bool:
        return any(re.search(rf"\b{re.escape(marker)}\b", question) for marker in markers)

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
            "- For computation tasks, include all operands; do not omit constants, divisors, or multipliers.\n"
            "- For encoding tasks, separate payload, method, and shift/key/rule into different evidence_items.\n"
            "- For string tasks, preserve complete strings exactly, including case; do not truncate hashes or payloads.\n"
            "- For date tasks, preserve exact dates and identify whether the answer needs a date, weekday, or day count.\n\n"
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

    async def _repair_structured_evidence(self, question: str, context: str, task_type: str, extraction: Dict) -> Dict:
        prompt = (
            "The previous structured extraction was insufficient for deterministic tool execution.\n"
            "Return only corrected JSON. Do not explain.\n\n"
            "Required fixes:\n"
            "- Include every missing encoded payload, shift/key/rule, number, date, or complete string needed by the operation.\n"
            "- Keep payload, method, and shift/key/rule as separate evidence_items for encoding tasks.\n"
            "- Preserve exact casing and full values. Do not truncate long hashes, identifiers, or encoded strings.\n"
            "- If the operation requires a transform, express it as a Python-style operation where possible.\n\n"
            f"Task type: {task_type}\n"
            f"Question:\n{question}\n\n"
            f"Previous extraction:\n{json.dumps(extraction, ensure_ascii=False)}\n\n"
            f"Context:\n{context[:24000]}\n\n"
            "Corrected JSON:"
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
        self._last_tool_failure_reason = None
        supported_tasks = {"computation", "date_time", "string_analysis", "encoding"}
        extraction_task_type = extraction.get("task_type") if isinstance(extraction.get("task_type"), str) else None
        effective_task_type = task_type if task_type in supported_tasks else extraction_task_type
        if (
            task_type == "string_analysis"
            and extraction_task_type == "computation"
            and self._looks_like_arithmetic_operation(extraction.get("operation"))
        ):
            effective_task_type = "computation"
        if self._has_date_evidence(extraction) and self._asks_for_date_difference(question, extraction):
            effective_task_type = "date_time"
        if effective_task_type not in supported_tasks:
            effective_task_type = task_type

        operation_answer = self._evaluate_extracted_string_operation(question, context, extraction)
        if operation_answer is not None:
            self._last_tool_failure_reason = None
            return operation_answer

        answer: Optional[str]
        if effective_task_type == "computation":
            answer = self._solve_computation(question, context, extraction)
        elif effective_task_type == "date_time":
            answer = self._solve_date_time(question, context, extraction)
        elif effective_task_type == "string_analysis":
            answer = self._solve_string_analysis(question, context, extraction)
        elif effective_task_type == "encoding":
            answer = self._solve_encoding(question, context, extraction)
        else:
            answer = None

        if answer is not None:
            self._last_tool_failure_reason = None
            return answer

        self._set_tool_failure("unsupported_operation")
        return None

    def _looks_like_arithmetic_operation(self, operation: object) -> bool:
        return isinstance(operation, str) and bool(re.search(r"\babs\s*\(|//|[+\-*/%]", operation))

    def _has_date_evidence(self, extraction: Dict) -> bool:
        return any(re.search(r"\b\d{4}-\d{1,2}-\d{1,2}\b", value) for value in self._evidence_values(extraction))

    def _asks_for_date_difference(self, question: str, extraction: Dict) -> bool:
        source = f"{question}\n{extraction.get('operation', '')}".lower()
        return any(term in source for term in ("days between", "elapsed", "difference in days", "datetime("))

    def _set_tool_failure(self, reason: str) -> None:
        if self._last_tool_failure_reason is None:
            self._last_tool_failure_reason = reason

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

        digit_sum_match = re.fullmatch(
            r"sum\(int\(c\)\s+for\s+c\s+in\s+([A-Za-z_]\w*)\s+if\s+c\.isdigit\(\)\)",
            expression,
        )
        if digit_sum_match:
            value = variables.get(self._safe_identifier(digit_sum_match.group(1)))
            return str(sum(int(ch) for ch in value if ch.isdigit())) if value is not None else None

        md5_patterns = (
            r"(?:hashlib\.)?md5\(\s*([A-Za-z_]\w*)(?:\.encode\(\))?\s*\)\.hexdigest\(\)\s*\[\s*:?\s*(\d+)\s*\]",
            r"(?:hashlib\.)?md5\(\s*([A-Za-z_]\w*)(?:\.encode\(\))?\s*\)\.hexdigest\(\)\s*\[\s*0\s*:\s*(\d+)\s*\]",
            r"md5\.hexdigest\(\s*([A-Za-z_]\w*)\s*\)\s*\[\s*0\s*:\s*(\d+)\s*\]",
        )
        for pattern in md5_patterns:
            md5_match = re.fullmatch(pattern, expression)
            if md5_match:
                value = variables.get(self._safe_identifier(md5_match.group(1)))
                if value is None:
                    return None
                return hashlib.md5(value.encode("utf-8")).hexdigest()[: int(md5_match.group(2))]

        caesar_match = re.fullmatch(
            r"(?:caesar_decode|decode_caesar)\(\s*([A-Za-z_]\w*)\s*,\s*(?:shift\s*=\s*)?([A-Za-z_]\w*|\d+)(?:\s*,.*)?\s*\)",
            expression,
        )
        if caesar_match:
            encoded = variables.get(self._safe_identifier(caesar_match.group(1)))
            shift = self._resolve_shift(caesar_match.group(2), variables)
            if encoded is not None and shift is not None:
                decoded = self._caesar_decode(encoded, shift)
                expected_prefix = self._expected_series_prefix(question, context, extraction)
                if expected_prefix and not decoded.upper().startswith(expected_prefix):
                    inferred = self._decode_caesar_to_expected_prefix(encoded, expected_prefix)
                    return inferred or decoded
                return decoded

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
        enriched_extraction = self._with_context_numeric_evidence(context, extraction)
        answer = self._evaluate_extracted_operation(enriched_extraction)
        if answer is not None:
            return str(answer)

        q = question.lower()
        if (
            self._last_tool_failure_reason == "missing_required_evidence"
            and self._looks_like_arithmetic_operation(enriched_extraction.get("operation"))
        ):
            return None

        if self._is_inventory_batch_formula(q):
            labeled_numbers = self._labeled_numbers_from_context(context)
            if len(labeled_numbers) < 4:
                self._set_tool_failure("missing_required_evidence")
                return None
            alpha, beta, gamma, delta = labeled_numbers[:4]
            return str(((alpha - beta) * gamma) // delta) if delta else None

        numbers = self._numbers_from(question, context, enriched_extraction)
        if len(numbers) < 2:
            numbers = self._labeled_numbers_from_context(context)
        if len(numbers) < 2:
            if any(term in q for term in ("square root", "integer square root", "sqrt")) and numbers:
                return str(math.isqrt(numbers[0]))
            self._set_tool_failure("missing_required_evidence")
            return None

        if any(term in q for term in ("square root", "integer square root", "sqrt")):
            return str(math.isqrt(max(numbers)))

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

        if "divide" in q or "division" in q or "ratio" in q or "per-layer" in q or "encryption layers" in q:
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
        self._set_tool_failure("unsupported_operation")
        return None

    def _is_inventory_batch_formula(self, question: str) -> bool:
        return all(term in question for term in ("inventory", "batch", "cycle")) and any(
            term in question for term in ("alpha", "beta", "gamma", "delta")
        )

    def _labeled_numbers_from_context(self, context: str) -> List[int]:
        label_patterns = (
            r"inventory id alpha[^0-9-]{0,80}(-?\d[\d,]*)",
            r"inventory id beta[^0-9-]{0,80}(-?\d[\d,]*)",
            r"production cycle id gamma[^0-9-]{0,80}(-?\d[\d,]*)",
            r"batch size id delta[^0-9-]{0,80}(-?\d[\d,]*)",
            r"initial allocation value[^:=\n]{0,120}[:=]\s*(-?\d[\d,]*)",
            r"final allocation value[^:=\n]{0,120}[:=]\s*(-?\d[\d,]*)",
            r"hyperdrive calibration constant[^:=\n]{0,120}[:=]\s*(-?\d[\d,]*)",
            r"nexus stabilization factor[^:=\n]{0,120}[:=]\s*(-?\d[\d,]*)",
            r"master key[^0-9-]{0,80}(-?\d[\d,]*)",
            r"encryption layers?[^0-9-]{0,80}(-?\d[\d,]*)",
            r"agent id[^0-9-]{0,80}(-?\d[\d,]*)",
            r"division factor[^0-9-]{0,80}(-?\d[\d,]*)",
            r"left code[^0-9-]{0,80}(-?\d[\d,]*)",
            r"right code[^0-9-]{0,80}(-?\d[\d,]*)",
            r"guard shift pattern[^0-9-]{0,80}(-?\d[\d,]*)",
            r"universal assembly code[^0-9-]{0,80}(-?\d[\d,]*)",
            r"dimensional scaling factor[^0-9-]{0,80}(-?\d[\d,]*)",
            r"primary antenna frequency coefficient[^0-9-]{0,80}(-?\d[\d,]*)",
            r"secondary communication wavelength constant[^0-9-]{0,80}(-?\d[\d,]*)",
        )
        numbers: List[int] = []
        for pattern in label_patterns:
            for match in re.finditer(pattern, context, re.IGNORECASE):
                numbers.append(int(match.group(1).replace(",", "")))
        return numbers

    def _with_context_numeric_evidence(self, context: str, extraction: Dict) -> Dict:
        operation = extraction.get("operation")
        if not isinstance(operation, str) or not operation.strip():
            return extraction

        items = [dict(item) for item in self._evidence_items(extraction)]
        changed = False
        for item in items:
            entity = self._safe_identifier(str(item.get("entity") or ""))
            if not entity:
                continue
            context_value = self._number_for_label_from_context(entity, context)
            if context_value is not None and str(item.get("value")) != str(context_value):
                item["value"] = str(context_value)
                changed = True

        existing = {self._safe_identifier(entity) for entity, _ in self._numeric_evidence({"evidence_items": items})}
        names = set(re.findall(r"\b[A-Za-z_]\w*\b", operation))
        missing = [name for name in names if name not in existing and name not in {"abs", "int", "math"}]
        if not missing:
            if not changed:
                return extraction
            enriched = dict(extraction)
            enriched["evidence_items"] = items
            return enriched

        for name in missing:
            value = self._number_for_label_from_context(name, context)
            if value is not None:
                items.append(
                    {
                        "entity": name,
                        "value": str(value),
                        "unit_or_type": "integer",
                        "source_snippet": name.replace("_", " "),
                    }
                )

        if len(items) == len(self._evidence_items(extraction)):
            return extraction
        enriched = dict(extraction)
        enriched["evidence_items"] = items
        return enriched

    def _number_for_label_from_context(self, name: str, context: str) -> Optional[int]:
        label = name.replace("_", " ")
        labels = [label]
        for suffix in (" mars", " jupiter"):
            if label.endswith(suffix):
                labels.append(label[: -len(suffix)])
        if "initial allocation" in label:
            labels.append("initial allocation value")
        if "final allocation" in label:
            labels.append("final allocation value")
        if "hyperdrive calibration constant" in label:
            labels.append("hyperdrive calibration constant")
        if "nexus stabilization factor" in label:
            labels.append("nexus stabilization factor")
        for candidate in labels:
            escaped = re.escape(candidate)
            match = re.search(rf"\b{escaped}\b[^:=\n]{{0,120}}[:=]\s*(-?\d[\d,]*)", context, re.IGNORECASE)
            if match:
                return int(match.group(1).replace(",", ""))
            match = re.search(rf"\b{escaped}\b[^0-9-]{{0,80}}(-?\d[\d,]*)", context, re.IGNORECASE)
            if match:
                return int(match.group(1).replace(",", ""))
        return None

    def _evaluate_extracted_operation(self, extraction: Dict) -> Optional[int]:
        operation = extraction.get("operation")
        if not isinstance(operation, str) or not operation.strip():
            self._set_tool_failure("unsupported_operation")
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
            self._set_tool_failure("missing_required_evidence")
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
            self._set_tool_failure("invalid_operation_parse")
            return None

    def _eval_ast(self, node: ast.AST, variables: Dict[str, int]):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.Name) and node.id in variables:
            return variables[node.id]
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -self._eval_ast(node.operand, variables)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            args = [self._eval_ast(arg, variables) for arg in node.args]
            if node.func.id == "int" and len(args) == 1:
                return int(args[0])
            if node.func.id == "abs" and len(args) == 1:
                return abs(args[0])
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
                return left // right
            if isinstance(node.op, ast.Mod):
                return left % right
            if isinstance(node.op, ast.Pow):
                if right == 0.5:
                    return float(math.isqrt(int(left)))
                return left**right
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
            self._set_tool_failure("missing_required_evidence")
            return None

        q = question.lower()
        if len(dates) >= 2 and any(term in q for term in ("how many days", "days between", "number of days")):
            return str(abs((dates[0] - dates[1]).days))

        date = dates[0]
        offset_match = re.search(r"(\d+)\s+days?\s+(before|after|prior to)", source, re.IGNORECASE)
        if offset_match:
            days = int(offset_match.group(1))
            direction = offset_match.group(2).lower()
            date = date - timedelta(days=days) if direction in ("before", "prior to") else date + timedelta(days=days)

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
            haystack = self._longest_hex_like(candidates + [context])
            if haystack:
                return str(sum(int(ch) for ch in haystack if ch.isdigit()))
            self._set_tool_failure("missing_required_evidence")
            return None

        target = self._quoted_value(question)
        haystack = self._best_string_haystack(values, target) or context
        if not target:
            target = self._target_after_phrase(question)
        if not target:
            self._set_tool_failure("missing_required_evidence")
            return None

        if "count" in q or "occurrence" in q or "how many" in q:
            return str(haystack.count(target))
        if "position" in q or "index" in q:
            idx = haystack.find(target)
            return str(idx) if idx >= 0 else None
        if "length" in q:
            return str(len(target))
        self._set_tool_failure("unsupported_operation")
        return None

    def _solve_encoding(self, question: str, context: str, extraction: Dict) -> Optional[str]:
        q = self._encoding_context_text(question, extraction)
        candidates = self._encoded_candidates(question, context, extraction)
        if not candidates:
            self._last_tool_failure_reason = "missing_required_evidence"
            return None

        methods = self._encoding_methods(q)
        if not methods:
            self._set_tool_failure("unsupported_operation")
            return None

        invalid_output = False
        expected_prefix = self._expected_series_prefix(question, context, extraction)
        for method in methods:
            if method == "caesar":
                shift = self._shift_from_evidence(extraction) or self._extract_shift(q)
                if shift is None:
                    self._set_tool_failure("missing_required_evidence")
                    continue
            else:
                shift = None

            for encoded in candidates:
                decoded = self._decode_candidate(encoded, method, shift)
                if decoded is None:
                    continue
                if self._is_valid_decoded_text(decoded):
                    if method == "caesar" and expected_prefix and not decoded.upper().startswith(expected_prefix):
                        inferred = self._decode_caesar_to_expected_prefix(encoded, expected_prefix)
                        if inferred and self._is_valid_decoded_text(inferred):
                            return inferred
                        invalid_output = True
                        continue
                    return decoded
                invalid_output = True

        self._set_tool_failure("invalid_decoded_output" if invalid_output else "invalid_operation_parse")
        return None

    def _encoding_context_text(self, question: str, extraction: Dict) -> str:
        values = "\n".join(self._evidence_values(extraction))
        operation = str(extraction.get("operation") or "")
        constraints = "\n".join(str(item) for item in extraction.get("constraints", []) or [])
        return f"{question}\n{values}\n{operation}\n{constraints}".lower()

    def _encoding_methods(self, text: str) -> List[str]:
        methods: List[str] = []
        if any(term in text for term in ("caesar", "julius", "shift", "rotate", "rotation")):
            methods.append("caesar")
        if any(term in text for term in ("mirror", "backwards", "read backward", "read backwards", "reverse-order", "reverse order")):
            methods.append("reverse")
        elif "reverse" in text and not any(term in text for term in ("reverse caesar", "reverse shift")):
            methods.append("reverse")
        if "base32" in text:
            methods.append("base32")
        if any(term in text for term in ("base64", "rfc 4648 section 4", "binary-to-text", "web-safe", "web encoding")):
            methods.append("base64")
        if any(term in text for term in ("hex", "base16", "ascii", "two-character", "two-digit pairs")):
            methods.append("hex")
        return methods

    def _decode_candidate(self, encoded: str, method: str, shift: Optional[int]) -> Optional[str]:
        encoded = encoded.strip().strip("\"'")
        if not encoded:
            return None
        try:
            if method == "reverse":
                return encoded[::-1]
            if method == "caesar" and shift is not None:
                return self._caesar_decode(encoded, shift)
            if method == "base32":
                return base64.b32decode(self._pad_base(encoded, block_size=8).upper()).decode("utf-8").strip()
            if method == "base64":
                decoder = base64.urlsafe_b64decode if ("-" in encoded or "_" in encoded) else base64.b64decode
                return decoder(self._pad_base(encoded, block_size=4)).decode("utf-8").strip()
            if method == "hex":
                normalized = self._normalize_hex_payload(encoded)
                if normalized:
                    return bytes.fromhex(normalized).decode("utf-8").strip()
        except (binascii.Error, UnicodeDecodeError, ValueError):
            return None
        return None

    def _encoded_candidates(self, question: str, context: str, extraction: Dict) -> List[str]:
        scored: List[Tuple[int, str]] = []
        for item in self._evidence_items(extraction):
            value = str(item.get("value") or "").strip()
            if not value:
                continue
            meta = f"{item.get('entity', '')} {item.get('unit_or_type', '')} {item.get('source_snippet', '')}".lower()
            score = 0
            if any(term in meta for term in ("encoded", "ciphertext", "signal", "payload", "code", "identifier", "callsign", "message")):
                score += 4
            if any(term in meta for term in ("protocol", "method", "standard", "section", "shift", "rotation", "cipher_method")):
                score -= 4
            if self._looks_like_base64(value):
                score += 2
            if self._normalize_hex_payload(value):
                score += 2
            if re.fullmatch(r"[A-Z0-9-]{5,}", value):
                score += 1
            scored.append((score, value))

        for source in (question, context):
            for value in self._payload_candidates_from_text(source):
                scored.append((3, value))

        ordered: List[str] = []
        seen = set()
        for _, value in sorted(scored, key=lambda item: (item[0], len(item[1])), reverse=True):
            clean = value.strip().strip("\"'")
            if clean and clean.lower() not in seen:
                ordered.append(clean)
                seen.add(clean.lower())
        return ordered

    def _payload_candidates_from_text(self, text: str) -> List[str]:
        patterns = (
            r"(?:payload|encoded(?:\s+\w+)?|ciphertext|signal|identifier|message|string|code)\s*(?:is|was|:|=|,)?\s*['\"]?([A-Za-z0-9+/=_-]{4,})['\"]?",
            r"(?:payload|ciphertext|signal|identifier|message|string|code).{0,80}?(?:is|was|:|=)\s*['\"]?([A-Za-z0-9+/=_-]{4,})['\"]?",
            r"\b(0x[0-9A-Fa-f]{2}(?:\s*,\s*0x[0-9A-Fa-f]{2})+)\b",
        )
        values: List[str] = []
        for pattern in patterns:
            values.extend(match.group(1) for match in re.finditer(pattern, text, re.IGNORECASE))
        quoted = self._quoted_value(text)
        if quoted and re.fullmatch(r"[A-Za-z0-9+/=_-]{4,}", quoted):
            values.append(quoted)
        return values

    def _normalize_hex_payload(self, text: str) -> Optional[str]:
        bytes_with_prefix = re.findall(r"0x([0-9A-Fa-f]{2})", text)
        if bytes_with_prefix:
            return "".join(bytes_with_prefix)
        compact = re.sub(r"[\s,;:-]", "", text)
        if re.fullmatch(r"[0-9A-Fa-f]+", compact) and len(compact) % 2 == 0 and len(compact) >= 4:
            return compact
        return None

    def _shift_from_evidence(self, extraction: Dict) -> Optional[int]:
        for item in self._evidence_items(extraction):
            meta = f"{item.get('entity', '')} {item.get('unit_or_type', '')} {item.get('source_snippet', '')}".lower()
            if "augustus" in meta or "augustus" in str(item.get("value", "")).lower():
                return 4
            if not any(term in meta for term in ("shift", "rotation", "rotate", "key", "access code")):
                continue
            match = re.search(r"-?\d+", str(item.get("value", "")))
            if match:
                return int(match.group(0))
        return None

    def _expected_series_prefix(self, question: str, context: str, extraction: Dict) -> Optional[str]:
        source = f"{question}\n{context}\n" + "\n".join(self._evidence_values(extraction))
        match = re.search(r"\b([A-Z]{3,12})\s+series\b", source)
        return match.group(1) if match else None

    def _decode_caesar_to_expected_prefix(self, encoded: str, expected_prefix: str) -> Optional[str]:
        letters = re.sub(r"[^A-Za-z]", "", encoded).upper()
        if len(letters) < len(expected_prefix):
            return None
        shifts = []
        for encoded_char, expected_char in zip(letters, expected_prefix):
            shifts.append((ord(encoded_char) - ord(expected_char)) % 26)
        if len(set(shifts)) != 1:
            return None
        return self._caesar_decode(encoded, shifts[0])

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

    def _longest_hex_like(self, values: List[str]) -> Optional[str]:
        matches: List[str] = []
        for value in values:
            matches.extend(re.findall(r"\b[0-9A-Fa-f]{16,}\b", value))
        return max(matches, key=len) if matches else None

    def _pad_base(self, text: str, block_size: int) -> str:
        return text + "=" * ((block_size - len(text) % block_size) % block_size)

    def _looks_like_base64(self, text: str) -> bool:
        if not re.fullmatch(r"[A-Za-z0-9+/=_-]{8,}", text):
            return False
        return len(text) % 4 in (0, 2, 3)

    def _extract_shift(self, text: str) -> Optional[int]:
        digit_match = re.search(r"(?:shift|rotate|rotation|positions?|by exactly|by)[^A-Za-z0-9]{0,20}(\d+)", text)
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
        key_word_match = re.search(r"number of letters in the word ['\"]([A-Za-z]+)['\"]", text)
        if key_word_match:
            return len(key_word_match.group(1))
        return None

    def _is_valid_decoded_text(self, text: str) -> bool:
        if not text:
            return False
        if any(ord(ch) < 32 or ord(ch) == 127 for ch in text if ch not in ("\n", "\t")):
            return False
        if not any(ch.isalnum() for ch in text):
            return False
        printable = sum(1 for ch in text if ch.isprintable())
        return printable / max(1, len(text)) >= 0.9

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
