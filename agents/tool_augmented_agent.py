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
    Planner-first agent: LLMs create a structured operation plan; Python executes
    only deterministic, verifiable tools.
    """

    SUPPORTED_TASKS = {"computation", "date_time", "string_analysis", "encoding"}
    AGENT_PROFILE = {
        "positioning": "Planner-first tool-augmented agent combining hybrid retrieval, LLM operation planning, deterministic Python execution, one-shot replanning on tool failure, validation, and trace metadata.",
        "expected_strengths": "Designed to let the LLM handle semantic field mapping and operation planning while Python handles exact arithmetic, date reasoning, string analysis, hashing, and encoding/decoding.",
        "expected_limits": "Quality depends on retrieved evidence completeness, LLM plan quality, validation coverage, operation parsing, deterministic tool coverage, and answer normalization.",
        "analysis_focus": "Use planner and tool diagnostics heavily. Attribute failures to retrieval completeness, operation_plan quality, validation_result, deterministic execution, replanning behavior, or answer normalization only when supported by traces and metrics. Do not recommend adding more case-specific hardcoded rules unless the data clearly shows a reusable missing tool primitive.",
    }

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        super().__init__(api_key=api_key, base_url=base_url)
        self.last_trace: Dict = {}
        self._last_tool_failure_reason: Optional[str] = None

    async def evaluate_model(self, prompt: Dict) -> str:
        question = prompt.get("question", "") or ""
        if not question:
            return "Missing required input data"

        context = self._select_context(prompt, question)
        plan = await self._create_operation_plan(question, context)
        answer, validation = self._run_plan(question, context, plan)
        planner_attempts = 1
        execution_mode = "planned_tool"

        if answer is None:
            repaired_plan = await self._repair_operation_plan(
                question,
                context,
                plan,
                self._last_tool_failure_reason or validation.get("status") or "unsupported_operation",
            )
            planner_attempts = 2
            if repaired_plan:
                repaired_answer, repaired_validation = self._run_plan(question, context, repaired_plan)
                if repaired_answer is not None:
                    plan = repaired_plan
                    answer = repaired_answer
                    validation = repaired_validation
                    execution_mode = "repaired_planned_tool"

        if answer is not None and str(answer).strip():
            self._record_trace(
                plan=plan,
                answer=answer,
                validation=validation,
                planner_attempts=planner_attempts,
                execution_mode=execution_mode,
                context=context,
            )
            return self.finalize_answer(str(answer))

        failure_reason = self._last_tool_failure_reason or validation.get("status") or "unsupported_operation"
        self.last_trace = {
            "agent": self.__class__.__name__,
            "path": "tool_augmented_failed",
            "task_type": self._plan_task_type(plan),
            "operation_plan": plan,
            "extraction": plan,
            "tool_answer": None,
            "validation_result": validation,
            "tool_failure_reason": failure_reason,
            "fallback_reason": failure_reason,
            "planner_attempts": planner_attempts,
            "execution_mode": "failed",
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
            "operation_plan": extraction or {},
            "extraction": extraction or {},
            "fallback_reason": reason,
            "tool_failure_reason": reason,
            "planner_attempts": 0,
            "execution_mode": "fallback_to_hybrid",
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

        return self._retrieve_with_hybrid(question, context_data, queries=self._tool_retrieval_queries(question))[
            "evidence_text"
        ]

    def _tool_retrieval_queries(self, question: str) -> List[str]:
        queries = [question]
        generic_terms = (
            "encoded string",
            "encoded payload",
            "payload",
            "hash",
            "date",
            "integer",
            "identifier",
            "code",
            "constant",
            "factor",
            "divisor",
            "batch",
            "cycle",
        )
        q = question.lower()
        for term in generic_terms:
            if term in q:
                queries.append(term)
        for token in re.findall(r"\b[A-Za-z][A-Za-z0-9_-]*\d[A-Za-z0-9_-]*\b", question):
            queries.append(token)
        for quoted in re.findall(r"['\"]([^'\"]{3,80})['\"]", question):
            queries.append(quoted)

        ordered: List[str] = []
        seen = set()
        for query in queries:
            clean = query.strip()
            if clean and clean.lower() not in seen:
                ordered.append(clean)
                seen.add(clean.lower())
        return ordered

    async def _create_operation_plan(self, question: str, context: str) -> Dict:
        prompt = self._planner_prompt(question, context)
        response = await self._create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            model=ECNU_PLUS_MODEL_NAME,
            enable_thinking=False,
            response_format={"type": "json_object"},
        )
        return self._normalize_plan(self._parse_json_object(response), question)

    async def _repair_operation_plan(self, question: str, context: str, plan: Dict, failure_reason: str) -> Dict:
        prompt = (
            "The previous operation plan failed deterministic execution.\n"
            "Return only corrected JSON using the same schema. Do not explain.\n\n"
            f"Failure reason: {failure_reason}\n"
            f"Question:\n{question}\n\n"
            f"Previous plan:\n{json.dumps(plan, ensure_ascii=False)}\n\n"
            f"Context:\n{context[:24000]}\n\n"
            "Corrected JSON:"
        )
        response = await self._create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            model=ECNU_PLUS_MODEL_NAME,
            enable_thinking=False,
            response_format={"type": "json_object"},
        )
        return self._normalize_plan(self._parse_json_object(response), question)

    async def _extract_structured_evidence(self, question: str, context: str, task_type: str = "") -> Dict:
        return await self._create_operation_plan(question, context)

    async def _repair_structured_evidence(self, question: str, context: str, task_type: str, extraction: Dict) -> Dict:
        return await self._repair_operation_plan(
            question,
            context,
            extraction,
            self._last_tool_failure_reason or "missing_required_evidence",
        )

    def _planner_prompt(self, question: str, context: str) -> str:
        return (
            "You are the planning component for a tool-augmented Needle-in-a-Haystack evaluator.\n"
            "Return only valid JSON. Do not explain.\n\n"
            "Schema:\n"
            "{\n"
            '  "task_type": "computation|date_time|string_analysis|encoding",\n'
            '  "evidence_items": [\n'
            '    {"entity": "stable_snake_case_name", "value": "exact value", '
            '"unit_or_type": "integer|date|string|payload|method|shift", '
            '"source_snippet": "short verbatim evidence"}\n'
            "  ],\n"
            '  "operation": "Python-style expression or concise tool operation",\n'
            '  "expected_answer_format": "integer|date|weekday|string|identifier|unknown",\n'
            '  "constraints": ["short execution constraints"]\n'
            "}\n\n"
            "Planning rules:\n"
            "- Decide the task type semantically; do not rely on keyword matching.\n"
            "- Include all operands, dates, strings, encoded payloads, methods, shifts, divisors, or constants needed.\n"
            "- Use the entity names in the operation when possible.\n"
            "- Prefer deterministic operations: arithmetic, date difference/weekday, count/reverse/slice, base64/hex/caesar decode.\n"
            "- If evidence is insufficient, still list what is present and make the missing fields explicit in constraints.\n\n"
            f"Question:\n{question}\n\n"
            f"Context:\n{context[:24000]}\n\n"
            "JSON:"
        )

    def _run_plan(self, question: str, context: str, plan: Dict) -> Tuple[Optional[str], Dict]:
        self._last_tool_failure_reason = None
        validation = self._validate_plan(plan)
        if validation["status"] != "ok":
            self._set_tool_failure(validation["status"])
            return None, validation

        answer = self._solve_with_tools(question, context, self._plan_task_type(plan), plan)
        if answer is None:
            validation = {"status": self._last_tool_failure_reason or "unsupported_operation"}
        return answer, validation

    def _validate_plan(self, plan: Dict) -> Dict:
        task_type = self._plan_task_type(plan)
        if task_type not in self.SUPPORTED_TASKS:
            return {"status": "unsupported_operation", "missing": ["task_type"]}
        if not self._evidence_items(plan):
            return {"status": "missing_required_evidence", "missing": ["evidence_items"]}
        return {"status": "ok"}

    def _record_trace(
        self,
        plan: Dict,
        answer: str,
        validation: Dict,
        planner_attempts: int,
        execution_mode: str,
        context: str,
    ) -> None:
        self.last_trace = {
            "agent": self.__class__.__name__,
            "path": "tool_augmented",
            "task_type": self._plan_task_type(plan),
            "operation_plan": plan,
            "extraction": plan,
            "tool_answer": answer,
            "validation_result": validation,
            "tool_failure_reason": None,
            "fallback_reason": None,
            "planner_attempts": planner_attempts,
            "execution_mode": execution_mode,
            "context_chars": len(context),
        }

    def _normalize_plan(self, plan: Dict, question: str) -> Dict:
        if not isinstance(plan, dict):
            plan = {}
        normalized = dict(plan)
        normalized["task_type"] = self._normalize_task_type(str(normalized.get("task_type") or "")) or self._classify_task(
            question
        )
        items = normalized.get("evidence_items") or normalized.get("evidence") or []
        normalized["evidence_items"] = [item for item in items if isinstance(item, dict)]
        normalized["operation"] = str(normalized.get("operation") or "").strip()
        normalized["expected_answer_format"] = str(normalized.get("expected_answer_format") or "unknown")
        constraints = normalized.get("constraints") or []
        normalized["constraints"] = constraints if isinstance(constraints, list) else [str(constraints)]
        return normalized

    def _normalize_task_type(self, value: str) -> str:
        value = value.lower().strip()
        aliases = {
            "math": "computation",
            "calculation": "computation",
            "date": "date_time",
            "datetime": "date_time",
            "string": "string_analysis",
            "text": "string_analysis",
            "decode": "encoding",
        }
        return value if value in self.SUPPORTED_TASKS else aliases.get(value, "")

    def _classify_task(self, question: str) -> str:
        q = question.lower()
        if any(term in q for term in ("decode", "encoded", "cipher", "base64", "base32", "base16", "hex")):
            return "encoding"
        if any(term in q for term in ("date", "weekday", "day of the week", "deadline", "launch", "elapsed")):
            return "date_time"
        if any(term in q for term in ("count", "occurrence", "reverse", "backwards", "substring", "character")):
            return "string_analysis"
        if any(term in q for term in ("calculate", "difference", "sum", "product", "divide", "ratio", "sqrt", "root")):
            return "computation"
        return "computation" if re.search(r"\d", question) else "general"

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

    def _solve_with_tools(self, question: str, context: str, task_type: str, plan: Dict) -> Optional[str]:
        self._last_tool_failure_reason = None
        task_type = self._plan_task_type(plan) if self._plan_task_type(plan) in self.SUPPORTED_TASKS else task_type

        operation_answer = self._evaluate_extracted_string_operation(question, context, plan)
        if operation_answer is not None:
            return operation_answer

        if self._has_date_evidence(plan) and self._asks_for_date_difference(question, plan):
            task_type = "date_time"

        solvers = {
            "computation": self._solve_computation,
            "date_time": self._solve_date_time,
            "string_analysis": self._solve_string_analysis,
            "encoding": self._solve_encoding,
        }
        solver = solvers.get(task_type)
        if not solver:
            self._set_tool_failure("unsupported_operation")
            return None
        return solver(question, context, plan)

    def _plan_task_type(self, plan: Dict) -> str:
        return self._normalize_task_type(str(plan.get("task_type") or ""))

    def _set_tool_failure(self, reason: str) -> None:
        if self._last_tool_failure_reason is None:
            self._last_tool_failure_reason = reason

    def _evidence_items(self, plan: Dict) -> List[Dict]:
        items = plan.get("evidence_items", []) or []
        return [item for item in items if isinstance(item, dict)]

    def _evidence_values(self, plan: Dict) -> List[str]:
        values = []
        for item in self._evidence_items(plan):
            if item.get("value") is not None:
                values.append(str(item["value"]))
        return values

    def _string_evidence(self, plan: Dict, context: str) -> Dict[str, str]:
        variables: Dict[str, str] = {}
        for idx, item in enumerate(self._evidence_items(plan)):
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

    def _evaluate_extracted_string_operation(self, question: str, context: str, plan: Dict) -> Optional[str]:
        operation = str(plan.get("operation") or "").strip()
        if not operation:
            return None
        variables = self._string_evidence(plan, context)
        if not variables:
            return None

        expression = operation.split(";")[-1].strip()

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
                return hashlib.md5(value.encode("utf-8")).hexdigest()[: int(md5_match.group(2))] if value else None

        caesar_match = re.fullmatch(
            r"(?:caesar_decode|decode_caesar)\(\s*([A-Za-z_]\w*)\s*,\s*(?:shift\s*=\s*)?([A-Za-z_]\w*|\d+)(?:\s*,.*)?\s*\)",
            expression,
        )
        if caesar_match:
            encoded = variables.get(self._safe_identifier(caesar_match.group(1)))
            shift_token = caesar_match.group(2)
            shift_value = int(shift_token) if shift_token.isdigit() else self._number_from_value(variables.get(shift_token))
            return self._caesar_decode(encoded, shift_value) if encoded is not None and shift_value is not None else None

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

    def _solve_computation(self, question: str, context: str, plan: Dict) -> Optional[str]:
        answer = self._evaluate_extracted_operation(plan)
        if answer is not None:
            return str(answer)

        q = question.lower()
        numbers = [value for _, value in self._numeric_evidence(plan)]
        if not numbers:
            self._set_tool_failure("missing_required_evidence")
            return None

        if any(term in q for term in ("square root", "integer square root", "sqrt")):
            return str(math.isqrt(max(numbers)))
        if len(numbers) < 2:
            self._set_tool_failure("missing_required_evidence")
            return None
        if "difference" in q or "absolute" in q:
            return str(abs(numbers[0] - numbers[1]))
        if "sum" in q or "total" in q:
            return str(sum(numbers))
        if "product" in q or "multiply" in q:
            product = 1
            for number in numbers:
                product *= number
            return str(product)
        if "divide" in q or "division" in q or "ratio" in q:
            dividend = max(numbers)
            divisors = [n for n in numbers if 0 < n != dividend]
            return str(dividend // min(divisors)) if divisors else None

        self._set_tool_failure("unsupported_operation")
        return None

    def _numeric_evidence(self, plan: Dict) -> List[Tuple[str, int]]:
        pairs: List[Tuple[str, int]] = []
        for idx, item in enumerate(self._evidence_items(plan)):
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

    def _evaluate_extracted_operation(self, plan: Dict) -> Optional[int]:
        operation = str(plan.get("operation") or "").strip()
        if not operation:
            self._set_tool_failure("unsupported_operation")
            return None

        variables: Dict[str, int] = {}
        replacements: List[Tuple[str, str]] = []
        for idx, (entity, value) in enumerate(self._numeric_evidence(plan)):
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

        expression = operation
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
                    return math.isqrt(int(left))
                return left**right
        raise ValueError("Unsupported expression")

    def _solve_date_time(self, question: str, context: str, plan: Dict) -> Optional[str]:
        source = "\n".join(self._evidence_values(plan)) + "\n" + question
        dates = self._extract_dates(source)
        if not dates:
            self._set_tool_failure("missing_required_evidence")
            return None

        q = f"{question}\n{plan.get('operation', '')}\n{plan.get('expected_answer_format', '')}".lower()
        if len(dates) >= 2 and any(term in q for term in ("how many days", "days between", "elapsed", "difference")):
            return str(abs((dates[0] - dates[1]).days))

        date = dates[0]
        offset_match = re.search(r"(\d+)\s+days?\s+(before|after|prior to)", source, re.IGNORECASE)
        if offset_match:
            days = int(offset_match.group(1))
            direction = offset_match.group(2).lower()
            date = date - timedelta(days=days) if direction in ("before", "prior to") else date + timedelta(days=days)

        if any(term in q for term in ("day of the week", "weekday", "what day", "weekday")):
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

    def _solve_string_analysis(self, question: str, context: str, plan: Dict) -> Optional[str]:
        q = question.lower()
        values = self._evidence_values(plan)
        candidates = values + [self._quoted_value(question) or "", self._quoted_value(context) or ""]

        if "sum of all hexadecimal digits" in q or "hexadecimal digits (0-9 only)" in q:
            haystack = self._longest_hex_like(candidates + [context])
            if haystack:
                return str(sum(int(ch) for ch in haystack if ch.isdigit()))
            self._set_tool_failure("missing_required_evidence")
            return None

        target = self._quoted_value(question) or self._target_after_phrase(question)
        haystack = self._best_string_haystack(values, target) or context
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

    def _solve_encoding(self, question: str, context: str, plan: Dict) -> Optional[str]:
        text = self._encoding_context_text(question, plan)
        candidates = self._encoded_candidates(question, context, plan)
        if not candidates:
            self._set_tool_failure("missing_required_evidence")
            return None

        methods = self._encoding_methods(text)
        if not methods:
            self._set_tool_failure("unsupported_operation")
            return None

        invalid_output = False
        for method in methods:
            shift = self._shift_from_evidence(plan) or self._extract_shift(text) if method == "caesar" else None
            if method == "caesar" and shift is None:
                self._set_tool_failure("missing_required_evidence")
                continue

            for encoded in candidates:
                decoded = self._decode_candidate(encoded, method, shift)
                if decoded and self._is_valid_decoded_text(decoded):
                    return decoded
                invalid_output = True

        self._set_tool_failure("invalid_decoded_output" if invalid_output else "invalid_operation_parse")
        return None

    def _encoding_context_text(self, question: str, plan: Dict) -> str:
        values = "\n".join(self._evidence_values(plan))
        operation = str(plan.get("operation") or "")
        constraints = "\n".join(str(item) for item in plan.get("constraints", []) or [])
        return f"{question}\n{values}\n{operation}\n{constraints}".lower()

    def _encoding_methods(self, text: str) -> List[str]:
        methods: List[str] = []
        if any(term in text for term in ("caesar", "julius", "shift", "rotate", "rotation")):
            methods.append("caesar")
        if any(term in text for term in ("mirror", "backward", "backwards", "reverse")):
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

    def _encoded_candidates(self, question: str, context: str, plan: Dict) -> List[str]:
        scored: List[Tuple[int, str]] = []
        for item in self._evidence_items(plan):
            value = str(item.get("value") or "").strip()
            if not value:
                continue
            meta = f"{item.get('entity', '')} {item.get('unit_or_type', '')} {item.get('source_snippet', '')}".lower()
            score = 0
            if any(term in meta for term in ("encoded", "ciphertext", "signal", "payload", "code", "identifier", "message")):
                score += 4
            if any(term in meta for term in ("method", "standard", "section", "shift", "rotation")):
                score -= 4
            if self._looks_like_base64(value):
                score += 2
            if self._normalize_hex_payload(value):
                score += 2
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

    def _shift_from_evidence(self, plan: Dict) -> Optional[int]:
        for item in self._evidence_items(plan):
            meta = f"{item.get('entity', '')} {item.get('unit_or_type', '')} {item.get('source_snippet', '')}".lower()
            if not any(term in meta for term in ("shift", "rotation", "rotate", "key")):
                continue
            match = re.search(r"-?\d+", str(item.get("value", "")))
            if match:
                return int(match.group(0))
        return None

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

    def _number_from_value(self, value: Optional[str]) -> Optional[int]:
        if value is None:
            return None
        match = re.search(r"-?\d+", str(value))
        return int(match.group(0)) if match else None

    def _has_date_evidence(self, plan: Dict) -> bool:
        return any(re.search(r"\b\d{4}-\d{1,2}-\d{1,2}\b", value) for value in self._evidence_values(plan))

    def _asks_for_date_difference(self, question: str, plan: Dict) -> bool:
        source = f"{question}\n{plan.get('operation', '')}".lower()
        return any(term in source for term in ("days between", "elapsed", "difference in days", "datetime("))

    def _safe_identifier(self, text: str) -> str:
        name = re.sub(r"[^0-9A-Za-z_]+", "_", text.strip().lower()).strip("_")
        if not name:
            return ""
        if name[0].isdigit():
            name = f"value_{name}"
        return name

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

    def _longest_hex_like(self, values: List[str]) -> Optional[str]:
        matches: List[str] = []
        for value in values:
            matches.extend(re.findall(r"\b[0-9A-Fa-f]{16,}\b", value))
        return max(matches, key=len) if matches else None

    def _longest_hex_after_hash(self, context: str) -> Optional[str]:
        matches = re.findall(r"(?:hash|token|vector|digest)[^:\n]{0,80}[:=]\s*([0-9A-Fa-f]{16,})", context, re.IGNORECASE)
        return max(matches, key=len) if matches else None

    def _normalize_hex_payload(self, text: str) -> Optional[str]:
        bytes_with_prefix = re.findall(r"0x([0-9A-Fa-f]{2})", text)
        if bytes_with_prefix:
            return "".join(bytes_with_prefix)
        compact = re.sub(r"[\s,;:-]", "", text)
        if re.fullmatch(r"[0-9A-Fa-f]+", compact) and len(compact) % 2 == 0 and len(compact) >= 4:
            return compact
        return None

    def _pad_base(self, text: str, block_size: int) -> str:
        return text + "=" * ((block_size - len(text) % block_size) % block_size)

    def _looks_like_base64(self, text: str) -> bool:
        if not re.fullmatch(r"[A-Za-z0-9+/=_-]{8,}", text):
            return False
        return len(text) % 4 in (0, 2, 3)

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
