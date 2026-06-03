import os
import random
from typing import Dict, Optional

from core.ecnu_constants import DEFAULT_ECNU_BASE_URL, ECNU_MAIN_MODEL_NAME
from .base_agent import ModelProvider


class BaselineAgent(ModelProvider):
    """
    Minimal baseline agent: randomly selects one file and extracts a fixed token window.
    Use it as a control group, not as the recommended evaluation agent.
    """

    AGENT_PROFILE = {
        "positioning": "Minimal control group. It uses a simple context window and does not perform hybrid retrieval or deterministic tool execution.",
        "expected_strengths": "Useful as a lower-bound baseline for checking whether the evaluation pipeline and prompt format are working.",
        "expected_limits": "Likely to miss deeply buried evidence, multi-document evidence joins, and exact computation-heavy tasks.",
        "analysis_focus": "Treat failures mainly as baseline capability gaps unless metrics show answer formatting or evaluator issues.",
    }

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        api_key = api_key or os.getenv("ECNU_API_KEY") or ""
        base_url = (base_url or os.getenv("ECNU_BASE_URL") or DEFAULT_ECNU_BASE_URL).rstrip("/")
        super().__init__(api_key=api_key, base_url=base_url)
        self.model_name = os.getenv("MODEL_NAME") or ECNU_MAIN_MODEL_NAME
        self.max_tokens_per_request = 10000

    async def evaluate_model(self, prompt: Dict) -> str:
        context_data = prompt.get("context_data") or {}
        context = prompt.get("context") or ""
        question = prompt.get("question") or ""
        if not question:
            return "Missing required input data"

        if context_data:
            selected_content = self._random_select_strategy(context_data)
        elif context:
            selected_content = self._truncate_text(context, self.max_tokens_per_request)
        else:
            return "Missing required input data"

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a strict answer extraction assistant for a Needle-in-a-Haystack evaluation. "
                    "Answer only from the provided context. Return only the final answer, with no explanation, "
                    "prefix, markdown, units unless explicitly requested, or reasoning trace."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Context:\n{selected_content}\n\n"
                    f"Question:\n{question}\n\n"
                    "Final answer only:"
                ),
            },
        ]

        response = await self._create_chat_completion(messages=messages)
        return self.finalize_answer(response.strip())

    def _random_select_strategy(self, context_data: Dict) -> str:
        files = context_data.get("files", [])
        if not files:
            return ""

        selected_file = random.choice(files)
        print(f"[Baseline] Randomly selected file: {selected_file['filename']}")

        content = selected_file.get("modified_content", "")
        return self._truncate_text(content, self.max_tokens_per_request)

    def _truncate_text(self, text: str, max_tokens: int) -> str:
        tokens = self.encode_text_to_tokens(text)
        if len(tokens) <= max_tokens:
            return text
        start_pos = random.randint(0, max(0, len(tokens) - max_tokens))
        return self.decode_tokens(tokens[start_pos : start_pos + max_tokens])
