import os
import random
from typing import Dict

from core.ecnu_constants import DEFAULT_ECNU_BASE_URL, ECNU_MAIN_MODEL_NAME
from .base_agent import ModelProvider


class ExampleAgent(ModelProvider):
    """
    Baseline agent: randomly selects 1 file and extracts a 10,000-token window.
    Use as a reference implementation when building your own agent.
    """

    def __init__(self, api_key: str, base_url: str):
        api_key = api_key or os.getenv("ECNU_API_KEY") or ""
        base_url = (base_url or os.getenv("ECNU_BASE_URL") or DEFAULT_ECNU_BASE_URL).rstrip("/")
        super().__init__(api_key=api_key, base_url=base_url)
        self.model_name = os.getenv("MODEL_NAME") or ECNU_MAIN_MODEL_NAME
        self.max_tokens_per_request = 10000

    async def evaluate_model(self, prompt: Dict) -> str:
        context_data = prompt["context_data"]
        question = prompt["question"]

        selected_content = self._random_select_strategy(context_data)

        messages = [
            {"role": "system", "content": "你是一个有帮助的 AI 助手。请基于给定上下文回答问题。"},
            {"role": "user", "content": f"上下文：\n{selected_content}\n\n问题：{question}\n\n答案："},
        ]

        response = await self._create_chat_completion(messages=messages)
        return self.finalize_answer(response.strip())

    def _random_select_strategy(self, context_data: Dict) -> str:
        files = context_data["files"]
        selected_file = random.choice(files)
        print(f"[Baseline] Randomly selected file: {selected_file['filename']}")

        content = selected_file["modified_content"]
        tokens = self.encode_text_to_tokens(content)

        if len(tokens) <= self.max_tokens_per_request:
            return content

        max_start = len(tokens) - self.max_tokens_per_request
        start_pos = random.randint(0, max_start)
        end_pos = start_pos + self.max_tokens_per_request

        print(f"[Baseline] Randomly extracted tokens {start_pos}-{end_pos} from {len(tokens)} total")
        return self.decode_tokens(tokens[start_pos:end_pos])
