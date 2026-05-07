from abc import ABC, abstractmethod
import json
import re
from typing import Dict, List, Optional


class ModelProvider(ABC):
    """
    Agent 实现的抽象基类。

    继承本类以实现属于自己的大海捞针测试 Agent。
    """

    def __init__(self, api_key: str, base_url: str):
        """
        初始化模型提供者。

        Args:
            api_key: LLM 服务的 API Key
            base_url: LLM 服务的基础地址
        """
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = "custom-agent"

    @abstractmethod
    async def evaluate_model(self, prompt: Dict) -> str:
        """
        根据给定 prompt 调用模型。

        需要在这里实现 Agent 的核心推理逻辑。

        Args:
            prompt: 包含上下文与问题等信息的字典

        Returns:
            模型返回的答案
        """
        ...

    @abstractmethod
    def generate_prompt(self, **kwargs) -> Dict:
        """
        生成传入模型的 prompt 结构。

        Args:
            **kwargs: 依据测试场景传入的灵活参数

        Returns:
            包含 prompt 信息的字典
        """
        ...

    @abstractmethod
    def encode_text_to_tokens(self, text: str) -> List[int]:
        """
        将文本编码为 tokens。

        Args:
            text: 需要编码的文本

        Returns:
            token ID 列表
        """
        ...

    @abstractmethod
    def decode_tokens(self, tokens: List[int], context_length: Optional[int] = None) -> str:
        """
        将 token ID 解码回文本。

        Args:
            tokens: token ID 列表
            context_length: 可选，限定解码长度

        Returns:
            解码后的文本
        """
        ...

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
        text = re.sub(r"^[\-\*•\d\.\)\(\s]+", "", text)
        text = text.strip('"'“”‘’`')
        text = re.sub(r"\s+", " ", text)
        return text
