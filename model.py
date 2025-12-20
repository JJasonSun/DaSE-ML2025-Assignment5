from abc import ABC, abstractmethod
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
