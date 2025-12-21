from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import os
import asyncio
from dotenv import load_dotenv
from openai import OpenAI


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
        self.base_url = base_url.rstrip('/')
        load_dotenv()
        self.model_name = os.getenv('MODEL_NAME')
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    async def _create_chat_completion(self,
                                      messages: List[Dict],
                                      temperature: float = 0,
                                      max_tokens: int = 500,
                                      timeout: int = 20) -> str:
        """在异步上下文中用 OpenAI 客户端调用 ChatCompletion 并返回文本。"""
        def _sync_call():
            # 从环境变量获取是否启用思考模式，默认为 false
            enable_think = os.getenv('ENABLE_THINK', 'false').lower() == 'true'
            extra_body = {}
            
            # 仅针对 GLM 模型或特定支持的模型启用/禁用思考模式
            # 如果 enable_think 为 false，则尝试禁用思考模式
            if not enable_think and "glm" in self.model_name.lower():
                extra_body = {"thinking": {"type": "disabled"}}

            return self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
                extra_body=extra_body
            )

        try:
            try:
                completion = await asyncio.to_thread(_sync_call)
            except AttributeError:
                loop = asyncio.get_running_loop()
                completion = await loop.run_in_executor(None, _sync_call)

            raw = completion.to_dict()
            print(f"Debug: Raw response: {raw}")
            return self._extract_content_from_response(raw)
        except Exception as exc:
            return f"API error: {exc}" if exc else "API error"

    def _extract_content_from_response(self, result: dict) -> str:
        """
        从API响应中提取内容，兼容多种格式。
        """
        try:
            choice = result.get('choices', [{}])[0]
            message = choice.get('message', {})
            
            # 1. 优先获取常规content
            content = message.get('content', '')
            if isinstance(content, str) and content.strip():
                return content.strip()
            
            # 2. 尝试获取reasoning_content（某些服务返回）
            reasoning = message.get('reasoning_content', '')
            if isinstance(reasoning, str) and reasoning.strip():
                return reasoning.strip()
            
            # 3. 检查是否有工具调用
            tool_calls = message.get('tool_calls')
            if tool_calls:
                return f"Tool calls detected: {tool_calls}"
            
            # 4. 返回finish_reason信息
            finish_reason = choice.get('finish_reason', 'unknown')
            return f"Empty response (finish_reason: {finish_reason})"
            
        except Exception as e:
            return f"Response parsing error: {str(e)[:50]}"

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
