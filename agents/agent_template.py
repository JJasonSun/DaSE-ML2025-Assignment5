from typing import List, Dict, Optional
from openai import AsyncOpenAI
import tiktoken
import random

from model import ModelProvider


class ExampleAgent(ModelProvider):
    """
    多文档检索 Agent 的示例实现。

    该基线实现仅用于演示接口，策略非常简单：
    - 随机选择 1 个文本文件
    - 随机截取 10000 个 token
    
    如需更佳效果，可考虑：
    - 汇总所有相关文件的信息
    - 使用向量检索的 RAG 流程
    - 基于相关性的智能文件选择
    - 与查询相关的上下文抽取
    """

    def __init__(self, api_key: str, base_url: str):
        super().__init__(api_key, base_url)
        self.model_name = "ecnu-max"
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.tokenizer = tiktoken.encoding_for_model("gpt-4")
        self.max_tokens_per_request = 10000

    async def evaluate_model(self, prompt: Dict) -> str:
        """
        处理多文档检索任务。

        基线策略：
        1. 在所有文件中随机选取 1 个
        2. 从该文件随机截取 10000 个 token
        3. 发送给模型生成答案
        
        该方案仅用于演示。

        Args:
            prompt: 包含 context_data 与 question 的字典

        Returns:
            模型回答
        """
        context_data = prompt['context_data']
        question = prompt['question']

        # 使用基线随机策略
        selected_content = self._random_select_strategy(context_data)

        messages = [
            {
                "role": "system",
                "content": "You are a helpful AI assistant. Answer the question based on the provided context."
            },
            {
                "role": "user",
                "content": f"Context:\n{selected_content}\n\nQuestion: {question}\n\nAnswer:"
            }
        ]

        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0,
            max_tokens=300
        )

        return response.choices[0].message.content

    def _random_select_strategy(self, context_data: Dict) -> str:
        """
        基线策略：随机选 1 个文件并截取 10000 个 token。

        该实现故意简化以展示接口，实际可实现更智能的检索。

        Args:
            context_data: 包含所有文件信息的字典

        Returns:
            截取后的文本内容
        """
        files = context_data['files']

        # 随机选择一个文件
        selected_file = random.choice(files)
        print(f"[Baseline] Randomly selected file: {selected_file['filename']}")

        content = selected_file['modified_content']
        tokens = self.encode_text_to_tokens(content)

        # 若文件长度不足上限则直接返回全部内容
        if len(tokens) <= self.max_tokens_per_request:
            return content

        # 随机截取一段文本
        max_start = len(tokens) - self.max_tokens_per_request
        start_pos = random.randint(0, max_start)
        end_pos = start_pos + self.max_tokens_per_request

        print(f"[Baseline] Randomly extracted tokens {start_pos}-{end_pos} from {len(tokens)} total")

        selected_tokens = tokens[start_pos:end_pos]
        return self.decode_tokens(selected_tokens)

    def generate_prompt(self, **kwargs) -> Dict:
        """
        生成传给模型的 prompt 结构。

        Args:
            **kwargs: 传入的上下文、问题等参数

        Returns:
            包含 prompt 信息的字典
        """
        return {
            'context_data': kwargs.get('context_data'),
            'question': kwargs.get('question')
        }

    def encode_text_to_tokens(self, text: str) -> List[int]:
        """将文本编码成 token 序列。"""
        return self.tokenizer.encode(text)

    def decode_tokens(self, tokens: List[int], context_length: Optional[int] = None) -> str:
        """将 token 序列解码为文本。"""
        if context_length:
            tokens = tokens[:context_length]
        return self.tokenizer.decode(tokens)
