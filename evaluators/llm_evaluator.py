import os
from openai import OpenAI
from typing import Dict
from .evaluator import Evaluator


class LLMEvaluator(Evaluator):
    """Evaluator that uses LLM to score responses against ground truth."""

    CRITERIA: Dict[str, str] = {
        "accuracy": """
Score 0: The answer is completely wrong or unrelated.
Score 3: The answer has minor relevance but contains major inaccuracies.
Score 5: The answer is partially correct but missing key information.
Score 7: The answer is mostly correct with minor omissions.
Score 10: The answer is completely accurate and matches the ground truth.
"""
    }

    def __init__(self, api_key: str, base_url: str, ground_truth: str, question: str):
        """Initialize the LLM evaluator."""
        # 优先从环境变量获取评测专用配置，实现评测与 Agent 的 API 隔离
        eval_api_key = os.getenv('EVAL_API_KEY') or os.getenv('API_KEY')
        eval_base_url = os.getenv('EVAL_BASE_URL') or os.getenv('BASE_URL')
        # 初始化 OpenAI 客户端
        self.client = OpenAI(api_key=eval_api_key, base_url=eval_base_url)
        self.ground_truth = ground_truth
        self.question = question
        # 从环境变量获取评测模型名称
        self.model_name = os.getenv('EVAL_MODEL_NAME') or os.getenv('MODEL_NAME')
        
        # 准备兜底配置
        self.ecnu_api_key = os.getenv('ECNU_API_KEY')
        self.ecnu_base_url = os.getenv('ECNU_BASE_URL')

    def _call_api(self, client: OpenAI, model: str, prompt: str) -> str:
        """封装 API 调用逻辑。"""
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system",
                 "content": "You are an expert evaluator. Respond only with a number from 0 to 10."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=10
        )
        if not completion or not getattr(completion, 'choices', None) or len(completion.choices) == 0:
            return None
        return completion.choices[0].message.content.strip()

    def evaluate_response(self, response: str) -> int:
        """Evaluate a response using LLM."""
        evaluation_prompt = f"""You are an expert evaluator. Your task is to score the answer based on how well it matches the ground truth.

Question: {self.question}
Ground Truth Answer: {self.ground_truth}
Answer: {response}

Scoring Criteria:
{self.CRITERIA['accuracy']}

Please evaluate the answer and respond with ONLY a single number from 0 to 10. Do not include any explanation or other text."""

        score_text = None
        
        # 1. 尝试主模型
        try:
            score_text = self._call_api(self.client, self.model_name, evaluation_prompt)
        except Exception as e:
            print(f"Primary evaluation model ({self.model_name}) failed: {e}")

        # 2. 兜底策略：如果主模型失败且有 ECNU 配置，换用 ecnu-plus
        if score_text is None and self.ecnu_api_key and self.ecnu_base_url:
            print(f"Switching to fallback model: ecnu-plus")
            try:
                ecnu_client = OpenAI(api_key=self.ecnu_api_key, base_url=self.ecnu_base_url)
                score_text = self._call_api(ecnu_client, "ecnu-plus", evaluation_prompt)
            except Exception as e:
                print(f"Fallback evaluation model (ecnu-plus) failed: {e}")

        if score_text is None:
            return 0

        # 3. 解析分数
        try:
            # 尝试提取第一个数字，以防模型返回了额外文字
            import re
            nums = re.findall(r'\d+', score_text)
            if nums:
                score = int(nums[0])
            else:
                score = 0
                
            if score < 0 or score > 10:
                score = max(0, min(10, score))
            return score
        except Exception as e:
            print(f"Error parsing score text '{score_text}': {e}")
            return 0
