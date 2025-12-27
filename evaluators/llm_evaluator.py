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
        self.ground_truth = ground_truth
        self.question = question
        
        # 1. 评测专用配置 (Primary)
        self.eval_api_key = os.getenv('EVAL_API_KEY')
        self.eval_base_url = os.getenv('EVAL_BASE_URL')
        self.eval_model_name = os.getenv('EVAL_MODEL_NAME')
        
        # 2. Agent 配置 (Fallback)
        self.agent_api_key = api_key
        self.agent_base_url = base_url
        self.agent_model_name = os.getenv('MODEL_NAME')

    def _call_api(self, client: OpenAI, model: str, prompt: str) -> str:
        """封装 API 调用逻辑。"""
        extra_body = {}
        # 评测时统一禁用思考模式，以获得快速且直接的分数输出
        if not model.lower().startswith("ecnu"):
            extra_body = {
                "thinking": {
                    "type": "disabled"
                }
            }

        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system",
                 "content": "You are an expert evaluator. Respond only with a number from 0 to 10."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=10,
            extra_body=extra_body if extra_body else None
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
        
        # 1. 尝试评测专用模型
        if self.eval_api_key and self.eval_base_url and self.eval_model_name:
            try:
                eval_client = OpenAI(api_key=self.eval_api_key, base_url=self.eval_base_url)
                score_text = self._call_api(eval_client, self.eval_model_name, evaluation_prompt)
            except Exception as e:
                print(f"Evaluation model ({self.eval_model_name}) failed: {e}")

        # 2. 兜底策略：使用 Agent 的 API
        if score_text is None:
            print(f"Switching to Agent API for evaluation fallback")
            try:
                agent_client = OpenAI(api_key=self.agent_api_key, base_url=self.agent_base_url)
                score_text = self._call_api(agent_client, self.agent_model_name, evaluation_prompt)
            except Exception as e:
                print(f"Agent API evaluation failed: {e}")

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
