import os
import json
from openai import OpenAI
from typing import Dict, Optional
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

    def __init__(self, api_key: str, base_url: str, ground_truth: str, question: str, model_name: Optional[str] = None):
        """Initialize the LLM evaluator."""
        # 优先级：EVAL 环境变量 > 主环境变量
        final_api_key = os.getenv('EVAL_API_KEY') or os.getenv('API_KEY')
        final_base_url = os.getenv('EVAL_BASE_URL') or os.getenv('BASE_URL')
        
        self.client = OpenAI(api_key=final_api_key, base_url=final_base_url)
        self.ground_truth = ground_truth
        self.question = question
        
        # model_name 同样优先使用环境变量配置
        self.model_name = os.getenv('EVAL_MODEL_NAME') or os.getenv('MODEL_NAME')

    def evaluate_response(self, response: str) -> int:
        """Evaluate a response using LLM."""
        evaluation_prompt = f"""You are an expert evaluator. Your task is to score the answer based on how well it matches the ground truth.

Question: {self.question}
Ground Truth Answer: {self.ground_truth}
Answer: {response}

Scoring Criteria:
{self.CRITERIA['accuracy']}

Please evaluate the answer and respond with a JSON object: {{"score": number}}."""

        content = ""
        try:
            # 检查模型是否支持 thinking 参数（如 GLM 系列）
            # 排除明确不支持的 ECNU 模型
            unsupported_models = ['ecnu-max', 'ecnu-plus', 'ecnu-turbo']
            supports_thinking = not any(m in self.model_name.lower() for m in unsupported_models)
            
            # 评测模型不需要思考模式，如果支持则显式禁用
            extra_body = None
            if supports_thinking:
                extra_body = {
                    "thinking": {
                        "type": "disabled"
                    }
                }

            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system",
                     "content": "You are an expert evaluator. Respond only with a JSON object containing the score: {\"score\": number}."},
                    {"role": "user", "content": evaluation_prompt}
                ],
                temperature=0,
                max_tokens=100,
                response_format={"type": "json_object"},
                extra_body=extra_body
            )
            
            content = completion.choices[0].message.content.strip()
            data = json.loads(content)
            score = int(data.get("score", 0))
            
            if score < 0 or score > 10:
                score = max(0, min(10, score))
            return score
            
        except Exception as e:
            print(f"Error during evaluation: {e}")
            # 兜底逻辑：尝试从文本中提取数字
            try:
                import re
                nums = re.findall(r'\d+', content)
                if nums:
                    return max(0, min(10, int(nums[0])))
            except:
                pass
            return 0
