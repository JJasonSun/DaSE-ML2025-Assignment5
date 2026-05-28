import re
import time
from openai import OpenAI
from typing import Dict, Optional
from .evaluator import Evaluator
from core.ecnu_constants import ECNU_PLUS_MODEL_NAME


class LLMEvaluator(Evaluator):
    """Evaluator that uses LLM to score responses against ground truth."""

    CRITERIA: Dict[str, str] = {
        "accuracy": """
0 分：答案完全错误或与问题无关。
3 分：答案有少量相关性，但包含重大错误。
5 分：答案部分正确，但缺少关键信息。
7 分：答案基本正确，但有轻微遗漏。
10 分：答案完全准确，并与标准答案匹配。
"""
    }

    def __init__(self, api_key: str, base_url: str, ground_truth: str, question: str):
        """Initialize the LLM evaluator."""
        self.ground_truth = ground_truth
        self.question = question
        self.eval_model_name = ECNU_PLUS_MODEL_NAME
        self.eval_client = OpenAI(api_key=api_key, base_url=base_url)

    def _call_api(self, prompt: str, max_retries: int = 3) -> Optional[str]:
        """封装 API 调用逻辑，含重试。"""
        extra_body = {"thinking": {"type": "disabled"}}

        for attempt in range(max_retries):
            try:
                completion = self.eval_client.chat.completions.create(
                    model=self.eval_model_name,
                    messages=[
                        {"role": "system",
                         "content": "你是专业评测员。请只返回 0 到 10 之间的一个数字。"},
                        {"role": "user", "content": prompt}
                    ],
                    extra_body=extra_body,
                )
                if not completion or not getattr(completion, 'choices', None) or len(completion.choices) == 0:
                    return None
                return completion.choices[0].message.content.strip()
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    print(f"[Evaluator] API call failed after {max_retries} attempts: {e}")
                    return None

    def evaluate_response(self, response: str) -> int:
        """Evaluate a response using LLM."""
        evaluation_prompt = f"""你是专业评测员。你的任务是根据模型答案与标准答案的匹配程度进行评分。

问题：{self.question}
标准答案：{self.ground_truth}
模型答案：{response}

评分标准：
{self.CRITERIA['accuracy']}

请评估模型答案，并且只返回 0 到 10 之间的一个数字。不要包含任何解释或其它文本。"""

        score_text = self._call_api(evaluation_prompt)

        if score_text is None:
            return 0

        try:
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
