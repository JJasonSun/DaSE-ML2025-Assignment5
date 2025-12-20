from openai import OpenAI
from typing import Dict
from .evaluator import Evaluator


class LLMEvaluator(Evaluator):
    """使用 LLM 依据标准答案进行打分的评测器。"""

    CRITERIA: Dict[str, str] = {
        "accuracy": """
得分 0：回答与问题无关或完全错误。
得分 3：略有相关性但存在明显错误。
得分 5：部分正确但缺少关键信息。
得分 7：基本正确，仅有少量遗漏。
得分 10：完全准确并与标准答案一致。
"""
    }

    def __init__(self, api_key: str, base_url: str, ground_truth: str, question: str):
        """初始化基于 LLM 的评测器。"""
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.ground_truth = ground_truth
        self.question = question

    def evaluate_response(self, response: str) -> int:
        """调用 LLM 对回答进行评估打分。"""
        evaluation_prompt = f"""你是一名资深评测员，请根据回答与标准答案的匹配程度打分。

问题：{self.question}
标准答案：{self.ground_truth}
模型回答：{response}

评分标准：
{self.CRITERIA['accuracy']}

请直接返回 0-10 的单个数字，不要包含任何解释或额外内容。"""

        completion = self.client.chat.completions.create(
            model="ecnu-max",
            messages=[
                {"role": "system",
                 "content": "你是一名评测员，仅返回 0 到 10 的数字。"},
                {"role": "user", "content": evaluation_prompt}
            ],
            temperature=0,
            max_tokens=10
        )
        score_text = completion.choices[0].message.content.strip()
        score = int(score_text)
        if score < 0 or score > 10:
            raise ValueError(f"Score out of range: {score}")
        return score
