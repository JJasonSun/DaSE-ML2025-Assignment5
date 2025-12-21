import os
import re
from dotenv import load_dotenv
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

    def __init__(self, api_key: str, base_url: str, ground_truth: str, question: str, model_name: str = None):
        """初始化基于 LLM 的评测器。"""
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.ground_truth = ground_truth
        self.question = question
        
        if model_name is None:
            load_dotenv()
            self.model_name = os.getenv('TEST_MODEL', os.getenv('MODEL_NAME', 'ecnu-max'))
        else:
            self.model_name = model_name

    def evaluate_response(self, response: str) -> int:
        """调用 LLM 对回答进行评估打分。"""
        evaluation_prompt = f"""你是一名资深评测员，请根据回答与标准答案的匹配程度打分。

问题：{self.question}
标准答案：{self.ground_truth}
模型回答：{response}

评分标准：
{self.CRITERIA['accuracy']}

请直接返回 0-10 的单个数字，不要包含任何解释或额外内容。"""

        try:
            # 增加 max_tokens 并尝试禁用某些模型的推理模式（如果支持）
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system",
                     "content": "你是一名评测员。你的任务是直接输出一个 0 到 10 之间的整数评分，严禁输出任何推理过程或解释。"},
                    {"role": "user", "content": evaluation_prompt}
                ],
                temperature=0,
                max_tokens=100  # 增加 token 限制以容纳可能的少量前导文字
            )
            
            # 兼容处理：优先获取 content，如果为空则尝试 reasoning_content
            message = completion.choices[0].message
            score_text = ""
            if hasattr(message, 'content') and message.content:
                score_text = message.content.strip()
            elif hasattr(message, 'reasoning_content') and message.reasoning_content:
                score_text = message.reasoning_content.strip()
            
            # 使用正则表达式提取数字
            import re
            match = re.search(r'\d+', score_text)
            if match:
                score = int(match.group())
            else:
                print(f"Warning: Could not find a score in LLM response: '{score_text}'")
                score = 0
                
            if score < 0: score = 0
            if score > 10: score = 10
            return score
        except Exception as e:
            print(f"Error during LLM evaluation: {e}")
            return 0
