import os
import re
import json
from openai import OpenAI
from typing import Dict, Optional
from dotenv import load_dotenv
from .evaluator import Evaluator


class LLMEvaluator(Evaluator):
    """Evaluator that uses LLM to score responses against ground truth."""

    CRITERIA: Dict[str, str] = {
        "accuracy": """
Score 0: The answer is completely wrong, unrelated, or says "I cannot generate a valid answer" when the information was actually present.
Score 3: The answer mentions some relevant keywords but fails to answer the core question or provides incorrect values.
Score 5: The answer is partially correct (e.g., got the date right but the day wrong, or vice versa) or missing critical details.
Score 8: The answer is mostly correct and matches the ground truth's meaning, but has minor formatting issues or extra unnecessary text.
Score 10: The answer is perfectly accurate, concise, and matches the ground truth exactly in meaning (especially for dates, codes, and numbers).
"""
    }

    def __init__(self, api_key: str, base_url: str, ground_truth: str, question: str, model_name: Optional[str] = None):
        """Initialize the LLM evaluator."""
        load_dotenv()
        self.model_name = model_name or os.getenv('MODEL_NAME', 'glm-4.5')
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.ground_truth = ground_truth
        self.question = question

    def evaluate_response(self, response: str) -> int:
        """Evaluate a response using LLM."""
        evaluation_prompt = f"""You are a strict and professional grader for a "Needle in a Haystack" retrieval task.
Your goal is to determine if the candidate's answer accurately reflects the Ground Truth.

Task Context:
- The candidate was asked to find a specific piece of information (the "needle") in a large collection of documents.
- Accuracy is paramount, especially for dates, days of the week, specific codes, and numerical values.

Question: {self.question}
Ground Truth Answer: {self.ground_truth}
Candidate Answer: {response}

Scoring Criteria:
{self.CRITERIA['accuracy']}

Evaluation Instructions:
1. Compare the Candidate Answer with the Ground Truth.
2. If the question asks for a specific fact (like a day of the week or a number) and the candidate provides the WRONG fact, the score MUST be 0, even if other parts of the answer are correct.
3. If the Candidate Answer is "I cannot generate a valid answer" or similar, score it 0.
4. Focus on semantic correctness. Minor capitalization or punctuation differences should not penalize the score.
5. Respond in JSON format: {{"score": integer}}.

Final Score:"""

        try:
            params = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": "You are an expert evaluator. Respond in JSON format: {\"score\": integer}."},
                    {"role": "user", "content": evaluation_prompt}
                ],
                "temperature": 0,
                "max_tokens": 100,
                "response_format": {"type": "json_object"}
            }

            # 只有不是 ecnu-max 和 ecnu-plus 时才添加 thinking: disabled 参数
            unsupported_models = ['ecnu-max', 'ecnu-plus']
            if not any(m in self.model_name.lower() for m in unsupported_models):
                params["extra_body"] = {
                    "thinking": {
                        "type": "disabled"
                    }
                }

            completion = self.client.chat.completions.create(**params)
            
            message = completion.choices[0].message
            content = message.content
            
            if content is None:
                return 0
            score_raw = content.strip()
            
            # 1. 预处理：剥离可能存在的 Markdown 代码块
            clean_raw = score_raw
            if "```" in clean_raw:
                json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", clean_raw, re.DOTALL)
                if json_match:
                    clean_raw = json_match.group(1)
            
            # 2. 尝试解析 JSON
            try:
                data = json.loads(clean_raw)
                score = int(data.get("score", 0))
            except Exception:
                # 3. 兜底：如果 JSON 解析失败，尝试用正则提取
                match = re.search(r'\"score\"\s*:\s*(\d+)', score_raw)
                if match:
                    score = int(match.group(1))
                else:
                    # 最后的尝试：寻找第一个 0-10 的数字
                    match = re.search(r'\b([0-9]|10)\b', score_raw)
                    score = int(match.group(1)) if match else 0
                
            if score < 0 or score > 10:
                return 0
            return score
        except Exception as e:
            print(f"Evaluation Error: {e}")
            return 0