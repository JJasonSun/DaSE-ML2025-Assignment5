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
0: The answer is completely wrong or irrelevant.
3: The answer is slightly related but contains major errors.
5: The answer is partially correct but misses key information.
7: The answer is mostly correct with minor omissions or formatting issues.
10: The answer is fully correct and matches the ground truth.
"""
    }

    def __init__(self, api_key: str, base_url: str, ground_truth: str, question: str):
        """Initialize the LLM evaluator."""
        self.ground_truth = ground_truth
        self.question = question
        self.eval_model_name = ECNU_PLUS_MODEL_NAME
        self.eval_client = OpenAI(api_key=api_key, base_url=base_url)

    def _call_api(self, prompt: str, max_retries: int = 3) -> Optional[str]:
        """Call the evaluator model with retry."""
        extra_body = {"thinking": {"type": "disabled"}}

        for attempt in range(max_retries):
            try:
                completion = self.eval_client.chat.completions.create(
                    model=self.eval_model_name,
                    messages=[
                        {"role": "system",
                         "content": "You are a strict evaluator. Return only one number from 0 to 10."},
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
        evaluation_prompt = f"""You are a strict evaluator. Score the model answer against the ground truth.

Question:
{self.question}

Ground truth:
{self.ground_truth}

Model answer:
{response}

Scoring rubric:
{self.CRITERIA['accuracy']}

Return only one number from 0 to 10. Do not include explanations or any other text."""

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
