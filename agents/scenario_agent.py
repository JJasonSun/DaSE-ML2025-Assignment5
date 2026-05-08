import json
import re
from typing import Dict, Optional

from core.ecnu_constants import ECNU_PLUS_MODEL_NAME
from agents.agent_plus import AdvancedRetrievalAgent

class ScenarioAwareAgent(AdvancedRetrievalAgent):
    """
    场景感知型 Agent：
    1. 自动识别题目场景 (encoding, string_analysis, computation, date_time)
    2. 针对不同场景采用不同的 Prompt 策略
    3. 继承 AdvancedRetrievalAgent 的混合检索能力
    """

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        super().__init__(api_key=api_key, base_url=base_url)
        self.scenario_prompts = self._get_scenario_prompts()

    def _get_scenario_prompts(self) -> Dict[str, Dict[str, str]]:
        base_system = (
            "You are a rigorous retrieval and reasoning expert specializing in {expertise_title}, "
            "operating in a Needle-in-a-Haystack scenario: locate precise evidence in a long context "
            "and derive the correct answer.\n\n"
            "## Protocol\n"
            "1) Decompose: Identify all question elements, constraints, and conditions that must be satisfied.\n"
            "2) Locate evidence: Scan the context for keywords, codes, numbers, dates, and other clues. "
            "Information may be scattered or partially obscured.\n"
            "3) Reason deeply: {reasoning_instruction} Perform precise logical deduction and arithmetic "
            "as needed. Ensure accuracy. Do NOT output your reasoning process.\n"
            "4) Cross-verify: Validate the candidate answer against known constraints. Eliminate contradictions.\n"
            "5) Output: Return ONLY the final answer. No explanation, no steps, no restating evidence.\n"
            "6) Fallback: If after exhaustive search and computation no answer is found, return \"Unknown\".\n\n"
            "## Strict Rules\n"
            "- Rely ONLY on the provided context and general reasoning/arithmetic. No external knowledge.\n"
            "- Infer before giving up — especially for date→weekday and number→arithmetic conversions. "
            "Do not abandon a question just because the answer is not explicitly stated."
        )

        user_template = (
            "Context:\n{context}\n\nQuestion: {question}\n\n"
            "Output only the final answer. No explanation."
        )

        return {
            "encoding": {
                "system": base_system.format(
                    expertise_title="data encoding and cryptography",
                    reasoning_instruction=(
                        "Identify the encoding scheme (Base64, Hex, Caesar cipher, etc.), "
                        "apply the correct decoding method step-by-step, and verify the result "
                        "is consistent with the surrounding context."
                    )
                ),
                "user": user_template
            },
            "string_analysis": {
                "system": base_system.format(
                    expertise_title="precise string and character analysis",
                    reasoning_instruction=(
                        "Perform character-level or word-level analysis with strict precision. "
                        "Count occurrences, determine positions, and extract substrings exactly — "
                        "no approximation."
                    )
                ),
                "user": user_template
            },
            "computation": {
                "system": base_system.format(
                    expertise_title="mathematical reasoning and calculation",
                    reasoning_instruction=(
                        "Extract all relevant numerical values, identify the required operations "
                        "(addition, subtraction, multiplication, division, etc.), and carry out "
                        "precise arithmetic. Handle units and ratios carefully."
                    )
                ),
                "user": user_template
            },
            "date_time": {
                "system": base_system.format(
                    expertise_title="temporal reasoning and calendar analysis",
                    reasoning_instruction=(
                        "Extract all dates and times. Compute weekdays, durations, and deadlines "
                        "through logical derivation. When only a date is given and the weekday is needed, "
                        "you MUST perform the calculation — do not skip it. Account for month lengths "
                        "and leap years."
                    )
                ),
                "user": user_template
            }
        }

    async def _classify_scenario(self, question: str) -> Optional[str]:
        classification_prompt = (
            "Classify the following question into exactly one scenario category. "
            "Return a JSON object with \"category\" and \"confidence\" (0-100%).\n\n"
            "Categories:\n"
            "1. encoding — Decoding Base64, Hex, ciphers, or other encoded strings.\n"
            "2. string_analysis — Character/word counting, position finding, or substring extraction.\n"
            "3. computation — Multi-step arithmetic, large-number calculation, or mathematical operations.\n"
            "4. date_time — Date, weekday, duration, or deadline calculation.\n\n"
            "If the question does not clearly fit any category, set \"category\" to \"none\".\n\n"
            f"Question: {question}\n\n"
            "JSON:"
        )
        
        messages = [{"role": "user", "content": classification_prompt}]
        
        # 使用 ECNU-plus 进行分类
        response = await self._create_chat_completion(
            messages=messages,
            model=ECNU_PLUS_MODEL_NAME,
            temperature=0,
            max_tokens=100,
            enable_thinking=False,
            response_format={"type": "json_object"}
        )
        
        try:
            # 处理可能包含 Markdown 代码块的情况
            clean_response = response.strip()
            if "```" in clean_response:
                match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", clean_response, re.DOTALL)
                if match:
                    clean_response = match.group(1)
            
            data = json.loads(clean_response)
            category = data.get("category", "none").lower()
            confidence = data.get("confidence", 0)
            
            # 将置信度归一化为 0-1
            if isinstance(confidence, str) and "%" in confidence:
                confidence = float(confidence.strip("%")) / 100
            elif isinstance(confidence, (int, float)) and confidence > 1:
                confidence = confidence / 100
                
            print(f"[Scenario] Classification: {category}, Confidence: {confidence}")

            # 只有当置信度足够高时才采纳分类结果
            if confidence >= 0.7:
                valid_categories = ["encoding", "string_analysis", "computation", "date_time"]
                for cat in valid_categories:
                    if cat in category:
                        return cat
            else:
                print(f"[Scenario] Confidence too low ({confidence}), falling back to default.")
                
        except Exception as e:
            print(f"[Scenario] Classification parsing failed: {e}")
            
        return None

    async def evaluate_model(self, prompt: Dict) -> str:
        question = prompt.get("question", "") or ""
        if not question:
            return "Missing required input data"
        
        # 1. 场景识别
        scenario = await self._classify_scenario(question)
        
        if scenario:
            print(f"[Scenario] Identified as: {scenario}")
            # 2. 动态设置场景特定的 Prompt
            original_prompts = self.prompts.copy()
            scenario_cfg = self.scenario_prompts.get(scenario)
            
            if scenario_cfg:
                self.prompts["system_prompt"] = scenario_cfg["system"]
                self.prompts["user_prompt_template"] = scenario_cfg["user"]

            try:
                # 3. 调用父类的 evaluate_model
                return await super().evaluate_model(prompt)
            finally:
                # 恢复原始 Prompt
                self.prompts = original_prompts
        else:
            print(f"[Scenario] No specific scenario identified, falling back to default prompts.")
            return await super().evaluate_model(prompt)
