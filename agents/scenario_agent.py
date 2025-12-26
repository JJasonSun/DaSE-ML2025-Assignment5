import asyncio
import json
import os
import re
from typing import Dict, List, Optional

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
            "You are a high-precision retrieval agent. Your task is to extract a specific 'needle' from the provided 'haystack' (context).\n\n"
            "### EXPERTISE:\n{expertise}\n\n"
            "### PROTOCOL:\n"
            "1. **Locate**: Scan context for keywords/codes from the question. The information IS present.\n"
            "2. **Verify**: Cross-reference the found data with all constraints in the question.\n"
            "3. **Reason**: {reasoning_instruction}\n"
            "4. **Output**: Return ONLY a JSON object: {{\"answer\": \"...\"}}.\n\n"
            "### CONSTRAINT:\n"
            "- No proactive planning.\n"
            "- No hallucinations.\n"
            "- No conversational filler."
        )
        
        user_template = "Context:\n{context}\n\nQuestion: {question}\n\nReturn your answer in JSON format: {{\"answer\": \"...\"}}"

        return {
            "encoding": {
                "system": base_system.format(
                    expertise="You are an expert in data encoding and decoding (Base64, Hex, Ciphers).",
                    reasoning_instruction="Identify the encoded string, decode it accurately using standard methods, and verify the result."
                ),
                "user": user_template
            },
            "string_analysis": {
                "system": base_system.format(
                    expertise="You are a precise string analysis expert (counting, indexing, substring matching).",
                    reasoning_instruction="Perform character-level or word-level analysis. Count exactly, find positions, or identify substrings as they appear. Do not summarize."
                ),
                "user": user_template
            },
            "computation": {
                "system": base_system.format(
                    expertise="You are a mathematical reasoning assistant.",
                    reasoning_instruction="Extract all relevant numerical values, identify the required operations, and perform the calculation step-by-step. Handle units and scales correctly."
                ),
                "user": user_template
            },
            "date_time": {
                "system": base_system.format(
                    expertise="You are a calendar and time-zone expert.",
                    reasoning_instruction="Extract dates/times. Calculate durations or deadlines considering month lengths and leap years. Standardize formats before calculating."
                ),
                "user": user_template
            }
        }

    async def _classify_scenario(self, question: str) -> Optional[str]:
        classification_prompt = (
            "You are a classification assistant. Categorize the question into one of these FOUR types:\n\n"
            "1. encoding: Decoding Base64, Hex, or ciphers. (e.g., 'Decode the message 50484F454E4958363335', 'Using Roman military encryption, decode XMXER552')\n"
            "2. string_analysis: Character/word counting, position, or substring analysis. (e.g., 'Calculate the sum of all numeric digits in the token string', 'Calculate the absolute difference between occurrences of a and E')\n"
            "3. computation: Mathematical calculations with large numbers or multiple steps. (e.g., 'Calculate the precise quarterly budget amount', 'Subtract verified coordinates from total and multiply by multiplier')\n"
            "4. date_time: Dates, days of week, durations, or deadlines. (e.g., 'What day of the week will it go live?', 'How many days between milestone completion and report deadline?')\n\n"
            "Return ONLY the category name (encoding, string_analysis, computation, or date_time). If unsure, return 'none'.\n\n"
            f"Question: {question}\n\n"
            "Category:"
        )
        
        messages = [{"role": "user", "content": classification_prompt}]
        
        # 使用 ecnu-plus 进行分类
        response = await self._create_chat_completion(
            messages=messages,
            model="ecnu-plus",
            temperature=0,
            max_tokens=10,
            enable_thinking=False
        )
        
        category = response.strip().lower()
        # 鲁棒性解析：检查返回字符串中是否包含关键字
        valid_categories = ["encoding", "string_analysis", "computation", "date_time"]
        for cat in valid_categories:
            if cat in category:
                return cat
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
