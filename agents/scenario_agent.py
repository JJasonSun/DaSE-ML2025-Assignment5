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
            "你是一个专注于 {expertise_title} 的高精度检索助手。\n\n"
            "### 任务：\n"
            "通过遵循有条理的推理过程，从提供的“大海”（上下文）中提取准确的“针”（答案）。\n\n"
            "### 协议：\n"
            "1. **分析**：将问题分解为核心要求和约束条件。\n"
            "2. **定位**：在上下文中扫描准确的关键词、代码或实体。信息很可能存在但可能被隐藏。\n"
            "3. **逐步推理**：{reasoning_instruction}\n"
            "4. **验证**：将你的发现与问题中的所有约束条件进行交叉引用，以确保 100% 的准确性。\n"
            "5. **输出**：仅返回一个 JSON 对象：{{\"answer\": \"...\"}}。\n\n"
            "### 约束条件：\n"
            "- **无幻觉**：仅使用提供的上下文。\n"
            "- **无对话废话**：不要解释你的过程或为什么信息可能缺失。\n"
            "- **严格回退**：只有在详尽搜索后信息确实不存在时，才返回 {{\"answer\": \"Unknown\"}}。\n"
            "- **无解释性失败**：严禁返回类似“上下文未提及...”之类的文本。只需在 JSON 中返回 \"Unknown\"。"
        )
        
        user_template = "Context:\n{context}\n\nQuestion: {question}\n\n请以 JSON 格式返回你的答案：{{\"answer\": \"...\"}}"

        return {
            "encoding": {
                "system": base_system.format(
                    expertise_title="数据编码与密码学",
                    reasoning_instruction="识别编码字符串（Base64、Hex 等），确定编码方法，并逐步执行转换。根据上下文验证解码结果。"
                ),
                "user": user_template
            },
            "string_analysis": {
                "system": base_system.format(
                    expertise_title="精准字符串分析",
                    reasoning_instruction="执行字符级或单词级分析。准确计算出现次数，识别精确位置，或提取子字符串。不要总结或近似。"
                ),
                "user": user_template
            },
            "computation": {
                "system": base_system.format(
                    expertise_title="数学推理与计算",
                    reasoning_instruction="提取所有相关的数值。识别所需的运算（加、减、乘、除等）。逐步执行计算，保持精度。仔细处理单位和比例。"
                ),
                "user": user_template
            },
            "date_time": {
                "system": base_system.format(
                    expertise_title="时间推理与日历分析",
                    reasoning_instruction="提取所有相关的日期和时间。逐步计算时长、截止日期或特定的星期几，考虑月份长度和闰年。"
                ),
                "user": user_template
            }
        }

    async def _classify_scenario(self, question: str) -> Optional[str]:
        classification_prompt = (
            "你是一个分类助手。请将问题归类为以下四种类型之一：\n\n"
            "1. encoding: 解码 Base64、Hex 或密码。（例如：'Decode the message 50484F454E4958363335', 'Using Roman military encryption, decode XMXER552'）\n"
            "2. string_analysis: 字符/单词计数、位置或子字符串分析。（例如：'Calculate the sum of all numeric digits in the token string', 'Calculate the absolute difference between occurrences of a and E'）\n"
            "3. computation: 涉及大数或多个步骤的数学计算。（例如：'Calculate the precise quarterly budget amount', 'Subtract verified coordinates from total and multiply by multiplier'）\n"
            "4. date_time: 日期、星期几、时长或截止日期。（例如：'What day of the week will it go live?', 'How many days between milestone completion and report deadline?'）\n\n"
            "仅返回类别名称（encoding, string_analysis, computation 或 date_time）。如果不确定，返回 'none'。\n\n"
            f"问题: {question}\n\n"
            "类别:"
        )
        
        messages = [{"role": "user", "content": classification_prompt}]
        
        # 使用 ecnu-max 进行分类
        response = await self._create_chat_completion(
            messages=messages,
            model="ecnu-max",
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
