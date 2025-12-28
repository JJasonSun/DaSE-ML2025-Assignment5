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
            "你是一名专注于 {expertise_title} 的严谨检索与推理专家，处于大海捞针场景：从长文本中精准定位并推导答案。\n\n"
            "【执行协议】\n"
            "1) 需求拆解：明确问题要素与约束，锁定必须满足的条件。\n"
            "2) 证据定位：逐段扫描上下文，捕捉关键词/代码/数值/日期等线索，信息可能被分散或遮蔽。\n"
            "3) 深度推理：{reasoning_instruction} 必要时进行精确的逻辑演绎与数值计算，确保推导过程严密，保证结果精确。\n"
            "4) 交叉校验：用已知约束验证候选答案，排除矛盾与遗漏。\n"
            "5) 输出约束：仅返回 JSON：{{\"answer\": \"...\"}}，不得附加其它文本。\n\n"
            "【严格守则】\n"
            "- 只依赖提供的上下文和通用推理/计算能力，不引入无关外部知识。\n"
            "- 不写过程、不做解释；若穷尽检索与计算仍无结果，返回 {{\"answer\": \"Unknown\"}}。\n"
            "- 信息缺省时先推断再放弃，尤其是日期→星期、数值→运算等，不得因未直述而放弃。"
        )
        
        user_template = (
            "Context:\n{context}\n\nQuestion: {question}\n\n"
            "仅返回 JSON：{{\"answer\": \"...\"}}，不要附加其它文字。"
        )

        return {
            "encoding": {
                "system": base_system.format(
                    expertise_title="数据编码与密码学",
                    reasoning_instruction="识别编码字符串（Base64、Hex 等），确定编码方法，通过逻辑推导完成编码/解码，逐步验证结果与上下文一致。"
                ),
                "user": user_template
            },
            "string_analysis": {
                "system": base_system.format(
                    expertise_title="精准字符串分析",
                    reasoning_instruction="执行字符级或单词级分析，通过严密的逻辑完成计数/位置/子串提取，保证精确，不要近似。"
                ),
                "user": user_template
            },
            "computation": {
                "system": base_system.format(
                    expertise_title="数学推理与计算",
                    reasoning_instruction="提取所有相关数值，识别所需运算（加/减/乘/除等），通过精确的数学推导完成计算，妥善处理单位与比例。"
                ),
                "user": user_template
            },
            "date_time": {
                "system": base_system.format(
                    expertise_title="时间推理与日历分析",
                    reasoning_instruction="提取所有日期和时间，通过逻辑推算确定星期几/时长/截止日期；如仅给日期需推星期，必须进行推导计算，不可跳过，考虑月份长度与闰年。"
                ),
                "user": user_template
            }
        }

    async def _classify_scenario(self, question: str) -> Optional[str]:
        classification_prompt = (
            "你是一个精确分类助手，请基于问题语义判定所属场景，并给出置信度（0-100%）。\n\n"
            "类别定义：\n"
            "1. encoding: 解码 Base64/Hex/密码。\n"
            "2. string_analysis: 字符/单词计数、位置或子串分析。\n"
            "3. computation: 多步或大数计算。\n"
            "4. date_time: 日期、星期几、时长、截止日期计算。\n\n"
            "输出：仅返回 JSON，含 'category' 与 'confidence'。若不确定，category=none。\n\n"
            f"问题: {question}\n\n"
            "JSON:"
        )
        
        messages = [{"role": "user", "content": classification_prompt}]
        
        # 使用 ecnu-max 进行分类
        response = await self._create_chat_completion(
            messages=messages,
            model="ecnu-max",
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
