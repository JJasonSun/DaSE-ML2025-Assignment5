import json
import re
from typing import Dict, Optional

from core.ecnu_constants import ECNU_PLUS_MODEL_NAME
from .agent_plus import AdvancedRetrievalAgent

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
            "你是专注于{expertise_title}的严谨检索与推理专家，正在处理 Needle-in-a-Haystack 长上下文任务："
            "你需要从长上下文中定位精确证据，并推导出正确答案。\n\n"
            "## 协议\n"
            "1. 拆解问题：识别题目中的所有元素、约束条件和必须满足的要求。\n"
            "2. 定位证据：在上下文中检索关键词、编号、数字、日期和其它线索。"
            "信息可能分散、隐藏或只在局部出现。\n"
            "3. 深度推理：{reasoning_instruction} 必要时进行精确逻辑推导和算术计算，"
            "并保证结果准确。不要输出推理过程。\n"
            "4. 交叉验证：用已知约束校验候选答案，排除矛盾结果。\n"
            "5. 输出：只返回最终答案。不要解释，不要列步骤，不要复述证据。\n"
            "6. 兜底：如果充分检索和计算后仍找不到答案，返回 \"Unknown\"。\n\n"
            "## 严格规则\n"
            "- 只能依赖给定上下文和必要的通用推理/算术，不要使用外部知识。\n"
            "- 在放弃前必须先尝试推断，尤其是日期到星期、数字到算术结果这类转换。"
            "不要因为答案没有被直接写出就立即放弃。"
        )

        user_template = (
            "上下文：\n{context}\n\n问题：{question}\n\n"
            "请只输出最终答案，不要解释。"
        )

        return {
            "encoding": {
                "system": base_system.format(
                    expertise_title="数据编码与密码学",
                    reasoning_instruction=(
                        "识别编码方案（如 Base64、Hex、Caesar cipher 等），"
                        "在内部应用正确的解码方法，并核对解码结果是否与上下文一致。"
                    )
                ),
                "user": user_template,
            },
            "string_analysis": {
                "system": base_system.format(
                    expertise_title="精确字符串与字符分析",
                    reasoning_instruction=(
                        "进行严格的字符级或词级分析，精确统计出现次数、判断位置并抽取子串，"
                        "不要近似估算。"
                    )
                ),
                "user": user_template,
            },
            "computation": {
                "system": base_system.format(
                    expertise_title="数学推理与计算",
                    reasoning_instruction=(
                        "抽取所有相关数值，识别所需运算（加、减、乘、除等），"
                        "并完成精确计算，注意单位、比例和整数除法语义。"
                    )
                ),
                "user": user_template,
            },
            "date_time": {
                "system": base_system.format(
                    expertise_title="时间推理与日历分析",
                    reasoning_instruction=(
                        "抽取所有日期和时间，计算星期、时长和截止时间。"
                        "如果题目给出日期但要求星期，必须在内部完成计算，不要跳过。"
                        "需要考虑月份天数和闰年。"
                    )
                ),
                "user": user_template,
            },
        }

    async def _classify_scenario(self, question: str) -> Optional[str]:
        classification_prompt = (
            "请将下面的问题严格分类到一个场景类别中。"
            "只返回 JSON 对象，包含 \"category\" 和 \"confidence\"（0-100%）。\n\n"
            "类别：\n"
            "1. encoding：Base64、Hex、移位密码或其它编码字符串的解码。\n"
            "2. string_analysis：字符/单词计数、位置查找或子串抽取。\n"
            "3. computation：多步算术、大数计算或数学运算。\n"
            "4. date_time：日期、星期、时长或截止时间计算。\n\n"
            "如果问题无法明确归入任何类别，请将 \"category\" 设为 \"none\"。\n\n"
            f"问题：{question}\n\n"
            "JSON："
        )
        
        messages = [{"role": "user", "content": classification_prompt}]
        
        response = await self._create_chat_completion(
            messages=messages,
            model=ECNU_PLUS_MODEL_NAME,
            enable_thinking=False,
            response_format={"type": "json_object"},
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
