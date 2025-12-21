from typing import Dict
from .evaluator import Evaluator


class StringMatchEvaluator(Evaluator):
    """通过精确字符串匹配打分的评测器。"""

    CRITERIA: Dict[str, str] = {
        "exact_match": """
得分 0：回答与标准答案不匹配。
得分 1：回答与标准答案完全一致。
"""
    }

    def __init__(self, ground_truth: str, case_sensitive: bool = False, strip_whitespace: bool = True):
        """
        初始化字符串匹配评测器。

        Args:
            ground_truth: 预期正确答案
            case_sensitive: 是否区分大小写（默认 False）
            strip_whitespace: 是否在比较前去除首尾空白（默认 True）
        """
        self.ground_truth = ground_truth
        self.case_sensitive = case_sensitive
        self.strip_whitespace = strip_whitespace

    def evaluate_response(self, response: str) -> int:
        """
        使用精确匹配评估回答。

        Args:
            response: 待评估的回答

        Returns:
            匹配返回 1，否则返回 0
        """
        # 准备待比较的字符串
        ground_truth = self.ground_truth
        response_to_check = response

        # 需要时移除首尾空白
        if self.strip_whitespace:
            ground_truth = ground_truth.strip()
            response_to_check = response_to_check.strip()

        # 不区分大小写时统一转为小写
        if not self.case_sensitive:
            ground_truth = ground_truth.lower()
            response_to_check = response_to_check.lower()

        # 执行精确匹配
        if response_to_check == ground_truth:
            return 1
        else:
            return 0
