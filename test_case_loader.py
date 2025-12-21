import json
from typing import Dict, List, Union


def load_test_cases(json_path: str) -> List[Dict]:
    """
    从 JSON 文件加载测试用例，兼容单个或多个用例格式。

    Args:
        json_path: 含测试用例的 JSON 路径

    Returns:
        包含 'needle'、'question'、'ground_truth' 的测试用例列表
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 判断是单个用例还是多个用例
    if isinstance(data, list):
        # 多个测试用例
        test_cases = data
    elif isinstance(data, dict):
        # 检查是否为带 test_cases 的包装格式
        if 'test_cases' in data:
            test_cases = data['test_cases']
        else:
            # 单个测试用例
            test_cases = [data]
    else:
        raise ValueError("Invalid JSON format: expected dict or list")

    # 校验所有测试用例
    required_fields = ['needle', 'question', 'ground_truth']
    for idx, test_case in enumerate(test_cases):
        for field in required_fields:
            if field not in test_case:
                raise ValueError(f"Test case {idx}: Missing required field '{field}'")

        # 若缺失则补充测试用例 ID
        if 'id' not in test_case:
            test_case['id'] = idx + 1

    return test_cases


def load_test_case(json_path: str) -> Dict:
    """
    仅加载单个测试用例（向后兼容）。

    Args:
        json_path: 含测试用例的 JSON 路径

    Returns:
        包含 'needle'、'question'、'ground_truth' 的字典
    """
    test_cases = load_test_cases(json_path)
    if len(test_cases) != 1:
        raise ValueError(f"Expected single test case, but found {len(test_cases)}")
    return test_cases[0]


def is_multi_needle(test_case: Dict) -> bool:
    """
    根据 needle 字段判断是否为多 needle 用例。

    Args:
        test_case: 测试用例字典

    Returns:
        多 needle 返回 True，否则 False
    """
    needle = test_case['needle']
    return isinstance(needle, list) and len(needle) > 1


def get_needles(test_case: Dict) -> List[str]:
    """
    从测试用例中提取 needle。

    Args:
        test_case: 测试用例字典

    Returns:
        needle 字符串列表
    """
    needle = test_case['needle']
    if isinstance(needle, list):
        return needle
    else:
        return [needle]
