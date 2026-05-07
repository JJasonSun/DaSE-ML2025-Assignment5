from dataclasses import dataclass
from typing import Optional

@dataclass
class CommandArgs:
    """命令行参数配置"""
    agent: str  # Agent 指定，格式为 "module.path:ClassName"，例如：agents.agent_template:ExampleAgent
    test_case_json: str  # 测试用例 JSON 文件路径。支持单个用例或包含多个用例的数组/封装结构

    api_key: Optional[str] = None  # 可选：覆盖环境变量中的 API_KEY（命令行优先）
    base_url: Optional[str] = None  # 可选：覆盖环境变量中的 BASE_URL（命令行优先）

    test_mode: Optional[str] = "multi"  # 测试模式：'multi'（多文档，多 needle）或 'single'（单文档，单 needle）
    evaluator_type: Optional[str] = "llm"  # 评估器类型：'string'（精确字符串匹配）或 'llm'（LLM 语义评分）
    haystack_dir: Optional[str] = "PaulGrahamEssays"  # 存放 haystack 文本文件的目录（用于插入 needles）
    results_version: Optional[int] = 1  # 结果保存的版本号，用于区分不同实验批次

    # 多文档（multi）模式相关参数：
    num_tests: Optional[int] = 5  # 每个测试用例执行的随机试验次数（每次 needle 插入位置不同）

    # 单文档（single）模式下用于扫描不同上下文长度的参数（以 token 计）：
    context_lengths_min: Optional[int] = 1000  # 最小上下文长度（token）
    context_lengths_max: Optional[int] = 100000  # 最大上下文长度（token）
    context_lengths_num_intervals: Optional[int] = 10  # 在 min 与 max 之间划分多少个区间用于测试

    # 针对 needle 在文档中插入深度（百分比）的扫描范围与划分：
    document_depth_percent_min: Optional[int] = 0  # 插入深度最小值（0 = 文档开头）
    document_depth_percent_max: Optional[int] = 100  # 插入深度最大值（100 = 文档末尾）
    document_depth_percent_intervals: Optional[int] = 10  # 深度划分区间数

    # 输出控制：
    save_results: Optional[bool] = False  # 是否将每次测试结果保存到 results/（json 文件）
    save_contexts: Optional[bool] = False  # 是否把生成的上下文文件写入 contexts/ 以便复查
    print_ongoing_status: Optional[bool] = True  # 是否在控制台打印详细的进行状态（便于监控与调试）
    skip_model_test: Optional[bool] = False  # 是否跳过运行前的模型健康检查