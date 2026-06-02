from dataclasses import dataclass
from typing import Optional


@dataclass
class CommandArgs:
    """命令行参数配置。"""

    agent: str  # Agent 路径，格式为 "module.path:ClassName"

    api_key: Optional[str] = None
    base_url: Optional[str] = None

    test_case_json: Optional[str] = None  # 默认使用 test_cases/test_cases_all_en.json
    num_samples: Optional[int] = 20  # 按类型均衡抽样的用例总数

    test_mode: Optional[str] = "multi"  # multi 或 single
    evaluator_type: Optional[str] = "llm"  # llm 或 string
    haystack_dir: Optional[str] = "PaulGrahamEssays"
    results_version: Optional[int] = 1

    # multi 模式参数
    num_tests: Optional[int] = 3

    # single 模式参数
    context_lengths_min: Optional[int] = 1000
    context_lengths_max: Optional[int] = 100000
    context_lengths_num_intervals: Optional[int] = 10
    document_depth_percent_min: Optional[int] = 0
    document_depth_percent_max: Optional[int] = 100
    document_depth_percent_intervals: Optional[int] = 10

    # 模型行为
    enable_thinking: Optional[bool] = False

    # 输出控制
    save_contexts: Optional[bool] = False
    print_ongoing_status: Optional[bool] = True
    generate_report: Optional[bool] = True
    reporter: Optional[str] = "reporters.deepseek_html_reporter:DeepSeekHtmlReporter"
    report_data_path: Optional[str] = None  # 默认 results/latest_<Agent>_evaluation_data.json
    report_output_path: Optional[str] = None  # 默认 results/<Agent>_evaluation_report.html
