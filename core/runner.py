from typing import List

from core.llm_multi_needle_haystack_tester import LLMMultiNeedleHaystackTester
from core.llm_single_needle_haystack_tester import LLMSingleNeedleHaystackTester
from core.test_case_loader import get_needles
from evaluators.llm_evaluator import LLMEvaluator
from evaluators.string_match_evaluator import StringMatchEvaluator
from core.agent_factory import load_agent
from core.config import CommandArgs

def run_single_test_case(test_case: dict, agent_spec: str, api_key: str,
                         base_url: str, args: CommandArgs) -> List[dict]:
    """
    运行单个测试用例。

    Args:
        test_case: 测试用例字典
        agent_spec: Agent 规范
        api_key: 接口密钥
        base_url: 接口基础地址
        args: 命令行参数

    Returns:
        测试结果列表
    """
    needles = get_needles(test_case)

    # 加载 Agent
    agent = load_agent(agent_spec, api_key, base_url)

    # 根据类型初始化评测器
    if args.evaluator_type == 'llm':
        evaluator = LLMEvaluator(
            api_key=api_key,
            base_url=base_url,
            ground_truth=test_case['ground_truth'],
            question=test_case['question'],
        )
    elif args.evaluator_type == 'string':
        evaluator = StringMatchEvaluator(
            ground_truth=test_case['ground_truth'],
            case_sensitive=False,
            strip_whitespace=True
        )
    else:
        raise ValueError(f"Invalid evaluator_type: {args.evaluator_type}. Must be 'string' or 'llm'")

    # 创建并运行测试器
    if args.test_mode == 'multi':
        tester = LLMMultiNeedleHaystackTester(
            model_to_test=agent,
            evaluator=evaluator,
            needles=needles,
            haystack_dir=args.haystack_dir,
            question=test_case['question'],
            results_version=args.results_version,
            num_tests=args.num_tests,
            save_contexts=args.save_contexts,
            print_ongoing_status=args.print_ongoing_status
        )
    else:
        if len(needles) != 1:
            raise ValueError(f"Single document mode requires exactly 1 needle, got {len(needles)}")

        tester = LLMSingleNeedleHaystackTester(
            model_to_test=agent,
            evaluator=evaluator,
            needle=needles[0],
            haystack_dir=args.haystack_dir,
            question=test_case['question'],
            results_version=args.results_version,
            context_lengths_min=args.context_lengths_min,
            context_lengths_max=args.context_lengths_max,
            context_lengths_num_intervals=args.context_lengths_num_intervals,
            document_depth_percent_min=args.document_depth_percent_min,
            document_depth_percent_max=args.document_depth_percent_max,
            document_depth_percent_intervals=args.document_depth_percent_intervals,
            save_contexts=args.save_contexts,
            print_ongoing_status=args.print_ongoing_status
        )

    tester.start_test()
    return tester.get_results()