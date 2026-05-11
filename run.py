import os

from dotenv import load_dotenv
from jsonargparse import CLI

from core.ecnu_constants import DEFAULT_ECNU_BASE_URL
from core.config import CommandArgs
from core.health_check import check_models
from core.runner import run_single_test_case
from core.test_case_loader import load_test_cases, get_needles, sample_test_cases

load_dotenv()

def main():
    """程序入口"""
    args = CLI(CommandArgs, as_positional=False)

    # 仅使用 ECNU 配置作为运行时来源
    api_key = args.api_key or os.getenv('ECNU_API_KEY')
    base_url = args.base_url or os.getenv('ECNU_BASE_URL', DEFAULT_ECNU_BASE_URL)

    if not api_key or not base_url:
        raise ValueError("ECNU_API_KEY and ECNU_BASE_URL must be provided via arguments or environment variables")

    if args.test_mode not in ['single', 'multi']:
        raise ValueError(f"test_mode must be 'single' or 'multi', got: {args.test_mode}")

    if args.evaluator_type not in ['string', 'llm']:
        raise ValueError(f"evaluator_type must be 'string' or 'llm', got: {args.evaluator_type}")

    if not args.skip_model_test:
        check_models(api_key, base_url, args.evaluator_type)

    # 加载测试用例（默认使用 test_cases_all_en.json）
    default_test_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_cases', 'test_cases_all_en.json')
    test_case_json = args.test_case_json or default_test_file
    test_cases = load_test_cases(test_case_json)

    # 按类型均衡抽样
    if args.num_samples and args.num_samples < len(test_cases):
        test_cases = sample_test_cases(test_cases, args.num_samples)
        print(f"\nSampled {len(test_cases)} test cases (balanced across types)")

    # single 模式下只保留单 needle 用例
    if args.test_mode == 'single':
        before = len(test_cases)
        test_cases = [tc for tc in test_cases if len(get_needles(tc)) == 1]
        skipped = before - len(test_cases)
        if skipped:
            print(f"[single mode] Skipped {skipped} multi-needle case(s), {len(test_cases)} remaining")
        if not test_cases:
            raise ValueError("No single-needle test cases available. Try increasing --num_samples.")

    print("\n" + "=" * 80)
    print(f"Test cases: {len(test_cases)} (from {os.path.basename(test_case_json)})")
    print(f"Evaluator Type: {args.evaluator_type}")
    print("=" * 80)

    # 逐个运行测试用例
    all_results = []
    test_case_summaries = []

    for idx, test_case in enumerate(test_cases, 1):
        test_id = test_case.get('id', idx)

        print("\n" + "=" * 80)
        print(f"Running Test Case {test_id} ({idx}/{len(test_cases)})")
        print("=" * 80)
        print(f"Question: {test_case['question']}")
        print(f"Ground Truth: {test_case['ground_truth']}")

        needles = get_needles(test_case)
        print(f"Needles: {len(needles)}")
        for i, needle in enumerate(needles, 1):
            print(f"  {i}. {needle[:100]}..." if len(needle) > 100 else f"  {i}. {needle}")
        print("=" * 80)

        try:
            results = run_single_test_case(
                test_case=test_case,
                agent_spec=args.agent,
                api_key=api_key,
                base_url=base_url,
                args=args
            )

            # 将测试用例 ID 和类型加入每条结果
            for result in results:
                result['test_case_id'] = test_id
                result['test_case_type'] = test_case.get('type', 'unknown')

            all_results.extend(results)

            # 计算当前测试用例的统计摘要
            scores = [r['score'] for r in results]
            summary = {
                'test_case_id': test_id,
                'question': test_case['question'][:100],
                'num_runs': len(results),
                'mean_score': sum(scores)/len(scores) if scores else 0,
                'max_score': max(scores) if scores else 0,
                'min_score': min(scores) if scores else 0
            }
            test_case_summaries.append(summary)

        except Exception as e:
            print(f"\n❌ Error running test case {test_id}: {e}")
            import traceback
            traceback.print_exc()

    # 打印最终统计
    print("\n" + "=" * 80)
    print("OVERALL TEST SUMMARY")
    print("=" * 80)
    print(f"Total test cases: {len(test_cases)}")
    print(f"Total test runs: {len(all_results)}")
    
    if test_case_summaries:
        print("\nPer-Test Case Summary:")
        for summary in test_case_summaries:
            print(f"  - Case {summary['test_case_id']}: Mean Score: {summary['mean_score']:.2f}, Range: [{summary['min_score']}, {summary['max_score']}]")

    if all_results:
        overall_mean = sum(r['score'] for r in all_results) / len(all_results)
        print(f"\nOverall Mean Score: {overall_mean:.2f}")
    print("=" * 80)

    # Auto-visualize results
    if args.visualize and all_results:
        from core.visualize import auto_visualize
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
        saved = auto_visualize(all_results, args.test_mode, output_dir)
        if saved:
            print(f"\nVisualization saved to:")
            for p in saved:
                print(f"  {p}")

if __name__ == '__main__':
    main()
