from typing import Any, List, Optional

from core.evaluation_data import build_evaluation_data
from core.report_paths import agent_report_stem
from reporters.deepseek_html_reporter import (
    DEFAULT_DEEPSEEK_BASE_URL,
    DEEPSEEK_REPORT_MODEL_NAME,
    DeepSeekHtmlReporter,
)


def generate_evaluation_report(
    results: List[dict],
    summaries: List[dict],
    args: Any,
    output_dir: str,
    api_key: str,
    base_url: str,
    model_name: str = DEEPSEEK_REPORT_MODEL_NAME,
) -> Optional[str]:
    """Compatibility wrapper for generating the default HTML report."""
    if not results:
        print("[Report] No results to report.")
        return None

    data = build_evaluation_data(results, summaries, args)
    report_name = f"{agent_report_stem(str(getattr(args, 'agent', 'unknown_agent')))}_evaluation_report.html"
    reporter = DeepSeekHtmlReporter(api_key=api_key, base_url=base_url, model_name=model_name)
    return reporter.generate(data, f"{output_dir}/{report_name}")
