import argparse
import os

from dotenv import load_dotenv

from core.evaluation_data import load_evaluation_data
from core.reporter_factory import load_reporter


DEFAULT_REPORTER = "reporters.deepseek_html_reporter:DeepSeekHtmlReporter"


def _project_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), path)


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Generate an HTML report from structured evaluation data.")
    parser.add_argument("--input", default="results/latest_evaluation_data.json")
    parser.add_argument("--output", default="results/evaluation_report.html")
    parser.add_argument("--reporter", default=DEFAULT_REPORTER)
    args = parser.parse_args()

    data = load_evaluation_data(_project_path(args.input))
    reporter_class = load_reporter(args.reporter)
    reporter = reporter_class()
    report_path = reporter.generate(data, _project_path(args.output))
    print(f"Evaluation report saved to: {report_path}")


if __name__ == "__main__":
    main()
