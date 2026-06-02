import re
from typing import Optional, Tuple


DEFAULT_REPORT_DATA_PATH = "results/latest_evaluation_data.json"
DEFAULT_REPORT_OUTPUT_PATH = "results/evaluation_report.html"


def agent_report_stem(agent_spec: str) -> str:
    """Return a filesystem-safe, readable agent name for report artifacts."""
    name = (agent_spec or "unknown_agent").split(":")[-1].split(".")[-1]
    stem = re.sub(r"[^0-9A-Za-z_-]+", "_", name).strip("_")
    return stem or "unknown_agent"


def agent_report_paths(agent_spec: str) -> Tuple[str, str]:
    stem = agent_report_stem(agent_spec)
    return (
        f"results/latest_{stem}_evaluation_data.json",
        f"results/{stem}_evaluation_report.html",
    )


def resolve_report_paths(
    agent_spec: str,
    data_path: Optional[str],
    output_path: Optional[str],
) -> Tuple[str, str]:
    default_data_path, default_output_path = agent_report_paths(agent_spec)
    return (
        default_data_path if data_path in (None, "", DEFAULT_REPORT_DATA_PATH) else data_path,
        default_output_path if output_path in (None, "", DEFAULT_REPORT_OUTPUT_PATH) else output_path,
    )
