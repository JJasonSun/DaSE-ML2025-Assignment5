import html
import os
import re
import time
from typing import Any, Dict, List

from openai import OpenAI

from reporters.base_reporter import BaseReporter

DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_REPORT_MODEL_NAME = "deepseek-v4-pro"

LABEL = {
    "title": "LLM Evaluation Report",
    "subtitle": "Structured evaluation dashboard and AI product analysis",
    "generated_at": "Generated at",
    "analysis_title": "AI Product Analysis",
    "analysis_model": "Analysis model",
    "model": "Test model",
    "test_mode": "Test mode",
    "total_runs": "Runs",
    "mean_score": "Mean score",
    "evaluator": "Evaluator",
    "cases": "Cases",
    "thinking": "Thinking",
    "enabled": "Enabled",
    "disabled": "Disabled",
    "overview": "Overview",
    "overall": "Score Distribution",
    "max_score": "Max score",
    "min_score": "Min score",
    "type_perf": "Performance by Type",
    "type": "Type",
    "sample_count": "Samples",
    "good_count": "Good",
    "bad_count": "Bad Cases",
    "multi_detail": "Multi-run Detail",
    "needle_depth": "Needle depth",
    "single_heatmap": "Single-mode Heatmap",
    "bad_examples": "Bad Cases",
    "score": "Score",
    "question": "Question",
    "ground_truth": "Ground truth",
    "response": "Model response",
    "tool_diagnostics": "Tool Diagnostics",
    "dimension": "Dimension",
    "label": "Label",
    "count": "Count",
    "execution_path": "Execution path",
    "task_type": "Task type",
    "fallback_reason": "Fallback reason",
    "ai_failed": "AI analysis generation failed: ",
    "missing_key": "DS_API_KEY is missing, so AI text analysis was skipped.",
}


class DeepSeekHtmlReporter(BaseReporter):
    def __init__(self, api_key: str = None, base_url: str = None, model_name: str = None):
        self.api_key = api_key or os.getenv("DS_API_KEY")
        self.base_url = base_url or os.getenv("DS_BASE_URL", DEFAULT_DEEPSEEK_BASE_URL)
        self.model_name = model_name or os.getenv("DS_MODEL_NAME", DEEPSEEK_REPORT_MODEL_NAME)

    def generate(self, data: Dict, output_path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        metrics = dict(data.get("metrics", {}))
        metrics["report_model"] = self.model_name
        analysis = self._generate_ai_analysis(data, metrics)
        html_report = self._render_html(data, metrics, analysis)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_report)

        print(f"[Report] Saved HTML evaluation report to {output_path}")
        return output_path

    def _generate_ai_analysis(self, data: Dict, metrics: Dict[str, Any]) -> str:
        if not self.api_key:
            return LABEL["missing_key"]

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        mode = str(metrics.get("test_mode", data.get("config", {}).get("test_mode", "unknown"))).lower()
        mode_focus = self._analysis_focus_for_mode(mode)
        system = (
            "You are a rigorous AI product evaluation analyst. "
            "You only write the narrative analysis section for an evaluation dashboard. "
            "Do not output HTML, XML, CSS, JavaScript, code blocks, or full web page structures. "
            "Output plain Markdown only. Write the content in Simplified Chinese. "
            "All claims must be grounded in the provided structured metrics, bad cases, and tool diagnostics. "
            "Do not invent missing data."
        )
        user = f"""
Generate the "AI Product Analysis" text for this evaluation dashboard.
Do not generate HTML.
Write in Simplified Chinese.

Required sections:
## Overall Conclusion
## Main Capability Gaps
## Bad Case Attribution
## Actionable Optimization Suggestions

Output constraints:
- Markdown text only.
- No HTML tags such as <html>, <body>, <section>, <div>, or <table>.
- No fenced code blocks.
- Do not say "below is the HTML/code".
- Do not invent model comparisons, user behavior, business outcomes, or unsupported causes.
- Use tool diagnostics to explain whether failures are more likely caused by retrieval, structured extraction,
  tool execution, fallback paths, or answer formatting.

Mode-specific analysis focus:
{mode_focus}

Structured evaluation data:
{{
  "config": {data.get("config", {})},
  "metrics": {metrics},
  "summaries": {data.get("summaries", [])[:20]}
}}
"""
        try:
            return self._chat_with_retry(
                client,
                model=self.model_name,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                enable_thinking=True,
            )
        except Exception as exc:
            return f"{LABEL['ai_failed']}{exc}"

    def _analysis_focus_for_mode(self, mode: str) -> str:
        if mode == "single":
            return (
                "- This is a single-document, single-needle long-context scan.\n"
                "- Focus on sensitivity to context length and needle insertion depth.\n"
                "- Identify low-score regions in the context_length x depth_percent grid.\n"
                "- Separate long-context degradation from answer extraction failures after the needle is found.\n"
                "- Suggestions should target prompting, retrieval fallback, context window strategy, and evaluation setup."
            )
        if mode == "multi":
            return (
                "- This is a multi-document, multi-needle retrieval and reasoning evaluation.\n"
                "- Focus on cross-document retrieval, multi-needle aggregation, and evidence composition.\n"
                "- Attribute failures to missing needles, irrelevant chunks, post-retrieval reasoning errors, "
                "tool execution failures, or answer format drift when supported by data.\n"
                "- Treat needle depth as supporting evidence only; do not over-attribute without data.\n"
                "- Suggestions should target hybrid retrieval, rerank thresholds, neighbor chunk expansion, "
                "tool augmentation, and scenario-specific reasoning."
            )
        return (
            "- The test mode is unknown. Stay conservative.\n"
            "- Analyze only explicit metrics and bad cases. Do not infer mode-specific causes without data."
        )

    def _chat_with_retry(
        self,
        client: OpenAI,
        model: str,
        messages: List[dict],
        enable_thinking: bool,
        max_retries: int = 3,
    ) -> str:
        params: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "extra_body": {"thinking": {"type": "enabled" if enable_thinking else "disabled"}},
        }
        for attempt in range(max_retries):
            try:
                completion = client.chat.completions.create(**params)
                message = completion.choices[0].message
                content = getattr(message, "content", None)
                if content:
                    return content.strip()
                reasoning = getattr(message, "reasoning_content", None)
                return (reasoning or "").strip()
            except Exception:
                if attempt >= max_retries - 1:
                    raise
                time.sleep(2**attempt)
        return ""

    def _render_html(self, data: Dict, metrics: Dict[str, Any], analysis: str) -> str:
        mode_section = self._render_single_section(metrics) if metrics.get("test_mode") == "single" else self._render_multi_section(metrics)
        analysis_html = _markdown_to_html(_sanitize_ai_analysis(analysis))
        return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{LABEL["title"]}</title>
  <style>
    :root {{
      --bg: #071018; --bg-2: #0d1723; --panel: rgba(13, 24, 36, .84);
      --ink: #e8f1f8; --muted: #8ea1b5; --line: rgba(123, 184, 210, .18);
      --line-strong: rgba(127, 211, 255, .38); --good: #35d18d; --partial: #f5b84b;
      --fail: #ff5f6d; --cyan: #42d6ff; --violet: #9c7cff; --shadow: 0 22px 70px rgba(0, 0, 0, .34);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0; color: var(--ink);
      background:
        linear-gradient(90deg, rgba(255,255,255,.035) 1px, transparent 1px) 0 0/42px 42px,
        linear-gradient(0deg, rgba(255,255,255,.03) 1px, transparent 1px) 0 0/42px 42px,
        radial-gradient(circle at 18% 8%, rgba(66, 214, 255, .22), transparent 34%),
        radial-gradient(circle at 86% 12%, rgba(156, 124, 255, .18), transparent 30%),
        linear-gradient(145deg, var(--bg), var(--bg-2) 48%, #050b11);
      font-family: "Aptos", "Segoe UI", "Microsoft YaHei", sans-serif; line-height: 1.58; min-height: 100vh;
    }}
    main {{ max-width: 1280px; margin: 0 auto; padding: 36px 22px 70px; }}
    header {{ display: grid; grid-template-columns: 1.4fr .6fr; gap: 24px; align-items: end; padding: 36px 0 30px; margin-bottom: 18px; border-bottom: 1px solid var(--line-strong); }}
    .eyebrow {{ color: var(--cyan); font: 800 12px/1.2 "Cascadia Mono", monospace; letter-spacing: .14em; text-transform: uppercase; }}
    h1 {{ margin: 10px 0; font-size: clamp(34px, 5vw, 62px); line-height: .98; letter-spacing: 0; font-weight: 850; }}
    h2 {{ margin: 0 0 16px; font-size: 18px; letter-spacing: .02em; }}
    .stamp {{ justify-self: end; min-width: 220px; border: 1px solid var(--line-strong); border-radius: 8px; padding: 17px 18px; background: linear-gradient(180deg, rgba(66,214,255,.13), rgba(10,20,31,.78)); box-shadow: var(--shadow); }}
    .stamp strong {{ display: block; font-size: 42px; line-height: 1; margin: 7px 0; color: #fff; text-shadow: 0 0 24px rgba(66,214,255,.42); }}
    .muted {{ color: var(--muted); }}
    section {{ background: linear-gradient(180deg, var(--panel), rgba(8, 17, 27, .88)); border: 1px solid var(--line); border-radius: 10px; padding: 22px; margin-bottom: 16px; box-shadow: var(--shadow); backdrop-filter: blur(14px); overflow: hidden; }}
    .grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }}
    .card {{ background: linear-gradient(180deg, rgba(255,255,255,.07), rgba(255,255,255,.025)); border: 1px solid var(--line); border-radius: 8px; padding: 15px; min-height: 96px; }}
    .metric-label {{ color: var(--muted); font: 800 11px/1.2 "Cascadia Mono", monospace; text-transform: uppercase; letter-spacing: .08em; }}
    .metric-value {{ font-size: 25px; font-weight: 780; margin-top: 9px; overflow-wrap: anywhere; color: #f7fbff; }}
    table {{ width: 100%; border-collapse: separate; border-spacing: 0; background: rgba(2,8,14,.22); border: 1px solid var(--line); border-radius: 8px; overflow: hidden; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 12px 10px; text-align: left; vertical-align: top; }}
    tr:last-child td {{ border-bottom: 0; }}
    th {{ color: #b7c7d8; font: 800 11px/1.2 "Cascadia Mono", monospace; text-transform: uppercase; letter-spacing: .08em; background: rgba(66,214,255,.06); }}
    .bar-row {{ display: grid; grid-template-columns: minmax(132px, 210px) 1fr 56px; align-items: center; gap: 12px; margin: 11px 0; }}
    .bar-label {{ color: var(--muted); font: 800 11px/1.2 "Cascadia Mono", monospace; overflow-wrap: anywhere; }}
    .bar-track {{ height: 18px; background: rgba(255,255,255,.06); border: 1px solid var(--line); border-radius: 999px; overflow: hidden; }}
    .bar-fill {{ height: 100%; border-radius: 999px; box-shadow: 0 0 18px currentColor; }}
    .score-pill {{ text-align: right; font-weight: 850; color: #fff; font-variant-numeric: tabular-nums; }}
    .analysis {{ display: grid; gap: 14px; font-size: 15px; color: #dbe8f3; }}
    .analysis h3 {{ margin: 4px 0 2px; color: #ffffff; font-size: 18px; }}
    .analysis p {{ margin: 0; color: #cfdae5; }}
    .analysis ul {{ margin: 0; padding-left: 20px; display: grid; gap: 7px; }}
    .heatmap-wrap {{ overflow-x: auto; }}
    .heatmap td, .heatmap th {{ text-align: center; white-space: nowrap; }}
    .heat-cell {{ font-weight: 850; border-radius: 6px; color: #081018; border: 1px solid rgba(255,255,255,.28); padding: 8px; }}
    @media (max-width: 880px) {{ header, .grid {{ grid-template-columns: 1fr; }} .stamp {{ justify-self: start; }} .bar-row {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
<main>
  <header>
    <div>
      <div class="eyebrow">Needle-in-a-Haystack Evaluation</div>
      <h1>{LABEL["title"]}</h1>
      <div class="muted">{LABEL["subtitle"]}</div>
    </div>
    <div class="stamp">
      <span class="muted">{LABEL["mean_score"]}</span>
      <strong>{float(metrics.get("mean_score", 0)):.2f}</strong>
      <span class="muted">{LABEL["generated_at"]}: {_e(data.get("generated_at", ""))}</span>
    </div>
  </header>
  {self._render_overview(metrics)}
  {self._render_outcomes(metrics)}
  {self._render_type_table(metrics)}
  {self._render_tool_diagnostics(metrics)}
  {mode_section}
  {self._render_bad_cases(metrics)}
  <section>
    <h2>{LABEL["analysis_title"]}</h2>
    <div class="analysis">{analysis_html}</div>
  </section>
</main>
</body>
</html>
"""

    def _render_overview(self, metrics: Dict[str, Any]) -> str:
        cards = [
            (LABEL["model"], metrics.get("model")),
            (LABEL["analysis_model"], metrics.get("report_model")),
            (LABEL["test_mode"], metrics.get("test_mode")),
            (LABEL["total_runs"], metrics.get("total_runs")),
            (LABEL["mean_score"], f"{float(metrics.get('mean_score', 0)):.2f}"),
            ("Agent", metrics.get("agent")),
            (LABEL["evaluator"], metrics.get("evaluator_type")),
            (LABEL["cases"], metrics.get("total_cases")),
            (LABEL["thinking"], LABEL["enabled"] if metrics.get("enable_thinking") else LABEL["disabled"]),
        ]
        return f'<section><h2>{LABEL["overview"]}</h2><div class="grid">' + "".join(
            f'<div class="card"><div class="metric-label">{_e(label)}</div><div class="metric-value">{_e(value)}</div></div>'
            for label, value in cards
        ) + "</div></section>"

    def _render_outcomes(self, metrics: Dict[str, Any]) -> str:
        counts = metrics.get("outcome_counts", {})
        total = max(1, int(metrics.get("total_runs", 0)))
        rows = []
        for label, color in [("Good", "var(--good)"), ("Partial", "var(--partial)"), ("Fail", "var(--fail)")]:
            count = int(counts.get(label, 0))
            percent = count / total * 100
            rows.append(
                f'<div class="bar-row"><div class="bar-label">{label}</div>'
                f'<div class="bar-track"><div class="bar-fill" style="width:{percent:.1f}%;background:{color}"></div></div>'
                f'<div class="score-pill">{count}</div></div>'
            )
        return f"""
<section>
  <h2>{LABEL["overall"]}</h2>
  <div class="grid">
    <div class="card"><div class="metric-label">{LABEL["max_score"]}</div><div class="metric-value">{float(metrics.get("max_score", 0)):.2f}</div></div>
    <div class="card"><div class="metric-label">{LABEL["min_score"]}</div><div class="metric-value">{float(metrics.get("min_score", 0)):.2f}</div></div>
    <div class="card"><div class="metric-label">Good</div><div class="metric-value">{counts.get("Good", 0)}</div></div>
    <div class="card"><div class="metric-label">Fail</div><div class="metric-value">{counts.get("Fail", 0)}</div></div>
  </div>
  {''.join(rows)}
</section>
"""

    def _render_type_table(self, metrics: Dict[str, Any]) -> str:
        rows = []
        for type_name, data in metrics.get("by_type", {}).items():
            rows.append(
                "<tr>"
                f"<td>{_e(type_name)}</td>"
                f"<td>{data['count']}</td>"
                f"<td>{data['mean_score']:.2f}</td>"
                f"<td>{data['good_cases']}</td>"
                f"<td>{data['bad_cases']}</td>"
                "</tr>"
            )
        return f"""
<section>
  <h2>{LABEL["type_perf"]}</h2>
  <table>
    <thead><tr><th>{LABEL["type"]}</th><th>{LABEL["sample_count"]}</th><th>{LABEL["mean_score"]}</th><th>{LABEL["good_count"]}</th><th>{LABEL["bad_count"]}</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
</section>
"""

    def _render_tool_diagnostics(self, metrics: Dict[str, Any]) -> str:
        diagnostics = metrics.get("tool_diagnostics", {}) or {}
        path_counts = diagnostics.get("path_counts", {}) or {}
        fallback_counts = diagnostics.get("fallback_counts", {}) or {}
        task_counts = diagnostics.get("task_counts", {}) or {}
        if not path_counts and not fallback_counts and not task_counts:
            return ""

        def rows(data: Dict[str, Any], label: str) -> str:
            if not data:
                return f"<tr><td>{label}</td><td>-</td><td>0</td></tr>"
            return "".join(
                f"<tr><td>{label}</td><td>{_e(key)}</td><td>{_e(value)}</td></tr>"
                for key, value in sorted(data.items())
            )

        return f"""
<section>
  <h2>{LABEL["tool_diagnostics"]}</h2>
  <table>
    <thead><tr><th>{LABEL["dimension"]}</th><th>{LABEL["label"]}</th><th>{LABEL["count"]}</th></tr></thead>
    <tbody>
      {rows(path_counts, LABEL["execution_path"])}
      {rows(task_counts, LABEL["task_type"])}
      {rows(fallback_counts, LABEL["fallback_reason"])}
    </tbody>
  </table>
</section>
"""

    def _render_multi_section(self, metrics: Dict[str, Any]) -> str:
        bars = []
        for row in metrics.get("per_run_scores", []):
            score = float(row["score"])
            color = _score_color(score)
            bars.append(
                f'<div class="bar-row"><div class="bar-label">Case {row["case_id"]} / #{_e(row["label"])}</div>'
                f'<div class="bar-track"><div class="bar-fill" style="width:{score * 10:.1f}%;background:{color}"></div></div>'
                f'<div class="score-pill">{score:.1f}</div></div>'
            )
        multi = metrics.get("multi_mode", {})
        depth = f'avg {multi.get("avg_needle_depth")}% / min {multi.get("min_needle_depth")}% / max {multi.get("max_needle_depth")}%'
        return f"""
<section>
  <h2>{LABEL["multi_detail"]}</h2>
  <p class="muted">{LABEL["needle_depth"]}: {_e(depth)}</p>
  {''.join(bars)}
</section>
"""

    def _render_single_section(self, metrics: Dict[str, Any]) -> str:
        single = metrics.get("single_mode", {})
        context_lengths = single.get("context_lengths", [])
        depth_percents = single.get("depth_percents", [])
        heatmap = single.get("heatmap", {})
        header = "".join(f"<th>{length}</th>" for length in context_lengths)
        rows = []
        for depth in depth_percents:
            cells = []
            row_data = heatmap.get(str(depth), heatmap.get(depth, {}))
            for length in context_lengths:
                score = row_data.get(str(length), row_data.get(length))
                if score is None:
                    cells.append("<td>-</td>")
                else:
                    cells.append(f'<td><div class="heat-cell" style="background:{_heat_color(float(score))};">{float(score):.1f}</div></td>')
            rows.append(f"<tr><th>{depth}%</th>{''.join(cells)}</tr>")
        return f"""
<section>
  <h2>{LABEL["single_heatmap"]}</h2>
  <div class="heatmap-wrap">
    <table class="heatmap">
      <thead><tr><th>Depth \\ Context</th>{header}</tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
  </div>
</section>
"""

    def _render_bad_cases(self, metrics: Dict[str, Any]) -> str:
        rows = []
        for item in metrics.get("bad_examples", []):
            rows.append(
                "<tr>"
                f"<td>{_e(item.get('test_case_id'))}</td>"
                f"<td>{_e(item.get('type'))}</td>"
                f"<td>{_e(item.get('score'))}</td>"
                f"<td>{_e(item.get('question'))}</td>"
                f"<td>{_e(item.get('ground_truth'))}</td>"
                f"<td>{_e(item.get('response'))}</td>"
                "</tr>"
            )
        return f"""
<section>
  <h2>{LABEL["bad_examples"]}</h2>
  <table>
    <thead><tr><th>Case ID</th><th>{LABEL["type"]}</th><th>{LABEL["score"]}</th><th>{LABEL["question"]}</th><th>{LABEL["ground_truth"]}</th><th>{LABEL["response"]}</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
</section>
"""


def _score_color(score: float) -> str:
    if score >= 8:
        return "var(--good)"
    if score >= 4:
        return "var(--partial)"
    return "var(--fail)"


def _heat_color(score: float) -> str:
    score = max(0.0, min(10.0, float(score)))
    if score < 4:
        return "#ff7c86"
    if score < 8:
        return "#f7c75b"
    return "#55d99d"


def _e(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def _markdown_to_html(text: str) -> str:
    if not text:
        return "<p></p>"

    blocks: List[str] = []
    list_items: List[str] = []
    paragraphs: List[str] = []

    def flush_list() -> None:
        nonlocal list_items
        if list_items:
            blocks.append("<ul>" + "".join(f"<li>{item}</li>" for item in list_items) + "</ul>")
            list_items = []

    def flush_paragraph() -> None:
        nonlocal paragraphs
        if paragraphs:
            blocks.append("<p>" + "<br>".join(paragraphs) + "</p>")
            paragraphs = []

    for raw_line in str(text).replace("\r\n", "\n").split("\n"):
        line = raw_line.strip()
        if not line:
            flush_paragraph()
            flush_list()
            continue

        heading = re.match(r"^#{1,4}\s+(.+)$", line)
        if heading:
            flush_paragraph()
            flush_list()
            blocks.append(f"<h3>{_inline_markdown(heading.group(1))}</h3>")
            continue

        bullet = re.match(r"^(?:[-*]\s+|\d+[.)]\s+)(.+)$", line)
        if bullet:
            flush_paragraph()
            list_items.append(_inline_markdown(bullet.group(1)))
            continue

        paragraphs.append(_inline_markdown(line))

    flush_paragraph()
    flush_list()
    return "".join(blocks) if blocks else "<p></p>"


def _sanitize_ai_analysis(text: str) -> str:
    if not text:
        return ""

    clean = str(text).strip()
    code_block = re.fullmatch(r"```(?:html|xml|markdown|md)?\s*(.*?)\s*```", clean, re.DOTALL | re.IGNORECASE)
    if code_block:
        clean = code_block.group(1).strip()

    clean = re.sub(r"```(?:html|xml|markdown|md)?", "", clean, flags=re.IGNORECASE)
    clean = clean.replace("```", "")
    clean = re.sub(r"(?is)<script.*?>.*?</script>", "", clean)
    clean = re.sub(r"(?is)<style.*?>.*?</style>", "", clean)

    if re.search(r"(?is)</?(html|body|section|div|table|tr|td|th|p|h[1-6]|ul|li)\b", clean):
        clean = re.sub(r"(?i)<br\s*/?>", "\n", clean)
        clean = re.sub(r"(?i)</(p|div|section|h[1-6]|li|tr)>", "\n", clean)
        clean = re.sub(r"(?is)<[^>]+>", "", clean)
        clean = html.unescape(clean)

    clean = re.sub(r"(?i)^\s*(here is|below is).{0,80}(html|code).*$", "", clean, flags=re.MULTILINE)
    return clean.strip()


def _inline_markdown(text: str) -> str:
    escaped = _e(text)
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped)
    escaped = re.sub(r"`([^`]+)`", r"<strong>\1</strong>", escaped)
    return escaped
