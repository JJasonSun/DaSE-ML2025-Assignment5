import html
import os
import time
from typing import Any, Dict, List

from openai import OpenAI

from reporters.base_reporter import BaseReporter

DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_REPORT_MODEL_NAME = "deepseek-v4-pro"


def _u(text: str) -> str:
    return text.encode("ascii").decode("unicode_escape")


LABEL = {
    "title": _u(r"LLM \u80fd\u529b\u8bc4\u6d4b\u62a5\u544a"),
    "subtitle": _u(r"\u7ed3\u6784\u5316\u6570\u636e\u770b\u677f\u4e0e AI \u4ea7\u54c1\u5206\u6790"),
    "generated_at": _u(r"\u751f\u6210\u65f6\u95f4"),
    "analysis_title": _u(r"AI \u4ea7\u54c1\u8bc4\u6d4b\u5206\u6790"),
    "analysis_model": _u(r"\u5206\u6790\u6a21\u578b"),
    "model": _u(r"\u4e3b\u6d4b\u6a21\u578b"),
    "test_mode": _u(r"\u6d4b\u8bd5\u6a21\u5f0f"),
    "total_runs": _u(r"\u8fd0\u884c\u6b21\u6570"),
    "mean_score": _u(r"\u5e73\u5747\u5206"),
    "evaluator": _u(r"\u8bc4\u5206\u5668"),
    "cases": _u(r"\u7528\u4f8b\u6570"),
    "thinking": _u(r"\u601d\u8003\u6a21\u5f0f"),
    "enabled": _u(r"\u5f00\u542f"),
    "disabled": _u(r"\u5173\u95ed"),
    "overview": _u(r"\u6838\u5fc3\u6307\u6807"),
    "overall": _u(r"\u5f97\u5206\u5206\u5e03"),
    "max_score": _u(r"\u6700\u9ad8\u5206"),
    "min_score": _u(r"\u6700\u4f4e\u5206"),
    "type_perf": _u(r"\u5206\u7c7b\u578b\u8868\u73b0"),
    "type": _u(r"\u7c7b\u578b"),
    "sample_count": _u(r"\u6837\u672c\u6570"),
    "good_count": _u(r"Good \u6570"),
    "bad_count": _u(r"Bad Case \u6570"),
    "multi_detail": _u(r"Multi \u6a21\u5f0f\u9010\u6b21\u8868\u73b0"),
    "needle_depth": _u(r"Needle \u6df1\u5ea6"),
    "single_heatmap": _u(r"Single \u6a21\u5f0f\u70ed\u529b\u56fe"),
    "bad_examples": _u(r"\u4f4e\u5206\u6837\u4f8b"),
    "score": _u(r"\u5206\u6570"),
    "question": _u(r"\u95ee\u9898"),
    "ground_truth": _u(r"\u6807\u51c6\u7b54\u6848"),
    "response": _u(r"\u6a21\u578b\u56de\u7b54"),
    "ai_failed": _u(r"AI \u5206\u6790\u751f\u6210\u5931\u8d25\uff1a"),
    "missing_key": _u(r"\u7f3a\u5c11 DS_API_KEY\uff0c\u5df2\u8df3\u8fc7 AI \u6587\u672c\u5206\u6790\u3002"),
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
        system = (
            "You are a rigorous AI product evaluation analyst. "
            "Analyze only the provided structured metrics and bad cases. "
            "Do not invent missing data. Write in Chinese for an AI product manager."
        )
        user = f"""
Generate the textual analysis section for an HTML evaluation dashboard.

Required sections:
1. Overall conclusion
2. Main capability gaps
3. Bad Case attribution
4. Actionable optimization suggestions

Constraints:
- Write in Chinese.
- Do not output full HTML.
- You may use short headings and bullet points.
- Do not invent model comparisons, user behavior, business outcomes, or unprovided causes.

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
        return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{LABEL["title"]}</title>
  <style>
    :root {{
      --ink: #18212f;
      --muted: #6c7687;
      --paper: #f4f0e8;
      --panel: rgba(255,255,255,.86);
      --line: #ded7cc;
      --good: #1f8f5f;
      --partial: #b97916;
      --fail: #c3423f;
      --navy: #14213d;
      --cyan: #0c8b95;
      --gold: #e5a93f;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      background:
        linear-gradient(135deg, rgba(20,33,61,.08) 0 25%, transparent 25% 50%, rgba(229,169,63,.12) 50% 75%, transparent 75%) 0 0/36px 36px,
        radial-gradient(circle at 12% 8%, rgba(12,139,149,.18), transparent 32%),
        radial-gradient(circle at 88% 18%, rgba(229,169,63,.22), transparent 30%),
        var(--paper);
      font-family: Georgia, "Times New Roman", "Microsoft YaHei", serif;
      line-height: 1.58;
    }}
    main {{ max-width: 1240px; margin: 0 auto; padding: 34px 22px 62px; }}
    header {{
      position: relative;
      display: grid;
      grid-template-columns: 1.4fr .6fr;
      gap: 24px;
      align-items: end;
      padding: 30px 0 24px;
      border-bottom: 3px double var(--navy);
      margin-bottom: 22px;
    }}
    .eyebrow {{ color: var(--cyan); font: 700 12px/1.2 "Consolas", monospace; letter-spacing: .12em; text-transform: uppercase; }}
    h1 {{ margin: 8px 0 8px; font-size: 44px; line-height: 1.02; letter-spacing: 0; }}
    h2 {{ margin: 0 0 14px; font-size: 20px; }}
    .stamp {{ justify-self: end; border: 2px solid var(--navy); padding: 14px 16px; background: rgba(255,255,255,.56); box-shadow: 8px 8px 0 rgba(20,33,61,.11); }}
    .stamp strong {{ display: block; font-size: 28px; }}
    .muted {{ color: var(--muted); }}
    section {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 20px;
      margin-bottom: 18px;
      box-shadow: 0 18px 45px rgba(42, 34, 22, .08);
      backdrop-filter: blur(10px);
    }}
    .grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }}
    .card {{
      background: linear-gradient(180deg, rgba(255,255,255,.92), rgba(255,255,255,.68));
      border: 1px solid var(--line);
      border-left: 4px solid var(--cyan);
      border-radius: 7px;
      padding: 15px;
      min-height: 96px;
    }}
    .metric-label {{ color: var(--muted); font: 700 12px/1.2 "Consolas", monospace; text-transform: uppercase; }}
    .metric-value {{ font-size: 25px; font-weight: 700; margin-top: 8px; overflow-wrap: anywhere; }}
    table {{ width: 100%; border-collapse: collapse; background: rgba(255,255,255,.52); }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 11px 9px; text-align: left; vertical-align: top; }}
    th {{ color: var(--muted); font: 700 12px/1.2 "Consolas", monospace; text-transform: uppercase; }}
    tr:hover td {{ background: rgba(12,139,149,.06); }}
    .bar-row {{ display: grid; grid-template-columns: 150px 1fr 56px; align-items: center; gap: 10px; margin: 10px 0; }}
    .bar-label {{ color: var(--muted); font: 700 12px/1.2 "Consolas", monospace; }}
    .bar-track {{ height: 18px; background: #e8e0d3; border: 1px solid rgba(20,33,61,.08); border-radius: 3px; overflow: hidden; }}
    .bar-fill {{ height: 100%; border-radius: 2px; }}
    .score-pill {{ text-align: right; font-weight: 800; }}
    .analysis {{ white-space: pre-wrap; font-size: 16px; }}
    .heatmap-wrap {{ overflow-x: auto; }}
    .heatmap td, .heatmap th {{ text-align: center; white-space: nowrap; }}
    .heat-cell {{ font-weight: 800; border-radius: 4px; color: #102033; border: 1px solid rgba(20,33,61,.1); }}
    @media (max-width: 880px) {{
      header {{ grid-template-columns: 1fr; }}
      .stamp {{ justify-self: start; }}
      .grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .bar-row {{ grid-template-columns: 1fr; }}
      h1 {{ font-size: 34px; }}
    }}
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
  {mode_section}
  {self._render_bad_cases(metrics)}
  <section>
    <h2>{LABEL["analysis_title"]}</h2>
    <div class="analysis">{_e(analysis)}</div>
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
                    cells.append(
                        f'<td><div class="heat-cell" style="background:{_heat_color(float(score))};padding:8px;">{float(score):.1f}</div></td>'
                    )
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
        return "#f3b0a7"
    if score < 8:
        return "#f0d68b"
    return "#95d3ae"


def _e(value: Any) -> str:
    return html.escape("" if value is None else str(value))
