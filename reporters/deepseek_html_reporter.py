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
    "title": 'LLM 能力评测报告',
    "subtitle": '结构化数据看板与 AI 产品分析',
    "generated_at": '生成时间',
    "analysis_title": 'AI 产品评测分析',
    "analysis_model": '分析模型',
    "model": '主测模型',
    "test_mode": '测试模式',
    "total_runs": '运行次数',
    "mean_score": '平均分',
    "evaluator": '评分器',
    "cases": '用例数',
    "thinking": '思考模式',
    "enabled": '开启',
    "disabled": '关闭',
    "overview": '核心指标',
    "overall": '得分分布',
    "max_score": '最高分',
    "min_score": '最低分',
    "type_perf": '分类型表现',
    "type": '类型',
    "sample_count": '样本数',
    "good_count": 'Good 数',
    "bad_count": 'Bad Case 数',
    "multi_detail": 'Multi 模式逐次表现',
    "needle_depth": 'Needle 深度',
    "single_heatmap": 'Single 模式热力图',
    "bad_examples": '低分样例',
    "score": '分数',
    "question": '问题',
    "ground_truth": '标准答案',
    "response": '模型回答',
    "ai_failed": 'AI 分析生成失败：',
    "missing_key": '缺少 DS_API_KEY，已跳过 AI 文本分析。',
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
            '你是严谨的 AI 产品评测分析师，只负责撰写报告中的文字解读。'
            + '禁止输出 HTML、XML、CSS、JavaScript、代码块或完整网页结构。'
            + '只能输出普通 Markdown 文本：短标题、短段落和项目符号。'
            + '所有结论必须基于给定的结构化指标、Bad Case 和工具链诊断，不要编造缺失数据。'
        )
        user = f"""
{'请为评测数据看板生成“AI 产品评测分析”文字，不要生成 HTML。'}

{'必须包含以下四个部分：'}
## {'总体结论'}
## {'主要能力短板'}
## Bad Case {'归因'}
## {'可执行优化建议'}

{'输出约束：'}
- {'只能使用 Markdown 文本。'}
- {'禁止输出任何 HTML 标签，例如 <html>、<body>、<section>、<div>、<table>。'}
- {'禁止输出 ```html 或其它代码块。'}
- {'不要写“下面是 HTML 代码”之类的话。'}
- {'不要编造模型对比、用户行为、业务结果或未提供的原因。'}
- {'必须结合工具链诊断，说明失败更可能发生在检索、结构化抽取、工具执行、回退路径还是答案格式。'}

{'模式专属分析重点：'}
{mode_focus}

{'结构化评测数据：'}
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
                '- 这是单文档、单 needle 的长上下文扫描。\n'
                + '- 优先分析模型对上下文长度和插入深度的敏感区间。\n'
                + '- 识别 context_length x depth_percent 网格中的低分区域。\n'
                + '- 判断失败是否更像长上下文退化、中间位置遗忘，或定位到 needle 后的答案抽取失败。\n'
                + '- 优化建议需要落到提示词设计、检索兜底、上下文窗口策略和评测设置。'
            )
        if mode == "multi":
            return (
                '- 这是多文档、多 needle 的检索与推理评测。\n'
                + '- 优先分析跨文档检索、多 needle 聚合和证据组合能力。\n'
                + '- 判断失败是否更可能来自漏掉某个 needle、召回无关 chunk、检索后的算术/日期/字符串推理错误，或答案格式漂移。\n'
                + '- needle 深度统计只能作为辅助证据；没有数据支持时，不要过度归因于深度。\n'
                + '- 优化建议需要落到混合检索、rerank 阈值、邻近 chunk 补充、工具增强和场景化推理。'
            )
        return (
            '- 当前测试模式未知。请保持保守，只分析显式指标和 Bad Case。\n'
            + '- 除非数据明确支持，否则不要推断模式专属原因。'
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
      --bg: #071018;
      --bg-2: #0d1723;
      --panel: rgba(13, 24, 36, .84);
      --panel-strong: rgba(18, 31, 46, .96);
      --ink: #e8f1f8;
      --muted: #8ea1b5;
      --line: rgba(123, 184, 210, .18);
      --line-strong: rgba(127, 211, 255, .38);
      --good: #35d18d;
      --partial: #f5b84b;
      --fail: #ff5f6d;
      --cyan: #42d6ff;
      --blue: #6c8cff;
      --violet: #9c7cff;
      --chip: rgba(66, 214, 255, .11);
      --shadow: 0 22px 70px rgba(0, 0, 0, .34);
    }}
    * {{ box-sizing: border-box; }}
    html {{ scroll-behavior: smooth; }}
    body {{
      margin: 0;
      color: var(--ink);
      background:
        linear-gradient(90deg, rgba(255,255,255,.035) 1px, transparent 1px) 0 0/42px 42px,
        linear-gradient(0deg, rgba(255,255,255,.03) 1px, transparent 1px) 0 0/42px 42px,
        radial-gradient(circle at 18% 8%, rgba(66, 214, 255, .22), transparent 34%),
        radial-gradient(circle at 86% 12%, rgba(156, 124, 255, .18), transparent 30%),
        linear-gradient(145deg, var(--bg), var(--bg-2) 48%, #050b11);
      font-family: "Aptos", "Segoe UI", "Microsoft YaHei", sans-serif;
      line-height: 1.58;
      min-height: 100vh;
    }}
    body::before {{
      content: "";
      position: fixed;
      inset: 0;
      pointer-events: none;
      background: linear-gradient(180deg, rgba(255,255,255,.04), transparent 18%, rgba(0,0,0,.24));
      mix-blend-mode: screen;
    }}
    main {{ max-width: 1280px; margin: 0 auto; padding: 36px 22px 70px; position: relative; }}
    header {{
      position: relative;
      display: grid;
      grid-template-columns: 1.4fr .6fr;
      gap: 24px;
      align-items: end;
      padding: 36px 0 30px;
      margin-bottom: 18px;
    }}
    header::after {{
      content: "";
      position: absolute;
      left: 0;
      right: 0;
      bottom: 0;
      height: 1px;
      background: linear-gradient(90deg, transparent, var(--cyan), var(--violet), transparent);
      box-shadow: 0 0 26px rgba(66, 214, 255, .45);
    }}
    .eyebrow {{
      display: inline-flex;
      align-items: center;
      gap: 10px;
      color: var(--cyan);
      font: 800 12px/1.2 "Cascadia Mono", "Consolas", monospace;
      letter-spacing: .14em;
      text-transform: uppercase;
    }}
    .eyebrow::before {{ content: ""; width: 9px; height: 9px; border-radius: 50%; background: var(--good); box-shadow: 0 0 16px var(--good); }}
    h1 {{ margin: 10px 0 10px; font-size: clamp(34px, 5vw, 62px); line-height: .98; letter-spacing: 0; font-weight: 850; }}
    h2 {{ margin: 0 0 16px; font-size: 18px; letter-spacing: .02em; }}
    .stamp {{
      justify-self: end;
      min-width: 220px;
      border: 1px solid var(--line-strong);
      border-radius: 8px;
      padding: 17px 18px;
      background: linear-gradient(180deg, rgba(66,214,255,.13), rgba(10,20,31,.78));
      box-shadow: var(--shadow), inset 0 1px 0 rgba(255,255,255,.08);
    }}
    .stamp strong {{ display: block; font-size: 42px; line-height: 1; margin: 7px 0; color: #fff; text-shadow: 0 0 24px rgba(66,214,255,.42); }}
    .muted {{ color: var(--muted); }}
    section {{
      position: relative;
      background: linear-gradient(180deg, var(--panel), rgba(8, 17, 27, .88));
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 22px;
      margin-bottom: 16px;
      box-shadow: var(--shadow);
      backdrop-filter: blur(14px);
      overflow: hidden;
    }}
    section::before {{
      content: "";
      position: absolute;
      inset: 0 0 auto 0;
      height: 2px;
      background: linear-gradient(90deg, var(--cyan), transparent 38%, var(--violet));
      opacity: .72;
    }}
    .grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }}
    .card {{
      background: linear-gradient(180deg, rgba(255,255,255,.07), rgba(255,255,255,.025));
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 15px;
      min-height: 96px;
      box-shadow: inset 0 1px 0 rgba(255,255,255,.06);
    }}
    .card:hover {{ border-color: var(--line-strong); transform: translateY(-1px); transition: .18s ease; }}
    .metric-label {{ color: var(--muted); font: 800 11px/1.2 "Cascadia Mono", "Consolas", monospace; text-transform: uppercase; letter-spacing: .08em; }}
    .metric-value {{ font-size: 25px; font-weight: 780; margin-top: 9px; overflow-wrap: anywhere; color: #f7fbff; }}
    table {{ width: 100%; border-collapse: separate; border-spacing: 0; background: rgba(2,8,14,.22); border: 1px solid var(--line); border-radius: 8px; overflow: hidden; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 12px 10px; text-align: left; vertical-align: top; }}
    tr:last-child td {{ border-bottom: 0; }}
    th {{ color: #b7c7d8; font: 800 11px/1.2 "Cascadia Mono", "Consolas", monospace; text-transform: uppercase; letter-spacing: .08em; background: rgba(66,214,255,.06); }}
    td {{ color: #d8e4ee; }}
    tr:hover td {{ background: rgba(66,214,255,.055); }}
    .bar-row {{ display: grid; grid-template-columns: minmax(132px, 210px) 1fr 56px; align-items: center; gap: 12px; margin: 11px 0; }}
    .bar-label {{ color: var(--muted); font: 800 11px/1.2 "Cascadia Mono", "Consolas", monospace; overflow-wrap: anywhere; }}
    .bar-track {{ height: 18px; background: rgba(255,255,255,.06); border: 1px solid var(--line); border-radius: 999px; overflow: hidden; box-shadow: inset 0 0 16px rgba(0,0,0,.28); }}
    .bar-fill {{ height: 100%; border-radius: 999px; box-shadow: 0 0 18px currentColor; }}
    .score-pill {{ text-align: right; font-weight: 850; color: #fff; font-variant-numeric: tabular-nums; }}
    .analysis {{
      display: grid;
      gap: 14px;
      font-size: 15px;
      color: #dbe8f3;
    }}
    .analysis h3 {{
      margin: 4px 0 2px;
      color: #ffffff;
      font-size: 18px;
      letter-spacing: .01em;
    }}
    .analysis p {{ margin: 0; color: #cfdae5; }}
    .analysis ul {{ margin: 0; padding-left: 20px; display: grid; gap: 7px; }}
    .analysis li::marker {{ color: var(--cyan); }}
    .analysis strong {{ color: #fff; font-weight: 850; }}
    .heatmap-wrap {{ overflow-x: auto; }}
    .heatmap td, .heatmap th {{ text-align: center; white-space: nowrap; }}
    .heat-cell {{ font-weight: 850; border-radius: 6px; color: #081018; border: 1px solid rgba(255,255,255,.28); box-shadow: inset 0 1px 0 rgba(255,255,255,.22); }}
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
  <h2>工具链诊断</h2>
  <table>
    <thead><tr><th>维度</th><th>标签</th><th>次数</th></tr></thead>
    <tbody>
      {rows(path_counts, "执行路径")}
      {rows(task_counts, "任务类型")}
      {rows(fallback_counts, "回退原因")}
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
        return "#ff7c86"
    if score < 8:
        return "#f7c75b"
    return "#55d99d"


def _e(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def _markdown_to_html(text: str) -> str:
    """Render a small, safe subset of Markdown used by report analysis output."""
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

    # If the model ignored instructions and produced HTML, keep the text content
    # rather than rendering source code or trusting model-generated markup.
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
