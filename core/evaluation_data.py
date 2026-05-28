import json
import os
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional


def build_evaluation_data(results: List[dict], summaries: List[dict], args: Any) -> Dict[str, Any]:
    metrics = build_metrics(results, summaries, args)
    return {
        "schema_version": 1,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "agent": str(getattr(args, "agent", "unknown")),
            "test_mode": str(getattr(args, "test_mode", "unknown")),
            "evaluator_type": str(getattr(args, "evaluator_type", "unknown")),
            "num_samples": getattr(args, "num_samples", None),
            "num_tests": getattr(args, "num_tests", None),
            "enable_thinking": bool(getattr(args, "enable_thinking", False)),
        },
        "metrics": metrics,
        "summaries": summaries,
        "results": results,
    }


def save_latest_evaluation_data(data: Dict[str, Any], path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return path


def load_evaluation_data(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_metrics(results: List[dict], summaries: List[dict], args: Any, report_model: Optional[str] = None) -> Dict[str, Any]:
    scores = [float(r.get("score", 0)) for r in results]
    model = str(results[0].get("model", "Unknown")) if results else "Unknown"
    test_mode = str(getattr(args, "test_mode", "unknown"))

    type_groups: Dict[str, List[float]] = defaultdict(list)
    for r in results:
        type_groups[str(r.get("test_case_type", "unknown"))].append(float(r.get("score", 0)))

    by_type = {
        t: {
            "count": len(vals),
            "mean_score": round(sum(vals) / len(vals), 2) if vals else 0,
            "bad_cases": sum(1 for v in vals if v < 4),
            "good_cases": sum(1 for v in vals if v >= 8),
        }
        for t, vals in sorted(type_groups.items())
    }

    outcome_counts = Counter(_score_bucket(s) for s in scores)
    metrics: Dict[str, Any] = {
        "model": model,
        "report_model": report_model,
        "agent": str(getattr(args, "agent", "unknown")),
        "test_mode": test_mode,
        "evaluator_type": str(getattr(args, "evaluator_type", "unknown")),
        "num_samples": getattr(args, "num_samples", None),
        "num_tests": getattr(args, "num_tests", None),
        "enable_thinking": bool(getattr(args, "enable_thinking", False)),
        "total_runs": len(results),
        "total_cases": len(summaries) or len({r.get("test_case_id") for r in results}),
        "mean_score": round(sum(scores) / len(scores), 2) if scores else 0,
        "min_score": round(min(scores), 2) if scores else 0,
        "max_score": round(max(scores), 2) if scores else 0,
        "outcome_counts": {
            "Good": outcome_counts.get("Good", 0),
            "Partial": outcome_counts.get("Partial", 0),
            "Fail": outcome_counts.get("Fail", 0),
        },
        "by_type": by_type,
        "bad_examples": _extract_bad_examples(results, limit=12),
        "tool_diagnostics": _tool_diagnostics(results),
    }

    if test_mode == "single":
        metrics["single_mode"] = _single_mode_metrics(results)
    else:
        metrics["multi_mode"] = _multi_mode_metrics(results)
        metrics["per_run_scores"] = _per_run_scores(results)

    return metrics


def _score_bucket(score: float) -> str:
    if score >= 8:
        return "Good"
    if score >= 4:
        return "Partial"
    return "Fail"


def _extract_bad_examples(results: List[dict], limit: int) -> List[Dict[str, Any]]:
    examples = []
    for r in sorted(results, key=lambda item: float(item.get("score", 0)))[:limit]:
        examples.append(
            {
                "test_case_id": r.get("test_case_id"),
                "type": r.get("test_case_type", "unknown"),
                "score": r.get("score"),
                "question": _truncate(str(r.get("question", "")), 160),
                "ground_truth": _truncate(str(r.get("ground_truth", "")), 120),
                "response": _truncate(str(r.get("model_response", "")), 200),
            }
        )
    return examples


def _single_mode_metrics(results: List[dict]) -> Dict[str, Any]:
    heatmap: Dict[float, Dict[int, float]] = defaultdict(dict)
    for r in results:
        if "context_length" not in r or "depth_percent" not in r:
            continue
        depth = round(float(r["depth_percent"]), 1)
        context_length = int(r["context_length"])
        heatmap[depth][context_length] = float(r.get("score", 0))

    return {
        "context_lengths": sorted({int(r.get("context_length", 0)) for r in results if "context_length" in r}),
        "depth_percents": sorted({round(float(r.get("depth_percent", 0)), 1) for r in results if "depth_percent" in r}),
        "heatmap": {depth: heatmap[depth] for depth in sorted(heatmap.keys())},
        "lowest_score_regions": [
            {
                "context_length": r.get("context_length"),
                "depth_percent": r.get("depth_percent"),
                "score": r.get("score"),
            }
            for r in sorted(results, key=lambda item: float(item.get("score", 0)))[:10]
        ],
    }


def _multi_mode_metrics(results: List[dict]) -> Dict[str, Any]:
    depths = []
    for r in results:
        for loc in r.get("needle_locations", []) or []:
            if "depth_percent" in loc:
                depths.append(float(loc["depth_percent"]))
    return {
        "avg_needle_depth": round(sum(depths) / len(depths), 2) if depths else None,
        "min_needle_depth": round(min(depths), 2) if depths else None,
        "max_needle_depth": round(max(depths), 2) if depths else None,
    }


def _per_run_scores(results: List[dict]) -> List[Dict[str, Any]]:
    rows = []
    for idx, r in enumerate(results, 1):
        rows.append(
            {
                "label": str(r.get("test_number", idx)),
                "score": float(r.get("score", 0)),
                "type": r.get("test_case_type", "unknown"),
                "case_id": r.get("test_case_id"),
            }
        )
    return rows


def _tool_diagnostics(results: List[dict]) -> Dict[str, Any]:
    traces = [r.get("agent_trace", {}) for r in results if isinstance(r.get("agent_trace"), dict)]
    path_counts = Counter(str(t.get("path", "unknown")) for t in traces)
    fallback_counts = Counter(str(t.get("fallback_reason")) for t in traces if t.get("fallback_reason"))
    task_counts = Counter(str(t.get("task_type", "unknown")) for t in traces)
    return {
        "path_counts": dict(path_counts),
        "fallback_counts": dict(fallback_counts),
        "task_counts": dict(task_counts),
    }


def _truncate(text: str, max_len: int) -> str:
    text = text.strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."
