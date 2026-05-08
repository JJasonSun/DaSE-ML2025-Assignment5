import os
from typing import List, Optional

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted")

PALETTE = sns.color_palette("muted")


def visualize_single_needle(results: List[dict], output_dir: str) -> Optional[str]:
    """Generate a heatmap for single-needle test results."""
    records = [r for r in results if "context_length" in r and "depth_percent" in r]
    if not records:
        print("[Visualize] No single-needle results to visualize.")
        return None

    df = pd.DataFrame(records)
    df["depth_percent"] = df["depth_percent"].round(1)
    df["context_length"] = df["context_length"].astype(int)
    df["score"] = df["score"].astype(float)

    pivot = df.pivot_table(index="depth_percent", columns="context_length", values="score", aggfunc="mean")

    fig, ax = plt.subplots(figsize=(12, 7))
    sns.heatmap(
        pivot, cmap="RdYlGn", annot=True, fmt=".1f", linewidths=0.5,
        cbar_kws={"label": "Score", "shrink": 0.8}, ax=ax,
        vmin=0, vmax=10,
    )
    model = df["model"].iloc[0] if "model" in df.columns else "Unknown"
    mean_score = df["score"].mean()
    ax.set_title(f"Single Needle — {model}  (mean={mean_score:.2f})", fontsize=14, pad=12)
    ax.set_xlabel("Context Length (tokens)", fontsize=11)
    ax.set_ylabel("Depth (%)", fontsize=11)
    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "single_needle_heatmap.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[Visualize] Saved heatmap to {path}")
    return path


def visualize_multi_needle(results: List[dict], output_dir: str) -> Optional[str]:
    """Generate a dashboard (bar + pie + stats) for multi-needle test results."""
    records = [r for r in results if "test_number" in r or "total_files" in r]
    if not records:
        print("[Visualize] No multi-needle results to visualize.")
        return None

    df = pd.DataFrame(records)
    model = df["model"].iloc[0] if "model" in df.columns else "Unknown"
    scores = df["score"].astype(float)
    mean_score = scores.mean()
    n = len(scores)

    # Categorize
    def categorize(s):
        if s >= 8:
            return "Good (>=8)"
        elif s >= 4:
            return "Partial (4-7)"
        else:
            return "Fail (<4)"

    df["Outcome"] = scores.apply(categorize)
    outcome_counts = df["Outcome"].value_counts()

    fig, (ax_bar, ax_pie) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [3, 2]})

    # --- Left: bar chart ---
    x_col = "test_number" if "test_number" in df.columns and not df["test_number"].isnull().all() else None
    if x_col is None:
        df["_idx"] = range(1, n + 1)
        x_col = "_idx"

    colors = [PALETTE[0] if s >= 8 else PALETTE[1] if s >= 4 else PALETTE[3] for s in scores]
    ax_bar.bar(df[x_col].astype(str), scores, color=colors, edgecolor="white", linewidth=0.5)
    ax_bar.axhline(mean_score, color="red", linestyle="--", linewidth=1.2, label=f"Mean={mean_score:.2f}")
    ax_bar.axhline(8, color="green", linestyle=":", linewidth=0.8, alpha=0.5, label="Good threshold")
    ax_bar.set_ylim(0, 10.5)
    ax_bar.set_xlabel("Test #", fontsize=11)
    ax_bar.set_ylabel("Score", fontsize=11)
    ax_bar.set_title("Per-Test Scores", fontsize=12)
    ax_bar.legend(fontsize=9)

    # --- Right: pie chart ---
    pie_colors = {"Good (>=8)": PALETTE[2], "Partial (4-7)": PALETTE[1], "Fail (<4)": PALETTE[3]}
    ordered_labels = ["Good (>=8)", "Partial (4-7)", "Fail (<4)"]
    sizes = [outcome_counts.get(l, 0) for l in ordered_labels]
    active_labels = [l for l, s in zip(ordered_labels, sizes) if s > 0]
    active_sizes = [s for s in sizes if s > 0]
    active_colors = [pie_colors[l] for l in active_labels]

    wedges, texts, autotexts = ax_pie.pie(
        active_sizes, labels=active_labels, autopct="%1.0f%%",
        colors=active_colors, startangle=90, textprops={"fontsize": 10},
    )
    for at in autotexts:
        at.set_fontweight("bold")
    ax_pie.set_title("Score Distribution", fontsize=12)

    fig.suptitle(f"Multi Needle — {model}  |  {n} tests, mean={mean_score:.2f}", fontsize=14, y=1.02)
    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"multi_needle_{model}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Visualize] Saved dashboard to {path}")
    return path


def visualize_bad_case_attribution(results: List[dict], output_dir: str) -> Optional[str]:
    """Generate a bad case attribution chart grouped by question type."""
    records = [r for r in results if "test_case_type" in r and "score" in r]
    if not records:
        print("[Visualize] No type-tagged results for bad case attribution.")
        return None

    df = pd.DataFrame(records)
    df["score"] = df["score"].astype(float)
    model = df["model"].iloc[0] if "model" in df.columns else "Unknown"

    # Aggregate by type
    by_type = df.groupby("test_case_type").agg(
        mean_score=("score", "mean"),
        count=("score", "size"),
        bad_count=("score", lambda s: (s < 4).sum()),
    ).sort_values("mean_score")

    if by_type.empty:
        return None

    fig, (ax_bar, ax_bad) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [3, 2]})

    # --- Left: mean score per type ---
    colors = [PALETTE[3] if m < 4 else PALETTE[1] if m < 8 else PALETTE[2] for m in by_type["mean_score"]]
    bars = ax_bar.barh(by_type.index, by_type["mean_score"], color=colors, edgecolor="white", linewidth=0.5)
    ax_bar.axvline(8, color="green", linestyle=":", linewidth=0.8, alpha=0.5, label="Good threshold")
    ax_bar.axvline(4, color="red", linestyle=":", linewidth=0.8, alpha=0.5, label="Fail threshold")
    ax_bar.set_xlim(0, 10.5)
    ax_bar.set_xlabel("Mean Score", fontsize=11)
    ax_bar.set_title("Mean Score by Question Type", fontsize=12)
    ax_bar.legend(fontsize=9)

    for bar, val in zip(bars, by_type["mean_score"]):
        ax_bar.text(val + 0.15, bar.get_y() + bar.get_height() / 2, f"{val:.1f}",
                    va="center", fontsize=10, fontweight="bold")

    # --- Right: bad case count per type ---
    has_bad = by_type["bad_count"].sum() > 0
    if has_bad:
        bad_colors = [PALETTE[3] if c > 0 else PALETTE[2] for c in by_type["bad_count"]]
        ax_bad.barh(by_type.index, by_type["bad_count"], color=bad_colors, edgecolor="white", linewidth=0.5)
        ax_bad.set_xlabel("Bad Cases (score < 4)", fontsize=11)
        ax_bad.set_title("Bad Case Count by Type", fontsize=12)
        for i, (idx, row) in enumerate(by_type.iterrows()):
            ax_bad.text(row["bad_count"] + 0.1, i, f"{int(row['bad_count'])}/{int(row['count'])}",
                        va="center", fontsize=10)
    else:
        ax_bad.text(0.5, 0.5, "No bad cases!", transform=ax_bad.transAxes,
                    ha="center", va="center", fontsize=14, color="green", fontweight="bold")
        ax_bad.set_title("Bad Case Count by Type", fontsize=12)
        ax_bad.set_yticks([])

    fig.suptitle(f"Bad Case Attribution — {model}", fontsize=14, y=1.02)
    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"bad_case_attribution_{model}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Visualize] Saved bad case attribution to {path}")
    return path


def auto_visualize(results: List[dict], test_mode: str, output_dir: str) -> List[str]:
    """Automatically visualize results based on test mode."""
    if not results:
        print("[Visualize] No results to visualize.")
        return []

    saved = []
    if test_mode == "single":
        path = visualize_single_needle(results, output_dir)
        if path:
            saved.append(path)
    else:
        path = visualize_multi_needle(results, output_dir)
        if path:
            saved.append(path)

    # Bad case attribution (both modes)
    path = visualize_bad_case_attribution(results, output_dir)
    if path:
        saved.append(path)

    return saved
