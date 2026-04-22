"""
Aggregate human eval + LLM judge results from model_comparison_scored-2.csv
Produces:
  - Console summary table
  - plots/  (6 charts saved as PNG)

Usage:
  python analyze_results.py
  python analyze_results.py --csv model_comparison_scored-2.csv
"""

import argparse
import os
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
TIER_ORDER  = ["low", "mid", "high"]
TIER_COLORS = {"low": "#60A5FA", "mid": "#F59E0B", "high": "#22C55E"}
AUTO_METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
METRIC_LABELS = {
    "faithfulness":      "Faithfulness",
    "answer_relevancy":  "Relevancy",
    "context_precision": "Ctx Precision",
    "context_recall":    "Ctx Recall",
}

plt.rcParams.update({
    "font.family":      "sans-serif",
    "font.size":        11,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "figure.dpi":       140,
})


# ── Load ──────────────────────────────────────────────────────────────────────
def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Normalise types
    df["human_eval_sample"] = df["human_eval_sample"].astype(str).str.strip().str.lower() == "true"
    df["human_score"]       = pd.to_numeric(df["human_score"],  errors="coerce")
    df["model_tier"]        = df["model_tier"].str.strip().str.lower()
    df["model_id"]          = df["model_id"].str.strip()
    # Keep only valid tiers
    df = df[df["model_tier"].isin(TIER_ORDER)]
    for m in AUTO_METRICS:
        df[m] = pd.to_numeric(df[m], errors="coerce")
    return df


# ── Aggregate ─────────────────────────────────────────────────────────────────
def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for mid, grp in df.groupby("model_id"):
        tier  = grp["model_tier"].iloc[0]
        label = grp["model_label"].iloc[0] if "model_label" in grp.columns else mid

        all_rows    = grp
        human_rows  = grp[grp["human_eval_sample"] == True]
        scored_rows = human_rows[human_rows["human_score"].notna()]

        row = {
            "model_id":    mid,
            "model_label": label,
            "tier":        tier,
            "total_rows":  len(all_rows),
            "human_flagged": len(human_rows),
            "human_scored":  len(scored_rows),
            "human_score_avg":  scored_rows["human_score"].mean(),
            "human_score_std":  scored_rows["human_score"].std(),
            "avg_response_time": all_rows["response_time_s"].mean() if "response_time_s" in all_rows else None,
            "total_cost_usd":    all_rows["cost_usd"].sum()         if "cost_usd"        in all_rows else None,
            "avg_input_tokens":  all_rows["input_tokens"].mean()    if "input_tokens"    in all_rows else None,
            "avg_output_tokens": all_rows["output_tokens"].mean()   if "output_tokens"   in all_rows else None,
        }
        for m in AUTO_METRICS:
            row[f"{m}_avg"] = all_rows[m].mean()
            row[f"{m}_std"] = all_rows[m].std()

        rows.append(row)

    agg = pd.DataFrame(rows)
    agg["tier_order"] = agg["tier"].map({t: i for i, t in enumerate(TIER_ORDER)})
    agg = agg.sort_values("tier_order").drop(columns="tier_order").reset_index(drop=True)
    return agg


# ── Print summary ─────────────────────────────────────────────────────────────
def print_summary(agg: pd.DataFrame):
    print("\n" + "=" * 80)
    print("  MODEL COMPARISON SUMMARY")
    print("=" * 80)

    # Human scores
    print("\n📊 Human Evaluation (flagged samples only)")
    print("-" * 60)
    h = agg[["model_id", "tier", "human_flagged", "human_scored", "human_score_avg", "human_score_std"]].copy()
    h.columns = ["Model", "Tier", "Flagged", "Scored", "Avg ★", "Std"]
    h["Avg ★"] = h["Avg ★"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "—")
    h["Std"]   = h["Std"].map(lambda x: f"±{x:.2f}" if pd.notna(x) else "—")
    print(h.to_string(index=False))

    # Auto metrics
    print("\n🤖 LLM Judge Scores (all rows)")
    print("-" * 70)
    cols = ["model_id", "tier"] + [f"{m}_avg" for m in AUTO_METRICS]
    a = agg[cols].copy()
    a.columns = ["Model", "Tier"] + [METRIC_LABELS[m] for m in AUTO_METRICS]
    for c in a.columns[2:]:
        a[c] = a[c].map(lambda x: f"{x:.3f}" if pd.notna(x) else "—")
    print(a.to_string(index=False))

    # Cost & speed
    if "total_cost_usd" in agg.columns:
        print("\n💰 Cost & Speed")
        print("-" * 60)
        cs = agg[["model_id", "tier", "total_cost_usd", "avg_response_time", "avg_output_tokens"]].copy()
        cs.columns = ["Model", "Tier", "Total Cost ($)", "Avg RT (s)", "Avg Out Tokens"]
        cs["Total Cost ($)"]  = cs["Total Cost ($)"].map(lambda x: f"${x:.4f}" if pd.notna(x) else "—")
        cs["Avg RT (s)"]      = cs["Avg RT (s)"].map(lambda x: f"{x:.2f}s" if pd.notna(x) else "—")
        cs["Avg Out Tokens"]  = cs["Avg Out Tokens"].map(lambda x: f"{int(x)}" if pd.notna(x) else "—")
        print(cs.to_string(index=False))

    print("\n" + "=" * 80 + "\n")


# ── Plots ─────────────────────────────────────────────────────────────────────
def short_label(label: str) -> str:
    """Shorten model label for axes."""
    return label.split("(")[0].strip()[:20]


def colors_for(agg: pd.DataFrame) -> list:
    return [TIER_COLORS.get(t, "#888") for t in agg["tier"]]


def plot_human_scores(agg: pd.DataFrame, out: str):
    scored = agg[agg["human_score_avg"].notna()].copy()
    if scored.empty:
        print("⚠️  No human scores found — skipping human score plot"); return

    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels  = [short_label(r["model_label"]) for _, r in scored.iterrows()]
    means   = scored["human_score_avg"].values
    stds    = scored["human_score_std"].fillna(0).values
    cols    = colors_for(scored)
    x       = np.arange(len(labels))

    bars = ax.bar(x, means, color=cols, width=0.55, zorder=3, edgecolor="white", linewidth=1.2)
    ax.errorbar(x, means, yerr=stds, fmt="none", color="#333", capsize=5, linewidth=1.5, zorder=4)

    for bar, v in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.06, f"{v:.2f}",
                ha="center", va="bottom", fontsize=10, fontweight="600")

    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 5.8); ax.set_ylabel("Average Human Score (1–5)")
    ax.set_title("Human Evaluation Scores by Model", fontsize=13, fontweight="600", pad=12)
    ax.axhline(means.mean(), color="#999", linestyle="--", linewidth=1, label=f"Mean: {means.mean():.2f}")
    ax.legend(fontsize=9)

    legend = [mpatches.Patch(color=TIER_COLORS[t], label=t.title()) for t in TIER_ORDER if t in scored["tier"].values]
    ax.legend(handles=legend + ax.get_legend_handles_labels()[0][-1:], fontsize=9)

    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  ✅ {out}")


def plot_auto_metrics_radar(agg: pd.DataFrame, out: str):
    metrics = AUTO_METRICS
    labels  = [METRIC_LABELS[m] for m in metrics]
    N       = len(metrics)
    angles  = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    for _, row in agg.iterrows():
        vals   = [row[f"{m}_avg"] for m in metrics]
        vals  += vals[:1]
        color  = TIER_COLORS.get(row["tier"], "#888")
        lbl    = short_label(row["model_label"])
        ax.plot(angles, vals, "o-", linewidth=2, color=color, label=lbl)
        ax.fill(angles, vals, alpha=0.08, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], fontsize=8)
    ax.set_title("LLM Judge Metrics — Radar", fontsize=13, fontweight="600", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.12), fontsize=9)

    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  ✅ {out}")


def plot_auto_metrics_bar(agg: pd.DataFrame, out: str):
    metrics = AUTO_METRICS
    x       = np.arange(len(metrics))
    n       = len(agg)
    width   = 0.7 / n
    fig, ax = plt.subplots(figsize=(9, 5))

    for i, (_, row) in enumerate(agg.iterrows()):
        vals  = [row[f"{m}_avg"] for m in metrics]
        errs  = [row[f"{m}_std"] for m in metrics]
        color = TIER_COLORS.get(row["tier"], "#888")
        lbl   = short_label(row["model_label"])
        offset = (i - n / 2 + 0.5) * width
        bars  = ax.bar(x + offset, vals, width * 0.9, color=color, label=lbl,
                       zorder=3, edgecolor="white", linewidth=0.8)
        ax.errorbar(x + offset, vals, yerr=errs, fmt="none",
                    color="#444", capsize=3, linewidth=1, zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABELS[m] for m in metrics], fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score (0–1)")
    ax.set_title("LLM Judge Scores by Metric & Model", fontsize=13, fontweight="600", pad=12)
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  ✅ {out}")


def plot_cost_vs_quality(agg: pd.DataFrame, out: str):
    scored = agg[agg["human_score_avg"].notna() & agg["total_cost_usd"].notna()].copy()
    if scored.empty:
        print("⚠️  No cost+human data — skipping cost vs quality plot"); return

    fig, ax = plt.subplots(figsize=(7, 5))
    for _, row in scored.iterrows():
        color = TIER_COLORS.get(row["tier"], "#888")
        lbl   = short_label(row["model_label"])
        ax.scatter(row["total_cost_usd"], row["human_score_avg"],
                   s=160, color=color, zorder=3, edgecolors="white", linewidth=1.5)
        ax.annotate(lbl, (row["total_cost_usd"], row["human_score_avg"]),
                    textcoords="offset points", xytext=(8, 4), fontsize=9)

    ax.set_xlabel("Total Cost (USD)")
    ax.set_ylabel("Avg Human Score (1–5)")
    ax.set_title("Cost vs Human Quality", fontsize=13, fontweight="600", pad=12)
    legend = [mpatches.Patch(color=TIER_COLORS[t], label=t.title()) for t in TIER_ORDER if t in scored["tier"].values]
    ax.legend(handles=legend, fontsize=9)

    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  ✅ {out}")


def plot_response_time(agg: pd.DataFrame, out: str):
    if "avg_response_time" not in agg.columns: return
    data = agg[agg["avg_response_time"].notna()]
    if data.empty: return

    fig, ax = plt.subplots(figsize=(7, 4))
    labels  = [short_label(r["model_label"]) for _, r in data.iterrows()]
    vals    = data["avg_response_time"].values
    cols    = colors_for(data)
    x       = np.arange(len(labels))

    bars = ax.barh(x, vals, color=cols, height=0.5, zorder=3, edgecolor="white")
    for bar, v in zip(bars, vals):
        ax.text(v + 0.05, bar.get_y() + bar.get_height()/2, f"{v:.2f}s",
                va="center", fontsize=10)

    ax.set_yticks(x); ax.set_yticklabels(labels)
    ax.set_xlabel("Average Response Time (s)")
    ax.set_title("Response Time by Model", fontsize=13, fontweight="600", pad=12)
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  ✅ {out}")


def plot_human_score_dist(df: pd.DataFrame, out: str):
    """Box plot of human score distribution per model."""
    scored = df[df["human_eval_sample"] & df["human_score"].notna()]
    if scored.empty:
        print("⚠️  No human scores — skipping distribution plot"); return

    models  = scored["model_id"].unique()
    data    = [scored[scored["model_id"]==m]["human_score"].values for m in models]
    labels  = [short_label(scored[scored["model_id"]==m]["model_label"].iloc[0]) for m in models]
    tiers   = [scored[scored["model_id"]==m]["model_tier"].iloc[0] for m in models]
    cols    = [TIER_COLORS.get(t,"#888") for t in tiers]

    fig, ax = plt.subplots(figsize=(7, 5))
    bp = ax.boxplot(data, patch_artist=True, widths=0.45,
                    medianprops=dict(color="white", linewidth=2))
    for patch, color in zip(bp["boxes"], cols):
        patch.set_facecolor(color); patch.set_alpha(0.75)
    for element in ["whiskers","caps","fliers"]:
        for item in bp[element]: item.set(color="#555")

    # overlay jitter
    for i, (d, c) in enumerate(zip(data, cols), 1):
        jitter = np.random.uniform(-0.12, 0.12, len(d))
        ax.scatter(i + jitter, d, color=c, alpha=0.6, s=30, zorder=3)

    ax.set_xticks(range(1, len(labels)+1))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0.5, 5.5)
    ax.set_ylabel("Human Score")
    ax.set_title("Human Score Distribution by Model", fontsize=13, fontweight="600", pad=12)

    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  ✅ {out}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="model_comparison_scored-2.csv")
    parser.add_argument("--out", default="plots")
    args = parser.parse_args()

    print(f"📂 Loading {args.csv} …")
    df  = load(args.csv)
    agg = aggregate(df)

    print_summary(agg)

    agg.to_csv("model_comparison_aggregated.csv", index=False)
    print("✅ Aggregated table → model_comparison_aggregated.csv\n")

    os.makedirs(args.out, exist_ok=True)
    print("📈 Generating plots …")
    plot_human_scores(agg,         f"{args.out}/01_human_scores.png")
    plot_auto_metrics_bar(agg,     f"{args.out}/02_auto_metrics_bar.png")
    plot_auto_metrics_radar(agg,   f"{args.out}/03_auto_metrics_radar.png")
    plot_cost_vs_quality(agg,      f"{args.out}/04_cost_vs_quality.png")
    plot_response_time(agg,        f"{args.out}/05_response_time.png")
    plot_human_score_dist(df,      f"{args.out}/06_human_score_dist.png")

    print(f"\n✅ All plots saved to {args.out}/")


if __name__ == "__main__":
    main()