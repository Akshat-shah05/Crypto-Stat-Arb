# experiments/build_analysis_pack.py

import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
PLOTS_DIR = ROOT / "plots"
OUT_DIR = RESULTS_DIR / "analysis_pack"
OUT_PLOTS = OUT_DIR / "plots"

OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PLOTS.mkdir(parents=True, exist_ok=True)


def strategy_label(row):
    return f"{row['kind']}_{int(row['horizon_hours'])}h_{row['condition']}"


def save_csv(df, name):
    path = OUT_DIR / name
    df.to_csv(path, index=False)
    print(f"Saved {path}")


def plot_heatmap(summary, kind, metric="sharpe"):
    df = summary[summary["kind"] == kind].copy()

    pivot = df.pivot_table(
        index="horizon_hours",
        columns="condition",
        values=metric,
        aggfunc="mean",
    ).sort_index()

    plt.figure(figsize=(12, 6))
    plt.imshow(pivot.values, aspect="auto")
    plt.colorbar(label=metric)

    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
    plt.yticks(range(len(pivot.index)), pivot.index.astype(int))

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.iloc[i, j]
            if pd.notna(val):
                plt.text(j, i, f"{val:.2f}", ha="center", va="center")

    plt.title(f"{kind.title()} {metric} Heatmap")
    plt.xlabel("Condition")
    plt.ylabel("Horizon Hours")
    plt.tight_layout()

    out = OUT_PLOTS / f"{kind}_{metric}_heatmap.png"
    plt.savefig(out)
    plt.close()
    print(f"Saved {out}")


def plot_turnover_vs_sharpe(summary):
    plt.figure(figsize=(9, 6))

    for condition, g in summary.groupby("condition"):
        plt.scatter(g["avg_turnover"], g["sharpe"], label=condition, alpha=0.75)

    plt.xlabel("Average Turnover")
    plt.ylabel("Net Sharpe")
    plt.title("Turnover vs Net Sharpe")
    plt.legend(fontsize=8)
    plt.tight_layout()

    out = OUT_PLOTS / "turnover_vs_sharpe.png"
    plt.savefig(out)
    plt.close()
    print(f"Saved {out}")


def plot_cost_drag(summary):
    df = summary.copy()
    df["strategy"] = df.apply(strategy_label, axis=1)

    top = df.sort_values("total_cost_drag", ascending=False).head(20)

    plt.figure(figsize=(12, 6))
    plt.bar(top["strategy"], top["total_cost_drag"])
    plt.xticks(rotation=75, ha="right")
    plt.ylabel("Total Cost Drag")
    plt.title("Highest Cost Drag Strategies")
    plt.tight_layout()

    out = OUT_PLOTS / "cost_drag_top20.png"
    plt.savefig(out)
    plt.close()
    print(f"Saved {out}")


def plot_btc_beta(summary):
    df = summary.copy()
    df["strategy"] = df.apply(strategy_label, axis=1)
    df["abs_beta"] = df["beta_to_btc"].abs()

    top = df.sort_values("abs_beta", ascending=False).head(20)

    plt.figure(figsize=(12, 6))
    plt.bar(top["strategy"], top["beta_to_btc"])
    plt.xticks(rotation=75, ha="right")
    plt.ylabel("BTC Beta")
    plt.title("Highest Absolute BTC Beta Strategies")
    plt.tight_layout()

    out = OUT_PLOTS / "btc_beta_top20.png"
    plt.savefig(out)
    plt.close()
    print(f"Saved {out}")


def find_timeseries_file(kind, horizon, condition):
    pattern = f"{kind}_{int(horizon)}h_{condition}_*bps_timeseries.csv"
    matches = list(RESULTS_DIR.glob(pattern))
    return matches[0] if matches else None


def plot_unconditional_equity_and_drawdowns(summary, kind):
    uncond = summary[
        (summary["kind"] == kind)
        & (summary["condition"] == "unconditional")
    ].sort_values("horizon_hours")

    if uncond.empty:
        return

    plt.figure(figsize=(11, 6))

    for _, row in uncond.iterrows():
        f = find_timeseries_file(row["kind"], row["horizon_hours"], row["condition"])
        if f is None:
            continue

        ts = pd.read_csv(f)
        ts["timestamp"] = pd.to_datetime(ts["timestamp"])
        plt.plot(ts["timestamp"], ts["net_equity"], label=f"{int(row['horizon_hours'])}h")

    plt.title(f"Unconditional {kind.title()} Net Equity")
    plt.xlabel("Date")
    plt.ylabel("Net Equity")
    plt.legend()
    plt.tight_layout()

    out = OUT_PLOTS / f"unconditional_{kind}_equity.png"
    plt.savefig(out)
    plt.close()
    print(f"Saved {out}")

    plt.figure(figsize=(11, 6))

    for _, row in uncond.iterrows():
        f = find_timeseries_file(row["kind"], row["horizon_hours"], row["condition"])
        if f is None:
            continue

        ts = pd.read_csv(f)
        ts["timestamp"] = pd.to_datetime(ts["timestamp"])

        equity = ts["net_equity"]
        drawdown = equity / equity.cummax() - 1

        plt.plot(ts["timestamp"], drawdown, label=f"{int(row['horizon_hours'])}h")

    plt.title(f"Unconditional {kind.title()} Drawdowns")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.legend()
    plt.tight_layout()

    out = OUT_PLOTS / f"unconditional_{kind}_drawdown.png"
    plt.savefig(out)
    plt.close()
    print(f"Saved {out}")


def build_condition_comparison(summary):
    base = summary[summary["condition"] == "unconditional"].copy()

    key_cols = ["kind", "horizon_hours"]
    base_cols = key_cols + [
        "sharpe",
        "annualized_return",
        "max_drawdown",
        "avg_turnover",
        "total_cost_drag",
        "beta_to_btc",
    ]

    base = base[base_cols].rename(columns={
        "sharpe": "uncond_sharpe",
        "annualized_return": "uncond_annualized_return",
        "max_drawdown": "uncond_max_drawdown",
        "avg_turnover": "uncond_avg_turnover",
        "total_cost_drag": "uncond_total_cost_drag",
        "beta_to_btc": "uncond_beta_to_btc",
    })

    comp = summary[summary["condition"] != "unconditional"].merge(
        base,
        on=key_cols,
        how="left",
    )

    comp["delta_sharpe"] = comp["sharpe"] - comp["uncond_sharpe"]
    comp["delta_annualized_return"] = (
        comp["annualized_return"] - comp["uncond_annualized_return"]
    )
    comp["delta_max_drawdown"] = comp["max_drawdown"] - comp["uncond_max_drawdown"]
    comp["delta_turnover"] = comp["avg_turnover"] - comp["uncond_avg_turnover"]
    comp["delta_cost_drag"] = comp["total_cost_drag"] - comp["uncond_total_cost_drag"]
    comp["delta_beta_to_btc"] = comp["beta_to_btc"] - comp["uncond_beta_to_btc"]

    comp["strategy"] = comp.apply(strategy_label, axis=1)

    return comp.sort_values("delta_sharpe", ascending=False)


def plot_cost_sensitivity_if_available():
    path = RESULTS_DIR / "cost_sensitivity.csv"

    if not path.exists():
        print("No cost_sensitivity.csv found. Skipping cost sensitivity plots.")
        return None

    df = pd.read_csv(path)

    plt.figure(figsize=(8, 5))
    plt.plot(df["cost_bps"], df["sharpe"], marker="o")
    plt.xlabel("Cost bps")
    plt.ylabel("Sharpe")
    plt.title("Sharpe vs Transaction Cost")
    plt.tight_layout()
    out = OUT_PLOTS / "cost_sensitivity_sharpe.png"
    plt.savefig(out)
    plt.close()
    print(f"Saved {out}")

    plt.figure(figsize=(8, 5))
    plt.plot(df["cost_bps"], df["annualized_return"], marker="o")
    plt.xlabel("Cost bps")
    plt.ylabel("Annualized Return")
    plt.title("Annualized Return vs Transaction Cost")
    plt.tight_layout()
    out = OUT_PLOTS / "cost_sensitivity_return.png"
    plt.savefig(out)
    plt.close()
    print(f"Saved {out}")

    positive = df[(df["sharpe"] > 0) & (df["annualized_return"] > 0)]
    breakeven = np.nan
    if not positive.empty:
        breakeven = positive["cost_bps"].max()

    return {
        "cost_sensitivity_rows": len(df),
        "breakeven_cost_bps_positive_sharpe_and_return": breakeven,
    }


def write_markdown_summary(summary, condition_comp, cost_info):
    summary = summary.copy()
    summary["strategy"] = summary.apply(strategy_label, axis=1)

    best = summary.sort_values("sharpe", ascending=False).head(5)
    worst = summary.sort_values("sharpe", ascending=True).head(5)
    best_conditioning = condition_comp.sort_values("delta_sharpe", ascending=False).head(5)
    worst_conditioning = condition_comp.sort_values("delta_sharpe", ascending=True).head(5)
    high_cost = summary.sort_values("total_cost_drag", ascending=False).head(5)
    high_beta = summary.assign(abs_beta=summary["beta_to_btc"].abs()).sort_values(
        "abs_beta",
        ascending=False,
    ).head(5)

    lines = []

    lines.append("# Analysis Summary\n")

    lines.append("## Top 5 Strategies by Net Sharpe\n")
    lines.append(best[[
        "strategy", "sharpe", "annualized_return", "max_drawdown",
        "avg_turnover", "total_cost_drag", "beta_to_btc", "alpha_t_stat"
    ]].to_markdown(index=False))
    lines.append("\n")

    lines.append("## Bottom 5 Strategies by Net Sharpe\n")
    lines.append(worst[[
        "strategy", "sharpe", "annualized_return", "max_drawdown",
        "avg_turnover", "total_cost_drag", "beta_to_btc", "alpha_t_stat"
    ]].to_markdown(index=False))
    lines.append("\n")

    lines.append("## Best Conditioning Improvements vs Unconditional\n")
    lines.append(best_conditioning[[
        "strategy", "sharpe", "uncond_sharpe", "delta_sharpe",
        "annualized_return", "delta_annualized_return",
        "avg_turnover", "delta_turnover"
    ]].to_markdown(index=False))
    lines.append("\n")

    lines.append("## Worst Conditioning Changes vs Unconditional\n")
    lines.append(worst_conditioning[[
        "strategy", "sharpe", "uncond_sharpe", "delta_sharpe",
        "annualized_return", "delta_annualized_return",
        "avg_turnover", "delta_turnover"
    ]].to_markdown(index=False))
    lines.append("\n")

    lines.append("## Highest Cost Drag Strategies\n")
    lines.append(high_cost[[
        "strategy", "sharpe", "annualized_return", "avg_turnover", "total_cost_drag"
    ]].to_markdown(index=False))
    lines.append("\n")

    lines.append("## Highest Absolute BTC Beta Strategies\n")
    lines.append(high_beta[[
        "strategy", "sharpe", "beta_to_btc", "r_squared", "alpha_t_stat"
    ]].to_markdown(index=False))
    lines.append("\n")

    if cost_info is not None:
        lines.append("## Cost Sensitivity\n")
        lines.append(str(cost_info))
        lines.append("\n")

    lines.append("## Interpretation Checklist\n")
    lines.append("- If high Sharpe strategies also have high turnover/cost drag, be skeptical.\n")
    lines.append("- If conditioning improves Sharpe versus unconditional, that is your main research result.\n")
    lines.append("- If BTC beta or R² is high, the strategy may be mostly crypto market exposure rather than stat arb.\n")
    lines.append("- If similar horizons/conditions work, the result is more credible than one isolated winner.\n")
    lines.append("- If gross-looking ideas die after costs, that is still a useful conclusion.\n")

    out = OUT_DIR / "analysis_summary.md"
    out.write_text("\n".join(lines))
    print(f"Saved {out}")


def main():
    summary_path = RESULTS_DIR / "experiment_summary.csv"

    if not summary_path.exists():
        raise FileNotFoundError(f"Missing {summary_path}")

    summary = pd.read_csv(summary_path)

    required_cols = [
        "kind",
        "horizon_hours",
        "condition",
        "sharpe",
        "annualized_return",
        "max_drawdown",
        "avg_turnover",
        "total_cost_drag",
        "beta_to_btc",
        "alpha_per_period",
        "alpha_t_stat",
        "r_squared",
    ]

    missing = [c for c in required_cols if c not in summary.columns]
    if missing:
        raise ValueError(f"Missing columns in experiment_summary.csv: {missing}")

    summary["strategy"] = summary.apply(strategy_label, axis=1)

    uncond = summary[summary["condition"] == "unconditional"].sort_values(
        ["kind", "horizon_hours"]
    )

    condition_comp = build_condition_comparison(summary)

    cost_turnover = summary[[
        "strategy", "kind", "horizon_hours", "condition",
        "sharpe", "annualized_return", "avg_turnover", "total_cost_drag"
    ]].sort_values("total_cost_drag", ascending=False)

    risk_exposure = summary[[
        "strategy", "kind", "horizon_hours", "condition",
        "max_drawdown", "beta_to_btc", "alpha_per_period",
        "alpha_t_stat", "r_squared"
    ]].sort_values("max_drawdown")

    top_by_sharpe = summary.sort_values("sharpe", ascending=False).head(15)
    bottom_by_sharpe = summary.sort_values("sharpe", ascending=True).head(15)
    worst_drawdown = summary.sort_values("max_drawdown", ascending=True).head(15)

    save_csv(uncond, "table_1_unconditional_performance.csv")
    save_csv(condition_comp, "table_2_conditioning_vs_unconditional.csv")
    save_csv(cost_turnover, "table_3_cost_turnover.csv")
    save_csv(risk_exposure, "table_4_risk_exposure.csv")
    save_csv(top_by_sharpe, "table_5_top_by_sharpe.csv")
    save_csv(bottom_by_sharpe, "table_6_bottom_by_sharpe.csv")
    save_csv(worst_drawdown, "table_7_worst_drawdowns.csv")

    for kind in ["momentum", "reversal"]:
        plot_heatmap(summary, kind, metric="sharpe")
        plot_unconditional_equity_and_drawdowns(summary, kind)

    plot_turnover_vs_sharpe(summary)
    plot_cost_drag(summary)
    plot_btc_beta(summary)

    cost_info = plot_cost_sensitivity_if_available()

    write_markdown_summary(summary, condition_comp, cost_info)

    print("\nDone. Look in:")
    print(f"  {OUT_DIR}")
    print(f"  {OUT_PLOTS}")


if __name__ == "__main__":
    main()