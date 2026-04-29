# experiments/run_all_experiments.py

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

import pandas as pd

from config import (
    PROCESSED_DIR,
    RESULTS_DIR,
    PLOTS_DIR,
    MARKET_ORDER_COST_BPS,
)
from features import add_features
from signals import add_signal
from portfolio import make_weights
from backtest import run_backtest
from metrics import summarize_returns, btc_alpha_beta
from plotting import plot_equity_curve, plot_drawdown


def apply_condition(df, signal_col, condition):
    df = df.copy()
    conditioned_col = f"{signal_col}_{condition}"

    if condition == "unconditional":
        df[conditioned_col] = df[signal_col]

    elif condition == "high_volume":
        df[conditioned_col] = df[signal_col].where(df["high_volume_shock"])

    elif condition == "low_volume":
        df[conditioned_col] = df[signal_col].where(df["low_volume_shock"])

    elif condition == "weekday":
        df[conditioned_col] = df[signal_col].where(~df["is_weekend"])

    elif condition == "weekend":
        df[conditioned_col] = df[signal_col].where(df["is_weekend"])

    elif condition == "high_market_vol":
        df[conditioned_col] = df[signal_col].where(df["high_market_vol"])

    elif condition == "low_market_vol":
        df[conditioned_col] = df[signal_col].where(~df["high_market_vol"])

    else:
        raise ValueError(f"Unknown condition: {condition}")

    return df, conditioned_col


def run_single_experiment(panel, kind, horizon, condition, cost_bps):
    df = add_signal(panel, kind=kind, horizon_hours=horizon)
    signal_col = f"{kind}_{horizon}h"

    df, conditioned_col = apply_condition(df, signal_col, condition)

    weights = make_weights(df, signal_col=conditioned_col)
    result = run_backtest(df, weights, cost_bps=cost_bps)

    ret_metrics = summarize_returns(result["net_return"])
    beta_metrics = btc_alpha_beta(result, df)

    summary = {
        "kind": kind,
        "horizon_hours": horizon,
        "condition": condition,
        "cost_bps": cost_bps,
        "avg_turnover": result["turnover"].mean(),
        "total_cost_drag": result["cost"].sum(),
        **ret_metrics,
        **beta_metrics,
    }

    name = f"{kind}_{horizon}h_{condition}_{cost_bps}bps"

    result.to_csv(RESULTS_DIR / f"{name}_timeseries.csv", index=False)

    plot_equity_curve(
        result,
        title=name,
        out_path=PLOTS_DIR / f"{name}_equity.png",
    )

    plot_drawdown(
        result,
        title=f"{name} Drawdown",
        out_path=PLOTS_DIR / f"{name}_drawdown.png",
    )

    return summary


def main():
    panel = pd.read_parquet(PROCESSED_DIR / "crypto_panel_1h.parquet")
    panel = add_features(panel)

    experiments = []

    signal_specs = [
        ("reversal", 1),
        ("reversal", 4),
        ("reversal", 12),
        ("reversal", 24),
        ("momentum", 24),
        ("momentum", 72),
        ("momentum", 168),
        ("momentum", 720),
    ]

    conditions = [
        "unconditional",
        "high_volume",
        "low_volume",
        "weekday",
        "weekend",
        "high_market_vol",
        "low_market_vol",
    ]

    for kind, horizon in signal_specs:
        for condition in conditions:
            print(f"Running {kind} {horizon}h {condition}")
            try:
                summary = run_single_experiment(
                    panel=panel,
                    kind=kind,
                    horizon=horizon,
                    condition=condition,
                    cost_bps=MARKET_ORDER_COST_BPS,
                )
                experiments.append(summary)
            except Exception as e:
                print(f"Failed: {kind} {horizon}h {condition}: {e}")

    summary_df = pd.DataFrame(experiments)
    summary_df.to_csv(RESULTS_DIR / "experiment_summary.csv", index=False)

    print(summary_df.sort_values("sharpe", ascending=False).head(20))


if __name__ == "__main__":
    main()