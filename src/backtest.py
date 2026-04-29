# src/backtest.py

import pandas as pd
import numpy as np


def run_backtest(
    panel: pd.DataFrame,
    weights: pd.DataFrame,
    cost_bps: float,
) -> pd.DataFrame:
    """
    weights at t earn fwd_simple_ret_1h from t to t+1.
    """
    if weights.empty:
        raise ValueError("Weights are empty. Check signal construction.")

    returns = panel[["timestamp", "symbol", "fwd_simple_ret_1h"]].copy()

    merged = weights.merge(
        returns,
        on=["timestamp", "symbol"],
        how="left",
    )

    merged["weighted_ret"] = merged["weight"] * merged["fwd_simple_ret_1h"]

    gross = (
        merged.groupby("timestamp")["weighted_ret"]
        .sum()
        .rename("gross_return")
    )

    weight_matrix = (
        weights.pivot(index="timestamp", columns="symbol", values="weight")
        .fillna(0.0)
        .sort_index()
    )

    turnover = (
        weight_matrix.diff()
        .abs()
        .sum(axis=1)
        .fillna(weight_matrix.abs().sum(axis=1))
        .rename("turnover")
    )

    cost_rate = cost_bps / 10_000
    costs = (turnover * cost_rate).rename("cost")

    result = pd.concat([gross, turnover, costs], axis=1).dropna()
    result["net_return"] = result["gross_return"] - result["cost"]

    result["gross_equity"] = (1 + result["gross_return"]).cumprod()
    result["net_equity"] = (1 + result["net_return"]).cumprod()

    return result.reset_index()