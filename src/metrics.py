# src/metrics.py

import numpy as np
import pandas as pd
import statsmodels.api as sm

from config import ANNUALIZATION_FACTOR


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    drawdown = equity / peak - 1
    return drawdown.min()


def summarize_returns(
    returns: pd.Series,
    annualization_factor: int = ANNUALIZATION_FACTOR,
) -> dict:
    returns = returns.dropna()

    if len(returns) == 0:
        return {}

    equity = (1 + returns).cumprod()

    ann_return = equity.iloc[-1] ** (annualization_factor / len(returns)) - 1
    ann_vol = returns.std() * np.sqrt(annualization_factor)

    sharpe = np.nan
    if returns.std() != 0:
        sharpe = returns.mean() / returns.std() * np.sqrt(annualization_factor)

    return {
        "annualized_return": ann_return,
        "annualized_volatility": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown(equity),
        "hit_rate": (returns > 0).mean(),
        "mean_return": returns.mean(),
        "median_return": returns.median(),
        "worst_return": returns.min(),
    }


def btc_alpha_beta(strategy_results: pd.DataFrame, panel: pd.DataFrame, btc_symbol="BTC/USD"):
    btc = panel[panel["symbol"] == btc_symbol][
        ["timestamp", "fwd_simple_ret_1h"]
    ].rename(columns={"fwd_simple_ret_1h": "btc_return"})

    data = strategy_results[["timestamp", "net_return"]].merge(
        btc,
        on="timestamp",
        how="inner",
    ).dropna()

    y = data["net_return"]
    X = sm.add_constant(data["btc_return"])

    model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 24})

    return {
        "alpha_per_period": model.params["const"],
        "beta_to_btc": model.params["btc_return"],
        "alpha_t_stat": model.tvalues["const"],
        "r_squared": model.rsquared,
    }