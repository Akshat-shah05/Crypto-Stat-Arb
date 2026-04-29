# src/signals.py

import numpy as np
import pandas as pd


def cross_sectional_zscore(df: pd.DataFrame, col: str) -> pd.Series:
    mean = df.groupby("timestamp")[col].transform("mean")
    std = df.groupby("timestamp")[col].transform("std")
    return (df[col] - mean) / std.replace(0, np.nan)


def add_signal(
    df: pd.DataFrame,
    kind: str,
    horizon_hours: int,
    vol_window_col: str = "vol_72h",
) -> pd.DataFrame:
    """
    kind: "momentum" or "reversal"
    horizon_hours: lookback horizon in hours
    """
    df = df.copy()
    df = df.sort_values(["symbol", "timestamp"])

    lookback_ret = df.groupby("symbol")["log_price"].diff(horizon_hours)

    if kind == "momentum":
        raw = lookback_ret
    elif kind == "reversal":
        raw = -lookback_ret
    else:
        raise ValueError("kind must be 'momentum' or 'reversal'")

    signal_name = f"{kind}_{horizon_hours}h"

    df[f"{signal_name}_raw"] = raw
    df[f"{signal_name}_vol_scaled"] = raw / df[vol_window_col].replace(0, np.nan)
    df[signal_name] = cross_sectional_zscore(df, f"{signal_name}_vol_scaled")

    return df