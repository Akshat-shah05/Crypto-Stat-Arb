# src/features.py

import numpy as np
import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["symbol", "timestamp"])

    g = df.groupby("symbol")

    df["vol_72h"] = (
        g["log_ret_1h"]
        .rolling(72, min_periods=24)
        .std()
        .reset_index(level=0, drop=True)
    )

    df["avg_volume_72h"] = (
        g["volume"]
        .rolling(72, min_periods=24)
        .mean()
        .reset_index(level=0, drop=True)
    )

    df["volume_shock"] = df["volume"] / df["avg_volume_72h"]
    df["high_volume_shock"] = df["volume_shock"] >= 1.5
    df["low_volume_shock"] = df["volume_shock"] <= 0.75

    df["weekday"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = df["weekday"] >= 5

    market_vol = (
        df.groupby("timestamp")["vol_72h"]
        .mean()
        .rename("market_vol_72h")
        .reset_index()
    )

    df = df.merge(market_vol, on="timestamp", how="left")

    # Simple non-lookahead-ish threshold: compare current market vol to its expanding median.
    mv = df[["timestamp", "market_vol_72h"]].drop_duplicates().sort_values("timestamp")
    mv["market_vol_expanding_median"] = mv["market_vol_72h"].expanding().median()
    mv["high_market_vol"] = mv["market_vol_72h"] > mv["market_vol_expanding_median"]

    df = df.merge(
        mv[["timestamp", "market_vol_expanding_median", "high_market_vol"]],
        on="timestamp",
        how="left",
    )

    return df