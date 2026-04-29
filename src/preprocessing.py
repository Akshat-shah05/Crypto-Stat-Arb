# src/preprocessing.py

import numpy as np
import pandas as pd
from pathlib import Path

from config import RAW_DIR, PROCESSED_DIR


def load_raw_data(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    files = list(raw_dir.glob("*.csv"))

    if not files:
        raise FileNotFoundError("No raw CSV files found. Run download_data.py first.")

    dfs = []
    for file in files:
        df = pd.read_csv(file)
        dfs.append(df)

    data = pd.concat(dfs, ignore_index=True)
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
    return data


def clean_panel(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = df.drop_duplicates(subset=["timestamp", "symbol"])
    df = df.sort_values(["symbol", "timestamp"])

    df = df[
        (df["close"] > 0)
        & (df["open"] > 0)
        & (df["high"] > 0)
        & (df["low"] > 0)
    ]

    cleaned = []

    for symbol, g in df.groupby("symbol"):
        g = g.sort_values("timestamp").set_index("timestamp")

        full_index = pd.date_range(
            start=g.index.min(),
            end=g.index.max(),
            freq="1h",
            tz="UTC",
        )

        g = g.reindex(full_index)
        g["symbol"] = symbol

        price_cols = ["open", "high", "low", "close"]
        g[price_cols] = g[price_cols].ffill(limit=2)

        # For missing candles, volume should not be invented aggressively.
        g["volume"] = g["volume"].fillna(0)

        cleaned.append(g.reset_index(names="timestamp"))

    panel = pd.concat(cleaned, ignore_index=True)
    panel = panel.sort_values(["symbol", "timestamp"])

    panel["log_price"] = np.log(panel["close"])
    panel["log_ret_1h"] = panel.groupby("symbol")["log_price"].diff()

    # This is the return earned after forming a signal at timestamp t.
    panel["fwd_simple_ret_1h"] = (
        panel.groupby("symbol")["close"].pct_change().shift(-1)
    )

    return panel


def main():
    raw = load_raw_data()
    panel = clean_panel(raw)

    out_path = PROCESSED_DIR / "crypto_panel_1h.parquet"
    panel.to_parquet(out_path, index=False)

    print(f"Saved processed panel to {out_path}")
    print(panel.head())


if __name__ == "__main__":
    main()