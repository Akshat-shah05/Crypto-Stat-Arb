# src/portfolio.py

import pandas as pd
import numpy as np

from config import GROSS_EXPOSURE, LONG_SHORT_QUANTILE, MIN_ASSETS


def make_weights(
    df: pd.DataFrame,
    signal_col: str,
    q: float = LONG_SHORT_QUANTILE,
    gross_exposure: float = GROSS_EXPOSURE,
    min_assets: int = MIN_ASSETS,
) -> pd.DataFrame:
    """
    At each timestamp:
    - long top q by signal
    - short bottom q by signal
    - long side sums to +gross/2
    - short side sums to -gross/2
    """
    rows = []

    for ts, g in df.groupby("timestamp"):
        s = g.set_index("symbol")[signal_col].dropna()

        if len(s) < min_assets:
            continue

        n_tail = max(1, int(len(s) * q))

        longs = s.nlargest(n_tail).index
        shorts = s.nsmallest(n_tail).index

        w = pd.Series(0.0, index=s.index)

        w.loc[longs] = gross_exposure / 2 / n_tail
        w.loc[shorts] = -gross_exposure / 2 / n_tail

        out = pd.DataFrame(
            {
                "timestamp": ts,
                "symbol": w.index,
                "weight": w.values,
            }
        )

        rows.append(out)

    if not rows:
        return pd.DataFrame(columns=["timestamp", "symbol", "weight"])

    return pd.concat(rows, ignore_index=True)