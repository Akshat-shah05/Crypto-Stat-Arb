# experiments/cost_sensitivity.py

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

import pandas as pd

from config import PROCESSED_DIR, RESULTS_DIR
from features import add_features
from signals import add_signal
from portfolio import make_weights
from backtest import run_backtest
from metrics import summarize_returns


def main():
    panel = pd.read_parquet(PROCESSED_DIR / "crypto_panel_1h.parquet")
    panel = add_features(panel)

    kind = "momentum"
    horizon = 168

    panel = add_signal(panel, kind=kind, horizon_hours=horizon)
    signal_col = f"{kind}_{horizon}h"

    weights = make_weights(panel, signal_col=signal_col)

    rows = []

    for cost_bps in [0, 7, 10, 20, 30, 50]:
        result = run_backtest(panel, weights, cost_bps=cost_bps)
        metrics = summarize_returns(result["net_return"])

        rows.append({
            "cost_bps": cost_bps,
            "avg_turnover": result["turnover"].mean(),
            **metrics,
        })

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / "cost_sensitivity.csv", index=False)
    print(out)


if __name__ == "__main__":
    main()