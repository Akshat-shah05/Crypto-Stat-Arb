# src/config.py

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
PLOTS_DIR = ROOT / "plots"

for d in [RAW_DIR, PROCESSED_DIR, RESULTS_DIR, PLOTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

EXCHANGE_ID = "coinbase"
TIMEFRAME = "1h"

START_DATE = "2021-01-01T00:00:00Z"
END_DATE = "2025-01-01T00:00:00Z"

SYMBOLS = [
    "BTC/USD",
    "ETH/USD",
    "SOL/USD",
    "XRP/USD",
    "ADA/USD",
    "DOGE/USD",
    "AVAX/USD",
    "LINK/USD",
    "LTC/USD",
    "BCH/USD",
    "DOT/USD",
    "UNI/USD",
    "ETC/USD",
    "FIL/USD",
    "NEAR/USD",
]

MARKET_ORDER_COST_BPS = 20
LIMIT_ORDER_COST_BPS = 7

GROSS_EXPOSURE = 1.0
LONG_SHORT_QUANTILE = 0.20
MIN_ASSETS = 8

ANNUALIZATION_FACTOR = 24 * 365