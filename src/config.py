# src/config.py

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
PLOTS_DIR = ROOT / "plots"

for d in [RAW_DIR, PROCESSED_DIR, RESULTS_DIR, PLOTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

EXCHANGE_ID = "binance"
SYMBOLS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "ADA/USDT",
    "DOGE/USDT", "AVAX/USDT", "LINK/USDT", "LTC/USDT", "BCH/USDT",
    "DOT/USDT", "UNI/USDT", "ETC/USDT", "FIL/USDT", "NEAR/USDT"
]
TIMEFRAME = "1h"

START_DATE = "2021-01-01T00:00:00Z"
END_DATE = "2025-01-01T00:00:00Z"


MARKET_ORDER_COST_BPS = 20
LIMIT_ORDER_COST_BPS = 7

GROSS_EXPOSURE = 1.0
LONG_SHORT_QUANTILE = 0.20
MIN_ASSETS = 8

ANNUALIZATION_FACTOR = 24 * 365