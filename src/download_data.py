# src/download_data.py

import time
import ccxt
import pandas as pd
from tqdm import tqdm

from config import EXCHANGE_ID, SYMBOLS, TIMEFRAME, START_DATE, END_DATE, RAW_DIR


def get_exchange():
    exchange_cls = getattr(ccxt, EXCHANGE_ID)
    exchange = exchange_cls({"enableRateLimit": True})
    exchange.load_markets()
    return exchange


def fetch_symbol_ohlcv(exchange, symbol, timeframe, start_date, end_date, limit=300):
    since = exchange.parse8601(start_date)
    end_ms = exchange.parse8601(end_date)
    timeframe_ms = exchange.parse_timeframe(timeframe) * 1000

    rows = []

    while since < end_ms:
        batch = exchange.fetch_ohlcv(
            symbol=symbol,
            timeframe=timeframe,
            since=since,
            limit=limit,
        )

        if not batch:
            break

        batch = [x for x in batch if x[0] < end_ms]
        rows.extend(batch)

        last_ts = batch[-1][0]
        next_since = last_ts + timeframe_ms

        if next_since <= since:
            break

        since = next_since
        time.sleep(exchange.rateLimit / 1000)

    df = pd.DataFrame(
        rows,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )

    if df.empty:
        return df

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["symbol"] = symbol
    return df


def main():
    exchange = get_exchange()

    for symbol in tqdm(SYMBOLS):
        if symbol not in exchange.markets:
            print(f"Skipping {symbol}: not available on {EXCHANGE_ID}")
            continue

        print(f"Downloading {symbol}")
        df = fetch_symbol_ohlcv(exchange, symbol, TIMEFRAME, START_DATE, END_DATE)

        if df.empty:
            print(f"No data for {symbol}")
            continue

        safe_symbol = symbol.replace("/", "_")
        out_path = RAW_DIR / f"{safe_symbol}_{TIMEFRAME}.csv"
        df.to_csv(out_path, index=False)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()