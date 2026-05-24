"""Fetch market data via yfinance.

For each bucket in config.YF_BUCKETS, we pull OHLCV and save adjusted close
to data/raw/<bucket>.csv. Adjusted close handles dividends + splits, which is
what you want for return-based features.
"""
from __future__ import annotations

import logging
import time
from typing import Mapping

import pandas as pd
import yfinance as yf

from src import config

log = logging.getLogger(__name__)


def fetch_yf_bucket(ticker_map: Mapping[str, str],
                    start: str = config.START_DATE,
                    end: str = config.END_DATE,
                    max_retries: int = 3,
                    sleep_sec: float = 1.5) -> pd.DataFrame:
    """Download adjusted close prices for every ticker in ticker_map.

    Returns a DataFrame indexed by date with one column per friendly name.
    """
    tickers = list(ticker_map.keys())
    if not tickers:
        return pd.DataFrame()

    for attempt in range(1, max_retries + 1):
        try:
            raw = yf.download(
                tickers=tickers,
                start=start,
                end=end,
                auto_adjust=False,
                progress=False,
                group_by="ticker",
                threads=True,
            )
            break
        except Exception as exc:
            log.warning("attempt %d/%d failed for batch %s: %s",
                        attempt, max_retries, tickers, exc)
            if attempt == max_retries:
                raise
            time.sleep(sleep_sec * attempt)

    if raw.empty:
        log.error("empty response for tickers %s", tickers)
        return pd.DataFrame()

    cols: dict[str, pd.Series] = {}
    if isinstance(raw.columns, pd.MultiIndex):
        for ticker, friendly in ticker_map.items():
            try:
                series = raw[ticker]["Adj Close"]
            except KeyError:
                log.warning("no Adj Close for %s — skipping", ticker)
                continue
            cols[friendly] = series
    else:
        # Single ticker case — yfinance returns flat columns
        only_ticker = tickers[0]
        friendly = ticker_map[only_ticker]
        cols[friendly] = raw["Adj Close"]

    out = pd.DataFrame(cols)
    out.index.name = "date"
    return out.sort_index()


def fetch_all_market() -> dict[str, pd.DataFrame]:
    config.ensure_dirs()
    results: dict[str, pd.DataFrame] = {}
    for bucket_name, ticker_map in config.YF_BUCKETS.items():
        log.info("=== %s (%d tickers) ===", bucket_name, len(ticker_map))
        df = fetch_yf_bucket(ticker_map)
        out_path = config.DATA_RAW / f"{bucket_name}.csv"
        df.to_csv(out_path)
        log.info("wrote %s (%s rows, %s cols)", out_path, len(df), df.shape[1])
        results[bucket_name] = df
    return results


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s — %(message)s")
    fetch_all_market()


if __name__ == "__main__":
    main()
