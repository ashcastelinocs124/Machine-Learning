"""Fetch FRED series via pandas_datareader (no API key needed).

Each bucket in config.FRED_BUCKETS is written to data/raw/<bucket>.csv with a
date index and one column per series.
"""
from __future__ import annotations

import logging
import time
from typing import Mapping

import pandas as pd
from pandas_datareader import data as pdr

from src import config

log = logging.getLogger(__name__)


def fetch_fred_series(series_map: Mapping[str, str],
                      start: str = config.START_DATE,
                      end: str = config.END_DATE,
                      max_retries: int = 3,
                      sleep_sec: float = 1.0) -> pd.DataFrame:
    """Fetch every series in ``series_map`` and return as a single DataFrame.

    Columns are the friendly names (keys of series_map); index is the date.
    Series with different native frequencies are aligned on a daily union
    index — gaps are preserved as NaN (resample downstream).
    """
    frames: list[pd.DataFrame] = []
    for friendly_name, series_id in series_map.items():
        for attempt in range(1, max_retries + 1):
            try:
                df = pdr.DataReader(series_id, "fred", start, end)
                df.columns = [friendly_name]
                frames.append(df)
                log.info("fetched %s (%s): %d rows", friendly_name, series_id, len(df))
                break
            except Exception as exc:
                log.warning("attempt %d/%d failed for %s (%s): %s",
                            attempt, max_retries, friendly_name, series_id, exc)
                if attempt == max_retries:
                    log.error("giving up on %s (%s)", friendly_name, series_id)
                else:
                    time.sleep(sleep_sec * attempt)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, axis=1)
    combined.index.name = "date"
    return combined.sort_index()


def fetch_all_fred() -> dict[str, pd.DataFrame]:
    config.ensure_dirs()
    results: dict[str, pd.DataFrame] = {}
    for bucket_name, series_map in config.FRED_BUCKETS.items():
        log.info("=== %s (%d series) ===", bucket_name, len(series_map))
        df = fetch_fred_series(series_map)
        out_path = config.DATA_RAW / f"{bucket_name}.csv"
        df.to_csv(out_path)
        log.info("wrote %s (%s rows, %s cols)", out_path, len(df), df.shape[1])
        results[bucket_name] = df
    return results


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s — %(message)s")
    fetch_all_fred()


if __name__ == "__main__":
    main()
