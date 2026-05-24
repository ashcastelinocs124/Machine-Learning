"""Run every fetcher end-to-end."""
from __future__ import annotations

import logging

from src.fetch_fred import fetch_all_fred
from src.fetch_market import fetch_all_market


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s — %(message)s")
    log = logging.getLogger(__name__)
    log.info("starting FRED fetch")
    fetch_all_fred()
    log.info("starting market fetch")
    fetch_all_market()
    log.info("done")


if __name__ == "__main__":
    main()
