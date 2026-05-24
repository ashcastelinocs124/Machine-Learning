# Macro Prediction — Project Guide

ML project for predicting market behavior (returns, regimes, vol) from a broad macro feature set covering economic activity, inflation, rates, credit stress, liquidity, oil shocks, AI capex, and sector returns.

## Data sources

| Bucket | Source | Notes |
|--------|--------|-------|
| ISM PMI, payrolls, retail sales, industrial production, jobless claims | FRED via `pandas_datareader` | No API key needed (public CSV endpoint) |
| CPI, PCE, breakevens, wages, oil (WTI/Brent) | FRED | Monthly + daily series mixed — handled in fetcher |
| 2Y/10Y/30Y nominal + real yields, Fed funds | FRED | Daily series (`DGS*`, `DFII*`, `DFF`) |
| HY/IG credit spreads, VIX | FRED | `BAMLH0A0HYM2`, `BAMLC0A0CM`, `VIXCLS` |
| Fed balance sheet, bank reserves, RRP, dollar index | FRED | Weekly + daily; resample later |
| Sector ETFs (XL*), SMH/SOXX, IWM, oil & dollar futures | yfinance | Free, no key |
| Hyperscaler capex, Nvidia revenue | TODO — manual CSV | Quarterly earnings; not on FRED |

FRED series IDs and yfinance tickers live in `src/config.py` — single source of truth.

## Project layout

```
macro-prediction/
├── CLAUDE.md
├── learnings.md
├── short_term_memory.md
├── long_term_memory.md
├── README.md
├── requirements.txt
├── data/
│   ├── raw/        # one CSV per source bucket
│   └── processed/  # joined / resampled feature sets (later)
├── notebooks/
└── src/
    ├── config.py       # series + ticker maps, date range
    ├── fetch_fred.py   # all FRED series
    ├── fetch_market.py # all yfinance tickers
    └── fetch_all.py    # orchestrator
```

## How to run

```bash
cd macro-prediction
pip install -r requirements.txt
python -m src.fetch_all            # fetch everything
python -m src.fetch_fred           # FRED only
python -m src.fetch_market         # market only
```

CSVs are written to `data/raw/<bucket>.csv` with a `date` index and one column per series.

## Conventions

- All time series are saved with a `date` column as the index, ISO format.
- FRED returns native frequencies (daily/weekly/monthly). Don't forward-fill at the fetch layer — keep raw fidelity, resample in the feature pipeline.
- yfinance returns adjusted close as `Adj Close`; we save adj close only for ETFs/indices (returns-friendly).
- Date range default: 2000-01-01 to today (overridable via `config.START_DATE`, `config.END_DATE`).

## Git push policy (HARD RULE)

Every push to a remote MUST go through the `/gitpush` skill. Never run `git push` or any push-equivalent directly in Bash.

## Learnings

This project maintains a `learnings.md` file at the project root. Add entries whenever you discover something interesting. Each entry must include a **Ref** subtitle pointing to the relevant CLAUDE.md section. Only read `learnings.md` when its contents are directly relevant to the current task.

Use the `/capture-learnings` skill at the end of sessions to do this automatically.

## Memory System

### Short-term memory (`short_term_memory.md`)
Holds a detailed log of the past 5 immediate tasks — what was done, why, and the outcome. When a new task is completed, append it. If there are more than 5 entries, summarize the oldest one into `long_term_memory.md` before removing it.

### Long-term memory (`long_term_memory.md`)
When a task ages out of short-term memory, write a condensed summary (2-3 lines) here. This preserves historical context without clutter.

**Pruning rule:** Every 10 sessions, review `long_term_memory.md` against `CLAUDE.md`. Delete any entries no longer relevant to the current state of the project.

### Loading priority
At the start of every session, read both files into context:
1. `short_term_memory.md` first — most important.
2. `long_term_memory.md` second — background context.

## Completed Work

### 2026-05-23 — Project scaffolding + data fetchers
- Created folder structure with `src/`, `data/raw/`, `data/processed/`, `notebooks/`
- Built `fetch_fred.py` (FRED via `pandas_datareader`, no API key required) covering economic activity, inflation, rates, credit, liquidity buckets
- Built `fetch_market.py` (yfinance) covering sector ETFs, semis, oil, dollar, VIX, IWM
- Centralized series/ticker maps in `src/config.py`
- Output: one CSV per bucket in `data/raw/`
