# Macro Prediction

ML project gathering a broad macro feature set for predicting market regimes / returns.

## Buckets

| Bucket | Examples |
|--------|----------|
| Economic activity | ISM PMI, payrolls, retail sales, industrial production, jobless claims |
| Inflation | CPI, PCE, breakevens, oil, wages |
| Rates | 2Y/10Y/30Y yields, real yields, Fed funds |
| Credit stress | HY/IG spreads, VIX |
| Liquidity | Fed balance sheet, bank reserves, reverse repo, dollar index |
| Oil shock | WTI, Brent, energy ETF |
| AI capex | Hyperscaler capex (manual), Nvidia revenue (manual), semis ETFs |
| Sector returns | SPDR sectors (XL*), SMH, SOXX, IWM, SPY |

## Setup

```bash
pip install -r requirements.txt
python -m src.fetch_all
```

CSVs land in `data/raw/`. See `CLAUDE.md` for full architecture and conventions.
