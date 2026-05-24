# Machine Learning Projects

A collection of end-to-end machine learning projects spanning structured prediction, feature engineering, and macro-economic forecasting.

## Projects

### [airbnb-prediction/](./airbnb-prediction/)

Predicting Airbnb listing prices across 7 major global cities (London, Rome, Bangkok, Barcelona, Amsterdam, Paris, Sydney) using CatBoost gradient boosting. A `room_type` + price-percentile proxy for property size lifted R² from ~0.43 to ~0.82 on London and Barcelona, and from 0.30 to 0.73 on Amsterdam. SHAP analysis confirms `estimated_bedrooms` dominates feature importance in every city.

**Stack:** Python, pandas, CatBoost, scikit-learn, SHAP.

### [macro-prediction/](./macro-prediction/)

A broad macro feature set for predicting market regimes, returns, and volatility. Pulls 50+ series across economic activity (ISM PMI, payrolls, retail sales), inflation (CPI, PCE, breakevens), rates (2Y/10Y/30Y nominal and real), credit stress (HY/IG spreads, VIX), liquidity (Fed balance sheet, RRP, dollar index), oil shocks, AI capex (hyperscalers, Nvidia, semis), and sector returns (SPDR XL*, SMH, SOXX, IWM).

**Stack:** Python, pandas_datareader (FRED), yfinance.

## Layout

```
.
├── README.md                  # this file
├── airbnb-prediction/         # CatBoost-based price prediction
│   └── README.md
└── macro-prediction/          # Macro feature ingestion for market prediction
    └── README.md
```

Each project folder is self-contained with its own README, dependencies, and run instructions.
