"""Single source of truth for series IDs, tickers, paths, and date range.

FRED series IDs: https://fred.stlouisfed.org/
yfinance tickers: https://finance.yahoo.com/
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

START_DATE = "2000-01-01"
END_DATE = date.today().isoformat()


# ---------------------------------------------------------------------------
# FRED series, grouped by bucket. Key = friendly name, value = FRED series ID.
# ---------------------------------------------------------------------------

FRED_ECONOMIC_ACTIVITY = {
    # Real GDP (quarterly, seasonally adjusted annual rate)
    "real_gdp": "GDPC1",
    # ISM Manufacturing PMI proxies (FRED discontinued ISM in 2016; use Markit / regional proxies)
    "ism_mfg_pmi_napm": "NAPM",                # legacy ISM Mfg PMI (history pre-2016)
    "ism_services_pmi_napmni": "NAPMNI",       # legacy ISM Services PMI
    "philly_fed_mfg": "GACDFSA066MSFRBPHI",    # Philly Fed mfg general activity (current proxy)
    "chicago_fed_natl": "CFNAI",               # Chicago Fed National Activity Index
    # Labor market
    "nonfarm_payrolls": "PAYEMS",
    "unemployment_rate": "UNRATE",
    "initial_claims": "ICSA",
    "continuing_claims": "CCSA",
    # Output / consumption
    "industrial_production": "INDPRO",
    "capacity_utilization": "TCU",
    "retail_sales": "RSAFS",
    "real_retail_sales": "RRSFS",
    "manufacturers_new_orders": "AMTMNO",
}

FRED_INFLATION = {
    "cpi_headline": "CPIAUCSL",
    "cpi_core": "CPILFESL",
    "pce_headline": "PCEPI",
    "pce_core": "PCEPILFE",
    "breakeven_5y": "T5YIE",
    "breakeven_10y": "T10YIE",
    "breakeven_5y5y": "T5YIFR",
    "avg_hourly_earnings": "CES0500000003",
    "wti_oil": "DCOILWTICO",
    "brent_oil": "DCOILBRENTEU",
}

FRED_RATES = {
    "fed_funds_effective": "DFF",
    "treasury_2y": "DGS2",
    "treasury_5y": "DGS5",
    "treasury_10y": "DGS10",
    "treasury_30y": "DGS30",
    "real_5y": "DFII5",
    "real_10y": "DFII10",
    "real_30y": "DFII30",
    "tips_10y_breakeven": "T10YIE",  # duplicate of inflation bucket for convenience
    "term_spread_10y_2y": "T10Y2Y",
}

FRED_CREDIT_STRESS = {
    "hy_oas": "BAMLH0A0HYM2",          # ICE BofA US High Yield OAS
    "ig_oas": "BAMLC0A0CM",            # ICE BofA US Corporate OAS
    "ccc_oas": "BAMLH0A3HYC",          # CCC & lower OAS
    "vix": "VIXCLS",
    "ted_spread_proxy": "TEDRATE",     # legacy TED; FRED discontinued — kept for history
    "financial_stress_stl": "STLFSI4", # St Louis Fed Financial Stress Index v4
}

FRED_LIQUIDITY = {
    "fed_balance_sheet": "WALCL",
    "bank_reserves": "WRESBAL",
    "reverse_repo": "RRPONTSYD",
    "dollar_index_broad": "DTWEXBGS",
    "dollar_index_majors": "DTWEXM",   # legacy; some history
    "m2": "M2SL",
}


# ---------------------------------------------------------------------------
# yfinance tickers
# ---------------------------------------------------------------------------

YF_SECTOR_ETFS = {
    "SPY": "spy_sp500",
    "XLB": "xlb_materials",
    "XLC": "xlc_communications",
    "XLE": "xle_energy",
    "XLF": "xlf_financials",
    "XLI": "xli_industrials",
    "XLK": "xlk_technology",
    "XLP": "xlp_consumer_staples",
    "XLRE": "xlre_real_estate",
    "XLU": "xlu_utilities",
    "XLV": "xlv_healthcare",
    "XLY": "xly_consumer_discretionary",
}

YF_SEMIS_AND_SMALL_CAP = {
    "SOXX": "soxx_semis",
    "SMH": "smh_semis",
    "IWM": "iwm_russell_2000",
    "QQQ": "qqq_nasdaq_100",
    "NVDA": "nvda",                  # proxy for hyperscaler / AI capex narrative
    "AMD": "amd",
    "TSM": "tsm",
}

YF_MACRO_ASSETS = {
    "^VIX": "vix",
    "DX-Y.NYB": "dxy_dollar_index",
    "CL=F": "wti_futures",
    "BZ=F": "brent_futures",
    "GC=F": "gold_futures",
    "HG=F": "copper_futures",
    "^TNX": "ust_10y_yield_index",   # ^TNX is 10Y yield * 100
    "^FVX": "ust_5y_yield_index",
    "^IRX": "ust_3m_yield_index",
    "^TYX": "ust_30y_yield_index",
}


# ---------------------------------------------------------------------------
# AI capex — manual quarterly CSV (not from FRED or yfinance).
# Values in billions USD. Compiled from SEC 10-Q/10-K filings.
# ---------------------------------------------------------------------------

AI_CAPEX_COLUMNS = [
    "msft_capex",
    "goog_capex",
    "amzn_capex",
    "meta_capex",
    "total_hyperscaler_capex",
    "nvda_revenue",
]


# ---------------------------------------------------------------------------
# Category groups — maps a category name to the feature columns it contains.
# Used by index_regression.py to build PCA-based composite indices.
# ---------------------------------------------------------------------------

CATEGORY_GROUPS: dict[str, list[str]] = {
    "economic_activity": [
        "nonfarm_payrolls", "unemployment_rate", "initial_claims",
        "continuing_claims", "industrial_production", "capacity_utilization",
        "manufacturers_new_orders",
    ],
    "consumer_spending": [
        "retail_sales", "real_retail_sales", "avg_hourly_earnings",
    ],
    "activity_indices": [
        "chicago_fed_natl", "philly_fed_mfg", "financial_stress_stl",
    ],
    "inflation": [
        "cpi_headline", "cpi_core", "pce_headline", "pce_core",
        "wti_oil", "brent_oil",
    ],
    "breakevens": [
        "breakeven_5y", "breakeven_10y", "breakeven_5y5y",
        "tips_10y_breakeven",
    ],
    "rates": [
        "fed_funds_effective", "treasury_2y", "treasury_5y", "treasury_10y",
        "treasury_30y", "real_5y", "real_10y", "term_spread_10y_2y",
    ],
    "credit_stress": [
        "vix", "ted_spread_proxy",
    ],
    "liquidity": [
        "fed_balance_sheet", "bank_reserves", "reverse_repo", "m2",
        "dollar_index_broad", "dollar_index_majors", "dxy_dollar_index",
    ],
    "commodities": [
        "wti_futures", "brent_futures", "copper_futures", "gold_futures",
    ],
    "equity_indices": [
        "spy_sp500", "qqq_nasdaq_100", "iwm_russell_2000",
    ],
    "sectors": [
        "xlb_materials", "xle_energy", "xlf_financials", "xli_industrials",
        "xlk_technology", "xlp_consumer_staples", "xlu_utilities",
        "xlv_healthcare", "xly_consumer_discretionary",
    ],
    "semis_tech": [
        "soxx_semis", "smh_semis", "nvda", "amd", "tsm",
    ],
    "ai_capex": [
        "msft_capex", "goog_capex", "amzn_capex", "meta_capex",
        "total_hyperscaler_capex", "nvda_revenue",
    ],
    "yield_indices": [
        "ust_3m_yield_index", "ust_5y_yield_index",
        "ust_10y_yield_index", "ust_30y_yield_index",
    ],
}


# ---------------------------------------------------------------------------
# Output file map — keeps raw CSV filenames declarative.
# ---------------------------------------------------------------------------

FRED_BUCKETS = {
    "fred_economic_activity": FRED_ECONOMIC_ACTIVITY,
    "fred_inflation": FRED_INFLATION,
    "fred_rates": FRED_RATES,
    "fred_credit_stress": FRED_CREDIT_STRESS,
    "fred_liquidity": FRED_LIQUIDITY,
}

YF_BUCKETS = {
    "market_sector_etfs": YF_SECTOR_ETFS,
    "market_semis_smallcap": YF_SEMIS_AND_SMALL_CAP,
    "market_macro_assets": YF_MACRO_ASSETS,
}


def ensure_dirs() -> None:
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
