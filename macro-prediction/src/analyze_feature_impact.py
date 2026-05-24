"""Identify which macro/market features most affect recent economic activity.

Constructs a GDP growth proxy from industrial production, retail sales, and
nonfarm payrolls, then ranks all other features by four methods: correlation,
Granger causality, gradient-boosting importance, and SHAP values.

Usage:
    python -m src.analyze_feature_impact
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy import stats as sp_stats
from sklearn.ensemble import GradientBoostingRegressor
from statsmodels.tsa.stattools import grangercausalitytests

from src.config import AI_CAPEX_COLUMNS, DATA_PROCESSED, DATA_RAW

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="shap")

RECENT_CUTOFF = "2023-01-01"

GDP_COLUMN = "real_gdp"

LEVEL_CHANGE_FEATURES = {
    "unemployment_rate", "capacity_utilization",
    "fed_funds_effective",
    "treasury_2y", "treasury_5y", "treasury_10y", "treasury_30y",
    "real_5y", "real_10y", "real_30y",
    "tips_10y_breakeven", "term_spread_10y_2y",
    "breakeven_5y", "breakeven_10y", "breakeven_5y5y",
    "hy_oas", "ig_oas", "ccc_oas",
    "vix", "ted_spread_proxy", "financial_stress_stl",
    "chicago_fed_natl", "philly_fed_mfg",
    "ust_10y_yield_index", "ust_5y_yield_index",
    "ust_3m_yield_index", "ust_30y_yield_index",
}

OUTPUT_DIR = DATA_PROCESSED


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_and_merge() -> pd.DataFrame:
    """Load all raw CSVs, merge on date, return a single daily DataFrame."""
    csv_files = sorted(DATA_RAW.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSVs found in {DATA_RAW}")

    frames: list[pd.DataFrame] = []
    for f in csv_files:
        df = pd.read_csv(f, parse_dates=["date"], index_col="date")
        frames.append(df)

    merged = pd.concat(frames, axis=1)
    merged = merged.loc[:, ~merged.columns.duplicated(keep="first")]
    merged.sort_index(inplace=True)
    return merged


def resample_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill then resample to month-end, taking last observation."""
    df = df.ffill()
    return df.resample("ME").last()


# ---------------------------------------------------------------------------
# Target construction
# ---------------------------------------------------------------------------

def build_gdp_target(quarterly_gdp: pd.Series) -> pd.Series:
    """QoQ annualized real GDP growth rate from GDPC1 levels."""
    gdp = quarterly_gdp.dropna()
    qoq = gdp.pct_change()
    annualized = ((1 + qoq) ** 4 - 1) * 100
    annualized.name = "gdp_growth"
    return annualized.dropna()


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def build_feature_matrix(monthly: pd.DataFrame) -> pd.DataFrame:
    """Quarterly feature matrix: 1-quarter momentum for all features except GDP."""
    exclude = {GDP_COLUMN}
    feature_cols = [c for c in monthly.columns if c not in exclude]
    feats = monthly[feature_cols].copy()

    quarterly = feats.resample("QE").last()

    transformed = pd.DataFrame(index=quarterly.index)
    for col in quarterly.columns:
        series = quarterly[col].dropna()
        if len(series) < 8:
            continue
        if col in LEVEL_CHANGE_FEATURES:
            transformed[col] = quarterly[col].diff(1)
        else:
            transformed[col] = quarterly[col].pct_change(1) * 100

    sparsity = transformed.isna().mean()
    keep = sparsity[sparsity < 0.30].index.tolist()
    ai_cols = [c for c in AI_CAPEX_COLUMNS if c in transformed.columns and c not in keep]
    keep.extend(ai_cols)
    transformed = transformed[keep]
    return transformed


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def correlation_analysis(
    features: pd.DataFrame, target: pd.Series
) -> pd.DataFrame:
    """Pearson + Spearman correlation, full and recent windows."""
    target_name = target.name
    aligned = features.join(target, how="inner").dropna()
    recent = aligned.loc[RECENT_CUTOFF:]

    results = []
    for col in features.columns:
        full = aligned[[col, target_name]].dropna()
        rec = recent[[col, target_name]].dropna()
        if len(full) < 20 or len(rec) < 6:
            continue
        if full[col].std() == 0 or rec[col].std() == 0:
            continue
        pr_full, _ = sp_stats.pearsonr(full[col], full[target_name])
        sp_full, _ = sp_stats.spearmanr(full[col], full[target_name])
        pr_rec, _ = sp_stats.pearsonr(rec[col], rec[target_name])
        sp_rec, _ = sp_stats.spearmanr(rec[col], rec[target_name])
        results.append({
            "feature": col,
            "pearson_full": pr_full,
            "spearman_full": sp_full,
            "pearson_recent": pr_rec,
            "spearman_recent": sp_rec,
            "abs_recent": abs(sp_rec),
        })
    df = pd.DataFrame(results).sort_values("abs_recent", ascending=False)
    return df.reset_index(drop=True)


def granger_analysis(
    features: pd.DataFrame, target: pd.Series, max_lag: int = 6
) -> pd.DataFrame:
    """Granger-causality min p-value across 1..max_lag for each feature."""
    target_name = target.name
    aligned = features.join(target, how="inner").dropna()
    results = []
    for col in features.columns:
        pair = aligned[[col, target_name]].dropna()
        if len(pair) < max_lag + 20:
            continue
        try:
            gc = grangercausalitytests(
                pair[[target_name, col]], maxlag=max_lag, verbose=False
            )
            min_p = min(gc[lag][0]["ssr_ftest"][1] for lag in range(1, max_lag + 1))
        except Exception:
            continue
        results.append({"feature": col, "granger_min_p": min_p})
    df = pd.DataFrame(results).sort_values("granger_min_p")
    return df.reset_index(drop=True)


def tree_importance(
    features: pd.DataFrame, target: pd.Series
) -> tuple[pd.DataFrame, GradientBoostingRegressor]:
    """GradientBoosting feature importance on full + recent windows."""
    target_name = target.name
    aligned = features.join(target, how="inner").dropna()
    recent = aligned.loc[RECENT_CUTOFF:]

    X_full = aligned.drop(columns=target_name)
    y_full = aligned[target_name]
    X_rec = recent.drop(columns=target_name)
    y_rec = recent[target_name]

    gbm_full = GradientBoostingRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, random_state=42,
    )
    gbm_full.fit(X_full, y_full)

    gbm_rec = GradientBoostingRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.8, random_state=42,
    )
    gbm_rec.fit(X_rec, y_rec)

    imp = pd.DataFrame({
        "feature": X_full.columns,
        "importance_full": gbm_full.feature_importances_,
        "importance_recent": gbm_rec.feature_importances_,
    }).sort_values("importance_recent", ascending=False).reset_index(drop=True)
    return imp, gbm_full


def shap_analysis(
    model: GradientBoostingRegressor,
    features: pd.DataFrame,
    target: pd.Series,
) -> pd.DataFrame:
    """SHAP mean absolute values from the full-window GBM."""
    target_name = target.name
    aligned = features.join(target, how="inner").dropna()
    X = aligned.drop(columns=target_name)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    mean_abs = np.abs(shap_values).mean(axis=0)
    df = pd.DataFrame({
        "feature": X.columns,
        "mean_abs_shap": mean_abs,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    return df, shap_values, X


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_correlation(corr_df: pd.DataFrame, path: Path) -> None:
    top = corr_df.head(20).copy()
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax, col, title in [
        (axes[0], "spearman_full", "Full History (2000–present)"),
        (axes[1], "spearman_recent", "Recent (2023–present)"),
    ]:
        colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in top[col]]
        ax.barh(top["feature"], top[col], color=colors)
        ax.set_xlabel("Spearman Correlation")
        ax.set_title(title)
        ax.invert_yaxis()
        ax.axvline(0, color="grey", linewidth=0.5)
    fig.suptitle("Top 20 Features — Correlation with Real GDP Growth", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_granger(granger_df: pd.DataFrame, path: Path) -> None:
    sig = granger_df[granger_df["granger_min_p"] < 0.05].head(20).copy()
    if sig.empty:
        sig = granger_df.head(15).copy()
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(sig["feature"], -np.log10(sig["granger_min_p"]), color="#3498db")
    ax.set_xlabel("-log10(p-value)")
    ax.set_title("Granger Causality — Features Predicting Real GDP Growth")
    ax.invert_yaxis()
    ax.axvline(-np.log10(0.05), color="red", linestyle="--", linewidth=0.8, label="p=0.05")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_importance(imp_df: pd.DataFrame, path: Path) -> None:
    top = imp_df.head(20).copy()
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    axes[0].barh(top["feature"], top["importance_full"], color="#9b59b6")
    axes[0].set_xlabel("Feature Importance")
    axes[0].set_title("Full History")
    axes[0].invert_yaxis()
    axes[1].barh(top["feature"], top["importance_recent"], color="#e67e22")
    axes[1].set_xlabel("Feature Importance")
    axes[1].set_title("Recent (2023–present)")
    axes[1].invert_yaxis()
    fig.suptitle("Gradient Boosting Feature Importance", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_shap(shap_values: np.ndarray, X: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        shap_values, X, plot_type="bar", max_display=20, show=False,
    )
    plt.title("SHAP Feature Importance (mean |SHAP|)", fontsize=13)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")


# ---------------------------------------------------------------------------
# Consensus ranking
# ---------------------------------------------------------------------------

def consensus_ranking(
    corr_df: pd.DataFrame,
    granger_df: pd.DataFrame,
    imp_df: pd.DataFrame,
    shap_df: pd.DataFrame,
    top_n: int = 15,
) -> pd.DataFrame:
    """Features appearing in the top_n across multiple methods."""
    sets = {
        "correlation": set(corr_df.head(top_n)["feature"]),
        "granger": set(granger_df.head(top_n)["feature"]),
        "gbm_importance": set(imp_df.head(top_n)["feature"]),
        "shap": set(shap_df.head(top_n)["feature"]),
    }
    all_feats = set()
    for s in sets.values():
        all_feats |= s

    rows = []
    for feat in all_feats:
        count = sum(feat in s for s in sets.values())
        methods = [name for name, s in sets.items() if feat in s]
        rows.append({"feature": feat, "methods_count": count, "methods": ", ".join(methods)})
    df = pd.DataFrame(rows).sort_values(
        ["methods_count", "feature"], ascending=[False, True]
    ).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_section(title: str) -> None:
    width = 72
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


def print_table(df: pd.DataFrame, n: int = 15) -> None:
    with pd.option_context("display.max_rows", n, "display.width", 120, "display.float_format", "{:.4f}".format):
        print(df.head(n).to_string(index=False))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Loading and merging raw data …")
    daily = load_and_merge()
    print(f"  {daily.shape[1]} columns, {daily.shape[0]} daily rows "
          f"({daily.index.min():%Y-%m-%d} → {daily.index.max():%Y-%m-%d})")

    monthly = resample_monthly(daily)
    print(f"  Resampled to {len(monthly)} monthly observations")

    print_section("REAL GDP GROWTH (GDPC1)")
    gdp_quarterly = monthly[GDP_COLUMN].resample("QE").last()
    target = build_gdp_target(gdp_quarterly)
    print(f"  Range: {target.index.min():%Y-%m-%d} → {target.index.max():%Y-%m-%d}")
    print(f"  {len(target)} quarterly observations")
    print(f"  Mean={target.mean():.2f}%  Std={target.std():.2f}%")
    recent_target = target.loc[RECENT_CUTOFF:]
    print(f"  Recent mean (since {RECENT_CUTOFF}): {recent_target.mean():.2f}%")
    print(f"  Last 8 quarters:")
    for dt, val in target.tail(8).items():
        print(f"    {dt:%Y-Q}{(dt.month - 1) // 3 + 1}  {val:+.2f}%")

    print("\nBuilding quarterly feature matrix …")
    features = build_feature_matrix(monthly)
    print(f"  {features.shape[1]} features retained (after dropping sparse ones)")

    # --- Correlation ---
    print_section("CORRELATION ANALYSIS")
    corr_df = correlation_analysis(features, target)
    print("\nTop 15 by |Spearman recent|:")
    print_table(corr_df[["feature", "spearman_recent", "spearman_full", "pearson_recent"]])

    # --- Granger ---
    print_section("GRANGER CAUSALITY")
    granger_df = granger_analysis(features, target, max_lag=4)
    sig_count = (granger_df["granger_min_p"] < 0.05).sum()
    print(f"\n{sig_count} features Granger-cause real GDP growth (p<0.05)")
    print("\nTop 15 by min p-value:")
    print_table(granger_df[["feature", "granger_min_p"]])

    # --- GBM Importance ---
    print_section("GRADIENT BOOSTING IMPORTANCE")
    imp_df, gbm_model = tree_importance(features, target)
    print("\nTop 15 (recent window):")
    print_table(imp_df[["feature", "importance_recent", "importance_full"]])

    # --- SHAP ---
    print_section("SHAP VALUES")
    shap_df, shap_vals, X_shap = shap_analysis(gbm_model, features, target)
    print("\nTop 15 by mean |SHAP|:")
    print_table(shap_df[["feature", "mean_abs_shap"]])

    # --- Consensus ---
    print_section("CONSENSUS RANKING")
    consensus = consensus_ranking(corr_df, granger_df, imp_df, shap_df)
    multi = consensus[consensus["methods_count"] >= 2]
    print(f"\n{len(multi)} features appear in top-15 of 2+ methods:\n")
    print_table(multi[["feature", "methods_count", "methods"]], n=len(multi))

    # --- Plots ---
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_correlation(corr_df, OUTPUT_DIR / "feature_impact_correlation.png")
    plot_granger(granger_df, OUTPUT_DIR / "feature_impact_granger.png")
    plot_importance(imp_df, OUTPUT_DIR / "feature_impact_importance.png")
    plot_shap(shap_vals, X_shap, OUTPUT_DIR / "feature_impact_shap.png")

    print(f"\nPlots saved to {OUTPUT_DIR}/")
    for name in [
        "feature_impact_correlation.png",
        "feature_impact_granger.png",
        "feature_impact_importance.png",
        "feature_impact_shap.png",
    ]:
        print(f"  ✓ {name}")


if __name__ == "__main__":
    main()
