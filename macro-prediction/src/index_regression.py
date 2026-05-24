"""Category-index regression for real GDP growth.

Groups 71 individual features into ~14 category indices via PCA (first
principal component), then fits a Gradient Boosting regression from
those indices to QoQ annualized real GDP growth.

Usage:
    python -m src.index_regression
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
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

from src.config import AI_CAPEX_COLUMNS, CATEGORY_GROUPS, DATA_PROCESSED, DATA_RAW

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="shap")

GDP_COLUMN = "real_gdp"
RECENT_CUTOFF = "2023-01-01"
OUTPUT_DIR = DATA_PROCESSED

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

SIGN_POSITIVE = {
    "economic_activity", "consumer_spending", "activity_indices",
    "equity_indices", "sectors", "semis_tech", "ai_capex",
}


# ---------------------------------------------------------------------------
# Data loading (reused from analyze_feature_impact)
# ---------------------------------------------------------------------------

def load_and_merge() -> pd.DataFrame:
    csv_files = sorted(DATA_RAW.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSVs found in {DATA_RAW}")
    frames = [pd.read_csv(f, parse_dates=["date"], index_col="date") for f in csv_files]
    merged = pd.concat(frames, axis=1)
    merged = merged.loc[:, ~merged.columns.duplicated(keep="first")]
    merged.sort_index(inplace=True)
    return merged


def resample_monthly(df: pd.DataFrame) -> pd.DataFrame:
    return df.ffill().resample("ME").last()


def build_gdp_target(quarterly_gdp: pd.Series) -> pd.Series:
    gdp = quarterly_gdp.dropna()
    qoq = gdp.pct_change()
    annualized = ((1 + qoq) ** 4 - 1) * 100
    annualized.name = "gdp_growth"
    return annualized.dropna()


def build_quarterly_features(monthly: pd.DataFrame) -> pd.DataFrame:
    exclude = {GDP_COLUMN}
    feature_cols = [c for c in monthly.columns if c not in exclude]
    quarterly = monthly[feature_cols].resample("QE").last()

    transformed = pd.DataFrame(index=quarterly.index)
    for col in quarterly.columns:
        series = quarterly[col].dropna()
        if len(series) < 8:
            continue
        if col in LEVEL_CHANGE_FEATURES:
            transformed[col] = quarterly[col].diff(1)
        else:
            transformed[col] = quarterly[col].pct_change(1) * 100
    return transformed


# ---------------------------------------------------------------------------
# PCA index construction
# ---------------------------------------------------------------------------

def build_category_indices(
    features: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, dict]]:
    """Build one PCA-based index per category.

    Returns:
        indices: DataFrame with one column per category index.
        meta: dict mapping category -> {variance_explained, loadings, columns_used}.
    """
    indices = pd.DataFrame(index=features.index)
    meta: dict[str, dict] = {}

    for cat_name, col_list in CATEGORY_GROUPS.items():
        available = [c for c in col_list if c in features.columns]
        if len(available) < 2:
            if len(available) == 1:
                s = features[available[0]].copy()
                indices[cat_name] = (s - s.mean()) / s.std()
                meta[cat_name] = {
                    "variance_explained": 1.0,
                    "loadings": {available[0]: 1.0},
                    "columns_used": available,
                }
            continue

        sub = features[available].dropna(axis=0, how="any")
        if len(sub) < 10:
            continue

        scaler = StandardScaler()
        z = scaler.fit_transform(sub)

        pca = PCA(n_components=1)
        pc1 = pca.fit_transform(z).ravel()

        loadings = dict(zip(available, pca.components_[0]))
        sign_ref = _pick_sign_reference(cat_name, loadings)
        if sign_ref < 0:
            pc1 = -pc1
            loadings = {k: -v for k, v in loadings.items()}

        idx_series = pd.Series(pc1, index=sub.index, name=cat_name)
        indices = indices.join(idx_series, how="left")

        meta[cat_name] = {
            "variance_explained": float(pca.explained_variance_ratio_[0]),
            "loadings": loadings,
            "columns_used": available,
        }

    return indices, meta


def _pick_sign_reference(cat_name: str, loadings: dict[str, float]) -> float:
    """Return +1 or -1 to align index so positive = expansionary."""
    top_var = max(loadings, key=lambda k: abs(loadings[k]))
    raw_sign = np.sign(loadings[top_var])

    if cat_name in SIGN_POSITIVE:
        return raw_sign
    return -raw_sign


# ---------------------------------------------------------------------------
# Regression + SHAP
# ---------------------------------------------------------------------------

def fit_and_evaluate(
    indices: pd.DataFrame, target: pd.Series
) -> tuple[GradientBoostingRegressor, pd.DataFrame, pd.DataFrame]:
    target_name = target.name
    aligned = indices.join(target, how="inner").dropna()
    recent = aligned.loc[RECENT_CUTOFF:]

    X_full, y_full = aligned.drop(columns=target_name), aligned[target_name]
    X_rec, y_rec = recent.drop(columns=target_name), recent[target_name]

    gbm_full = GradientBoostingRegressor(
        n_estimators=300, max_depth=3, learning_rate=0.05,
        subsample=0.8, random_state=42,
    )
    gbm_full.fit(X_full, y_full)

    gbm_rec = GradientBoostingRegressor(
        n_estimators=200, max_depth=2, learning_rate=0.05,
        subsample=0.8, random_state=42,
    )
    gbm_rec.fit(X_rec, y_rec)

    imp = pd.DataFrame({
        "category": X_full.columns,
        "importance_full": gbm_full.feature_importances_,
        "importance_recent": gbm_rec.feature_importances_,
    }).sort_values("importance_recent", ascending=False).reset_index(drop=True)

    pred_full = pd.Series(gbm_full.predict(X_full), index=X_full.index, name="predicted")
    r2_full = r2_score(y_full, pred_full)
    pred_rec = pd.Series(gbm_rec.predict(X_rec), index=X_rec.index, name="predicted")
    r2_rec = r2_score(y_rec, pred_rec)

    fit_df = pd.DataFrame({"actual": y_full, "predicted": pred_full})

    print(f"  R-squared (full history):  {r2_full:.3f}")
    print(f"  R-squared (recent 2023+):  {r2_rec:.3f}")

    return gbm_full, imp, fit_df


def run_shap(
    model: GradientBoostingRegressor,
    indices: pd.DataFrame,
    target: pd.Series,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    target_name = target.name
    aligned = indices.join(target, how="inner").dropna()
    X = aligned.drop(columns=target_name)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    mean_abs = np.abs(shap_values).mean(axis=0)
    df = pd.DataFrame({
        "category": X.columns,
        "mean_abs_shap": mean_abs,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    return df, shap_values, X


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_shap_bar(shap_values: np.ndarray, X: pd.DataFrame, path: Path) -> None:
    shap.summary_plot(shap_values, X, plot_type="bar", max_display=20, show=False)
    plt.title("Category Index SHAP Importance for GDP Growth", fontsize=13)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")


def plot_fit(fit_df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(fit_df.index, fit_df["actual"], label="Actual GDP Growth", linewidth=1.5)
    ax.plot(fit_df.index, fit_df["predicted"], label="Predicted (GBM)", linewidth=1.5, linestyle="--")
    ax.axhline(0, color="grey", linewidth=0.5)
    ax.set_ylabel("QoQ Annualized GDP Growth (%)")
    ax.set_title("Actual vs Predicted Real GDP Growth (Category-Index GBM)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_loadings(meta: dict[str, dict], path: Path) -> None:
    cats = sorted(meta.keys())
    all_vars: set[str] = set()
    for info in meta.values():
        all_vars.update(info["loadings"].keys())
    all_vars_sorted = sorted(all_vars)

    matrix = np.zeros((len(cats), len(all_vars_sorted)))
    for i, cat in enumerate(cats):
        for var, weight in meta[cat]["loadings"].items():
            j = all_vars_sorted.index(var)
            matrix[i, j] = weight

    fig, ax = plt.subplots(figsize=(20, 8))
    im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_yticks(range(len(cats)))
    ax.set_yticklabels(cats)
    ax.set_xticks(range(len(all_vars_sorted)))
    ax.set_xticklabels(all_vars_sorted, rotation=90, fontsize=7)
    ax.set_title("PCA Loadings by Category Index")
    fig.colorbar(im, ax=ax, shrink=0.6, label="PC1 Loading")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_section(title: str) -> None:
    width = 72
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


def print_table(df: pd.DataFrame, n: int = 15) -> None:
    with pd.option_context(
        "display.max_rows", n, "display.width", 120,
        "display.float_format", "{:.4f}".format,
    ):
        print(df.head(n).to_string(index=False))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Loading and merging raw data …")
    daily = load_and_merge()
    monthly = resample_monthly(daily)
    print(f"  {daily.shape[1]} columns → {len(monthly)} monthly rows")

    print_section("REAL GDP GROWTH (GDPC1)")
    gdp_q = monthly[GDP_COLUMN].resample("QE").last()
    target = build_gdp_target(gdp_q)
    print(f"  {len(target)} quarterly obs  "
          f"({target.index.min():%Y-%m-%d} → {target.index.max():%Y-%m-%d})")
    print(f"  Mean={target.mean():.2f}%  Std={target.std():.2f}%")

    print("\nBuilding quarterly features …")
    features = build_quarterly_features(monthly)
    print(f"  {features.shape[1]} individual features")

    print_section("PCA CATEGORY INDICES")
    indices, meta = build_category_indices(features)
    print(f"  {indices.shape[1]} category indices built\n")

    for cat in sorted(meta.keys()):
        info = meta[cat]
        var_pct = info["variance_explained"] * 100
        top3 = sorted(info["loadings"].items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        top3_str = ", ".join(f"{k} ({v:+.2f})" for k, v in top3)
        print(f"  {cat:22s}  var_explained={var_pct:5.1f}%  top: {top3_str}")

    print_section("GBM REGRESSION")
    model, imp_df, fit_df = fit_and_evaluate(indices, target)

    print("\nCategory importance (recent window):")
    print_table(imp_df[["category", "importance_recent", "importance_full"]])

    print_section("SHAP ANALYSIS")
    shap_df, shap_vals, X_shap = run_shap(model, indices, target)
    print("\nCategory ranking by mean |SHAP|:")
    print_table(shap_df[["category", "mean_abs_shap"]])

    print_section("ACTUAL vs PREDICTED (last 8 quarters)")
    tail = fit_df.tail(8)
    for dt, row in tail.iterrows():
        print(f"  {dt:%Y-Q}{(dt.month - 1) // 3 + 1}  "
              f"actual={row['actual']:+6.2f}%  predicted={row['predicted']:+6.2f}%")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_shap_bar(shap_vals, X_shap, OUTPUT_DIR / "index_regression_shap.png")
    plot_fit(fit_df, OUTPUT_DIR / "index_regression_fit.png")
    plot_loadings(meta, OUTPUT_DIR / "index_regression_loadings.png")

    print(f"\nPlots saved to {OUTPUT_DIR}/")
    for name in [
        "index_regression_shap.png",
        "index_regression_fit.png",
        "index_regression_loadings.png",
    ]:
        print(f"  ✓ {name}")


if __name__ == "__main__":
    main()
