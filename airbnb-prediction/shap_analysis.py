"""SHAP deep-dive for the Airbnb price-prediction CatBoost models.

Trains a per-city CatBoost model (identical to catboost_price_predict.py),
then generates rich SHAP visualisations for each city:

  1. Beeswarm   — per-observation impact direction + magnitude
  2. Bar        — global mean |SHAP| ranking
  3. Dependence — scatter of top features vs SHAP value
  4. Waterfall  — single-listing explanations (priciest + cheapest predicted)
  5. Force      — HTML force plots for sampled listings

Outputs land in  shap_output/<city>/

Usage:
    python shap_analysis.py                   # all cities
    python shap_analysis.py --city London     # single city
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="shap")

INPUT_FILE = "airbnb_cleaned.csv"
TARGET = "price"
OUTPUT_ROOT = Path("shap_output")

NUMERIC_FEATURES = [
    "latitude", "longitude", "minimum_nights",
    "calculated_host_listings_count", "availability_365",
    "estimated_bedrooms",
]
CAT_FEATURES = ["room_type", "neighbourhood"]
ALL_FEATURES = NUMERIC_FEATURES + CAT_FEATURES

TOP_N = 15
TOP_N_DEPENDENCE = 6


# ---------------------------------------------------------------------------
# Data prep  (mirrors catboost_price_predict.py exactly)
# ---------------------------------------------------------------------------

def add_estimated_bedrooms(df: pd.DataFrame, train_idx) -> pd.DataFrame:
    train_entire = df.loc[train_idx]
    train_entire = train_entire[train_entire["room_type"] == "Entire home/apt"]
    p33 = train_entire[TARGET].quantile(0.33)
    p66 = train_entire[TARGET].quantile(0.66)

    def _est(row):
        if row["room_type"] in ("Shared room", "Hotel room"):
            return 0
        if row["room_type"] == "Private room":
            return 1
        price = row[TARGET]
        if pd.isna(price) or price <= p33:
            return 1
        if price <= p66:
            return 2
        return 3

    df["estimated_bedrooms"] = df.apply(_est, axis=1)
    return df


def prepare_city(city_df: pd.DataFrame):
    base_cols = [c for c in NUMERIC_FEATURES if c != "estimated_bedrooms"]
    df = city_df[base_cols + CAT_FEATURES + [TARGET]].copy()

    for col in base_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in CAT_FEATURES:
        df[col] = df[col].astype(str).fillna("unknown")

    df = df.dropna(subset=[TARGET])
    df[base_cols] = df[base_cols].fillna(0)

    train_idx, test_idx = train_test_split(
        df.index, test_size=0.2, random_state=42,
    )
    df = add_estimated_bedrooms(df, train_idx)

    X_train = df.loc[train_idx, ALL_FEATURES]
    X_test = df.loc[test_idx, ALL_FEATURES]
    y_train = df.loc[train_idx, TARGET]
    y_test = df.loc[test_idx, TARGET]

    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def train_model(X_train, y_train):
    cat_indices = [ALL_FEATURES.index(c) for c in CAT_FEATURES]
    model = CatBoostRegressor(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3,
        random_seed=42,
        verbose=0,
        cat_features=cat_indices,
    )
    model.fit(X_train, y_train)
    return model


# ---------------------------------------------------------------------------
# SHAP computation
# ---------------------------------------------------------------------------

def compute_shap(model: CatBoostRegressor, X: pd.DataFrame) -> shap.Explanation:
    explainer = shap.TreeExplainer(model)
    explanation = explainer(X)
    return explanation


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_beeswarm(explanation: shap.Explanation, city: str, out: Path) -> None:
    plt.figure(figsize=(12, 8))
    shap.plots.beeswarm(explanation, max_display=TOP_N, show=False)
    plt.title(f"{city} — SHAP Beeswarm", fontsize=14, pad=12)
    plt.tight_layout()
    plt.savefig(out / "beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close("all")


def plot_bar(explanation: shap.Explanation, city: str, out: Path) -> None:
    plt.figure(figsize=(10, 7))
    shap.plots.bar(explanation, max_display=TOP_N, show=False)
    plt.title(f"{city} — Mean |SHAP| Feature Importance", fontsize=14, pad=12)
    plt.tight_layout()
    plt.savefig(out / "bar.png", dpi=150, bbox_inches="tight")
    plt.close("all")


def plot_dependence(
    explanation: shap.Explanation, X: pd.DataFrame, city: str, out: Path,
) -> None:
    mean_abs = np.abs(explanation.values).mean(axis=0)
    n = min(TOP_N_DEPENDENCE, len(mean_abs))
    top_idx = np.argsort(mean_abs)[::-1][:n]
    top_feats = [X.columns[i] for i in top_idx]

    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    axes = np.atleast_2d(axes)

    for i, feat in enumerate(top_feats):
        r, c = divmod(i, cols)
        ax = axes[r, c]
        feat_idx = list(X.columns).index(feat)
        sv = explanation.values[:, feat_idx]
        fv = X[feat].values if X[feat].dtype.kind in "iufb" else np.arange(len(sv))

        ax.scatter(fv, sv, c=sv, cmap="coolwarm", alpha=0.5, s=14, edgecolors="none")
        ax.set_xlabel(feat, fontsize=10)
        ax.set_ylabel("SHAP value", fontsize=10)
        ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
        ax.set_title(feat, fontsize=11)

    for i in range(n, rows * cols):
        r, c = divmod(i, cols)
        axes[r, c].set_visible(False)

    fig.suptitle(f"{city} — SHAP Dependence (Top {n} Features)", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(out / "dependence.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_waterfall(
    explanation: shap.Explanation,
    y_pred: np.ndarray,
    city: str,
    out: Path,
) -> None:
    hi_idx = int(np.argmax(y_pred))
    plt.figure(figsize=(10, 7))
    shap.plots.waterfall(explanation[hi_idx], max_display=TOP_N, show=False)
    plt.title(f"{city} — Highest Predicted Price (${y_pred[hi_idx]:.0f})", fontsize=13, pad=12)
    plt.tight_layout()
    plt.savefig(out / "waterfall_high.png", dpi=150, bbox_inches="tight")
    plt.close("all")

    lo_idx = int(np.argmin(y_pred))
    plt.figure(figsize=(10, 7))
    shap.plots.waterfall(explanation[lo_idx], max_display=TOP_N, show=False)
    plt.title(f"{city} — Lowest Predicted Price (${y_pred[lo_idx]:.0f})", fontsize=13, pad=12)
    plt.tight_layout()
    plt.savefig(out / "waterfall_low.png", dpi=150, bbox_inches="tight")
    plt.close("all")


def save_force_html(
    explainer_expected: float,
    shap_values_matrix: np.ndarray,
    X: pd.DataFrame,
    city: str,
    out: Path,
    n_samples: int = 200,
) -> None:
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X), size=min(n_samples, len(X)), replace=False)
    idx.sort()

    force = shap.force_plot(
        explainer_expected,
        shap_values_matrix[idx],
        X.iloc[idx],
        show=False,
    )
    shap.save_html(str(out / "force.html"), force)


# ---------------------------------------------------------------------------
# Per-city driver
# ---------------------------------------------------------------------------

def run_city(city: str, city_df: pd.DataFrame) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {city}  ({len(city_df):,} listings)")
    print(f"{'=' * 60}")

    X_train, X_test, y_train, y_test = prepare_city(city_df)
    print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    model = train_model(X_train, y_train)
    y_pred = model.predict(X_test)
    from sklearn.metrics import r2_score, mean_absolute_error
    print(f"  R²:  {r2_score(y_test, y_pred):.4f}  |  MAE: {mean_absolute_error(y_test, y_pred):.2f}")

    print("  Computing SHAP values (test set) …")
    explanation = compute_shap(model, X_test)

    mean_abs = np.abs(explanation.values).mean(axis=0)
    ranking = sorted(zip(X_test.columns, mean_abs), key=lambda x: x[1], reverse=True)
    print(f"\n  {'Feature':<35} {'Mean |SHAP|':>12}")
    print(f"  {'─' * 35} {'─' * 12}")
    for feat, val in ranking:
        print(f"  {feat:<35} {val:>12.2f}")

    out = OUTPUT_ROOT / city.lower().replace(" ", "_")
    out.mkdir(parents=True, exist_ok=True)
    print(f"\n  Saving plots → {out}/")

    plot_beeswarm(explanation, city, out)
    print(f"    ✓ beeswarm.png")

    plot_bar(explanation, city, out)
    print(f"    ✓ bar.png")

    plot_dependence(explanation, X_test, city, out)
    print(f"    ✓ dependence.png")

    plot_waterfall(explanation, y_pred, city, out)
    print(f"    ✓ waterfall_high.png")
    print(f"    ✓ waterfall_low.png")

    explainer = shap.TreeExplainer(model)
    raw_shap = explainer.shap_values(X_test)
    save_force_html(explainer.expected_value, raw_shap, X_test, city, out)
    print(f"    ✓ force.html")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="SHAP analysis for Airbnb price models")
    parser.add_argument("--city", type=str, default=None, help="Run for a single city only")
    args = parser.parse_args()

    df = pd.read_csv(INPUT_FILE, low_memory=False)
    df["city"] = df["city"].astype(str).str.strip()
    cities = sorted(df["city"].unique())

    if args.city:
        match = [c for c in cities if c.lower() == args.city.lower()]
        if not match:
            print(f"City '{args.city}' not found. Available: {', '.join(cities)}")
            return
        cities = match

    print(f"SHAP Analysis — {len(cities)} city/cities")

    for city in cities:
        city_df = df[df["city"] == city]
        if city_df[TARGET].dropna().empty:
            print(f"\nSkipping {city} (no price data)")
            continue
        run_city(city, city_df)

    print(f"\nDone. All outputs in {OUTPUT_ROOT}/")


if __name__ == "__main__":
    main()
