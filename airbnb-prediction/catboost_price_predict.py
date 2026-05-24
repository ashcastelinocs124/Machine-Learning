import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

INPUT_FILE = "airbnb_cleaned.csv"
TARGET = "price"

NUMERIC_FEATURES = [
    "latitude", "longitude", "minimum_nights",
    "calculated_host_listings_count", "availability_365",
    "estimated_bedrooms",
]
CAT_FEATURES = ["room_type", "neighbourhood"]

ALL_FEATURES = NUMERIC_FEATURES + CAT_FEATURES

OLD_R2 = {
    "Amsterdam": 0.2956, "Bangkok": 0.3438, "Barcelona": 0.6241,
    "London": 0.5096, "Rome": 0.4337,
}


def add_estimated_bedrooms(df, train_idx):
    """Leakage-safe: percentile bins computed from training set only."""
    train_entire = df.loc[train_idx]
    train_entire = train_entire[train_entire["room_type"] == "Entire home/apt"]
    p33 = train_entire[TARGET].quantile(0.33)
    p66 = train_entire[TARGET].quantile(0.66)

    def est_bedrooms(row):
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

    df["estimated_bedrooms"] = df.apply(est_bedrooms, axis=1)
    return df


def train_city_model(city: str, city_df: pd.DataFrame) -> dict:
    base_cols = [c for c in NUMERIC_FEATURES if c != "estimated_bedrooms"]
    df = city_df[base_cols + CAT_FEATURES + [TARGET]].copy()

    for col in base_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in CAT_FEATURES:
        df[col] = df[col].astype(str).fillna("unknown")

    df = df.dropna(subset=[TARGET])
    df[base_cols] = df[base_cols].fillna(0)

    train_idx, test_idx = train_test_split(
        df.index, test_size=0.2, random_state=42
    )

    df = add_estimated_bedrooms(df, train_idx)

    X_train = df.loc[train_idx, ALL_FEATURES]
    X_test = df.loc[test_idx, ALL_FEATURES]
    y_train = df.loc[train_idx, TARGET]
    y_test = df.loc[test_idx, TARGET]

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

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    importances = model.get_feature_importance()
    feat_imp = sorted(
        zip(ALL_FEATURES, importances), key=lambda x: x[1], reverse=True
    )

    return {
        "city": city,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "mae": mae,
        "r2": r2,
        "feature_importances": feat_imp,
    }


def main():
    df = pd.read_csv(INPUT_FILE, low_memory=False)
    df["city"] = df["city"].astype(str).str.strip()
    cities = sorted(df["city"].unique())

    results = []
    for city in cities:
        print(f"Training model for {city}...")
        city_df = df[df["city"] == city]
        result = train_city_model(city, city_df)
        results.append(result)

    print("\n" + "=" * 78)
    print("CATBOOST FEATURE IMPORTANCE PER CITY (+ estimated_bedrooms)")
    print("=" * 78)

    for r in results:
        print(f"\n  {r['city']}  (train: {r['train_size']:,} | test: {r['test_size']:,})")
        print(f"  MAE: {r['mae']:.2f}  |  R²: {r['r2']:.4f}")
        print(f"  {'Feature':<40} {'Importance':>12}")
        print("  " + "-" * 54)
        for feat, imp in r["feature_importances"]:
            bar = "█" * int(imp / 2)
            print(f"  {feat:<40} {imp:>10.2f}%  {bar}")

    print("\n" + "=" * 78)
    print("SUMMARY: Top predictor of price per city")
    print("=" * 78)
    print(f"  {'City':<15} {'Top Feature':<30} {'Importance':>12} {'R²':>8}")
    print("  " + "-" * 67)
    for r in results:
        top_feat, top_imp = r["feature_importances"][0]
        print(f"  {r['city']:<15} {top_feat:<30} {top_imp:>10.2f}%  {r['r2']:>8.4f}")

    print("\n" + "=" * 78)
    print("R² COMPARISON: Before vs After adding estimated_bedrooms")
    print("=" * 78)
    print(f"  {'City':<15} {'Old R²':>10} {'New R²':>10} {'Change':>10} {'% Improve':>12}")
    print("  " + "-" * 58)
    for r in results:
        old = OLD_R2.get(r["city"], 0)
        new = r["r2"]
        delta = new - old
        pct = (delta / old * 100) if old > 0 else 0
        arrow = "+" if delta > 0 else ""
        print(f"  {r['city']:<15} {old:>10.4f} {new:>10.4f} {arrow}{delta:>9.4f} {arrow}{pct:>10.1f}%")


if __name__ == "__main__":
    main()
