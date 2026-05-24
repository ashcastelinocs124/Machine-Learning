import pandas as pd
import numpy as np

INPUT_FILE = "airbnb_top_cities.csv"
OUTPUT_FILE = "airbnb_cleaned.csv"

NUMERIC_COLS = [
    "latitude", "longitude", "price", "minimum_nights",
    "number_of_reviews", "reviews_per_month",
    "calculated_host_listings_count", "availability_365",
    "number_of_reviews_ltm",
]


def remove_outliers_iqr(group):
    prices = group["price"]
    q1 = prices.quantile(0.25)
    q3 = prices.quantile(0.75)
    iqr = q3 - q1
    return group[(prices >= q1 - 1.5 * iqr) & (prices <= q3 + 1.5 * iqr)]


def main():
    df = pd.read_csv(INPUT_FILE, low_memory=False)
    df["city"] = df["city"].astype(str).str.strip()
    df = df[df["city"].str.isalpha()]

    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["price"])
    print(f"Rows with valid price: {len(df):,}\n")

    # --- Remove outliers per city ---
    cleaned = df.groupby("city", group_keys=False).apply(remove_outliers_iqr)
    cleaned.to_csv(OUTPUT_FILE, index=False)

    removed = len(df) - len(cleaned)
    print(f"Outliers removed:  {removed:,}")
    print(f"Cleaned dataset:   {len(cleaned):,} rows  ->  saved to '{OUTPUT_FILE}'\n")

    # --- Correlation with price per city ---
    corr_cols = [c for c in NUMERIC_COLS if c != "price"]

    print("=" * 72)
    print("PEARSON CORRELATION WITH PRICE (per city, outliers removed)")
    print("=" * 72)

    summary_rows = []

    for city in sorted(cleaned["city"].unique()):
        sub = cleaned[cleaned["city"] == city]

        corrs = sub[corr_cols + ["price"]].corr()["price"].drop("price").sort_values(
            key=abs, ascending=False
        )

        print(f"\n  {city}  ({len(sub):,} listings)")
        print(f"  {'Variable':<35} {'r':>8}  Strength")
        print("  " + "-" * 58)

        for var, r in corrs.items():
            strength = (
                "strong" if abs(r) >= 0.5
                else "moderate" if abs(r) >= 0.3
                else "weak" if abs(r) >= 0.1
                else "negligible"
            )
            marker = " ***" if abs(r) >= 0.5 else " **" if abs(r) >= 0.3 else ""
            print(f"  {var:<35} {r:>8.4f}  {strength}{marker}")

        top_var = corrs.index[0]
        top_r = corrs.iloc[0]
        summary_rows.append({"city": city, "top_variable": top_var, "r": top_r})

    # --- Summary table ---
    print("\n" + "=" * 72)
    print("SUMMARY: Highest correlated variable with price per city")
    print("=" * 72)
    print(f"  {'City':<15} {'Top Variable':<35} {'r':>8}")
    print("  " + "-" * 58)
    for row in summary_rows:
        print(f"  {row['city']:<15} {row['top_variable']:<35} {row['r']:>8.4f}")


if __name__ == "__main__":
    main()
