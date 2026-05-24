import pandas as pd
from pathlib import Path

INPUT_FILE = "airbnb_top_cities.csv"
OUTPUT_DIR = Path("data_by_city")

def main():
    df = pd.read_csv(INPUT_FILE)

    df["city"] = df["city"].astype(str).str.strip()
    df = df[df["city"].str.isalpha()]

    OUTPUT_DIR.mkdir(exist_ok=True)

    print(f"Total rows: {len(df):,}")
    print(f"Cities found: {df['city'].nunique()}\n")
    print(f"{'City':<15} {'Listings':>10}")
    print("-" * 27)

    for city, group in sorted(df.groupby("city"), key=lambda x: -len(x[1])):
        filename = OUTPUT_DIR / f"{city.lower().replace(' ', '_')}.csv"
        group.to_csv(filename, index=False)
        print(f"{city:<15} {len(group):>10,}")

    print(f"\nCSV files saved to '{OUTPUT_DIR}/'")


if __name__ == "__main__":
    main()
