# Airbnb Price Prediction Across Top Global Cities

A machine learning project that predicts Airbnb listing prices across 7 major cities using CatBoost gradient boosting, with feature engineering to improve model accuracy.

## Dataset

**Source:** Airbnb listings scraped March 2026, covering ~294K listings across 7 cities.

| City | Raw Listings | After Cleaning |
|---|---|---|
| London | 96,871 | 57,768 |
| Paris | 81,853 | No price data |
| Rome | 37,652 | 31,041 |
| Bangkok | 28,806 | 21,119 |
| Barcelona | 19,410 | 14,535 |
| Sydney | 17,730 | No price data |
| Amsterdam | 10,480 | 5,573 |

Paris and Sydney were excluded due to missing price data. Outliers were removed per city using the IQR method (Q1 - 1.5*IQR to Q3 + 1.5*IQR), eliminating ~9,900 extreme values.

## Approach

### 1. Exploratory Analysis

Grouped listings by city and examined price distributions. Outlier rates ranged from 4.9% (Barcelona) to 9.3% (Bangkok).

### 2. Correlation Testing

Pearson correlation with price revealed weak linear relationships across all cities — the strongest was Barcelona's `minimum_nights` at r = -0.35. This signaled that simple linear features alone would not suffice.

### 3. CatBoost Baseline (7 features)

Trained a CatBoost regressor per city using the available numeric and categorical features:

- `latitude`, `longitude`, `minimum_nights`, `calculated_host_listings_count`, `availability_365`
- `room_type`, `neighbourhood`

| City | R² | MAE |
|---|---|---|
| Barcelona | 0.624 | 39.23 |
| London | 0.510 | 46.92 |
| Rome | 0.434 | 37.03 |
| Bangkok | 0.344 | 489.35 |
| Amsterdam | 0.296 | 67.14 |

R² was low because the dataset lacked the most important price driver: **property size** (bedrooms, bathrooms, sqft).

### 4. Feature Engineering

We introduced `estimated_bedrooms` — a proxy for property size derived from `room_type` and within-city price percentiles:

| Room Type | Estimated Bedrooms |
|---|---|
| Shared room / Hotel room | 0 |
| Private room | 1 |
| Entire home (price <= 33rd pctl) | 1 |
| Entire home (price 33rd-66th pctl) | 2 |
| Entire home (price > 66th pctl) | 3+ |

Percentile thresholds were computed on the training set only to prevent data leakage.

### 5. CatBoost with Feature Engineering (8 features)

| City | Old R² | New R² | Improvement | MAE |
|---|---|---|---|---|
| **London** | 0.510 | **0.818** | +60.5% | 27.35 |
| **Barcelona** | 0.624 | **0.816** | +30.7% | 26.88 |
| **Rome** | 0.434 | **0.741** | +70.9% | 23.28 |
| **Amsterdam** | 0.296 | **0.733** | +148.0% | 38.39 |
| **Bangkok** | 0.344 | **0.661** | +92.1% | 324.33 |

### 6. Top Price Predictor by City

`estimated_bedrooms` dominated feature importance in every city:

| City | Top Feature | Importance |
|---|---|---|
| London | estimated_bedrooms | 75.7% |
| Amsterdam | estimated_bedrooms | 62.8% |
| Rome | estimated_bedrooms | 60.2% |
| Bangkok | estimated_bedrooms | 54.9% |
| Barcelona | estimated_bedrooms | 52.1% |

## Key Findings

1. **Property size is the single biggest price driver** across all cities, even when approximated from price bins.
2. **Room type** is the second most important feature, especially in London (where entire homes average 2.5x the price of private rooms).
3. **Location** (longitude, latitude, neighbourhood) matters most in Rome, where proximity to the historic center significantly affects price.
4. **Minimum nights** is uniquely important in Barcelona (14.4%), reflecting its short-stay tourist premium.
5. **Review-based features** (number of reviews, reviews per month) had negligible predictive power and were dropped without loss.

## Project Structure

```
.
├── airbnb_top_cities.csv           # Raw dataset (294K listings, 7 cities)
├── airbnb_cleaned.csv              # Cleaned dataset (outliers removed)
├── data_by_city/                   # Individual CSV per city
│   ├── london.csv
│   ├── paris.csv
│   ├── rome.csv
│   ├── bangkok.csv
│   ├── barcelona.csv
│   ├── sydney.csv
│   └── amsterdam.csv
├── group_by_city.py                # Script to split data by city
├── clean_and_correlate.py          # Outlier removal + Pearson correlation
├── catboost_price_predict.py       # CatBoost training + feature engineering
└── README.md
```

## How to Run

```bash
pip install pandas numpy catboost scikit-learn

# Step 1: Group data by city
python group_by_city.py

# Step 2: Clean data + correlation analysis
python clean_and_correlate.py

# Step 3: Train CatBoost models with feature engineering
python catboost_price_predict.py
```

## Requirements

- Python 3.9+
- pandas
- numpy
- catboost
- scikit-learn
