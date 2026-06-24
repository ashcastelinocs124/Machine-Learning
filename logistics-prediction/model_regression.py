"""
Linear regression for trip duration prediction.

Target : actual_duration_hours
Split  : chronological
  train = trips dispatched 2022-01-01 .. 2024-06-30
  test  = trips dispatched 2024-07-01 .. 2024-12-31
Features are leak-safe (typical_distance_miles replaces actual_distance_miles;
historical aggregates use strictly-prior dispatch dates).
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (r2_score, mean_absolute_error,
                             mean_squared_error, max_error)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA = Path(__file__).parent
df = pd.read_csv(DATA / "features_trips.csv")
df["dispatch_date"] = pd.to_datetime(df["dispatch_date"])
df["load_date"]     = pd.to_datetime(df["load_date"])

# ---------------------------------------------------------------- target + split
y = df["actual_duration_hours"]
cutoff = pd.Timestamp("2024-07-01")
train_mask = df["dispatch_date"] < cutoff
test_mask  = ~train_mask

drop = ["actual_duration_hours", "trip_id", "dispatch_date", "load_date",
        "hire_date", "contract_start_date", "truck_acquisition_date",
        "trailer_acquisition_date"]
# high-cardinality entity ids: drop driver/truck/customer (too many levels for
# a linear model); keep route_id (only 57 levels) as categorical
high_card_drop = ["driver_id", "truck_id", "customer_id"]
feature_df = df.drop(columns=drop)

X_train, y_train = feature_df[train_mask].drop(columns=high_card_drop), y[train_mask]
X_test,  y_test  = feature_df[test_mask].drop(columns=high_card_drop),  y[test_mask]
print(f"Train: {len(X_train):,} rows  target mean={y_train.mean():.2f}h  std={y_train.std():.2f}h")
print(f"Test : {len(X_test):,} rows  target mean={y_test.mean():.2f}h  std={y_test.std():.2f}h")
print(f"Test period: {df.loc[test_mask,'dispatch_date'].min().date()} -> "
      f"{df.loc[test_mask,'dispatch_date'].max().date()}")

# ---------------------------------------------------------------- column typing
num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X_train.select_dtypes(exclude=[np.number]).columns.tolist()
print(f"\nNumeric cols ({len(num_cols)})")
print(f"Categorical cols ({len(cat_cols)}): {cat_cols}")

# ---------------------------------------------------------------- preprocessing
numeric_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale",  StandardScaler()),
])
categorical_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", max_categories=50)),
])
pre = ColumnTransformer([
    ("num", numeric_pipe,     num_cols),
    ("cat", categorical_pipe, cat_cols),
])
model = Pipeline([
    ("pre", pre),
    ("reg", LinearRegression()),
])

# ---------------------------------------------------------------- fit + evaluate
model.fit(X_train, y_train)
pred = model.predict(X_test)

r2   = r2_score(y_test, pred)
mae  = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
mape = (np.abs(pred - y_test) / y_test).mean() * 100

# baselines
base_mean_pred  = np.full_like(y_test, y_train.mean())
base_median     = np.full_like(y_test, y_train.median())
typical_pred    = df.loc[test_mask, "typical_transit_days"].values * 24.0  # transit days -> hours
# distance-only naive: duration = mean_speed^-1 * distance; estimate mph from train
mph = (df.loc[train_mask,"typical_distance_miles"] / y_train).mean()
dist_pred = df.loc[test_mask,"typical_distance_miles"].values / mph

print("\n================ TEST PERFORMANCE ================")
print(f"R2          : {r2:.4f}")
print(f"MAE         : {mae:.3f} hours  ({mae*60:.0f} min)")
print(f"RMSE        : {rmse:.3f} hours")
print(f"MAPE        : {mape:.2f}%")
print(f"Max error   : {max_error(y_test, pred):.2f} hours")

print("\n---------------- Baselines (test) ----------------")
print(f"Predict train mean ({y_train.mean():.2f}h): "
      f"R2={r2_score(y_test, base_mean_pred):.4f}  MAE={mean_absolute_error(y_test, base_mean_pred):.3f}")
print(f"Predict train median ({y_train.median():.2f}h): "
      f"R2={r2_score(y_test, base_median):.4f}  MAE={mean_absolute_error(y_test, base_median):.3f}")
print(f"typical_transit_days*24:    "
      f"R2={r2_score(y_test, typical_pred):.4f}  MAE={mean_absolute_error(y_test, typical_pred):.3f}")
print(f"distance / train-mph ({mph:.2f}mph):  "
      f"R2={r2_score(y_test, dist_pred):.4f}  MAE={mean_absolute_error(y_test, dist_pred):.3f}")

# error distribution
err = pred - y_test
print("\nError distribution (pred - actual), hours:")
print(pd.Series(err).describe().round(3).to_string())
print(f"\nWithin +/-15 min: {(np.abs(err) <= 0.25).mean():.1%}")
print(f"Within +/-30 min: {(np.abs(err) <= 0.5).mean():.1%}")
print(f"Within +/-1 hr : {(np.abs(err) <= 1.0).mean():.1%}")

# ---------------------------------------------------------------- top coefficients
feat_names = model.named_steps["pre"].get_feature_names_out()
coefs = model.named_steps["reg"].coef_
coef_df = (pd.DataFrame({"feature": feat_names, "coef": coefs})
           .assign(abs=lambda d: d.coef.abs())
           .sort_values("abs", ascending=False))
print("\nTop 15 features by |coef| (units = hours per 1-scaled-std):")
print(coef_df.head(15).to_string(index=False, float_format=lambda x: f"{x:+.3f}"))
