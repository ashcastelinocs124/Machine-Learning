"""
XGBoost regression for trip duration prediction.

Target : actual_duration_hours
Split  : chronological (identical to model_regression.py)
  train      = trips dispatched 2022-01-01 .. 2024-06-30
  early_stop = last 3 months of train (2024-04-01 .. 2024-06-30) for early stopping
  test       = trips dispatched 2024-07-01 .. 2024-12-31  (held out, untouched)

Features are leak-safe (typical_distance_miles replaces actual_distance_miles;
historical aggregates use strictly-prior dispatch dates). Native XGBoost
handling of categorical columns + missing values, so no OneHotEncoder / imputer
needed (unlike the linear model).
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (r2_score, mean_absolute_error,
                             mean_squared_error, max_error)
import xgboost as xgb

DATA = Path(__file__).parent
df = pd.read_csv(DATA / "features_trips.csv")
df["dispatch_date"] = pd.to_datetime(df["dispatch_date"])
df["load_date"]     = pd.to_datetime(df["load_date"])

# ---------------------------------------------------------------- target + split
y = df["actual_duration_hours"]
test_cutoff   = pd.Timestamp("2024-07-01")
val_cutoff    = pd.Timestamp("2024-04-01")   # last 3mo of train -> early stop
train_mask = df["dispatch_date"] < test_cutoff
test_mask  = ~train_mask
val_mask   = train_mask & (df["dispatch_date"] >= val_cutoff)
tr_mask    = train_mask & (df["dispatch_date"] < val_cutoff)

# columns to never use as features
drop = ["actual_duration_hours", "trip_id", "dispatch_date", "load_date",
        "hire_date", "contract_start_date", "truck_acquisition_date",
        "trailer_acquisition_date"]
# high-cardinality entity ids: drop driver/truck/customer (tree models handle
# high cardinality poorly without huge depth); keep route_id as categorical
high_card_drop = ["driver_id", "truck_id", "customer_id"]
feature_df = df.drop(columns=drop).drop(columns=high_card_drop)

# XGBoost wants categoricals as pandas 'category' dtype
cat_cols = feature_df.select_dtypes(exclude=[np.number]).columns.tolist()
for c in cat_cols:
    feature_df[c] = feature_df[c].astype("category")

X_tr,  y_tr  = feature_df[tr_mask],  y[tr_mask]
X_val, y_val = feature_df[val_mask], y[val_mask]
X_te,  y_te  = feature_df[test_mask], y[test_mask]
print(f"Train      : {len(X_tr):,} rows  (<= {val_cutoff.date()})  target mean={y_tr.mean():.2f}h")
print(f"Early-stop : {len(X_val):,} rows  ({val_cutoff.date()} -> {test_cutoff.date()})  target mean={y_val.mean():.2f}h")
print(f"Test       : {len(X_te):,} rows  ({test_cutoff.date()} ->)  target mean={y_te.mean():.2f}h")
print(f"Categorical cols ({len(cat_cols)}): {cat_cols}")

# ---------------------------------------------------------------- fit
model = xgb.XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    reg_lambda=1.0,
    tree_method="hist",
    enable_categorical=True,
    early_stopping_rounds=50,
    eval_metric="rmse",
    n_jobs=-1,
    random_state=42,
)
model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
print(f"\nBest iteration: {model.best_iteration}  (best val RMSE={model.best_score:.4f})")

# ---------------------------------------------------------------- evaluate
pred = model.predict(X_te)
r2   = r2_score(y_te, pred)
mae  = mean_absolute_error(y_te, pred)
rmse = np.sqrt(mean_squared_error(y_te, pred))
mape = (np.abs(pred - y_te) / y_te).mean() * 100

print("\n================ TEST PERFORMANCE ================")
print(f"R2          : {r2:.4f}")
print(f"MAE         : {mae:.3f} hours  ({mae*60:.0f} min)")
print(f"RMSE        : {rmse:.3f} hours")
print(f"MAPE        : {mape:.2f}%")
print(f"Max error   : {max_error(y_te, pred):.2f} hours")

# baselines for head-to-head (same as model_regression.py)
base_mean  = np.full_like(y_te, y_tr.mean())
mph = (df.loc[tr_mask,"typical_distance_miles"] / y_tr).mean()
dist_pred = df.loc[test_mask,"typical_distance_miles"].values / mph

print("\n---------------- Head-to-head (test) ----------------")
print(f"{'Model':<32s} {'R2':>8s} {'MAE(h)':>9s} {'RMSE(h)':>9s} {'MaxErr(h)':>10s}")
print(f"{'XGBoost (this model)':<32s} {r2:>8.4f} {mae:>9.3f} {rmse:>9.3f} {max_error(y_te,pred):>10.2f}")
print(f"{'Linear regression (prev)':<32s} {0.9740:>8.4f} {1.698:>9.3f} {2.300:>9.3f} {10.90:>10.2f}")
print(f"{'distance / train-mph':<32s} {r2_score(y_te,dist_pred):>8.4f} {mean_absolute_error(y_te,dist_pred):>9.3f} {np.sqrt(mean_squared_error(y_te,dist_pred)):>9.3f} {max_error(y_te,dist_pred):>10.2f}")
print(f"{'predict train mean':<32s} {r2_score(y_te,base_mean):>8.4f} {mean_absolute_error(y_te,base_mean):>9.3f} {np.sqrt(mean_squared_error(y_te,base_mean)):>9.3f} {max_error(y_te,base_mean):>10.2f}")

# error distribution + hit rates
err = pred - y_te
print("\nError distribution (pred - actual), hours:")
print(pd.Series(err).describe().round(3).to_string())
print(f"\nWithin +/-15 min: {(np.abs(err) <= 0.25).mean():.1%}  (linear was 13.7%)")
print(f"Within +/-30 min: {(np.abs(err) <= 0.5).mean():.1%}  (linear was 24.6%)")
print(f"Within +/-1 hr : {(np.abs(err) <= 1.0).mean():.1%}  (linear was 42.3%)")

# ---------------------------------------------------------------- feature importance (gain)
imp = (pd.DataFrame({
        "feature": model.get_booster().feature_names,
        "gain":    model.get_booster().get_score(importance_type="gain").values()
          if False else None,  # placeholder; use sklearn-style below for reliability
      }))
# use the sklearn API importance (total gain across splits) for a reliable mapping
sk_imp = model.feature_importances_
imp = (pd.DataFrame({"feature": feature_df.columns, "gain": sk_imp})
       .sort_values("gain", ascending=False))
total = imp["gain"].sum()
imp["share"] = imp["gain"] / total

print("\nTop 15 features by gain (importance_type='gain', share of total):")
print(imp.head(15).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print(f"\ntypical_distance_miles share: "
      f"{imp.loc[imp.feature=='typical_distance_miles','share'].iloc[0]:.1%}")
top5_share = imp.head(5)["share"].sum()
print(f"Top 5 features share of total gain: {top5_share:.1%}")
