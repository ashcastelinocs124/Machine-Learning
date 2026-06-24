"""
XGBoost regression for predicting detention_minutes at pickup/delivery events.

Target : detention_minutes
Grain  : one row per delivery_event (both Pickup and Delivery, 170,820 rows)
Split  : chronological
  train      = events scheduled 2022-01-01 .. 2024-06-30
  early_stop = 2024-04-01 .. 2024-06-30 (last 3mo of train)
  test       = events scheduled 2024-07-01 .. 2024-12-31  (held out)

Leak-safe: only pre-event features. Detention is an outcome of the event;
we predict it from what's known at schedule/dispatch time.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, max_error
import xgboost as xgb

DATA = Path(__file__).parent
df = pd.read_csv(DATA / "features_detention.csv")
df["scheduled_datetime"] = pd.to_datetime(df["scheduled_datetime"])
df["dispatch_date"]      = pd.to_datetime(df["dispatch_date"])

# ---------------------------------------------------------------- target + split
y = df["detention_minutes"]
test_cutoff = pd.Timestamp("2024-07-01")
val_cutoff  = pd.Timestamp("2024-04-01")
train_mask = df["scheduled_datetime"] < test_cutoff
test_mask  = ~train_mask
val_mask   = train_mask & (df["scheduled_datetime"] >= val_cutoff)
tr_mask    = train_mask & (df["scheduled_datetime"] < val_cutoff)

# columns to never use as features
drop = ["detention_minutes", "event_id", "scheduled_datetime", "dispatch_date",
        "load_date", "hire_date", "contract_start_date",
        "truck_acquisition_date", "trailer_acquisition_date"]
# high-cardinality entity ids: drop driver/truck/customer; keep route_id, facility_id (small)
high_card_drop = ["driver_id", "truck_id", "customer_id"]
feature_df = df.drop(columns=drop).drop(columns=high_card_drop)

# XGBoost categorical dtype
cat_cols = feature_df.select_dtypes(exclude=[np.number]).columns.tolist()
for c in cat_cols:
    feature_df[c] = feature_df[c].astype("category")

X_tr,  y_tr  = feature_df[tr_mask],  y[tr_mask]
X_val, y_val = feature_df[val_mask], y[val_mask]
X_te,  y_te  = feature_df[test_mask], y[test_mask]
print(f"Train      : {len(X_tr):,}  target mean={y_tr.mean():.2f} min  std={y_tr.std():.2f}")
print(f"Early-stop : {len(X_val):,}  target mean={y_val.mean():.2f} min")
print(f"Test       : {len(X_te):,}  target mean={y_te.mean():.2f} min  std={y_te.std():.2f} min")
print(f"Test period: {df.loc[test_mask,'scheduled_datetime'].min().date()} -> "
      f"{df.loc[test_mask,'scheduled_datetime'].max().date()}")
print(f"Features ({feature_df.shape[1]}): {len(feature_df.columns)-len(cat_cols)} numeric + {len(cat_cols)} categorical")

# ---------------------------------------------------------------- fit
model = xgb.XGBRegressor(
    n_estimators=1500, learning_rate=0.03, max_depth=6,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
    reg_lambda=1.0, tree_method="hist", enable_categorical=True,
    early_stopping_rounds=50, eval_metric="rmse", n_jobs=-1, random_state=42,
)
model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
print(f"\nBest iteration: {model.best_iteration}  (best val RMSE={model.best_score:.4f})")

# ---------------------------------------------------------------- evaluate
pred = model.predict(X_te)
r2   = r2_score(y_te, pred)
mae  = mean_absolute_error(y_te, pred)
rmse = np.sqrt(mean_squared_error(y_te, pred))
mape = (np.abs(pred - y_te) / y_te.replace(0, np.nan)).mean() * 100
err = pred - y_te

print("\n================ TEST PERFORMANCE ================")
print(f"R2          : {r2:.4f}")
print(f"MAE         : {mae:.3f} min  ({mae/60:.2f} h)")
print(f"RMSE        : {rmse:.3f} min")
print(f"MAPE        : {mape:.2f}%")
print(f"Max error   : {max_error(y_te, pred):.2f} min")
print(f"Within +/-15min: {(np.abs(err)<=15).mean():.1%}")
print(f"Within +/-30min: {(np.abs(err)<=30).mean():.1%}")
print(f"Within +/-1hr  : {(np.abs(err)<=60).mean():.1%}")

# baselines
base_mean   = np.full_like(y_te, y_tr.mean())
# event_type baseline: predict mean for that event_type from train
et_mean = df.loc[tr_mask].groupby("event_type")["detention_minutes"].mean()
et_pred = df.loc[test_mask, "event_type"].map(et_mean).values
# hour bucket baseline (the strongest signal we saw)
def hour_bucket(h): return "overnight" if (h<6 or h>=19) else "daytime"
hb_mean = df.loc[tr_mask].assign(hb=df.loc[tr_mask,"scheduled_datetime"].dt.hour.apply(hour_bucket)).groupby("hb")["detention_minutes"].mean()
hb_pred = pd.Series(df.loc[test_mask,"scheduled_datetime"].dt.hour.values).apply(hour_bucket).map(hb_mean).values

print("\n---------------- Head-to-head (test) ----------------")
print(f"{'Model':<32s} {'R2':>8s} {'MAE(min)':>10s} {'RMSE(min)':>10s}")
print(f"{'XGBoost (all features)':<32s} {r2:>8.4f} {mae:>10.3f} {rmse:>10.3f}")
print(f"{'event_type mean':<32s} {r2_score(y_te,et_pred):>8.4f} {mean_absolute_error(y_te,et_pred):>10.3f} {np.sqrt(mean_squared_error(y_te,et_pred)):>10.3f}")
print(f"{'hour-bucket mean (overnight/daytime)':<32s} {r2_score(y_te,hb_pred):>8.4f} {mean_absolute_error(y_te,hb_pred):>10.3f} {np.sqrt(mean_squared_error(y_te,hb_pred)):>10.3f}")
print(f"{'predict train mean':<32s} {r2_score(y_te,base_mean):>8.4f} {mean_absolute_error(y_te,base_mean):>10.3f} {np.sqrt(mean_squared_error(y_te,base_mean)):>10.3f}")

# ---------------------------------------------------------------- feature importance
imp = pd.DataFrame({"feature": feature_df.columns, "gain": model.feature_importances_})
imp = imp.sort_values("gain", ascending=False)
imp["share"] = imp["gain"] / imp["gain"].sum()
print("\nTop 15 features by gain (share of total):")
print(imp.head(15).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
top5_share = imp.head(5)["share"].sum()
print(f"\nTop 5 features share of total gain: {top5_share:.1%}")

# error distribution
print("\nError distribution (pred - actual), minutes:")
print(pd.Series(err).describe().round(2).to_string())
