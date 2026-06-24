"""
XGBoost regression for trip duration -- TOP-5 features only.

Same chronological split and params as model_xgboost.py, but restricted to the
5 features that carried 94.2% of total gain in the full model:
  route_id (cat), typical_distance_miles, typical_transit_days,
  fuel_surcharge, fuel_surcharge_rate
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, max_error
import xgboost as xgb

DATA = Path(__file__).parent
df = pd.read_csv(DATA / "features_trips.csv")
df["dispatch_date"] = pd.to_datetime(df["dispatch_date"])

TOP5 = ["route_id", "typical_distance_miles", "typical_transit_days",
        "fuel_surcharge", "fuel_surcharge_rate"]

y = df["actual_duration_hours"]
test_cutoff = pd.Timestamp("2024-07-01")
val_cutoff  = pd.Timestamp("2024-04-01")
train_mask = df["dispatch_date"] < test_cutoff
test_mask  = ~train_mask
val_mask   = train_mask & (df["dispatch_date"] >= val_cutoff)
tr_mask    = train_mask & (df["dispatch_date"] < val_cutoff)

X = df[TOP5].copy()
X["route_id"] = X["route_id"].astype("category")   # only cat in TOP5

X_tr, y_tr  = X[tr_mask],  y[tr_mask]
X_val, y_val = X[val_mask], y[val_mask]
X_te, y_te  = X[test_mask], y[test_mask]
print(f"Train: {len(X_tr):,}  Early-stop: {len(X_val):,}  Test: {len(X_te):,}")
print(f"Features: {TOP5}")

model = xgb.XGBRegressor(
    n_estimators=1000, learning_rate=0.05, max_depth=6,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
    reg_lambda=1.0, tree_method="hist", enable_categorical=True,
    early_stopping_rounds=50, eval_metric="rmse", n_jobs=-1, random_state=42,
)
model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
print(f"Best iteration: {model.best_iteration}  (best val RMSE={model.best_score:.4f})")

pred = model.predict(X_te)
r2, mae = r2_score(y_te, pred), mean_absolute_error(y_te, pred)
rmse = np.sqrt(mean_squared_error(y_te, pred))
mape = (np.abs(pred - y_te) / y_te).mean() * 100
err = pred - y_te

print("\n================ TEST PERFORMANCE (TOP-5 ONLY) ================")
print(f"R2: {r2:.4f}   MAE: {mae:.3f}h ({mae*60:.0f}min)   RMSE: {rmse:.3f}h   MAPE: {mape:.2f}%   MaxErr: {max_error(y_te,pred):.2f}h")
print(f"Within +/-15min: {(np.abs(err)<=0.25).mean():.1%}  +/-30min: {(np.abs(err)<=0.5).mean():.1%}  +/-1hr: {(np.abs(err)<=1.0).mean():.1%}")

print("\n---------------- Head-to-head (test) ----------------")
print(f"{'Model':<28s} {'R2':>8s} {'MAE(h)':>9s} {'RMSE(h)':>9s} {'MaxErr(h)':>10s}")
print(f"{'XGBoost top-5':<28s} {r2:>8.4f} {mae:>9.3f} {rmse:>9.3f} {max_error(y_te,pred):>10.2f}")
print(f"{'XGBoost all-59':<28s} {0.9731:>8.4f} {1.714:>9.3f} {2.336:>9.3f} {12.93:>10.2f}")
print(f"{'Linear all-59':<28s} {0.9740:>8.4f} {1.698:>9.3f} {2.300:>9.3f} {10.90:>10.2f}")
mph = (df.loc[tr_mask,"typical_distance_miles"] / y_tr).mean()
dp = df.loc[test_mask,"typical_distance_miles"].values / mph
print(f"{'distance / train-mph':<28s} {r2_score(y_te,dp):>8.4f} {mean_absolute_error(y_te,dp):>9.3f} {np.sqrt(mean_squared_error(y_te,dp)):>9.3f} {max_error(y_te,dp):>10.2f}")

print("\nGain share (top-5 model):")
sk_imp = model.feature_importances_
imp = pd.DataFrame({"feature": TOP5, "gain": sk_imp}).sort_values("gain", ascending=False)
imp["share"] = imp["gain"] / imp["gain"].sum()
print(imp.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
