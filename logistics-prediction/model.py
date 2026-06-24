"""
Logistic regression for on-time delivery prediction.

Split : chronological
  train = deliveries scheduled 2022-01-01 .. 2024-06-30
  test  = deliveries scheduled 2024-07-01 .. 2024-12-31
This evaluates the model on the most recent 6 months, the way it would be
used in production. Historical aggregate features are already leak-safe
(strictly-prior dispatch dates), so the time split keeps them valid.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report, roc_curve)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA = Path(__file__).parent
df = pd.read_csv(DATA / "features.csv")
df["scheduled_datetime"] = pd.to_datetime(df["scheduled_datetime"])
df["dispatch_date"]      = pd.to_datetime(df["dispatch_date"])

# ---------------------------------------------------------------- target + split
y = df["on_time_flag"].astype(int)            # 1 = on time, 0 = late
cutoff = pd.Timestamp("2024-07-01")
train_mask = df["scheduled_datetime"] < cutoff
test_mask  = ~train_mask

# columns to never use as features
drop = ["on_time_flag", "event_id", "scheduled_datetime", "dispatch_date",
        "scheduled_date", "load_date", "hire_date", "contract_start_date",
        "truck_acquisition_date", "trailer_acquisition_date"]
# raw entity ids -> treat as categorical (high-cardinality ones dropped to keep
# the linear model interpretable and fast)
id_cols = ["driver_id", "truck_id", "customer_id", "route_id", "facility_id"]
feature_df = df.drop(columns=drop)

X_train, y_train = feature_df[train_mask], y[train_mask]
X_test,  y_test  = feature_df[test_mask],  y[test_mask]
print(f"Train: {len(X_train):,} rows  ({y_train.mean():.3%} on-time)")
print(f"Test : {len(X_test):,} rows  ({y_test.mean():.3%} on-time)")
print(f"Test period: {df.loc[test_mask,'scheduled_datetime'].min().date()} -> "
      f"{df.loc[test_mask,'scheduled_datetime'].max().date()}")

# ---------------------------------------------------------------- column typing
# drop very-high-cardinality ids from categoricals (keep route_id/facility_id:
# small counts). Drop driver/truck/customer ids.
high_card_drop = ["driver_id", "truck_id", "customer_id"]
X_train = X_train.drop(columns=high_card_drop)
X_test  = X_test.drop(columns=high_card_drop)

num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X_train.select_dtypes(exclude=[np.number]).columns.tolist()
print(f"\nNumeric cols ({len(num_cols)}): {num_cols}")
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
    ("clf", LogisticRegression(
        class_weight="balanced",     # ~45/55 split
        max_iter=2000,
        solver="lbfgs",
        n_jobs=-1,
    )),
])

# ---------------------------------------------------------------- fit + evaluate
model.fit(X_train, y_train)
proba = model.predict_proba(X_test)[:, 1]
pred  = (proba >= 0.5).astype(int)

print("\n================ TEST PERFORMANCE ================")
print(f"Accuracy    : {accuracy_score(y_test, pred):.4f}")
print(f"Precision   : {precision_score(y_test, pred):.4f}  (of predicted on-time, share truly on-time)")
print(f"Recall      : {recall_score(y_test, pred):.4f}  (of true on-time, share caught)")
print(f"F1 score    : {f1_score(y_test, pred):.4f}")
print(f"ROC-AUC     : {roc_auc_score(y_test, proba):.4f}")
tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
print(f"\nConfusion matrix (rows=actual [late,on-time], cols=pred [late,on-time]):")
print(f"  late     : TN={tn:,}  FP={fp:,}")
print(f"  on-time  : FN={fn:,}  TP={tp:,}")

# baseline = always predict majority class
maj = y_train.value_counts().idxmax()
base_acc = (y_test == maj).mean()
print(f"\nBaseline (always predict '{maj}'): {base_acc:.4f}")
print(f"Test prevalence (on-time): {y_test.mean():.4f}")

print("\nClassification report:")
print(classification_report(y_test, pred, target_names=["late","on-time"], digits=4))

# ---------------------------------------------------------------- top coefficients
# map encoded feature names back, show top positive/negative drivers
feat_names = (model.named_steps["pre"]
              .get_feature_names_out())
coefs = model.named_steps["clf"].coef_[0]
coef_df = (pd.DataFrame({"feature": feat_names, "coef": coefs})
           .assign(abs=lambda d: d.coef.abs())
           .sort_values("abs", ascending=False))
print("\nTop 15 features by |coef| (positive coef -> more likely on-time):")
print(coef_df.head(15).to_string(index=False, float_format=lambda x: f"{x:+.3f}"))

# ---------------------------------------------------------------- ROC operating point
# find threshold that maximizes F1 on train (reported for reference)
train_proba = model.predict_proba(X_train)[:, 1]
best_thr, best_f1 = 0.5, 0
for thr in np.linspace(0.3, 0.7, 41):
    f1 = f1_score(y_train, (train_proba >= thr).astype(int))
    if f1 > best_f1:
        best_thr, best_f1 = thr, f1
pred_opt = (proba >= best_thr).astype(int)
print(f"\nWith train-tuned threshold={best_thr:.2f} (best train F1={best_f1:.4f}):")
print(f"  Test F1={f1_score(y_test, pred_opt):.4f}  "
      f"Precision={precision_score(y_test, pred_opt):.4f}  "
      f"Recall={recall_score(y_test, pred_opt):.4f}")
