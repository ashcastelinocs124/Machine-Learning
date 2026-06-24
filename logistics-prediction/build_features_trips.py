"""
Build a trip-grain feature table for predicting actual_duration_hours.

Target : actual_duration_hours  (from trips)
Grain  : one row per trip (trip_id)

Leak-safe design: only dispatch-time-knowable features are kept.
Post-trip columns (actual_distance_miles, fuel_gallons_used, average_mpg,
idle_time_hours, trip_status) are dropped. typical_distance_miles from routes
is used instead of actual_distance_miles (r=0.998 with actual).
Historical aggregates use strictly-prior dispatch dates (cumsum.shift(1)).
"""

from pathlib import Path
import numpy as np
import pandas as pd

DATA = Path(__file__).parent
OUT = DATA / "features_trips.csv"

# ---------------------------------------------------------------- imports
drivers   = pd.read_csv(DATA / "drivers.csv")
trucks    = pd.read_csv(DATA / "trucks.csv")
trailers  = pd.read_csv(DATA / "trailers.csv")
customers = pd.read_csv(DATA / "customers.csv")
routes    = pd.read_csv(DATA / "routes.csv")
loads     = pd.read_csv(DATA / "loads.csv")
trips     = pd.read_csv(DATA / "trips.csv")

# rename overlapping columns BEFORE merging
drivers = drivers.rename(columns={"home_terminal": "driver_home_terminal"})
trucks  = trucks.rename(columns={
    "model_year": "truck_model_year",
    "acquisition_date": "truck_acquisition_date",
    "status": "truck_status",
    "home_terminal": "truck_home_terminal",
})
trailers = trailers.rename(columns={
    "model_year": "trailer_model_year",
    "acquisition_date": "trailer_acquisition_date",
    "status": "trailer_status",
    "vin": "trailer_vin",
})

# ---------------------------------------------------------------- base grain = trips
t = trips.copy()
t["dispatch_date"] = pd.to_datetime(t["dispatch_date"])
# keep target; drop post-trip leakage
t = t.drop(columns=["actual_distance_miles", "fuel_gallons_used",
                    "average_mpg", "idle_time_hours", "trip_status"])

# ---------------------------------------------------------------- joins
# trip -> load (customer/route + load features)
load_cols = ["load_id","customer_id","route_id","load_date","load_type","weight_lbs",
             "pieces","revenue","fuel_surcharge","accessorial_charges","booking_type"]
t = t.merge(loads[load_cols], on="load_id", how="left")
t["load_date"] = pd.to_datetime(t["load_date"])

# dimension tables
t = t.merge(routes, on="route_id", how="left")
t = t.merge(customers, on="customer_id", how="left")
t = t.merge(drivers,  on="driver_id", how="left")
t = t.merge(trucks,   on="truck_id", how="left")
t = t.merge(trailers, on="trailer_id", how="left")

# ---------------------------------------------------------------- date parsing
t["hire_date"]               = pd.to_datetime(t["hire_date"])
t["date_of_birth"]           = pd.to_datetime(t["date_of_birth"])
t["contract_start_date"]     = pd.to_datetime(t["contract_start_date"])
t["truck_acquisition_date"]  = pd.to_datetime(t["truck_acquisition_date"])
t["trailer_acquisition_date"]= pd.to_datetime(t["trailer_acquisition_date"])

# ---------------------------------------------------------------- time features
t["dispatch_dayofweek"] = t["dispatch_date"].dt.dayofweek
t["dispatch_month"]     = t["dispatch_date"].dt.month
t["dispatch_year"]      = t["dispatch_date"].dt.year
t["dispatch_is_weekend"]= (t["dispatch_dayofweek"] >= 5).astype(int)
t["days_booking_to_dispatch"] = (t["dispatch_date"] - t["load_date"]).dt.days

# ---------------------------------------------------------------- derived entity features
t["driver_age_years"]     = (t["dispatch_date"] - t["date_of_birth"]).dt.days / 365.25
t["driver_tenure_days"]   = (t["dispatch_date"] - t["hire_date"]).dt.days
t["customer_tenure_days"] = (t["dispatch_date"] - t["contract_start_date"]).dt.days
t["truck_age_years"]      = t["dispatch_date"].dt.year - t["truck_model_year"]
t["trailer_age_years"]    = t["dispatch_date"].dt.year - t["trailer_model_year"]
t["truck_age_days"]       = (t["dispatch_date"] - t["truck_acquisition_date"]).dt.days
t["trailer_age_days"]     = (t["dispatch_date"] - t["trailer_acquisition_date"]).dt.days
t["revenue_per_mile"]     = t["revenue"] / t["typical_distance_miles"]
t["weight_per_piece"]     = t["weight_lbs"] / t["pieces"]

# ---------------------------------------------------------------- historical aggregates
# prior trip DURATION (the target) and prior MPG by truck / driver.
# strictly prior by dispatch_date -> leak-safe.
raw = trips[["trip_id","truck_id","driver_id","dispatch_date",
             "actual_duration_hours","actual_distance_miles","fuel_gallons_used"]].copy()
raw["dispatch_date"] = pd.to_datetime(raw["dispatch_date"])
raw["trip_mpg"] = raw["actual_distance_miles"] / raw["fuel_gallons_used"]

def prior_stats(df, key, val):
    s = df.sort_values([key, "dispatch_date", "trip_id"]).copy()
    cnt = s.groupby(key).cumcount()
    csum = s.groupby(key)[val].transform(lambda x: x.cumsum().shift(1))
    s["_cnt"], s["_mean"] = cnt, csum / cnt
    return s

for key, pref in [("truck_id","truck"), ("driver_id","driver")]:
    s_dur = prior_stats(raw, key, "actual_duration_hours")
    raw[f"{pref}_prior_trips"]      = s_dur["_cnt"]
    raw[f"{pref}_prior_duration"]   = s_dur["_mean"]
    s_mpg = prior_stats(raw, key, "trip_mpg")
    raw[f"{pref}_prior_mpg"]        = s_mpg["_mean"]

t = t.merge(raw[["trip_id","truck_prior_trips","truck_prior_duration","truck_prior_mpg",
                 "driver_prior_trips","driver_prior_duration","driver_prior_mpg"]],
            on="trip_id", how="left")

# restore stable order
t = t.sort_values("trip_id").reset_index(drop=True)

# ---------------------------------------------------------------- drop leakage / PII / raw ids
leakage = ["actual_distance_miles", "fuel_gallons_used", "average_mpg",
           "idle_time_hours", "trip_status"]
pii_junk = ["first_name","last_name","license_number","date_of_birth","termination_date",
            "vin","trailer_vin","unit_number","trailer_number","fuel_card_number"]
raw_ids = ["load_id","trailer_id"]   # keep trip_id + driver/truck/customer/route ids
drop_cols = [c for c in leakage + pii_junk + raw_ids if c in t.columns]
t = t.drop(columns=drop_cols)

# ---------------------------------------------------------------- save & report
t.to_csv(OUT, index=False)
print("Saved:", OUT)
print("Rows:", len(t), " Cols:", t.shape[1])
print("\nTarget (actual_duration_hours):")
print(t["actual_duration_hours"].describe().round(2).to_string())
print("\nColumn groups:")
target = ["actual_duration_hours"]
ids    = ["trip_id","driver_id","truck_id","customer_id","route_id"]
times  = ["dispatch_date","dispatch_dayofweek","dispatch_month","dispatch_year",
          "dispatch_is_weekend","days_booking_to_dispatch","load_date"]
hist   = [c for c in t.columns if "_prior_" in c]
others = [c for c in t.columns if c not in ids+target+times+hist]
print("  identifiers :", ids)
print("  target      :", target)
print("  time feats  :", times)
print("  historical  :", hist)
print("  other feats :", others)
print("\nNull % (top 10):")
print((t.isnull().mean()*100).round(2).sort_values(ascending=False).head(10).to_string())
