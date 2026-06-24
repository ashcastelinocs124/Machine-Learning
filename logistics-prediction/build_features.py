"""
Build a feature table for predicting on-time delivery.

Target  : on_time_flag  (from delivery_events where event_type == 'Delivery')
Grain   : one row per delivery event (event_id)

All features are knowable at dispatch time. Post-trip / post-event columns
(actual_datetime, detention_minutes, trip performance, statuses) are dropped.
Historical aggregates use only records with dispatch_date STRICTLY BEFORE the
current event's dispatch_date (cumsum().shift(1)) to prevent target leakage.
"""

from pathlib import Path
import numpy as np
import pandas as pd

DATA = Path(__file__).parent
OUT = DATA / "features.csv"

# ---------------------------------------------------------------- imports
drivers   = pd.read_csv(DATA / "drivers.csv")
trucks    = pd.read_csv(DATA / "trucks.csv")
trailers  = pd.read_csv(DATA / "trailers.csv")
customers = pd.read_csv(DATA / "customers.csv")
facilities= pd.read_csv(DATA / "facilities.csv")
routes    = pd.read_csv(DATA / "routes.csv")
loads     = pd.read_csv(DATA / "loads.csv")
trips     = pd.read_csv(DATA / "trips.csv")
events    = pd.read_csv(DATA / "delivery_events.csv")

# rename overlapping columns BEFORE merging to avoid suffix collisions
drivers = drivers.rename(columns={
    "home_terminal": "driver_home_terminal",
})
trucks = trucks.rename(columns={
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

# ---------------------------------------------------------------- base grain
delv = events[events["event_type"] == "Delivery"].copy()
delv["scheduled_datetime"] = pd.to_datetime(delv["scheduled_datetime"])
delv["actual_datetime"]    = pd.to_datetime(delv["actual_datetime"])  # dropped later
# events already has load_id; rename its location cols to avoid facility-merge collision
delv = delv.rename(columns={"location_city":"event_city","location_state":"event_state"})

# ---------------------------------------------------------------- joins
# delivery -> trip (driver/truck/trailer/dispatch_date); events already has load_id
trip_cols = ["trip_id","driver_id","truck_id","trailer_id","dispatch_date"]
delv = delv.merge(trips[trip_cols], on="trip_id", how="left")
delv["dispatch_date"] = pd.to_datetime(delv["dispatch_date"])

# trip -> load (customer/route + load features)
load_cols = ["load_id","customer_id","route_id","load_date","load_type","weight_lbs",
             "pieces","revenue","fuel_surcharge","accessorial_charges","booking_type"]
delv = delv.merge(loads[load_cols], on="load_id", how="left")
delv["load_date"] = pd.to_datetime(delv["load_date"])

# dimension tables
delv = delv.merge(routes, on="route_id", how="left")
delv = delv.merge(customers, on="customer_id", how="left")
delv = delv.merge(drivers,  on="driver_id", how="left")
delv = delv.merge(trucks,   on="truck_id", how="left")
delv = delv.merge(trailers, on="trailer_id", how="left")
delv = delv.merge(facilities.rename(columns={
        "facility_name":"delv_facility_name","facility_type":"delv_facility_type",
        "city":"delv_city","state":"delv_state","latitude":"delv_latitude",
        "longitude":"delv_longitude","dock_doors":"delv_dock_doors",
        "operating_hours":"delv_operating_hours"}),
      on="facility_id", how="left")

# ---------------------------------------------------------------- date parsing
delv["hire_date"]             = pd.to_datetime(delv["hire_date"])
delv["date_of_birth"]         = pd.to_datetime(delv["date_of_birth"])
delv["contract_start_date"]   = pd.to_datetime(delv["contract_start_date"])
delv["truck_acquisition_date"]    = pd.to_datetime(delv["truck_acquisition_date"])
delv["trailer_acquisition_date"]  = pd.to_datetime(delv["trailer_acquisition_date"])

# ---------------------------------------------------------------- time features
delv["scheduled_date"]      = delv["scheduled_datetime"].dt.normalize()  # datetime at midnight
delv["scheduled_hour"]      = delv["scheduled_datetime"].dt.hour
delv["scheduled_dayofweek"] = delv["scheduled_datetime"].dt.dayofweek
delv["scheduled_month"]     = delv["scheduled_datetime"].dt.month
delv["scheduled_is_weekend"]= (delv["scheduled_dayofweek"] >= 5).astype(int)

delv["dispatch_dayofweek"] = delv["dispatch_date"].dt.dayofweek
delv["dispatch_month"]     = delv["dispatch_date"].dt.month

delv["days_dispatch_to_scheduled"] = (delv["scheduled_date"]
        - delv["dispatch_date"]).dt.days
delv["days_booking_to_dispatch"]   = (delv["dispatch_date"]
        - delv["load_date"]).dt.days

# buffer (positive = more days than typical; negative = tight schedule)
delv["scheduled_transit_gap_days"] = delv["days_dispatch_to_scheduled"] - delv["typical_transit_days"]

# ---------------------------------------------------------------- derived entity features
delv["driver_age_years"]     = (delv["dispatch_date"] - delv["date_of_birth"]).dt.days / 365.25
delv["driver_tenure_days"]   = (delv["dispatch_date"] - delv["hire_date"]).dt.days
delv["customer_tenure_days"] = (delv["dispatch_date"] - delv["contract_start_date"]).dt.days
delv["truck_age_years"]      = delv["dispatch_date"].dt.year - delv["truck_model_year"]
delv["trailer_age_years"]    = delv["dispatch_date"].dt.year - delv["trailer_model_year"]
delv["truck_age_days"]       = (delv["dispatch_date"] - delv["truck_acquisition_date"]).dt.days
delv["trailer_age_days"]     = (delv["dispatch_date"] - delv["trailer_acquisition_date"]).dt.days

delv["revenue_per_mile"]    = delv["revenue"] / delv["typical_distance_miles"]
delv["weight_per_piece"]    = delv["weight_lbs"] / delv["pieces"]

# ---------------------------------------------------------------- historical aggregates
# (a) trip-level prior MPG by truck / driver (strictly prior by dispatch_date)
t = trips[["trip_id","truck_id","driver_id","dispatch_date",
           "actual_distance_miles","fuel_gallons_used"]].copy()
t["dispatch_date"] = pd.to_datetime(t["dispatch_date"])
t["trip_mpg"] = t["actual_distance_miles"] / t["fuel_gallons_used"]

def prior_mean(df, key, val):
    s = df.sort_values([key, "dispatch_date", "trip_id"]).copy()
    cnt = s.groupby(key).cumcount()
    sm  = s.groupby(key)[val].transform(lambda x: x.cumsum().shift(1))
    s["_cnt"], s["_sm"] = cnt, sm
    return s

for key, pref in [("truck_id","truck"), ("driver_id","driver")]:
    s = prior_mean(t, key, "trip_mpg")
    t[f"{pref}_prior_trips"] = s["_cnt"]
    t[f"{pref}_prior_mpg"]   = s["_sm"] / s["_cnt"]

delv = delv.merge(t[["trip_id","truck_prior_trips","truck_prior_mpg",
                     "driver_prior_trips","driver_prior_mpg"]], on="trip_id", how="left")

# (b) delivery-event-level prior on-time rate, grouped by driver / route / customer / truck
def event_prior(df, key, out_prefix):
    s = df.sort_values([key, "dispatch_date", "event_id"]).copy()
    cnt = s.groupby(key).cumcount()
    sm  = s.groupby(key)["on_time_flag"].transform(lambda x: x.cumsum().shift(1))
    s[out_prefix + "_past_deliveries"]   = cnt
    s[out_prefix + "_past_on_time_rate"] = sm / cnt
    return s[[out_prefix + "_past_deliveries", out_prefix + "_past_on_time_rate"]]

for key, pref in [("driver_id","driver"), ("route_id","route"),
                  ("customer_id","customer"), ("truck_id","truck_ev")]:
    delv = delv.join(event_prior(delv, key, pref))

# restore a stable row order
delv = delv.sort_values("event_id").reset_index(drop=True)

# ---------------------------------------------------------------- drop leakage / PII / raw ids
leakage = [
    "actual_datetime", "detention_minutes", "event_type",
    "actual_distance_miles", "actual_duration_hours", "fuel_gallons_used",
    "average_mpg", "idle_time_hours", "trip_status", "load_status",
]
pii_junk = [
    "first_name","last_name","license_number","date_of_birth","termination_date",
    "vin","trailer_vin","unit_number","trailer_number","fuel_card_number",
]
raw_ids = ["load_id","trip_id","trailer_id"]   # keep event_id, driver/truck/customer/route/facility ids
drop_cols = [c for c in leakage + pii_junk + raw_ids if c in delv.columns]
delv = delv.drop(columns=drop_cols)

# ---------------------------------------------------------------- save & report
delv.to_csv(OUT, index=False)
print("Saved:", OUT)
print("Rows:", len(delv), " Cols:", delv.shape[1])
print("\nTarget (on_time_flag):")
print(delv["on_time_flag"].value_counts(dropna=False).to_string())
print("\nColumn groups:")
ids    = ["event_id","driver_id","truck_id","customer_id","route_id","facility_id"]
target = ["on_time_flag"]
times  = [c for c in delv.columns if c.startswith(("scheduled_","dispatch_","days_"))
          or c == "scheduled_transit_gap_days"]
hist    = [c for c in delv.columns if "_past_" in c or "_prior_" in c]
others  = [c for c in delv.columns if c not in ids+target+times+hist]
print("  identifiers :", ids)
print("  target      :", target)
print("  time feats  :", times)
print("  historical  :", hist)
print("  other feats :", others)
print("\nNull % (top 15):")
print((delv.isnull().mean()*100).round(2).sort_values(ascending=False).head(15).to_string())
