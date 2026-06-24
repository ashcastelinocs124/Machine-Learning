"""
Build an event-grain feature table for predicting detention_minutes.

Target : detention_minutes  (from delivery_events)
Grain  : one row per delivery event (event_id) -- BOTH Pickup and Delivery (170,820 rows)

Leak-safe: only pre-event-knowable features kept. Post-event columns
(actual_datetime, on_time_flag) are dropped. Detention is the outcome of the
event, so we predict it from what's known at dispatch/schedule time:
scheduled time, facility, route, driver, truck, customer, load, equipment.
"""

from pathlib import Path
import numpy as np
import pandas as pd

DATA = Path(__file__).parent
OUT = DATA / "features_detention.csv"

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

# rename overlapping columns BEFORE merging
drivers = drivers.rename(columns={"home_terminal": "driver_home_terminal"})
trucks  = trucks.rename(columns={
    "model_year": "truck_model_year", "acquisition_date": "truck_acquisition_date",
    "status": "truck_status", "home_terminal": "truck_home_terminal"})
trailers = trailers.rename(columns={
    "model_year": "trailer_model_year", "acquisition_date": "trailer_acquisition_date",
    "status": "trailer_status", "vin": "trailer_vin"})

# ---------------------------------------------------------------- base grain = ALL events (pickup + delivery)
ev = events.copy()
ev["scheduled_datetime"] = pd.to_datetime(ev["scheduled_datetime"])
# rename event location cols to avoid collision with facilities
ev = ev.rename(columns={"location_city":"event_city","location_state":"event_state"})

# ---------------------------------------------------------------- joins
# event -> trip (driver/truck/trailer/dispatch_date/load)
trip_cols = ["trip_id","driver_id","truck_id","trailer_id","dispatch_date","actual_distance_miles"]
ev = ev.merge(trips[trip_cols], on="trip_id", how="left")
ev["dispatch_date"] = pd.to_datetime(ev["dispatch_date"])

# trip -> load (customer/route + load features)
load_cols = ["load_id","customer_id","route_id","load_date","load_type","weight_lbs",
             "pieces","revenue","fuel_surcharge","accessorial_charges","booking_type"]
ev = ev.merge(loads[load_cols], on="load_id", how="left")
ev["load_date"] = pd.to_datetime(ev["load_date"])

# dimension tables
ev = ev.merge(routes, on="route_id", how="left")
ev = ev.merge(customers, on="customer_id", how="left")
ev = ev.merge(drivers,  on="driver_id", how="left")
ev = ev.merge(trucks,   on="truck_id", how="left")
ev = ev.merge(trailers, on="trailer_id", how="left")
ev = ev.merge(facilities.rename(columns={
        "facility_name":"delv_facility_name","facility_type":"delv_facility_type",
        "city":"delv_city","state":"delv_state","latitude":"delv_latitude",
        "longitude":"delv_longitude","dock_doors":"delv_dock_doors",
        "operating_hours":"delv_operating_hours"}),
      on="facility_id", how="left")

# ---------------------------------------------------------------- date parsing
ev["hire_date"]               = pd.to_datetime(ev["hire_date"])
ev["date_of_birth"]           = pd.to_datetime(ev["date_of_birth"])
ev["contract_start_date"]     = pd.to_datetime(ev["contract_start_date"])
ev["truck_acquisition_date"]  = pd.to_datetime(ev["truck_acquisition_date"])
ev["trailer_acquisition_date"]= pd.to_datetime(ev["trailer_acquisition_date"])

# ---------------------------------------------------------------- time features (KEY - hour has real signal)
ev["scheduled_hour"]       = ev["scheduled_datetime"].dt.hour
ev["scheduled_dayofweek"]  = ev["scheduled_datetime"].dt.dayofweek
ev["scheduled_month"]      = ev["scheduled_datetime"].dt.month
ev["scheduled_is_weekend"] = (ev["scheduled_dayofweek"] >= 5).astype(int)
ev["is_overnight"]         = ((ev["scheduled_hour"] < 6) | (ev["scheduled_hour"] >= 19)).astype(int)
ev["dispatch_dayofweek"]   = ev["dispatch_date"].dt.dayofweek
ev["dispatch_month"]       = ev["dispatch_date"].dt.month
ev["days_dispatch_to_scheduled"] = (ev["scheduled_datetime"].dt.normalize() - ev["dispatch_date"]).dt.days
ev["days_booking_to_dispatch"]   = (ev["dispatch_date"] - ev["load_date"]).dt.days

# ---------------------------------------------------------------- derived entity features
ev["driver_age_years"]     = (ev["dispatch_date"] - ev["date_of_birth"]).dt.days / 365.25
ev["driver_tenure_days"]   = (ev["dispatch_date"] - ev["hire_date"]).dt.days
ev["customer_tenure_days"] = (ev["dispatch_date"] - ev["contract_start_date"]).dt.days
ev["truck_age_years"]      = ev["dispatch_date"].dt.year - ev["truck_model_year"]
ev["trailer_age_years"]    = ev["dispatch_date"].dt.year - ev["trailer_model_year"]
ev["revenue_per_mile"]     = ev["revenue"] / ev["typical_distance_miles"]
ev["weight_per_piece"]     = ev["weight_lbs"] / ev["pieces"]

# ---------------------------------------------------------------- historical aggregates (leak-safe: strictly-prior by dispatch_date)
# prior detention at this facility / route / customer (the most relevant history for detention)
def event_prior(df, key, out_prefix):
    s = df.sort_values([key, "dispatch_date", "event_id"]).copy()
    cnt = s.groupby(key).cumcount()
    csum = s.groupby(key)["detention_minutes"].transform(lambda x: x.cumsum().shift(1))
    s[out_prefix + "_past_events"]   = cnt
    s[out_prefix + "_past_detention"] = csum / cnt
    return s[[out_prefix + "_past_events", out_prefix + "_past_detention"]]

for key, pref in [("facility_id","facility"), ("route_id","route"),
                  ("customer_id","customer"), ("driver_id","driver")]:
    ev = ev.join(event_prior(ev, key, pref))

# restore stable order
ev = ev.sort_values("event_id").reset_index(drop=True)

# ---------------------------------------------------------------- drop leakage / PII / raw ids
leakage = ["actual_datetime", "on_time_flag",         # post-event
           "actual_distance_miles", "actual_duration_hours", "fuel_gallons_used",
           "average_mpg", "idle_time_hours", "trip_status", "load_status"]
pii_junk = ["first_name","last_name","license_number","date_of_birth","termination_date",
            "vin","trailer_vin","unit_number","trailer_number","fuel_card_number"]
raw_ids = ["load_id","trip_id","trailer_id"]   # keep event_id + driver/truck/customer/route/facility
drop_cols = [c for c in leakage + pii_junk + raw_ids if c in ev.columns]
ev = ev.drop(columns=drop_cols)

# ---------------------------------------------------------------- save & report
ev.to_csv(OUT, index=False)
print("Saved:", OUT)
print("Rows:", len(ev), " Cols:", ev.shape[1])
print("\nTarget (detention_minutes):")
print(ev["detention_minutes"].describe().round(2).to_string())
print("\nEvent type mix:")
print(ev["event_type"].value_counts().to_string())
print("\nColumn groups:")
target = ["detention_minutes"]
ids    = ["event_id","driver_id","truck_id","customer_id","route_id","facility_id"]
times  = [c for c in ev.columns if c.startswith(("scheduled_","dispatch_","days_","is_"))]
hist   = [c for c in ev.columns if "_past_" in c]
others = [c for c in ev.columns if c not in ids+target+times+hist]
print(f"  identifiers : {ids}")
print(f"  target      : {target}")
print(f"  time feats  : {times}")
print(f"  historical  : {hist}")
print(f"  other feats : {others}")
print("\nNull % (top 10):")
print((ev.isnull().mean()*100).round(2).sort_values(ascending=False).head(10).to_string())
