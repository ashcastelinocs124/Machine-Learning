"""
Initial EDA + business KPIs + profitability deep-dive for the logistics dataset.

Sections:
  1. Data profile          (row counts, date ranges, key nulls, dupes)
  2. Seasonal patterns     (load volume, rates by month)
  3. Customer analysis     (revenue by customer, service levels)
  4. Driver performance    (MPG, revenue/mile, on-time, idle)
  5. Route profitability   (revenue vs fuel+maint+safety by lane)
  6. Fleet utilization     (miles/truck, revenue/asset, utilization rate)
  7. Fuel efficiency       (MPG trends, fuel cost by route)
  8. Maintenance           (cost per mile, downtime impact)
  9. Safety                (incident rates, preventable accidents)
 10. Profitability deep-dive (margin by route / customer / driver / month)

Profitability model:
  total_revenue = loads.revenue + fuel_surcharge + accessorial_charges  (billed to customer)
  fuel_cost     = trips.fuel_gallons_used x mean(fuel_purchases.price_per_gallon)
                  (NOT sum(fuel_purchases.total_cost) -- that table's gallons are
                  unscaled to trip distance, a data-generation bug)
  safety_cost   = vehicle_damage + cargo_damage + claim_amount per trip
  maintenance   = per-truck (allocated to trips by miles for trip-level margin)
  net_margin (truck)   = total_revenue - fuel_cost - safety_cost - allocated maintenance
"""

from pathlib import Path
import numpy as np
import pandas as pd
pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 30)

DATA = Path(__file__).parent
def load(f, **k): return pd.read_csv(DATA / f, **k)

drivers   = load("drivers.csv")
trucks    = load("trucks.csv")
trailers  = load("trailers.csv")
customers = load("customers.csv")
facilities= load("facilities.csv")
routes    = load("routes.csv")
loads     = load("loads.csv")
trips     = load("trips.csv")
fuel      = load("fuel_purchases.csv")
maint     = load("maintenance_records.csv")
events    = load("delivery_events.csv")
safety    = load("safety_incidents.csv")
dmm       = load("driver_monthly_metrics.csv")
tum       = load("truck_utilization_metrics.csv")

def hr(title): print(f"\n{'='*78}\n{title}\n{'='*78}")

# ================================================================ 1. DATA PROFILE
hr("1. DATA PROFILE")
tables = [("drivers",drivers),("trucks",trucks),("trailers",trailers),
          ("customers",customers),("facilities",facilities),("routes",routes),
          ("loads",loads),("trips",trips),("fuel_purchases",fuel),
          ("maintenance_records",maint),("delivery_events",events),
          ("safety_incidents",safety),("driver_monthly_metrics",dmm),
          ("truck_utilization_metrics",tum)]
total = 0
print(f"{'table':<28s} {'rows':>9s} {'cols':>5s} {'date col':<22s} {'min':>12s} {'max':>12s}")
for name, df in tables:
    total += len(df)
    datecol = next((c for c in df.columns if "date" in c or "datetime" in c), None)
    dmin = dmax = ""
    if datecol:
        s = pd.to_datetime(df[datecol], errors="coerce")
        dmin, dmax = str(s.min().date()), str(s.max().date())
    print(f"{name:<28s} {len(df):>9,} {df.shape[1]:>5} {datecol or '-':<22s} {dmin:>12s} {dmax:>12s}")
print(f"\nTotal records across all tables: {total:,}")
# nulls + dupes on key tables
print("\nKey nulls / duplicate PKs:")
for name, df, pk in [("loads",loads,"load_id"),("trips",trips,"trip_id"),
                     ("delivery_events",events,"event_id"),("fuel_purchases",fuel,"fuel_purchase_id")]:
    dup = df[pk].duplicated().sum()
    print(f"  {name:<20s} dup PKs={dup}  null% top3={((df.isnull().mean()*100).sort_values(ascending=False).head(3)).round(2).to_dict()}")

# ================================================================ 2. SEASONAL PATTERNS
hr("2. SEASONAL PATTERNS (loads by month, 2022-2024)")
l = loads.copy(); l["load_date"] = pd.to_datetime(l["load_date"])
l["ym"] = l["load_date"].dt.to_period("M")
seas = l.groupby("ym").agg(loads=("load_id","count"),
                           avg_revenue=("revenue","mean"),
                           avg_weight=("weight_lbs","mean")).round(2)
print(seas.to_string())
print("\nMonth-of-year (pooled 3yr):")
l["mo"] = l["load_date"].dt.month
print(l.groupby("mo").agg(loads=("load_id","count"), avg_rev=("revenue","mean"),
                          avg_weight=("weight_lbs","mean")).round(2).to_string())
# rate per mile by month
lm = l.merge(routes[["route_id","typical_distance_miles"]], on="route_id")
lm["rpm"] = lm["revenue"] / lm["typical_distance_miles"]
print("\nAvg rate/mile by month-of-year:")
print(lm.groupby(lm["load_date"].dt.month)["rpm"].mean().round(3).to_string())

# ================================================================ 3. CUSTOMER ANALYSIS
hr("3. CUSTOMER ANALYSIS (revenue, loads, service)")
lc = loads.merge(customers[["customer_id","customer_name","customer_type"]], on="customer_id")
cust = lc.groupby(["customer_name","customer_type"]).agg(
        loads=("load_id","count"),
        total_revenue=("revenue","sum"),
        avg_revenue=("revenue","mean"),
        avg_weight=("weight_lbs","mean")).round(2).sort_values("total_revenue", ascending=False)
print("Top 10 customers by total revenue:")
print(cust.head(10).to_string())
print(f"\nCustomers: {customers['customer_id'].nunique()}  | by type:")
print(customers["customer_type"].value_counts().to_string())
print(f"\nAccount status: {customers['account_status'].value_counts().to_dict()}")

# ================================================================ 4. DRIVER PERFORMANCE
hr("4. DRIVER PERFORMANCE (monthly metrics, top/bottom)")
print("Driver monthly metrics summary:")
print(dmm[["trips_completed","total_miles","total_revenue","average_mpg",
          "on_time_delivery_rate","average_idle_hours"]].describe().round(2).to_string())
# per-driver aggregate
drv_agg = dmm.groupby("driver_id").agg(
        trips=("trips_completed","sum"),
        miles=("total_miles","sum"),
        revenue=("total_revenue","sum"),
        avg_mpg=("average_mpg","mean"),
        on_time=("on_time_delivery_rate","mean"),
        idle_h=("average_idle_hours","mean")).round(2)
drv_agg["rev_per_mile"] = (drv_agg["revenue"]/drv_agg["miles"]).round(2)
drv_agg = drv_agg.merge(drivers[["driver_id","years_experience"]], on="driver_id")
print("\nTop 5 drivers by total revenue:")
print(drv_agg.sort_values("revenue", ascending=False).head(5).to_string())
print("\nBottom 5 drivers by total revenue:")
print(drv_agg.sort_values("revenue").head(5).to_string())
print("\nBest 5 by MPG (min 5000 miles):")
print(drv_agg[drv_agg.miles>=5000].sort_values("avg_mpg", ascending=False).head(5).to_string())
print("\nOn-time rate: overall mean {:.1%}  | spread {:.1%}..{:.1%}".format(
      drv_agg["on_time"].mean(), drv_agg["on_time"].min(), drv_agg["on_time"].max()))

# ================================================================ 5. ROUTE PROFITABILITY
hr("5. ROUTE PROFITABILITY (revenue vs fuel+maint+safety by lane)")
# NOTE: fuel_purchases.gallons is unscaled to trip distance (data-gen bug), so
# we use trips.fuel_gallons_used x diesel price instead. Diesel price comes from
# the mean of fuel_purchases.price_per_gallon (the per-gallon price is valid).
DIESEL = fuel["price_per_gallon"].mean()
print(f"Diesel price used: ${DIESEL:.2f}/gal  (mean of fuel_purchases.price_per_gallon)")
print(f"NOTE: fuel cost = trips.fuel_gallons_used x ${DIESEL:.2f}  (NOT fuel_purchases.total_cost, "
      f"which is unscaled to distance)")
# trip-level costs
si_trip = (safety.assign(safety_cost=safety["vehicle_damage_cost"].fillna(0)
                         +safety["cargo_damage_cost"].fillna(0)
                         +safety["claim_amount"].fillna(0))
           .groupby("trip_id")["safety_cost"].sum())
maint_truck = maint.groupby("truck_id")["total_cost"].sum().rename("maint_cost")
truck_miles = trips.groupby("truck_id")["actual_distance_miles"].sum().rename("truck_miles")

t = trips.merge(loads[["load_id","customer_id","route_id","revenue",
                       "fuel_surcharge","accessorial_charges"]], on="load_id")
t = t.merge(routes[["route_id","origin_city","destination_city","typical_distance_miles",
                    "base_rate_per_mile"]], on="route_id")
t["total_revenue"] = t["revenue"] + t["fuel_surcharge"] + t["accessorial_charges"]
t["fuel_cost"] = t["fuel_gallons_used"] * DIESEL          # corrected fuel cost
t = t.merge(si_trip, on="trip_id", how="left")
t[["safety_cost"]] = t[["safety_cost"]].fillna(0)
t["contribution"] = t["total_revenue"] - t["fuel_cost"] - t["safety_cost"]
# allocate maintenance by miles
t = t.merge(maint_truck, on="truck_id", how="left").merge(truck_miles, on="truck_id", how="left")
t["maint_alloc"] = (t["maint_cost"] / t["truck_miles"]) * t["actual_distance_miles"]
t["maint_alloc"] = t["maint_alloc"].fillna(0)
t["net_margin"] = t["contribution"] - t["maint_alloc"]

route_p = t.groupby(["route_id","origin_city","destination_city"]).agg(
        trips=("trip_id","count"),
        total_revenue=("total_revenue","sum"),
        fuel_cost=("fuel_cost","sum"),
        maint_alloc=("maint_alloc","sum"),
        safety_cost=("safety_cost","sum"),
        net_margin=("net_margin","sum"),
        avg_margin_per_trip=("net_margin","mean")).round(0).sort_values("net_margin")
route_p["margin_pct"] = (route_p["net_margin"]/route_p["total_revenue"]*100).round(1)
print("Top 5 routes by net margin:")
print(route_p.tail(5).to_string())
print("\nBottom 5 routes by net margin:")
print(route_p.head(5).to_string())
print(f"\nRoutes: {route_p.shape[0]}  | profitable: {(route_p['net_margin']>0).sum()}  "
      f"| loss-making: {(route_p['net_margin']<=0).sum()}")
print(f"Total net margin: ${route_p['net_margin'].sum():,.0f}  "
      f"| avg margin%: {route_p['margin_pct'].mean():.1f}%")

# ================================================================ 6. FLEET UTILIZATION
hr("6. FLEET UTILIZATION (truck metrics)")
print("Truck utilization metrics summary:")
print(tum[["trips_completed","total_miles","total_revenue","average_mpg",
          "downtime_hours","utilization_rate"]].describe().round(2).to_string())
truck_u = tum.groupby("truck_id").agg(
        trips=("trips_completed","sum"),
        miles=("total_miles","sum"),
        revenue=("total_revenue","sum"),
        maint_events=("maintenance_events","sum"),
        maint_cost=("maintenance_cost","sum"),
        downtime=("downtime_hours","sum"),
        util_rate=("utilization_rate","mean")).round(2)
truck_u["rev_per_mile"] = (truck_u["revenue"]/truck_u["miles"]).round(2)
truck_u = truck_u.merge(trucks[["truck_id","make","truck_model_year" if "truck_model_year" in trucks else "model_year","fuel_type"]], on="truck_id")
print("\nTop 5 trucks by utilization rate:")
print(truck_u.sort_values("util_rate", ascending=False).head(5).to_string())
print("\nMost downtime (bottom 5 by utilization):")
print(truck_u.sort_values("util_rate").head(5).to_string())
print(f"\nFleet: {trucks['truck_id'].nunique()} trucks  | active: "
      f"{(trucks['status']=='Active').sum() if 'status' in trucks else 'n/a'}")
print(f"Trucks with util data: {tum['truck_id'].nunique()}  | avg utilization: {tum['utilization_rate'].mean():.1%}")

# ================================================================ 7. FUEL EFFICIENCY
hr("7. FUEL EFFICIENCY (MPG, fuel cost)")
tr_f = trips.copy()
tr_f["trip_mpg"] = tr_f["actual_distance_miles"] / tr_f["fuel_gallons_used"]
tr_f["dispatch_date"] = pd.to_datetime(tr_f["dispatch_date"])
tr_f["year"] = tr_f["dispatch_date"].dt.year
print("MPG by year:")
print(tr_f.groupby("year")["trip_mpg"].agg(["mean","median","std"]).round(2).to_string())
print("\nMPG by truck make:")
tr_f = tr_f.merge(trucks[["truck_id","make"]], on="truck_id")
print(tr_f.groupby("make")["trip_mpg"].agg(["mean","count"]).round(2).sort_values("mean",ascending=False).to_string())
# fuel cost per route (using corrected trips.fuel_gallons_used x DIESEL, not fuel_purchases)
fc_route = trips.merge(loads[["load_id","route_id"]], on="load_id").merge(
            routes[["route_id","origin_city","destination_city","typical_distance_miles"]], on="route_id")
fc_route = fc_route.groupby(["route_id","origin_city","destination_city"]).agg(
            trips=("trip_id","count"),
            gallons=("fuel_gallons_used","sum"),
            miles=("typical_distance_miles","first")).round(0)
fc_route["fuel_cost"] = (fc_route["gallons"] * DIESEL).round(0)
fc_route["fuel_cost_per_mile"] = (fc_route["fuel_cost"] / (fc_route["miles"] * fc_route["trips"])).round(2)
fc_route["fuel_cost_per_gallon"] = DIESEL
print(f"\nFuel cost per mile by route (using trips.fuel_gallons_used x ${DIESEL:.2f}/gal):")
print(fc_route.sort_values("fuel_cost_per_mile").head(5).to_string())
print(fc_route.sort_values("fuel_cost_per_mile").tail(5).to_string())

# ================================================================ 8. MAINTENANCE
hr("8. MAINTENANCE (cost per mile, downtime, by type)")
print("Maintenance by type:")
print(maint.groupby("maintenance_type").agg(
        records=("maintenance_id","count"),
        avg_total_cost=("total_cost","mean"),
        avg_downtime_h=("downtime_hours","mean"),
        total_cost=("total_cost","sum")).round(2).sort_values("total_cost",ascending=False).to_string())
# cost per mile by truck
mt = maint.groupby("truck_id")["total_cost"].sum().rename("total_maint_cost")
tm = trips.groupby("truck_id")["actual_distance_miles"].sum().rename("miles")
cpm = pd.concat([mt, tm], axis=1).dropna()
cpm["cost_per_mile"] = (cpm["total_maint_cost"]/cpm["miles"]).round(3)
print(f"\nMaintenance cost per mile: mean={cpm['cost_per_mile'].mean():.3f}  "
      f"median={cpm['cost_per_mile'].median():.3f}")
print(f"Total maint cost: ${maint['total_cost'].sum():,.0f}  | total downtime: {maint['downtime_hours'].sum():,.0f}h")
print("\nWorst 5 trucks by maint cost per mile (min 10k miles):")
print(cpm[cpm.miles>=10000].sort_values("cost_per_mile",ascending=False).head(5).round(2).to_string())

# ================================================================ 9. SAFETY
hr("9. SAFETY (incident rates, preventable, by type)")
print(f"Total incidents: {len(safety)}  | preventable: {safety['preventable_flag'].sum()} "
      f"({safety['preventable_flag'].mean():.1%})  | injuries: {safety['injury_flag'].sum()}")
print(f"At-fault: {safety['at_fault_flag'].sum()} ({safety['at_fault_flag'].mean():.1%})")
print(f"Total cost: vehicle ${safety['vehicle_damage_cost'].sum():,.0f}  "
      f"cargo ${safety['cargo_damage_cost'].sum():,.0f}  claims ${safety['claim_amount'].sum():,.0f}")
print("\nIncidents by type:")
print(safety.groupby("incident_type").agg(
        count=("incident_id","count"),
        preventable=("preventable_flag","sum"),
        injuries=("injury_flag","sum"),
        avg_claim=("claim_amount","mean")).round(0).sort_values("count",ascending=False).to_string())
# incident rate per 1000 trips
print(f"\nIncident rate: {len(safety)/len(trips)*1000:.2f} per 1,000 trips")
si_driver = safety.groupby("driver_id").size().rename("incidents")
print("\nDrivers with most incidents:")
print(si_driver.sort_values(ascending=False).head(5).to_string())

# ================================================================ 10. PROFITABILITY DEEP-DIVE
hr("10. PROFITABILITY DEEP-DIVE (margin by route / customer / driver / month)")
print(f"\nFleet-wide P&L:")
print(f"  Total revenue (billed): ${t['total_revenue'].sum():>15,.0f}")
print(f"  Fuel cost:              ${t['fuel_cost'].sum():>15,.0f}  ({t['fuel_cost'].sum()/t['total_revenue'].sum()*100:.1f}% of rev)")
print(f"  Maintenance (alloc):    ${t['maint_alloc'].sum():>15,.0f}  ({t['maint_alloc'].sum()/t['total_revenue'].sum()*100:.1f}% of rev)")
print(f"  Safety cost:            ${t['safety_cost'].sum():>15,.0f}  ({t['safety_cost'].sum()/t['total_revenue'].sum()*100:.1f}% of rev)")
print(f"  NET MARGIN:             ${t['net_margin'].sum():>15,.0f}  ({t['net_margin'].sum()/t['total_revenue'].sum()*100:.1f}% of rev)")

# by customer
cust_p = t.merge(customers[["customer_id","customer_name"]], on="customer_id").groupby("customer_name").agg(
        trips=("trip_id","count"), revenue=("total_revenue","sum"),
        margin=("net_margin","sum")).round(0).sort_values("margin")
cust_p["margin_pct"] = (cust_p["margin"]/cust_p["revenue"]*100).round(1)
print("\nTop/bottom 5 customers by net margin:")
print(cust_p.tail(5).to_string())
print(cust_p.head(5).to_string())

# by driver
drv_p = t.groupby("driver_id").agg(trips=("trip_id","count"),
        revenue=("total_revenue","sum"), margin=("net_margin","sum")).round(0)
drv_p = drv_p.merge(drivers[["driver_id","years_experience"]], on="driver_id")
drv_p["margin_pct"] = (drv_p["margin"]/drv_p["revenue"]*100).round(1)
print("\nTop 5 drivers by net margin:")
print(drv_p.sort_values("margin",ascending=False).head(5).to_string())
print("\nBottom 5 drivers by net margin:")
print(drv_p.sort_values("margin").head(5).to_string())

# by month
t["dispatch_date"] = pd.to_datetime(t["dispatch_date"])
t["ym"] = t["dispatch_date"].dt.to_period("M")
mon_p = t.groupby("ym").agg(trips=("trip_id","count"),
        revenue=("total_revenue","sum"),
        fuel=("fuel_cost","sum"),
        maint=("maint_alloc","sum"),
        safety=("safety_cost","sum"),
        margin=("net_margin","sum")).round(0)
mon_p["margin_pct"] = (mon_p["margin"]/mon_p["revenue"]*100).round(1)
print("\nMonthly P&L:")
print(mon_p.to_string())

# correlation: years_experience vs margin
print(f"\nDriver years_experience vs margin%: r={drv_p['years_experience'].corr(drv_p['margin_pct']).round(3)}")
print(f"Driver years_experience vs revenue:  r={drv_p['years_experience'].corr(drv_p['revenue']).round(3)}")
