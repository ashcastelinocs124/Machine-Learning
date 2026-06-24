"""
Generate all charts for the PDF report. Saves PNGs to report_charts/.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
})

DATA = Path(__file__).parent
CHARTS = DATA / "report_charts"
CHARTS.mkdir(exist_ok=True)

BLUE = "#1e3c78"; LIGHT = "#5b7fc3"; RED = "#c0392b"; GREEN = "#27ae60"
ORANGE = "#e67e22"; GREY = "#95a5a6"

# load data
trips = pd.read_csv(DATA / "trips.csv")
loads = pd.read_csv(DATA / "loads.csv")
routes = pd.read_csv(DATA / "routes.csv")
events = pd.read_csv(DATA / "delivery_events.csv")
fuel = pd.read_csv(DATA / "fuel_purchases.csv")
maint = pd.read_csv(DATA / "maintenance_records.csv")
safety = pd.read_csv(DATA / "safety_incidents.csv")
drivers = pd.read_csv(DATA / "drivers.csv")
customers = pd.read_csv(DATA / "customers.csv")
dmm = pd.read_csv(DATA / "driver_monthly_metrics.csv")

DIESEL = fuel["price_per_gallon"].mean()

# ================================================================ 1. Fleet P&L bar chart
fig, ax = plt.subplots(figsize=(6, 3.2))
t = trips.merge(loads[["load_id","revenue","fuel_surcharge","accessorial_charges"]], on="load_id")
rev = (t["revenue"]+t["fuel_surcharge"]+t["accessorial_charges"]).sum()
fc = (t["fuel_gallons_used"]*DIESEL).sum()
mc = maint["total_cost"].sum()
sc = (safety["vehicle_damage_cost"].fillna(0)+safety["cargo_damage_cost"].fillna(0)+safety["claim_amount"].fillna(0)).sum()
nm = rev - fc - mc - sc
labels = ["Revenue", "Fuel", "Maint.", "Safety", "Net Margin"]
vals = [rev, -fc, -mc, -sc, nm]
colors = [BLUE, RED, ORANGE, RED, GREEN]
ax.bar(labels, [abs(v) for v in vals], color=colors, width=0.6, edgecolor="white")
for i,(l,v) in enumerate(zip(labels,vals)):
    ax.text(i, abs(v)+rev*0.01, f"${v/1e6:.0f}M", ha="center", fontsize=8, fontweight="bold")
ax.set_ylabel("USD")
ax.set_title("Fleet P&L (2022-2024, corrected fuel)")
ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x,_: f"${x/1e6:.0f}M"))
plt.tight_layout(); plt.savefig(CHARTS/"01_pnl_waterfall.png"); plt.close()

# ================================================================ 2. Model performance comparison
fig, ax = plt.subplots(figsize=(6, 2.8))
models = ["On-time\n(LogReg)", "Duration\n(Linear)", "Duration\n(XGBoost)", "Detention\n(XGBoost)"]
r2s = [0.005, 0.974, 0.974, 0.043]
aucs = [0.505, None, None, None]
x = np.arange(len(models))
bars = ax.bar(x, [0.005, 0.974, 0.974, 0.043], color=[GREY, BLUE, LIGHT, ORANGE], width=0.55, edgecolor="white")
ax.axhline(0.5, color=RED, linestyle="--", linewidth=0.8, label="Random baseline (AUC=0.5)")
for i,(m,v) in enumerate(zip(models, [0.005,0.974,0.974,0.043])):
    label = f"AUC={v:.3f}" if i==0 else f"R2={v:.3f}"
    ax.text(i, v+0.02, label, ha="center", fontsize=8, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(models)
ax.set_ylabel("AUC (on-time) / R2 (regression)")
ax.set_title("Model Performance Summary")
ax.set_ylim(0, 1.1)
ax.legend(loc="upper left", fontsize=7)
plt.tight_layout(); plt.savefig(CHARTS/"02_model_performance.png"); plt.close()

# ================================================================ 3. On-time signal: corr with features
fig, ax = plt.subplots(figsize=(6, 3))
de = events[events.event_type=="Delivery"].copy()
de["on_time_flag"] = de["on_time_flag"].astype(int)
de["scheduled_datetime"] = pd.to_datetime(de["scheduled_datetime"])
de["scheduled_transit_gap"] = (de["scheduled_datetime"].dt.normalize() -
    pd.to_datetime(de.merge(trips[["trip_id","dispatch_date"]],on="trip_id")["dispatch_date"]).dt.normalize()).dt.days
de_merged = de.merge(trips[["trip_id","dispatch_date","actual_distance_miles"]],on="trip_id")
de_merged = de_merged.merge(loads[["load_id","route_id"]],on="load_id")
de_merged = de_merged.merge(routes[["route_id","typical_distance_miles"]],on="route_id")
gap = de_merged.assign(bucket=pd.cut(de_merged["scheduled_transit_gap"],[-100,-2,-1,0,1,2,100]))
g = gap.groupby("bucket", observed=True)["on_time_flag"].mean()
ax.bar(range(len(g)), g.values, color=LIGHT, width=0.6, edgecolor="white")
ax.axhline(g.mean(), color=RED, linestyle="--", linewidth=1, label=f"Mean={g.mean():.3f}")
ax.set_xticks(range(len(g))); ax.set_xticklabels([str(x) for x in g.index], fontsize=7)
ax.set_ylabel("On-time rate")
ax.set_title("On-time rate by transit buffer (FLAT = no signal)")
ax.set_ylim(0.3, 0.6)
ax.legend()
plt.tight_layout(); plt.savefig(CHARTS/"03_ontime_no_signal.png"); plt.close()

# ================================================================ 4. Duration vs distance scatter
fig, ax = plt.subplots(figsize=(5, 3.2))
sample = trips.sample(3000, random_state=42)
ax.scatter(sample["actual_distance_miles"], sample["actual_duration_hours"],
           s=3, alpha=0.3, color=LIGHT)
z = np.polyfit(trips["actual_distance_miles"], trips["actual_duration_hours"], 1)
xline = np.array([0, trips["actual_distance_miles"].max()])
ax.plot(xline, np.polyval(z, xline), color=RED, linewidth=1.5, label=f"y={z[0]:.2f}x+{z[1]:.1f}")
ax.set_xlabel("Distance (miles)"); ax.set_ylabel("Duration (hours)")
ax.set_title(f"Trip duration = distance / 56mph (r=0.99)")
ax.legend(fontsize=7)
plt.tight_layout(); plt.savefig(CHARTS/"04_duration_vs_distance.png"); plt.close()

# ================================================================ 5. XGBoost feature importance (duration)
fig, ax = plt.subplots(figsize=(6, 2.8))
feats = ["route_id", "typical_distance_miles", "typical_transit_days", "fuel_surcharge_rate", "all others"]
shares = [73.6, 26.4, 0.0, 0.1, 0.0]
colors = [BLUE, LIGHT, GREY, GREY, GREY]
bars = ax.barh(feats[::-1], shares[::-1], color=colors[::-1], edgecolor="white")
for bar, s in zip(bars, shares[::-1]):
    ax.text(bar.get_width()+1, bar.get_y()+bar.get_height()/2, f"{s:.1f}%",
            va="center", fontsize=8, fontweight="bold")
ax.set_xlabel("Gain share (%)")
ax.set_title("Trip Duration: Feature Importance (top-4 model)")
ax.set_xlim(0, 85)
plt.tight_layout(); plt.savefig(CHARTS/"05_duration_importance.png"); plt.close()

# ================================================================ 6. Detention by hour
fig, ax = plt.subplots(figsize=(6, 2.8))
ev = events.copy()
ev["scheduled_datetime"] = pd.to_datetime(ev["scheduled_datetime"])
hourly = ev.groupby(ev["scheduled_datetime"].dt.hour)["detention_minutes"].mean()
colors_h = [ORANGE if (h<6 or h>=19) else BLUE for h in hourly.index]
ax.bar(hourly.index, hourly.values, color=colors_h, width=0.7, edgecolor="white")
ax.set_xlabel("Scheduled hour"); ax.set_ylabel("Detention (min)")
ax.set_title("Detention by time of day (overnight = +20 min)")
ax.set_xticks(range(0,24,2))
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=BLUE,label="Daytime"),Patch(color=ORANGE,label="Overnight")], fontsize=7)
plt.tight_layout(); plt.savefig(CHARTS/"06_detention_by_hour.png"); plt.close()

# ================================================================ 7. Detention feature importance
fig, ax = plt.subplots(figsize=(6, 2.8))
feats = ["event_type", "is_overnight", "days_dispatch_to_sched", "typical_distance", "scheduled_hour", "others"]
shares = [28.2, 19.3, 17.2, 3.2, 3.0, 29.1]
colors = [BLUE, ORANGE, LIGHT, GREY, GREY, GREY]
ax.barh(feats[::-1], shares[::-1], color=colors[::-1], edgecolor="white")
for i,(s) in enumerate(shares[::-1]):
    ax.text(s+0.5, i, f"{s:.1f}%", va="center", fontsize=8, fontweight="bold")
ax.set_xlabel("Gain share (%)")
ax.set_title("Detention: Feature Importance")
ax.set_xlim(0, 35)
plt.tight_layout(); plt.savefig(CHARTS/"07_detention_importance.png"); plt.close()

# ================================================================ 8. Driver ICC (interchangeable)
fig, ax = plt.subplots(figsize=(5, 2.5))
outcomes = ["Distance", "MPG", "Idle", "Duration"]
iccs = [0.0014, 0.0016, 0.0014, 0.0012]
ax.barh(outcomes[::-1], iccs[::-1], color=GREY, edgecolor="white")
ax.axvline(0.01, color=RED, linestyle="--", linewidth=1, label="Small effect threshold")
ax.set_xlabel("ICC (intraclass correlation)")
ax.set_title("Driver identity explains ~0.1% of variance\n(drivers are interchangeable)")
ax.legend(fontsize=7)
ax.set_xlim(0, 0.02)
plt.tight_layout(); plt.savefig(CHARTS/"08_driver_icc.png"); plt.close()

# ================================================================ 9. Experience vs outcomes (flat)
fig, axes = plt.subplots(1, 2, figsize=(6, 2.5))
tr = trips.merge(drivers[["driver_id","years_experience"]], on="driver_id")
tr = tr[tr.years_experience.notna()]
tr["mpg"] = tr.actual_distance_miles / tr.fuel_gallons_used
buckets = pd.cut(tr.years_experience, [0,5,10,15,20,30], labels=["0-5","5-10","10-15","15-20","20-30"])
g1 = tr.groupby(buckets, observed=True)["actual_distance_miles"].mean()
g2 = tr.groupby(buckets, observed=True)["mpg"].mean()
axes[0].bar(g1.index.astype(str), g1.values, color=LIGHT, edgecolor="white")
axes[0].set_title("Trip distance by experience (FLAT)", fontsize=9)
axes[0].set_ylabel("Avg distance (mi)")
axes[1].bar(g2.index.astype(str), g2.values, color=BLUE, edgecolor="white")
axes[1].set_title("MPG by experience (FLAT)", fontsize=9)
axes[1].set_ylabel("Avg MPG")
axes[1].set_ylim(6.4, 6.6)
for a in axes:
    a.tick_params(axis="x", labelsize=7)
plt.tight_layout(); plt.savefig(CHARTS/"09_experience_flat.png"); plt.close()

# ================================================================ 10. Route margins (corrected)
fig, ax = plt.subplots(figsize=(6, 3.2))
t = trips.merge(loads[["load_id","route_id","revenue","fuel_surcharge","accessorial_charges"]], on="load_id")
t = t.merge(routes[["route_id","origin_city","destination_city","typical_distance_miles"]], on="route_id")
t["total_rev"] = t["revenue"]+t["fuel_surcharge"]+t["accessorial_charges"]
t["fuel_cost"] = t["fuel_gallons_used"]*DIESEL
t["margin"] = t["total_rev"]-t["fuel_cost"]
rp = t.groupby(["route_id","origin_city","destination_city","typical_distance_miles"]).agg(
    margin=("margin","sum"), rev=("total_rev","sum")).reset_index()
rp["margin_pct"] = rp["margin"]/rp["rev"]*100
rp = rp.sort_values("margin_pct")
rp["label"] = rp["origin_city"].str[:8] + " -> " + rp["destination_city"].str[:8]
colors_r = [RED if m<60 else (ORANGE if m<70 else GREEN) for m in rp["margin_pct"]]
ax.barh(range(len(rp)), rp["margin_pct"], color=colors_r, edgecolor="white")
ax.set_yticks(range(0, len(rp), 5))
ax.set_yticklabels([rp["label"].iloc[i] for i in range(0,len(rp),5)], fontsize=6)
ax.set_xlabel("Margin %")
ax.set_title("Route profitability (all 58 routes, corrected fuel)")
ax.axvline(70, color="black", linestyle=":", linewidth=0.5)
plt.tight_layout(); plt.savefig(CHARTS/"10_route_margins.png"); plt.close()

# ================================================================ 11. Fuel bug: before vs after
fig, axes = plt.subplots(1, 2, figsize=(6, 2.5))
# before (broken fuel_purchases)
fp_trip = fuel.groupby("trip_id")["total_cost"].sum()
t_b = trips.merge(fp_trip, on="trip_id", how="left").merge(loads[["load_id","route_id"]],on="load_id")
t_b = t_b.merge(routes[["route_id","typical_distance_miles"]],on="route_id")
b = t_b.assign(short=t_b.typical_distance_miles<500).groupby("short")["total_cost"].mean()
axes[0].bar(["Short-haul\n(<500mi)","Long-haul\n(>500mi)"], [b.get(False,0), b.get(True,0)],
           color=[RED, RED], edgecolor="white")
axes[0].set_title("BEFORE: fuel_purchases\n(broken, unscaled)", fontsize=9, color=RED)
axes[0].set_ylabel("Avg fuel cost/trip ($)")
# after (corrected)
t_a = trips.copy(); t_a["fuel_cost"] = t_a["fuel_gallons_used"]*DIESEL
t_a = t_a.merge(loads[["load_id","route_id"]],on="load_id").merge(routes[["route_id","typical_distance_miles"]],on="route_id")
a = t_a.assign(short=t_a.typical_distance_miles<500).groupby("short")["fuel_cost"].mean()
axes[1].bar(["Short-haul\n(<500mi)","Long-haul\n(>500mi)"], [a.get(False,0), a.get(True,0)],
           color=[GREEN, GREEN], edgecolor="white")
axes[1].set_title("AFTER: trips.fuel_gallons_used\n(corrected)", fontsize=9, color=GREEN)
axes[1].set_ylabel("Avg fuel cost/trip ($)")
plt.tight_layout(); plt.savefig(CHARTS/"11_fuel_bug.png"); plt.close()

# ================================================================ 12. Monthly loads (seasonality)
fig, ax = plt.subplots(figsize=(6, 2.2))
l = loads.copy(); l["load_date"] = pd.to_datetime(l["load_date"])
l["ym"] = l["load_date"].dt.to_period("M")
monthly = l.groupby("ym").size()
ax.plot(range(len(monthly)), monthly.values, color=BLUE, linewidth=1.5)
ax.fill_between(range(len(monthly)), monthly.values, alpha=0.2, color=BLUE)
ax.set_ylabel("Loads/month")
ax.set_title("Monthly load volume (no seasonality)")
ax.set_xticks([0,12,24,35])
ax.set_xticklabels(["Jan 2022","Jan 2023","Jan 2024","Dec 2024"], fontsize=7)
ax.set_ylim(2000, 2600)
plt.tight_layout(); plt.savefig(CHARTS/"12_seasonality.png"); plt.close()

print(f"Generated {len(list(CHARTS.glob('*.png')))} charts in {CHARTS}/")
for f in sorted(CHARTS.glob("*.png")):
    print(f"  {f.name}  ({f.stat().st_size//1024}KB)")
