"""
Generate a concise, chart-heavy PDF report.
"""
from fpdf import FPDF
from pathlib import Path

OUT = Path(__file__).parent / "logistics_analysis_report.pdf"
CHARTS = Path(__file__).parent / "report_charts"

class Report(FPDF):
    def header(self):
        if self.page_no() > 1:
            self.set_font("Helvetica", "I", 7)
            self.set_text_color(150, 150, 150)
            self.cell(0, 4, "Logistics Operations Database - Analysis Report", align="R")
            self.ln(5)
    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 7)
        self.set_text_color(150, 150, 150)
        self.cell(0, 8, f"Page {self.page_no()}", align="C")

    def section(self, title, num=None):
        self.ln(3)
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(30, 60, 120)
        self.cell(0, 7, f"{num}. {title}" if num else title)
        self.ln(7)
        self.set_draw_color(30, 60, 120)
        self.line(15, self.get_y(), 195, self.get_y())
        self.ln(3)
        self.set_text_color(0, 0, 0)

    def body(self, text):
        self.set_font("Helvetica", "", 9.5)
        self.multi_cell(0, 5, text)
        self.ln(1)

    def key(self, text):
        self.set_fill_color(240, 245, 255)
        self.set_font("Helvetica", "B", 9.5)
        self.set_text_color(30, 60, 120)
        self.multi_cell(0, 5.5, f"KEY FINDING: {text}", fill=True)
        self.ln(2)
        self.set_text_color(0, 0, 0)

    def chart(self, filename, w=170):
        path = CHARTS / filename
        if path.exists():
            x = (210 - w) / 2
            self.image(str(path), x=x, w=w)
            self.ln(2)

    def table(self, headers, rows, col_widths=None):
        if col_widths is None:
            col_widths = [180/len(headers)] * len(headers)
        self.set_font("Helvetica", "B", 8.5)
        self.set_fill_color(30, 60, 120)
        self.set_text_color(255, 255, 255)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 6, h, border=1, fill=True, align="C")
        self.ln(6)
        self.set_text_color(0, 0, 0)
        self.set_font("Helvetica", "", 8.5)
        for row in rows:
            for i, val in enumerate(row):
                align = "R" if i > 0 and any(c.isdigit() for c in str(val)) else "L"
                self.cell(col_widths[i], 5.5, str(val), border=1, align=align)
            self.ln(5.5)
        self.ln(2)

pdf = Report()
pdf.set_auto_page_break(True, margin=15)
pdf.add_page()

# ================================================================ TITLE
pdf.set_font("Helvetica", "B", 20)
pdf.set_text_color(30, 60, 120)
pdf.cell(0, 10, "Logistics Operations Database", align="C")
pdf.ln(9)
pdf.set_font("Helvetica", "B", 13)
pdf.set_text_color(80, 80, 80)
pdf.cell(0, 7, "Analysis & Predictive Modeling Report", align="C")
pdf.ln(5)
pdf.set_font("Helvetica", "", 9)
pdf.set_text_color(120, 120, 120)
pdf.cell(0, 5, "Dataset: yogape/logistics-operations-database  |  549,706 records  |  2022-2024", align="C")
pdf.ln(3)
pdf.cell(0, 5, "June 2026", align="C")
pdf.ln(6)

# Summary box
pdf.set_fill_color(245, 248, 255)
pdf.set_font("Helvetica", "", 9.5)
pdf.set_text_color(0, 0, 0)
summary = (
    "Three predictive targets were modeled: on-time delivery (AUC 0.505 -- random noise), "
    "trip duration (R2 0.974 -- distance-driven), and detention minutes (R2 0.04 -- weak "
    "signal from event type and time of day). Driver profile carries no predictive signal "
    "(ICC ~0.001). A fuel_purchases data bug was found and corrected, revealing 72% fleet margins."
)
pdf.multi_cell(0, 5.5, summary, fill=True)
pdf.ln(3)

pdf.chart("02_model_performance.png", w=160)

# ================================================================ 1. DATASET & ON-TIME
pdf.add_page()
pdf.section("Dataset & On-Time Delivery", 1)
pdf.body(
    "14 tables, 549,706 records (2022-2024). Target: on_time_flag (delivery events, 85,410 rows). "
    "Logistic regression with chronological split (train 70,973 / test 14,437)."
)
pdf.table(
    ["Metric", "Value", "Interpretation"],
    [["AUC", "0.505", "= coin flip (no signal)"],
     ["Accuracy", "0.508", "Below 0.557 majority baseline"],
     ["Top feature corr", "0.007", "Effectively zero"]],
    col_widths=[30, 30, 120]
)
pdf.chart("03_ontime_no_signal.png", w=160)
pdf.key("on_time_flag is random Bernoulli(0.45) noise. Flat across every cut: transit buffer, "
        "route, customer, month, lead time. Even a timestamp-derived label is flat. Unpredictable.")

# ================================================================ 2. TRIP DURATION
pdf.add_page()
pdf.section("Trip Duration Prediction", 2)
pdf.body(
    "Target: actual_duration_hours (85,410 trips). Linear & XGBoost, chronological split. "
    "typical_distance_miles replaces actual_distance_miles (leak-safe, r=0.998)."
)
pdf.chart("04_duration_vs_distance.png", w=140)
pdf.table(
    ["Model", "R2", "MAE (h)", "RMSE (h)"],
    [["Linear (59 feats)", "0.974", "1.698", "2.300"],
     ["XGBoost (59 feats)", "0.973", "1.714", "2.336"],
     ["XGBoost (top 4)", "0.974", "1.694", "2.299"],
     ["distance/mph baseline", "0.974", "1.689", "2.305"]],
    col_widths=[55, 30, 30, 30]
)
pdf.chart("05_duration_importance.png", w=160)
pdf.key("Duration = distance / 56mph + noise. route_id + typical_distance_miles = 100% of gain. "
        "Dropping 55 features improved the model. No nonlinear structure to exploit.")

# ================================================================ 3. DETENTION
pdf.add_page()
pdf.section("Detention Minutes Prediction", 3)
pdf.body(
    "Target: detention_minutes (170,820 events, both pickup+delivery). Not distance-driven (r=0.001). "
    "XGBoost, chronological split (train 127,617 / test 28,780)."
)
pdf.chart("06_detention_by_hour.png", w=170)
pdf.table(
    ["Model", "R2", "MAE (min)"],
    [["XGBoost (75 feats)", "0.043", "58.1"],
     ["event_type mean", "0.047", "58.0"],
     ["predict train mean", "-0.000", "59.0"]],
    col_widths=[60, 40, 40]
)
pdf.chart("07_detention_importance.png", w=160)
pdf.key("Real but weak signal (R2=0.04). Deliveries +30 min vs pickups; overnight +20 min; "
        "short lead time increases detention. 96% of variance unexplained -- needs features "
        "not in dataset (dock windows, staffing, weather).")

# ================================================================ 4. DRIVER PROFILE
pdf.add_page()
pdf.section("Driver Profile: No Signal", 4)
pdf.body(
    "Tested: years_experience, age, tenure, CDL, home_terminal, license_state vs all outcomes. "
    "4 grouping strategies, ANOVA, ICC, trip-assignment randomness."
)
pdf.chart("09_experience_flat.png", w=160)
pdf.chart("08_driver_icc.png", w=130)
pdf.key("Driver identity explains 0.1% of variance (ICC ~0.001). Trip assignment is random "
        "w.r.t. experience (same distances, routes, trucks for all). Profile is decorative.")

# ================================================================ 5. FUEL BUG & PROFITABILITY
pdf.add_page()
pdf.section("Data Bug & Profitability", 5)
pdf.body(
    "fuel_purchases.csv gallons are unscaled to trip distance (data-gen bug). 92-mile trips "
    "get same fuel bill as 2,500-mile trips. Fix: use trips.fuel_gallons_used x $3.90/gal."
)
pdf.chart("11_fuel_bug.png", w=160)
pdf.chart("01_pnl_waterfall.png", w=160)
pdf.key("Bug inflated fuel by $22M. After correction: all 58 routes profitable, fleet margin "
        "72% ($215M), fuel 24.7% of revenue (was 32%).")

# ================================================================ 6. ROUTES & SEASONALITY
pdf.add_page()
pdf.section("Routes & Seasonality", 6)
pdf.chart("10_route_margins.png", w=175)
pdf.body("All 58 routes profitable (corrected). Short-haul (92-167 mi) lowest margins (50-78%); "
         "long-haul (2,500+ mi) highest (75-80%).")
pdf.chart("12_seasonality.png", w=170)
pdf.body("No seasonality: ~2,300-2,500 loads/month, ~$3,070 avg revenue, ~$2.20/mile rate, "
         "every month. Margins stable at ~72% throughout 2022-2024.")

# ================================================================ 7. CONCLUSIONS
pdf.add_page()
pdf.section("Conclusions", 7)
pdf.table(
    ["Target", "Best Model", "Performance", "Signal"],
    [["On-time delivery", "LogReg", "AUC 0.505", "None (noise)"],
     ["Trip duration", "XGBoost top-4", "R2 0.974", "Distance (100%)"],
     ["Detention minutes", "XGBoost", "R2 0.043", "Event type, time (4%)"]],
    col_widths=[35, 35, 35, 50]
)
pdf.body(
    "This is a synthetic dataset where distance is the only real driver of trip outcomes, "
    "on-time is random noise, and driver profile is disconnected from performance. A fuel "
    "data bug masked true profitability (72% margins, all routes profitable). For real-world "
    "logistics prediction, the valuable targets would be detention (with dock/staffing/weather "
    "features) and fuel efficiency (with driver-skill variation)."
)

pdf.output(str(OUT))
print(f"Report saved: {OUT}  ({OUT.stat().st_size//1024}KB, {pdf.page_no()} pages)")
