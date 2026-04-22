# ============================================================
# PFIZER GCS PROJECT — STEP 2: Deep Analysis + Charts
# Vaishnavi Mallikarjungajarla | Northeastern University
# ============================================================
# This script produces 6 charts saved as PNG files:
#   chart1_delay_by_shipping_mode.png
#   chart2_delay_trend_over_time.png
#   chart3_top_late_categories.png
#   chart4_anomaly_detection.png
#   chart5_demand_forecast.png
#   chart6_kpi_summary.png
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

# ── STYLE ────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor':   '#F8F9FA',
    'axes.grid':        True,
    'grid.alpha':       0.3,
    'font.family':      'sans-serif',
    'axes.spines.top':  False,
    'axes.spines.right':False,
})
PFIZER_BLUE  = '#0093D0'
PFIZER_DARK  = '#003087'
ALERT_RED    = '#E63946'
SAFE_GREEN   = '#2DC653'
AMBER        = '#F4A261'

print("=" * 60)
print("STEP 2 — DEEP ANALYSIS")
print("=" * 60)

# ── LOAD CLEANED DATA ────────────────────────────────────────
df = pd.read_csv('supply_chain_clean.csv', encoding='latin-1',
                 parse_dates=['order_date_dateorders', 'shipping_date_dateorders'])
print(f"Loaded {len(df):,} rows\n")

# ════════════════════════════════════════════════════════════
# CHART 1 — DELAY RATE BY SHIPPING MODE
# ════════════════════════════════════════════════════════════
print("Building Chart 1: Delay by shipping mode...")

delay_mode = (
    df.groupby('shipping_mode')['is_delayed']
    .agg(['mean', 'count'])
    .reset_index()
)
delay_mode.columns = ['shipping_mode', 'delay_rate', 'orders']
delay_mode['delay_rate_pct'] = delay_mode['delay_rate'] * 100
delay_mode = delay_mode.sort_values('delay_rate_pct', ascending=True)

fig, ax = plt.subplots(figsize=(10, 5))
colors = [ALERT_RED if r > 60 else AMBER if r > 40 else SAFE_GREEN
          for r in delay_mode['delay_rate_pct']]
bars = ax.barh(delay_mode['shipping_mode'], delay_mode['delay_rate_pct'],
               color=colors, edgecolor='white', linewidth=0.5)

for bar, val, cnt in zip(bars, delay_mode['delay_rate_pct'], delay_mode['orders']):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
            f'{val:.1f}%  ({cnt:,} orders)', va='center', fontsize=10)

ax.axvline(57.3, color=PFIZER_DARK, linestyle='--', linewidth=1.2, label='Overall avg 57.3%')
ax.set_xlabel('Late Delivery Rate (%)', fontsize=11)
ax.set_title('Late Delivery Rate by Shipping Mode\nPfizer GCS — Operational Risk Analysis',
             fontsize=13, fontweight='bold', color=PFIZER_DARK)
ax.legend(fontsize=9)
ax.set_xlim(0, 105)
plt.tight_layout()
plt.savefig('chart1_delay_by_shipping_mode.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved chart1_delay_by_shipping_mode.png")

# Print insight
worst = delay_mode.iloc[-1]
best  = delay_mode.iloc[0]
print(f"  INSIGHT: '{worst['shipping_mode']}' has highest delay rate "
      f"({worst['delay_rate_pct']:.1f}%)")
print(f"  INSIGHT: '{best['shipping_mode']}' has lowest delay rate "
      f"({best['delay_rate_pct']:.1f}%)")

# ════════════════════════════════════════════════════════════
# CHART 2 — MONTHLY DELAY TREND OVER TIME
# ════════════════════════════════════════════════════════════
print("\nBuilding Chart 2: Delay trend over time...")

df['year_month'] = df['order_date_dateorders'].dt.to_period('M')
monthly = (
    df.groupby('year_month')
    .agg(
        total_orders   = ('order_id', 'count'),
        delayed_orders = ('is_delayed', 'sum'),
        avg_delay_days = ('delay_days', 'mean'),
        total_sales    = ('sales', 'sum')
    )
    .reset_index()
)
monthly['delay_rate_pct'] = monthly['delayed_orders'] / monthly['total_orders'] * 100
monthly['year_month_dt'] = monthly['year_month'].dt.to_timestamp()

fig, ax1 = plt.subplots(figsize=(12, 5))
ax2 = ax1.twinx()

ax1.fill_between(monthly['year_month_dt'], monthly['delay_rate_pct'],
                 alpha=0.15, color=ALERT_RED)
ax1.plot(monthly['year_month_dt'], monthly['delay_rate_pct'],
         color=ALERT_RED, linewidth=2, label='Delay rate %')
ax2.bar(monthly['year_month_dt'], monthly['total_orders'],
        width=20, alpha=0.3, color=PFIZER_BLUE, label='Order volume')

ax1.set_ylabel('Late Delivery Rate (%)', color=ALERT_RED, fontsize=11)
ax2.set_ylabel('Order Volume', color=PFIZER_BLUE, fontsize=11)
ax1.set_xlabel('Month', fontsize=11)
ax1.tick_params(axis='y', labelcolor=ALERT_RED)
ax2.tick_params(axis='y', labelcolor=PFIZER_BLUE)
ax1.set_title('Monthly Delay Rate vs Order Volume\nTrend Analysis for Operational Planning',
              fontsize=13, fontweight='bold', color=PFIZER_DARK)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)

plt.tight_layout()
plt.savefig('chart2_delay_trend_over_time.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved chart2_delay_trend_over_time.png")

# ════════════════════════════════════════════════════════════
# CHART 3 — TOP LATE CATEGORIES & REGIONS
# ════════════════════════════════════════════════════════════
print("\nBuilding Chart 3: Top late categories & regions...")

cat_delay = (
    df.groupby('category_name')['is_delayed']
    .agg(['mean', 'count'])
    .reset_index()
)
cat_delay.columns = ['category', 'delay_rate', 'orders']
cat_delay = cat_delay[cat_delay['orders'] > 500]
cat_delay['delay_pct'] = cat_delay['delay_rate'] * 100
cat_delay = cat_delay.sort_values('delay_pct', ascending=False).head(10)

region_delay = (
    df.groupby('order_region')['is_delayed']
    .agg(['mean', 'count'])
    .reset_index()
)
region_delay.columns = ['region', 'delay_rate', 'orders']
region_delay['delay_pct'] = region_delay['delay_rate'] * 100
region_delay = region_delay.sort_values('delay_pct', ascending=False).head(10)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Category
bars1 = ax1.barh(cat_delay['category'], cat_delay['delay_pct'],
                 color=PFIZER_BLUE, edgecolor='white')
for bar, val in zip(bars1, cat_delay['delay_pct']):
    ax1.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
             f'{val:.1f}%', va='center', fontsize=9)
ax1.set_xlabel('Late Delivery Rate (%)', fontsize=10)
ax1.set_title('Top 10 Categories\nby Delay Rate', fontsize=12,
              fontweight='bold', color=PFIZER_DARK)
ax1.set_xlim(0, 100)

# Region
bars2 = ax2.barh(region_delay['region'], region_delay['delay_pct'],
                 color=PFIZER_DARK, edgecolor='white')
for bar, val in zip(bars2, region_delay['delay_pct']):
    ax2.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
             f'{val:.1f}%', va='center', fontsize=9)
ax2.set_xlabel('Late Delivery Rate (%)', fontsize=10)
ax2.set_title('Top 10 Regions\nby Delay Rate', fontsize=12,
              fontweight='bold', color=PFIZER_DARK)
ax2.set_xlim(0, 100)

plt.suptitle('Delay Hotspot Analysis — Category & Region Breakdown',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('chart3_top_late_categories.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved chart3_top_late_categories.png")

# ════════════════════════════════════════════════════════════
# CHART 4 — ANOMALY DETECTION (Isolation Forest)
# ════════════════════════════════════════════════════════════
print("\nBuilding Chart 4: Anomaly detection...")

features = ['days_for_shipping_real', 'days_for_shipment_scheduled',
            'delay_days', 'order_item_quantity', 'sales', 'order_profit_per_order']
features = [f for f in features if f in df.columns]

sample = df[features].dropna().sample(n=min(20000, len(df)), random_state=42)
iso = IsolationForest(contamination=0.05, random_state=42, n_estimators=100)
sample['anomaly'] = iso.fit_predict(sample)
sample['anomaly_label'] = sample['anomaly'].map({1: 'Normal', -1: 'Anomaly'})

n_anomalies = (sample['anomaly'] == -1).sum()
anomaly_pct = n_anomalies / len(sample) * 100
print(f"  Detected {n_anomalies:,} anomalies ({anomaly_pct:.1f}% of orders)")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Scatter: delay_days vs sales
normal  = sample[sample['anomaly'] ==  1]
anomaly = sample[sample['anomaly'] == -1]
ax1.scatter(normal['delay_days'],  normal['sales'],
            c=PFIZER_BLUE, alpha=0.2, s=8, label='Normal')
ax1.scatter(anomaly['delay_days'], anomaly['sales'],
            c=ALERT_RED, alpha=0.6, s=20, label=f'Anomaly ({n_anomalies:,})')
ax1.set_xlabel('Delay Days', fontsize=10)
ax1.set_ylabel('Sales ($)', fontsize=10)
ax1.set_title('Anomaly Detection\nDelay Days vs Sales', fontsize=12,
              fontweight='bold', color=PFIZER_DARK)
ax1.legend(fontsize=9)

# Bar: anomaly by shipping mode (merge back)
df_sample = df.loc[sample.index].copy()
df_sample['anomaly'] = sample['anomaly'].values
anomaly_mode = df_sample.groupby('shipping_mode')['anomaly'].apply(
    lambda x: (x == -1).sum() / len(x) * 100
).reset_index()
anomaly_mode.columns = ['shipping_mode', 'anomaly_rate']
anomaly_mode = anomaly_mode.sort_values('anomaly_rate', ascending=True)

ax2.barh(anomaly_mode['shipping_mode'], anomaly_mode['anomaly_rate'],
         color=ALERT_RED, edgecolor='white', alpha=0.8)
for i, (_, row) in enumerate(anomaly_mode.iterrows()):
    ax2.text(row['anomaly_rate'] + 0.1, i, f"{row['anomaly_rate']:.1f}%",
             va='center', fontsize=10)
ax2.set_xlabel('Anomaly Rate (%)', fontsize=10)
ax2.set_title('Anomaly Rate\nby Shipping Mode', fontsize=12,
              fontweight='bold', color=PFIZER_DARK)
ax2.set_xlim(0, 15)

plt.suptitle('Isolation Forest Anomaly Detection — Operational Risk Flags',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('chart4_anomaly_detection.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved chart4_anomaly_detection.png")

# ════════════════════════════════════════════════════════════
# CHART 5 — DEMAND FORECASTING
# ════════════════════════════════════════════════════════════
print("\nBuilding Chart 5: Demand forecasting...")

monthly_demand = (
    df.groupby('year_month')
    .agg(total_qty=('order_item_quantity', 'sum'))
    .reset_index()
)
monthly_demand['month_num'] = range(len(monthly_demand))
monthly_demand['year_month_dt'] = monthly_demand['year_month'].dt.to_timestamp()

# Linear regression forecast
X = monthly_demand['month_num'].values.reshape(-1, 1)
y = monthly_demand['total_qty'].values
model = LinearRegression()
model.fit(X, y)

# Forecast 6 months ahead
last_month = monthly_demand['month_num'].max()
future_months = np.arange(last_month + 1, last_month + 7).reshape(-1, 1)
forecast = model.predict(future_months)
last_date = monthly_demand['year_month_dt'].max()
future_dates = [last_date + pd.DateOffset(months=i+1) for i in range(6)]

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(monthly_demand['year_month_dt'], monthly_demand['total_qty'],
        color=PFIZER_BLUE, linewidth=2, marker='o', markersize=4, label='Actual demand')
ax.plot(monthly_demand['year_month_dt'], model.predict(X),
        color=PFIZER_DARK, linewidth=1.5, linestyle='--', alpha=0.6, label='Trend')
ax.plot(future_dates, forecast,
        color=ALERT_RED, linewidth=2, marker='s', markersize=5,
        linestyle='--', label='6-month forecast')

# Confidence band
std_resid = np.std(y - model.predict(X))
ax.fill_between(future_dates,
                forecast - 1.5 * std_resid,
                forecast + 1.5 * std_resid,
                alpha=0.15, color=ALERT_RED, label='Confidence band')

ax.axvline(last_date, color='gray', linestyle=':', linewidth=1)
ax.text(last_date, ax.get_ylim()[1] * 0.95, '  Forecast start',
        fontsize=9, color='gray')
ax.set_xlabel('Month', fontsize=11)
ax.set_ylabel('Total Order Quantity', fontsize=11)
ax.set_title('Supply Chain Demand Forecasting — 6-Month Outlook\nLinear Trend + Confidence Band',
             fontsize=13, fontweight='bold', color=PFIZER_DARK)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig('chart5_demand_forecast.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved chart5_demand_forecast.png")

# ════════════════════════════════════════════════════════════
# CHART 6 — KPI SUMMARY DASHBOARD
# ════════════════════════════════════════════════════════════
print("\nBuilding Chart 6: KPI summary dashboard...")

total_orders    = len(df)
delay_rate      = df['is_delayed'].mean() * 100
avg_delay_days  = df.loc[df['is_delayed']==1, 'delay_days'].mean()
total_sales     = df['sales'].sum()
avg_profit      = df['order_profit_per_order'].mean()
on_time_rate    = 100 - delay_rate
top_mode        = df['shipping_mode'].value_counts().index[0]
top_mode_pct    = df['shipping_mode'].value_counts(normalize=True).iloc[0] * 100

fig = plt.figure(figsize=(14, 8))
fig.patch.set_facecolor('white')

gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.5, wspace=0.4)

def kpi_box(ax, value, label, color, sub=''):
    ax.set_facecolor(color)
    ax.text(0.5, 0.58, value, transform=ax.transAxes,
            ha='center', va='center', fontsize=20, fontweight='bold', color='white')
    ax.text(0.5, 0.22, label, transform=ax.transAxes,
            ha='center', va='center', fontsize=9, color='white', alpha=0.9)
    if sub:
        ax.text(0.5, 0.05, sub, transform=ax.transAxes,
                ha='center', va='center', fontsize=8, color='white', alpha=0.7)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

# Row 1: KPI boxes
ax1 = fig.add_subplot(gs[0, 0])
kpi_box(ax1, f'{total_orders:,}', 'Total Orders', PFIZER_BLUE)

ax2 = fig.add_subplot(gs[0, 1])
kpi_box(ax2, f'{delay_rate:.1f}%', 'Late Delivery Rate', ALERT_RED, 'Above 50% = high risk')

ax3 = fig.add_subplot(gs[0, 2])
kpi_box(ax3, f'{on_time_rate:.1f}%', 'On-Time Rate', SAFE_GREEN)

ax4 = fig.add_subplot(gs[0, 3])
kpi_box(ax4, f'{avg_delay_days:.1f}d', 'Avg Delay (when late)', AMBER)

ax5 = fig.add_subplot(gs[1, 0])
kpi_box(ax5, f'${total_sales/1e6:.1f}M', 'Total Sales', PFIZER_DARK)

ax6 = fig.add_subplot(gs[1, 1])
kpi_box(ax6, f'${avg_profit:.1f}', 'Avg Profit/Order', PFIZER_BLUE)

ax7 = fig.add_subplot(gs[1, 2])
kpi_box(ax7, f'{anomaly_pct:.1f}%', 'Anomaly Rate', ALERT_RED, 'Isolation Forest')

ax8 = fig.add_subplot(gs[1, 3])
kpi_box(ax8, f'{top_mode_pct:.0f}%', f'{top_mode}', PFIZER_DARK, 'Most used ship mode')

# Row 2: Delivery status donut
ax_donut = fig.add_subplot(gs[2, :2])
status_counts = df['delivery_status'].value_counts()
donut_colors  = [ALERT_RED, SAFE_GREEN, PFIZER_BLUE, AMBER]
wedges, texts, autotexts = ax_donut.pie(
    status_counts.values,
    labels=status_counts.index,
    autopct='%1.1f%%',
    colors=donut_colors[:len(status_counts)],
    startangle=90,
    wedgeprops=dict(width=0.5)
)
for t in autotexts:
    t.set_fontsize(9)
ax_donut.set_title('Delivery Status Breakdown', fontsize=11,
                   fontweight='bold', color=PFIZER_DARK)

# Row 2: Order volume by segment
ax_seg = fig.add_subplot(gs[2, 2:])
seg = df['customer_segment'].value_counts()
ax_seg.bar(seg.index, seg.values,
           color=[PFIZER_BLUE, PFIZER_DARK, AMBER][:len(seg)],
           edgecolor='white')
for i, (idx, val) in enumerate(seg.items()):
    ax_seg.text(i, val + 200, f'{val:,}', ha='center', fontsize=9)
ax_seg.set_ylabel('Orders', fontsize=10)
ax_seg.set_title('Orders by Customer Segment', fontsize=11,
                 fontweight='bold', color=PFIZER_DARK)
ax_seg.set_facecolor('#F8F9FA')

fig.suptitle('Supply Chain Operations — KPI Dashboard\nPfizer GCS Operational Analytics Project',
             fontsize=14, fontweight='bold', color=PFIZER_DARK, y=1.01)

plt.savefig('chart6_kpi_summary.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved chart6_kpi_summary.png")

# ════════════════════════════════════════════════════════════
# PRINT ALL KEY INSIGHTS (for your interview talking points)
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("KEY INSIGHTS FOR YOUR PFIZER INTERVIEW")
print("=" * 60)
print(f"""
1. DELAY RATE: {delay_rate:.1f}% of all orders are delivered late
   → Highest risk shipping mode: {worst['shipping_mode']} ({worst['delay_rate_pct']:.1f}% late)
   → Lowest risk shipping mode : {best['shipping_mode']} ({best['delay_rate_pct']:.1f}% late)

2. CYCLE TIME: When orders are delayed, avg delay = {avg_delay_days:.1f} days
   → Max delay in dataset = {df['delay_days'].max()} days
   → Negative delay = early delivery (min = {df['delay_days'].min()} days)

3. ANOMALIES: {n_anomalies:,} anomalous orders detected ({anomaly_pct:.1f}%)
   → These are orders with unusual combinations of qty, delay, profit
   → Flagging these early prevents supply disruptions

4. DEMAND: Forecasting shows {'upward' if model.coef_[0] > 0 else 'downward'} trend
   → Slope = {model.coef_[0]:.1f} units/month
   → Use for procurement and inventory planning

5. REVENUE IMPACT: ${total_sales/1e6:.1f}M total sales in dataset
   → Avg profit per order = ${avg_profit:.2f}
   → Delayed orders likely cost margin (visible in anomaly chart)
""")

print("Step 2 COMPLETE. 6 charts saved in your SupplyChain folder.")
print("Run step3_dashboard.py next for the interactive Streamlit dashboard.")