# ============================================================
# PFIZER GCS — FIXED ML FORECAST + POWER BI DATA EXPORT
# Uses WEEKLY aggregation (150+ rows vs 31 monthly)
# Vaishnavi Mallikarjungajarla | Northeastern University
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

PFIZER_BLUE = '#0093D0'
PFIZER_DARK = '#003087'
ALERT_RED   = '#E63946'
SAFE_GREEN  = '#2DC653'
AMBER       = '#F4A261'

plt.rcParams.update({
    'figure.facecolor': 'white', 'axes.facecolor': '#F8F9FA',
    'axes.grid': True, 'grid.alpha': 0.3, 'font.family': 'sans-serif',
    'axes.spines.top': False, 'axes.spines.right': False,
})

print("=" * 60)
print("FIXED ML FORECASTING — WEEKLY AGGREGATION")
print("=" * 60)

df = pd.read_csv('supply_chain_clean.csv', encoding='latin-1',
                 parse_dates=['order_date_dateorders'])
print(f"Loaded {len(df):,} rows")

# ── WEEKLY AGGREGATION (gives ~150 rows) ─────────────────────
df['week'] = df['order_date_dateorders'].dt.to_period('W')
weekly = (
    df.groupby('week')
    .agg(
        total_qty      = ('order_item_quantity', 'sum'),
        total_orders   = ('order_id', 'count'),
        total_sales    = ('sales', 'sum'),
        avg_profit     = ('order_profit_per_order', 'mean'),
        delay_rate     = ('is_delayed', 'mean'),
        avg_delay_days = ('delay_days', 'mean'),
        avg_discount   = ('order_item_discount_rate', 'mean'),
    )
    .reset_index()
)
weekly['week_num']  = range(len(weekly))
weekly['week_dt']   = weekly['week'].dt.to_timestamp()
weekly = weekly[weekly['total_orders'] > 10].reset_index(drop=True)
weekly['week_num']  = range(len(weekly))
print(f"Weekly rows: {len(weekly)}")

# ── FEATURE ENGINEERING (25 features) ────────────────────────
for lag in [1, 2, 3, 4]:
    weekly[f'qty_lag_{lag}']   = weekly['total_qty'].shift(lag)
    weekly[f'sales_lag_{lag}'] = weekly['total_sales'].shift(lag)
    weekly[f'delay_lag_{lag}'] = weekly['delay_rate'].shift(lag)

weekly['qty_roll4_mean']    = weekly['total_qty'].shift(1).rolling(4).mean()
weekly['qty_roll4_std']     = weekly['total_qty'].shift(1).rolling(4).std()
weekly['qty_roll8_mean']    = weekly['total_qty'].shift(1).rolling(8).mean()
weekly['sales_roll4_mean']  = weekly['total_sales'].shift(1).rolling(4).mean()
weekly['week_of_year']      = weekly['week_dt'].dt.isocalendar().week.astype(int)
weekly['month_of_year']     = weekly['week_dt'].dt.month
weekly['quarter']           = weekly['week_dt'].dt.quarter
weekly['week_sin']          = np.sin(2 * np.pi * weekly['week_of_year'] / 52)
weekly['week_cos']          = np.cos(2 * np.pi * weekly['week_of_year'] / 52)
weekly['week_trend']        = weekly['week_num']
weekly['stockout_risk_score']= (
    (1 - weekly['total_qty'] / weekly['total_qty'].max()) * 0.6 +
    weekly['delay_rate'] * 0.4
)
weekly['order_variance'] = weekly['total_orders'].pct_change().fillna(0)

feature_cols = [
    'qty_lag_1','qty_lag_2','qty_lag_3','qty_lag_4',
    'sales_lag_1','sales_lag_2','sales_lag_3','sales_lag_4',
    'delay_lag_1','delay_lag_2','delay_lag_3','delay_lag_4',
    'qty_roll4_mean','qty_roll4_std','qty_roll8_mean','sales_roll4_mean',
    'week_of_year','month_of_year','quarter',
    'week_sin','week_cos','week_trend',
    'avg_profit','delay_rate','stockout_risk_score','order_variance',
]
feature_cols = [f for f in feature_cols if f in weekly.columns]

model_df = weekly.dropna(subset=feature_cols + ['total_qty']).copy().reset_index(drop=True)
print(f"Rows for modeling: {len(model_df)}")

X = model_df[feature_cols].values
y = model_df['total_qty'].values

# ── TIME-SERIES CROSS VALIDATION ─────────────────────────────
print("\nRunning Time-Series Cross Validation (5 splits)...")
tscv = TimeSeriesSplit(n_splits=5)

models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(
        n_estimators=300, max_depth=6, min_samples_leaf=3,
        random_state=42, n_jobs=-1),
    'XGBoost': xgb.XGBRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbosity=0),
}

results = {}
for name, model in models.items():
    maes, mapes = [], []
    for train_idx, test_idx in tscv.split(X):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        model.fit(X_tr, y_tr)
        y_pred = np.maximum(model.predict(X_te), 0)
        maes.append(mean_absolute_error(y_te, y_pred))
        mape = np.mean(np.abs((y_te - y_pred) / np.maximum(y_te, 1))) * 100
        mapes.append(mape)
    acc = max(0, 100 - np.mean(mapes))
    results[name] = {'MAE': np.mean(maes), 'MAPE': np.mean(mapes), 'Accuracy': acc}
    print(f"  {name:22s} | Accuracy: {acc:.1f}% | MAE: {np.mean(maes):.0f}")

# ── FINAL TRAIN/TEST SPLIT ───────────────────────────────────
split = len(X) - 8
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

preds = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds[name] = np.maximum(model.predict(X_test), 0)

# ── CHART 7: MODEL COMPARISON ────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax1 = axes[0]
names = list(results.keys())
accs  = [results[m]['Accuracy'] for m in names]
bars  = ax1.bar(names, accs, color=[PFIZER_BLUE, SAFE_GREEN, PFIZER_DARK],
                edgecolor='white', width=0.5)
for bar, val in zip(bars, accs):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{val:.1f}%', ha='center', fontsize=12, fontweight='bold')
ax1.set_ylabel('Forecast Accuracy (%)', fontsize=11)
ax1.set_title('Model Accuracy Comparison\n(Time-Series Cross Validation)',
              fontsize=12, fontweight='bold', color=PFIZER_DARK)
ax1.set_ylim(0, 115)
ax1.axhline(80, color=AMBER, linestyle='--', linewidth=1.2, label='80% target')
ax1.legend(fontsize=9)

ax2 = axes[1]
test_dates = model_df['week_dt'].values[split:]
ax2.plot(test_dates, y_test, 'o-', color=PFIZER_BLUE, lw=2, ms=6,
         label='Actual', zorder=3)
plot_colors = [SAFE_GREEN, AMBER, PFIZER_DARK]
for (name, pred), col in zip(preds.items(), plot_colors):
    ax2.plot(test_dates, pred, 's--', color=col, lw=1.5, ms=5,
             label=f'{name}', alpha=0.85)
ax2.set_ylabel('Weekly Order Quantity', fontsize=11)
ax2.set_title('Actual vs Predicted — Last 8 Weeks\nTest Set Validation',
              fontsize=12, fontweight='bold', color=PFIZER_DARK)
ax2.legend(fontsize=9)
plt.suptitle('ML Forecasting Pipeline — Random Forest + XGBoost vs Baseline',
             fontsize=13, fontweight='bold', color=PFIZER_DARK, y=1.02)
plt.tight_layout()
plt.savefig('chart7_ml_forecast_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved chart7_ml_forecast_comparison.png")

# ── CHART 8: FEATURE IMPORTANCE ──────────────────────────────
rf_model  = models['Random Forest']
xgb_model = models['XGBoost']
rf_model.fit(X, y)
xgb_model.fit(X, y)

rf_imp  = pd.Series(rf_model.feature_importances_,  index=feature_cols).sort_values(ascending=False).head(12)
xgb_imp = pd.Series(xgb_model.feature_importances_, index=feature_cols).sort_values(ascending=False).head(12)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.barh(rf_imp.index[::-1],  rf_imp.values[::-1],  color=SAFE_GREEN,  edgecolor='white')
ax1.set_xlabel('Importance', fontsize=10)
ax1.set_title('Random Forest — Top Features', fontsize=12, fontweight='bold', color=PFIZER_DARK)
ax2.barh(xgb_imp.index[::-1], xgb_imp.values[::-1], color=PFIZER_DARK, edgecolor='white')
ax2.set_xlabel('Importance', fontsize=10)
ax2.set_title('XGBoost — Top Features', fontsize=12, fontweight='bold', color=PFIZER_DARK)
plt.suptitle('Feature Importance — What Drives Demand Forecasting',
             fontsize=13, fontweight='bold', color=PFIZER_DARK, y=1.02)
plt.tight_layout()
plt.savefig('chart8_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved chart8_feature_importance.png")

# ════════════════════════════════════════════════════════════
# POWER BI DATA EXPORTS — 5 CSV files
# Load these into Power BI as separate tables
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("EXPORTING POWER BI DATA FILES")
print("=" * 60)

# 1. KPI Summary table
df_clean = pd.read_csv('supply_chain_clean.csv', encoding='latin-1',
                       parse_dates=['order_date_dateorders'])
df_clean['year_month'] = df_clean['order_date_dateorders'].dt.to_period('M').astype(str)

kpi_summary = pd.DataFrame([{
    'metric': 'Total Orders',       'value': len(df_clean),        'unit': 'orders'},
    {'metric': 'Late Delivery Rate', 'value': round(df_clean['is_delayed'].mean()*100,1), 'unit': '%'},
    {'metric': 'On-Time Rate',       'value': round((1-df_clean['is_delayed'].mean())*100,1), 'unit': '%'},
    {'metric': 'Avg Delay Days',     'value': round(df_clean.loc[df_clean['is_delayed']==1,'delay_days'].mean(),1), 'unit': 'days'},
    {'metric': 'Total Sales',        'value': round(df_clean['sales'].sum(),0), 'unit': 'USD'},
    {'metric': 'Avg Profit/Order',   'value': round(df_clean['order_profit_per_order'].mean(),2), 'unit': 'USD'},
    {'metric': 'Anomaly Rate',       'value': 5.0,  'unit': '%'},
    {'metric': 'Forecast Accuracy',  'value': round(max(r['Accuracy'] for r in results.values()),1), 'unit': '%'},
])
kpi_summary.to_csv('powerbi_kpi_summary.csv', index=False)
print("  Saved powerbi_kpi_summary.csv  (KPI cards in Power BI)")

# 2. Monthly trend table
monthly_trend = (
    df_clean.groupby('year_month')
    .agg(
        total_orders   = ('order_id', 'count'),
        total_qty      = ('order_item_quantity', 'sum'),
        total_sales    = ('sales', 'sum'),
        delayed_orders = ('is_delayed', 'sum'),
        avg_delay_days = ('delay_days', 'mean'),
    ).reset_index()
)
monthly_trend['delay_rate_pct']  = (monthly_trend['delayed_orders'] / monthly_trend['total_orders'] * 100).round(1)
monthly_trend['on_time_rate_pct']= (100 - monthly_trend['delay_rate_pct']).round(1)
monthly_trend['order_variance']  = monthly_trend['total_orders'].pct_change().fillna(0).round(3)
monthly_trend.to_csv('powerbi_monthly_trend.csv', index=False)
print("  Saved powerbi_monthly_trend.csv  (line/bar trend charts)")

# 3. Delay by shipping mode + category
delay_mode = (
    df_clean.groupby('shipping_mode')['is_delayed']
    .agg(['mean','count']).reset_index()
)
delay_mode.columns = ['shipping_mode','delay_rate','order_count']
delay_mode['delay_rate_pct'] = (delay_mode['delay_rate']*100).round(1)
delay_mode.to_csv('powerbi_delay_by_mode.csv', index=False)
print("  Saved powerbi_delay_by_mode.csv  (bar chart by mode)")

delay_cat = (
    df_clean.groupby('category_name')['is_delayed']
    .agg(['mean','count']).reset_index()
)
delay_cat.columns = ['category','delay_rate','order_count']
delay_cat = delay_cat[delay_cat['order_count'] > 200]
delay_cat['delay_rate_pct'] = (delay_cat['delay_rate']*100).round(1)
delay_cat.to_csv('powerbi_delay_by_category.csv', index=False)
print("  Saved powerbi_delay_by_category.csv  (bar chart by category)")

# 4. Forecast vs actual + stockout risk
best_model_name = max(results, key=lambda m: results[m]['Accuracy'])
best_model_obj  = models[best_model_name]
best_model_obj.fit(X, y)

# Last 12 weeks actual
actual_12 = model_df[['week_dt','total_qty']].tail(12).copy()
actual_12.columns = ['date','actual_qty']
actual_12['type'] = 'Actual'

# 6 weeks forecast
last_row = model_df[feature_cols].iloc[-1].values.copy()
future_preds, future_dates = [], []
last_date = model_df['week_dt'].iloc[-1]
for i in range(6):
    pred = float(max(best_model_obj.predict(last_row.reshape(1,-1))[0], 0))
    future_preds.append(round(pred))
    future_dates.append(last_date + pd.DateOffset(weeks=i+1))
    if 'qty_lag_4' in feature_cols:
        last_row[feature_cols.index('qty_lag_4')] = last_row[feature_cols.index('qty_lag_3')]
    if 'qty_lag_3' in feature_cols:
        last_row[feature_cols.index('qty_lag_3')] = last_row[feature_cols.index('qty_lag_2')]
    if 'qty_lag_2' in feature_cols:
        last_row[feature_cols.index('qty_lag_2')] = last_row[feature_cols.index('qty_lag_1')]
    if 'qty_lag_1' in feature_cols:
        last_row[feature_cols.index('qty_lag_1')] = pred

forecast_rows = pd.DataFrame({
    'date':       future_dates,
    'actual_qty': future_preds,
    'type':       'Forecast'
})
forecast_full = pd.concat([actual_12, forecast_rows], ignore_index=True)
forecast_full['stockout_risk']   = forecast_full['actual_qty'].apply(
    lambda x: 'HIGH' if x < 2000 else ('MEDIUM' if x < 4000 else 'LOW'))
forecast_full['anomaly_flag']    = (forecast_full['actual_qty'] < 1500).astype(int)
forecast_full['lower_bound']     = (forecast_full['actual_qty'] * 0.85).round()
forecast_full['upper_bound']     = (forecast_full['actual_qty'] * 1.15).round()
forecast_full.to_csv('powerbi_forecast_vs_actual.csv', index=False)
print("  Saved powerbi_forecast_vs_actual.csv  (forecast line chart)")

# 5. Region delay heatmap data
region_delay = (
    df_clean.groupby(['order_region','shipping_mode'])['is_delayed']
    .agg(['mean','count']).reset_index()
)
region_delay.columns = ['region','shipping_mode','delay_rate','orders']
region_delay['delay_rate_pct'] = (region_delay['delay_rate']*100).round(1)
region_delay.to_csv('powerbi_region_heatmap.csv', index=False)
print("  Saved powerbi_region_heatmap.csv  (matrix/heatmap visual)")

# ════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)
best = results[best_model_name]
print(f"""
Best model       : {best_model_name}
Forecast accuracy: {best['Accuracy']:.1f}%
MAE              : {best['MAE']:.0f} units/week
Features used    : {len(feature_cols)}

All models:""")
for name, r in results.items():
    print(f"  {name:22s}: {r['Accuracy']:.1f}% accuracy")

print(f"""
Power BI files saved (load all 5 into Power BI Desktop):
  powerbi_kpi_summary.csv         → KPI cards (8 metrics)
  powerbi_monthly_trend.csv       → Line/bar trend charts
  powerbi_delay_by_mode.csv       → Delay by shipping mode
  powerbi_delay_by_category.csv   → Delay by category
  powerbi_forecast_vs_actual.csv  → Forecast vs actual + stockout risk
  powerbi_region_heatmap.csv      → Region x mode heatmap

What to say in interview:
  "I built a Random Forest + XGBoost forecasting pipeline on
   {len(model_df)} weeks of supply chain data, achieving {best['Accuracy']:.0f}%
   forecast accuracy using {len(feature_cols)} engineered features including
   lag variables, rolling windows, and business KPIs like stockout risk
   and order variance."
""")