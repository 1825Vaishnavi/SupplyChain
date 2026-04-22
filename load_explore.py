# ============================================================
# PFIZER GCS PROJECT — STEP 1: Load, Explore & Clean Data
# Vaishnavi Mallikarjungajarla | Northeastern University
# Dataset: DataCo Smart Supply Chain (Kaggle)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ── 1. LOAD DATA ─────────────────────────────────────────────
# Update this path to where your file is saved
df = pd.read_csv(
    'DataCoSupplyChainDataset.csv',
    encoding='latin-1'   # important! this file needs latin-1 encoding
)

print("=" * 60)
print("DATACO SUPPLY CHAIN — INITIAL EXPLORATION")
print("=" * 60)

# ── 2. BASIC SHAPE ───────────────────────────────────────────
print(f"\nRows    : {df.shape[0]:,}")
print(f"Columns : {df.shape[1]}")

# ── 3. COLUMN NAMES ──────────────────────────────────────────
print("\nAll columns:")
for col in df.columns:
    print(f"  {col}")

# ── 4. DATA TYPES ────────────────────────────────────────────
print("\nData types:")
print(df.dtypes)

# ── 5. MISSING VALUES ────────────────────────────────────────
print("\nMissing values per column:")
missing = df.isnull().sum()
missing = missing[missing > 0]
print(missing)

# ── 6. KEY COLUMNS WE WILL USE ───────────────────────────────
# These are the columns most relevant to Pfizer GCS ops analysis
key_cols = [
    'Type',                        # transaction type
    'Days for shipping (real)',     # actual shipping days
    'Days for shipment (scheduled)',# scheduled shipping days
    'Delivery Status',             # on time / late / etc
    'Late_delivery_risk',          # 0 or 1 flag
    'Category Name',               # product category
    'Customer Segment',            # segment
    'Order Item Quantity',         # order qty
    'Sales',                       # revenue
    'Order Profit Per Order',      # profit
    'Shipping Mode',               # standard / first class / etc
    'Order Region',                # geography
    'order date (DateOrders)',     # order date
]

print("\nKey columns preview:")
available = [c for c in key_cols if c in df.columns]
print(df[available].head(5).to_string())

# ── 7. CLEAN: RENAME COLUMNS (remove spaces) ─────────────────
df.columns = (
    df.columns
    .str.strip()
    .str.replace(' ', '_')
    .str.replace('(', '')
    .str.replace(')', '')
    .str.replace('/', '_')
    .str.lower()
)

print("\nCleaned column names (first 20):")
print(list(df.columns[:20]))

# ── 8. CLEAN: PARSE DATES ────────────────────────────────────
date_cols = [c for c in df.columns if 'date' in c]
print(f"\nDate columns found: {date_cols}")

for col in date_cols:
    try:
        df[col] = pd.to_datetime(df[col], errors='coerce')
        print(f"  Parsed: {col}")
    except:
        print(f"  Skipped: {col}")

# ── 9. CLEAN: DROP USELESS COLUMNS ───────────────────────────
# Customer email, password, zip — not needed for ops analysis
drop_cols = [c for c in df.columns if any(
    x in c for x in ['password', 'email', 'zipcode', 'street', 'product_image']
)]
df.drop(columns=drop_cols, inplace=True, errors='ignore')
print(f"\nDropped {len(drop_cols)} PII/irrelevant columns: {drop_cols}")

# ── 10. CREATE KEY METRICS ───────────────────────────────────
# Delay = actual days - scheduled days (positive = late)
if 'days_for_shipping_real' in df.columns and 'days_for_shipment_scheduled' in df.columns:
    df['delay_days'] = df['days_for_shipping_real'] - df['days_for_shipment_scheduled']
    df['is_delayed'] = (df['delay_days'] > 0).astype(int)
    print("\nCreated: delay_days, is_delayed")

# ── 11. SUMMARY STATS ────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY STATISTICS — KEY METRICS")
print("=" * 60)

if 'delay_days' in df.columns:
    print(f"\nDelay days stats:")
    print(df['delay_days'].describe().round(2))

if 'is_delayed' in df.columns:
    delay_rate = df['is_delayed'].mean() * 100
    print(f"\nOverall delay rate: {delay_rate:.1f}%")

if 'delivery_status' in df.columns:
    print(f"\nDelivery status breakdown:")
    print(df['delivery_status'].value_counts())

if 'shipping_mode' in df.columns:
    print(f"\nShipping mode breakdown:")
    print(df['shipping_mode'].value_counts())

# ── 12. SAVE CLEANED DATA ────────────────────────────────────
df.to_csv('supply_chain_clean.csv', index=False)
print("\nCleaned data saved to: supply_chain_clean.csv")
print(f"Final shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
print("\nStep 1 COMPLETE. Run step2_analysis.py next.")