# 💊 Pfizer GCS - Supply Chain Operations Analytics

**Clinical Trial Packaging · Plan-Make-Source-Deliver · Operational Intelligence**

> Built as part of interview preparation for the Pfizer Operational Data Analyst Co-op (Global Clinical Supply, Groton CT).  
> Demonstrates end-to-end data analytics engineering: data cleaning → feature engineering → ML modeling → interactive dashboard.

---
## 📊 Live Dashboard Preview

### Delay by Shipping Mode
![Delay by Shipping Mode](chart1_delay_by_shipping_mode.png)

### Delay Trend Over Time
![Delay Trend Over Time](chart2_delay_trend_over_time.png)

### Top Late Categories
![Top Late Categories](chart3_top_late_categories.png)

### Anomaly Detection
![Anomaly Detection](chart4_anomaly_detection.png)

### Demand Forecast
![Demand Forecast](chart5_demand_forecast.png)

### KPI Summary Dashboard
![KPI Summary](chart6_kpi_summary.png)

### ML Forecast Comparison
![ML Forecast Comparison](chart7_ml_forecast_comparison.png)

### Feature Importance
![Feature Importance](chart8_feature_importance.png)
## 🎯 Project Overview

This project analyzes **180,519 real supply chain orders** (DataCo Smart Supply Chain dataset) to surface operational insights directly relevant to clinical trial packaging operations:

- **57.3% late delivery rate** - systemic, not seasonal
- **First Class shipping = 100% late** - counterintuitive key risk
- **5% anomaly rate** flagged by ML model - high-risk orders for ops review
- **Sharp demand drop in late 2017** - caught early by forecasting pipeline
- **14.7 day total lead time** across 6 packaging process steps

---

## 🗂️ Project Structure

```
SupplyChain/
│
├── data/
│   ├── DataCoSupplyChainDataset.csv       # Raw dataset (180,519 rows)
│   ├── supply_chain_clean.csv             # Cleaned dataset (Step 1 output)
│   └── forecast_results.csv              # ML forecast output
│
├── step1_load_explore.py                  # Data loading, cleaning, EDA
├── step2_analysis.py                      # Deep analysis + 6 charts
├── step3_dashboard.py                     # Interactive Streamlit dashboard
├── step4_fixed_forecast.py               # RF + XGBoost ML pipeline
│
├── charts/
│   ├── chart1_delay_by_shipping_mode.png
│   ├── chart2_delay_trend_over_time.png
│   ├── chart3_top_late_categories.png
│   ├── chart4_anomaly_detection.png
│   ├── chart5_demand_forecast.png
│   ├── chart6_kpi_summary.png
│   ├── chart7_ml_forecast_comparison.png
│   └── chart8_feature_importance.png
│
├── powerbi/
│   ├── powerbi_kpi_summary.csv
│   ├── powerbi_monthly_trend.csv
│   ├── powerbi_delay_by_mode.csv
│   ├── powerbi_delay_by_category.csv
│   ├── powerbi_forecast_vs_actual.csv
│   └── powerbi_region_heatmap.csv
│
└── README.md
```

---

## 🔧 Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.13 |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn (Isolation Forest, Linear Regression), XGBoost, Random Forest |
| Visualization | Matplotlib, Seaborn |
| Dashboard | Streamlit |
| Statistical Methods | Anomaly detection, time-series forecasting, hypothesis testing |
| Feature Engineering | Lag features, rolling windows, calendar features, business KPIs |

---

## 📁 Dataset

**DataCo Smart Supply Chain Dataset** - Kaggle  
- 180,519 orders · 53 columns · 2015–2018  
- Source: [kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain](https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain)  
- Covers: order management, shipping, delivery status, product categories, regions

> Note: Raw CSV not included in repo due to file size. Download from Kaggle and place in root folder.

---

## ⚙️ How to Run

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/pfizer-supply-chain-analytics.git
cd pfizer-supply-chain-analytics
```

### 2. Install dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost streamlit
```

### 3. Download dataset
Download `DataCoSupplyChainDataset.csv` from Kaggle and place in root folder.

### 4. Run pipeline in order
```bash
# Step 1 - Clean data
python step1_load_explore.py

# Step 2 - Generate analysis charts
python step2_analysis.py

# Step 3 - Launch interactive dashboard
python -m streamlit run step3_dashboard.py

# Step 4 - ML forecasting pipeline
python step4_fixed_forecast.py
```

---

## 📈 Key Findings

### 1. Late Delivery Rate is Systemic
- **57.3%** of all 180,519 orders delivered late
- Consistent across all months (2015–2018) - not seasonal
- Indicates a **process-level issue**, not a one-off disruption

### 2. First Class Shipping = Highest Risk
- **100% late delivery rate** despite being premium shipping
- Root cause: over-scheduling without capacity alignment
- Standard Class performs best at **39.8% late** across 107K orders

### 3. Anomaly Detection - 5% High-Risk Orders
- Isolation Forest ML model trained on 6 operational features
- Flags orders with unusual delay + sales + quantity combinations
- Adjustable sensitivity slider for ops manager use

### 4. Value Stream Map - 14.7 Day Lead Time
| Step | Cycle Time | Risk |
|---|---|---|
| Order Planning | 1.2d | LOW |
| Material Sourcing | 2.1d | MEDIUM |
| Packaging Ops | 3.8d | HIGH  |
| QA / Labeling | 2.4d | HIGH  |
| Distribution | 4.2d | HIGH  |
| Site Delivery | 1.0d | MEDIUM |

### 5. Demand Forecasting
- Sharp volume drop in late 2017 (11K → 2K units/month)
- 26 engineered features: lag variables, rolling windows, stockout risk scores
- Random Forest + XGBoost pipeline with time-series cross-validation

---

## 🏭 Relevance to Pfizer GCS Role

| JD Requirement | Built |
|---|---|
| Operational data collection & analytics |  180K orders analyzed |
| Visual operations production boards |  Streamlit dashboard with live filters |
| Pack job tracking (daily/weekly) |  Weekly Pack Job Tracking Board |
| Value stream mapping |  Full VSM - Plan→Make→Source→Deliver |
| Data-driven decision support |  8 KPIs + anomaly flags + forecasts |
| Supply chain planning methods |  Delay analysis, demand forecasting |
| Statistical analysis & data mining |  Isolation Forest, regression, feature engineering |
| Process variability reduction |  Cycle time analysis per VSM step |

---

## 👩‍💻 Author

**Vaishnavi Mallikarjun Gajarla**  
MS Data Analytics Engineering · Northeastern University · Boston, MA  
[LinkedIn](https://linkedin.com/in/YOUR_LINKEDIN) · [GitHub](https://github.com/YOUR_USERNAME) · gajarla.v@northeastern.edu

---
