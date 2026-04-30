# 🧹 Data Preprocessing & Feature Engineering

> A complete end-to-end data preprocessing pipeline applied to a customer purchase dataset — preparing raw, messy data into a clean, ML-ready format.

---

## 📌 Project Overview

This project demonstrates a full data preprocessing workflow covering data ingestion from multiple sources, exploratory analysis, cleaning, transformation, encoding, scaling, and feature engineering. The final output is a machine-learning-ready dataset suitable for binary classification tasks such as **customer churn prediction**.

**Problem Type:** Binary Classification  
**Target Variable:** `Churn` (0 = No, 1 = Yes)  
**Goal:** Predict whether a customer will churn based on purchase behaviour and demographic features.

---

## 📂 Project Structure

```
├── Data_Preprocessing.ipynb       # Main Jupyter Notebook
├── customer.csv                   # Customer demographic data
├── transactions.json              # Transaction records
├── products.db / products.sql     # Product data (SQLite)
├── processed_customer_data.csv    # Final cleaned & engineered dataset
└── eda_report.html                # Auto-generated EDA report (ydata-profiling)
```

---

## 🔄 Workflow / Steps

### Step 1 — Problem Definition & Plan
- Defined the data science project lifecycle (9 stages: Problem → Deployment → Monitoring)
- Framed the dataset as a **binary classification** problem
- Identified key input features: `age`, `gender`, `income`, `purchase_amount`

---

### Step 2 — Data Import & Understanding
Data was loaded from **4 different sources** and merged into a single unified dataset:

| Source | Format | Library Used |
|--------|--------|--------------|
| Customer data | CSV | `pandas.read_csv` |
| Transactions | JSON | `pandas.read_json` |
| Products | SQL (SQLite) | `sqlite3` + `pd.read_sql` |
| Users | REST API | `requests` |

All four sources were merged using `customer_id` and `product_id` as join keys.

---

### Step 3 — Exploratory Data Analysis (EDA)
- **Univariate Analysis** — Histograms and skewness check for `age`, `income`, `amount`
- **Bivariate Analysis** — Correlation matrix, boxplots (Income vs Purchase, Age vs Purchase)
- **Multivariate Analysis** — Heatmap, pairplot, and city-level group aggregations
- Auto-generated a full EDA report using `ydata-profiling` → saved as `eda_report.html`

---

### Step 4 — Handling Missing Data

| Technique | Applied To |
|-----------|-----------|
| Simple Imputer (mean) | Numerical columns: `age`, `income`, `amount` |
| Simple Imputer (most frequent) | Categorical columns: `gender`, `city`, `payment_mode` |
| Mode-based filling | `city` column |
| Missing Indicator | `income` — tracked which rows were originally missing |
| Random Sample Imputation | `income` — realistic replacement |
| KNN Imputer (`k=5`) | Numerical columns |
| MICE / Iterative Imputer | Numerical columns (most accurate) |
| Complete Case Analysis | Dropped rows with missing values (for comparison) |

> ✅ **Best Result:** KNN and MICE outperformed simple imputation by leveraging inter-feature relationships.

---

### Step 5 — Outlier Detection & Handling

| Method | Description |
|--------|-------------|
| Z-Score | Flags values where \|Z\| > 3 |
| IQR | Flags values outside Q1 − 1.5×IQR and Q3 + 1.5×IQR |
| Percentile | Uses 1st and 99th percentile as thresholds |
| Winsorization | Caps extreme values at percentile limits |

> ✅ **Best Result:** IQR method was more reliable for skewed columns like `income` and `amount`. Z-score assumes normality and over-flagged legitimate high values.

---

### Step 6 — Handling Mixed & Date/Time Variables
- Converted `date` column to `datetime` format
- Engineered `days_since_last_purchase` feature (recency signal)
- Extracted numeric parts from mixed-format IDs (`transaction_id`, `product_id`) using regex

---

### Step 7 — Encoding Categorical Variables

| Encoding Type | Applied To | Reason |
|--------------|-----------|--------|
| Label Encoding | `gender` | Binary nominal column |
| One-Hot Encoding | `city`, `fav_category` | Nominal (no order) |
| Ordinal Encoding | `income_level` (Low/Medium/High) | Ranked categories |
| Binning | `income` → groups | Discretization of continuous values |

> ✅ **Best Practice:** One-Hot for nominal data (no ordering), Ordinal for ranked data (preserve order).

---

### Step 8 — Feature Scaling

| Scaler | Behaviour |
|--------|-----------|
| `StandardScaler` | Centres to mean=0, std=1 |
| `MinMaxScaler` | Scales to [0, 1] range |
| `MaxAbsScaler` | Scales by max absolute value |
| `RobustScaler` | Uses median and IQR — resistant to outliers |
| `Normalizer` | Scales each row to unit norm |

Also used `ColumnTransformer` to apply different scalers to different columns in a single pipeline step.

> ✅ **Best Result:** `RobustScaler` worked best since the dataset contained outliers in `income` and `amount`.

---

### Step 9 — Feature Construction & Transformation

**Engineered Features:**
- `purchase_per_day` — Spending intensity relative to recency (`amount / (days_since + 1)`)
- `income_log` — Log-transformed income to reduce skewness
- `amount_log` — Log-transformed transaction amount
- `income_power` — Yeo-Johnson power transformation for normality
- `amount_power` — Yeo-Johnson power transformation
- `income_bin` — Binned into Low / Medium / High / Very High
- `frequent_buyer` — Binary flag (1 if spending > average)

---

### Step 10 — Final Output
- Exported cleaned dataset as `processed_customer_data.csv`
- Verified shape, column list, missing values, and summary statistics
- Dataset confirmed ML-ready

---

## 🛠️ Tech Stack

| Category | Libraries |
|----------|-----------|
| Data Manipulation | `pandas`, `numpy` |
| Visualisation | `matplotlib`, `seaborn` |
| EDA | `ydata-profiling` |
| Preprocessing | `scikit-learn` |
| Database | `sqlite3` |
| API | `requests` |
| Statistics | `scipy` |

---

## ⚙️ Installation & Setup

```bash
# Clone the repository
git clone https://github.com/your-username/data-preprocessing.git
cd data-preprocessing

# Install required libraries
pip install pandas numpy matplotlib seaborn scikit-learn scipy ydata-profiling requests
```

Then open the notebook:

```bash
jupyter notebook Data_Preprocessing.ipynb
```

---

## 📊 Key Findings

- **Best Imputation:** MICE and KNN imputation produced more accurate fills than simple mean/mode imputation
- **Best Outlier Method:** IQR was more reliable than Z-score for right-skewed distributions
- **Best Scaler:** `RobustScaler` handled outlier-heavy columns most effectively
- **Best Encoding:** One-Hot Encoding for nominal, Ordinal Encoding for ranked categories
- **Key Engineered Features:** `purchase_per_day`, `frequent_buyer`, log/power transforms significantly improved data quality for ML models

---

## 👤 Author

**Arnob Maity**  
Data Science Project — Data Preprocessing & Feature Engineering  

---

## 📄 License

This project is for educational purposes.
