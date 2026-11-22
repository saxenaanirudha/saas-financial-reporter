# SaaS Financial Reporter 📊

[![CI — SaaS Financial Reporter](https://github.com/saxenaanirudha/saas-financial-reporter/actions/workflows/ci.yml/badge.svg)](https://github.com/saxenaanirudha/saas-financial-reporter/actions/workflows/ci.yml)

An AI-assisted SaaS financial reporting and customer analytics project featuring KPI analysis, cash-flow forecasting, automated reporting, an interactive Streamlit dashboard, and ML-based churn risk prediction.

> **Note on Datasets**: All financial and customer records used in this project are synthetic and illustrative, designed to simulate realistic SaaS business dynamics without using proprietary data.

---

## ⚡ Quick Start

```powershell
# 1. Run the automated analytics pipeline (computes KPIs, forecasts, PDF report, ML model)
python main.py

# 2. Launch the interactive browser dashboard
streamlit run app.py
```

---

## 🖥️ Dashboard Preview

The project includes an interactive Streamlit web dashboard providing a non-technical presentation layer for executives, investors, and team members to explore financial performance, cash trajectories, and customer risk scoring.

![SaaS Financial Reporter Executive Dashboard](docs/images/saas-executive-overview.png)

### Cash Flow & Forecast
![Cash Flow Forecast Dashboard](docs/images/saas-cashflow.png)

### Customer Churn Intelligence
![Customer Churn Intelligence Dashboard](docs/images/saas-customer-churn-intelligence.png)

---

## 🌟 Key Highlights

- **MRR and ARR Analytics**: Computes Monthly Recurring Revenue (MRR) and Annualized Recurring Revenue (ARR) run-rates.
- **MRR Growth Analysis**: Measures Month-over-Month (MoM) revenue growth rates and net revenue expansions.
- **Customer Acquisition Cost (CAC)**: Tracks marketing spend efficiency per newly acquired customer.
- **Customer Churn & Retention Analysis**: Reconstructs customer counts chronologically to evaluate monthly subscriber churn and retention rates.
- **Historical Operating Cash-Flow Analysis**: Tracks monthly cash inflows against categorized operating expenses and rolls forward cash balances *(cash flow tracking, not GAAP accounting profit)*.
- **6-Month Trend-Based Cash-Flow Forecasting**: Projects 6-month forward cash trajectories using component-by-component OLS linear regression *(trend-based projection, not guaranteed financial predictions)*.
- **Automated Management/Investor PDF Reporting**: Generates a multi-page, publication-grade executive PDF report with dynamic KPI tables and 10 embedded high-resolution charts via ReportLab.
- **Streamlit Browser Dashboard**: Interactive presentation interface (`streamlit run app.py`) for non-technical stakeholders to explore financial KPIs, cash-flow forecasts, churn intelligence, reports, and project information.
- **Customer-Level Churn Prediction using Logistic Regression**: End-to-end Scikit-Learn classification pipeline trained on separate customer behavioral data *(illustrative synthetic dataset; not claimed as production performance)*.
- **Low / Medium / High Churn-Risk Classification**: Categorizes customer churn probabilities into actionable risk tiers for targeted retention intervention.
- **Automated Test Suite**: 203 automated unit, integration, concurrency, and security regression tests verifying formulas, data integrity, thread safety, and error handling.
- **Modular Pipeline Architecture**: Clean, decoupled architecture with automated directory initialization and artifact validation.

---

## 📖 Table of Contents
1. [Overview](#-overview)
2. [Business Problem](#-business-problem)
3. [Key Features](#-key-features)
4. [Interactive Streamlit Dashboard](#-interactive-streamlit-dashboard)
5. [Architecture](#-architecture)
6. [Project Structure](#-project-structure)
7. [Financial Dataset](#-financial-dataset)
8. [Customer Churn Dataset](#-customer-churn-dataset)
9. [Financial KPI Logic & Formulas](#-financial-kpi-logic--formulas)
10. [Historical Cash Flow Analysis](#-historical-cash-flow-analysis)
11. [Six-Month Cash Flow Forecasting](#-six-month-cash-flow-forecasting)
12. [Customer Churn Machine Learning](#-customer-churn-machine-learning)
13. [Generated Outputs & Artifacts](#-generated-outputs--artifacts)
14. [Installation & Setup](#-installation--setup)
15. [Running the Project](#-running-the-project)
16. [Running Tests](#-running-tests)
17. [Security & Quality Assurance — Barn Door Protocol](#-security--quality-assurance--barn-door-protocol)
18. [Example Workflow](#-example-workflow)
19. [Assumptions & Limitations](#-assumptions--limitations)
20. [AI-Assisted Development Approach](#-ai-assisted-development-approach)
21. [Skills Demonstrated](#-skills-demonstrated)
22. [Future Improvements](#-future-improvements)
23. [Disclaimer](#-disclaimer)

---

## 🎯 Overview

The **SaaS Financial Reporter** automates the periodic financial analysis and reporting workflow for a subscription-based business model. Built in Python, it integrates:
- **Core SaaS Metrics**: Monthly Recurring Revenue (MRR), Annual Recurring Revenue (ARR), MoM Revenue Growth, and Customer Acquisition Cost (CAC).
- **Customer Base Dynamics**: Roll-forward customer ledger reconstruction, monthly churn rates, and retention percentages.
- **Operating Cash Flow**: Inflow vs. multi-category outflow tracking, monthly net flow, and cumulative balance roll-forward.
- **Trend-Based Forecasting**: 6-month statistical forecasting of revenue and individual expense components using linear regression.
- **Automated Investor Reporting**: Generation of presentation-grade, multi-page PDF executive summaries with embedded high-resolution charts.
- **Interactive Web Interface**: A Streamlit browser application designed for non-technical stakeholders to interactively inspect financial performance and ML churn scoring.
- **Predictive Churn Risk Modeling**: A customer-level Machine Learning classification pipeline (Logistic Regression) that estimates churn probabilities and classifies accounts into actionable risk tiers.

---

## 💼 Business Problem

In early-stage and growing SaaS companies, monthly financial and customer reporting is frequently performed manually across disconnected spreadsheets. This manual workflow suffers from:
- **Repetitive Calculation Overhead**: Re-calculating MRR, ARR, CAC, and subscriber counts month after month.
- **Reporting Disconnect**: Financial metrics (revenue and cash) are rarely bridged with operational data (churn and usage).
- **Limited Forward Visibility**: Cash runways and expense trends are often projected using crude manual estimates.
- **Delayed Retention Intervention**: High-risk accounts are identified only after cancellations have already occurred.

This project demonstrates how a modular Python data pipeline can automate data validation, financial calculations, forecasting, chart rendering, PDF publication, interactive exploration, and predictive churn risk scoring in a unified, reproducible run.

> *Note: This project demonstrates reporting workflow automation and is not a substitute for audited accounting or professional financial advisory services.*

---

## 🚀 Key Features

### 1. Financial Performance & Revenue
- **MRR & ARR Tracking**: Computes monthly recurring revenue and annualized revenue run-rate.
- **Month-over-Month Growth**: Calculates MoM percentage expansion and net revenue additions.
- **Customer Acquisition Cost (CAC)**: Measures acquisition spend efficiency per new subscriber.

### 2. Subscriber Base & Retention Dynamics
- **Customer Base Roll-Forward**: Reconstructs opening, new, churned, and closing customer counts.
- **Churn & Retention Rates**: Calculates monthly subscriber churn rate % and retention rate % (safely handling initial baseline limitations).

### 3. Historical Cash Flow Analysis
- **Cash Inflow vs. Outflow**: Maps MRR inflows against marketing, operating, payroll, and administrative expenses.
- **Rolling Cash Ledger**: Computes monthly net cash flow and rolls forward closing cash balances.

### 4. Six-Month Cash Flow Forecasting
- **Component-by-Component OLS Regression**: Fits independent linear trend models on revenue and each expense line.
- **Non-Negative Clamping**: Revenue and expense lines are bounded at $\ge 0$ while allowing net cash flow and balances to fluctuate realistically.
- **Runway & Cash Trajectory**: Projects 6 months of forward closing balances and net cash burn.

### 5. Automated Management & Investor PDF Report
- **Multi-Page Executive Report**: Generates a publication-grade PDF using ReportLab with dynamic timestamps, KPI snapshot grids, financial statements, and 10 embedded high-resolution charts.

### 6. Interactive Presentation Dashboard (Streamlit)
- **Executive KPI Dashboard**: Live interactive cards, metric deltas, and responsive financial charts.
- **Cash Flow & Forecast Explorer**: Visual breakdown of historical cash flows, forward runway projections, and monthly expense structures.
- **Customer Churn Intelligence**: Real-time account risk filtering, feature importance explanations, and interactive risk tier distributions.
- **Reports & Artifact Viewer**: In-browser PDF report download, dataset inspectors, and system architecture summaries.

### 7. Customer-Level Churn Machine Learning Pipeline
- **Separate Behavioral Dataset**: Evaluates 500 customer records without contaminating aggregated financial data.
- **Scikit-Learn Pipeline**: Integrates `StandardScaler` (numerical), `OneHotEncoder` (categorical), and `LogisticRegression`.
- **Anti-Leakage Safeguards**: Preprocessing is fit strictly on training splits (`X_train`); identifiers and post-cancellation fields are excluded.
- **Actionable Risk Tiers**: Categorizes predicted probabilities into **Low** ($<0.30$), **Medium** ($0.30–0.60$), and **High** ($\ge 0.60$) risk segments.
- **Feature Importance**: Extracts learned coefficients and odds ratios to identify top churn risk drivers.

### 8. Reliability & Execution Orchestration
- **Pipeline Coordinator (`src/pipeline.py`)**: Drives end-to-end execution with decoupled stages.
- **Severity-Aware Error Handling**: Critical errors halt execution safely; non-critical failures log warnings and produce `PARTIAL_SUCCESS`.
- **Automatic Directory & Artifact Validation**: Auto-creates `charts/`, `reports/`, `models/` and verifies on-disk output integrity.

---

## 🖥️ Interactive Streamlit Dashboard

To provide a non-technical, client-friendly presentation layer, the repository includes a Streamlit browser dashboard:

```powershell
streamlit run app.py
```

### Dashboard Capabilities:
- **Executive Overview**: High-level financial KPIs (MRR, ARR, CAC, churn rate, cash balance) with interactive month-over-month trend charts.
- **Cash Flow & Forecast**: Interactive breakdown of historical operating expenses, net cash flow, and 6-month linear forward projections.
- **Customer Churn Intelligence**: Exploration of customer churn risk tiers (Low, Medium, High), search and filtering across individual account profiles, and feature importance drivers.
- **Reports & Data Viewer**: In-browser inspection of raw datasets, generated metrics, and one-click download of the generated executive PDF report.
- **Architecture & Workflow**: Visual documentation of data flow, modeling assumptions, and engineering architecture.

---

## 🏛️ Architecture

```text
Monthly Financial Dataset (CSV)                 Customer-Level Churn Dataset (CSV)
             │                                                  │
             ▼                                                  ▼
     [ Data Ingestion ]                                 [ Data Ingestion ]
    (Schema & Range Check)                             (Schema & Target Check)
             │                                                  │
             ├──────────────────────┐                           ▼
             ▼                      ▼                   [ Stratified Split ]
   [ Revenue & CAC Engine ]   [ Cash Flow Engine ]      (80% Train / 20% Test)
             │                      │                           │
             ├──────────────────────┘                           ▼
             ▼                                          [ Pipeline Preprocessing ]
   [ Churn & Retention ]                               (StandardScaler + OneHotEncoder)
             │                                                  │
             ▼                                                  ▼
   [ Historical Comparison ]                            [ Logistic Regression ]
    (MoM & Baseline Trends)                            (Fit on Training Data Only)
             │                                                  │
             ▼                                                  ▼
   [ 10 Financial Charts ]                             [ Test Set Evaluation ]
   (Matplotlib Agg Engine)                             (Accuracy, F1, ROC-AUC, CM)
             │                                                  │
             ▼                                                  ▼
   [ 6-Month OLS Forecast ]                             [ Probability Scoring ]
   (Component Linear Trends)                           (Low / Med / High Risk Tiers)
             │                                                  │
             ▼                                                  ▼
   [ PDF Investor Report ]                             [ Churn Risk CSV & Charts ]
   (ReportLab Master File)                             (customer_churn_risk.csv)
             │                                                  │
             ├──────────────────────────────────────────────────┤
             ▼                                                  ▼
    [ Console Dashboard ]                              [ Streamlit Dashboard ]
   (CLI Summary Pipeline)                             (Browser Presentation Layer)
             │                                                  │
             └──────────────────────┬───────────────────────────┘
                                    │
                                    ▼
                     [ Pipeline Execution Summary ]
                    (Artifact Validation & Status)
```

---

## 📁 Project Structure

```text
saas-financial-reporter/
│
├── .github/
│   └── workflows/
│       └── ci.yml                     # Multi-version GitHub Actions CI/CD workflow
│
├── data/
│   ├── saas_financial_data.csv        # Aggregated monthly financial data (6 months)
│   ├── customer_churn_data.csv        # Synthetic customer-level churn data (500 records)
│   └── generate_churn_data.py         # Deterministic customer churn data generator
│
├── docs/
│   └── images/                        # Documentation and preview screenshots
│       ├── saas-executive-overview.png
│       ├── saas-cashflow.png
│       └── saas-customer-churn-intelligence.png
│
├── src/                               # Core application modules
│   ├── __init__.py
│   ├── config.py                      # Centralized path definitions and constants
│   ├── exceptions.py                  # Domain exception classes
│   ├── pipeline.py                    # Pipeline coordinator and orchestration engine
│   ├── data_loader.py                 # Financial data loading and validation
│   ├── revenue_dashboard.py           # MRR, ARR, and revenue growth calculations
│   ├── cac_calculator.py              # Customer Acquisition Cost calculations
│   ├── churn_retention_analyzer.py    # Customer roll-forward, churn & retention metrics
│   ├── chart_generator.py             # Financial KPI and forecast chart generator
│   ├── kpi_comparator.py              # Historical period-over-period KPI comparisons
│   ├── cash_flow_analyzer.py          # Operating cash flow and rolling balance ledger
│   ├── cash_flow_forecaster.py        # 6-Month linear cash flow forecasting engine
│   ├── pdf_report_generator.py        # Automated investor/management PDF report
│   ├── reporting_dashboard.py         # Consolidated console reporting dashboard
│   └── churn_model.py                 # Logistic regression churn ML model & risk scoring
│
├── models/                            # Model output artifacts
│   ├── .gitkeep                       # Folder tracking
│   ├── churn_model.joblib             # Serialized trained scikit-learn pipeline (generated at runtime; not committed)
│   └── churn_model_metrics.json       # Serialized test evaluation metrics (generated at runtime; not committed)
│
├── charts/                            # Visual chart outputs
│   ├── .gitkeep                       # Folder tracking
│   ├── mrr_trend.png                  # MRR monthly trajectory (generated at runtime; not committed)
│   ├── arr_trend.png                  # ARR annualized trajectory (generated at runtime; not committed)
│   ├── cac_trend.png                  # Customer Acquisition Cost trajectory (generated at runtime; not committed)
│   ├── churn_retention_trend.png      # Churn & retention rate comparison (generated at runtime; not committed)
│   ├── customer_base_growth.png       # Active subscriber base growth (generated at runtime; not committed)
│   ├── mrr_growth_rate.png            # MoM MRR growth rate percentage (generated at runtime; not committed)
│   ├── net_cash_flow_trend.png        # Monthly net cash flow bars (generated at runtime; not committed)
│   ├── cash_balance_trend.png         # Closing cash balance trajectory (generated at runtime; not committed)
│   ├── cash_flow_forecast.png         # Historical vs 6-month net cash flow forecast (generated at runtime; not committed)
│   ├── cash_balance_forecast.png      # Historical vs 6-month cash balance forecast (generated at runtime; not committed)
│   ├── churn_confusion_matrix.png     # Churn ML confusion matrix heatmap (generated at runtime; not committed)
│   └── churn_roc_curve.png            # Churn ML ROC-AUC curve (generated at runtime; not committed)
│
├── reports/                           # Executive reports
│   ├── .gitkeep                       # Folder tracking
│   ├── saas_financial_report.pdf      # Master management/investor PDF report (generated at runtime; not committed)
│   └── customer_churn_risk.csv        # Scored customer accounts with risk tiers (generated at runtime; not committed)
│
├── tests/                             # Unit & Integration test suite (180 passing tests)
│   ├── __init__.py
│   ├── test_data_loader.py
│   ├── test_revenue_dashboard.py
│   ├── test_cac_calculator.py
│   ├── test_churn_retention_analyzer.py
│   ├── test_chart_generator.py
│   ├── test_kpi_comparator.py
│   ├── test_cash_flow_analyzer.py
│   ├── test_cash_flow_forecaster.py
│   ├── test_pdf_report_generator.py
│   ├── test_reporting_dashboard.py
│   ├── test_churn_model.py
│   ├── test_pipeline.py
│   └── test_end_to_end_regression.py  # End-to-end mathematical regression tests
│
├── app.py                             # Interactive Streamlit browser dashboard
├── main.py                            # Automated pipeline CLI entry point
├── requirements.txt                   # Minimal project dependencies
├── .gitignore                         # Git ignore configuration (excludes runtime artifacts)
├── PROJECT_SUMMARY.md                 # Project architecture & quick reference
├── GITHUB_DESCRIPTION.txt             # Repository description & topic tags
├── RESUME_PROJECT_ENTRY.md            # Resume & portfolio snippets
├── INTERVIEW_QA.md                    # 20 common technical & domain interview Q&As
└── README.md                          # Comprehensive technical documentation
```

---

## 📊 Financial Dataset

The financial pipeline reads from `data/saas_financial_data.csv`. This dataset represents 6 consecutive months of operating metrics for an illustrative SaaS business:

| Column | Type | Description | Example |
| :--- | :--- | :--- | :--- |
| `Month` | String | Calendar month label (`%b-%Y`) | `Jan-2025` |
| `NewCustomers` | Integer | New subscriber additions acquired during the month | `105` |
| `ChurnedCustomers` | Integer | Existing subscriber accounts cancelled during the month | `0` |
| `MRR` | Float | Total Monthly Recurring Revenue recognized | `60000.0` |
| `MarketingSpend` | Float | Marketing & customer acquisition expenditure | `15000.0` |
| `OperatingExpenses` | Float | Software tools, cloud hosting, and infrastructure costs | `12000.0` |
| `PayrollExpenses` | Float | Team salaries, contractor fees, and compensation | `20000.0` |
| `OtherExpenses` | Float | Legal, administrative, and miscellaneous expenses | `3000.0` |

> *All financial records are synthetic illustrative data designed to demonstrate pipeline functionality.*

---

## 👥 Customer Churn Dataset

The Machine Learning classification model reads from `data/customer_churn_data.csv`. This customer-level dataset contains 500 individual account records:

| Feature | Type | Description |
| :--- | :--- | :--- |
| `CustomerID` | String | Unique customer identifier (excluded from training) |
| `SignupDate` | String | Account creation date (excluded from training) |
| `Plan` | Categorical | Subscription plan tier (`Basic`, `Pro`, `Enterprise`) |
| `MonthlyFee` | Float | Contracted monthly recurring fee ($) |
| `TenureMonths` | Integer | Total active tenure in months |
| `LastLoginDaysAgo` | Integer | Recency of last product login (days) |
| `SupportTickets` | Integer | Number of customer support inquiries filed |
| `MonthlyUsageHours`| Float | Total platform engagement time in the past month (hours) |
| `PaymentFailures` | Integer | Number of recurring billing failures encountered |
| `DiscountPercent` | Float | Discount percentage applied to subscription |
| `ContractType` | Categorical | Contract term (`Monthly` vs. `Annual`) |
| `AutoRenew` | Binary | Automatic renewal status (`1` = Active, `0` = Disabled) |
| `Churned` | Binary | Target variable (`1` = Churned, `0` = Active) |

### Why Separate Datasets?
- **Aggregated Financial Data (`data/saas_financial_data.csv`)** operates at the monthly cohort level, providing top-down financial and cash flow visibility.
- **Customer-Level Data (`data/customer_churn_data.csv`)** captures granular behavioral interactions required for individualized risk modeling without creating artificial joins.

---

## 📐 Financial KPI Logic & Formulas

### 1. Monthly Recurring Revenue (MRR) & Annual Recurring Revenue (ARR)
$$\text{MRR} = \text{Monthly Subscription Revenue Recognized}$$
$$\text{ARR} = \text{MRR} \times 12$$

### 2. Month-over-Month MRR Growth Rate
$$\text{MoM Growth } (\%) = \frac{\text{MRR}_t - \text{MRR}_{t-1}}{\text{MRR}_{t-1}} \times 100$$
*(First month growth is explicitly undefined / `NaN` due to lack of a preceding baseline).*

### 3. Customer Acquisition Cost (CAC)
$$\text{CAC} = \frac{\text{Marketing Spend}}{\text{New Customers Acquired}}$$

### 4. Reconstructed Customer Base Roll-Forward
Because monthly aggregated data lacks explicit beginning balance records, the subscriber base is reconstructed chronologically:
- **Month 1 Beginning Customers**: Initialized to `0`.
- **Ending Customers**: $\text{Beginning Customers} + \text{New Customers} - \text{Churned Customers}$
- **Month $N$ Beginning Customers**: $\text{Ending Customers}_{N-1}$

### 5. Customer Churn Rate & Retention Rate
$$\text{Churn Rate } (\%) = \frac{\text{Churned Customers}_t}{\text{Beginning Customers}_t} \times 100$$
$$\text{Retention Rate } (\%) = 100 - \text{Churn Rate } (\%)$$
*(First month churn/retention is recorded as `NaN` because Month 1 beginning customer base is 0).*

---

## 💵 Historical Cash Flow Analysis

The cash flow module models operational liquidity under a transparent simplified cash ledger:

1. **Cash Inflow**: Modeled directly as Monthly Recurring Revenue ($\text{Cash Inflow} = \text{MRR}$).
2. **Total Operating Cash Outflow**:
   $$\text{Total Outflow} = \text{MarketingSpend} + \text{OperatingExpenses} + \text{PayrollExpenses} + \text{OtherExpenses}$$
3. **Net Cash Flow**:
   $$\text{Net Cash Flow} = \text{Cash Inflow} - \text{Total Outflow}$$
4. **Cash Balance Roll-Forward**:
   - $\text{Month 1 Opening Balance} = \$100,000 \text{ (Default configured baseline)}$
   - $\text{Closing Balance}_t = \text{Opening Balance}_t + \text{Net Cash Flow}_t$
   - $\text{Opening Balance}_{t+1} = \text{Closing Balance}_t$

> **Important Distinction**: Operating cash flow represents liquidity movement through the business and is distinct from GAAP accounting profit (which would require accounting for accruals, depreciation, and amortization).

---

## 🔮 Six-Month Cash Flow Forecasting

To estimate future cash burn and runway without opaque assumptions, the forecaster fits independent Ordinary Least Squares (OLS) linear trend models on historical monthly series ($t = 1 \dots N$):

1. **Component Trend Models**:
   - $\widehat{\text{Inflow}}(t) = \beta_0^{\text{in}} + \beta_1^{\text{in}} t$
   - $\widehat{\text{Marketing}}(t) = \beta_0^{\text{mkt}} + \beta_1^{\text{mkt}} t$
   - $\widehat{\text{Operating}}(t) = \beta_0^{\text{ops}} + \beta_1^{\text{ops}} t$
   - $\widehat{\text{Payroll}}(t) = \beta_0^{\text{pay}} + \beta_1^{\text{pay}} t$
   - $\widehat{\text{Other}}(t) = \beta_0^{\text{oth}} + \beta_1^{\text{oth}} t$
2. **Non-Negative Clamping**:
   $$\widehat{\text{Component}}(t) = \max(0, \widehat{\text{Component}}(t))$$
3. **Derived Totals**:
   $$\widehat{\text{Total Outflow}}(t) = \sum \widehat{\text{Expenses}}(t)$$
   $$\widehat{\text{Net Cash Flow}}(t) = \widehat{\text{Inflow}}(t) - \widehat{\text{Total Outflow}}(t)$$
4. **Projected Balance Roll-Forward**:
   The first forecast month opens with the latest historical closing cash balance and rolls forward across the 6 projected months.

> **Forecasting Nature**: These projections represent linear trend extrapolations based on historical trajectory for planning scenarios, not guaranteed future financial outcomes.

---

## 🤖 Customer Churn Machine Learning

### ML Pipeline Workflow
1. **Stratified Split**: 80% Training / 20% Testing (`stratify=y`, `random_state=42`).
2. **ColumnTransformer Preprocessing**:
   - **Numerical Features** (`StandardScaler`): `MonthlyFee`, `TenureMonths`, `LastLoginDaysAgo`, `SupportTickets`, `MonthlyUsageHours`, `PaymentFailures`, `DiscountPercent`, `AutoRenew`.
   - **Categorical Features** (`OneHotEncoder(drop='first', handle_unknown='ignore')`): `Plan`, `ContractType`.
3. **Classifier**: `LogisticRegression(max_iter=1000, random_state=42)`.
4. **Evaluation**: Held-out test set metrics computed with `zero_division=0`:
   - Accuracy, Precision, Recall, F1-Score, ROC-AUC Score, and Confusion Matrix ($TN, FP, FN, TP$).

### Operational Risk Tiers
Predicted probabilities $P(\text{Churn} = 1)$ are mapped to portfolio action tiers:
- **Low Risk ($P < 0.30$)**: Account healthy; standard lifecycle communications.
- **Medium Risk ($0.30 \le P < 0.60$)**: Moderate engagement decline; automated check-ins.
- **High Risk ($P \ge 0.60$)**: Elevated churn probability; prioritized for Customer Success outreach.

### Data Leakage Protections
- Preprocessing transformations are fit strictly on `X_train` inside the scikit-learn `Pipeline`.
- Non-predictive identifiers (`CustomerID`) and timestamp columns (`SignupDate`) are dropped before fitting.

---

## 📁 Generated Outputs & Artifacts

When the pipeline executes, the following verified artifacts are produced (runtime artifacts are ignored by Git via `.gitignore`):

| Output Artifact | Location | Commit Status | Description |
| :--- | :--- | :--- | :--- |
| **PDF Investor Report** | `reports/saas_financial_report.pdf` | *(generated at runtime; not committed)* | Comprehensive multi-page management & investor summary |
| **Customer Risk Scoring CSV** | `reports/customer_churn_risk.csv` | *(generated at runtime; not committed)* | All 500 accounts scored and ranked by churn probability |
| **Serialized Churn Model** | `models/churn_model.joblib` | *(generated at runtime; not committed)* | Trained scikit-learn pipeline binary |
| **Model Evaluation Metrics** | `models/churn_model_metrics.json` | *(generated at runtime; not committed)* | Test set metrics in structured JSON format |
| **Financial KPI Charts (10)** | `charts/*.png` | *(generated at runtime; not committed)* | High-resolution trend plots (MRR, ARR, CAC, Cash Flow, Forecast) |
| **ML Evaluation Charts (2)** | `charts/churn_*.png` | *(generated at runtime; not committed)* | Confusion Matrix heatmap & ROC-AUC curve |

---

## 🛠️ Installation & Setup

### 1. Prerequisites
- **Python 3.9, 3.10, 3.11, or 3.12**
- Git

### 2. Clone the Repository
```powershell
git clone https://github.com/saxenaanirudha/saas-financial-reporter.git
cd saas-financial-reporter
```

### 3. Create & Activate Virtual Environment
**Windows PowerShell:**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install Dependencies
```powershell
pip install -r requirements.txt
```

---

## 🏃 Running the Project

### 1. Run the Automated Analytics & ML Pipeline
To execute the complete end-to-end reporting pipeline and ML workflow:

```powershell
python main.py
```

#### What Happens During Pipeline Execution:
1. Output directories (`charts/`, `reports/`, `models/`) are initialized automatically.
2. Monthly financial data (`data/saas_financial_data.csv`) is validated and loaded.
3. Revenue, CAC, and Churn/Retention analytics are computed.
4. Historical cash flow and 6-month statistical forecasts are generated.
5. High-resolution PNG charts and master PDF report (`reports/saas_financial_report.pdf`) are rendered.
6. Consolidated console dashboard displays core business health.
7. Customer churn dataset (`data/customer_churn_data.csv`) is processed and trained via Logistic Regression.
8. Customer risk report (`reports/customer_churn_risk.csv`) and model binaries are persisted.
9. Final artifact validation verifies all 16 generated files on disk.

### 2. Launch the Interactive Presentation Dashboard
To launch the client-ready Streamlit web application:

```powershell
streamlit run app.py
```

The dashboard will open automatically in your default browser (typically at `http://localhost:8501`), providing a visual, interactive interface to explore all financial KPIs, runway forecasts, customer risk segments, and generated PDF reports.

---

## 🧪 Running Tests

The test suite contains **203 automated unit, integration, and security regression tests** verifying formulas, data validation, chart rendering, PDF generation, ML leakage prevention, CSV formula injection defense, model HMAC provenance signing, thread synchronization, and full pipeline reliability:

```powershell
# Verbose test execution (runs all 203 tests across 16 test modules)
pytest -v

# Quiet summary check
pytest -q
```

---

## 🛡️ Security & Quality Assurance — Barn Door Protocol

The project has undergone rigorous application security auditing and engineering hardening following the **Barn Door Protocol**, achieving a **PASS (RELEASE READY)** verdict with **203 automated tests passing cleanly** across 16 test modules. Full methodology, vulnerability analysis, and behavioral TDD verification evidence are documented in [`BARN_DOOR_QA_REPORT.md`](BARN_DOOR_QA_REPORT.md).

### Key Security & Reliability Mitigations:

- **Spreadsheet Formula Injection Prevention (CWE-1236 / SEC-01 & BLK-02):**
  Defensively sanitizes string exports in `sanitize_dataframe_for_csv()`. Cells starting with formula triggers (`=`, `+`, `-`, `@`, `\t`, `\r`) — including values preceded by whitespace, spaces, tabs, or newlines (`" =cmd|..."`, `"\t+1234"`, `"\n-cmd"`) — are safely prefixed with a single quote (`'`). Already quoted cells (`'`) are preserved without double-escaping, and native numeric columns (`float`, `int`, e.g., negative cash flows `-1500.50`) remain untransformed.
- **HMAC-SHA256 Model Provenance Verification (CWE-502 / SEC-02 & BLK-01):**
  Serialized scikit-learn model artifacts (`churn_model.joblib`) are paired with a cryptographically signed HMAC-SHA256 provenance manifest (`churn_model_provenance.json`). Deserialization verifies file size bounds (50 MB safety cap), path containment (preventing directory traversal attacks), SHA-256 binary hash, and HMAC signature prior to unpickling.
- **Production Key Management & Rotation (BLK-01):**
  Hardcoded fallback signing keys are strictly rejected. In production/strict mode (`require_secret=True`, `SAAS_STRICT_SECURITY=1`, or `SAAS_ENV=production`), a missing `SAAS_MODEL_SECRET` raises `SecurityValidationError` without exposing key bytes in logs. In local development and CI workflows, an ephemeral process-isolated random secret (`os.urandom(32)`) is generated in memory, preventing signature forgery against static keys while enabling test suites to run unhindered.
- **Thread-Safe Matplotlib Rendering (CONC-01 / BLK-03):**
  100% of Matplotlib operations across `src/chart_generator.py`, `src/churn_model.py` (confusion matrix & ROC curve), and `app.py` (risk distribution & feature impact bar charts) are synchronized under `MATPLOTLIB_LOCK` via `@thread_safe_chart`. All `plt.close()` calls are protected within the locked context to prevent cross-session figure bleeding and state corruption during concurrent multi-user Streamlit sessions.
- **Atomic File Persistence (CONC-02 / BLK-04):**
  Model binaries and provenance manifests are staged via temporary files and committed using atomic `os.replace` under `ARTIFACT_PUBLISH_LOCK` (`atomic_publish_model_and_provenance`). Customer risk CSV exports and metrics JSON are written atomically via `atomic_write_bytes()`. ReportLab PDF compilation writes to temporary files before atomic commitment, eliminating torn writes or corrupted outputs during pipeline execution.
- **ReportLab PDF XML Entity Escaping (CWE-116 / SEC-03):**
  All dynamic data, table cells, headers, and narrative summaries injected into ReportLab PDF flowables are sanitized via `safe_xml_escape()` (`&`, `<`, `>`, `"`, `'`), preventing XML parsing crashes.
- **Resource Exhaustion & Denial of Service Protection (CWE-400 / SEC-05):**
  Input datasets are strictly bounded to prevent memory exhaustion (financial data capped at 10,000 rows; churn data capped at 500,000 rows).
- **Sanitized Error Handling (CWE-209 / SEC-04):**
  User-facing UI exception handlers log technical stack traces to server event logs while displaying clean, sanitized error banners to prevent system path disclosure.

### Secure Production Secret Configuration:

To enforce strict cryptographic provenance in production environments, configure `SAAS_MODEL_SECRET` via environment variables or a secret vault:

```bash
# Linux / macOS (bash / zsh)
export SAAS_MODEL_SECRET="<your-strong-random-hex-secret-min-32-chars>"
export SAAS_STRICT_SECURITY=1

# Windows (PowerShell)
$env:SAAS_MODEL_SECRET="<your-strong-random-hex-secret-min-32-chars>"
$env:SAAS_STRICT_SECURITY="1"
```

> **Key Rotation**: To rotate the signing key, update `SAAS_MODEL_SECRET` in your environment, retrain the model via `python main.py`, and re-publish the newly signed provenance manifest. Never commit or log signing keys.

### Verified Protections vs. Documented Limitations:

| Dimension | Verified Protections | Documented Limitations |
| :--- | :--- | :--- |
| **Concurrency Scope** | Multi-threaded safety within a single process via `MATPLOTLIB_LOCK` and `ARTIFACT_PUBLISH_LOCK`. | Multi-process model/manifest publication uses dual `os.replace` calls. While individual file swaps are OS-atomic, mutual transactionality across separate operating system processes requires an external distributed lock. |
| **Application Scope** | Secure internal management reporting and analytical dashboarding. | Single-tenant architecture; not designed for untrusted multi-tenant database partitioning. |
| **Data Scope** | Tested against malicious formula injection, XML payloads, and oversized datasets. | Simplified cash-flow ledger treats MRR as immediate cash inflow without modeling accounts receivable delays. |

---

## 🔄 Example Workflow

1. **Update Financial Data**: Place updated monthly figures into `data/saas_financial_data.csv`.
2. **Execute Pipeline**: Run `python main.py`.
3. **Inspect Console Dashboard**: Review executive revenue summary, MoM KPI movements, and overall status.
4. **Open PDF Report**: Open `reports/saas_financial_report.pdf` for executive or investor review.
5. **Review At-Risk Accounts**: Open `reports/customer_churn_risk.csv` to review high-probability churn accounts for proactive intervention.
6. **Explore in Browser**: Run `streamlit run app.py` to present insights interactively.

---

## ⚠️ Assumptions & Limitations

- **Synthetic Datasets**: Both the financial dataset (6 months) and customer churn dataset (500 records) are illustrative synthetic data created for portfolio demonstration.
- **Simplified Cash Flow Ledger**: Cash inflow is modeled directly from MRR without tracking deferred revenue, accounts receivable aging, or billing timing differences. Operating cash flow is not GAAP accounting profit.
- **Linear Trend Forecasting**: OLS trend forecasting assumes historical growth patterns continue; it represents planning scenarios and does not guarantee future financial outcomes or model macroeconomic shocks.
- **Customer Base Reconstruction**: Aggregate historical subscriber counts are reconstructed chronologically assuming Month 1 begins with 0 customers.
- **Model Portfolio Scope**: The churn classification model demonstrates a clean ML workflow and is not calibrated for live automated production decisioning without real product telemetry.

---

## 🤖 AI-Assisted Development Approach

This project was developed using an **AI-assisted software engineering workflow**. AI tools (such as Claude, ChatGPT, and Antigravity) supported implementation, boilerplate generation, and debugging, while project direction was focused on core business requirements, financial logic formulation, statistical methodology, rigorous test coverage, and end-to-end workflow integration.

The primary objective of this project was to construct a transparent, production-style analytics and reporting lifecycle—from raw data validation and financial mathematics to machine learning integration, test-driven quality assurance, and interactive presentation—ensuring every formula, edge case, and architectural decision is transparent, interview-defensible, and thoroughly validated.

---

## 🛠️ Skills Demonstrated

- **Financial Analytics & SaaS Unit Economics**: MRR, ARR, CAC, MoM revenue growth, subscriber churn rate, and customer base roll-forwards.
- **Cash Flow Modeling & Forecasting**: Multi-category expense tracking, net cash flow calculation, and OLS linear trend forecasting.
- **Machine Learning & Predictive Modeling**: Scikit-Learn pipelines, `ColumnTransformer`, `StandardScaler`, `OneHotEncoder`, `LogisticRegression`, and risk stratification.
- **Data Leakage Awareness**: Training/test split discipline, transformer fitting isolation, and identifier exclusion.
- **Automated Document & Chart Generation**: Multi-page PDF generation via ReportLab and publication-grade Matplotlib visualizations.
- **Interactive Web Development**: Streamlit dashboard presentation layer with caching and modular data connectivity.
- **Robust Software Engineering**: Decoupled architecture, domain exceptions, input DataFrame immutability, automated directory preparation, and artifact validation.
- **Test-Driven Quality Assurance**: 203 comprehensive unit, integration, and security regression tests using `pytest`.
- **CI/CD Automation**: GitHub Actions multi-version Python test matrix (Python 3.9–3.12) and artifact upload workflows.

---

## 🔮 Future Improvements

- **Extended Historical Data**: Incorporate multi-year financial history to support seasonal ARIMA / Holt-Winters forecasting.
- **Cohort Retention Heatmaps**: Build triangle cohort retention matrices tracking subscriber longevity over time.
- **Threshold Calibration**: Implement cost-sensitive threshold optimization based on Customer Lifetime Value (LTV) vs. retention outreach cost.
- **Database Integration**: Connect ingestion pipelines to PostgreSQL or cloud data warehouses (BigQuery/Snowflake).

---

## ⚖️ Disclaimer

This project uses synthetic portfolio data and simplified financial and machine learning assumptions. It is designed solely for educational, portfolio, and demonstration purposes. It does not constitute investment advice, audited financial accounting, or production-ready decisioning software.
