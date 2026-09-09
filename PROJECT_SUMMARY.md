# Project Summary — SaaS Financial Reporter

Quick reference and interview preparation guide for the **SaaS Financial Reporter** project.

---

## 📌 Project in One Sentence
An automated Python analytics and machine learning pipeline that transforms monthly SaaS operational data into recurring revenue metrics, historical cash-flow analysis, 6-month statistical forecasts, an executive investor PDF report, and customer churn risk scores.

---

## 🎯 Problem Solved
Early-stage SaaS businesses often track finances and retention in disconnected spreadsheets, causing repetitive calculation overhead, lack of forward cash visibility, and reactive retention workflows. This project automates the entire analytical lifecycle into a single, reliable Python execution that outputs financial dashboards, PDF summaries, and proactive customer risk scoring.

---

## 🏛️ Architecture Overview
The project maintains a strict separation of concerns across two primary workflows:
1. **Financial Reporting Pipeline**: Ingests monthly aggregated metrics (`saas_financial_data.csv`), validates schema integrity, calculates revenue/CAC/churn KPIs, performs cash flow accounting, projects 6-month linear forecasts, renders 10 charts, compiles a master PDF, and prints a console dashboard.
2. **Customer Churn ML Pipeline**: Ingests individual customer behavior (`customer_churn_data.csv`), performs a stratified 80/20 train/test split, trains a scikit-learn Logistic Regression pipeline (`StandardScaler` + `OneHotEncoder`), evaluates metrics on held-out test data, categorizes accounts into Low/Medium/High risk tiers, and outputs `customer_churn_risk.csv`.

---

## 🚀 Core Features
- **Recurring Revenue**: MRR, ARR, and Month-over-Month growth rates.
- **Unit Economics**: Customer Acquisition Cost (CAC) and efficiency trend tracking.
- **Retention Dynamics**: Reconstructed customer base roll-forward, subscriber churn %, and retention %.
- **Operating Cash Flow**: Cash inflow (MRR) vs. 4 expense categories, net cash flow, and rolling balances.
- **6-Month Forecasting**: Ordinary Least Squares (OLS) linear trend forecasting on individual components.
- **Automated Reporting**: Master multi-page PDF generated via ReportLab with 10 high-resolution Matplotlib charts.
- **Predictive ML**: Logistic Regression classifier predicting churn probability and risk tier classification.
- **Execution Reliability**: Pipeline coordinator (`src/pipeline.py`), domain exceptions, automated directory creation, and artifact validation.

---

## 💡 Biggest Technical Decisions
1. **Two Distinct Datasets**: Kept monthly aggregate financial data separate from customer-level behavioral data to prevent artificial joins and ensure each dataset serves its proper domain.
2. **Component-by-Component Forecasting**: Modeled revenue and each expense line independently with linear regression and floored at 0, rather than treating net cash flow as a black box.
3. **Scikit-Learn Pipeline with Preprocessing Encapsulation**: Enforced `StandardScaler` and `OneHotEncoder` fitting strictly on training partitions (`X_train`) to completely eliminate test data leakage.
4. **Decoupled Orchestration**: Refactored `main.py` into a thin entry point delegating to `src/pipeline.py`, returning deterministic `SUCCESS`, `PARTIAL_SUCCESS`, or `FAILED` execution statuses.

---

## ⚠️ Biggest Limitations
- **Synthetic Datasets**: Datasets represent illustrative sample data for portfolio demonstration.
- **Simplified Cash Flow**: Cash inflow is modeled directly from MRR without modeling accounts receivable aging, payment delays, or deferred revenue.
- **Linear Trend Forecasting**: Assumes historical linear patterns continue; does not capture seasonality or macroeconomic shocks.
- **Reconstructed Customer Base**: Customer count begins at 0 in Month 1 due to absence of historical baseline records.

---

## 🔮 What I Would Improve Next
1. **Cohort Retention Analysis**: Implement triangular subscriber cohort retention heatmaps.
2. **Web Simulation Dashboard**: Build an interactive Streamlit or FastAPI dashboard with parameter sliders for cash runway scenario testing.
3. **Cost-Sensitive Thresholds**: Optimize churn probability cutoff thresholds based on Customer Lifetime Value (LTV) vs. retention outreach cost.

---

## ⏱️ 60-Second Interview Explanation

> "I built an AI-assisted SaaS financial reporting and customer analytics pipeline in Python. The project takes monthly SaaS financial metrics, validates data integrity, and calculates key unit economics like MRR, ARR, CAC, and subscriber churn. It performs historical cash-flow analysis, forecasts the next six months of cash position using component-wise linear regression, and automatically compiles a multi-page PDF investor report with charts.
>
> In parallel, I built a customer-level machine learning pipeline using Logistic Regression on behavioral data to predict churn probabilities and categorize accounts into operational risk tiers. I focused on clean architecture, preventing data leakage, thorough test coverage with 180 pytest tests, and keeping all mathematical assumptions transparent."

---

## ⏱️ 2-Minute Interview Explanation

> "The motivation behind this project was to automate the recurring financial reporting and retention analytics workflow for subscription businesses. Often, teams spend days calculating MRR, CAC, and cash burn across spreadsheets while identifying churned customers only after they cancel.
>
> I structured the project around two decoupled workflows. The first is a monthly financial pipeline that ingests financial records, reconstructs the customer base roll-forward, tracks operating cash flow across marketing, payroll, and operational expenses, and projects future runway using linear regression. This feeds directly into an automated ReportLab PDF generator and a consolidated console dashboard.
>
> The second workflow is a predictive machine learning pipeline that operates on customer-level data. To demonstrate sound ML engineering, I used a scikit-learn Pipeline combining `StandardScaler` and `OneHotEncoder` with `LogisticRegression`. Preprocessing is fit strictly on training splits to prevent data leakage, non-predictive identifiers like CustomerID and signup dates are excluded, and accounts are scored into Low, Medium, and High risk tiers for proactive Customer Success outreach.
>
> I developed this project using an AI-assisted workflow, focusing on validating business logic, designing test scenarios across 180 unit and regression tests, and implementing reliable error handling so non-critical chart warnings produce partial success rather than crashing the run."
