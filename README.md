# Quantitative Asset Allocation

A research suite of five quantitative-finance projects on asset allocation, risk management, and
return forecasting. The work follows a single thesis: **classical mean–variance optimization is
fragile on real market data** — covariance matrices are ill-conditioned, returns are
non-stationary, and optimal subset selection is combinatorially intractable — and the practical
fixes come from **regularization and machine learning**. Each project pairs a financial problem
with a method that respects these empirical realities, and reports honest out-of-sample results.

> All headline numbers below are taken from executed notebook outputs in this repository.

---

## Projects at a Glance

| Project | Universe / Scale | Core method | Headline result |
|---|---|---|---|
| [Ex-Post ML Techniques](#1-ex-post-ml-techniques-for-portfolio-optimization) | 12 NSE stocks, 5y | Ledoit–Wolf shrinkage + Hierarchical Risk Parity | Covariance condition number **29.3 → 13.3 (−55%)**; HRP **~157% lower turnover** than mean–variance |
| [Portfolio Replication](#2-portfolio-replication-index-tracking) | **S&P 500 — 559 stocks**, 5y | L1 (Lasso) sparse tracking vs. autoencoder factor map | Tracked index with **43 of 559 names (−92%)**, test TE **0.0045/day**; beat autoencoder **~16×** on out-of-sample TE |
| [ML Pipeline](#3-ml-pipeline-for-portfolio-optimization) | 6 NSE stocks, 5y daily, **44 features** | FinBERT sentiment + GBMs/LSTM under walk-forward CV | Leakage-free pipeline; **FinBERT over 5,426 headlines**; HRP-based portfolio backtest framework |
| [Goal-Based Optimization](#4-goal-based-portfolio-optimization) | 5 equities, 20y horizon | Vectorized Monte Carlo + combinatorial / SLSQP search | **5,000 paths × 70 allocations × 20y**; liability timing shown to drive optimal allocation |
| [Modern Portfolio Theory](#5-modern-portfolio-theory-baseline) | 10 of 30 stocks, 3y | Markowitz efficient frontier | GMV vs. Max-Sharpe tangency portfolios; classical baseline |

---

## 1. Ex-Post ML Techniques for Portfolio Optimization

**Problem.** Markowitz optimization assumes the sample covariance matrix is well estimated. In
practice it is highly ill-conditioned, producing extreme weights and high turnover out-of-sample.

**Approach.** Applied **statistical regularization** (Ledoit–Wolf shrinkage) and **structural
regularization** (Hierarchical Risk Parity) to a 12-stock NSE universe over 5 years.

**Results (measured).**
- Shrinkage reduced the covariance **condition number from 29.34 to 13.29 (−55%)** at a fitted
  shrinkage intensity of **0.164**.
- HRP — correlation-distance clustering, quasi-diagonalization, and recursive bisection — allocated
  risk **without predefined sector labels** and produced **~157% lower portfolio turnover** than the
  mean–variance allocation.

**Skills.** Covariance estimation, unsupervised clustering, numerical stability analysis.

## 2. Portfolio Replication (Index Tracking)

**Problem.** Tracking the S&P 500 with a small subset of constituents minimizes transaction costs
and liquidity drag, but choosing the optimal subset is an NP-hard mixed-integer quadratic program
over 559 names.

**Approach.** Two competing methods benchmarked head-to-head over 5 years
(train 2020–2025-06, holdout 2025-07–12):
1. **Lasso (L1) regression** — exploits the natural sparsity of L1 to select constituents and weights
   in one step; alpha swept to trade sparsity against tracking error.
2. **Latent factor mapping** — a PyTorch autoencoder (559 → 128 → *k* latent) plus hierarchical
   clustering to pick representative names.

**Results (measured).**
- Lasso tracked the index with **43 of 559 stocks (−92% cardinality)** at a holdout tracking error
  of **0.0045/day**.
- The autoencoder factor map needed 50 stocks for a holdout TE of **0.075/day** — roughly **16×
  worse**, demonstrating that a well-chosen regularized linear model can dominate a deeper,
  more complex one on this task.

**Skills.** High-dimensional sparse regression, factor modeling, tracking-error minimization,
model benchmarking.

## 3. ML Pipeline for Portfolio Optimization

**Problem.** Price action alone is weakly predictive; alpha requires fusing microstructure,
macro, fundamentals, and alternative data — without leaking the future into the past.

**Approach.** An end-to-end pipeline over a 6-stock NSE universe, 5 years of daily data:
- **Feature engineering — 44 features:** market microstructure, fundamental ratios (revenue,
  margins, growth), macro indicators (USD/GBP/EUR/JPY, G-Sec and T-bill yields, RBI rate, Brent,
  inflation), and **FinBERT sentiment scored over 5,426 scraped news headlines** (55% daily
  coverage). All inputs are **lagged to eliminate look-ahead bias** and scaled with a
  `RobustScaler`; winsorization is fit on the training window only.
- **Models:** XGBoost, LightGBM, and a 2-layer PyTorch LSTM (lookback 20, dropout 0.2),
  evaluated under **5-fold walk-forward validation**. The final model is selected on
  **walk-forward RMSE — never the test set** — explicitly avoiding the temporal leakage that
  standard K-fold CV introduces in time series.
- **Construction & backtest:** HRP allocation (recursive bisection, cophenetic correlation 0.88)
  with a reproducible backtest reporting Sharpe / Sortino / Calmar, drawdown, and turnover across
  Equal-Weight, Risk-Adjusted, and Pure-HRP portfolios.

**Honest finding.** Daily directional accuracy lands near **0.50** — single-name daily returns are
close to unpredictable — which is precisely what a leakage-free evaluation is supposed to reveal.
The value here is the **methodology and infrastructure**, not an inflated backtest.

**Skills.** Data-pipeline engineering, NLP (transformers), feature engineering, rigorous
out-of-sample validation.

## 4. Goal-Based Portfolio Optimization

**Problem.** Investors face discrete, time-bound liabilities; missing one incurs a borrowing
penalty, making sequence-of-returns risk critical.

**Approach.** A **vectorized Monte Carlo engine** simulating a 20-year horizon with sequential cash
outflows and a 12% borrowing penalty on shortfalls, across **5,000 return paths × 70 discrete weight
allocations**. Two liability schedules (aggressive-early vs. backloaded) are compared, and discrete
brute-force optima are validated against continuous **SLSQP** optimization.

**Finding.** The **timing of liabilities fundamentally changes the optimal allocation** —
front-loaded goals demand lower-volatility mixes, while backloaded goals tolerate higher expected
return and larger drawdowns.

**Skills.** Stochastic / path-dependent simulation, combinatorial search, constrained optimization.

## 5. Modern Portfolio Theory (Baseline)

**Problem.** Establish the classical expected-return / variance trade-off the regularized methods
aim to improve on.

**Approach.** Markowitz optimization on a 10-stock universe (selected by correlation structure from
30 candidates) over 3 years: Monte Carlo efficient frontier, Global Minimum Variance, and the
Max-Sharpe tangency portfolio against a defined risk-free rate.

**Skills.** Financial mathematics, linear algebra, data visualization.

---

## Technology Stack

- **Language:** Python
- **Data:** Pandas, NumPy
- **ML / DL:** Scikit-learn, XGBoost, LightGBM, PyTorch, FinBERT (Hugging Face Transformers)
- **Optimization / Math:** SciPy (SLSQP), SciPy hierarchical clustering
- **Simulation:** Custom vectorized Monte Carlo engines
- **Visualization:** Matplotlib, Seaborn, Plotly

## Repository Structure

```text
.
├── Ex-Post ML techniques for Portfolio Optimization
│   └── notebooks/  collecting_data_and_preprocess · part1 · part2
├── Portfolio Replication
│   └── notebooks/  Lasso Regression · Latent Factor Mapping
├── ML Pipeline for Portfolio Optimization
│   └── notebooks/  P01_Scrapping · P02_Pre-Processing · P03_Sentiment ·
│                   P04_Portfolio_Construction · Prediction/<TICKER>_Stock_Prediction
├── Goal Based Portfolio Optimization
│   └── Solution.ipynb
└── Modern Portfolio Theory
    └── Notebook/  Modern Portfolio Theory.ipynb
```

Each project folder contains its own problem statement, data, and a written report (PDF).

## Quantitative Techniques

Covariance estimation & shrinkage · hierarchical risk parity · sparse / L1 regression ·
factor modeling & autoencoders · time-series forecasting (GBMs, LSTM) · walk-forward validation ·
NLP sentiment analysis · Monte Carlo simulation · constrained & combinatorial optimization ·
financial econometrics.
