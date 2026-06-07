# Quantitative Asset Allocation

## Overview

This repository contains a collection of quantitative finance projects focused on asset allocation, risk management, and return forecasting. The underlying progression of the work moves from classical mean-variance frameworks to regularized machine learning methodologies. The primary objective is to build practical trading algorithms and portfolios that respect the empirical realities of financial data, such as ill-conditioned covariance matrices, non-stationarity, and computational constraints in combinatorial optimization.

## Project Portfolio

### Ex-Post Machine Learning Techniques for Portfolio Optimization

* **Objective**: Stabilize the efficient frontier and reduce portfolio turnover.
* **Financial context**: Standard Markowitz optimization assumes sample covariance matrices are perfectly estimated. In reality, these matrices are highly ill-conditioned, leading to extreme weight allocations and high turnover when applied to equity markets.
* **Methodology**: Applied statistical regularization and structural regularization to a universe of 12 NSE stocks over a 5-year period.
* **Models / techniques used**: Ledoit-Wolf Shrinkage estimator, Hierarchical Risk Parity (HRP), recursive bisection, and bootstrap resampling.
* **Key findings**: Shrinking the sample covariance matrix significantly lowered the condition number. HRP clustered statistically similar assets without relying on predefined sector labels, which materially reduced out-of-sample portfolio turnover compared to traditional mean-variance methods.
* **Skills demonstrated**: Covariance matrix estimation, unsupervised learning (clustering), algorithm implementation, and financial stability analysis.

### Goal-Based Portfolio Optimization

* **Objective**: Construct a static portfolio to maximize the probability of achieving sequential, time-bound financial goals.
* **Financial context**: Investors often face discrete liabilities at specific future intervals (e.g., intermediate liquidity needs before retirement). Failure to meet a liability incurs a borrowing penalty, making sequence of returns risk critical.
* **Methodology**: Modeled a 20-year investment horizon with specific cash outflows and borrowing penalties using 5 uncorrelated Indian equities. Evaluated two distinct withdrawal sequences (aggressive early withdrawals vs. backloaded withdrawals).
* **Models / techniques used**: Monte Carlo simulation (5,000 paths per portfolio), brute-force combinatorial optimization on discrete weights, and probability maximization.
* **Key findings**: The timing of liabilities fundamentally alters the optimal asset allocation. Front-loaded liabilities demanded lower volatility combinations, while back-loaded liabilities allowed for higher expected returns despite larger drawdowns.
* **Skills demonstrated**: Stochastic modeling, path-dependent option pricing logic, simulation scaling, and computational search.

### Machine Learning Pipeline for Portfolio Optimization

* **Objective**: Forecast daily equity returns using a multimodal dataset and construct a forward-looking portfolio.
* **Financial context**: Pure price action often lacks predictive power. Alpha generation requires synthesizing market microstructure, macroeconomic indicators, fundamental ratios, and alternative data.
* **Methodology**: Built a predictive system for a 6-stock Indian equity universe using 5 years of daily data. The system incorporated lagged features to eliminate look-ahead bias and normalized inputs to handle financial outliers.
* **Models / techniques used**: Pre-trained transformers (FinBERT) for NLP sentiment analysis on financial news, time-series walk-forward nested cross-validation, and recursive feature elimination.
* **Key findings**: Alternative sentiment data improved directional accuracy during earnings windows. Walk-forward validation proved that standard K-fold cross-validation severely overestimates performance in financial time series due to temporal leakage.
* **Skills demonstrated**: End-to-end data pipeline engineering, natural language processing, feature engineering, and rigorous out-of-sample testing.

### Portfolio Replication

* **Objective**: Replicate the daily return profile of the S&P 500 Index using a strictly limited subset of its constituent stocks (k <= 50).
* **Financial context**: Index tracking under cardinality constraints is necessary for minimizing transaction costs and managing liquidity, but selecting the optimal subset is computationally intractable (NP-hard).
* **Methodology**: Approximated the mixed-integer quadratic programming (MIQP) problem over a 5-year period using heuristic and regularized approaches to minimize tracking error.
* **Models / techniques used**: Lasso Regression (L1 regularization) to enforce sparsity and Latent Factor Mapping.
* **Key findings**: The tradeoff between sparsity and tracking error is non-linear. The models successfully identified a subset that matched benchmark returns while avoiding unintentional sector drift.
* **Skills demonstrated**: High-dimensional regression, factor modeling, optimization heuristics, and tracking error minimization.

### Modern Portfolio Theory

* **Objective**: Engineer a weighting strategy that maximizes risk-adjusted performance using a defined multi-asset universe.
* **Financial context**: The fundamental baseline for quantitative asset management involves mapping the expected return and variance trade-off across uncorrelated assets.
* **Methodology**: Analyzed a 10-stock universe selected from a 30-stock dataset based on historical correlation structures. Evaluated long-only constraints over a 3-year period.
* **Models / techniques used**: Markowitz optimization, Monte Carlo efficient frontier generation, correlation matrix analysis, and Global Minimum Variance (GMV) modeling.
* **Key findings**: Identified the optimal tangent portfolio against a predefined risk-free rate and demonstrated the distinct weight distributions between the GMV and Maximum Sharpe portfolios.
* **Skills demonstrated**: Core financial mathematics, linear algebra, and data visualization.

## Quantitative Techniques

* Statistical modeling
* Time-series forecasting
* Machine learning
* Feature engineering
* Portfolio analytics
* Optimization
* Financial econometrics

## Technology Stack

* **Languages**: Python
* **Data Processing**: Pandas, NumPy
* **Machine Learning**: Scikit-learn, FinBERT
* **Optimization & Math**: SciPy
* **Simulation**: Custom Monte Carlo engines
* **Visualization**: Matplotlib, Seaborn

## Repository Structure

```text
.
|-- Ex-Post ML techniques for Portfolio Optimization
|   |-- notebooks
|   |   |-- collecting_data_and_preprocess.ipynb
|   |   |-- part1.ipynb
|   |   `-- part2.ipynb
|-- Goal Based Portfolio Optimization
|   `-- Solution.ipynb
|-- ML Pipleline for Portfolio Optimization
|   |-- notebooks
|   |   |-- P01_Scrapping.ipynb
|   |   |-- P02_Pre-Processing.ipynb
|   |   |-- P03_Sentiment.ipynb
|   |   `-- P04_Portfolio_Construction.ipynb
|-- Modern Portfolio Theory
|   `-- Notebook
|       `-- Modern Portfolio Theory.ipynb
`-- Portfolio Replication
    `-- notebooks
        |-- Lasso Regression.ipynb
        `-- Latent Factor Mapping.ipynb
```
