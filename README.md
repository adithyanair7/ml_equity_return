# ML Equity Return Prediction

This repository implements a complete research framework for forecasting and backtesting cross-sectional equity returns using modern machine learning techniques. The design emphasizes methodological rigor, bias control, and realistic simulation of market frictions.

## Overview

The project provides an end-to-end system that covers:
- Data preparation from raw panel data (e.g., CRSP/Compustat or Nasdaq-100).
- Feature engineering for predictive signals (momentum, volatility, liquidity, value-style proxies).
- Machine learning model training using Random Forests and XGBoost in a pooled cross-sectional walk-forward setup.
- Bias-safe backtesting with transaction costs, volatility targeting, lagged execution, and delisting return integration.

The framework is designed to be extensible for both academic research and professional quantitative finance applications.

## Features

- **Walk-forward model training:** Rolling windows with configurable train/test splits and constant-time stepping.
- **Models:** Random Forests, XGBoost, or ensemble combinations.
- **Cross-sectional normalization:** Per-date z-scoring to avoid leakage from scale differences.
- **Backtesting framework:** Non-overlapping rebalances, lagged execution, turnover-based costs, per-name caps, and dispersion gating.
- **Outputs:** Predictions, equity curves, weight histories, and summary statistics.

## Repository Structure

```
├── data/
│   └── universe/          # Input datasets (features, scores, panel data)
├── results/               # Backtest outputs (equity curves, weights, summaries)
├── src/                   # Source code for training and portfolio backtesting
│   ├── data_loader.py
│   ├── features.py
│   ├── portfolio.py
│   └── train_model.py
├── notebooks/             # Example research notebooks
│   ├── 01_feature_engineering_demo.ipynb
│   └── 02_backtest_demo.ipynb
└── README.md
```

## Usage

1. **Generate features**  
   ```bash
   python src/features.py
   ```

2. **Train models**  
   ```bash
   python src/train_model.py --model ensemble --step 21 --zscore
   ```

3. **Run backtest**  
   ```bash
   python src/portfolio.py --scores data/universe/scores.csv --features data/universe/features.csv --horizon 21 --soft_weighting
   ```

## Example Results

A typical run of the system delivered performance metrics such as:  
- Sharpe Ratio ≈ 1.9  
- Max Drawdown ≈ -22%  
- CAGR ≈ 28%  
- Hit Rate ≈ 56%  

These results are post-costs with turnover-adjusted transaction fees and lagged execution.

## Notebooks

The repository includes Jupyter notebooks in the `notebooks/` directory:
- **01_feature_engineering_demo.ipynb** – Demonstrates engineered features from raw panel data.
- **02_backtest_demo.ipynb** – Illustrates backtest execution and analysis of results.

## License

This project is released under the MIT License for educational and research purposes.
