# Backtest Module

The backtest module contains the core backtesting engine and performance analysis tools for evaluating trading strategies. It provides comprehensive diagnostics, risk management features, and detailed performance reporting.

## 📁 Directory Structure

```
backtest/
├── engine/                       # Core backtesting engine
│   ├── __init__.py
│   ├── backtest_runner.py        # Main backtest execution
│   ├── config_loader.py          # Configuration loading
│   ├── data_loader.py            # Data loading and preparation
│   ├── data_utils.py             # Data utility functions
│   ├── README.md                 # Detailed engine documentation
│   └── run_backtest.py           # Main backtest script
├── kpi/                          # Performance metrics
│   ├── __init__.py
│   └── calculator.py             # KPI calculation utilities
└── reports/                      # Backtest results and outputs
    └── (various report directories)
```

## 🎯 Key Features

- **Backtrader Integration**: Built on the Backtrader framework
- **Comprehensive Diagnostics**: Detailed performance metrics and analysis
- **Risk Management**: Advanced risk controls and position sizing
- **Multiple Frequency Support**: Day, 1-minute, and 5-minute data
- **Walk-Forward Testing**: Robust out-of-sample validation
- **Parameter Optimization**: Support for hierarchical parameter sweeps

## 🏗️ Backtest Architecture

### Core Components
1. **Engine Core**: Main backtest execution and coordination
2. **Data Loading**: Qlib data integration and feature loading
3. **Strategy Execution**: Trade execution and portfolio management
4. **Risk Management**: Position limits, volatility targeting, and constraints
5. **Performance Analysis**: Comprehensive metrics and reporting

### Workflow
1. **Data Preparation**: Load features and predictions
2. **Strategy Setup**: Configure trading strategy parameters
3. **Backtest Execution**: Run historical simulation
4. **Results Analysis**: Generate performance metrics and reports
5. **Optimization**: Parameter tuning and strategy refinement

## 📊 Performance Metrics

The backtest engine provides comprehensive performance analysis:

### Core Metrics
- **Sharpe Ratio**: Risk-adjusted returns
- **CAGR**: Compound Annual Growth Rate
- **Maximum Drawdown (MDD)**: Worst peak-to-trough decline
- **Information Coefficient (IC)**: Prediction accuracy
- **Turnover Rate**: Portfolio churn
- **Cost Analysis**: Transaction costs and slippage

### Advanced Diagnostics
- Daily performance breakdown
- Long/short leg contributions
- Sector and liquidity analysis
- ADV limit compliance
- Water-filling allocation details

## 🚀 Quick Start

### Basic Backtest
```bash
python backtest/engine/run_backtest.py \
  --qlib_dir ~/.qlib/qlib_data/us_data \
  --preds artifacts/preds/weekly/predictions.parquet \
  --features_path artifacts/features_day.parquet \
  --start 2020-01-01 --end 2024-12-31 \
  --out_dir backtest/reports/basic_test
```

### Advanced Configuration
```bash
python backtest/engine/run_backtest.py \
  --qlib_dir ~/.qlib/qlib_data/us_data \
  --preds artifacts/preds/weekly/predictions.parquet \
  --features_path artifacts/features_day.parquet \
  --start 2017-01-01 --end 2024-12-31 \
  --top_k 40 --short_k 8 \
  --long_exposure 1.0 --short_exposure -0.6 \
  --weight_scheme equal \
  --membership_buffer 0.4 --smooth_eta 0.85 \
  --neutralize "beta,sector,liq,size" \
  --target_vol 0.08 --leverage_cap 2.0 \
  --max_pos_per_name 0.05 --hard_cap \
  --adv_limit_pct 0.01 \
  --short_timing_mom63 --short_timing_threshold 0.0 \
  --out_dir backtest/reports/advanced_test
```

## ⚙️ Configuration

### Key Parameters
- `--top_k` / `--short_k`: Number of long/short positions
- `--weight_scheme`: Weighting methodology (equal, icdf)
- `--neutralize`: Risk factor neutralization
- `--target_vol`: Target annualized volatility
- `--max_pos_per_name`: Single position limit
- `--adv_limit_pct`: ADV-based position limits
- `--hard_cap`: Enable precise water-filling allocation

### Execution Options
- `--trade_at`: Trade execution time (open, close)
- `--exec_lag`: Execution lag in days
- `--anchor_symbol`: Benchmark symbol for calendar alignment

## 📈 Output Files

Backtest results include:

### Core Outputs
- `equity_curve.csv`: Portfolio value over time
- `portfolio_returns.csv`: Daily returns
- `per_day_ext.csv`: Extended daily diagnostics
- `orders.csv`: Trade execution details
- `positions.csv`: Portfolio holdings snapshots

### Summary Files
- `summary.json`: Overall performance summary
- `kpis.json`: Detailed performance metrics
- `parameters.json`: Backtest configuration

## 🔗 Related Modules

- **[Data](../data/README.md)**: Provides features and market data
- **[Models](../models/README.md)**: Generates prediction signals
- **[Strategies](../strategies/README.md)**: Implements trading logic
- **[Scripts](../scripts/README.md)**: Provides optimization utilities

## ⚠️ Important Notes

1. **Data Quality**: Ensure data is properly adjusted and validated
2. **Out-of-Sample**: Use proper walk-forward testing methodology
3. **Transaction Costs**: Include realistic cost assumptions
4. **Liquidity Constraints**: Consider market impact and execution
5. **Model Risk**: Understand limitations of prediction models

## 🧪 Testing

Run backtest validation:
```bash
# Test with dummy data
python backtest/engine/run_backtest.py \
  --qlib_dir ~/.qlib/qlib_data/us_data \
  --preds artifacts/preds/dummy_predictions.parquet \
  --features_path artifacts/features_day.parquet \
  --start 2023-01-01 --end 2023-03-31 \
  --out_dir backtest/reports/test_run
```

## 📖 Detailed Documentation

For comprehensive documentation on the backtest engine, including advanced features, parameter explanations, and usage examples, see:

**[Backtest Engine Documentation](engine/README.md)**

The engine README provides detailed information on:
- Advanced parameter configurations
- Risk management features
- Performance diagnostics
- Best practices and troubleshooting
- Version-specific features and changes

---

*For detailed backtest engine usage, parameters, and advanced features, refer to the [engine documentation](engine/README.md).*
