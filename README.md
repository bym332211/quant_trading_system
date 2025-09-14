# Quant Trading System

A comprehensive, extensible quantitative trading system designed for US equities with crypto-ready architecture. The system follows a four-layer architecture: **Data → Features → Models → Backtest/Execution**.

## 🏗️ System Architecture

```
quant_trading_system/
├── data/          # Data processing and management
├── models/        # Machine learning models (LightGBM)
├── strategies/    # Trading strategies framework  
├── backtest/      # Backtesting engine and analysis
├── scripts/       # Utility scripts and tools
├── config/        # Configuration management
└── artifacts/     # Generated features and predictions
```

## 🚀 Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
```

### Basic Usage
1. **Data Preparation**: Build Qlib data and features
2. **Model Training**: Train LightGBM models
3. **Backtesting**: Run comprehensive backtests
4. **Analysis**: Review performance metrics

### Example Command
```bash
python backtest/engine/run_backtest.py \
  --qlib_dir ~/.qlib/qlib_data/us_data \
  --preds artifacts/preds/weekly/predictions.parquet \
  --features_path artifacts/features_day.parquet \
  --start 2020-01-01 --end 2024-12-31
```

## 📊 Key Features

- **Multi-frequency Data**: Day/1min/5min data support with Qlib `.bin` format
- **Machine Learning**: LightGBM-based cross-sectional stock selection
- **Advanced Backtesting**: Backtrader-based framework with comprehensive diagnostics
- **Risk Management**: Neutralization, target volatility, hard caps, %ADV limits
- **Walk-Forward Testing**: Robust out-of-sample validation
- **Extensible**: Designed for easy extension to crypto markets

## 🔗 Module Documentation

For detailed usage of each module, refer to the specific README files:

- **[Data Module](data/README.md)** - Data processing and management
- **[Models Module](models/README.md)** - LightGBM machine learning models  
- **[Strategies Module](strategies/README.md)** - Trading strategies framework
- **[Backtest Module](backtest/README.md)** - Backtesting engine and analysis
- **[Scripts Module](scripts/README.md)** - Utility scripts and tools

## 📈 Current Status (v3.4.x)

### ✅ Implemented Features
- Complete backtest engine with diagnostic outputs
- Hard cap water-filling for precise position sizing
- Short-leg timing based on SPY momentum
- Comprehensive KPI reporting (Sharpe, CAGR, MDD, turnover, costs)
- Data contracts and schema validation
- Hierarchical parameter grid search

### 🚧 In Progress
- Covariance/TE constraints
- Impact/fill modeling
- Enhanced reporting

### 📋 Planned Features
- Minute-level execution
- Monitoring/CI integration
- Crypto market adaptation

## 🎯 Performance Highlights

The system supports sophisticated parameter optimization with hierarchical grid search:

```bash
python scripts/hierarchical_sweep_sharpe.py \
  --qlib_dir "/path/to/qlib_data" \
  --preds "artifacts/preds/weekly/predictions.parquet" \
  --features_path "artifacts/features_day.parquet" \
  --start "2017-01-01" --end "2024-12-31" \
  --stage1_preset coarse --run_stage2
```

## 🤝 Contributing

This project follows a modular architecture. When adding new features:
1. Follow existing data contracts and interfaces
2. Maintain backward compatibility where possible
3. Add comprehensive tests for new functionality
4. Update relevant documentation

## 📝 License

This project is designed for research and educational purposes in quantitative trading.

---

*For detailed module-specific documentation, please refer to the individual module README files.*
