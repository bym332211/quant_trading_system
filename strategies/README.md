# Strategies Module

The strategies module contains trading strategy implementations that use model predictions to generate trading signals and manage portfolio construction. It includes entry/exit strategies, position sizing, and risk management components.

## 📁 Directory Structure

```
strategies/
├── __init__.py                 # Module initialization
├── stock_selection.py          # Stock selection utilities
├── base/                       # Base classes and registry
│   ├── base_strategy.py        # Abstract strategy base class
│   └── strategy_registry.py    # Strategy registration system
├── entry_strategies/           # Entry signal generation
│   ├── __init__.py
│   ├── coordinator.py          # Entry strategy coordination
│   └── icdf_equal_strategy.py  # ICDF and equal weighting
├── exit_strategies/            # Exit signal generation
│   ├── __init__.py
│   ├── coordinator.py          # Exit strategy coordination
│   ├── tech_stop_loss.py       # Technical stop loss
│   └── volatility_exit.py      # Volatility-based exit
├── lgbm_weekly/                # LightGBM weekly strategy
│   └── predict.py              # Prediction integration
└── xsecrebalance/              # Sector rebalancing strategy
    ├── __init__.py
    ├── data_preparation.py     # Sector data preparation
    └── strategy.py             # Sector rebalancing logic
```

## 🎯 Key Features

- **Modular Design**: Separate entry, exit, and risk components
- **Multiple Weighting Schemes**: Equal weight, ICDF (Inverse CDF), and custom weighting
- **Risk Management**: Built-in stop loss and volatility controls
- **Sector Awareness**: Sector-neutral and sector-rotation strategies
- **Extensible Framework**: Easy to add new strategy types
- **Backtest Integration**: Seamless integration with backtesting engine

## 🏗️ Strategy Architecture

### Strategy Components
1. **Entry Strategies**: Determine when and how to enter positions
   - Equal weighting
   - ICDF (Inverse Cumulative Distribution Function) weighting
   - Momentum-based entry
   - Mean reversion entry

2. **Exit Strategies**: Determine when to exit positions
   - Technical stop loss
   - Volatility-based exits
   - Time-based exits
   - Profit-taking rules

3. **Risk Management**: Position sizing and risk controls
   - Volatility targeting
   - Position limits
   - Drawdown controls
   - Sector constraints

## 📊 Strategy Types

### 1. LGBM Weekly Strategy
Uses LightGBM predictions for weekly rebalancing:
- Cross-sectional stock ranking
- Weekly prediction updates
- Integration with backtest engine

### 2. Sector Rebalancing Strategy
Sector-aware portfolio construction:
- Sector neutrality constraints
- Sector rotation signals
- Dynamic sector weighting

### 3. Technical Strategies
Rule-based approaches:
- Momentum strategies
- Mean reversion strategies
- Breakout strategies

## 🚀 Usage Examples

### Basic Strategy Usage
```python
from strategies.entry_strategies import EqualWeightStrategy
from strategies.exit_strategies import VolatilityExitStrategy

# Create strategy instances
entry_strategy = EqualWeightStrategy(top_k=50)
exit_strategy = VolatilityExitStrategy(volatility_threshold=0.02)

# Generate signals
signals = entry_strategy.generate_signals(predictions, current_date)
exit_signals = exit_strategy.generate_exit_signals(positions, market_data)
```

### LGBM Weekly Strategy
```bash
python -m strategies.lgbm_weekly.predict \
  --model_path ../models/lgbm/weekly_model.pkl \
  --features_path ../artifacts/features_day.parquet \
  --output_path ../artifacts/preds/weekly_signals.parquet
```

### Sector Rebalancing
```bash
python -m strategies.xsecrebalance.strategy \
  --sector_data ../data/instrument_sector.csv \
  --predictions ../artifacts/preds/predictions.parquet \
  --output_path ../artifacts/sector_rebalanced.parquet
```

## ⚙️ Configuration

Strategy configuration uses:
- **Strategy-specific parameters** passed during initialization
- **Global configuration** from `../config/config.yaml`
- **Runtime parameters** for dynamic adjustment

### Common Parameters
- `top_k`: Number of positions to hold
- `weight_scheme`: Weighting methodology (equal, icdf, custom)
- `rebalance_frequency`: How often to rebalance
- `risk_parameters`: Volatility targets, position limits
- `sector_constraints`: Sector neutrality rules

## 📈 Performance Considerations

### Strategy Selection
1. **Market Regime Adaptation**: Different strategies work best in different market conditions
2. **Transaction Costs**: Consider impact of turnover on strategy performance
3. **Liquidity Constraints**: Ensure strategy is executable given liquidity
4. **Model Risk**: Understand limitations of underlying prediction models

### Risk Management
1. **Position Sizing**: Use appropriate position sizing for risk control
2. **Diversification**: Maintain adequate diversification across positions
3. **Stop Losses**: Implement stop losses to limit losses
4. **Volatility Control**: Manage portfolio volatility to target levels

## 🛠️ Development Guidelines

### Adding New Strategies
1. **Inherit from BaseStrategy**: Follow the base class interface
2. **Implement Required Methods**: generate_signals(), update(), etc.
3. **Register Strategy**: Add to strategy registry for discovery
4. **Add Configuration**: Support configurable parameters
5. **Test Thoroughly**: Include unit tests and backtests

### Best Practices
1. **Clear Signal Logic**: Strategies should have well-defined entry/exit rules
2. **Parameter Robustness**: Test across different parameter values
3. **Out-of-Sample Testing**: Validate on unseen data
4. **Transaction Cost Awareness**: Model realistic trading costs

## 🔗 Related Modules

- **[Models](../models/README.md)**: Provides prediction signals
- **[Backtest](../backtest/README.md)**: Evaluates strategy performance
- **[Data](../data/README.md)**: Provides market data and features

## ⚠️ Important Notes

1. **Forward Testing**: Always forward test strategies before live deployment
2. **Parameter Optimization**: Avoid overfitting to historical data
3. **Market Impact**: Consider strategy's potential market impact
4. **Regulatory Compliance**: Ensure strategies comply with relevant regulations

## 🧪 Testing

Run strategy tests:
```bash
python -m pytest tests/strategies/ -v
```

---

*For strategy development and performance optimization, refer to the main project documentation and backtest results.*
