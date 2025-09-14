# Scripts Module

The scripts module contains utility scripts for data management, system maintenance, parameter optimization, and various operational tasks. These scripts support the main quant trading system functionality.

## 📁 Directory Structure

```
scripts/
├── dump_bin.py                    # Export Qlib binary data
├── hierarchical_sweep_sharpe.py   # Two-stage parameter optimization
├── make_dummy_preds.py            # Generate dummy predictions for testing
├── rebuild_us_1day.bat            # Windows batch script for daily data rebuild
├── rebuild_us_1day.sh             # Linux shell script for daily data rebuild
├── rebuild_us_1min.sh             # Linux script for 1-minute data rebuild
├── rebuild_us_5min.sh             # Linux script for 5-minute data rebuild
├── refine_from_best.py            # Refine parameters from best results
└── sweep_sharpe_focus.py          # Single-stage parameter optimization
```

## 🎯 Key Features

- **Data Management**: Scripts for rebuilding and maintaining Qlib datasets
- **Parameter Optimization**: Hierarchical and focused parameter sweeps
- **Testing Utilities**: Tools for generating test data and predictions
- **Cross-Platform Support**: Both Windows (.bat) and Linux (.sh) scripts
- **Automation**: Scripts for automating repetitive tasks
- **Performance Optimization**: Tools for maximizing strategy performance

## 📊 Script Categories

### 1. Data Management Scripts
**Purpose**: Manage and maintain Qlib data infrastructure

- `rebuild_us_1day.sh` / `rebuild_us_1day.bat`: Rebuild daily US market data
- `rebuild_us_1min.sh`: Rebuild 1-minute frequency data
- `rebuild_us_5min.sh`: Rebuild 5-minute frequency data
- `dump_bin.py`: Export Qlib binary data to other formats

### 2. Parameter Optimization Scripts
**Purpose**: Optimize strategy parameters for maximum Sharpe ratio

- `hierarchical_sweep_sharpe.py`: Two-stage optimization (coarse → fine)
- `sweep_sharpe_focus.py`: Single-stage focused optimization
- `refine_from_best.py`: Refine parameters from existing best results

### 3. Testing and Utility Scripts
**Purpose**: Support development and testing workflows

- `make_dummy_preds.py`: Generate dummy predictions for testing
- Various helper scripts for system maintenance

## 🚀 Usage Examples

### Rebuild Qlib Data
```bash
# Rebuild daily data (Linux/Mac)
./scripts/rebuild_us_1day.sh

# Rebuild daily data (Windows)
scripts\rebuild_us_1day.bat

# Rebuild minute data
./scripts/rebuild_us_1min.sh
./scripts/rebuild_us_5min.sh
```

### Parameter Optimization
```bash
# Hierarchical parameter sweep (recommended)
python scripts/hierarchical_sweep_sharpe.py \
  --qlib_dir "/path/to/qlib_data" \
  --preds "artifacts/preds/weekly/predictions.parquet" \
  --features_path "artifacts/features_day.parquet" \
  --start "2017-01-01" --end "2024-12-31" \
  --out_root "backtest/reports/hier_sweep" \
  --stage1_preset coarse --run_stage2

# Focused parameter sweep
python scripts/sweep_sharpe_focus.py \
  --qlib_dir "/path/to/qlib_data" \
  --preds "artifacts/preds/predictions.parquet" \
  --features_path "artifacts/features_day.parquet" \
  --start "2020-01-01" --end "2024-12-31" \
  --preset focused \
  --out_dir "backtest/reports/focused_sweep"
```

### Generate Test Data
```bash
# Create dummy predictions for testing
python scripts/make_dummy_preds.py \
  --output "artifacts/preds/dummy_predictions.parquet" \
  --start_date "2020-01-01" \
  --end_date "2024-12-31" \
  --num_instruments 500
```

### Export Data
```bash
# Export Qlib binary data
python scripts/dump_bin.py \
  --qlib_dir "/path/to/qlib_data" \
  --output_dir "./exported_data" \
  --instruments "SPY,AAPL,MSFT" \
  --fields "close,volume,high,low"
```

## ⚙️ Configuration

Script configuration uses:
- **Command-line arguments** for runtime parameters
- **Environment variables** for system-specific settings
- **Global configuration** from `../config/config.yaml` where applicable

### Common Parameters for Optimization Scripts
- `--qlib_dir`: Path to Qlib data directory
- `--preds`: Path to predictions file
- `--features_path`: Path to features file
- `--start`/`--end`: Backtest date range
- `--out_root`/`--out_dir`: Output directory for results
- `--preset`: Parameter preset (coarse, medium, focused, fine)

## 📈 Optimization Methodology

### Hierarchical Sweep Approach
1. **Stage 1 (Coarse)**: Broad parameter exploration
2. **Stage 2 (Fine)**: Focused optimization around best parameters
3. **Neighborhood Exploration**: Systematic exploration around optimum

### Parameter Grids
Scripts support optimization of key parameters:
- `top_k`: Number of positions
- `weight_scheme`: Equal vs ICDF weighting
- `membership_buffer`: Entry/exit buffer
- `target_vol`: Target volatility
- `smooth_eta`: Smoothing parameter
- `max_pos_per_name`: Position limits
- `adv_limit_pct`: ADV limit percentage

## 🛠️ Best Practices

### Data Management
1. **Regular Rebuilds**: Schedule regular data rebuilds to ensure data quality
2. **Version Control**: Keep track of data versions and rebuild dates
3. **Backup**: Maintain backups of critical data files
4. **Validation**: Verify data integrity after rebuilds

### Parameter Optimization
1. **Walk-Forward**: Use walk-forward validation for robust optimization
2. **Out-of-Sample**: Always validate on out-of-sample data
3. **Multiple Time Periods**: Test across different market regimes
4. **Transaction Costs**: Include realistic cost assumptions

### Script Maintenance
1. **Documentation**: Keep scripts well-documented
2. **Error Handling**: Implement robust error handling
3. **Logging**: Include comprehensive logging
4. **Testing**: Test scripts before production use

## 🔗 Related Modules

- **[Data](../data/README.md)**: Scripts interact with data processing
- **[Backtest](../backtest/README.md)**: Optimization scripts use backtest engine
- **[Models](../models/README.md)**: Scripts support model evaluation

## ⚠️ Important Notes

1. **Resource Intensive**: Parameter optimization can be computationally expensive
2. **Data Requirements**: Ensure data is properly prepared before running scripts
3. **Result Interpretation**: Carefully interpret optimization results to avoid overfitting
4. **Production Readiness**: Test scripts thoroughly before production use

## 🧪 Testing

Run script functionality tests:
```bash
# Test data generation
python scripts/make_dummy_preds.py --test

# Test parameter optimization with small dataset
python scripts/sweep_sharpe_focus.py \
  --qlib_dir "/path/to/qlib_data" \
  --preds "artifacts/preds/dummy_predictions.parquet" \
  --start "2023-01-01" --end "2023-06-30" \
  --preset test
```

---

*For detailed usage of specific scripts, refer to the script help messages (`--help`) and main project documentation.*
