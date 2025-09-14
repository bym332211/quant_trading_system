# Data Module

The data module handles all data processing, management, and preparation for the quant trading system. It includes scripts for downloading S&P500 data, building Qlib datasets, generating features, and data validation.

## 📁 Directory Structure

```
data/
├── download_sp500.py          # Download S&P500 constituent data
├── build_qlib_us.py           # Build Qlib US market dataset
├── build_factors.py           # Generate features from Qlib data
├── make_instrument_sector.py  # Create instrument sector mappings
├── make_instrument_shares.py  # Process instrument share data
├── make_sp500_list.py         # Generate S&P500 stock lists
├── verify_adjusted_qlib.py    # Validate Qlib adjusted data
├── sp500_tickers.txt          # S&P500 ticker list
├── instrument_sector.csv      # Instrument sector classifications
├── data/                      # Raw data storage
├── download/                  # Downloaded data files
└── sp500new/                  # New S&P500 data
```

## 🎯 Key Features

- **Automated Data Download**: Download S&P500 constituent data
- **Qlib Integration**: Build and manage Qlib-formatted datasets
- **Feature Engineering**: Generate technical indicators and factors
- **Data Validation**: Ensure data quality and consistency
- **Multiple Frequencies**: Support for day/1min/5min data

## 📊 Data Contracts

### Features Schema (features_day.parquet)
- **Primary Key**: (instrument, datetime) - UTC timezone naive
- **Required Columns**:
  - `instrument`: Stock ticker symbol
  - `datetime`: Timestamp
  - `$open`, `$high`, `$low`, `$close`, `$vwap`, `$volume`: Price and volume data
  - `ret_1`, `ret_5`: 1-day and 5-day returns
  - `adv_20`: 20-day average dollar volume
  - `mom_20`, `vol_20`: 20-day momentum and volatility
  - `mkt_beta_60`: 60-day market beta
  - `ln_dollar_vol_20`: Log of 20-day dollar volume
  - `ind_*`: Industry one-hot encodings
  - `liq_bucket_*`: Liquidity bucket indicators

### Data Generation Rules
- All price fields are forward-adjusted
- NA values during warm-up periods are normal
- Features are generated from properly adjusted Qlib data

## 🚀 Usage Examples

### Download S&P500 Data
```bash
python download_sp500.py
```

### Build Qlib US Dataset
```bash
python build_qlib_us.py --region us --interval 1d
```

### Generate Features
```bash
python build_factors.py --output features_day.parquet --frequency day
```

### Validate Adjusted Data
```bash
python verify_adjusted_qlib.py --qlib_dir ~/.qlib/qlib_data/us_data
```

## 🔧 Configuration

The data module uses the global configuration from `config/config.yaml` for paths and settings. Key configuration options include:

- `qlib_dir`: Path to Qlib data directory
- `data_paths`: Various data storage locations
- `download_settings`: Data download parameters
- `feature_generation`: Feature calculation parameters

## ⚠️ Important Notes

1. **Data Freshness**: Always ensure data is up-to-date before running models or backtests
2. **Memory Usage**: Feature generation can be memory-intensive for large datasets
3. **Disk Space**: Qlib datasets require significant storage space
4. **Time Zones**: All timestamps are UTC and timezone-naive
5. **Adjustment**: Price data is forward-adjusted for consistency

## 🛠️ Troubleshooting

### Common Issues

1. **Missing Data**: Run the download scripts to ensure all required data is available
2. **Qlib Errors**: Verify Qlib installation and data directory structure
3. **Memory Errors**: Use smaller date ranges or increase system memory
4. **File Permissions**: Ensure write permissions for data directories

### Data Validation

Use the verification scripts to check data quality:
```bash
python verify_adjusted_qlib.py
python -c "import pandas as pd; df = pd.read_parquet('features_day.parquet'); print(df.info())"
```

## 📈 Performance Considerations

- **Batch Processing**: Process data in chunks for large datasets
- **Caching**: Use Parquet format for efficient storage and retrieval
- **Parallel Processing**: Some scripts support parallel execution
- **Incremental Updates**: Update only new data when possible

## 🔗 Related Modules

- **[Models](../models/README.md)**: Uses generated features for training
- **[Backtest](../backtest/README.md)**: Uses features and predictions for backtesting
- **[Scripts](../scripts/README.md)**: Contains data rebuilding utilities

---

*For questions or issues with data processing, check the main project README or create an issue.*
