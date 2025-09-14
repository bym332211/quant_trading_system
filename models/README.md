# Models Module

The models module contains machine learning implementations for quantitative trading, currently focused on LightGBM for cross-sectional stock selection with weekly/monthly rebalancing frequencies.

## 📁 Directory Structure

```
models/
└── lgbm/                         # LightGBM implementation
    ├── dataset.py               # Dataset preparation and feature engineering
    ├── evaluate.py              # Model evaluation and performance metrics
    ├── inference.py             # Model inference and prediction generation
    ├── test.py                  # Unit tests and validation
    └── train.py                 # Model training and hyperparameter optimization
```

## 🎯 Key Features

- **LightGBM Integration**: Gradient boosting framework for stock prediction
- **Cross-Sectional Modeling**: Rank stocks within time periods
- **Weekly/Monthly Frequency**: Support for different rebalancing schedules
- **Feature Importance**: Analyze which factors drive predictions
- **Walk-Forward Validation**: Robust out-of-sample testing
- **Extensible Architecture**: Designed to support additional models (XGBoost, Ridge, Linear)

## 🏗️ Model Architecture

### Training Pipeline
1. **Data Preparation**: Load features and create training datasets
2. **Feature Engineering**: Create technical indicators and transformations
3. **Label Generation**: Define target variables for supervised learning
4. **Cross-Validation**: Time-series aware validation splits
5. **Hyperparameter Tuning**: Optimize model parameters
6. **Model Training**: Train final model on full dataset

### Prediction Pipeline
1. **Feature Extraction**: Generate features for prediction period
2. **Model Inference**: Generate stock scores/ranks
3. **Post-Processing**: Apply any necessary transformations
4. **Output Generation**: Save predictions in required format

## 📊 Data Requirements

### Input Features
- Historical price data (OHLCV)
- Technical indicators and factors
- Fundamental data (if available)
- Market regime indicators

### Output Predictions
Predictions are saved in Parquet format with columns:
- `instrument`: Stock ticker symbol
- `datetime`: Prediction date (UTC timezone-naive)
- `score`: Raw prediction score
- `rank`: Optional rank within time period

## 🚀 Usage Examples

### Train LightGBM Model
```bash
python train.py \
  --features_path ../artifacts/features_day.parquet \
  --output_dir ./models/weekly \
  --start_date 2010-01-01 \
  --end_date 2020-12-31 \
  --frequency weekly
```

### Generate Predictions
```bash
python inference.py \
  --model_path ./models/weekly/model.pkl \
  --features_path ../artifacts/features_day.parquet \
  --output_path ../artifacts/preds/weekly/predictions.parquet \
  --start_date 2021-01-01 \
  --end_date 2024-12-31
```

### Evaluate Model Performance
```bash
python evaluate.py \
  --predictions_path ../artifacts/preds/weekly/predictions.parquet \
  --features_path ../artifacts/features_day.parquet \
  --output_dir ./evaluation_results
```

## ⚙️ Configuration

Model configuration is managed through:
- **Command-line arguments** for training/inference parameters
- **Hyperparameter files** for model-specific settings
- **Global config** in `../config/config.yaml` for paths and defaults

### Key Hyperparameters
- `learning_rate`: Step size shrinkage
- `num_leaves`: Maximum tree leaves
- `max_depth`: Maximum tree depth
- `min_data_in_leaf`: Minimum data in leaves
- `feature_fraction`: Feature sampling ratio
- `bagging_fraction`: Data sampling ratio
- `lambda_l1`, `lambda_l2`: L1 and L2 regularization

## 📈 Performance Metrics

### Training Metrics
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- R-squared
- Feature importance scores

### Trading Metrics (via Backtest)
- Sharpe Ratio
- CAGR (Compound Annual Growth Rate)
- Maximum Drawdown (MDD)
- Information Coefficient (IC)
- Turnover rate
- Cost analysis

## 🛠️ Best Practices

### Data Handling
1. **Avoid Lookahead Bias**: Ensure no future information leakage
2. **Proper Validation**: Use walk-forward or expanding window validation
3. **Feature Stability**: Monitor feature importance over time
4. **Out-of-Sample Testing**: Always test on unseen time periods

### Model Development
1. **Start Simple**: Begin with linear models before complex ensembles
2. **Regularize Appropriately**: Use L1/L2 regularization to prevent overfitting
3. **Monitor Performance**: Track metrics across time periods
4. **Version Control**: Keep track of model versions and hyperparameters

## 🔗 Related Modules

- **[Data](../data/README.md)**: Provides features and raw data
- **[Backtest](../backtest/README.md)**: Evaluates model performance through backtesting
- **[Strategies](../strategies/README.md)**: Uses predictions for trading decisions

## ⚠️ Important Notes

1. **Data Leakage**: Ensure strict time separation between training and testing
2. **Market Regimes**: Models may perform differently in bull/bear markets
3. **Overfitting**: Regular monitoring and validation is essential
4. **Implementation Risk**: Real-world performance may differ from backtests

## 🧪 Testing

Run model tests to ensure functionality:
```bash
python test.py
```

---

*For model development best practices and performance optimization, refer to the main project documentation.*
