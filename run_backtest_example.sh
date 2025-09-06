#!/bin/bash
# 运行回测的示例命令

# 使用长线基础策略
echo "运行长线基础策略..."
python backtest/engine/run_backtest.py \
    --strategy_key long_only_baseline \
    --qlib_dir "/home/ec2-user/.qlib/qlib_data/us_data" \
    --preds "artifacts/preds/weekly/predictions.parquet" \
    --features_path "artifacts/features_day.parquet" \
    --start "2017-01-01" \
    --end "2024-12-31" \
    --out_dir "backtest/reports/long_only_baseline" \
    --verbose

# 使用长短结合策略
echo "运行长短结合策略..."
python backtest/engine/run_backtest.py \
    --strategy_key long_short_tv08 \
    --qlib_dir "/home/ec2-user/.qlib/qlib_data/us_data" \
    --preds "artifacts/preds/weekly/predictions.parquet" \
    --features_path "artifacts/features_day.parquet" \
    --start "2017-01-01" \
    --end "2024-12-31" \
    --out_dir "backtest/reports/long_short_tv08" \
    --verbose

# 使用ICDF激进策略
echo "运行ICDF激进策略..."
python backtest/engine/run_backtest.py \
    --strategy_key icdf_aggressive \
    --qlib_dir "/home/ec2-user/.qlib/qlib_data/us_data" \
    --preds "artifacts/preds/weekly/predictions.parquet" \
    --features_path "artifacts/features_day.parquet" \
    --start "2017-01-01" \
    --end "2024-12-31" \
    --out_dir "backtest/reports/icdf_aggressive" \
    --verbose

# 使用自定义参数覆盖
echo "运行自定义参数策略..."
python backtest/engine/run_backtest.py \
    --strategy_key long_short_tv08 \
    --top_k 30 \
    --short_k 5 \
    --target_vol 0.10 \
    --qlib_dir "/home/ec2-user/.qlib/qlib_data/us_data" \
    --preds "artifacts/preds/weekly/predictions.parquet" \
    --features_path "artifacts/features_day.parquet" \
    --start "2017-01-01" \
    --end "2024-12-31" \
    --out_dir "backtest/reports/custom_params" \
    --verbose

echo "所有回测任务已启动！"
