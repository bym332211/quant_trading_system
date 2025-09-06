# 策略配置结构说明

## 新的配置结构

系统现在支持分层策略配置，每个策略包含三个主要部分：

### 1. 选股配置 (selection)
- `top_k`: 选股数量（多头）
- `short_k`: 做空数量（空头）  
- `membership_buffer`: 换手缓冲比例
- `use_rank`: 排名使用模式 ("auto" | "rank" | "score")

### 2. 入场策略配置 (entry)
- `neutralize_items`: 中性化项目列表 ["beta", "sector", "size", "liq"]
- `ridge_lambda`: 岭回归参数
- `long_exposure`: 多头暴露度
- `short_exposure`: 空头暴露度
- `max_pos_per_name`: 单票最大仓位
- `weight_scheme`: 权重方案 ("equal" | "icdf")
- `smooth_eta`: 平滑系数
- `target_vol`: 目标波动率
- `leverage_cap`: 杠杆上限
- `hard_cap`: 是否使用硬上限

### 3. 出场策略配置 (exit)
包含两个主要出场策略：

#### 技术指标止损 (tech_stop_loss)
- `atr_multiplier`: ATR乘数
- `atr_period`: ATR周期
- `ma_stop_period`: 移动平均止损周期
- `bollinger_period`: 布林带周期
- `bollinger_std`: 布林带标准差倍数
- `rsi_period`: RSI周期
- `rsi_overbought`: RSI超买阈值
- `rsi_oversold`: RSI超卖阈值
- `enabled_methods`: 启用的止损方法

#### 波动率出场 (volatility_exit)
- `vol_multiplier`: 波动率乘数
- `vol_period`: 波动率计算周期
- `market_vol_period`: 市场波动率周期
- `vol_filter_threshold`: 波动率过滤阈值
- `max_position_days`: 最大持仓天数
- `enabled_methods`: 启用的波动率方法

## 使用方法

### 1. 选择策略
```bash
python backtest/engine/run_backtest.py \
    --strategy_key long_only_baseline \
    --qlib_dir "/path/to/qlib/data" \
    --preds "path/to/predictions.parquet" \
    --features_path "path/to/features.parquet" \
    --start "2017-01-01" \
    --end "2024-12-31" \
    --out_dir "results/long_only"
```

### 2. 覆盖特定参数
```bash
python backtest/engine/run_backtest.py \
    --strategy_key long_short_tv08 \
    --top_k 30 \
    --short_k 10 \
    --target_vol 0.12 \
    --qlib_dir "/path/to/qlib/data" \
    --preds "path/to/predictions.parquet" \
    --features_path "path/to/features.parquet" \
    --start "2017-01-01" \
    --end "2024-12-31" \
    --out_dir "results/long_short_custom"
```

### 3. 添加新策略
在 `config/config.yaml` 中的 `strategies` 部分添加新的策略配置：

```yaml
strategies:
  my_custom_strategy:
    selection:
      top_k: 60
      short_k: 15
      membership_buffer: 0.3
      use_rank: "auto"
    
    entry:
      neutralize_items: ["beta", "sector"]
      ridge_lambda: 1e-6
      long_exposure: 1.0
      short_exposure: -0.7
      max_pos_per_name: 0.04
      weight_scheme: "equal"
      smooth_eta: 0.65
      target_vol: 0.18
      leverage_cap: 2.0
      hard_cap: true
    
    exit:
      tech_stop_loss:
        atr_multiplier: 2.0
        atr_period: 14
        ma_stop_period: 20
        bollinger_period: 20
        bollinger_std: 2.0
        rsi_period: 14
        rsi_overbought: 70.0
        rsi_oversold: 30.0
        enabled_methods: ["atr", "ma", "bollinger"]
      
      volatility_exit:
        vol_multiplier: 2.0
        vol_period: 20
        market_vol_period: 63
        vol_filter_threshold: 0.3
        max_position_days: 50
        enabled_methods: ["vol_stop", "time_exit"]
      
      enabled_strategies: ["tech_stop_loss", "volatility_exit"]
```

## 向后兼容性

系统仍然支持旧的配置格式，如果没有指定 `--strategy_key` 参数，会使用全局默认配置。

## 配置验证

可以使用测试脚本来验证配置：
```bash
python test_config_loading.py
python test_backtest_config.py
```

## 注意事项

1. CLI参数会覆盖配置文件中的参数
2. 确保所有必需的参数都在配置文件中定义
3. 出场策略配置现在完全从配置文件加载，不再需要硬编码
4. 支持动态启用/禁用不同的出场策略方法
