# 回测交易问题：只有2024年才有交易

## 问题描述

在运行量化交易回测时，发现只有2024年才有交易记录，2020-2023年没有交易。具体表现为：

- **回测时间范围**: 2020-01-01 到 2024-12-31
- **实际交易年份**: 只有2024年有交易（7198个订单）
- **预期行为**: 所有年份（2020-2024）都应该有交易

## 已排查事项

### ✅ 数据完整性检查
- [x] 预测数据文件 `preds_y5_2020_2024.parquet` 包含2020-2024年完整数据
- [x] 各年份数据分布均匀（2020:124927, 2021:125635, 2022:125480, 2023:125209, 2024:126997）
- [x] 数据准备阶段没有过滤任何年份的数据

### ✅ 执行日期生成检查  
- [x] 周频调度函数 `weekly_schedule` 工作正常
- [x] 全年都有正确的执行日期（53个执行日/年）
- [x] 执行日期范围正确（2020-01-02 到 2024-12-28）

### ✅ 策略逻辑检查
- [x] 单独测试各月份（1月、6月、12月）都有交易
- [x] 策略参数配置正确（top_k=50, membership_buffer=0.35等）
- [x] 策略权重计算逻辑正常

### ✅ 数据准备检查
- [x] 数据准备函数 `prepare_predictions` 没有过滤数据
- [x] 时间戳格式正确，边界检查通过
- [x] 所有年份数据在准备后都完整保留

## 未排查事项 / 可疑点

### 🔍 执行日期映射问题
- [ ] 检查 `asof_map_schedule_to_pred` 函数的映射逻辑
- [ ] 验证调度日到预测日的映射准确性
- [ ] 调试映射过程中是否存在年份不匹配

### 🔍 数据加载时序问题
- [ ] 检查价格数据加载时序（QLib数据可能存在问题）
- [ ] 验证锚定符号（SPY）的交易日期完整性
- [ ] 检查执行延迟（exec_lag）参数的影响

### 🔍 策略执行环境问题
- [ ] 检查回测引擎的全局状态管理
- [ ] 验证策略实例是否在不同年份间正确重置
- [ ] 调试权重平滑参数的影响

### 🔍 数据特征问题
- [ ] 检查特征数据（features_day.parquet）的完整性
- [ ] 验证中性化处理（beta,sector）是否影响选股
- [ ] 分析波动率限制对交易的影响

## ##TODO

### 下一步调查内容

1. **调试执行日期映射**
   - 在 `prepare_execution_dates` 函数中添加详细调试输出
   - 检查每个执行日期映射到的预测日期
   - 验证是否存在年份跳跃或错误映射

2. **检查QLib数据加载**
   - 验证2020-2023年的价格数据是否可用
   - 检查锚定符号SPY在各年份的交易日期
   - 确认数据加载没有时序问题

3. **分析策略状态管理**
   - 检查策略实例是否在年份间正确重置
   - 验证平滑参数是否导致权重缓慢变化
   - 调试策略的初始状态设置

4. **完整回测流程跟踪**
   - 添加端到端的调试输出，跟踪从数据准备到交易执行的完整流程
   - 记录每个执行日的选股结果和权重计算
   - 比较不同年份的执行日志差异

### 调查方法

```python
# 示例调试代码 - 添加到 prepare_execution_dates 函数
print(f"执行日期映射详细检查:")
for sched_date, pred_date in sched2pred.items():
    year_match = "✅" if sched_date.year == pred_date.year else "❌"
    print(f"  {year_match} 调度日: {sched_date} -> 预测日: {pred_date}")
    if sched_date.year != pred_date.year:
        print(f"    警告: 年份不匹配! {sched_date.year} -> {pred_date.year}")
```

## 环境信息

- **项目路径**: `c:/workspace/quant_trading_system`
- **预测数据**: `artifacts/preds/preds_y5_2020_2024.parquet`
- **特征数据**: `artifacts/features_day.parquet`
- **QLib数据**: `~/.qlib/qlib_data/us_data`
- **回测引擎**: `backtest/engine/run_backtest.py`

## 相关文件

- `strategies/xsecrebalance/data_preparation.py` - 数据准备逻辑
- `backtest/engine/data_loader.py` - 数据加载和日期映射
- `strategies/entry_strategies/icdf_equal_strategy.py` - 策略逻辑

---

**每次执行调查后更新此文档，记录新的发现和待办事项**
