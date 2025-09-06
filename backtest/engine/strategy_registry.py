"""
策略注册模块 - 避免循环导入问题
"""
from backtest.engine.base_strategy import StrategyFactory
from backtest.engine.strategy import XSecRebalance

# 注册所有策略
StrategyFactory.register_strategy("xsec_rebalance", XSecRebalance)
