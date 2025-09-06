# strategies/exit_strategies/__init__.py
# -*- coding: utf-8 -*-
"""
出场策略模块 - 支持多种出场策略的协调执行

包含：
- 技术指标止损策略
- 波动率动态出场策略
- 主协调器：支持多个策略同时运行
"""

from __future__ import annotations
from typing import Dict, List, Optional, Protocol
import pandas as pd
import numpy as np


class ExitStrategy(Protocol):
    """出场策略接口"""
    
    def should_exit(self, 
                   symbol: str, 
                   current_price: float, 
                   entry_price: float, 
                   position_size: float,
                   historical_data: pd.DataFrame,
                   current_date: pd.Timestamp) -> bool:
        """判断是否应该出场"""
        ...
    
    def get_name(self) -> str:
        """获取策略名称"""
        ...

# 导出具体的策略类
from .tech_stop_loss import TechStopLossStrategy
from .volatility_exit import VolatilityExitStrategy
from .coordinator import ExitStrategyCoordinator

__all__ = [
    'ExitStrategy',
    'TechStopLossStrategy',
    'VolatilityExitStrategy',
    'ExitStrategyCoordinator'
]
