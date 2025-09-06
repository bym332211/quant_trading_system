# strategies/entry_strategies/__init__.py
# -*- coding: utf-8 -*-
"""
入场策略模块 - 支持多种入场策略的协调执行

包含：
- 基础入场策略接口
- 多种入场策略实现
- 主协调器：支持多个策略同时运行
"""

from __future__ import annotations
from typing import Dict, List, Optional, Protocol, Tuple
import pandas as pd
import numpy as np


class EntryStrategy(Protocol):
    """入场策略接口"""
    
    def generate_entry_weights(self,
                             g: pd.DataFrame,
                             prev_weights: Dict[str, float],
                             expos_df: Optional[pd.DataFrame] = None,
                             vol_df: Optional[pd.DataFrame] = None,
                             allow_shorts: bool = True,
                             reb_counter: Optional[int] = None) -> Dict[str, float]:
        """生成入场权重"""
        ...
    
    def get_name(self) -> str:
        """获取策略名称"""
        ...

# 导出具体的策略类
from .icdf_equal_strategy import ICDFEqualStrategy
from .coordinator import EntryStrategyCoordinator

__all__ = [
    'EntryStrategy',
    'ICDFEqualStrategy',
    'EntryStrategyCoordinator'
]
