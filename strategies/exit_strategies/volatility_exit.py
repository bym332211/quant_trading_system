# strategies/exit_strategies/volatility_exit.py
# -*- coding: utf-8 -*-
"""
波动率动态出场策略
基于波动率调整出场条件：
- 历史波动率止损
- 市场波动率因子调整
- 波动率过滤机制
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from . import ExitStrategy


class VolatilityExitStrategy:
    """波动率动态出场策略"""
    
    def __init__(self,
                 vol_multiplier: float = 2.0,
                 vol_period: int = 20,
                 market_vol_period: int = 63,
                 vol_filter_threshold: float = 0.3,
                 max_position_days: int = 63,
                 enabled_methods: list = None):
        """
        初始化波动率出场策略
        
        参数:
            vol_multiplier: 波动率止损乘数
            vol_period: 波动率计算周期
            market_vol_period: 市场波动率计算周期
            vol_filter_threshold: 波动率过滤阈值
            max_position_days: 最大持仓天数
            enabled_methods: 启用的波动率方法列表
        """
        self.vol_multiplier = vol_multiplier
        self.vol_period = vol_period
        self.market_vol_period = market_vol_period
        self.vol_filter_threshold = vol_filter_threshold
        self.max_position_days = max_position_days
        self.enabled_methods = enabled_methods or ["vol_stop", "market_vol", "vol_filter", "time_exit"]
        
        # 缓存计算结果
        self._vol_cache: Dict[str, float] = {}
        self._market_vol_cache: Dict[str, float] = {}
        self._position_days: Dict[str, int] = {}
    
    def get_name(self) -> str:
        return "volatility_exit"
    
    def should_exit(self, 
                   symbol: str, 
                   current_price: float, 
                   entry_price: float, 
                   position_size: float,
                   historical_data: pd.DataFrame,
                   current_date: pd.Timestamp) -> bool:
        """
        判断是否应该基于波动率出场
        
        参数:
            symbol: 股票代码
            current_price: 当前价格
            entry_price: 入场价格
            position_size: 持仓数量（正数为多头，负数为空头）
            historical_data: 历史价格数据
            current_date: 当前日期
            
        返回:
            bool: 是否应该出场
        """
        if historical_data.empty or len(historical_data) < max(self.vol_period, self.market_vol_period):
            return False
        
        # 更新持仓天数
        self._update_position_days(symbol, current_date)
        
        # 多头和空头使用不同的出场逻辑
        is_long = position_size > 0
        
        exit_signals = []
        
        # 波动率止损
        if "vol_stop" in self.enabled_methods:
            vol_exit = self._check_volatility_stop(symbol, current_price, entry_price, is_long, historical_data)
            exit_signals.append(vol_exit)
        
        # 市场波动率调整
        if "market_vol" in self.enabled_methods:
            market_vol_exit = self._check_market_volatility(symbol, is_long, historical_data)
            exit_signals.append(market_vol_exit)
        
        # 波动率过滤
        if "vol_filter" in self.enabled_methods:
            vol_filter_exit = self._check_volatility_filter(symbol, historical_data)
            exit_signals.append(vol_filter_exit)
        
        # 时间出场
        if "time_exit" in self.enabled_methods:
            time_exit = self._check_time_exit(symbol)
            exit_signals.append(time_exit)
        
        # 任一信号触发即出场
        return any(exit_signals)
    
    def _update_position_days(self, symbol: str, current_date: pd.Timestamp):
        """更新持仓天数"""
        if symbol not in self._position_days:
            self._position_days[symbol] = 1
        else:
            self._position_days[symbol] += 1
    
    def _check_volatility_stop(self, symbol: str, current_price: float, entry_price: float, 
                              is_long: bool, data: pd.DataFrame) -> bool:
        """检查波动率止损"""
        vol = self._calculate_volatility(symbol, data)
        if vol is None:
            return False
        
        stop_distance = vol * self.vol_multiplier
        
        if is_long:
            # 多头：当前价格跌破入场价 - 波动率×乘数
            return current_price <= (entry_price - stop_distance)
        else:
            # 空头：当前价格突破入场价 + 波动率×乘数
            return current_price >= (entry_price + stop_distance)
    
    def _calculate_volatility(self, symbol: str, data: pd.DataFrame) -> Optional[float]:
        """计算历史波动率"""
        if symbol in self._vol_cache:
            return self._vol_cache[symbol]
        
        try:
            returns = data["close"].astype(float).pct_change().dropna()
            if len(returns) < self.vol_period:
                return None
            
            # 计算年化波动率
            daily_vol = returns.rolling(window=self.vol_period).std().iloc[-1]
            annual_vol = daily_vol * np.sqrt(252)
            
            # 转换为价格波动幅度
            price_vol = data["close"].iloc[-1] * annual_vol
            
            self._vol_cache[symbol] = price_vol
            return price_vol
        except:
            return None
    
    def _check_market_volatility(self, symbol: str, is_long: bool, data: pd.DataFrame) -> bool:
        """检查市场波动率调整"""
        market_vol = self._calculate_market_volatility(symbol, data)
        if market_vol is None:
            return False
        
        # 高市场波动率环境下，收紧出场条件
        # 这里可以根据具体策略调整阈值
        high_vol_threshold = 0.25  # 25%的年化波动率视为高波动
        
        if market_vol > high_vol_threshold:
            # 在高波动环境下，如果持仓与市场方向相反，考虑出场
            # 这里简化处理，实际可以根据更复杂的逻辑
            return True
        
        return False
    
    def _calculate_market_volatility(self, symbol: str, data: pd.DataFrame) -> Optional[float]:
        """计算市场波动率（使用SPY作为市场代表）"""
        # 这里简化处理，实际应该使用市场指数数据
        # 假设data中包含市场数据或使用外部数据源
        try:
            # 使用自身数据作为代理（实际应用中应该使用市场指数）
            returns = data["close"].astype(float).pct_change().dropna()
            if len(returns) < self.market_vol_period:
                return None
            
            daily_vol = returns.rolling(window=self.market_vol_period).std().iloc[-1]
            annual_vol = daily_vol * np.sqrt(252)
            
            self._market_vol_cache[symbol] = annual_vol
            return annual_vol
        except:
            return None
    
    def _check_volatility_filter(self, symbol: str, data: pd.DataFrame) -> bool:
        """检查波动率过滤"""
        vol = self._calculate_volatility(symbol, data)
        if vol is None:
            return False
        
        # 如果波动率超过阈值，考虑出场
        current_price = data["close"].iloc[-1]
        vol_ratio = vol / current_price
        
        return vol_ratio > self.vol_filter_threshold
    
    def _check_time_exit(self, symbol: str) -> bool:
        """检查时间出场"""
        if symbol in self._position_days:
            return self._position_days[symbol] >= self.max_position_days
        return False
    
    def reset_position_days(self, symbol: str):
        """重置持仓天数（平仓时调用）"""
        if symbol in self._position_days:
            del self._position_days[symbol]
    
    def clear_cache(self):
        """清空缓存"""
        self._vol_cache.clear()
        self._market_vol_cache.clear()
        self._position_days.clear()
