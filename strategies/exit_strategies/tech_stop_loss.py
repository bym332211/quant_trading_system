# strategies/exit_strategies/tech_stop_loss.py
# -*- coding: utf-8 -*-
"""
技术指标止损策略
包含多种技术指标止损方法：
- ATR止损
- 移动平均线止损  
- 布林带止损
- RSI超买超卖止损
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, Optional
from . import ExitStrategy


class TechStopLossStrategy:
    """技术指标止损策略"""
    
    def __init__(self,
                 atr_multiplier: float = 2.0,
                 atr_period: int = 14,
                 ma_stop_period: int = 20,
                 bollinger_period: int = 20,
                 bollinger_std: float = 2.0,
                 rsi_period: int = 14,
                 rsi_overbought: float = 70.0,
                 rsi_oversold: float = 30.0,
                 enabled_methods: list = None):
        """
        初始化技术指标止损策略
        
        参数:
            atr_multiplier: ATR止损乘数
            atr_period: ATR计算周期
            ma_stop_period: 移动平均线止损周期
            bollinger_period: 布林带周期
            bollinger_std: 布林带标准差倍数
            rsi_period: RSI计算周期
            rsi_overbought: RSI超买阈值
            rsi_oversold: RSI超卖阈值
            enabled_methods: 启用的止损方法列表
        """
        self.atr_multiplier = atr_multiplier
        self.atr_period = atr_period
        self.ma_stop_period = ma_stop_period
        self.bollinger_period = bollinger_period
        self.bollinger_std = bollinger_std
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.enabled_methods = enabled_methods or ["atr", "ma", "bollinger", "rsi"]
        
        # 缓存计算结果
        self._atr_cache: Dict[str, float] = {}
        self._ma_cache: Dict[str, float] = {}
        self._bollinger_cache: Dict[str, tuple] = {}
        self._rsi_cache: Dict[str, float] = {}
    
    def get_name(self) -> str:
        return "tech_stop_loss"
    
    def should_exit(self, 
                   symbol: str, 
                   current_price: float, 
                   entry_price: float, 
                   position_size: float,
                   historical_data: pd.DataFrame,
                   current_date: pd.Timestamp) -> bool:
        """
        判断是否应该基于技术指标止损出场
        
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
        if historical_data.empty or len(historical_data) < max(self.atr_period, self.ma_stop_period, self.bollinger_period, self.rsi_period):
            return False
        
        # 多头和空头使用不同的止损逻辑
        is_long = position_size > 0
        
        exit_signals = []
        
        # ATR止损
        if "atr" in self.enabled_methods:
            atr_exit = self._check_atr_stop(symbol, current_price, entry_price, is_long, historical_data)
            exit_signals.append(atr_exit)
        
        # 移动平均线止损
        if "ma" in self.enabled_methods:
            ma_exit = self._check_ma_stop(symbol, current_price, is_long, historical_data)
            exit_signals.append(ma_exit)
        
        # 布林带止损
        if "bollinger" in self.enabled_methods:
            bollinger_exit = self._check_bollinger_stop(symbol, current_price, is_long, historical_data)
            exit_signals.append(bollinger_exit)
        
        # RSI止损
        if "rsi" in self.enabled_methods:
            rsi_exit = self._check_rsi_stop(symbol, current_price, is_long, historical_data)
            exit_signals.append(rsi_exit)
        
        # 任一信号触发即出场
        return any(exit_signals)
    
    def _check_atr_stop(self, symbol: str, current_price: float, entry_price: float, 
                       is_long: bool, data: pd.DataFrame) -> bool:
        """检查ATR止损"""
        atr = self._calculate_atr(symbol, data)
        if atr is None:
            return False
        
        stop_distance = atr * self.atr_multiplier
        
        if is_long:
            # 多头：当前价格跌破入场价 - ATR×乘数
            return current_price <= (entry_price - stop_distance)
        else:
            # 空头：当前价格突破入场价 + ATR×乘数
            return current_price >= (entry_price + stop_distance)
    
    def _calculate_atr(self, symbol: str, data: pd.DataFrame) -> Optional[float]:
        """计算ATR"""
        if symbol in self._atr_cache:
            return self._atr_cache[symbol]
        
        try:
            high = data["high"].astype(float)
            low = data["low"].astype(float)
            close = data["close"].astype(float)
            
            # 计算真实波幅
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # 计算ATR
            atr = tr.rolling(window=self.atr_period).mean().iloc[-1]
            self._atr_cache[symbol] = atr
            return atr
        except:
            return None
    
    def _check_ma_stop(self, symbol: str, current_price: float, is_long: bool, data: pd.DataFrame) -> bool:
        """检查移动平均线止损"""
        ma = self._calculate_ma(symbol, data)
        if ma is None:
            return False
        
        if is_long:
            # 多头：价格跌破移动平均线
            return current_price < ma
        else:
            # 空头：价格突破移动平均线
            return current_price > ma
    
    def _calculate_ma(self, symbol: str, data: pd.DataFrame) -> Optional[float]:
        """计算移动平均线"""
        if symbol in self._ma_cache:
            return self._ma_cache[symbol]
        
        try:
            close = data["close"].astype(float)
            ma = close.rolling(window=self.ma_stop_period).mean().iloc[-1]
            self._ma_cache[symbol] = ma
            return ma
        except:
            return None
    
    def _check_bollinger_stop(self, symbol: str, current_price: float, is_long: bool, data: pd.DataFrame) -> bool:
        """检查布林带止损"""
        bollinger = self._calculate_bollinger(symbol, data)
        if bollinger is None:
            return False
        
        upper, lower = bollinger
        
        if is_long:
            # 多头：价格跌破布林带下轨
            return current_price < lower
        else:
            # 空头：价格突破布林带上轨
            return current_price > upper
    
    def _calculate_bollinger(self, symbol: str, data: pd.DataFrame) -> Optional[tuple]:
        """计算布林带"""
        if symbol in self._bollinger_cache:
            return self._bollinger_cache[symbol]
        
        try:
            close = data["close"].astype(float)
            ma = close.rolling(window=self.bollinger_period).mean()
            std = close.rolling(window=self.bollinger_period).std()
            
            upper = ma.iloc[-1] + std.iloc[-1] * self.bollinger_std
            lower = ma.iloc[-1] - std.iloc[-1] * self.bollinger_std
            
            result = (upper, lower)
            self._bollinger_cache[symbol] = result
            return result
        except:
            return None
    
    def _check_rsi_stop(self, symbol: str, current_price: float, is_long: bool, data: pd.DataFrame) -> bool:
        """检查RSI止损"""
        rsi = self._calculate_rsi(symbol, data)
        if rsi is None:
            return False
        
        if is_long:
            # 多头：RSI超买
            return rsi >= self.rsi_overbought
        else:
            # 空头：RSI超卖
            return rsi <= self.rsi_oversold
    
    def _calculate_rsi(self, symbol: str, data: pd.DataFrame) -> Optional[float]:
        """计算RSI"""
        if symbol in self._rsi_cache:
            return self._rsi_cache[symbol]
        
        try:
            close = data["close"].astype(float)
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
            
            rs = gain.iloc[-1] / loss.iloc[-1] if loss.iloc[-1] != 0 else 0
            rsi = 100 - (100 / (1 + rs))
            
            self._rsi_cache[symbol] = rsi
            return rsi
        except:
            return None
