# -*- coding: utf-8 -*-
"""
出场策略协调器
负责协调多个出场策略的执行
支持同时运行多个出场策略，任一策略触发即出场
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from . import ExitStrategy
from .tech_stop_loss import TechStopLossStrategy
from .volatility_exit import VolatilityExitStrategy


class ExitStrategyCoordinator:
    """出场策略协调器"""
    
    def __init__(self,
                 tech_stop_loss_config: Optional[Dict] = None,
                 volatility_exit_config: Optional[Dict] = None,
                 enabled_strategies: List[str] = None):
        self.enabled_strategies = enabled_strategies or [
            # "tech_stop_loss", 
            "volatility_exit"]
        self.strategies: Dict[str, ExitStrategy] = {}
        
        if "tech_stop_loss" in self.enabled_strategies:
            tech_config = tech_stop_loss_config or {}
            self.strategies["tech_stop_loss"] = TechStopLossStrategy(**tech_config)
        
        if "volatility_exit" in self.enabled_strategies:
            vol_config = volatility_exit_config or {}
            self.strategies["volatility_exit"] = VolatilityExitStrategy(**vol_config)
        
        # 入场记录
        self.entry_prices: Dict[str, float] = {}
        self.entry_dates: Dict[str, pd.Timestamp] = {}
        self.position_sizes: Dict[str, float] = {}
    
    def record_entry(self, symbol: str, entry_price: float, 
                     position_size: float, entry_date: pd.Timestamp):
        self.entry_prices[symbol] = float(entry_price)
        self.entry_dates[symbol] = pd.Timestamp(entry_date)
        self.position_sizes[symbol] = float(position_size)
    
    def record_exit(self, symbol: str):
        # 清理本地记录
        self.entry_prices.pop(symbol, None)
        self.entry_dates.pop(symbol, None)
        self.position_sizes.pop(symbol, None)
        
        # 通知各策略重置/清理 ✅ 属性名加引号，并用 getattr 调用
        for strategy in self.strategies.values():
            reset_fn = getattr(strategy, "reset_position_days", None)
            if callable(reset_fn):
                reset_fn(symbol)
            clear_fn = getattr(strategy, "clear_cache", None)
            if callable(clear_fn):
                clear_fn()
    
    def should_exit(self, 
                    symbol: str, 
                    current_price: float, 
                    historical_data: pd.DataFrame,
                    current_date: pd.Timestamp) -> Tuple[bool, List[str]]:
        # 无入场记录则不判出场
        if symbol not in self.entry_prices:
            return False, []
        
        entry_price = self.entry_prices[symbol]
        position_size = self.position_sizes.get(symbol, 0.0)
        
        triggered: List[str] = []
        for name, strategy in self.strategies.items():
            try:
                if strategy.should_exit(
                    symbol=symbol,
                    current_price=float(current_price),
                    entry_price=float(entry_price),
                    position_size=float(position_size),
                    historical_data=historical_data,
                    current_date=pd.Timestamp(current_date),
                ):
                    triggered.append(name)
            except Exception as e:
                print(f"[exit] Error executing {name} for {symbol}: {e}")
                continue
        
        return (len(triggered) > 0), triggered
    
    def get_strategy_status(self, symbol: str) -> Dict[str, Any]:
        status: Dict[str, Any] = {}
        if symbol in self.entry_prices:
            status.update({
                "entry_price": self.entry_prices[symbol],
                "position_size": self.position_sizes.get(symbol, 0.0),
                "entry_date": self.entry_dates.get(symbol),
            })
        
        # 各策略内部缓存（仅在存在时返回） ✅ 属性名加引号
        for name, strategy in self.strategies.items():
            s: Dict[str, Any] = {}
            if name == "tech_stop_loss":
                if hasattr(strategy, "_atr_cache") and symbol in strategy._atr_cache:
                    s["atr"] = strategy._atr_cache[symbol]
                if hasattr(strategy, "_ma_cache") and symbol in strategy._ma_cache:
                    s["ma"] = strategy._ma_cache[symbol]
                if hasattr(strategy, "_bollinger_cache") and symbol in strategy._bollinger_cache:
                    s["bollinger"] = strategy._bollinger_cache[symbol]
                if hasattr(strategy, "_rsi_cache") and symbol in strategy._rsi_cache:
                    s["rsi"] = strategy._rsi_cache[symbol]
            elif name == "volatility_exit":
                if hasattr(strategy, "_vol_cache") and symbol in strategy._vol_cache:
                    s["volatility"] = strategy._vol_cache[symbol]
                if hasattr(strategy, "_market_vol_cache") and symbol in strategy._market_vol_cache:
                    s["market_vol"] = strategy._market_vol_cache[symbol]
                if hasattr(strategy, "_position_days") and symbol in strategy._position_days:
                    s["position_days"] = strategy._position_days[symbol]
            status[name] = s
        return status
    
    def clear_all_cache(self):
        self.entry_prices.clear()
        self.entry_dates.clear()
        self.position_sizes.clear()
        for strategy in self.strategies.values():
            clear_fn = getattr(strategy, "clear_cache", None)
            if callable(clear_fn):
                clear_fn()
