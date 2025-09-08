# -*- coding: utf-8 -*-
"""
入场策略协调器
负责协调多个入场策略的执行
支持同时运行多个入场策略，按权重组合结果
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from . import EntryStrategy
from .icdf_equal_strategy import ICDFEqualStrategy, waterfill_two_legs


class EntryStrategyCoordinator:
    """入场策略协调器"""
    
    def __init__(self,
                 icdf_equal_config: Optional[Dict] = None,
                 enabled_strategies: List[str] = None,
                 strategy_weights: Optional[Dict[str, float]] = None,
                 # 组合后归一（可选）
                 post_normalize: bool = False,
                 long_exposure: float = 1.0,
                 short_exposure: float = -1.0,
                 max_pos_per_name: float = 0.05):
        self.enabled_strategies = enabled_strategies or ["icdf_equal"]
        self.strategy_weights = strategy_weights or {"icdf_equal": 1.0}
        self.post_normalize = bool(post_normalize)
        self._long_exposure = float(long_exposure)
        self._short_exposure = float(short_exposure)
        self._max_pos = float(max_pos_per_name)
        self.strategies: Dict[str, EntryStrategy] = {}
        
        if "icdf_equal" in self.enabled_strategies:
            config = icdf_equal_config or {}
            self.strategies["icdf_equal"] = ICDFEqualStrategy(**config)
        
        # 状态变量
        self.prev_weights: Dict[str, float] = {}
        self.reb_counter = 0
    
    def generate_entry_weights(self,
                              g: pd.DataFrame,
                              prev_weights: Dict[str, float],
                              expos_df: Optional[pd.DataFrame] = None,
                              vol_df: Optional[pd.DataFrame] = None,
                              allow_shorts: bool = True,
                              reb_counter: Optional[int] = None) -> Dict[str, float]:
        """
        生成入场权重 - 协调多个策略的结果
        
        参数:
            g: 预测数据DataFrame
            prev_weights: 上一期权重
            expos_df: 暴露数据DataFrame
            vol_df: 波动率数据DataFrame
            allow_shorts: 是否允许做空
            reb_counter: 调仓计数器
            
        返回:
            目标权重字典（多个策略的加权组合）
        """
        if reb_counter is not None:
            self.reb_counter = reb_counter
        
        # 收集所有策略的结果
        strategy_results = {}
        for name, strategy in self.strategies.items():
            if name in self.enabled_strategies:
                try:
                    weights = strategy.generate_entry_weights(
                        g=g,
                        prev_weights=prev_weights,
                        expos_df=expos_df,
                        vol_df=vol_df,
                        allow_shorts=allow_shorts,
                        reb_counter=self.reb_counter,
                    )
                    strategy_results[name] = weights
                except Exception as e:
                    print(f"[entry] Error executing {name} strategy: {e}")
                    continue
        
        # 如果没有策略成功执行，返回空权重
        if not strategy_results:
            return {}
        
        # 如果只有一个策略，直接返回其结果
        if len(strategy_results) == 1:
            result = next(iter(strategy_results.values()))
            self.prev_weights = result.copy()
            self.reb_counter += 1
            return result
        
        # 多个策略：按权重组合结果
        combined_weights = self._combine_strategy_weights(strategy_results)

        # 组合后归一（可选）：确保两腿目标敞口与单名上限
        if self.post_normalize:
            combined_weights = waterfill_two_legs(
                combined_weights,
                long_exposure=self._long_exposure,
                short_exposure=self._short_exposure,
                max_pos_per_name=self._max_pos,
                allow_shorts=allow_shorts,
            )
        
        # 更新状态
        self.prev_weights = combined_weights.copy()
        self.reb_counter += 1
        
        return combined_weights
    
    def _combine_strategy_weights(self, strategy_results: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """组合多个策略的权重结果"""
        # 收集所有股票代码
        all_symbols = set()
        for weights in strategy_results.values():
            all_symbols.update(weights.keys())
        
        # 计算加权组合
        combined = {}
        total_weight = sum(self.strategy_weights.get(name, 0.0) for name in strategy_results.keys())
        
        if total_weight <= 0:
            # 如果权重总和为0或负数，使用等权重
            strategy_names = list(strategy_results.keys())
            weight_per_strategy = 1.0 / len(strategy_names) if strategy_names else 0.0
            
            for symbol in all_symbols:
                weighted_sum = 0.0
                for name in strategy_names:
                    weighted_sum += strategy_results[name].get(symbol, 0.0) * weight_per_strategy
                combined[symbol] = weighted_sum
        else:
            # 使用配置的权重
            for symbol in all_symbols:
                weighted_sum = 0.0
                for name, weights in strategy_results.items():
                    strategy_weight = self.strategy_weights.get(name, 0.0) / total_weight
                    weighted_sum += weights.get(symbol, 0.0) * strategy_weight
                combined[symbol] = weighted_sum
        
        return combined
    
    def get_strategy_status(self) -> Dict[str, Any]:
        """获取策略状态信息"""
        status = {
            "enabled_strategies": self.enabled_strategies,
            "strategy_weights": self.strategy_weights,
            "reb_counter": self.reb_counter,
        }
        
        # 添加各策略的状态
        for name, strategy in self.strategies.items():
            status[name] = {
                "prev_weights_count": len(getattr(strategy, 'prev_weights', {})),
            }
        
        return status
    
    def clear_all_cache(self):
        """清理所有缓存"""
        self.prev_weights.clear()
        for strategy in self.strategies.values():
            if hasattr(strategy, 'prev_weights'):
                strategy.prev_weights.clear()
