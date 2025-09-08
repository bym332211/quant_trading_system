"""
策略抽象基类模块 - 定义策略接口和工厂模式
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple, Type
import backtrader as bt
import pandas as pd


class BaseStrategy(ABC):
    """策略抽象基类 - 包含完整的交易流程"""
    
    @abstractmethod
    def prepare_data(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        策略特定的数据准备
        
        Args:
            config: 策略配置
            
        Returns:
            准备好的数据字典
        """
        pass
    
    @abstractmethod
    def prepare(self, config: Dict[str, Any], data: Dict[str, Any]) -> None:
        """
        准备阶段：根据配置加载所有子模块
        
        Args:
            config: 策略配置字典
            data: 回测数据字典
        """
        pass
    
    @abstractmethod
    def entry_strategy(self, dtoday: pd.Timestamp, preds_df: pd.DataFrame, 
                      prev_weights: Dict[str, float]) -> Dict[str, float]:
        """
        入场策略：生成初步目标权重
        
        Args:
            dtoday: 当前日期
            preds_df: 预测数据DataFrame
            prev_weights: 上一期权重
            
        Returns:
            目标权重字典 {symbol: weight}
        """
        pass
    
    @abstractmethod
    def exit_strategy(self, dtoday: pd.Timestamp) -> List[Tuple[str, List[str]]]:
        """
        出场策略：检查是否需要出场
        
        Args:
            dtoday: 当前日期
            
        Returns:
            需要出场的标的及触发策略列表 [(symbol, [strategy_names])]
        """
        pass
    
    @abstractmethod
    def risk_management(self, tgt_weights: Dict[str, float], 
                       dtoday: pd.Timestamp) -> Dict[str, float]:
        """
        风险管理：应用风控规则
        
        Args:
            tgt_weights: 目标权重
            dtoday: 当前日期
            
        Returns:
            经过风控处理后的权重
        """
        pass
    
    @abstractmethod
    def position_sizing(self, tgt_weights: Dict[str, float],
                       dtoday: pd.Timestamp, port_value: float) -> Dict[str, float]:
        """
        仓位管理：最终仓位确定
        
        Args:
            tgt_weights: 目标权重
            dtoday: 当前日期
            port_value: 组合价值
            
        Returns:
            最终确定的仓位权重
        """
        pass
    
    @abstractmethod
    def get_diagnostics(self) -> Dict[str, Any]:
        """
        获取诊断信息
        
        Returns:
            各子模块的诊断统计信息
        """
        pass
    
    @abstractmethod
    def update_state(self, new_weights: Dict[str, float]) -> None:
        """
        更新策略状态
        
        Args:
            new_weights: 新的权重配置
        """
        pass


class StrategyFactory:
    """策略工厂 - 根据配置创建策略实例"""
    
    _strategies: Dict[str, Type[BaseStrategy]] = {}
    
    @classmethod
    def create_strategy(cls, strategy_config: Dict[str, Any]) -> Type[BaseStrategy]:
        """
        获取策略类
        
        Args:
            strategy_config: 策略配置
            
        Returns:
            策略类
            
        Raises:
            ValueError: 如果策略类型未知
        """
        strategy_name = strategy_config.get("name", "xsec_rebalance")
        
        if strategy_name not in cls._strategies:
            raise ValueError(f"未知策略类型: {strategy_name}")
        
        return cls._strategies[strategy_name]
    
    @classmethod
    def register_strategy(cls, name: str, strategy_class: Type[BaseStrategy]) -> None:
        """
        注册新策略类型
        
        Args:
            name: 策略名称
            strategy_class: 策略类
        """
        cls._strategies[name] = strategy_class
    
    @classmethod
    def list_strategies(cls) -> List[str]:
        """列出所有已注册的策略"""
        return list(cls._strategies.keys())
