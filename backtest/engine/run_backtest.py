#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_backtest.py  (v3.4.1 with diagnostics + short-leg timing + hard-cap water-filling)

重构版本 - 使用模块化结构简化代码
"""

from __future__ import annotations
import sys
from pathlib import Path

# --- fallback: allow running this file directly without -m by injecting project root ---
try:
    _PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))
except Exception:
    pass

# 导入重构后的模块
from backtest.engine.config_loader import ConfigLoader
from backtest.engine.backtest_runner import BacktestRunner


def main():
    """主入口函数 - 简化为协调各个模块"""
    # 解析CLI参数
    args = ConfigLoader.parse_args()
    
    # 加载配置文件
    cfg = ConfigLoader.load_config(args.config)
    
    # 合并配置和参数
    config = ConfigLoader.merge_config_with_args(args, cfg)
    
    # 创建回测运行器
    runner = BacktestRunner(config)
    
    # 准备数据
    data = runner.prepare_data()
    
    # 运行回测
    results = runner.run_backtest(data)
    
    # 保存结果
    runner.save_results(results, data)


if __name__ == "__main__":
    main()
