"""
配置加载模块 - 提取自 run_backtest.py 的配置处理逻辑
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
import yaml


class ConfigLoader:
    """处理CLI参数和YAML配置的加载与合并"""
    
    @staticmethod
    def parse_args() -> argparse.Namespace:
        """解析CLI参数"""
        ap = argparse.ArgumentParser()
        ap.add_argument("--config", default="config/config.yaml", help="YAML 配置文件路径（可选）")
        ap.add_argument("--strategy_key", default=None, help="从 config.strategies 选择一个 key 覆盖参数")

        ap.add_argument("--qlib_dir", required=True)
        ap.add_argument("--preds", required=True)
        ap.add_argument("--features_path", required=True)
        ap.add_argument("--start", required=True)
        ap.add_argument("--end", required=True)

        ap.add_argument("--trade_at", choices=["open","close"], default="open")
        ap.add_argument("--exec_lag", type=int, default=0, help="执行延迟 N 个交易日（0=同日）")

        ap.add_argument("--neutralize", default="", help="beta,sector,liq,size")
        ap.add_argument("--ridge_lambda", type=float, default=1e-6)

        # 由 config 决定，CLI 仅在传入时覆盖
        ap.add_argument("--top_k", type=int, default=None)
        ap.add_argument("--short_k", type=int, default=None)
        ap.add_argument("--membership_buffer", type=float, default=None)

        ap.add_argument("--long_exposure", type=float, default=1.0)
        ap.add_argument("--short_exposure", type=float, default=-1.0)
        ap.add_argument("--max_pos_per_name", type=float, default=0.05)
        ap.add_argument("--weight_scheme", choices=["equal","icdf"], default="equal")

        ap.add_argument("--smooth_eta", type=float, default=0.6)

        ap.add_argument("--target_vol", type=float, default=0.0, help="年化目标波动 (0 关闭)")
        ap.add_argument("--ewm_halflife", type=int, default=20, help="EWM 半衰期（用于 sigma 估计）")
        ap.add_argument("--leverage_cap", type=float, default=2.0)
        ap.add_argument("--adv_limit_pct", type=float, default=0.0, help="单次换手 ADV 限制，例如 0.005=0.5%%ADV")

        # 成本与撮合
        ap.add_argument("--commission_bps", type=float, default=1.0)
        ap.add_argument("--slippage_bps", type=float, default=5.0)
        ap.add_argument("--cash", type=float, default=1_000_000.0)
        ap.add_argument("--anchor_symbol", default="SPY")

        # NEW: 短腿择时参数
        ap.add_argument("--short_timing_mom63", action="store_true")
        ap.add_argument("--short_timing_threshold", type=float, default=0.0)
        ap.add_argument("--short_timing_lookback", type=int, default=63)

        # NEW: 硬上限开关
        ap.add_argument("--hard_cap", action="store_true")

        ap.add_argument("--out_dir", required=True)
        ap.add_argument("--verbose", action="store_true")
        return ap.parse_args()

    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """加载YAML配置文件"""
        cfg_path = Path(config_path).expanduser()
        if cfg_path.exists():
            with open(cfg_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}

    @staticmethod
    def merge_config_with_args(args: argparse.Namespace, cfg: Dict[str, Any]) -> Dict[str, Any]:
        """合并CLI参数和配置文件"""
        strategies_cfg = cfg.get("strategies") or {}
        active_strategy_key = args.strategy_key
        
        # 如果没有指定策略key，使用第一个策略或默认配置
        if not active_strategy_key and strategies_cfg:
            active_strategy_key = list(strategies_cfg.keys())[0]
        
        # 获取选股配置
        sel_cfg = {}
        if active_strategy_key and active_strategy_key in strategies_cfg:
            strategy_cfg = strategies_cfg[active_strategy_key]
            sel_cfg = strategy_cfg.get("selection", {})
        else:
            # 向后兼容：使用全局默认配置
            sel_cfg = cfg.get("selection", {})
        
        # CLI参数覆盖配置
        if args.top_k is not None:
            sel_cfg["top_k"] = int(args.top_k)
        if args.short_k is not None:
            sel_cfg["short_k"] = int(args.short_k)
        if args.membership_buffer is not None:
            sel_cfg["membership_buffer"] = float(args.membership_buffer)

        # 获取入场策略配置
        entry_cfg = {}
        if active_strategy_key and active_strategy_key in strategies_cfg:
            strategy_cfg = strategies_cfg[active_strategy_key]
            entry_cfg = strategy_cfg.get("entry_strategies", {})
        
        # 获取出场策略配置
        exit_cfg = {}
        if active_strategy_key and active_strategy_key in strategies_cfg:
            strategy_cfg = strategies_cfg[active_strategy_key]
            exit_cfg = strategy_cfg.get("exit_strategies", {})
        
        # 合并CLI参数与配置
        neutralize_items = entry_cfg.get("neutralize_items", [])
        ridge_lambda = entry_cfg.get("ridge_lambda", 1e-6)
        long_exposure = entry_cfg.get("long_exposure", 1.0)
        short_exposure = entry_cfg.get("short_exposure", -1.0)
        max_pos_per_name = entry_cfg.get("max_pos_per_name", 0.05)
        weight_scheme = entry_cfg.get("weight_scheme", "equal")
        smooth_eta = entry_cfg.get("smooth_eta", 0.6)
        target_vol = entry_cfg.get("target_vol", 0.0)
        leverage_cap = entry_cfg.get("leverage_cap", 2.0)
        hard_cap = entry_cfg.get("hard_cap", False)
        
        # CLI参数覆盖
        if args.neutralize:
            neutralize_items = [s.strip().lower() for s in args.neutralize.split(",") if s.strip()]
        if args.ridge_lambda is not None:
            ridge_lambda = args.ridge_lambda
        if args.long_exposure is not None:
            long_exposure = args.long_exposure
        if args.short_exposure is not None:
            short_exposure = args.short_exposure
        if args.max_pos_per_name is not None:
            max_pos_per_name = args.max_pos_per_name
        if args.weight_scheme:
            weight_scheme = args.weight_scheme
        if args.smooth_eta is not None:
            smooth_eta = args.smooth_eta
        if args.target_vol is not None:
            target_vol = args.target_vol
        if args.leverage_cap is not None:
            leverage_cap = args.leverage_cap
        if args.hard_cap:
            hard_cap = True

        # 构建策略配置
        strategy_config = {
            "name": active_strategy_key or "xsec_rebalance",
            "selection": {
                "top_k": int(sel_cfg.get("top_k", 50)),
                "short_k": int(sel_cfg.get("short_k", 50)),
                "membership_buffer": float(sel_cfg.get("membership_buffer", 0.2)),
                "use_rank": str(sel_cfg.get("use_rank", "auto")).strip().lower(),
            },
            "entry": {
                "neutralize_items": neutralize_items,
                "ridge_lambda": ridge_lambda,
                "long_exposure": long_exposure,
                "short_exposure": short_exposure,
                "max_pos_per_name": max_pos_per_name,
                "weight_scheme": weight_scheme,
                "smooth_eta": smooth_eta,
                "target_vol": target_vol,
                "leverage_cap": leverage_cap,
                "hard_cap": hard_cap,
            },
            "exit": exit_cfg,
        }

        return {
            "strategy": strategy_config,
            "selection": strategy_config["selection"],
            "entry": strategy_config["entry"],
            "exit": strategy_config["exit"],
            "args": vars(args),
        }
