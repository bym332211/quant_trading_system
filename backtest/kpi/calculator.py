"""
KPI计算器 - 用于计算回测性能指标
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
import json
from pathlib import Path

class KPICalculator:
    """KPI计算器类，用于计算各种回测性能指标"""
    
    @staticmethod
    def calculate_basic_kpis(eq_df: pd.DataFrame, ret_df: pd.DataFrame, 
                           dd_analysis: Dict[str, Any]) -> Dict[str, float]:
        """
        计算基本KPI指标
        
        Args:
            eq_df: 权益曲线DataFrame，包含datetime和value列
            ret_df: 收益率DataFrame，包含datetime和ret列
            dd_analysis: 回撤分析结果
            
        Returns:
            基本KPI指标字典
        """
        ret = ret_df["ret"].to_numpy()
        ann = np.sqrt(252.0)
        
        # 计算夏普比率
        sharpe = float(np.nanmean(ret) / (np.nanstd(ret, ddof=1) + 1e-12) * ann) if len(ret) > 2 else float('nan')
        
        # 计算年化收益率(CAGR)
        if len(eq_df) > 1:
            cagr = float((eq_df["value"].iloc[-1] / eq_df["value"].iloc[0]) ** (252.0 / max(1, len(eq_df))) - 1.0)
        else:
            cagr = float('nan')
        
        # 计算最大回撤
        mdd = float(dd_analysis.get('max', {}).get('drawdown', np.nan))
        
        return {
            "sharpe": sharpe,
            "cagr": cagr,
            "mdd_pct": mdd
        }
    
    @staticmethod
    def calculate_diagnostic_kpis(diag_df: pd.DataFrame) -> Dict[str, float]:
        """
        计算诊断相关的KPI指标
        
        Args:
            diag_df: 诊断DataFrame
            
        Returns:
            诊断KPI指标字典
        """
        # 换手率指标
        turn_mean = float(diag_df["turnover_post"].mean()) if "turnover_post" in diag_df else float('nan')
        turn_p90 = float(diag_df["turnover_post"].quantile(0.9)) if "turnover_post" in diag_df else float('nan')
        
        # ADV限制相关指标
        adv_hit_days = float((diag_df["adv_clip_names"] > 0).mean()) if "adv_clip_names" in diag_df else 0.0
        adv_clip_avg = float(diag_df["adv_clip_ratio"].replace([np.inf, -np.inf], np.nan).fillna(0.0).mean()) if "adv_clip_ratio" in diag_df else 0.0
        
        # 多空暴露指标
        gross_long_avg = float(diag_df["gross_long"].mean()) if "gross_long" in diag_df else float('nan')
        gross_short_avg = float(diag_df["gross_short"].mean()) if "gross_short" in diag_df else float('nan')
        
        return {
            "turnover_mean": turn_mean,
            "turnover_p90": turn_p90,
            "adv_clip_days_frac": adv_hit_days,
            "adv_clip_ratio_avg": adv_clip_avg,
            "gross_long_avg": gross_long_avg,
            "gross_short_avg": gross_short_avg
        }
    
    @staticmethod
    def calculate_leg_sharpe_ratios(diag_df: pd.DataFrame) -> Dict[str, float]:
        """
        计算长短腿的夏普比率
        
        Args:
            diag_df: 诊断DataFrame，需要包含ret_long和ret_short列
            
        Returns:
            长短腿夏普比率字典
        """
        ann = np.sqrt(252.0)
        
        if {"ret_long", "ret_short"} <= set(diag_df.columns):
            rl = diag_df["ret_long"].to_numpy()
            rs = diag_df["ret_short"].to_numpy()
            
            sharpe_long = float(np.nanmean(rl) / (np.nanstd(rl, ddof=1) + 1e-12) * ann) if len(rl) > 2 else float('nan')
            sharpe_short = float(np.nanmean(rs) / (np.nanstd(rs, ddof=1) + 1e-12) * ann) if len(rs) > 2 else float('nan')
        else:
            sharpe_long = sharpe_short = float('nan')
        
        return {
            "sharpe_long": sharpe_long,
            "sharpe_short": sharpe_short
        }
    
    @staticmethod
    def create_summary(args: Any, eq_df: pd.DataFrame, strat: Any, 
                      price_map: Dict[str, pd.DataFrame], sel_cfg: Dict[str, Any]) -> Dict[str, Any]:
        """
        创建策略总结信息
        
        Args:
            args: 命令行参数
            eq_df: 权益曲线DataFrame
            strat: 策略实例
            price_map: 价格数据字典
            sel_cfg: 选择配置
            
        Returns:
            策略总结字典
        """
        return {
            "start": args.start, "end": args.end,
            "cash_init": float(eq_df["value"].iloc[0]) if len(eq_df) else float(args.cash),
            "cash_end": float(eq_df["value"].iloc[-1]) if len(eq_df) else float(args.cash),
            "top_k": sel_cfg.get("top_k", 50), 
            "short_k": sel_cfg.get("short_k", 50),
            "membership_buffer": sel_cfg.get("membership_buffer", 0.2),
            "selection_use_rank_mode": sel_cfg.get("use_rank", "auto"),
            "long_exposure": args.long_exposure, 
            "short_exposure": args.short_exposure,
            "commission_bps": args.commission_bps, 
            "slippage_bps": args.slippage_bps,
            "trade_at": args.trade_at, 
            "neutralize": [s for s in [s.strip() for s in args.neutralize.split(",")] if s],
            "smooth_eta": args.smooth_eta,
            "days": int(len(eq_df)), 
            "rebalance_days": len(set(list(strat.p.preds_by_exec.keys()))) if hasattr(strat, 'p') and hasattr(strat.p, 'preds_by_exec') else 0,
            "universe_size": len(price_map),
            "avg_candidates_per_reb": float(np.mean([len(g) for g in strat.p.preds_by_exec.values()])) if hasattr(strat, 'p') and hasattr(strat.p, 'preds_by_exec') else 0.0,
            "exec_lag": int(args.exec_lag),
            "target_vol": float(args.target_vol),
            "weight_scheme": args.weight_scheme,
            "adv_limit_pct": float(args.adv_limit_pct),
            "hard_cap": bool(args.hard_cap),
            "short_timing_mom63": bool(args.short_timing_mom63),
            "short_timing_threshold": float(args.short_timing_threshold),
            "short_timing_lookback": int(args.short_timing_lookback),
        }
    
    @staticmethod
    def calculate_all_kpis(args: Any, eq_df: pd.DataFrame, ret_df: pd.DataFrame, 
                          diag_df: pd.DataFrame, dd_analysis: Dict[str, Any],
                          strat: Any, price_map: Dict[str, pd.DataFrame], 
                          sel_cfg: Dict[str, Any], commission_total: float) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        计算所有KPI指标
        
        Args:
            args: 命令行参数
            eq_df: 权益曲线DataFrame
            ret_df: 收益率DataFrame
            diag_df: 诊断DataFrame
            dd_analysis: 回撤分析结果
            strat: 策略实例
            price_map: 价格数据字典
            sel_cfg: 选择配置
            commission_total: 总佣金
            
        Returns:
            (summary_dict, kpis_dict) 元组
        """
        # 计算基本KPI
        basic_kpis = KPICalculator.calculate_basic_kpis(eq_df, ret_df, dd_analysis)
        
        # 计算诊断KPI
        diagnostic_kpis = KPICalculator.calculate_diagnostic_kpis(diag_df)
        
        # 计算长短腿夏普比率
        leg_sharpe = KPICalculator.calculate_leg_sharpe_ratios(diag_df)
        
        # 创建策略总结
        summary = KPICalculator.create_summary(args, eq_df, strat, price_map, sel_cfg)
        
        # 合并所有KPI指标
        kpis = {
            **basic_kpis,
            **diagnostic_kpis,
            **leg_sharpe,
            "commission_total": float(commission_total)
        }
        
        # 将基本KPI也加入到summary中
        summary.update(basic_kpis)
        
        return summary, kpis
    
    @staticmethod
    def save_kpis_to_files(out_dir: str, summary: Dict[str, Any], kpis: Dict[str, Any]):
        """
        保存KPI指标到文件
        
        Args:
            out_dir: 输出目录
            summary: 策略总结字典
            kpis: KPI指标字典
        """
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        
        # 保存summary
        with open(out_path / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        # 保存kpis
        with open(out_path / "kpis.json", "w") as f:
            json.dump(kpis, f, indent=2)
