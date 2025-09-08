"""
回测运行器模块 - 提取自 run_backtest.py 的回测执行逻辑
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List, Set
import pandas as pd
import backtrader as bt
import json

# KPI计算模块
from backtest.kpi.calculator import KPICalculator

# 数据加载模块
from backtest.engine.data_loader import (
    ensure_inst_dt, weekly_schedule, asof_map_schedule_to_pred,
    load_qlib_ohlcv, build_exposures_map, compute_vol_adv_maps,
    _compute_short_timing_dates
)


class BacktestRunner:
    """回测执行器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.args = config["args"]
        
    def prepare_data(self) -> Dict[str, Any]:
        """准备回测数据 - 委托给策略特定的数据准备方法"""
        from strategies.base.strategy_registry import StrategyFactory
        
        # 从配置中获取策略配置
        strategy_config = self.config.get("strategy", {"name": "xsec_rebalance"})
        
        # 使用工厂获取策略类
        strategy_class = StrategyFactory.create_strategy(strategy_config)
        
        # 直接调用策略类的prepare_data方法（静态方法调用）
        # Backtrader策略类不能直接实例化，需要由Cerebro实例化
        data = strategy_class.prepare_data(self.config)
        
        return data

    def run_backtest(self, data: Dict[str, Any]) -> Any:
        """运行回测"""
        from strategies.base.strategy_registry import StrategyFactory
        
        # 从配置中获取策略配置
        strategy_config = self.config.get("strategy", {"name": "xsec_rebalance"})
        
        # 使用工厂获取策略类
        strategy_class = StrategyFactory.create_strategy(strategy_config)
        
        cerebro = bt.Cerebro(stdstats=True)
        cerebro.broker.setcash(float(self.args["cash"]))
        cerebro.broker.setcommission(commission=float(self.args["commission_bps"]) / 10000.0)
        cerebro.broker.set_slippage_perc(perc=float(self.args["slippage_bps"]) / 10000.0,
                                         slip_open=True, slip_limit=True, slip_match=True)
        if self.args["trade_at"] == "close":
            cerebro.broker.set_coc(True)

        for sym, df in data["price_map"].items():
            datafeed = bt.feeds.PandasData(dataname=df, name=sym)
            cerebro.adddata(datafeed)

        # 添加策略类（Backtrader会负责实例化）
        cerebro.addstrategy(strategy_class,
            preds_by_exec=data["preds_by_exec"],
            exec_dates=data["exec_dates"],
            exposures_by_date=data["expos_map"],
            vol_by_date=data["vol_map"],
            adv_by_date=data["adv_map"],
            neutralize_items=tuple(data["neutral_list"]),
            ridge_lambda=self.args["ridge_lambda"],
            trade_at=self.args["trade_at"],

            # 来自 config/CLI 合并
            top_k=self.config["selection"]["top_k"],
            short_k=self.config["selection"]["short_k"],
            membership_buffer=self.config["selection"]["membership_buffer"],
            selection_use_rank_mode=self.config["selection"]["use_rank"],

            long_exposure=self.config["entry"]["long_exposure"], 
            short_exposure=self.config["entry"]["short_exposure"],
            max_pos_per_name=self.config["entry"]["max_pos_per_name"],
            weight_scheme=self.config["entry"]["weight_scheme"],
            smooth_eta=self.config["entry"]["smooth_eta"],
            target_vol=self.config["entry"]["target_vol"],
            leverage_cap=self.config["entry"]["leverage_cap"],
            adv_limit_pct=self.config["entry"].get("adv_limit_pct", 0.0),
            # NEW
            short_timing_on=bool(self.args.get("short_timing_mom63", False)),
            short_timing_dates=data["short_allow_dates"],
            hard_cap=bool(self.config["entry"]["hard_cap"]),
            verbose=bool(self.args["verbose"]),
            # 入场策略配置
            entry_strategies_config = self.config["entry"],
            # 出场策略配置
            exit_strategies_config=self.config["exit"],
        )
        cerebro.addanalyzer(bt.analyzers.TimeReturn, timeframe=bt.TimeFrame.Days, _name='timeret', fund=False)
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='dd')

        return cerebro.run(maxcpus=1)

    def save_results(self, results: Any, data: Dict[str, Any]) -> None:
        """保存回测结果"""
        strat = results[0]
        out_dir = Path(self.args["out_dir"]).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        # 导出曲线
        eq = pd.DataFrame(strat.val_records).drop_duplicates(subset=["datetime"]).sort_values("datetime")
        eq["ret"] = eq["value"].pct_change().fillna(0.0)
        eq.to_csv(out_dir / "equity_curve.csv", index=False)

        # 导出逐日收益
        tr = results[0].analyzers.timeret.get_analysis()
        if tr:
            retdf = pd.DataFrame({"datetime": list(tr.keys()), "ret": list(tr.values())})
            retdf["datetime"] = pd.to_datetime(retdf["datetime"]).sort_values()
        else:
            retdf = eq[["datetime","ret"]].copy()
        retdf.to_csv(out_dir / "portfolio_returns.csv", index=False)

        # 逐日诊断导出
        diagdf = pd.DataFrame(strat.diag_daily).drop_duplicates(subset=["datetime"]).sort_values("datetime")
        diagdf.to_csv(out_dir / "per_day_ext.csv", index=False)

        # 订单与持仓
        pd.DataFrame(strat.order_records).to_csv(out_dir / "orders.csv", index=False)
        pd.DataFrame(strat.pos_records).to_csv(out_dir / "positions.csv", index=False)

        # KPI计算
        dd = results[0].analyzers.dd.get_analysis()
        commission_total = float(strat._commission_cum if hasattr(strat, "_commission_cum") else 0.0)
        
        # 使用KPICalculator计算所有指标
        # 需要将字典形式的args转换回Namespace对象
        from types import SimpleNamespace
        args_obj = SimpleNamespace(**self.args)
        
        summary, kpis = KPICalculator.calculate_all_kpis(
            args=args_obj,
            eq_df=eq,
            ret_df=retdf,
            diag_df=diagdf,
            dd_analysis=dd,
            strat=strat,
            price_map=data["price_map"],
            sel_cfg=self.config["selection"],
            commission_total=commission_total
        )
        
        # 保存KPI指标到文件
        KPICalculator.save_kpis_to_files(out_dir, summary, kpis)
        print("[summary]", json.dumps(summary, indent=2))
        print(f"[saved] -> {out_dir}")
