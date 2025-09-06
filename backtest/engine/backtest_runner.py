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
        """准备回测数据"""
        # 预测数据
        preds_all = ensure_inst_dt(pd.read_parquet(Path(self.args["preds"]).expanduser().resolve()))
        mask = (preds_all["datetime"] >= pd.Timestamp(self.args["start"])) & (preds_all["datetime"] <= pd.Timestamp(self.args["end"]))
        preds_all = preds_all.loc[mask].copy()
        if preds_all.empty:
            raise RuntimeError("预测为空。")
        pred_days = pd.DatetimeIndex(preds_all["datetime"].unique()).sort_values()

        # 初始 universe：窗口内出现过的所有票 + 锚
        anchor_sym = self.args["anchor_symbol"].upper() if self.args.get("anchor_symbol") else None
        universe_all = set(preds_all["instrument"].astype(str).str.upper().unique().tolist())
        if anchor_sym:
            universe_all.add(anchor_sym)

        # 行情（锚在第一位）
        price_map = load_qlib_ohlcv(sorted(list(universe_all)), start=self.args["start"], end=self.args["end"], qlib_dir=self.args["qlib_dir"])
        if anchor_sym and anchor_sym in price_map:
            anchor_days = pd.DatetimeIndex(price_map[anchor_sym].index)
            anchor_first = {anchor_sym: price_map.pop(anchor_sym)}
            price_map = {**anchor_first, **price_map}
        else:
            anchor_days = pd.DatetimeIndex(next(iter(price_map.values())).index)

        # 周频锚定 -> 源预测日 as-of 映射 -> exec_lag 推进到"执行日"
        sched_anchor = weekly_schedule(anchor_days)
        sched2pred = asof_map_schedule_to_pred(sched_anchor, pred_days)
        if not sched2pred:
            raise RuntimeError("调仓日与预测日无法 as-of 映射（窗口内没有预测）")
        exec_dates = []
        anchor_list = sorted(pd.DatetimeIndex(anchor_days).tolist())
        pos_map = {d: i for i, d in enumerate(anchor_list)}
        for sd in sched2pred.keys():
            i = pos_map.get(sd)
            if i is None:
                continue
            j = i + max(0, int(self.args["exec_lag"]))
            if j < len(anchor_list):
                exec_dates.append(anchor_list[j])
        exec_dates = sorted(pd.DatetimeIndex(exec_dates).unique().tolist())
        if not exec_dates:
            raise RuntimeError("exec_dates 为空（检查 exec_lag 与日期范围）")

        # 聚合预测：源日 -> 截面
        preds_all = preds_all.copy()
        keep_cols = ["instrument","score"] + (["rank"] if "rank" in preds_all.columns else [])
        preds_all["dt_norm"] = preds_all["datetime"].dt.normalize()
        preds_by_src = {d: g[keep_cols].copy() for d, g in preds_all.groupby("dt_norm")}

        # exec 日 -> 源预测日
        exec2pred_src = {}
        for sd, ps in sched2pred.items():
            i = pos_map.get(sd)
            if i is None: continue
            j = i + max(0, int(self.args["exec_lag"]))
            if j < len(anchor_list):
                exec2pred_src[anchor_list[j]] = ps

        # 最终 universe：参加过映射的票
        mapped_src_days = sorted(set(exec2pred_src.values()))
        final_universe = set(preds_all[preds_all["dt_norm"].isin(mapped_src_days)]["instrument"].astype(str).str.upper().unique().tolist())
        if anchor_sym:
            final_universe.add(anchor_sym)
        price_map = {sym: df for sym, df in price_map.items() if sym in final_universe}

        # exec->pred 截面（仅保留有行情的票）
        preds_by_exec = {}
        for ed, src in exec2pred_src.items():
            g = preds_by_src.get(src)
            if g is None: continue
            g2 = g[g["instrument"].isin(price_map.keys())].copy()
            preds_by_exec[pd.Timestamp(ed).normalize()] = g2

        # 中性化暴露、波动/ADV
        neutral_list = [s.strip().lower() for s in self.args["neutralize"].split(",") if s.strip()]
        expos_map = build_exposures_map(self.args["features_path"], universe=set(price_map.keys()),
                                        dates=list(preds_by_exec.keys()), use_items=neutral_list)
        vol_map, adv_map = compute_vol_adv_maps(self.args["features_path"], universe=set(price_map.keys()),
                                                dates=list(preds_by_exec.keys()), halflife=int(self.args["ewm_halflife"]))

        # 短腿择时日期集合
        short_allow_dates: Set[pd.Timestamp] = set()
        if self.args.get("short_timing_mom63") and anchor_sym and anchor_sym in price_map:
            short_allow_dates = _compute_short_timing_dates(
                price_map[anchor_sym],
                exec_dates=list(preds_by_exec.keys()),
                lookback=int(self.args["short_timing_lookback"]),
                thr=float(self.args["short_timing_threshold"])
            )

        return {
            "price_map": price_map,
            "preds_by_exec": preds_by_exec,
            "exec_dates": list(preds_by_exec.keys()),
            "expos_map": expos_map,
            "vol_map": vol_map,
            "adv_map": adv_map,
            "short_allow_dates": short_allow_dates,
            "neutral_list": neutral_list,
        }

    def run_backtest(self, data: Dict[str, Any]) -> Any:
        """运行回测"""
        from backtest.engine.strategy_registry import StrategyFactory
        
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
