"""
回测运行器模块 - 提取自 run_backtest.py 的回测执行逻辑
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List, Set
import pandas as pd
import backtrader as bt
import json
import numpy as np

# KPI计算模块
from backtest.kpi.calculator import KPICalculator


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
            liq_bucket_by_date=data.get("liq_bucket_by_date", {}),
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
            entry_strategies_config=self.config["entry"],
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
            retdf = pd.DataFrame({
                "datetime": pd.to_datetime(list(tr.keys())),
                "ret": list(tr.values()),
            }).sort_values("datetime")
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

        # === 新增：行业 & 流动性桶 P&L 归一化报告（[-1,1]，保留2位小数） ===
        try:
            pos_path = out_dir / "positions.csv"
            if pos_path.exists():
                df_pos = pd.read_csv(pos_path, parse_dates=["datetime"])  # columns: datetime,instrument,size,price,value
                if not df_pos.empty:
                    # 准备价格与收益
                    close_map = {}
                    for sym, dfp in data["price_map"].items():
                        s = dfp[["close"]].copy()
                        s.index = pd.to_datetime(s.index)
                        close_map[str(sym).upper()] = s.sort_index()

                    # 分组容器
                    sector_pnl = {}
                    liq_pnl = {}
                    sector_liq_pnl = {}

                    expos_map = data.get("expos_map", {}) or {}
                    liq_map = data.get("liq_bucket_by_date", {}) or {}

                    # 兜底行业映射（若当日 exposures 中无 ind_*）
                    sector_fallback = {}
                    try:
                        sec_csv = Path("data/instrument_sector.csv")
                        if sec_csv.exists():
                            sdf = pd.read_csv(sec_csv)
                            if {"instrument", "sector"} <= set(sdf.columns):
                                sdf["instrument"] = sdf["instrument"].astype(str).str.upper()
                                sector_fallback = dict(zip(sdf["instrument"], sdf["sector"].astype(str)))
                    except Exception:
                        pass

                    for dt, g in df_pos.groupby("datetime"):
                        dt = pd.Timestamp(dt).normalize()
                        # 权重（按绝对持仓归一化，适配可能的多空）
                        vals = g[["instrument", "value"]].copy()
                        vals["instrument"] = vals["instrument"].astype(str).str.upper()
                        denom = float(np.abs(vals["value"]).sum())
                        if denom <= 0:
                            continue
                        vals["w"] = vals["value"] / denom

                        # 次一交易日收益
                        contrib = []
                        for _, row in vals.iterrows():
                            sym = row["instrument"]
                            w = float(row["w"])
                            cs = close_map.get(sym)
                            if cs is None or cs.empty:
                                continue
                            idx = cs.index.searchsorted(dt, side="left")
                            if idx < 0 or idx + 1 >= len(cs.index):
                                continue
                            d0 = cs.index[idx]
                            if d0 != dt:
                                # 若当日不在价格索引中，尝试找下一个可用交易日
                                if idx >= len(cs.index) or idx + 1 >= len(cs.index):
                                    continue
                                d0 = cs.index[idx]
                            # 次日
                            d1 = cs.index.searchsorted(d0, side="right")
                            if isinstance(d1, np.ndarray):
                                d1 = int(d1)
                            if d1 >= len(cs.index):
                                continue
                            p0 = float(cs.iloc[idx]["close"])
                            p1 = float(cs.iloc[d1]["close"])
                            if not np.isfinite(p0) or p0 == 0:
                                continue
                            r = (p1 / p0) - 1.0
                            contrib.append((sym, w * r))

                        if not contrib:
                            continue

                        # 行业映射（当日）
                        sec_map = {}
                        exdf = expos_map.get(dt)
                        if exdf is not None and not exdf.empty:
                            exdf = exdf.copy()
                            exdf["instrument"] = exdf["instrument"].astype(str).str.upper()
                            ind_cols = [c for c in exdf.columns if str(c).startswith("ind_")]
                            if ind_cols:
                                sub = exdf[["instrument"] + ind_cols].copy()
                                for _, rr in sub.iterrows():
                                    inst = rr["instrument"]
                                    if len(ind_cols) == 1:
                                        sec = ind_cols[0].replace("ind_", "").strip()
                                    else:
                                        vals_np = rr[ind_cols].astype(float).to_numpy()
                                        j = int(np.nanargmax(vals_np)) if len(vals_np) else 0
                                        sec = ind_cols[j].replace("ind_", "").strip()
                                    sec_map[inst] = sec or "Unknown"
                        # 若仍无行业，使用兜底 CSV
                        if not sec_map and sector_fallback:
                            sec_map = sector_fallback.copy()

                        # 流动性桶映射（当日）
                        lb_map = {}
                        ldf = liq_map.get(dt)
                        if ldf is not None and not ldf.empty and "liq_bucket" in ldf.columns:
                            ldf = ldf.copy()
                            ldf["instrument"] = ldf["instrument"].astype(str).str.upper()
                            lb_map = dict(zip(ldf["instrument"], ldf["liq_bucket"]))

                        # 聚合贡献
                        for sym, c in contrib:
                            sec = sec_map.get(sym, sector_fallback.get(sym, "Unknown"))
                            sector_pnl[sec] = sector_pnl.get(sec, 0.0) + float(c)
                            lb = lb_map.get(sym, None)
                            # 无法分配桶时，标记为 liq_unknown（与数据准备阶段一致）
                            lb_key = f"liq_{int(lb)}" if lb is not None and pd.notna(lb) else "liq_unknown"
                            liq_pnl[lb_key] = liq_pnl.get(lb_key, 0.0) + float(c)
                            # joint
                            sector_liq_pnl.setdefault(sec, {})
                            sector_liq_pnl[sec][lb_key] = sector_liq_pnl[sec].get(lb_key, 0.0) + float(c)

                    def _normalize_round(d: dict) -> dict:
                        if not d:
                            return {}
                        m = max((abs(v) for v in d.values()), default=0.0)
                        if m <= 0:
                            return {k: 0.00 for k in d}
                        out = {k: float(np.clip(v / m, -1.0, 1.0)) for k, v in d.items()}
                        # 两位小数
                        return {k: float(f"{v:.2f}") for k, v in out.items()}

                    # 若仅需行业×流动性矩阵，可跳过单独的行业与流动性聚合输出
                    # joint normalization across all cells
                    all_vals = [v for dct in sector_liq_pnl.values() for v in dct.values()]
                    max_abs = max([abs(v) for v in all_vals], default=0.0)
                    if max_abs <= 0:
                        joint_norm = {s: {b: 0.00 for b in dct} for s, dct in sector_liq_pnl.items()}
                    else:
                        joint_norm = {
                            s: {b: float(f"{float(np.clip(v / max_abs, -1.0, 1.0)):.2f}") for b, v in dct.items()}
                            for s, dct in sector_liq_pnl.items()
                        }

                    # 不再输出 JSON 版本，仅保留 CSV 宽表

                    # 仅输出行业×流动性：长表与宽表 CSV
                    try:
                        # 1) 归一化矩阵（现有输出）：仅输出宽表 CSV（sectors 作为行，liq_buckets 作为列）
                        if joint_norm:
                            rows = []
                            for s, dct in joint_norm.items():
                                for b, v in dct.items():
                                    rows.append((s, b, v))
                            df_long = pd.DataFrame(rows, columns=["sector", "liq_bucket", "value"]) 
                            if not df_long.empty:
                                df_wide = df_long.pivot_table(index="sector", columns="liq_bucket", values="value", aggfunc="first")
                                cols = list(df_wide.columns)
                                num_cols = [c for c in cols if isinstance(c, str) and c.startswith("liq_") and c.split("_")[-1].isdigit()]
                                num_cols_sorted = sorted(num_cols, key=lambda x: int(x.split("_")[-1]))
                                other_cols = [c for c in cols if c not in num_cols]
                                col_order = num_cols_sorted + [c for c in other_cols if c != "liq_unknown"] + (["liq_unknown"] if "liq_unknown" in other_cols else [])
                                df_wide = df_wide.reindex(columns=[c for c in col_order if c in df_wide.columns])
                                df_wide.to_csv(out_dir / "pnl_by_sector_liq_wide.csv", float_format='%.2f')

                        # 2) 原始未归一化贡献（新增）：输出长表与宽表，便于多时间段/多 run 汇总
                        if sector_liq_pnl:
                            raw_rows = []
                            for s, dct in sector_liq_pnl.items():
                                for b, v in dct.items():
                                    raw_rows.append((s, b, float(v)))
                            df_raw_long = pd.DataFrame(raw_rows, columns=["sector", "liq_bucket", "value"])
                            if not df_raw_long.empty:
                                df_raw_long.to_csv(out_dir / "pnl_by_sector_liq_raw_long.csv", index=False)
                                df_raw_wide = df_raw_long.pivot_table(index="sector", columns="liq_bucket", values="value", aggfunc="sum").fillna(0.0)
                                # 列排序与归一化版本一致
                                cols = list(df_raw_wide.columns)
                                num_cols = [c for c in cols if isinstance(c, str) and c.startswith("liq_") and c.split("_")[-1].isdigit()]
                                num_cols_sorted = sorted(num_cols, key=lambda x: int(x.split("_")[-1]))
                                other_cols = [c for c in cols if c not in num_cols]
                                col_order = num_cols_sorted + [c for c in other_cols if c != "liq_unknown"] + (["liq_unknown"] if "liq_unknown" in other_cols else [])
                                df_raw_wide = df_raw_wide.reindex(columns=[c for c in col_order if c in df_raw_wide.columns])
                                df_raw_wide.to_csv(out_dir / "pnl_by_sector_liq_raw_wide.csv", float_format='%.2f')
                    except Exception as e:
                        try:
                            print(f"[warn] failed to output sector×liq CSV: {e}")
                        except Exception:
                            pass
        except Exception as e:
            # 不影响主流程
            try:
                print(f"[warn] failed to build group P&L: {e}")
            except Exception:
                pass
