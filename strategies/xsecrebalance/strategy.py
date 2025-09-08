"""
策略模块 - 包含 XSecRebalance 策略类
"""
from __future__ import annotations
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np
import backtrader as bt

# external selection module
from strategies import ExitStrategyCoordinator
from strategies.entry_strategies import EntryStrategyCoordinator

# 数据加载模块中的工具函数
from backtest.engine.data_loader import apply_adv_limit

# 策略基类
from strategies.base.base_strategy import BaseStrategy


class XSecRebalance(bt.Strategy):
    params = dict(
        preds_by_exec=None,
        exec_dates=None,
        exposures_by_date=None,
        vol_by_date=None,
        adv_by_date=None,
        neutralize_items=(),
        ridge_lambda=1e-6,

        trade_at="open",
        # 从 config/CLI 来
        top_k=50, short_k=50,
        membership_buffer=0.2,
        selection_use_rank_mode="auto",  # "auto" | "rank" | "score"

        long_exposure=1.0, short_exposure=-1.0,
        max_pos_per_name=0.05,
        weight_scheme="equal",

        smooth_eta=0.6,

        target_vol=0.0,
        leverage_cap=2.0,
        adv_limit_pct=0.0,

        # 短腿择时
        short_timing_on=False,
        short_timing_dates=None,   # set of dates 允许做空

        # 硬上限
        hard_cap=False,

        # 出场策略配置
        exit_strategies_config=None,
        
        # 入场策略配置
        entry_strategies_config=None,

        verbose=False,
    )

    def __init__(self):
        if self.p.preds_by_exec is None or self.p.exec_dates is None:
            raise ValueError("preds_by_exec / exec_dates 未传入")
        self._preds = self.p.preds_by_exec
        self._exec = set(pd.to_datetime(self.p.exec_dates).tolist())
        self._expos = self.p.exposures_by_date or {}
        self._vol   = self.p.vol_by_date or {}
        self._adv   = self.p.adv_by_date or {}
        self._neutral = tuple(self.p.neutralize_items) if self.p.neutralize_items else tuple()
        self._short_allow = set(self.p.short_timing_dates) if self.p.short_timing_dates else set()

        self.data2sym = {d: d._name for d in self.datas}
        self.sym2data = {d._name: d for d in self.datas}

        # 记录
        self.val_records, self.order_records, self.pos_records = [], [], []
        self.prev_weights: dict[str, float] = {}
        self.reb_counter = 0

        # 诊断记录
        self.diag_daily = []
        self._commission_cum = 0.0

        if str(self.p.trade_at).lower() == "close":
            self.broker.set_coc(True)

        # === 入场策略实例 ===
        # 存储入场策略配置（用于后续初始化）
        self._entry_strategies_config = getattr(self.p, "entry_strategies_config", {})
        
        # 初始化入场策略协调器
        icdf_equal_config = self._entry_strategies_config.get("icdf_equal", {})
        enabled_strategies = self._entry_strategies_config.get("enabled_strategies", ["icdf_equal"])
        strategy_weights = self._entry_strategies_config.get("strategy_weights", {"icdf_equal": 1.0})
        
        self.entry = EntryStrategyCoordinator(
            icdf_equal_config=icdf_equal_config,
            enabled_strategies=enabled_strategies,
            strategy_weights=strategy_weights
        )
        
        # 存储出场策略配置（用于后续初始化）
        self._exit_strategies_config = getattr(self.p, "exit_strategies_config", {})

    # ----- Backtrader callbacks -----
    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        rec = {
            "datetime": pd.Timestamp(bt.num2date(order.executed.dt or self.datas[0].datetime[0]).date()),
            "instrument": order.data._name if order.data else None,
            "status": order.getstatusname(),
            "size": float(order.size),
            "price": float(order.executed.price or np.nan),
            "value": float(order.executed.value or np.nan),
            "commission": float(order.executed.comm or np.nan),
        }
        self._commission_cum += float(order.executed.comm or 0.0)
        self.order_records.append(rec)
        if self.p.verbose:
            print("[order]", rec)

    def notify_trade(self, trade):
        if trade.isclosed:
            rec = {
                "datetime": pd.Timestamp(bt.num2date(self.datas[0].datetime[0]).date()),
                "instrument": trade.data._name,
                "pnl": float(trade.pnl),
                "pnlcomm": float(trade.pnlcomm),
                "price": float(trade.price),
                "size": float(trade.size),
                "status": "TRADE_CLOSED",
            }
            self.order_records.append(rec)
            if self.p.verbose:
                print("[trade]", rec)

    # ----- Core -----
    def next(self):
        dtoday = pd.Timestamp(bt.num2date(self.datas[0].datetime[0]).date()).normalize()

        # 组合权益记录
        port_val = float(self.broker.getvalue())
        port_cash = float(self.broker.getcash())
        self.val_records.append({"datetime": dtoday, "value": port_val, "cash": port_cash})

        # ---------- 日度收益拆腿（昨日权重 * 今日 close/prev_close） ----------
        w_prev = pd.Series(self.prev_weights, dtype=float)
        ret_map = {}
        for d in self.datas:
            if len(d) < 2:
                continue
            prev_c = float(d.close[-1])
            curr_c = float(d.close[0])
            if prev_c > 0:
                ret_map[d._name] = (curr_c / prev_c - 1.0)
        if len(w_prev) > 0 and ret_map:
            rets = pd.Series(ret_map).reindex(w_prev.index).fillna(0.0)
            total_ret = float((w_prev * rets).sum())
            long_ret  = float((w_prev.clip(lower=0.0) * rets).sum())
            short_ret = float((w_prev.clip(upper=0.0) * rets).sum())
        else:
            total_ret, long_ret, short_ret = 0.0, 0.0, 0.0

        # 记录持仓快照
        for d in self.datas:
            pos = self.getposition(d)
            if pos.size != 0:
                self.pos_records.append({"datetime": dtoday, "instrument": d._name,
                                         "size": float(pos.size), "price": float(pos.price),
                                         "value": float(pos.size * d.close[0])})
        
        # ---------- 出场策略检查（每日执行） ----------
        # 初始化出场策略协调器（如果尚未初始化）
        if not hasattr(self, 'exit_coordinator'):
            # 从配置中获取出场策略参数
            tech_stop_loss_config = self._exit_strategies_config.get("tech_stop_loss", {})
            volatility_exit_config = self._exit_strategies_config.get("volatility_exit", {})
            enabled_strategies = self._exit_strategies_config.get("enabled_strategies", ["tech_stop_loss", "volatility_exit"])
            
            self.exit_coordinator = ExitStrategyCoordinator(
                tech_stop_loss_config=tech_stop_loss_config,
                volatility_exit_config=volatility_exit_config,
                enabled_strategies=enabled_strategies
            )
        
        # 检查每个持仓是否需要出场
        exit_symbols = []
        for d in self.datas:
            pos = self.getposition(d)
            if pos.size != 0:
                symbol = d._name
                current_price = float(d.close[0])
                
                # 获取历史价格数据（最近60天）
                lookback = 60
                if len(d) >= lookback:
                    historical_data = pd.DataFrame({
                        'open': [float(d.open[i]) for i in range(-lookback, 0)],
                        'high': [float(d.high[i]) for i in range(-lookback, 0)],
                        'low': [float(d.low[i]) for i in range(-lookback, 0)],
                        'close': [float(d.close[i]) for i in range(-lookback, 0)],
                        'volume': [float(d.volume[i]) for i in range(-lookback, 0)]
                    }, index=pd.date_range(end=dtoday, periods=lookback, freq='D'))
                    
                    # 记录入场信息（如果尚未记录）
                    if symbol not in self.exit_coordinator.entry_prices:
                        # 使用平均入场价格
                        entry_price = float(pos.price) if pos.price != 0 else current_price
                        self.exit_coordinator.record_entry(
                            symbol=symbol,
                            entry_price=entry_price,
                            position_size=float(pos.size),
                            entry_date=dtoday
                        )
                    
                    # 检查是否需要出场
                    should_exit, triggered_strategies = self.exit_coordinator.should_exit(
                        symbol=symbol,
                        current_price=current_price,
                        historical_data=historical_data,
                        current_date=dtoday
                    )
                    
                    if should_exit:
                        exit_symbols.append((symbol, triggered_strategies))
        
        # 执行出场
        for symbol, strategies in exit_symbols:
            d = self.sym2data.get(symbol)
            if d:
                if self.p.verbose:
                    print(f"[exit] {symbol} triggered by strategies: {strategies}")
                self.order_target_percent(data=d, target=0.0)
                self.exit_coordinator.record_exit(symbol)

        # ---------- 非执行日：记录后返回 ----------
        if dtoday not in self._exec:
            self.diag_daily.append({
                "datetime": dtoday,
                "ret": total_ret,
                "ret_long": long_ret,
                "ret_short": short_ret,
                "turnover_pre": 0.0,
                "turnover_post": 0.0,
                "adv_clip_ratio": 0.0,
                "adv_clip_names": 0,
                "gross_long": float(w_prev.clip(lower=0.0).sum()),
                "gross_short": float(-w_prev.clip(upper=0.0).sum()),
                "commission_cum": float(self._commission_cum),
            })
            return

        # ---------- 选股+入场权重：委托给 EntryStrategy ----------
        g = self._preds.get(dtoday)
        if g is None or g.empty:
            if self.p.verbose:
                print(f"[warn] no predictions for exec day {dtoday.date()}")
            self.diag_daily.append({
                "datetime": dtoday, "ret": total_ret, "ret_long": long_ret, "ret_short": short_ret,
                "turnover_pre": 0.0, "turnover_post": 0.0,
                "adv_clip_ratio": 0.0, "adv_clip_names": 0,
                "gross_long": float(w_prev.clip(lower=0.0).sum()),
                "gross_short": float(-w_prev.clip(upper=0.0).sum()),
                "commission_cum": float(self._commission_cum),
            })
            return

        # 短腿择时：不允许做空则在入场策略中清空 short 候选
        allow_shorts_today = (not self.p.short_timing_on) or (dtoday in self._short_allow)

        # 使用 EntryStrategy 生成"ADV 限速之前"的目标权重
        tgt_pre_adv = self.entry.generate_entry_weights(
            g=g,
            prev_weights=self.prev_weights or {},
            expos_df=self._expos.get(dtoday),
            vol_df=self._vol.get(dtoday),
            allow_shorts=allow_shorts_today,
            reb_counter=int(self.reb_counter),
        )

        # ---------- %ADV 限速 ----------
        adv_df = self._adv.get(dtoday)
        tgt, diag_adv = apply_adv_limit(
            self.prev_weights, tgt_pre_adv, adv_df,
            self.broker.getvalue(), adv_limit_pct=float(self.p.adv_limit_pct)
        )

        # ---------- 诊断：turnover / adv clip ----------
        # 注意：此处使用"更新前"的 prev_weights 计算换手
        w_prev_ser = pd.Series(self.prev_weights, dtype=float)
        w_pre_ser  = pd.Series(tgt_pre_adv, dtype=float)
        w_post_ser = pd.Series(tgt, dtype=float)
        all_idx = sorted(set(w_prev_ser.index) | set(w_pre_ser.index) | set(w_post_ser.index))
        w_prev_ser = w_prev_ser.reindex(all_idx).fillna(0.0)
        w_pre_ser  = w_pre_ser.reindex(all_idx).fillna(0.0)
        w_post_ser = w_post_ser.reindex(all_idx).fillna(0.0)
        delta_pre  = (w_pre_ser - w_prev_ser).abs().sum()
        delta_post = (w_post_ser - w_prev_ser).abs().sum()
        turnover_pre  = 0.5 * float(delta_pre)
        turnover_post = 0.5 * float(delta_post)
        adv_clip_ratio = float(diag_adv["clip_sum"] / diag_adv["delta_pre_sum"]) if diag_adv["delta_pre_sum"] > 0 else 0.0

        # ---------- 下单 ----------
        for sym, tw in tgt.items():
            d = self.sym2data.get(sym)
            if d is None:
                if self.p.verbose:
                    print(f"[skip] {sym} has no datafeed")
                continue
            self.order_target_percent(data=d, target=float(tw))
        for d in self.datas:
            sym = d._name
            if sym in tgt: 
                continue
            pos = self.getposition(d)
            if pos.size != 0:
                self.order_target_percent(data=d, target=0.0)

        # 保存诊断记录
        self.diag_daily.append({
            "datetime": dtoday,
            "ret": total_ret,
            "ret_long": long_ret,
            "ret_short": short_ret,
            "turnover_pre": turnover_pre,
            "turnover_post": turnover_post,
            "adv_clip_ratio": adv_clip_ratio,
            "adv_clip_names": int(diag_adv.get("hit_names", 0)),
            "gross_long": float(w_prev_ser.clip(lower=0.0).sum()),
            "gross_short": float(-w_prev_ser.clip(upper=0.0).sum()),
            "commission_cum": float(self._commission_cum),
        })

        # 同步 prev_weights（以 ADV 限速后的最终权重为准）
        self.prev_weights = tgt.copy()
        self.entry.prev_weights = tgt.copy()
        self.reb_counter += 1

    # ----- BaseStrategy 接口实现 -----
    
    def prepare(self, config: Dict[str, Any], data: Dict[str, Any]) -> None:
        """准备阶段：根据配置和数据初始化策略参数"""
        # 存储配置和数据引用
        self.config = config
        self.data = data
        
        # 设置策略参数
        self.p.preds_by_exec = data["preds_by_exec"]
        self.p.exec_dates = data["exec_dates"]
        self.p.exposures_by_date = data["expos_map"]
        self.p.vol_by_date = data["vol_map"]
        self.p.adv_by_date = data["adv_map"]
        self.p.neutralize_items = tuple(data["neutral_list"])
        
        # 设置其他参数
        args = config.get("args", {})
        self.p.ridge_lambda = args.get("ridge_lambda", 1e-6)
        self.p.trade_at = args.get("trade_at", "open")
        
        # 设置选择配置
        selection_config = config.get("selection", {})
        self.p.top_k = selection_config.get("top_k", 50)
        self.p.short_k = selection_config.get("short_k", 50)
        self.p.membership_buffer = selection_config.get("membership_buffer", 0.2)
        self.p.selection_use_rank_mode = selection_config.get("use_rank", "auto")
        
        # 设置入场配置
        entry_config = config.get("entry", {})
        self.p.long_exposure = entry_config.get("long_exposure", 1.0)
        self.p.short_exposure = entry_config.get("short_exposure", -1.0)
        self.p.max_pos_per_name = entry_config.get("max_pos_per_name", 0.05)
        self.p.weight_scheme = entry_config.get("weight_scheme", "equal")
        self.p.smooth_eta = entry_config.get("smooth_eta", 0.6)
        self.p.target_vol = entry_config.get("target_vol", 0.0)
        self.p.leverage_cap = entry_config.get("leverage_cap", 2.0)
        self.p.adv_limit_pct = entry_config.get("adv_limit_pct", 0.0)
        self.p.hard_cap = entry_config.get("hard_cap", False)
        
        # 设置短腿择时
        self.p.short_timing_on = bool(args.get("short_timing_mom63", False))
        self.p.short_timing_dates = data.get("short_allow_dates", set())
        
        # 设置出场策略配置
        self.p.exit_strategies_config = config.get("exit", {})
        self.p.verbose = bool(args.get("verbose", False))
        
        # 重新初始化内部数据结构
        self._preds = self.p.preds_by_exec
        self._exec = set(pd.to_datetime(self.p.exec_dates).tolist())
        self._expos = self.p.exposures_by_date or {}
        self._vol   = self.p.vol_by_date or {}
        self._adv   = self.p.adv_by_date or {}
        self._neutral = tuple(self.p.neutralize_items) if self.p.neutralize_items else tuple()
        self._short_allow = set(self.p.short_timing_dates) if self.p.short_timing_dates else set()
        
        # 存储入场策略配置
        self._entry_strategies_config = config.get("entry", {})
        
        # 重新初始化入场策略协调器
        icdf_equal_config = self._entry_strategies_config.get("icdf_equal", {})
        enabled_strategies = self._entry_strategies_config.get("enabled_strategies", ["icdf_equal"])
        strategy_weights = self._entry_strategies_config.get("strategy_weights", {"icdf_equal": 1.0})
        
        self.entry = EntryStrategyCoordinator(
            icdf_equal_config=icdf_equal_config,
            enabled_strategies=enabled_strategies,
            strategy_weights=strategy_weights
        )
        
        # 存储出场策略配置
        self._exit_strategies_config = getattr(self.p, "exit_strategies_config", {})
    
    def entry_strategy(self, dtoday: pd.Timestamp, preds_df: pd.DataFrame, 
                      prev_weights: Dict[str, float]) -> Dict[str, float]:
        """入场策略：生成初步目标权重"""
        g = self._preds.get(dtoday)
        if g is None or g.empty:
            return {}
        
        allow_shorts_today = (not self.p.short_timing_on) or (dtoday in self._short_allow)
        
        return self.entry.generate_entry_weights(
            g=g,
            prev_weights=prev_weights or {},
            expos_df=self._expos.get(dtoday),
            vol_df=self._vol.get(dtoday),
            allow_shorts=allow_shorts_today,
            reb_counter=int(self.reb_counter),
        )
    
    def exit_strategy(self, dtoday: pd.Timestamp) -> List[Tuple[str, List[str]]]:
        """出场策略：检查是否需要出场"""
        if not hasattr(self, 'exit_coordinator'):
            # 初始化出场策略协调器
            tech_stop_loss_config = self._exit_strategies_config.get("tech_stop_loss", {})
            volatility_exit_config = self._exit_strategies_config.get("volatility_exit", {})
            enabled_strategies = self._exit_strategies_config.get("enabled_strategies", ["tech_stop_loss", "volatility_exit"])
            
            self.exit_coordinator = ExitStrategyCoordinator(
                tech_stop_loss_config=tech_stop_loss_config,
                volatility_exit_config=volatility_exit_config,
                enabled_strategies=enabled_strategies
            )
        
        exit_symbols = []
        for d in self.datas:
            pos = self.getposition(d)
            if pos.size != 0:
                symbol = d._name
                current_price = float(d.close[0])
                
                # 获取历史价格数据
                lookback = 60
                if len(d) >= lookback:
                    historical_data = pd.DataFrame({
                        'open': [float(d.open[i]) for i in range(-lookback, 0)],
                        'high': [float(d.high[i]) for i in range(-lookback, 0)],
                        'low': [float(d.low[i]) for i in range(-lookback, 0)],
                        'close': [float(d.close[i]) for i in range(-lookback, 0)],
                        'volume': [float(d.volume[i]) for i in range(-lookback, 0)]
                    }, index=pd.date_range(end=dtoday, periods=lookback, freq='D'))
                    
                    # 记录入场信息
                    if symbol not in self.exit_coordinator.entry_prices:
                        entry_price = float(pos.price) if pos.price != 0 else current_price
                        self.exit_coordinator.record_entry(
                            symbol=symbol,
                            entry_price=entry_price,
                            position_size=float(pos.size),
                            entry_date=dtoday
                        )
                    
                    # 检查是否需要出场
                    should_exit, triggered_strategies = self.exit_coordinator.should_exit(
                        symbol=symbol,
                        current_price=current_price,
                        historical_data=historical_data,
                        current_date=dtoday
                    )
                    
                    if should_exit:
                        exit_symbols.append((symbol, triggered_strategies))
        
        return exit_symbols
    
    def risk_management(self, tgt_weights: Dict[str, float], 
                       dtoday: pd.Timestamp) -> Dict[str, float]:
        """风险管理：应用风控规则"""
        # 这里可以添加额外的风控逻辑
        # 当前的风险管理已经在 next() 方法中通过 EntryStrategy 处理
        return tgt_weights
    
    def position_sizing(self, tgt_weights: Dict[str, float],
                       dtoday: pd.Timestamp, port_value: float) -> Dict[str, float]:
        """仓位管理：最终仓位确定"""
        adv_df = self._adv.get(dtoday)
        tgt, _ = apply_adv_limit(
            self.prev_weights, tgt_weights, adv_df,
            port_value, adv_limit_pct=float(self.p.adv_limit_pct)
        )
        return tgt
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """获取诊断信息"""
        return {
            "commission_cum": self._commission_cum,
            "reb_counter": self.reb_counter,
            "exit_stats": getattr(self.exit_coordinator, 'get_stats', lambda: {})()
        }
    
    @classmethod
    def prepare_data(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        """XSecRebalance 策略的数据准备"""
        from backtest.engine.data_utils import (
            prepare_predictions, prepare_price_data, prepare_execution_dates,
            prepare_exposures, prepare_vol_adv, prepare_short_timing,
            get_universe_from_predictions, filter_universe_by_mapped_predictions
        )
        
        args = config["args"]
        anchor_sym = args["anchor_symbol"].upper() if args.get("anchor_symbol") else None
        
        # 1. 准备预测数据
        preds_by_src, preds_all = prepare_predictions(config)
        
        # 2. 获取初始universe
        initial_universe = get_universe_from_predictions(preds_all, anchor_sym)
        
        # 3. 准备价格数据
        price_map = prepare_price_data(config, initial_universe)
        
        # 4. 准备执行日期映射
        exec2pred_src, exec_dates_set = prepare_execution_dates(config, price_map, preds_by_src)
        
        # 5. 过滤最终universe
        mapped_src_days = set(exec2pred_src.values())
        final_universe = filter_universe_by_mapped_predictions(preds_all, mapped_src_days, anchor_sym)
        
        # 6. 过滤价格数据
        price_map = {sym: df for sym, df in price_map.items() if sym in final_universe}
        
        # 7. 准备按执行日分组的预测数据
        preds_by_exec = {}
        for ed, src in exec2pred_src.items():
            g = preds_by_src.get(src)
            if g is None: continue
            g2 = g[g["instrument"].isin(price_map.keys())].copy()
            preds_by_exec[pd.Timestamp(ed).normalize()] = g2
        
        # 8. 准备暴露数据
        expos_map = prepare_exposures(config, final_universe, set(preds_by_exec.keys()))
        
        # 9. 准备波动率和ADV数据
        vol_map, adv_map = prepare_vol_adv(config, final_universe, set(preds_by_exec.keys()))
        
        # 10. 准备短腿择时数据
        short_allow_dates = prepare_short_timing(config, price_map, set(preds_by_exec.keys()))
        
        # 11. 中性化列表
        neutral_list = [s.strip().lower() for s in args["neutralize"].split(",") if s.strip()]
        
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
    
    def update_state(self, new_weights: Dict[str, float]) -> None:
        """更新策略状态"""
        self.prev_weights = new_weights.copy()
        # 更新协调器的状态
        self.entry.prev_weights = new_weights.copy()
        # 更新各个策略的状态
        for strategy in self.entry.strategies.values():
            if hasattr(strategy, 'prev_weights'):
                strategy.prev_weights = new_weights.copy()
