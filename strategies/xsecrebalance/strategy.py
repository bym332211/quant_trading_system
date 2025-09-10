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
# 注：XSecRebalance 直接继承 Backtrader 的 Strategy；
# BaseStrategy 仅作为文档化接口存在，工厂返回的是 Backtrader 策略类。


class XSecRebalance(bt.Strategy):
    params = dict(
        preds_by_exec=None,
        exec_dates=None,
        exposures_by_date=None,
        vol_by_date=None,
        adv_by_date=None,
        liq_bucket_by_date=None,
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
        self._liq_buckets = self.p.liq_bucket_by_date or {}
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
        
        # 初始化入场策略协调器（合并顶层 entry 配置 → icdf 配置）
        icdf_equal_config = self._build_icdf_config(self._entry_strategies_config)
        enabled_strategies = self._entry_strategies_config.get("enabled_strategies", ["icdf_equal"])
        strategy_weights = self._entry_strategies_config.get("strategy_weights", {"icdf_equal": 1.0})

        self.entry = EntryStrategyCoordinator(
            icdf_equal_config=icdf_equal_config,
            enabled_strategies=enabled_strategies,
            strategy_weights=strategy_weights,
            # 组合后进行两腿归一与上限控制（使用顶层 entry 配置）
            post_normalize=True,
            long_exposure=float(self.p.long_exposure),
            short_exposure=float(self.p.short_exposure),
            max_pos_per_name=float(self.p.max_pos_per_name),
        )
        
        # 存储出场策略配置（用于后续初始化）
        self._exit_strategies_config = getattr(self.p, "exit_strategies_config", {})

        # 历史数据缓存与动态窗口
        self._hist_cache: dict[str, pd.DataFrame] = {}
        self._exit_lookback = self._compute_exit_lookback(self._exit_strategies_config)
        # 读取过滤配置
        self._filters_cfg = (self._entry_strategies_config or {}).get("filters", {}) or {}

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
        
        # 先增量更新历史缓存（所有数据源），避免每次重建 DataFrame
        self._update_all_hist_cache(dtoday)

        # 检查每个持仓是否需要出场
        exit_symbols = []
        for d in self.datas:
            pos = self.getposition(d)
            if pos.size != 0:
                symbol = d._name
                current_price = float(d.close[0])
                
                # 使用缓存的历史窗口；若样本不足则跳过出场判断
                historical_data = self._hist_cache.get(symbol)
                if historical_data is not None and len(historical_data) >= self._exit_lookback:
                    
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

        # 应用入场过滤（行业×流动性桶）
        expos_df_today = self._expos.get(dtoday)
        if g is not None and not g.empty:
            g = self._apply_entry_filters(dtoday, g, expos_df_today)

        # 使用 EntryStrategy 生成"ADV 限速之前"的目标权重
        tgt_pre_adv = self.entry.generate_entry_weights(
            g=g,
            prev_weights=self.prev_weights or {},
            expos_df=expos_df_today,
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
        # 与 EntryStrategyCoordinator 的计数对齐：只在协调器侧自增
        self.reb_counter = self.entry.reb_counter

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
        
        # 重新初始化入场策略协调器（配置可能变化）
        icdf_equal_config = self._build_icdf_config(self._entry_strategies_config)
        enabled_strategies = self._entry_strategies_config.get("enabled_strategies", ["icdf_equal"])
        strategy_weights = self._entry_strategies_config.get("strategy_weights", {"icdf_equal": 1.0})

        self.entry = EntryStrategyCoordinator(
            icdf_equal_config=icdf_equal_config,
            enabled_strategies=enabled_strategies,
            strategy_weights=strategy_weights,
            post_normalize=True,
            long_exposure=float(self.p.long_exposure),
            short_exposure=float(self.p.short_exposure),
            max_pos_per_name=float(self.p.max_pos_per_name),
        )
        
        # 存储出场策略配置并更新动态窗口
        self._exit_strategies_config = getattr(self.p, "exit_strategies_config", {})
        self._exit_lookback = self._compute_exit_lookback(self._exit_strategies_config)
        # 重置历史缓存（配置变化时）
        self._hist_cache = {}
    
    def entry_strategy(self, dtoday: pd.Timestamp, preds_df: pd.DataFrame, 
                      prev_weights: Dict[str, float]) -> Dict[str, float]:
        """入场策略：生成初步目标权重"""
        g = self._preds.get(dtoday)
        if g is None or g.empty:
            return {}
        
        allow_shorts_today = (not self.p.short_timing_on) or (dtoday in self._short_allow)
        
        result = self.entry.generate_entry_weights(
            g=g,
            prev_weights=prev_weights or {},
            expos_df=self._expos.get(dtoday),
            vol_df=self._vol.get(dtoday),
            allow_shorts=allow_shorts_today,
            reb_counter=int(self.reb_counter),
        )
        # 同步本地计数，避免双重自增
        self.reb_counter = self.entry.reb_counter
        return result
    
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
            "reb_counter": self.entry.reb_counter,
            "exit_stats": getattr(self.exit_coordinator, 'get_stats', lambda: {})()
        }
    
    @classmethod
    def prepare_data(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        """XSecRebalance 策略的数据准备"""
        from strategies.xsecrebalance.data_preparation import (
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
        
        # 9. 准备波动率和ADV数据 + 流动性桶
        vol_map, adv_map, liq_map = prepare_vol_adv(config, final_universe, set(preds_by_exec.keys()))
        
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
            "liq_bucket_by_date": liq_map,
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

    def _build_icdf_config(self, entry_cfg: Dict[str, Any]) -> Dict[str, Any]:
        """合并顶层 entry 配置到 icdf_equal 子配置，确保参数一致性。"""
        sub = (entry_cfg or {}).get("icdf_equal", {}) or {}
        base = entry_cfg or {}
        # 仅挑选 ICDF 支持的键进行合并（子配置优先）
        # 注意：filters 属于上层配置，不能传入子策略构造函数
        keys = [
            "neutralize_items", "ridge_lambda", "top_k", "short_k",
            "membership_buffer", "selection_use_rank_mode", "long_exposure",
            "short_exposure", "max_pos_per_name", "weight_scheme", "smooth_eta",
            "target_vol", "leverage_cap", "hard_cap", "verbose",
        ]
        merged = {k: base[k] for k in keys if k in base}
        merged.update(sub)
        return merged

    def _apply_entry_filters(self, dtoday: pd.Timestamp, g: pd.DataFrame, expos_df: pd.DataFrame | None) -> pd.DataFrame:
        """按配置过滤不入场的标的：行业×流动性桶。

        配置示例（config.entry.filters.sector_liq_exclude）：
          filters:
            sector_liq_exclude:
              Technology: [4, 3]
              Energy: [3]
        """
        rules = (self._filters_cfg or {}).get("sector_liq_exclude", {}) or {}
        if not isinstance(rules, dict) or g is None or g.empty:
            return g

        # instrument -> sector (from ind_* one-hot in exposures)
        inst2sector: dict[str, str] = {}
        if expos_df is not None and not expos_df.empty:
            sector_cols = [c for c in expos_df.columns if isinstance(c, str) and c.startswith("ind_")]
            if sector_cols:
                edf = expos_df[["instrument"] + sector_cols].copy()
                for _, row in edf.iterrows():
                    inst = str(row["instrument"]).upper()
                    sec = None
                    for c in sector_cols:
                        try:
                            if float(row.get(c, 0.0)) > 0.5:
                                sec = c[4:]
                                break
                        except Exception:
                            continue
                    if sec:
                        inst2sector[inst] = sec

        # instrument -> liq_bucket (from precomputed map)
        inst2lb: dict[str, int] = {}
        liq_df = self._liq_buckets.get(dtoday)
        if liq_df is not None and not liq_df.empty and {"instrument","liq_bucket"} <= set(liq_df.columns):
            for _, r in liq_df.iterrows():
                try:
                    inst2lb[str(r["instrument"]).upper()] = int(r["liq_bucket"])
                except Exception:
                    continue

        def is_filtered(inst: str) -> bool:
            sec = inst2sector.get(inst)
            lb = inst2lb.get(inst)
            if sec is None or lb is None:
                return False
            buckets = rules.get(sec)
            if buckets is None:
                return False
            if isinstance(buckets, dict):
                buckets = [k for k, v in buckets.items() if v]
            try:
                return int(lb) in set(int(x) for x in buckets)
            except Exception:
                return False

        g2 = g.copy()
        g2["instrument"] = g2["instrument"].astype(str).str.upper()
        mask = g2["instrument"].map(lambda x: not is_filtered(x))
        return g2.loc[mask].reset_index(drop=True)

    # ----- helpers -----
    def _compute_exit_lookback(self, exit_cfg: Dict[str, Any]) -> int:
        """根据启用的出场策略，动态确定所需历史窗口长度。"""
        tech = (exit_cfg or {}).get("tech_stop_loss", {})
        vol  = (exit_cfg or {}).get("volatility_exit", {})

        periods = [
            int(tech.get("atr_period", 14)),
            int(tech.get("ma_stop_period", 20)),
            int(tech.get("bollinger_period", 20)),
            int(tech.get("rsi_period", 14)),
            int(vol.get("vol_period", 20)),
            int(vol.get("market_vol_period", 63)),
        ]
        # 至少留足 60 根，避免初期频繁不足
        return max(periods + [60])

    def _update_all_hist_cache(self, dtoday: pd.Timestamp) -> None:
        """增量更新所有数据源的历史窗口缓存。"""
        lookback = self._exit_lookback
        for d in self.datas:
            sym = d._name
            # 若数据长度不足则跳过
            if len(d) < 1:
                continue
            # 读取当前bar
            rec = {
                'open': float(d.open[0]),
                'high': float(d.high[0]),
                'low': float(d.low[0]),
                'close': float(d.close[0]),
                'volume': float(d.volume[0]) if not np.isnan(float(d.volume[0])) else 0.0,
            }
            # 使用真实交易日索引
            idx = pd.Timestamp(bt.num2date(d.datetime[0]).date())
            prev = self._hist_cache.get(sym)
            if prev is None:
                df = pd.DataFrame([rec], index=[idx])
            else:
                # 仅在新bar时追加
                if idx not in prev.index:
                    df = pd.concat([prev, pd.DataFrame([rec], index=[idx])])
                else:
                    # 回填/覆盖当日（如 close 模式多次 next）
                    df = prev.copy()
                    df.loc[idx] = rec
            # 截断窗口大小
            if len(df) > lookback:
                df = df.iloc[-lookback:]
            self._hist_cache[sym] = df
