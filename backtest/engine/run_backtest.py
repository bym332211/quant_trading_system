#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_backtest.py  (v3.4.1 with diagnostics + short-leg timing + hard-cap water-filling)

在 v3.3.1 基础上新增：
- 短腿择时：基于锚标的（默认 SPY）63 日动量，使用 T-1 信息，**动量 ≤ 阈值时允许做空**
- 硬上限 Water-filling：在单票上限下精确分配多/空腿目标权重（不二次超限），替代 clip+归一

权重流程：中性化 → 平滑 → 目标波动缩放 → 硬上限 → %ADV 限速
"""

from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import pandas as pd
import numpy as np
import backtrader as bt
import qlib
from qlib.data import D
import yaml  # pip install pyyaml

# KPI计算模块
from backtest.kpi.calculator import KPICalculator

# --- fallback: allow running this file directly without -m by injecting project root ---
try:
    _PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))
except Exception:
    pass

# external selection module
from strategies import select_members_with_buffer, EntryStrategy, ExitStrategyCoordinator


# ----------------------------- Utils -----------------------------
def ensure_inst_dt(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if "instrument" not in d.columns:
        for c in ["symbol","ticker","Instrument","Symbol","TICKER"]:
            if c in d.columns:
                d = d.rename(columns={c:"instrument"})
                break
    if "datetime" not in d.columns:
        for c in ["date","Date","timestamp","Timestamp","DATETIME"]:
            if c in d.columns:
                d = d.rename(columns={c:"datetime"})
                break
    d["instrument"] = d["instrument"].astype(str).str.upper()
    d["datetime"] = pd.to_datetime(d["datetime"], utc=False)
    return d


def weekly_schedule(trading_days: pd.DatetimeIndex) -> list[pd.Timestamp]:
    idx = pd.DatetimeIndex(trading_days).sort_values()
    df = pd.DataFrame(index=idx)
    iso = df.index.isocalendar()
    tmp = df.reset_index(names="dt")
    tmp["y"] = iso.year.to_numpy()
    tmp["w"] = iso.week.to_numpy()
    sched = tmp.groupby(["y","w"], sort=True)["dt"].min().sort_values().tolist()
    return [pd.Timestamp(d).normalize() for d in sched]


def asof_map_schedule_to_pred(sched_dates: list[pd.Timestamp],
                              pred_days: pd.DatetimeIndex) -> dict[pd.Timestamp, pd.Timestamp]:
    pred_days = pd.DatetimeIndex(pred_days).sort_values()
    mapping = {}
    for d in sorted(pd.DatetimeIndex(sched_dates)):
        pos = pred_days.searchsorted(d, side="right") - 1
        if pos >= 0:
            mapping[pd.Timestamp(d).normalize()] = pd.Timestamp(pred_days[pos]).normalize()
    return mapping  # {schedule_day -> pred_day}


def load_qlib_ohlcv(symbols: list[str], start: str, end: str, qlib_dir: str) -> dict[str, pd.DataFrame]:
    qlib.init(provider_uri=str(Path(qlib_dir).expanduser().resolve()), region="us")
    fields = ["$open","$high","$low","$close","$volume"]
    raw = D.features(symbols, fields, start_time=start, end_time=end, freq="day")
    if raw.empty:
        raise RuntimeError("Qlib 返回空数据。")
    out = {}
    for sym, df_sym in raw.groupby(level=0):
        sub = df_sym.droplevel(0).sort_index()
        dfbt = pd.DataFrame({
            "open":   sub["$open"].astype(float),
            "high":   sub["$high"].astype(float),
            "low":    sub["$low"].astype(float),
            "close":  sub["$close"].astype(float),
            "volume": sub["$volume"].astype(float),
        })
        dfbt.index = pd.to_datetime(dfbt.index, utc=False)
        out[str(sym).upper()] = dfbt
    return out


def build_exposures_map(features_path: str, universe: set[str], dates: list[pd.Timestamp], use_items: list[str]) -> dict[pd.Timestamp, pd.DataFrame]:
    if not use_items:
        return {}
    use_items = [x.strip().lower() for x in use_items]
    dfh = pd.read_parquet(Path(features_path).expanduser().resolve(), columns=None)
    all_cols = set(dfh.columns)
    need = {"instrument","datetime"}
    if "beta" in use_items: need.add("mkt_beta_60")
    if "size" in use_items: need.add("ln_dollar_vol_20")
    if "sector" in use_items: need |= {c for c in all_cols if c.startswith("ind_")}
    if "liq"   in use_items: need |= {c for c in all_cols if c.startswith("liq_bucket_")}
    need = [c for c in need if c in all_cols]
    df = dfh[need].copy()
    df["instrument"] = df["instrument"].astype(str).str.upper()
    df["datetime"] = pd.to_datetime(df["datetime"], utc=False)
    df = df[df["instrument"].isin(universe) & df["datetime"].isin(dates)]
    out = {}
    for d, g in df.groupby("datetime"):
        out[pd.Timestamp(d).normalize()] = g.reset_index(drop=True)
    return out


def compute_vol_adv_maps(features_path: str, universe: set[str], dates: list[pd.Timestamp],
                         halflife: int = 20) -> tuple[dict[pd.Timestamp, pd.DataFrame], dict[pd.Timestamp, pd.DataFrame]]:
    """
    返回:
      vol_map[date]: DataFrame[instrument, sigma]  (日波动率, EWM std, 以 date-1 为止)
      adv_map[date]: DataFrame[instrument, adv_dollar] (≈ ADV20 * vwap, 以 date-1 为止)
    """
    need = ["instrument","datetime","ret_1","adv_20","$vwap"]
    df = pd.read_parquet(Path(features_path).expanduser().resolve(), columns=None)
    # 稳健性增强 #1：缺列时降级
    exist = set(df.columns)
    subset = [c for c in need if c in exist]
    df = df[subset].copy()
    df["instrument"] = df["instrument"].astype(str).str.upper()
    df["datetime"]   = pd.to_datetime(df["datetime"], utc=False)
    df = df[df["instrument"].isin(universe)]
    last_need = max(dates) if dates else df["datetime"].max()
    df = df[df["datetime"] <= last_need]

    # sigma (EWM std) 以 T-1 截止；若无 ret_1 或有效样本不足则空映射
    vol_map: dict[pd.Timestamp, pd.DataFrame] = {}
    if "ret_1" in df.columns and not df.empty:
        vol = (df.sort_values(["instrument","datetime"])
                 .groupby("instrument", group_keys=False)["ret_1"]
                 .apply(lambda s: s.ewm(halflife=halflife, min_periods=max(5, halflife//2)).std())
                 .to_frame("sigma"))
        vol = vol.join(df[["instrument","datetime"]]).sort_values(["instrument","datetime"])
        vol["sigma"] = vol.groupby("instrument", group_keys=False)["sigma"].shift(1)
        vol = vol.dropna(subset=["sigma"])
        if not vol.empty:
            vol_map = {pd.Timestamp(d).normalize() : g[["instrument","sigma"]].reset_index(drop=True)
                       for d, g in vol.groupby("datetime") if pd.Timestamp(d).normalize() in set(dates)}

    # ADV$
    adv_map: dict[pd.Timestamp, pd.DataFrame] = {}
    if "$vwap" in df.columns and "adv_20" in df.columns and not df.empty:
        df["adv_dollar"] = (df["adv_20"].astype(float).shift(1) * df["$vwap"].astype(float).shift(1)).clip(lower=0.0)
        adv_map = {pd.Timestamp(d).normalize(): g[["instrument","adv_dollar"]].dropna().reset_index(drop=True)
                   for d, g in df.groupby("datetime") if pd.Timestamp(d).normalize() in set(dates)}

    return vol_map, adv_map








def apply_adv_limit(prev_w: dict[str,float], tgt_w: dict[str,float],
                    adv_df: pd.DataFrame | None, port_value: float, adv_limit_pct: float) -> tuple[dict[str,float], dict]:
    """
    限制单次换手 |Δw| ≤ adv_limit_pct * ADV$ / PortValue
    返回 (新权重, 诊断字典)
    """
    diag = {"hit_names": 0, "clip_sum": 0.0, "delta_pre_sum": 0.0}
    if adv_limit_pct is None or adv_limit_pct <= 0 or adv_df is None or adv_df.empty:
        return tgt_w, diag
    lim = float(adv_limit_pct)
    adv_map = dict(zip(adv_df["instrument"], adv_df["adv_dollar"]))
    out = {}
    for sym, tw in tgt_w.items():
        pw = prev_w.get(sym, 0.0)
        adv_dol = float(adv_map.get(sym, np.nan))
        if np.isnan(adv_dol) or adv_dol <= 0 or port_value <= 0:
            out[sym] = tw
            continue
        max_delta = lim * (adv_dol / port_value)
        delta_pre = tw - pw
        delta = np.clip(delta_pre, -max_delta, max_delta)
        if abs(delta) + 1e-12 < abs(delta_pre):
            diag["hit_names"] += 1
            diag["clip_sum"] += float(abs(delta_pre) - abs(delta))
        diag["delta_pre_sum"] += float(abs(delta_pre))
        out[sym] = pw + float(delta)
    return out, diag




# ----------------------------- Strategy -----------------------------
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
        self.entry = EntryStrategy(
            neutralize_items=self._neutral,
            ridge_lambda=float(self.p.ridge_lambda),
            top_k=int(self.p.top_k),
            short_k=int(self.p.short_k),
            membership_buffer=float(self.p.membership_buffer or 0.0),
            selection_use_rank_mode=str(getattr(self.p, "selection_use_rank_mode", "auto")),
            long_exposure=float(self.p.long_exposure),
            short_exposure=float(self.p.short_exposure),
            max_pos_per_name=float(self.p.max_pos_per_name or 0.0),
            weight_scheme=str(self.p.weight_scheme),
            smooth_eta=float(self.p.smooth_eta or 0.0),
            target_vol=float(self.p.target_vol or 0.0),
            leverage_cap=float(self.p.leverage_cap or 10.0),
            hard_cap=bool(self.p.hard_cap),
            verbose=bool(self.p.verbose),
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

        # 使用 EntryStrategy 生成“ADV 限速之前”的目标权重
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
        # 注意：此处使用“更新前”的 prev_weights 计算换手
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


# ----------------------------- CLI & main -----------------------------
def parse_args():
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
    ap.add_argument("--adv_limit_pct", type=float, default=0.0, help="单次换手 %ADV 限制，例如 0.005=0.5%%ADV")

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


def _compute_short_timing_dates(anchor_df: pd.DataFrame,
                                exec_dates: list[pd.Timestamp],
                                lookback: int = 63,
                                thr: float = 0.0) -> set[pd.Timestamp]:
    """
    使用锚标的收盘价计算动量：mom = close / close.shift(lookback) - 1
    采用 T-1 信息：signal_asof = (mom <= thr).shift(1)
    在 signal_asof 为 True 的执行日**允许做空**。
    """
    # 稳健性增强 #2：历史不足或无效则“全允许做空”，避免误关短腿
    if anchor_df is None or anchor_df.empty or "close" not in anchor_df.columns:
        return set(pd.DatetimeIndex(exec_dates))
    s = anchor_df["close"].astype(float).dropna()
    if s.empty or len(s) <= int(lookback) + 1:
        return set(pd.DatetimeIndex(exec_dates))

    mom = s / s.shift(int(lookback)) - 1.0
    sig = (mom <= float(thr)).astype(float)
    sig_asof = sig.shift(1)  # 用 T-1
    idx = s.index
    allowed = set()
    for ed in pd.DatetimeIndex(exec_dates):
        i = idx.searchsorted(ed) - 1
        if i >= 0 and not np.isnan(sig_asof.iloc[i]):
            if bool(sig_asof.iloc[i] == 1.0):
                allowed.add(pd.Timestamp(ed).normalize())
        elif i >= 0:
            # 缺失时放行
            allowed.add(pd.Timestamp(ed).normalize())
    return allowed


def main():
    args = parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- 读取 YAML ---
    cfg = {}
    cfg_path = Path(getattr(args, "config", "config/config.yaml")).expanduser()
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

    # 策略配置加载：支持新的分层结构
    strategies_cfg = (cfg.get("strategies") or {})
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

    sel_top_k    = int(sel_cfg.get("top_k", 50))
    sel_short_k  = int(sel_cfg.get("short_k", 50))
    sel_buffer   = float(sel_cfg.get("membership_buffer", 0.2))
    sel_use_rank = str(sel_cfg.get("use_rank", "auto")).strip().lower()
    
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

    # 预测
    preds_all = ensure_inst_dt(pd.read_parquet(Path(args.preds).expanduser().resolve()))
    mask = (preds_all["datetime"] >= pd.Timestamp(args.start)) & (preds_all["datetime"] <= pd.Timestamp(args.end))
    preds_all = preds_all.loc[mask].copy()
    if preds_all.empty:
        raise RuntimeError("预测为空。")
    pred_days = pd.DatetimeIndex(preds_all["datetime"].unique()).sort_values()

    # 初始 universe：窗口内出现过的所有票 + 锚
    anchor_sym = args.anchor_symbol.upper() if args.anchor_symbol else None
    universe_all = set(preds_all["instrument"].astype(str).str.upper().unique().tolist())
    if anchor_sym:
        universe_all.add(anchor_sym)

    # 行情（锚在第一位）
    price_map = load_qlib_ohlcv(sorted(list(universe_all)), start=args.start, end=args.end, qlib_dir=args.qlib_dir)
    if anchor_sym and anchor_sym in price_map:
        anchor_days = pd.DatetimeIndex(price_map[anchor_sym].index)
        anchor_first = {anchor_sym: price_map.pop(anchor_sym)}
        price_map = {**anchor_first, **price_map}
    else:
        anchor_days = pd.DatetimeIndex(next(iter(price_map.values())).index)

    # 周频锚定 -> 源预测日 as-of 映射 -> exec_lag 推进到“执行日”
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
        j = i + max(0, int(args.exec_lag))
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
        j = i + max(0, int(args.exec_lag))
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
    neutral_list = [s.strip().lower() for s in args.neutralize.split(",") if s.strip()]
    expos_map = build_exposures_map(args.features_path, universe=set(price_map.keys()),
                                    dates=list(preds_by_exec.keys()), use_items=neutral_list)
    vol_map, adv_map = compute_vol_adv_maps(args.features_path, universe=set(price_map.keys()),
                                            dates=list(preds_by_exec.keys()), halflife=int(args.ewm_halflife))

    # NEW: 短腿择时日期集合
    short_allow_dates = set()
    if args.short_timing_mom63 and anchor_sym and anchor_sym in price_map:
        short_allow_dates = _compute_short_timing_dates(
            price_map[anchor_sym],
            exec_dates=list(preds_by_exec.keys()),
            lookback=int(args.short_timing_lookback),
            thr=float(args.short_timing_threshold)
        )

    # Backtrader
    cerebro = bt.Cerebro(stdstats=True)
    cerebro.broker.setcash(float(args.cash))
    cerebro.broker.setcommission(commission=float(args.commission_bps) / 10000.0)
    cerebro.broker.set_slippage_perc(perc=float(args.slippage_bps) / 10000.0,
                                     slip_open=True, slip_limit=True, slip_match=True)
    if args.trade_at == "close":
        cerebro.broker.set_coc(True)

    for sym, df in price_map.items():
        datafeed = bt.feeds.PandasData(dataname=df, name=sym)
        cerebro.adddata(datafeed)

    cerebro.addstrategy(
        XSecRebalance,
        preds_by_exec=preds_by_exec,
        exec_dates=list(preds_by_exec.keys()),
        exposures_by_date=expos_map,
        vol_by_date=vol_map,
        adv_by_date=adv_map,
        neutralize_items=tuple(neutral_list),
        ridge_lambda=args.ridge_lambda,
        trade_at=args.trade_at,

        # 来自 config/CLI 合并
        top_k=sel_top_k,
        short_k=sel_short_k,
        membership_buffer=sel_buffer,
        selection_use_rank_mode=sel_use_rank,

        long_exposure=args.long_exposure, short_exposure=args.short_exposure,
        max_pos_per_name=args.max_pos_per_name,
        weight_scheme=args.weight_scheme,
        smooth_eta=args.smooth_eta,
        target_vol=args.target_vol,
        leverage_cap=args.leverage_cap,
        adv_limit_pct=args.adv_limit_pct,
        # NEW
        short_timing_on=bool(args.short_timing_mom63),
        short_timing_dates=short_allow_dates,
        hard_cap=bool(args.hard_cap),
        verbose=args.verbose,
        
        # 出场策略配置
        exit_strategies_config=exit_cfg,
    )
    cerebro.addanalyzer(bt.analyzers.TimeReturn, timeframe=bt.TimeFrame.Days, _name='timeret', fund=False)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='dd')

    results = cerebro.run(maxcpus=1)
    strat: XSecRebalance = results[0]

    # 导出曲线
    eq = pd.DataFrame(strat.val_records).drop_duplicates(subset=["datetime"]).sort_values("datetime")
    eq["ret"] = eq["value"].pct_change().fillna(0.0)
    out_dir.mkdir(parents=True, exist_ok=True)
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

    # KPI计算 - 使用新的KPICalculator模块
    dd = results[0].analyzers.dd.get_analysis()
    commission_total = float(strat._commission_cum if hasattr(strat, "_commission_cum") else 0.0)
    
    # 使用KPICalculator计算所有指标
    summary, kpis = KPICalculator.calculate_all_kpis(
        args=args,
        eq_df=eq,
        ret_df=retdf,
        diag_df=diagdf,
        dd_analysis=dd,
        strat=strat,
        price_map=price_map,
        sel_cfg=sel_cfg,
        commission_total=commission_total
    )
    
    # 保存KPI指标到文件
    KPICalculator.save_kpis_to_files(out_dir, summary, kpis)
    print("[summary]", json.dumps(summary, indent=2))
    print(f"[saved] -> {out_dir}")

if __name__ == "__main__":
    main()
