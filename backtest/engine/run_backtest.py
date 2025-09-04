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
import argparse, json
from pathlib import Path
import pandas as pd
import numpy as np
import backtrader as bt
import qlib
from qlib.data import D


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
            vol_map = {pd.Timestamp(d).normalize(): g[["instrument","sigma"]].reset_index(drop=True)
                       for d, g in vol.groupby("datetime") if pd.Timestamp(d).normalize() in set(dates)}

    # ADV$
    adv_map: dict[pd.Timestamp, pd.DataFrame] = {}
    if "$vwap" in df.columns and "adv_20" in df.columns and not df.empty:
        df["adv_dollar"] = (df["adv_20"].astype(float).shift(1) * df["$vwap"].astype(float).shift(1)).clip(lower=0.0)
        adv_map = {pd.Timestamp(d).normalize(): g[["instrument","adv_dollar"]].dropna().reset_index(drop=True)
                   for d, g in df.groupby("datetime") if pd.Timestamp(d).normalize() in set(dates)}

    return vol_map, adv_map


def neutralize_weights(targets: dict[str, float], expos_df: pd.DataFrame,
                       ridge_lambda: float = 1e-6, drop_dummy: bool = True,
                       keep_cols: list[str] | None = None) -> dict[str, float]:
    if expos_df is None or expos_df.empty or not targets:
        return targets.copy()
    df = expos_df.copy()
    cols = [c for c in df.columns if c not in ("instrument","datetime")]
    if keep_cols:
        cols = [c for c in cols if c in keep_cols]
    if not cols:
        return targets.copy()
    tdf = pd.DataFrame({"instrument": list(targets.keys()), "w": list(targets.values())})
    tdf["instrument"] = tdf["instrument"].astype(str).str.upper()
    dfm = tdf.merge(df[["instrument"] + cols], on="instrument", how="left").dropna()
    if dfm.empty:
        return targets.copy()
    use_cols = []
    for c in cols:
        v = dfm[c].to_numpy()
        if not np.allclose(v, v[0]):
            use_cols.append(c)
    if drop_dummy:
        for prefix in ("ind_", "liq_bucket_"):
            cand = [c for c in use_cols if c.startswith(prefix)]
            if len(cand) > 1:
                use_cols.remove(sorted(cand)[-1])
    if not use_cols:
        return targets.copy()
    X = dfm[use_cols].to_numpy(dtype=float)
    y = dfm["w"].to_numpy(dtype=float)
    try:
        XtX = X.T @ X
        beta = np.linalg.solve(XtX + ridge_lambda * np.eye(XtX.shape[0]), X.T @ y)
    except np.linalg.LinAlgError:
        return targets.copy()
    resid = y - X @ beta
    out = {ins: float(w) for ins, w in zip(dfm["instrument"], resid)}
    for k, v in targets.items():
        out.setdefault(k, float(v))
    return out


# --------- NEW: Water-filling 单腿分配 + 两腿封装 ----------
def _waterfill_one_leg(raw_pos: dict[str, float], target_sum: float, cap: float | dict[str,float]) -> dict[str, float]:
    """raw_pos>=0 的目标，按 cap 做硬上限分配，使和=target_sum（若不可行则全封顶）。"""
    raw = {k: max(0.0, float(v)) for k, v in raw_pos.items() if float(v) > 0}
    if not raw or target_sum <= 0:
        return {k: 0.0 for k in raw_pos}
    # caps
    if isinstance(cap, dict):
        caps = {k: float(cap.get(k, np.inf)) for k in raw.keys()}
    else:
        caps = {k: float(cap) for k in raw.keys()}
    # 不可行：总 cap < 目标
    if sum(caps.values()) + 1e-12 < float(target_sum):
        return {k: caps.get(k, 0.0) for k in raw_pos}
    A = set(raw.keys())
    w = {k: 0.0 for k in raw_pos}
    R = float(target_sum)
    # 循环 water-filling
    while A:
        denom = sum(raw[k] for k in A)
        if denom <= 0:
            # 原始权重全 0：平均分剩余
            share = R / len(A)
            for k in list(A):
                take = min(share, caps.get(k, np.inf))
                w[k] = take
            break
        s = R / denom
        overflow = [k for k in A if s * raw[k] > caps.get(k, np.inf) + 1e-15]
        if not overflow:
            for k in A:
                w[k] = s * raw[k]
            break
        for k in overflow:
            w[k] = caps.get(k, np.inf)
        R -= sum(caps.get(k, np.inf) for k in overflow)
        A -= set(overflow)
        if R <= 1e-15:  # 刚好分完
            break
    # 保持原 keys
    for k in raw_pos:
        w.setdefault(k, 0.0)
    return w


def waterfill_two_legs(weights: dict[str,float], long_exposure: float, short_exposure: float,
                       max_pos_per_name: float, allow_shorts: bool = True) -> dict[str,float]:
    """把净权重拆两腿做 water-filling，然后合并。"""
    w = pd.Series(weights, dtype=float)
    pos_raw = {k: float(v) for k,v in w[w>0].items()}
    neg_raw = {k: float(-v) for k,v in w[w<0].items()}  # 用正数做空腿分配
    cap = float(max(0.0, max_pos_per_name or 0.0))
    # 多腿
    wL = _waterfill_one_leg(pos_raw, max(0.0, float(long_exposure)), cap)
    # 空腿
    target_short = max(0.0, float(-short_exposure)) if allow_shorts else 0.0
    wS_pos = _waterfill_one_leg(neg_raw, target_short, cap) if target_short > 0 else {k:0.0 for k in neg_raw}
    # 合并
    out = {k: 0.0 for k in weights}
    for k,v in wL.items(): out[k] = out.get(k,0.0) + float(v)
    for k,v in wS_pos.items(): out[k] = out.get(k,0.0) - float(v)
    return out


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


def icdf_weights(ranks: np.ndarray) -> np.ndarray:
    """
    ranks: 1..K  (1=最好)
    返回未经缩放的 shape(K,) 权重，按 Φ^{-1}((r-0.5)/(K+1)) 计算，越靠前越大
    """
    def erfinv(x):
        a = 0.147
        ln = np.log(1 - x**2)
        s = np.sign(x)
        return s * np.sqrt(np.sqrt((2/(np.pi*a) + ln/2)**2 - ln/a) - (2/(np.pi*a) + ln/2))
    p = (ranks - 0.5) / (len(ranks) + 1.0)
    z = np.sqrt(2.0) * erfinv(2*p - 1)
    return z


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
        top_k=50, short_k=50,
        long_exposure=1.0, short_exposure=-1.0,
        max_pos_per_name=0.05,
        weight_scheme="equal",

        membership_buffer=0.2,
        smooth_eta=0.6,

        target_vol=0.0,
        leverage_cap=2.0,
        adv_limit_pct=0.0,

        # NEW: 短腿择时
        short_timing_on=False,
        short_timing_dates=None,   # set of dates 允许做空

        # NEW: 硬上限
        hard_cap=False,

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

    # ----- Helpers -----
    def _members_with_buffer(self, g: pd.DataFrame) -> tuple[list[str], list[str], pd.DataFrame]:
        buf = float(self.p.membership_buffer or 0.0)
        top_k = int(self.p.top_k); short_k = int(self.p.short_k)
        g = g.copy()

        use_rank = ("rank" in g.columns and
                    pd.to_numeric(g["rank"], errors="coerce").notna().sum() >= max(1, top_k + short_k))

        if use_rank:
            g["rank"] = pd.to_numeric(g["rank"], errors="coerce")
            g = g.sort_values("rank", na_position="last", kind="mergesort")
            # longs
            enter_long_thr = top_k
            exit_long_thr  = int(np.ceil(top_k * (1.0 + buf)))
            longs_enter = set(g.head(enter_long_thr)["instrument"])
            longs_keep = {ins for ins, w in self.prev_weights.items() if w > 0}
            longs_ok = set(g[g["rank"] <= exit_long_thr]["instrument"])
            longs = list((longs_enter | (longs_keep & longs_ok)) if top_k > 0 else set())
            # shorts
            g_tail = g.iloc[::-1].copy()
            enter_short_thr = short_k
            exit_short_thr  = int(np.ceil(short_k * (1.0 + buf)))
            shorts_enter = set(g_tail.head(enter_short_thr)["instrument"])
            shorts_keep = {ins for ins, w in self.prev_weights.items() if w < 0}
            shorts_ok = set(g_tail[g_tail["rank"] <= exit_short_thr]["instrument"])
            shorts = list((shorts_enter | (shorts_keep & shorts_ok)) if short_k > 0 else set())
        else:
            g = g.dropna(subset=["score"]).sort_values("score", ascending=False)
            longs_enter = set(g.head(top_k)["instrument"])
            exit_idx = int(np.ceil(top_k * (1.0 + buf)))
            longs_exit_zone = set(g.head(exit_idx)["instrument"])
            longs_keep = {ins for ins, w in self.prev_weights.items() if w > 0}
            longs = list((longs_enter | (longs_keep & longs_exit_zone)) if top_k > 0 else set())

            g_rev = g.iloc[::-1]
            shorts_enter = set(g_rev.head(short_k)["instrument"])
            exit_idx_s = int(np.ceil(short_k * (1.0 + buf)))
            shorts_exit_zone = set(g_rev.head(exit_idx_s)["instrument"])
            shorts_keep = {ins for ins, w in self.prev_weights.items() if w < 0}
            shorts = list((shorts_enter | (shorts_keep & shorts_exit_zone)) if short_k > 0 else set())

        if self.reb_counter < 10 and self.p.verbose:
            print(f"[rebalance {self.reb_counter}] longs={len(longs)} shorts={len(shorts)} (candidates={len(g)})")
        return longs, shorts, g

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

        # ---------- 选股 ----------
        g = self._preds.get(dtoday)
        if g is None or g.empty:
            if self.p.verbose:
                print(f"[warn] no predictions for exec day {dtoday.date()]}")
            self.diag_daily.append({
                "datetime": dtoday, "ret": total_ret, "ret_long": long_ret, "ret_short": short_ret,
                "turnover_pre": 0.0, "turnover_post": 0.0,
                "adv_clip_ratio": 0.0, "adv_clip_names": 0,
                "gross_long": float(w_prev.clip(lower=0.0).sum()),
                "gross_short": float(-w_prev.clip(upper=0.0).sum()),
                "commission_cum": float(self._commission_cum),
            })
            return

        longs, shorts, g_sorted = self._members_with_buffer(g)

        # 短腿择时：不允许做空则清空 short 候选
        allow_shorts_today = (not self.p.short_timing_on) or (dtoday in self._short_allow)
        if not allow_shorts_today:
            shorts = []

        # ---------- 权重生成（raw） ----------
        tgt = {}
        if self.p.weight_scheme == "icdf":
            if longs:
                dfL = g_sorted[g_sorted["instrument"].isin(longs)].copy()
                rankL = (np.arange(1, len(dfL)+1)).astype(float)
                wL = icdf_weights(rankL)
                wL = np.maximum(wL, 0.0)
                if wL.sum() > 0:
                    wL = wL / wL.sum() * max(0.0, float(self.p.long_exposure))
                for ins, w in zip(dfL["instrument"], wL):
                    tgt[ins] = tgt.get(ins, 0.0) + float(w)
            if shorts:
                dfS = g_sorted[g_sorted["instrument"].isin(shorts)].copy().iloc[::-1]
                rankS = (np.arange(1, len(dfS)+1)).astype(float)
                wS = -np.maximum(icdf_weights(rankS), 0.0)
                if -wS.sum() > 0 and self.p.short_exposure < 0:
                    wS = wS / (-wS.sum()) * float(self.p.short_exposure)
                for ins, w in zip(dfS["instrument"], wS):
                    tgt[ins] = tgt.get(ins, 0.0) + float(w)
        else:
            if len(longs) > 0 and self.p.long_exposure != 0.0:
                w = float(self.p.long_exposure) / float(len(longs))
                for s in longs: tgt[s] = tgt.get(s, 0.0) + w
            if len(shorts) > 0 and self.p.short_exposure != 0.0:
                w = float(self.p.short_exposure) / float(len(shorts))
                for s in shorts: tgt[s] = tgt.get(s, 0.0) + w

        # ---------- 中性化 ----------
        keep_cols = []
        if "beta" in self._neutral: keep_cols.append("mkt_beta_60")
        if "size" in self._neutral: keep_cols.append("ln_dollar_vol_20")
        expos_df = self._expos.get(dtoday)
        if expos_df is not None:
            if "sector" in self._neutral: keep_cols += [c for c in expos_df.columns if c.startswith("ind_")]
            if "liq"    in self._neutral: keep_cols += [c for c in expos_df.columns if c.startswith("liq_bucket_")]
        tgt = neutralize_weights(tgt, expos_df, ridge_lambda=self.p.ridge_lambda, drop_dummy=True, keep_cols=keep_cols)

        # ---------- 平滑 ----------
        eta = float(self.p.smooth_eta or 0.0)
        if eta > 0 and getattr(self, "prev_weights", None):
            w_prev_ser = pd.Series(self.prev_weights, dtype=float)
            w_tgt_ser  = pd.Series(tgt, dtype=float)
            all_idx = sorted(set(w_prev_ser.index) | set(w_tgt_ser.index))
            w_prev_ser = w_prev_ser.reindex(all_idx).fillna(0.0)
            w_tgt_ser  = w_tgt_ser.reindex(all_idx).fillna(0.0)
            w_new = eta * w_prev_ser + (1.0 - eta) * w_tgt_ser
            tgt = {k: float(v) for k, v in w_new.items()}

        # ---------- 目标波动率（先做缩放，再做硬上限） ----------
        tv = float(self.p.target_vol or 0.0)
        if tv > 0:
            sigma_df = self._vol.get(dtoday)
            if sigma_df is not None and not sigma_df.empty:
                sigmap = dict(zip(sigma_df["instrument"], sigma_df["sigma"]))
                w = pd.Series(tgt, dtype=float)
                s2 = np.array([ (sigmap.get(k, np.nan) or np.nan)**2 * (w.get(k,0.0)**2) for k in w.index ])
                s2 = s2[~np.isnan(s2)]
                if s2.size > 0 and np.nansum(s2) > 0:
                    est_ann = float(np.sqrt(np.nansum(s2)) * np.sqrt(252.0))
                    if est_ann > 1e-8:
                        scale = min(float(self.p.leverage_cap or 10.0), tv / est_ann)
                        w = (w * scale)
                        tgt = {k: float(v) for k, v in w.clip(-1.0, 1.0).items()}

        # ---------- 硬上限（water-filling） ----------
        if self.p.hard_cap and (self.p.max_pos_per_name is not None) and self.p.max_pos_per_name > 0:
            tgt = waterfill_two_legs(
                tgt,
                long_exposure=self.p.long_exposure,
                short_exposure=self.p.short_exposure,
                max_pos_per_name=float(self.p.max_pos_per_name),
                allow_shorts=allow_shorts_today
            )
        else:
            # 软上限回退：分别归一正负腿并裁剪
            w = pd.Series(tgt, dtype=float)
            w_pos = w[w>0]; w_neg = -w[w<0]
            if w_pos.sum() > 0:
                w_pos = w_pos / w_pos.sum() * max(0.0, float(self.p.long_exposure))
            if w_neg.sum() > 0 and allow_shorts_today:
                w_neg = w_neg / w_neg.sum() * max(0.0, float(-self.p.short_exposure))
            else:
                w_neg = w_neg * 0.0
            w2 = pd.concat([w_pos, -w_neg]).reindex(w.index).fillna(0.0)
            cap = float(self.p.max_pos_per_name or 0.0)
            if cap > 0:
                w2 = w2.clip(-cap, cap)
            tgt = {k: float(v) for k,v in w2.items()}

        # 为诊断记录 pre-ADV（硬上限后）
        tgt_pre_adv = tgt.copy()

        # ---------- %ADV 限速 ----------
        adv_df = self._adv.get(dtoday)
        tgt, diag_adv = apply_adv_limit(self.prev_weights, tgt, adv_df, self.broker.getvalue(), adv_limit_pct=float(self.p.adv_limit_pct))

        # ---------- 诊断：turnover / adv clip ----------
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
            if sym in tgt: continue
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

        self.prev_weights = tgt.copy()
        self.reb_counter += 1


# ----------------------------- CLI & main -----------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qlib_dir", required=True)
    ap.add_argument("--preds", required=True)
    ap.add_argument("--features_path", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)

    ap.add_argument("--trade_at", choices=["open","close"], default="open")
    ap.add_argument("--exec_lag", type=int, default=0, help="执行延迟 N 个交易日（0=同日）")

    ap.add_argument("--neutralize", default="", help="beta,sector,liq,size")
    ap.add_argument("--ridge_lambda", type=float, default=1e-6)

    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--short_k", type=int, default=50)
    ap.add_argument("--long_exposure", type=float, default=1.0)
    ap.add_argument("--short_exposure", type=float, default=-1.0)
    ap.add_argument("--max_pos_per_name", type=float, default=0.05)
    ap.add_argument("--weight_scheme", choices=["equal","icdf"], default="equal")

    ap.add_argument("--membership_buffer", type=float, default=0.2)
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

    # NEW: 短腿择时参数（动量 ≤ 阈值时允许做空；信号用 T-1）
    ap.add_argument("--short_timing_mom63", action="store_true", help="启用短腿择时：SPY 63日动量≤阈值时允许做空（用T-1信息）")
    ap.add_argument("--short_timing_threshold", type=float, default=0.0, help="短腿择时动量阈值，默认0.0")
    ap.add_argument("--short_timing_lookback", type=int, default=63, help="短腿择时回看窗口，默认63")

    # NEW: 硬上限开关（由 CLI 控制；不再强制 True）
    ap.add_argument("--hard_cap", action="store_true", help="使用硬上限 water-filling 单票上限分配")

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
        top_k=args.top_k, short_k=args.short_k,
        long_exposure=args.long_exposure, short_exposure=args.short_exposure,
        max_pos_per_name=args.max_pos_per_name,
        weight_scheme=args.weight_scheme,
        membership_buffer=args.membership_buffer,
        smooth_eta=args.smooth_eta,
        target_vol=args.target_vol,
        leverage_cap=args.leverage_cap,
        adv_limit_pct=args.adv_limit_pct,
        # NEW
        short_timing_on=bool(args.short_timing_mom63),
        short_timing_dates=short_allow_dates,
        hard_cap=bool(args.hard_cap),  # 修复：尊重 CLI，默认 False，仅传入时启用
        verbose=args.verbose,
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

    # KPI
    dd = results[0].analyzers.dd.get_analysis()
    ret = retdf["ret"].to_numpy()
    ann = np.sqrt(252.0)
    sharpe = float(np.nanmean(ret) / (np.nanstd(ret, ddof=1) + 1e-12) * ann) if len(ret) > 2 else float('nan')
    cagr = float((eq["value"].iloc[-1] / eq["value"].iloc[0]) ** (252.0 / max(1, len(eq))) - 1.0) if len(eq) > 1 else float('nan')
    mdd = float(dd.get('max', {}).get('drawdown', np.nan))

    # 诊断聚合
    turn_mean = float(diagdf["turnover_post"].mean()) if "turnover_post" in diagdf else float('nan')
    turn_p90  = float(diagdf["turnover_post"].quantile(0.9)) if "turnover_post" in diagdf else float('nan')
    adv_hit_days = float((diagdf["adv_clip_names"] > 0).mean()) if "adv_clip_names" in diagdf else 0.0
    adv_clip_avg = float(diagdf["adv_clip_ratio"].replace([np.inf,-np.inf], np.nan).fillna(0.0).mean()) if "adv_clip_ratio" in diagdf else 0.0
    gross_long_avg  = float(diagdf["gross_long"].mean()) if "gross_long" in diagdf else float('nan')
    gross_short_avg = float(diagdf["gross_short"].mean()) if "gross_short" in diagdf else float('nan')
    # 长短腿 Sharpe
    if {"ret_long","ret_short"} <= set(diagdf.columns):
        rl = diagdf["ret_long"].to_numpy()
        rs = diagdf["ret_short"].to_numpy()
        sharpe_long  = float(np.nanmean(rl) / (np.nanstd(rl, ddof=1) + 1e-12) * ann) if len(rl) > 2 else float('nan')
        sharpe_short = float(np.nanmean(rs) / (np.nanstd(rs, ddof=1) + 1e-12) * ann) if len(rs) > 2 else float('nan')
    else:
        sharpe_long = sharpe_short = float('nan')

    summary = {
        "start": args.start, "end": args.end,
        "cash_init": float(eq["value"].iloc[0]) if len(eq) else float(args.cash),
        "cash_end": float(eq["value"].iloc[-1]) if len(eq) else float(args.cash),
        "top_k": args.top_k, "short_k": args.short_k,
        "long_exposure": args.long_exposure, "short_exposure": args.short_exposure,
        "commission_bps": args.commission_bps, "slippage_bps": args.slippage_bps,
        "trade_at": args.trade_at, "neutralize": [s for s in [s.strip() for s in args.neutralize.split(",")] if s],
        "membership_buffer": args.membership_buffer, "smooth_eta": args.smooth_eta,
        "days": int(len(eq)), "CAGR": cagr, "Sharpe": sharpe, "MDD_pct": mdd,
        "rebalance_days": len(set(list(results[0].p.preds_by_exec.keys()))),
        "universe_size": len(price_map),
        "avg_candidates_per_reb": float(np.mean([len(g) for g in results[0].p.preds_by_exec.values()])),
        "exec_lag": int(args.exec_lag),
        "target_vol": float(args.target_vol),
        "weight_scheme": args.weight_scheme,
        "adv_limit_pct": float(args.adv_limit_pct),
        "hard_cap": bool(args.hard_cap),
        "short_timing_mom63": bool(args.short_timing_mom63),
        "short_timing_threshold": float(args.short_timing_threshold),
        "short_timing_lookback": int(args.short_timing_lookback),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("[summary]", json.dumps(summary, indent=2))

    # 额外 KPI 导出
    kpis = {
        "turnover_mean": turn_mean,
        "turnover_p90": turn_p90,
        "adv_clip_days_frac": adv_hit_days,
        "adv_clip_ratio_avg": adv_clip_avg,
        "gross_long_avg": gross_long_avg,
        "gross_short_avg": gross_short_avg,
        "commission_total": float(strat._commission_cum if hasattr(strat, "_commission_cum") else 0.0),
        "sharpe_long": sharpe_long,
        "sharpe_short": sharpe_short,
    }
    with open(out_dir / "kpis.json", "w") as f:
        json.dump(kpis, f, indent=2)
    print(f"[saved] -> {out_dir}")

if __name__ == "__main__":
    main()
