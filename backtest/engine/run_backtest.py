#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_backtest.py  (v3.3.1 with diagnostics)

在 v3.3 基础上新增：
- 每日 turnover（pre-ADV / post-ADV），%ADV 限速命中率（裁剪票数与裁剪权重占比）
- 多/空腿日度贡献（ret_long, ret_short），平均敞口、长短腿 Sharpe、总佣金
- 导出 per_day_ext.csv 与 kpis.json

保持兼容：
- exec_lag / target_vol / %ADV 限速 / 权重方案 equal|icdf / 中性化 / 平滑 等均保留
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
    miss = set(need) - set(df.columns)
    if miss:
        # 允许无 adv/vwap（则只生成 vol_map）
        need = [c for c in need if c in df.columns]
    df = df[need].copy()
    df["instrument"] = df["instrument"].astype(str).str.upper()
    df["datetime"]   = pd.to_datetime(df["datetime"], utc=False)
    df = df[df["instrument"].isin(universe)]
    last_need = max(dates) if dates else df["datetime"].max()
    df = df[df["datetime"] <= last_need]

    vol = (df.sort_values(["instrument","datetime"])
             .groupby("instrument", group_keys=False)["ret_1"]
             .apply(lambda s: s.ewm(halflife=halflife, min_periods=max(5, halflife//2)).std())
             .to_frame("sigma"))
    vol = vol.join(df[["instrument","datetime"]]).sort_values(["instrument","datetime"])
    vol["sigma"] = vol.groupby("instrument", group_keys=False)["sigma"].shift(1)
    vol = vol.dropna(subset=["sigma"])
    vol_map = {pd.Timestamp(d).normalize(): g[["instrument","sigma"]].reset_index(drop=True)
               for d, g in vol.groupby("datetime") if pd.Timestamp(d).normalize() in set(dates)}

    adv_map = {}
    if "$vwap" in df.columns and "adv_20" in df.columns:
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


def rescale_and_clip(weights: dict[str, float],
                     long_exposure: float, short_exposure: float,
                     max_pos_per_name: float) -> dict[str, float]:
    if not weights:
        return {}
    w = pd.Series(weights, dtype=float)
    w_pos = w[w > 0]
    w_neg = -w[w < 0]
    if w_pos.sum() > 0:
        w_pos = w_pos / w_pos.sum() * max(0.0, float(long_exposure))
    if w_neg.sum() > 0:
        w_neg = w_neg / w_neg.sum() * max(0.0, float(-short_exposure))
    w2 = pd.concat([w_pos, -w_neg]).reindex(w.index).fillna(0.0)
    if max_pos_per_name and max_pos_per_name > 0:
        cap = float(max_pos_per_name)
        w2 = w2.clip(-cap, cap)
        pos = w2[w2 > 0].sum(); neg = -w2[w2 < 0].sum()
        if pos > 0 and long_exposure > 0:
            w2[w2 > 0] *= (long_exposure / pos)
        if neg > 0 and short_exposure < 0:
            w2[w2 < 0] *= (-short_exposure / neg)
    return {k: float(v) for k, v in w2.items()}


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

        self.data2sym = {d: d._name for d in self.datas}
        self.sym2data = {d._name: d for d in self.datas}

        # 记录
        self.val_records, self.order_records, self.pos_records = [], [], []
        self.prev_weights: dict[str, float] = {}
        self.reb_counter = 0

        # 诊断记录
        self.diag_daily = []   # 每日：ret, ret_long, ret_short, turnover_pre/post, adv_clip_ratio 等
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

        # ---------- 日度收益拆腿（使用昨日权重 * 今日 C2C 收益） ----------
        # 注意：这里的 ret 是 close/prev_close - 1，与 trade_at 无关（统一度量组合表现）
        w_prev = pd.Series(self.prev_weights, dtype=float)
        ret_map = {}
        for d in self.datas:
            if len(d) < 2:  # 需要至少两根K
                continue
            prev_c = float(d.close[-1])
            curr_c = float(d.close[0])
            if prev_c > 0:
                ret_map[d._name] = (curr_c / prev_c - 1.0)
        if len(w_prev) > 0 and ret_map:
            rets = pd.Series(ret_map).reindex(w_prev.index).fillna(0.0)
            total_ret = float((w_prev * rets).sum())
            long_ret  = float((w_prev.clip(lower=0.0) * rets).sum())
            short_ret = float((w_prev.clip(upper=0.0) * rets).sum())  # 负数代表短腿贡献为负
        else:
            total_ret, long_ret, short_ret = 0.0, 0.0, 0.0

        # 记录持仓快照
        for d in self.datas:
            pos = self.getposition(d)
            if pos.size != 0:
                self.pos_records.append({"datetime": dtoday, "instrument": d._name,
                                         "size": float(pos.size), "price": float(pos.price),
                                         "value": float(pos.size * d.close[0])})

        # ---------- 若不是执行日，记诊断后返回 ----------
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

        longs, shorts, g_sorted = self._members_with_buffer(g)

        # ---------- 权重生成 ----------
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
        tgt = rescale_and_clip(tgt, self.p.long_exposure, self.p.short_exposure, self.p.max_pos_per_name)

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

        # ---------- %ADV 限速 ----------
        adv_df = self._adv.get(dtoday)
        tgt_pre_adv = tgt.copy()
        tgt, diag_adv = apply_adv_limit(self.prev_weights, tgt, adv_df, self.broker.getvalue(), adv_limit_pct=float(self.p.adv_limit_pct))

        # ---------- 目标波动率 ----------
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
                        w = (w * scale).clip(-1.0, 1.0)
                        tgt = {k: float(v) for k, v in w.items()}

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
    ap.add_argument("--short_score_q", type=float, default=1.0,
                help="仅做空 score 分位 < q 的票；1.0 表示不限制")
    ap.add_argument("--short_liq_min_bucket", type=int, default=-1,
                help="仅做空流动性桶 >= 此阈值的票；-1 表示不限制")

    ap.add_argument("--membership_buffer", type=float, default=0.2)
    ap.add_argument("--smooth_eta", type=float, default=0.6)

    ap.add_argument("--target_vol", type=float, default=0.0, help="年化目标波动 (0 关闭)")
    ap.add_argument("--ewm_halflife", type=int, default=20, help="EWM 半衰期（用于 sigma 估计）")
    ap.add_argument("--leverage_cap", type=float, default=2.0)
    ap.add_argument("--adv_limit_pct", type=float, default=0.0, help="单次换手 %ADV 限制，例如 0.005=0.5%%ADV")

    ap.add_argument("--commission_bps", type=float, default=1.0)
    ap.add_argument("--slippage_bps", type=float, default=5.0)
    ap.add_argument("--cash", type=float, default=1_000_000.0)
    ap.add_argument("--anchor_symbol", default="SPY")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--verbose", action="store_true")
    return ap.parse_args()


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
        if i is None:  # 理论不应发生
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

    # 导出逐日收益（Backtrader analyzer 或我们自己算的）
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
    # 长短腿 Sharpe（用我们拆分的日度贡献）
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
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("[summary]", json.dumps(summary, indent=2))

    # 额外 KPI 导出
    kpis = {
        "turnover_mean": turn_mean,
        "turnover_p90": turn_p90,
        "adv_clip_days_frac": adv_hit_days,     # 有裁剪的天数占比
        "adv_clip_ratio_avg": adv_clip_avg,     # 被裁剪的换手占比（对有目标换手的截面）
        "gross_long_avg": gross_long_avg,
        "gross_short_avg": gross_short_avg,
        "commission_total": float(results[0]._commission_cum if hasattr(results[0], "_commission_cum") else 0.0),
        "sharpe_long": sharpe_long,
        "sharpe_short": sharpe_short,
    }
    with open(out_dir / "kpis.json", "w") as f:
        json.dump(kpis, f, indent=2)
    print(f"[saved] -> {out_dir}")

if __name__ == "__main__":
    main()
