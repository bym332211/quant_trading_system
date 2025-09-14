"""
数据加载模块 - 提取自 run_backtest.py 的数据处理逻辑
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Tuple, Set, List
import pandas as pd
import numpy as np
import qlib
from qlib.data import D


def ensure_inst_dt(df: pd.DataFrame) -> pd.DataFrame:
    """确保DataFrame包含instrument和datetime列"""
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


def weekly_schedule(trading_days: pd.DatetimeIndex) -> List[pd.Timestamp]:
    """生成周频调仓计划"""
    idx = pd.DatetimeIndex(trading_days).sort_values()
    df = pd.DataFrame(index=idx)
    iso = df.index.isocalendar()
    tmp = df.reset_index(names="dt")
    tmp["y"] = iso.year.to_numpy()
    tmp["w"] = iso.week.to_numpy()
    sched = tmp.groupby(["y","w"], sort=True)["dt"].min().sort_values().tolist()
    return [pd.Timestamp(d).normalize() for d in sched]


def asof_map_schedule_to_pred(sched_dates: List[pd.Timestamp],
                             pred_days: pd.DatetimeIndex) -> Dict[pd.Timestamp, pd.Timestamp]:
    """映射调仓日到预测日"""
    pred_days = pd.DatetimeIndex(pred_days).sort_values()
    mapping = {}
    for d in sorted(pd.DatetimeIndex(sched_dates)):
        pos = pred_days.searchsorted(d, side="right") - 1
        if pos >= 0:
            mapping[pd.Timestamp(d).normalize()] = pd.Timestamp(pred_days[pos]).normalize()
    return mapping  # {schedule_day -> pred_day}


def load_qlib_ohlcv(symbols: List[str], start: str, end: str, qlib_dir: str) -> Dict[str, pd.DataFrame]:
    """加载QLib OHLCV数据"""
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


def build_exposures_map(features_path: str, universe: Set[str], dates: List[pd.Timestamp], 
                       use_items: List[str]) -> Dict[pd.Timestamp, pd.DataFrame]:
    """构建暴露度映射"""
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
    
    # 应用as-of回填，确保所有执行日都有映射数据
    return _asof_fill_map(out, dates)


def compute_vol_adv_maps(features_path: str, universe: Set[str], dates: List[pd.Timestamp],
                        halflife: int = 20) -> Tuple[Dict[pd.Timestamp, pd.DataFrame], Dict[pd.Timestamp, pd.DataFrame], Dict[pd.Timestamp, pd.DataFrame]]:
    """
    计算波动率和ADV映射
    返回:
      vol_map[date]: DataFrame[instrument, sigma]  (日波动率, EWM std, 以 date-1 为止)
      adv_map[date]: DataFrame[instrument, adv_dollar] (≈ ADV20 * vwap, 以 date-1 为止)
    """
    need = ["instrument","datetime","ret_1","adv_20","$vwap","ln_dollar_vol_20"]
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
    vol_map: Dict[pd.Timestamp, pd.DataFrame] = {}
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
    adv_map: Dict[pd.Timestamp, pd.DataFrame] = {}
    if "$vwap" in df.columns and "adv_20" in df.columns and not df.empty:
        df["adv_dollar"] = (df["adv_20"].astype(float).shift(1) * df["$vwap"].astype(float).shift(1)).clip(lower=0.0)
        adv_map = {pd.Timestamp(d).normalize(): g[["instrument","adv_dollar"]].dropna().reset_index(drop=True)
                   for d, g in df.groupby("datetime") if pd.Timestamp(d).normalize() in set(dates)}

    # Liquidity buckets by day (0..4) based on ln_dollar_vol_20 cross-sectional quintiles
    liq_map: Dict[pd.Timestamp, pd.DataFrame] = {}
    if "ln_dollar_vol_20" in df.columns and not df.empty:
        for d, g in df.groupby("datetime"):
            dt_norm = pd.Timestamp(d).normalize()
            if dt_norm not in set(dates):
                continue
            s = g[["instrument","ln_dollar_vol_20"]].dropna()
            if s.empty or len(s) < 10:
                continue
            try:
                q = pd.qcut(s["ln_dollar_vol_20"], q=5, labels=False, duplicates='drop').astype("Int64")
            except ValueError:
                q = pd.cut(s["ln_dollar_vol_20"], bins=5, labels=False, duplicates='drop').astype("Int64")
            liq_map[dt_norm] = pd.DataFrame({
                "instrument": s["instrument"].astype(str).str.upper().values,
                "liq_bucket": q.values,
            })

    # 对每个映射应用as-of回填，确保所有执行日都有映射数据
    vol_map_filled = _asof_fill_map(vol_map, dates)
    adv_map_filled = _asof_fill_map(adv_map, dates)
    liq_map_filled = _asof_fill_map(liq_map, dates)
    
    return vol_map_filled, adv_map_filled, liq_map_filled


def apply_adv_limit(prev_w: Dict[str, float], tgt_w: Dict[str, float],
                   adv_df: pd.DataFrame | None, port_value: float, adv_limit_pct: float) -> Tuple[Dict[str, float], Dict]:
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


def _asof_fill_map(target_map: Dict[pd.Timestamp, pd.DataFrame], 
                  exec_dates: List[pd.Timestamp]) -> Dict[pd.Timestamp, pd.DataFrame]:
    """
    对缺失的执行日进行as-of回填
    
    参数:
        target_map: 原始映射字典 {date: DataFrame}
        exec_dates: 所有需要映射的执行日列表
        
    返回:
        填充后的映射字典，确保所有执行日都有对应的映射数据
    """
    if not target_map:
        return {}
    
    # 获取所有有映射数据的日期并排序
    keys_sorted = pd.DatetimeIndex(sorted(target_map.keys()))
    if keys_sorted.empty:
        return {}
    
    filled_map = {}
    
    for exec_date in exec_dates:
        exec_date_norm = pd.Timestamp(exec_date).normalize()
        if exec_date_norm in target_map:
            filled_map[exec_date_norm] = target_map[exec_date_norm]
        else:
            # 找到之前最近的有效日
            pos = keys_sorted.searchsorted(exec_date_norm, side='right') - 1
            if pos >= 0:
                nearest_date = keys_sorted[pos]
                filled_map[exec_date_norm] = target_map[nearest_date]
            # 如果没有之前的有效日，则跳过（保持为空）
    
    return filled_map


def _compute_short_timing_dates(anchor_df: pd.DataFrame,
                               exec_dates: List[pd.Timestamp],
                               lookback: int = 63,
                               thr: float = 0.0) -> Set[pd.Timestamp]:
    """
    使用锚标的收盘价计算动量：mom = close / close.shift(lookback) - 1
    采用 T-1 信息：signal_asof = (mom <= thr).shift(1)
    在 signal_asof 为 True 的执行日**允许做空**。
    """
    # 稳健性增强 #2：历史不足或无效则"全允许做空"，避免误关短腿
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
