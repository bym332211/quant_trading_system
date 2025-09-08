"""
XSecRebalance策略专用的数据准备模块
包含XSecRebalance特有的数据准备逻辑
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Set, Tuple
import pandas as pd
import numpy as np

from backtest.engine.data_loader import (
    ensure_inst_dt, weekly_schedule, asof_map_schedule_to_pred,
    load_qlib_ohlcv, build_exposures_map, compute_vol_adv_maps,
    _compute_short_timing_dates
)


def prepare_predictions(config: Dict[str, Any]) -> Tuple[Dict[pd.Timestamp, pd.DataFrame], pd.DataFrame]:
    """准备预测数据 - XSecRebalance特有"""
    args = config["args"]
    
    # 预测数据
    preds_all = ensure_inst_dt(pd.read_parquet(Path(args["preds"]).expanduser().resolve()))
    mask = (preds_all["datetime"] >= pd.Timestamp(args["start"])) & (preds_all["datetime"] <= pd.Timestamp(args["end"]))
    preds_all = preds_all.loc[mask].copy()
    if preds_all.empty:
        raise RuntimeError("预测为空。")
    pred_days = pd.DatetimeIndex(preds_all["datetime"].unique()).sort_values()

    # 聚合预测：源日 -> 截面
    preds_all_copy = preds_all.copy()
    keep_cols = ["instrument","score"] + (["rank"] if "rank" in preds_all_copy.columns else [])
    preds_all_copy["dt_norm"] = preds_all_copy["datetime"].dt.normalize()
    preds_by_src = {d: g[keep_cols].copy() for d, g in preds_all_copy.groupby("dt_norm")}

    # ---- 关键修复：下游需要在 preds_all 上使用 dt_norm，这里同步补上一列（最小改动） ----
    preds_all["dt_norm"] = preds_all["datetime"].dt.normalize()

    return preds_by_src, preds_all


def prepare_price_data(config: Dict[str, Any], universe: Set[str]) -> Dict[str, pd.DataFrame]:
    """准备价格数据 - XSecRebalance特有"""
    args = config["args"]
    
    # 行情数据
    price_map = load_qlib_ohlcv(sorted(list(universe)), start=args["start"], end=args["end"], qlib_dir=args["qlib_dir"])
    
    # 如果有锚定符号，放在第一位
    anchor_sym = args["anchor_symbol"].upper() if args.get("anchor_symbol") else None
    if anchor_sym and anchor_sym in price_map:
        anchor_days = pd.DatetimeIndex(price_map[anchor_sym].index)
        anchor_first = {anchor_sym: price_map.pop(anchor_sym)}
        price_map = {**anchor_first, **price_map}
    
    return price_map


def prepare_execution_dates(config: Dict[str, Any], price_map: Dict[str, pd.DataFrame], 
                          preds_by_src: Dict[pd.Timestamp, pd.DataFrame]) -> Tuple[Dict[pd.Timestamp, pd.Timestamp], Set[pd.Timestamp]]:
    """准备执行日期映射 - XSecRebalance特有"""
    args = config["args"]
    
    # 获取锚定日期
    anchor_sym = args["anchor_symbol"].upper() if args.get("anchor_symbol") else None
    if anchor_sym and anchor_sym in price_map:
        anchor_days = pd.DatetimeIndex(price_map[anchor_sym].index)
    else:
        anchor_days = pd.DatetimeIndex(next(iter(price_map.values())).index)

    # 周频锚定 -> 源预测日 as-of 映射
    sched_anchor = weekly_schedule(anchor_days)
    pred_days = pd.DatetimeIndex(preds_by_src.keys()).sort_values()
    sched2pred = asof_map_schedule_to_pred(sched_anchor, pred_days)
    if not sched2pred:
        raise RuntimeError("调仓日与预测日无法 as-of 映射（窗口内没有预测）")

    # exec_lag 推进到"执行日"
    exec_dates = []
    anchor_list = sorted(pd.DatetimeIndex(anchor_days).tolist())
    pos_map = {d: i for i, d in enumerate(anchor_list)}
    for sd in sched2pred.keys():
        i = pos_map.get(sd)
        if i is None:
            continue
        j = i + max(0, int(args["exec_lag"]))
        if j < len(anchor_list):
            exec_dates.append(anchor_list[j])
    exec_dates = sorted(pd.DatetimeIndex(exec_dates).unique().tolist())
    if not exec_dates:
        raise RuntimeError("exec_dates 为空（检查 exec_lag 与日期范围）")

    # exec 日 -> 源预测日
    exec2pred_src = {}
    for sd, ps in sched2pred.items():
        i = pos_map.get(sd)
        if i is None: continue
        j = i + max(0, int(args["exec_lag"]))
        if j < len(anchor_list):
            exec2pred_src[anchor_list[j]] = ps

    return exec2pred_src, set(exec_dates)


def prepare_exposures(config: Dict[str, Any], universe: Set[str], exec_dates: Set[pd.Timestamp]) -> Dict[pd.Timestamp, pd.DataFrame]:
    """准备暴露数据 - XSecRebalance特有"""
    args = config["args"]
    
    neutral_list = [s.strip().lower() for s in args["neutralize"].split(",") if s.strip()]
    expos_map = build_exposures_map(args["features_path"], universe=universe,
                                    dates=list(exec_dates), use_items=neutral_list)
    return expos_map


def prepare_vol_adv(config: Dict[str, Any], universe: Set[str], exec_dates: Set[pd.Timestamp]) -> Tuple[Dict[pd.Timestamp, pd.DataFrame], Dict[pd.Timestamp, pd.DataFrame]]:
    """准备波动率和ADV数据 - XSecRebalance特有"""
    args = config["args"]
    
    vol_map, adv_map = compute_vol_adv_maps(args["features_path"], universe=universe,
                                            dates=list(exec_dates), halflife=int(args["ewm_halflife"]))
    return vol_map, adv_map


def prepare_short_timing(config: Dict[str, Any], price_map: Dict[str, pd.DataFrame], exec_dates: Set[pd.Timestamp]) -> Set[pd.Timestamp]:
    """准备短腿择时数据 - XSecRebalance特有"""
    args = config["args"]
    
    short_allow_dates: Set[pd.Timestamp] = set()
    anchor_sym = args["anchor_symbol"].upper() if args.get("anchor_symbol") else None
    if args.get("short_timing_mom63") and anchor_sym and anchor_sym in price_map:
        short_allow_dates = _compute_short_timing_dates(
            price_map[anchor_sym],
            exec_dates=list(exec_dates),
            lookback=int(args["short_timing_lookback"]),
            thr=float(args["short_timing_threshold"])
        )
    
    return short_allow_dates


def get_universe_from_predictions(preds_all: pd.DataFrame, anchor_sym: str = None) -> Set[str]:
    """从预测数据获取universe - XSecRebalance特有"""
    universe_all = set(preds_all["instrument"].astype(str).str.upper().unique().tolist())
    if anchor_sym:
        universe_all.add(anchor_sym.upper())
    return universe_all


def filter_universe_by_mapped_predictions(preds_all: pd.DataFrame, mapped_src_days: Set[pd.Timestamp], 
                                        anchor_sym: str = None) -> Set[str]:
    """根据映射的预测日过滤universe - XSecRebalance特有"""
    final_universe = set(preds_all[preds_all["dt_norm"].isin(mapped_src_days)]["instrument"].astype(str).str.upper().unique().tolist())
    if anchor_sym:
        final_universe.add(anchor_sym.upper())
    return final_universe
