# strategies/stock_selection.py
# -*- coding: utf-8 -*-
"""
选股（membership）模块：
- 支持 rank 优先（auto 模式下，当 rank 列存在且非空样本覆盖 top_k+short_k）
- 支持 score 降序
- 支持 membership_buffer：进入/退出缓冲，降低换手
- 保留上期在“保留区”的标的（long/short 各自）
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict


def _should_use_rank(g: pd.DataFrame, top_k: int, short_k: int) -> bool:
    if "rank" not in g.columns:
        return False
    ranks = pd.to_numeric(g["rank"], errors="coerce")
    need = max(1, int(top_k) + int(short_k))
    return ranks.notna().sum() >= need


def select_members_with_buffer(
    g: pd.DataFrame,
    prev_weights: Dict[str, float],
    top_k: int,
    short_k: int,
    membership_buffer: float = 0.2,
    use_rank_mode: str = "auto",   # "auto" | "rank" | "score"
    reb_counter: int | None = None,
    verbose: bool = False,
) -> Tuple[List[str], List[str], pd.DataFrame]:
    """
    返回: (longs, shorts, g_sorted)
    g_sorted: 若按 rank 则按 rank 升序；若按 score 则按 score 降序
    """
    buf = float(membership_buffer or 0.0)
    g = g.copy()

    mode = (use_rank_mode or "auto").strip().lower()
    if mode == "rank":
        use_rank = ("rank" in g.columns)
    elif mode == "score":
        use_rank = False
    else:
        use_rank = _should_use_rank(g, top_k, short_k)

    if use_rank:
        g["rank"] = pd.to_numeric(g["rank"], errors="coerce")
        g = g.sort_values("rank", na_position="last", kind="mergesort")

        # longs
        enter_long_thr = int(top_k)
        exit_long_thr  = int(np.ceil(top_k * (1.0 + buf)))
        longs_enter = set(g.head(max(0, enter_long_thr))["instrument"])
        longs_keep  = {ins for ins, w in (prev_weights or {}).items() if w > 0}
        longs_ok    = set(g[g["rank"] <= exit_long_thr]["instrument"])
        longs = list((longs_enter | (longs_keep & longs_ok)) if top_k > 0 else set())

        # shorts：从尾部
        g_tail = g.iloc[::-1].copy()
        enter_short_thr = int(short_k)
        exit_short_thr  = int(np.ceil(short_k * (1.0 + buf)))
        shorts_enter = set(g_tail.head(max(0, enter_short_thr))["instrument"])
        shorts_keep  = {ins for ins, w in (prev_weights or {}).items() if w < 0}
        # 用正序 rank 的阈值界定“保留区”，但选尾部进入
        shorts_ok = set(g[g["rank"] <= exit_short_thr].iloc[::-1]["instrument"])
        shorts = list((shorts_enter | (shorts_keep & shorts_ok)) if short_k > 0 else set())

        g_sorted = g

    else:
        # 按 score 降序
        g = g.dropna(subset=["score"]).sort_values("score", ascending=False)

        # longs
        enter_long_thr = int(top_k)
        exit_long_thr  = int(np.ceil(top_k * (1.0 + buf)))
        longs_enter    = set(g.head(max(0, enter_long_thr))["instrument"])
        longs_exit_zone= set(g.head(max(0, exit_long_thr))["instrument"])
        longs_keep     = {ins for ins, w in (prev_weights or {}).items() if w > 0}
        longs = list((longs_enter | (longs_keep & longs_exit_zone)) if top_k > 0 else set())

        # shorts：从尾部
        g_rev = g.iloc[::-1]
        enter_short_thr = int(short_k)
        exit_short_thr  = int(np.ceil(short_k * (1.0 + buf)))
        shorts_enter    = set(g_rev.head(max(0, enter_short_thr))["instrument"])
        shorts_exit_zone= set(g_rev.head(max(0, exit_short_thr))["instrument"])
        shorts_keep     = {ins for ins, w in (prev_weights or {}).items() if w < 0}
        shorts = list((shorts_enter | (shorts_keep & shorts_exit_zone)) if short_k > 0 else set())

        g_sorted = g

    if (reb_counter is not None) and (reb_counter < 10) and verbose:
        print(f"[rebalance {reb_counter}] longs={len(longs)} shorts={len(shorts)} (candidates={len(g_sorted)})")

    return longs, shorts, g_sorted
