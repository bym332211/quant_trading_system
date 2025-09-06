# strategies/entry_strategy.py
# -*- coding: utf-8 -*-
"""
入场策略模块 - 将入场逻辑从主策略中剥离出来

包含：
- 权重生成（ICDF 和等权重）
- 中性化处理
- 平滑处理
- 目标波动率缩放
- 硬上限处理
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from strategies.stock_selection import select_members_with_buffer


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


def neutralize_weights(targets: dict[str, float], expos_df: pd.DataFrame,
                       ridge_lambda: float = 1e-6, drop_dummy: bool = True,
                       keep_cols: list[str] | None = None) -> dict[str, float]:
    """中性化权重处理"""
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


class EntryStrategy:
    """入场策略类 - 处理所有入场相关的逻辑"""
    
    def __init__(self, 
                 neutralize_items: Tuple[str, ...] = (),
                 ridge_lambda: float = 1e-6,
                 top_k: int = 50,
                 short_k: int = 50,
                 membership_buffer: float = 0.2,
                 selection_use_rank_mode: str = "auto",
                 long_exposure: float = 1.0,
                 short_exposure: float = -1.0,
                 max_pos_per_name: float = 0.05,
                 weight_scheme: str = "equal",
                 smooth_eta: float = 0.6,
                 target_vol: float = 0.0,
                 leverage_cap: float = 2.0,
                 hard_cap: bool = False,
                 verbose: bool = False):
        self.neutralize_items = neutralize_items
        self.ridge_lambda = ridge_lambda
        self.top_k = top_k
        self.short_k = short_k
        self.membership_buffer = membership_buffer
        self.selection_use_rank_mode = selection_use_rank_mode
        self.long_exposure = long_exposure
        self.short_exposure = short_exposure
        self.max_pos_per_name = max_pos_per_name
        self.weight_scheme = weight_scheme
        self.smooth_eta = smooth_eta
        self.target_vol = target_vol
        self.leverage_cap = leverage_cap
        self.hard_cap = hard_cap
        self.verbose = verbose
        
        # 状态变量
        self.prev_weights: Dict[str, float] = {}
        self.reb_counter = 0
    
    def generate_entry_weights(self,
                              g: pd.DataFrame,
                              prev_weights: Dict[str, float],
                              expos_df: Optional[pd.DataFrame] = None,
                              vol_df: Optional[pd.DataFrame] = None,
                              allow_shorts: bool = True,
                              reb_counter: Optional[int] = None) -> Dict[str, float]:
        """
        生成入场权重
        
        参数:
            g: 预测数据DataFrame
            prev_weights: 上一期权重
            expos_df: 暴露数据DataFrame
            vol_df: 波动率数据DataFrame
            allow_shorts: 是否允许做空
            reb_counter: 调仓计数器
            
        返回:
            目标权重字典
        """
        if reb_counter is not None:
            self.reb_counter = reb_counter
        
        # 1. 选股
        longs, shorts, g_sorted = select_members_with_buffer(
            g=g,
            prev_weights=prev_weights,
            top_k=self.top_k,
            short_k=self.short_k,
            membership_buffer=self.membership_buffer,
            use_rank_mode=self.selection_use_rank_mode,
            reb_counter=self.reb_counter,
            verbose=self.verbose,
        )
        
        # 如果不允许做空，清空short候选
        if not allow_shorts:
            shorts = []
        
        # 2. 权重生成（raw）
        tgt = self._generate_raw_weights(longs, shorts, g_sorted)
        
        # 3. 中性化
        tgt = self._apply_neutralization(tgt, expos_df)
        
        # 4. 平滑
        tgt = self._apply_smoothing(tgt, prev_weights)
        
        # 5. 目标波动率缩放
        tgt = self._apply_volatility_scaling(tgt, vol_df)
        
        # 6. 硬上限处理
        tgt = self._apply_hard_cap(tgt, allow_shorts)
        
        # 更新状态
        self.prev_weights = tgt.copy()
        self.reb_counter += 1
        
        return tgt
    
    def _generate_raw_weights(self, longs: List[str], shorts: List[str], g_sorted: pd.DataFrame) -> Dict[str, float]:
        """生成原始权重"""
        tgt = {}
        
        if self.weight_scheme == "icdf":
            if longs:
                dfL = g_sorted[g_sorted["instrument"].isin(longs)].copy()
                rankL = (np.arange(1, len(dfL)+1)).astype(float)
                wL = icdf_weights(rankL)
                wL = np.maximum(wL, 0.0)
                if wL.sum() > 0:
                    wL = wL / wL.sum() * max(0.0, float(self.long_exposure))
                for ins, w in zip(dfL["instrument"], wL):
                    tgt[ins] = tgt.get(ins, 0.0) + float(w)
            if shorts:
                dfS = g_sorted[g_sorted["instrument"].isin(shorts)].copy().iloc[::-1]
                rankS = (np.arange(1, len(dfS)+1)).astype(float)
                wS = -np.maximum(icdf_weights(rankS), 0.0)
                if -wS.sum() > 0 and self.short_exposure < 0:
                    wS = wS / (-wS.sum()) * float(self.short_exposure)
                for ins, w in zip(dfS["instrument"], wS):
                    tgt[ins] = tgt.get(ins, 0.0) + float(w)
        else:
            if len(longs) > 0 and self.long_exposure != 0.0:
                w = float(self.long_exposure) / float(len(longs))
                for s in longs: tgt[s] = tgt.get(s, 0.0) + w
            if len(shorts) > 0 and self.short_exposure != 0.0:
                w = float(self.short_exposure) / float(len(shorts))
                for s in shorts: tgt[s] = tgt.get(s, 0.0) + w
        
        return tgt
    
    def _apply_neutralization(self, tgt: Dict[str, float], expos_df: Optional[pd.DataFrame]) -> Dict[str, float]:
        """应用中性化处理"""
        keep_cols = []
        if "beta" in self.neutralize_items: keep_cols.append("mkt_beta_60")
        if "size" in self.neutralize_items: keep_cols.append("ln_dollar_vol_20")
        if expos_df is not None:
            if "sector" in self.neutralize_items: keep_cols += [c for c in expos_df.columns if c.startswith("ind_")]
            if "liq"    in self.neutralize_items: keep_cols += [c for c in expos_df.columns if c.startswith("liq_bucket_")]
        
        return neutralize_weights(tgt, expos_df, ridge_lambda=self.ridge_lambda, drop_dummy=True, keep_cols=keep_cols)
    
    def _apply_smoothing(self, tgt: Dict[str, float], prev_weights: Dict[str, float]) -> Dict[str, float]:
        """应用平滑处理"""
        eta = float(self.smooth_eta or 0.0)
        if eta > 0 and prev_weights:
            w_prev_ser = pd.Series(prev_weights, dtype=float)
            w_tgt_ser  = pd.Series(tgt, dtype=float)
            all_idx = sorted(set(w_prev_ser.index) | set(w_tgt_ser.index))
            w_prev_ser = w_prev_ser.reindex(all_idx).fillna(0.0)
            w_tgt_ser  = w_tgt_ser.reindex(all_idx).fillna(0.0)
            w_new = eta * w_prev_ser + (1.0 - eta) * w_tgt_ser
            return {k: float(v) for k, v in w_new.items()}
        return tgt
    
    def _apply_volatility_scaling(self, tgt: Dict[str, float], vol_df: Optional[pd.DataFrame]) -> Dict[str, float]:
        """应用波动率缩放"""
        tv = float(self.target_vol or 0.0)
        if tv > 0 and vol_df is not None and not vol_df.empty:
            sigmap = dict(zip(vol_df["instrument"], vol_df["sigma"]))
            w = pd.Series(tgt, dtype=float)
            s2 = np.array([ (sigmap.get(k, np.nan) or np.nan)**2 * (w.get(k,0.0)**2) for k in w.index ])
            s2 = s2[~np.isnan(s2)]
            if s2.size > 0 and np.nansum(s2) > 0:
                est_ann = float(np.sqrt(np.nansum(s2)) * np.sqrt(252.0))
                if est_ann > 1e-8:
                    scale = min(float(self.leverage_cap or 10.0), tv / est_ann)
                    w = (w * scale)
                    return {k: float(v) for k, v in w.clip(-1.0, 1.0).items()}
        return tgt
    
    def _apply_hard_cap(self, tgt: Dict[str, float], allow_shorts: bool) -> Dict[str, float]:
        """应用硬上限处理"""
        if self.hard_cap and (self.max_pos_per_name is not None) and self.max_pos_per_name > 0:
            return waterfill_two_legs(
                tgt,
                long_exposure=self.long_exposure,
                short_exposure=self.short_exposure,
                max_pos_per_name=float(self.max_pos_per_name),
                allow_shorts=allow_shorts
            )
        else:
            # 软上限回退：分别归一正负腿并裁剪
            w = pd.Series(tgt, dtype=float)
            w_pos = w[w>0]; w_neg = -w[w<0]
            if w_pos.sum() > 0:
                w_pos = w_pos / w_pos.sum() * max(0.0, float(self.long_exposure))
            if w_neg.sum() > 0 and allow_shorts:
                w_neg = w_neg / w_neg.sum() * max(0.0, float(-self.short_exposure))
            else:
                w_neg = w_neg * 0.0
            w2 = pd.concat([w_pos, -w_neg]).reindex(w.index).fillna(0.0)
            cap = float(self.max_pos_per_name or 0.0)
            if cap > 0:
                w2 = w2.clip(-cap, cap)
            return {k: float(v) for k,v in w2.items()}
