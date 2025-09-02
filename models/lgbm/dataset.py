#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
models/lgbm/dataset.py

用途
- 读取 features parquet（由 data/build_factors.py 生成）
- 逐日横截面预处理：winsor → zscore（可选）
- 训练期标签中性化（y ~ exposures，取残差）
- 构建训练集（X, y）与推理用单日截面（X_t）
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union
import warnings
import numpy as np
import pandas as pd


# -----------------------------
# 配置结构
# -----------------------------
@dataclass
class NeutralizeConfig:
    enable: bool = True
    on: str = "label"  # "label" | "features" | "scores"
    exposures: Sequence[str] = ()  # 支持通配符，如 ["mkt_beta_60","ln_mktcap","ind_*"]
    ridge_lambda: float = 1e-8
    center_exposures: bool = True
    drop_dummy: bool = True  # 对 one-hot 组自动 drop 1 列


@dataclass
class PreprocessConfig:
    winsor: float = 0.01         # 逐日截面 winsor 百分位；0 关闭
    zscore: bool = True          # 逐日截面标准化
    neutralize: NeutralizeConfig = field(default_factory=NeutralizeConfig)  # <-- 修正


@dataclass
class DataConfig:
    features_path: str = "artifacts/features_day.parquet"
    freq: str = "day"
    label: str = "y_fwd_5"
    features: Sequence[str] = (
        "mom_20", "vol_20", "vwap_spread", "ma20_gap", "logv_zn_20", "rng_hl", "oc"
    )
    # 过滤相关（可选）
    min_price: float = 0.5              # 最低价格过滤（按 $close 或 close）
    min_adv_usd: float = 1e6            # ADV_$ = adv_20 * $vwap 的美元阈值
    min_names_per_day: int = 50         # 每日至少多少只股票保留
    start: str = "2010-01-01"
    end: str = "2100-01-01"


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)                 # <-- 修正
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)  # <-- 修正


# -----------------------------
# 工具函数
# -----------------------------
def _get_close_col(df: pd.DataFrame) -> str:
    if "$close" in df.columns:
        return "$close"
    if "close" in df.columns:
        return "close"
    raise KeyError("未找到 $close/close 列。请检查 features parquet。")

def _expand_wildcards(all_cols: Sequence[str], patterns: Sequence[str]) -> List[str]:
    cols = []
    s = set(all_cols)
    for p in patterns:
        if "*" in p or "?" in p:
            import fnmatch
            matched = [c for c in all_cols if fnmatch.fnmatch(c, p)]
            cols.extend(matched)
        else:
            if p in s:
                cols.append(p)
    # 去重但保持顺序
    seen = set()
    out = []
    for c in cols:
        if c not in seen:
            out.append(c); seen.add(c)
    return out

def _groupby_apply(df: pd.DataFrame, key: str, func):
    """兼容 pandas <2.1 / >=2.1 的 groupby.apply include_groups 参数差异"""
    gb = df.groupby(key, group_keys=False)
    try:
        return gb.apply(func, include_groups=False)
    except TypeError:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            return gb.apply(func)


# -----------------------------
# 横截面处理器
# -----------------------------
class CrossSectionProcessor:
    def __init__(self, winsor: float = 0.01, zscore: bool = True):
        self.winsor = float(winsor) if winsor else 0.0
        self.do_z = bool(zscore)

    @staticmethod
    def _winsorize_vec(x: pd.Series, p: float) -> pd.Series:
        if x.size == 0 or not np.isfinite(x).any():
            return x
        lo, hi = x.quantile(p), x.quantile(1 - p)
        return x.clip(lo, hi)

    def winsorize(self, df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
        if not self.winsor or self.winsor <= 0:
            return df
        cols = [c for c in cols if c in df.columns]
        if not cols:
            return df

        def _f(g: pd.DataFrame) -> pd.DataFrame:
            gg = g.copy()
            for c in cols:
                s = gg[c].astype(float)
                gg[c] = self._winsorize_vec(s, self.winsor)
            return gg

        return _groupby_apply(df, "datetime", _f)

    def zscore(self, df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
        if not self.do_z:
            return df
        cols = [c for c in cols if c in df.columns]
        if not cols:
            return df

        def _f(g: pd.DataFrame) -> pd.DataFrame:
            gg = g.copy()
            for c in cols:
                s = gg[c].astype(float)
                mu = s.mean()
                sd = s.std(ddof=0)
                if not np.isfinite(sd) or sd == 0:
                    gg[c] = 0.0
                else:
                    gg[c] = (s - mu) / sd
            return gg

        return _groupby_apply(df, "datetime", _f)

    @staticmethod
    def _auto_drop_dummy(expos_cols: List[str]) -> List[str]:
        """
        简单按前缀归组并每组 drop 最后一列，避免 dummy trap
        """
        prefixes = ["ind_", "size_bucket_", "liq_bucket_"]
        keep = expos_cols.copy()
        for p in prefixes:
            grp = [c for c in expos_cols if c.startswith(p)]
            if len(grp) >= 2:
                drop = grp[-1]
                if drop in keep:
                    keep.remove(drop)
        return keep

    @staticmethod
    def _ridge_residual(y: np.ndarray, E: np.ndarray, lam: float) -> np.ndarray:
        # y: (n,), E: (n,k)
        # beta = (E'E + lam I)^-1 E'y ; resid = y - E beta
        if E.size == 0:
            return y.copy()
        k = E.shape[1]
        XtX = E.T @ E
        if lam > 0:
            XtX = XtX + lam * np.eye(k)
        try:
            beta = np.linalg.solve(XtX, E.T @ y)
        except np.linalg.LinAlgError:
            beta = np.linalg.pinv(XtX) @ (E.T @ y)
        return y - E @ beta

    def neutralize(
        self,
        df: pd.DataFrame,
        target_col: str,
        exposures: Sequence[str],
        ridge_lambda: float = 1e-8,
        center_exposures: bool = True,
        drop_dummy: bool = True,
    ) -> pd.DataFrame:
        """
        逐日：target ~ exposures 取残差；返回 DataFrame（target_col 被替换为 resid）
        """
        expos_cols = [c for c in exposures if c in df.columns]
        if not expos_cols or target_col not in df.columns:
            return df

        if drop_dummy:
            expos_cols = self._auto_drop_dummy(expos_cols)

        def _f(g: pd.DataFrame) -> pd.DataFrame:
            gg = g.copy()
            y = gg[target_col].astype(float).to_numpy()
            X = gg[expos_cols].astype(float).to_numpy()

            # 丢零方差/全NA列
            keep_idx = []
            for j in range(X.shape[1]):
                col = X[:, j]
                finite = np.isfinite(col)
                if finite.sum() < 3:
                    continue
                v = np.nanvar(col[finite])
                if not np.isfinite(v) or v == 0:
                    continue
                keep_idx.append(j)
            if not keep_idx:
                return gg
            X = X[:, keep_idx]

            # 缺失填当日均值；可选中心化
            for j in range(X.shape[1]):
                col = X[:, j]
                if np.any(~np.isfinite(col)):
                    finite = np.isfinite(col)
                    mu = np.nanmean(col[finite]) if finite.any() else 0.0
                    col[~finite] = mu
                    X[:, j] = col
            if center_exposures:
                mu = X.mean(axis=0, keepdims=True)
                X = X - mu

            # 加常数项
            ones = np.ones((X.shape[0], 1), dtype=float)
            Xc = np.concatenate([ones, X], axis=1)

            yy = y.copy()
            mask = np.isfinite(yy)
            if mask.sum() < 3:
                return gg

            resid = np.full_like(yy, fill_value=np.nan, dtype=float)
            resid[mask] = self._ridge_residual(yy[mask], Xc[mask], ridge_lambda)
            gg[target_col] = resid
            return gg

        return _groupby_apply(df, "datetime", _f)


# -----------------------------
# 数据集构建器
# -----------------------------
class DatasetBuilder:
    def __init__(self, cfg: Union[Config, Dict]):
        # 允许直接传 dict
        if isinstance(cfg, dict):
            dc = cfg.get("data", {})
            pc = cfg.get("preprocess", {})
            nc = (pc.get("neutralize", {}) if isinstance(pc, dict) else {})
            self.cfg = Config(
                data=DataConfig(
                    features_path=dc.get("features_path", DataConfig.features_path),
                    freq=dc.get("freq", DataConfig.freq),
                    label=dc.get("label", DataConfig.label),
                    features=tuple(dc.get("features", DataConfig.features)),
                    min_price=dc.get("min_price", DataConfig.min_price),
                    min_adv_usd=dc.get("min_adv_usd", DataConfig.min_adv_usd),
                    min_names_per_day=dc.get("min_names_per_day", DataConfig.min_names_per_day),
                    start=dc.get("start", DataConfig.start),
                    end=dc.get("end", DataConfig.end),
                ),
                preprocess=PreprocessConfig(
                    winsor=pc.get("winsor", PreprocessConfig.winsor),
                    zscore=pc.get("zscore", PreprocessConfig.zscore),
                    neutralize=NeutralizeConfig(
                        enable=nc.get("enable", True),
                        on=nc.get("on", "label"),
                        exposures=tuple(nc.get("exposures", ())),
                        ridge_lambda=nc.get("ridge_lambda", 1e-8),
                        center_exposures=nc.get("center_exposures", True),
                        drop_dummy=nc.get("drop_dummy", True),
                    ),
                ),
            )
        else:
            self.cfg = cfg

        self.cs = CrossSectionProcessor(
            winsor=self.cfg.preprocess.winsor,
            zscore=self.cfg.preprocess.zscore,
        )

    # ---------- 数据载入 & 过滤 ----------
    def load_features(self, path: Optional[str] = None) -> pd.DataFrame:
        path = path or self.cfg.data.features_path
        df = pd.read_parquet(path)
        required = {"instrument", "datetime"}
        if not required <= set(df.columns):
            raise ValueError(f"features 缺列：{required - set(df.columns)}")
        df["instrument"] = df["instrument"].astype(str).str.upper()
        df["datetime"] = pd.to_datetime(df["datetime"], utc=False)

        # 时间裁剪
        mask = (df["datetime"] >= pd.Timestamp(self.cfg.data.start)) & \
               (df["datetime"] <= pd.Timestamp(self.cfg.data.end))
        df = df.loc[mask].copy()

        # 可交易过滤
        close_col = _get_close_col(df)
        if self.cfg.data.min_price is not None:
            df = df[df[close_col] >= float(self.cfg.data.min_price)]
        vwap_col = "$vwap" if "$vwap" in df.columns else ("vwap" if "vwap" in df.columns else None)
        if vwap_col is not None and "adv_20" in df.columns and self.cfg.data.min_adv_usd is not None:
            adv_usd = (df["adv_20"].astype(float) * df[vwap_col].astype(float))
            df = df[adv_usd >= float(self.cfg.data.min_adv_usd)]

        if self.cfg.data.min_names_per_day and self.cfg.data.min_names_per_day > 0:
            ct = df.groupby("datetime")["instrument"].nunique()
            keep_dt = ct[ct >= int(self.cfg.data.min_names_per_day)].index
            df = df[df["datetime"].isin(keep_dt)]

        return df.sort_values(["datetime", "instrument"]).reset_index(drop=True)

    # ---------- 训练集构建 ----------
    def build_train_xy(
        self,
        df: pd.DataFrame,
        start: Optional[str] = None,
        end: Optional[str] = None,
        use_resid_label: bool = True,
        feature_cols: Optional[Sequence[str]] = None,
        drop_na: bool = True,
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        start = start or self.cfg.data.start
        end = end or self.cfg.data.end
        label = self.cfg.data.label
        if label not in df.columns:
            raise KeyError(f"label 列不存在：{label}")

        m = (df["datetime"] >= pd.Timestamp(start)) & (df["datetime"] <= pd.Timestamp(end))
        d = df.loc[m].copy()

        feats_all = list(d.columns)
        feats = feature_cols or self.cfg.data.features
        feats = list(_expand_wildcards(feats_all, feats))

        d = self.cs.winsorize(d, feats)
        d = self.cs.zscore(d, feats)

        if use_resid_label and self.cfg.preprocess.neutralize.enable and self.cfg.preprocess.neutralize.on == "label":
            expos = list(_expand_wildcards(feats_all, self.cfg.preprocess.neutralize.exposures))
            if expos:
                d = self.cs.neutralize(
                    d, target_col=label, exposures=expos,
                    ridge_lambda=self.cfg.preprocess.neutralize.ridge_lambda,
                    center_exposures=self.cfg.preprocess.neutralize.center_exposures,
                    drop_dummy=self.cfg.preprocess.neutralize.drop_dummy,
                )

        cols_need = feats + [label, "instrument", "datetime"]
        d = d[cols_need]
        if drop_na:
            d = d.dropna(subset=feats + [label])

        panel = d[["instrument", "datetime"]].copy()
        X = d[feats].astype(float)
        y = d[label].astype(float)
        return X, y, panel

    # ---------- 单日截面（推理输入） ----------
    def build_cross_section(
        self,
        df: pd.DataFrame,
        date: Union[str, pd.Timestamp],
        feature_cols: Optional[Sequence[str]] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        dt = pd.Timestamp(date)
        d = df[df["datetime"] == dt].copy()
        if d.empty:
            raise ValueError(f"指定日期无数据：{dt}")

        feats_all = list(d.columns)
        feats = feature_cols or self.cfg.data.features
        feats = list(_expand_wildcards(feats_all, feats))

        d = self.cs.winsorize(d, feats)
        d = self.cs.zscore(d, feats)

        panel = d[["instrument", "datetime"]].copy()
        X = d[feats].astype(float)
        return X, panel

    # ---------- 推理后分数中性化（可选接口） ----------
    def neutralize_scores(
        self,
        df_scores: pd.DataFrame,
        df_exposures: pd.DataFrame,
        score_col: str = "score",
        exposures: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        required = {"instrument", "datetime", score_col}
        if not required <= set(df_scores.columns):
            raise ValueError(f"df_scores 缺列：{required - set(df_scores.columns)}")
        expos = exposures or self.cfg.preprocess.neutralize.exposures
        expos = list(expos) if expos else []
        if not expos:
            return df_scores

        cols = ["instrument", "datetime"] + list(expos)
        d = df_scores.merge(df_exposures[cols], on=["instrument", "datetime"], how="left")

        d = self.cs.neutralize(
            d,
            target_col=score_col,
            exposures=_expand_wildcards(d.columns, expos),
            ridge_lambda=self.cfg.preprocess.neutralize.ridge_lambda,
            center_exposures=self.cfg.preprocess.neutralize.center_exposures,
            drop_dummy=self.cfg.preprocess.neutralize.drop_dummy,
        )
        return d
