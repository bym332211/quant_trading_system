#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
models/lgbm/dataset.py

用途
- 读取 features parquet（由 data/build_factors.py 生成）
- 逐日横截面预处理：winsor → zscore（可选）
- 训练期标签中性化（y ~ exposures，取残差）
- 构建训练集（X, y）与推理用单日截面（X_t）

稳健性
- 兼容 instrument/datetime 可能出现在索引或被命名为别名（symbol/ticker、date/timestamp）
- 兼容 pandas 版本差异的 groupby.apply 行为
- dataclass 使用 default_factory 以兼容 Python 3.12
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union
from pathlib import Path
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
    exposures: Sequence[str] = ()  # 支持通配符，如 ["mkt_beta_60","ln_mktcap","ind_*","liq_bucket_*"]
    ridge_lambda: float = 1e-8
    center_exposures: bool = True
    drop_dummy: bool = True  # 对 one-hot 组自动 drop 1 列


@dataclass
class PreprocessConfig:
    winsor: float = 0.01         # 逐日截面 winsor 百分位；0 关闭
    zscore: bool = True          # 逐日截面标准化
    neutralize: NeutralizeConfig = field(default_factory=NeutralizeConfig)


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
    data: DataConfig = field(default_factory=DataConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)


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
            out.append(c)
            seen.add(c)
    return out


def _groupby_apply(df: pd.DataFrame, key: str, func):
    """
    兼容：key 既可能是列名，也可能是索引层级名。
    优先按列分组；若不在列中则尝试按索引 level 分组。
    并在返回后确保 key 仍然是列（若丢失则按索引对齐拼回）。
    """
    cols = set(df.columns)
    idx_names = list(getattr(df.index, "names", []) or [])
    if key in cols:
        gb = df.groupby(key, group_keys=False)
        key_is_column = True
    elif key in idx_names:
        gb = df.groupby(level=key, group_keys=False)
        key_is_column = False
    else:
        raise KeyError(f"{key} not found in columns or index levels; "
                       f"columns={list(df.columns)[:10]}..., index_names={idx_names}")

    try:
        res = gb.apply(func, include_groups=False)
    except TypeError:
        # 兼容老版本 pandas
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            res = gb.apply(func)

    # 确保 key 作为列存在
    if isinstance(res, pd.DataFrame) and key not in res.columns:
        if key_is_column and key in df.columns:
            # 索引对齐拼回
            res = res.join(df[[key]], how="left")
        elif (not key_is_column) and key in idx_names:
            # 如果 key 在原索引层级上，reset 一次索引再保证列存在
            if key in list(getattr(res.index, "names", []) or []):
                res = res.reset_index()
            # 若还没列，尝试从原 df 取
            if key not in res.columns and key in df.columns:
                res = res.join(df[[key]], how="left")
    return res


def _coerce_inst_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """
    将不同命名/索引形态的 instrument/datetime 统一为标准列。
    """
    d = df.copy()

    # 如果 instrument 在索引里
    if "instrument" not in d.columns:
        idx_names = list(getattr(d.index, "names", []) or [])
        if "instrument" in idx_names:
            d = d.reset_index()

    # 常见别名 → instrument
    if "instrument" not in d.columns:
        for cand in ["symbol", "ticker", "Instrument", "Symbol", "TICKER"]:
            if cand in d.columns:
                d = d.rename(columns={cand: "instrument"})
                break

    # 如果 datetime 在索引里
    if "datetime" not in d.columns:
        idx_names = list(getattr(d.index, "names", []) or [])
        if "datetime" in idx_names:
            d = d.reset_index()

    # 常见别名 → datetime
    if "datetime" not in d.columns:
        for cand in ["date", "Date", "timestamp", "Timestamp", "DATETIME"]:
            if cand in d.columns:
                d = d.rename(columns={cand: "datetime"})
                break

    # 规范类型
    if "instrument" in d.columns:
        d["instrument"] = d["instrument"].astype(str).str.upper()
    if "datetime" in d.columns:
        d["datetime"] = pd.to_datetime(d["datetime"], utc=False, errors="coerce")

    return d


def _ensure_datetime_column(df: pd.DataFrame) -> pd.DataFrame:
    """若 datetime 在索引上，则 reset 成列；保证存在 datetime 列。"""
    d = df.copy()
    if "datetime" not in d.columns and "datetime" in (getattr(d.index, "names", []) or []):
        d = d.reset_index()
    if "datetime" not in d.columns:
        raise KeyError("缺少 datetime 列；请检查上游数据。")
    return d


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
        """
        使用 transform 逐列逐日处理，避免 groupby.apply 丢列。
        """
        if not self.winsor or self.winsor <= 0:
            return df
        cols = [c for c in cols if c in df.columns]
        if not cols:
            return df
        out = df.copy()
        g = out.groupby("datetime")
        for c in cols:
            out[c] = g[c].transform(lambda s: self._winsorize_vec(s.astype(float), self.winsor))
        return out

    def zscore(self, df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
        """
        使用 transform 逐列逐日标准化，避免 groupby.apply 丢列。
        """
        if not self.do_z:
            return df
        cols = [c for c in cols if c in df.columns]
        if not cols:
            return df
        out = df.copy()
        g = out.groupby("datetime")
        for c in cols:
            s = out[c].astype(float)
            mu = g[c].transform("mean")
            # 使用 ddof=0 的方差
            var0 = g[c].transform(lambda x: x.astype(float).var(ddof=0))
            sd = np.sqrt(var0)
            # 0 方差 → 置 0
            z = (s - mu) / sd.replace(0, np.nan)
            out[c] = z.fillna(0.0)
        return out

    @staticmethod
    def _auto_drop_dummy(expos_cols: List[str]) -> List[str]:
        """
        简单按前缀归组并每组 drop 最后一列，避免 dummy trap。
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

        # 仍用 _groupby_apply，但它现在会在必要时把 key 拼回列
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
        p = Path(path)
        if not p.is_absolute():
            # 把相对路径视为 “项目根目录” 相对路径
            repo_root = Path(__file__).resolve().parents[2]  # models/lgbm/ 上两级就是仓库根
            p = (repo_root / p).resolve()
        if not p.exists():
            raise FileNotFoundError(f"features parquet 不存在：{p}")

        df = pd.read_parquet(p)
        # 统一 instrument/datetime（容错别名/索引）
        df = _coerce_inst_datetime(df)

        required = {"instrument", "datetime"}
        if not required <= set(df.columns):
            have = list(df.columns)
            idxn = getattr(df.index, "names", None)
            raise ValueError(f"features 缺列：{required - set(df.columns)} | "
                             f"现有列示例：{have[:30]} | index names={idxn}")

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

        # 强校验 & 兜底
        must = {"instrument", "datetime", label}
        missing = must - set(df.columns)
        if missing:
            raise KeyError(
                f"build_train_xy 需要列 {must}，但当前缺少：{missing}。"
                "很可能你在进入 build_train_xy 前做了列筛选，把 datetime 或 instrument 丢了。"
            )
        d = df.loc[(df["datetime"] >= pd.Timestamp(start)) & (df["datetime"] <= pd.Timestamp(end))].copy()
        d = _ensure_datetime_column(d)

        feats_all = list(d.columns)
        feats = list(feature_cols) if feature_cols else list(self.cfg.data.features)
        feats = list(_expand_wildcards(feats_all, feats))

        # 横截面 winsor & zscore
        d = self.cs.winsorize(d, feats)
        d = self.cs.zscore(d, feats)

        # 标签中性化（可选）
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
        cols_need = [c for c in cols_need if c in d.columns]  # 防御
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
        # 兜底：保证 datetime 是列
        d0 = _ensure_datetime_column(df)
        dt = pd.Timestamp(date)
        d = d0[d0["datetime"] == dt].copy()
        if d.empty:
            raise ValueError(f"指定日期无数据：{dt}")

        feats_all = list(d.columns)
        feats = list(feature_cols) if feature_cols else list(self.cfg.data.features)
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
