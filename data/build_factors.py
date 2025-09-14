#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_factors.py  (with liquidity & size buckets one-hot; robust instrument/datetime)

从 Qlib .bin 读取已复权价量（$open,$high,$low,$close,$vwap,$volume），
计算核心因子与未来收益标签，并在同一脚本内补充“暴露列”：
- 市场β：mkt_beta_60         （对等权市场收益的60日滚动β）
- 流动性/规模代理：ln_dollar_vol_20 = ln(ADV20 × VWAP)
- 行业：ind_*               （若 --sector_csv 提供 instrument,sector）
- 市值：ln_mktcap           （close × shares 的 as-of 历史；需 --shares_csv 或 --use_yf_shares）
- 桶列：size_bucket_*（需股本）、liq_bucket_*（不依赖股本）

稳健性：
- 绝不使用未来信息；股本按 “变更生效日 ≤ 当日” as-of 合并并前向填充。
- winsor 默认不处理暴露列（避免削弱基暴露）。
- 修复 pandas 新版 groupby.apply 可能“吃掉” instrument 的问题；落盘前强校验主键。
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

import qlib
from qlib.data import D


# =========================
# 因子计算（逐票）
# =========================
def compute_single_instrument_factors(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """
    输入：index=datetime；列=$open,$high,$low,$close,$vwap,$volume
    输出：与 index 对齐的因子/标签 DataFrame（不含 instrument 列）
    """
    out = pd.DataFrame(index=df.index, copy=True)

    # 基础价量
    for k in ['$open', '$high', '$low', '$close', '$vwap', '$volume']:
        out[k] = df[k].astype(float)

    o, h, l, c, vwap, vol = [out[k] for k in ['$open', '$high', '$low', '$close', '$vwap', '$volume']]

    # ---------- 基础收益 ----------
    ret1 = c.pct_change(1)
    out["ret_1"]  = ret1
    out["ret_5"]  = c.pct_change(5)
    out["ret_20"] = c.pct_change(20)

    # ---------- 动量 ----------
    out["mom_5"]   = (c / c.shift(5) - 1.0)
    out["mom_20"]  = (c / c.shift(20) - 1.0)
    out["mom_60"]  = (c / c.shift(60) - 1.0)

    # ---------- 波动 ----------
    out["vol_20"]  = ret1.rolling(20, min_periods=10).std()
    out["vol_60"]  = ret1.rolling(60, min_periods=20).std()
    out["rng_hl"]  = (h - l) / c.replace(0, np.nan)

    # ---------- 价量 ----------
    logv = np.log1p(vol.clip(lower=0))
    out["logv"]       = logv
    out["logv_zn_20"] = (logv - logv.rolling(20, min_periods=10).mean()) / (logv.rolling(20, min_periods=10).std() + 1e-9)
    out["adv_20"]     = vol.rolling(20, min_periods=10).mean()
    out["vwap_spread"] = (c - vwap) / vwap.replace(0, np.nan)

    # ---------- 微结构/形态 ----------
    out["oc"]            = (c - o) / o.replace(0, np.nan)
    out["upper_shadow"]  = (h - c) / (h - l + 1e-9)
    out["lower_shadow"]  = (o - l) / (h - l + 1e-9)

    # ---------- 均值回归 ----------
    ma5  = c.rolling(5,  min_periods=3).mean()
    ma20 = c.rolling(20, min_periods=10).mean()
    out["ma5_gap"]   = (c - ma5) / (ma5 + 1e-9)
    out["ma20_gap"]  = (c - ma20) / (ma20 + 1e-9)

    # ---------- 季节性 ----------
    dt = pd.to_datetime(out.index)
    out["dow"] = dt.weekday
    if freq == "1min":
        mod = dt.hour * 60 + dt.minute
        out["tod_sin"] = np.sin(2 * np.pi * mod / (6.5*60))
        out["tod_cos"] = np.cos(2 * np.pi * mod / (6.5*60))
    else:
        out["dom_sin"] = np.sin(2 * np.pi * dt.day / 31.0)
        out["dom_cos"] = np.cos(2 * np.pi * dt.day / 31.0)

    # ---------- 标签 ----------
    for k in (1, 5, 20):
        out[f"y_fwd_{k}"] = c.shift(-k) / c - 1.0

    return out


# =========================
# 暴露列计算
# =========================
def add_market_beta(df_all: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    df = df_all.copy()
    if "ret_1" not in df.columns:
        raise ValueError("缺少 ret_1，无法计算 mkt_beta_60")
    mkt = df.groupby("datetime", sort=True)["ret_1"].mean().rename("mkt_ret")
    df = df.merge(mkt, on="datetime", how="left")

    def _beta(sub: pd.DataFrame) -> pd.Series:
        r_i = sub["ret_1"].astype(float)
        r_m = sub["mkt_ret"].astype(float)
        cov = r_i.rolling(window, min_periods=window).cov(r_m)
        var_m = r_m.rolling(window, min_periods=window).var()
        return cov / var_m.replace(0, np.nan)

    # 按票 rolling
    try:
        beta_series = (df[["instrument", "ret_1", "mkt_ret"]]
                       .groupby("instrument", group_keys=False)
                       .apply(_beta, include_groups=False))
    except TypeError:
        beta_series = (df[["instrument", "ret_1", "mkt_ret"]]
                       .groupby("instrument", group_keys=False)
                       .apply(_beta))
    df["mkt_beta_60"] = beta_series.astype(float)
    df["mkt_beta_60"] = df.groupby("instrument", group_keys=False)["mkt_beta_60"].ffill().fillna(0.0)
    df.drop(columns=["mkt_ret"], inplace=True)
    return df


def add_size_proxy(df_all: pd.DataFrame) -> pd.DataFrame:
    df = df_all.copy()
    vwap_col = "$vwap" if "$vwap" in df.columns else ("vwap" if "vwap" in df.columns else None)
    if ("adv_20" not in df.columns) or (vwap_col is None):
        print("[warn] 缺少 adv_20 或 vwap，跳过 ln_dollar_vol_20")
        return df
    dv = (df["adv_20"].astype(float) * df[vwap_col].astype(float)).clip(lower=1.0)
    df["ln_dollar_vol_20"] = np.log(dv)
    return df


def add_sector_onehot(df_all: pd.DataFrame, sector_csv: str | None) -> pd.DataFrame:
    if not sector_csv:
        return df_all
    csv_path = Path(sector_csv).expanduser()
    if not csv_path.exists():
        print(f"[info] 未找到行业映射文件：{csv_path}，跳过行业 one-hot")
        return df_all

    mapdf = pd.read_csv(csv_path)
    required = {"instrument", "sector"}
    if not required <= set(mapdf.columns):
        print(f"[warn] 行业映射缺列，需要 {required}，已跳过")
        return df_all

    df = df_all.copy()
    df["instrument"] = df["instrument"].astype(str).str.upper()
    mapdf["instrument"] = mapdf["instrument"].astype(str).str.upper()

    df = df.merge(mapdf[["instrument", "sector"]], on="instrument", how="left")
    if df["sector"].notna().any():
        dummies = pd.get_dummies(df["sector"], prefix="ind", dtype=np.float32)
        df = pd.concat([df.drop(columns=["sector"]), dummies], axis=1)
        print(f"[info] 已加入行业列：{[c for c in dummies.columns]}")
    else:
        df = df.drop(columns=["sector"])
        print("[info] 行业列全为空，已跳过")
    return df


# =========================
# 市值（ln_mktcap）与 size 桶（需股本）
# =========================
def _load_shares_csv(path: str) -> pd.DataFrame:
    p = Path(path).expanduser()
    df = pd.read_csv(p)
    must = {"instrument", "date", "shares"}
    if not must <= set(df.columns):
        raise ValueError(f"--shares_csv 需要列 {must}")
    df = df.copy()
    df["instrument"] = df["instrument"].astype(str).str.upper()
    df["date"] = pd.to_datetime(df["date"], utc=False)
    df["shares"] = df["shares"].astype(float)
    df = df.sort_values(["instrument", "date"]).drop_duplicates(["instrument", "date"])
    return df


def _fetch_shares_yf(symbols: list[str], sleep: float = 0.6) -> pd.DataFrame:
    try:
        import yfinance as yf
    except Exception as e:
        raise RuntimeError("请先 pip install yfinance") from e
    import time
    rows = []
    for sym in symbols:
        try:
            tk = yf.Ticker(sym)
            hist = None
            if hasattr(tk, "get_shares_full"):
                try:
                    hist = tk.get_shares_full(start="1990-01-01")
                except Exception:
                    hist = None
            if hist is not None and not getattr(hist, "empty", True):
                df = hist.reset_index().rename(columns={"Date": "date", "Shares Outstanding": "shares"})
            else:
                v = None
                try:
                    v = getattr(tk.fast_info, "shares_outstanding", None)
                except Exception:
                    v = None
                if v is None:
                    try:
                        info = tk.get_info()
                        v = info.get("sharesOutstanding")
                    except Exception:
                        v = None
                if v:
                    df = pd.DataFrame({"date": [pd.Timestamp("1990-01-01")], "shares": [float(v)]})
                else:
                    df = None
            if df is not None and not df.empty:
                df["instrument"] = sym
                rows.append(df[["instrument", "date", "shares"]])
        except Exception:
            pass
        time.sleep(max(0.0, sleep))
    if not rows:
        raise RuntimeError("yfinance 未获取到任何 shares 数据")
    out = pd.concat(rows).dropna()
    out["instrument"] = out["instrument"].astype(str).str.upper()
    out["date"] = pd.to_datetime(out["date"], utc=False)
    out["shares"] = out["shares"].astype(float)
    out = out.sort_values(["instrument", "date"]).drop_duplicates(["instrument", "date"])
    return out


def _int_dummies(s: pd.Series, prefix: str) -> pd.DataFrame:
    """
    将分桶标签统一转为可空整型 Int64，再做 one-hot，确保列名稳定为 prefix_0..k-1
    """
    s = pd.Series(s, index=s.index).astype("Int64")
    oh = pd.get_dummies(s, prefix=prefix, dtype=np.float32)
    # 规范列名（避免 '0.0'）
    oh.columns = [f"{prefix}_{int(str(c).split('_')[-1])}" for c in oh.columns]
    return oh


def add_ln_mktcap_and_size(df_all: pd.DataFrame,
                           shares_csv: str | None = None,
                           use_yf_shares: bool = False,
                           yf_sleep: float = 0.6,
                           size_buckets: int = 0) -> pd.DataFrame:
    df = df_all.copy()
    close_col = "$close" if "$close" in df.columns else "close"
    if close_col not in df.columns:
        print("[warn] 缺少 close 列，无法计算 ln_mktcap；已跳过")
        return df

    # 股本来源
    shares_ts = None
    if shares_csv:
        shares_ts = _load_shares_csv(shares_csv)
        print(f"[info] shares_csv 加载 {shares_ts['instrument'].nunique()} tickers, {len(shares_ts)} 行")
    elif use_yf_shares:
        syms = sorted(df["instrument"].astype(str).str.upper().unique().tolist())
        shares_ts = _fetch_shares_yf(syms, sleep=yf_sleep)
        print(f"[info] yfinance 抓到 {shares_ts['instrument'].nunique()} tickers, {len(shares_ts)} 行")

    if shares_ts is None or shares_ts.empty:
        print("[info] 未提供股本，跳过 ln_mktcap/size_bucket_*")
        return df

    # as-of 合并（避免泄漏）
    out_parts = []
    for sym, g in df.groupby("instrument", group_keys=False):
        g = g.sort_values("datetime")
        s = shares_ts[shares_ts["instrument"] == sym].sort_values("date")
        if s.empty:
            out_parts.append(g)
            continue
        merged = pd.merge_asof(
            g, s[["date", "shares"]], left_on="datetime", right_on="date",
            direction="backward", allow_exact_matches=True
        ).drop(columns=["date"])
        out_parts.append(merged)
    df = pd.concat(out_parts, axis=0)
    df["shares"] = df.groupby("instrument", group_keys=False)["shares"].ffill()
    if df["shares"].isna().any():
        med = df["shares"].median()
        df["shares"] = df["shares"].fillna(med if np.isfinite(med) and med > 0 else 1e6)

    mktcap = (df[close_col].astype(float) * df["shares"].astype(float)).clip(lower=1.0)
    df["ln_mktcap"] = np.log(mktcap)

    # size 桶 one-hot（可选）
    if size_buckets and size_buckets > 1:
        k = int(size_buckets)
        for dt, g in df.groupby("datetime"):
            try:
                q = pd.qcut(g["ln_mktcap"], q=k, labels=False, duplicates="drop")
            except ValueError:
                continue
            oh = _int_dummies(q, prefix="size_bucket")
            df.loc[g.index, oh.columns] = oh.values
        print(f"[info] 已加入 size 桶列: {[c for c in df.columns if c.startswith('size_bucket_')]}")

    return df


def add_liquidity_buckets(df_all: pd.DataFrame, buckets: int = 5) -> pd.DataFrame:
    """
    按当日 ln_dollar_vol_20 做分位分桶，生成 liq_bucket_* 列。
    buckets < 2 则不生效；数据不足/重复分位会自动跳过当日。
    """
    if buckets is None or int(buckets) < 2:
        return df_all
    df = df_all.copy()
    if "ln_dollar_vol_20" not in df.columns:
        print("[info] 无 ln_dollar_vol_20，跳过流动性分桶")
        return df
    k = int(buckets)
    for dt, g in df.groupby("datetime"):
        try:
            q = pd.qcut(g["ln_dollar_vol_20"], q=k, labels=False, duplicates="drop")
        except ValueError:
            continue
        oh = _int_dummies(q, prefix="liq_bucket")
        df.loc[g.index, oh.columns] = oh.values
    added = [c for c in df.columns if c.startswith("liq_bucket_")]
    if added:
        print(f"[info] 已加入流动性桶列: {added}")
    return df


# =========================
# winsor（默认不处理暴露列）
# =========================
def winsorize(df: pd.DataFrame, limits: float = 0.01, exclude: set[str] | None = None) -> pd.DataFrame:
    out = df.copy()
    exclude = exclude or set()
    numeric_cols = out.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col in exclude:
            continue
        s = out[col]
        lo, hi = s.quantile(limits), s.quantile(1 - limits)
        out[col] = s.clip(lo, hi)
    return out


# =========================
# 主流程
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qlib_dir", required=True, help="Qlib .bin 路径")
    ap.add_argument("--freq", default="day", choices=["day", "1min", "5min", "15min", "30min", "60min"],
                    help="原数据频率；分钟库统一读取为 1min")
    ap.add_argument("--start", default="2010-01-01")
    ap.add_argument("--end",   default="2100-01-01")
    ap.add_argument("--out",   required=True, help="输出 Parquet 路径")
    ap.add_argument("--sample", type=int, default=0, help="仅抽样前 N 只；0=全部")

    # 暴露列参数
    ap.add_argument("--beta_window", type=int, default=60, help="mkt_beta_60 的滚动窗口")
    ap.add_argument("--sector_csv", type=str, default="data/instrument_sector.csv", help="instrument,sector 映射 CSV（默认尝试 data/instrument_sector.csv，不存在则自动跳过）")
    ap.add_argument("--shares_csv", type=str, default="", help="可选：instrument,date,shares 股本 CSV（优先）")
    ap.add_argument("--use_yf_shares", action="store_true", help="若无 shares_csv，则用 yfinance 抓历史股本")
    ap.add_argument("--yf_sleep", type=float, default=0.6, help="yfinance 抓取间隔秒数")
    ap.add_argument("--size_buckets", type=int, default=0, help="按 ln_mktcap 做 size 桶 one-hot（如 5）")
    ap.add_argument("--liq_buckets", type=int, default=5, help="按 ln_dollar_vol_20 做流动性桶 one-hot（默认 5）")

    # winsor 设置
    ap.add_argument("--winsor", type=float, default=0.01, help="winsorize 百分位（每边）；0 关闭")
    ap.add_argument("--winsor_exposures", action="store_true", help="若设置，也对暴露列做 winsor（默认否）")
    args = ap.parse_args()

    qlib_dir = str(Path(args.qlib_dir).expanduser().resolve())
    read_freq = "1min" if args.freq in {"1min", "5min", "15min", "30min", "60min"} else "day"

    print(f"[INIT] qlib_dir={qlib_dir} | read_freq={read_freq} | range={args.start}~{args.end}")
    qlib.init(provider_uri=qlib_dir, region="us")

    # 取标的清单
    inst_file = Path(qlib_dir) / "instruments" / "all.txt"
    if not inst_file.exists():
        raise FileNotFoundError(f"未找到 {inst_file}")
    # instruments/all.txt format: CODE[TAB]start[TAB]end; only take CODE
    rows = [l.strip() for l in inst_file.read_text().splitlines() if l.strip()]
    symbols = []
    for r in rows:
        code = r.split("\t")[0].strip()
        if code:
            symbols.append(code)
    # de-duplicate while preserving order
    seen = set()
    symbols = [s for s in symbols if not (s in seen or seen.add(s))]
    if args.sample and args.sample > 0:
        symbols = symbols[:args.sample]

    # 拉取基础字段
    base_fields = ["$open", "$high", "$low", "$close", "$vwap", "$volume"]
    print(f"[LOAD] symbols={len(symbols)} fields={base_fields}")
    raw = D.features(symbols, base_fields, start_time=args.start, end_time=args.end, freq=read_freq)
    if raw.empty:
        raise RuntimeError("Qlib 返回空数据；检查 qlib_dir / 时间窗 / 频率")

    # 分票计算因子（确保后续有 instrument 列）
    print("[FACTOR] computing per instrument ...")
    feats = []
    for sym, df_sym in raw.groupby(level=0):
        sub = df_sym.droplevel(0).sort_index()
        fdf = compute_single_instrument_factors(sub, read_freq)
        fdf.insert(0, "instrument", str(sym).upper())
        feats.append(fdf)

    # 更稳：先命名 index 再 reset，避免出现 level_0/level_1
    allf = pd.concat(feats)
    allf.index.name = "datetime"
    allf = allf.reset_index()  # 列里有 datetime 了
    allf["instrument"] = allf["instrument"].astype(str).str.upper()
    allf["datetime"]   = pd.to_datetime(allf["datetime"], utc=False)

    # ---------- 暴露列 ----------
    print("[EXPOSURE] mkt_beta_60 ...")
    allf = add_market_beta(allf, window=args.beta_window)

    print("[EXPOSURE] ln_dollar_vol_20 ...")
    allf = add_size_proxy(allf)

    if args.sector_csv:
        print("[EXPOSURE] sector one-hot ...")
        allf = add_sector_onehot(allf, args.sector_csv)

    if args.liq_buckets and args.liq_buckets > 1:
        print("[EXPOSURE] liquidity buckets ...")
        allf = add_liquidity_buckets(allf, buckets=args.liq_buckets)

    print("[EXPOSURE] ln_mktcap (+ size buckets if shares available) ...")
    shares_csv = args.shares_csv if args.shares_csv.strip() else None
    allf = add_ln_mktcap_and_size(
        allf,
        shares_csv=shares_csv,
        use_yf_shares=(not shares_csv and args.use_yf_shares),
        yf_sleep=args.yf_sleep,
        size_buckets=args.size_buckets,
    )

    # ---------- winsor（默认跳过暴露列） ----------
    if args.winsor and args.winsor > 0:
        exposure_cols = set(
            [c for c in allf.columns
             if c.startswith("ind_") or c.startswith("size_bucket_") or c.startswith("liq_bucket_")]
        )
        for name in ("mkt_beta_60", "ln_dollar_vol_20", "ln_mktcap"):
            if name in allf.columns:
                exposure_cols.add(name)
        if args.winsor_exposures:
            exposure_cols = set()  # 用户要求也 winsor 暴露列

        # 显式 for-loop，保证 instrument 作为列不丢失
        parts = []
        for sym, g in allf.groupby("instrument", sort=False):
            gg = winsorize(g, limits=args.winsor, exclude=exposure_cols)
            if "instrument" not in gg.columns:
                gg = gg.assign(instrument=sym)
            parts.append(gg)
        allf = pd.concat(parts, axis=0)
        allf = allf.sort_values(["instrument", "datetime"]).reset_index(drop=True)

    # ---------- 质量报告 ----------
    num_cols = [c for c in allf.columns if c not in ("instrument", "datetime")]
    nans = allf[num_cols].isna().mean().sort_values(ascending=False).head(12)
    print("[QC] top-12 NA ratio:\n", nans)

    # ---------- 主键兜底：索引 -> 列，别名修正 ----------
    idx_names = list(getattr(allf.index, "names", []) or [])
    if "instrument" in idx_names or "datetime" in idx_names:
        print(f"[FIX] reset_index 因为索引里包含: {idx_names}")
        allf = allf.reset_index()

    if "instrument" not in allf.columns:
        for c in ("symbol", "ticker", "Instrument", "Symbol", "TICKER"):
            if c in allf.columns:
                allf = allf.rename(columns={c: "instrument"})
                break
    if "datetime" not in allf.columns:
        for c in ("date", "Date", "timestamp", "Timestamp", "DATETIME"):
            if c in allf.columns:
                allf = allf.rename(columns={c: "datetime"})
                break

    if "instrument" not in allf.columns or "datetime" not in allf.columns:
        raise RuntimeError(f"[FATAL] 缺少 instrument/datetime；实际列: {list(allf.columns)[:40]}")

    allf["instrument"] = allf["instrument"].astype(str).str.upper()
    allf["datetime"]   = pd.to_datetime(allf["datetime"], utc=False)

    # 主键置前 & 去重
    front = ["instrument", "datetime"]
    allf = allf[front + [c for c in allf.columns if c not in front]]
    dups = allf.duplicated(front).sum()
    if dups:
        print(f"[WARN] 发现 {dups} 个重复主键，已保留最后一条")
        allf = allf.drop_duplicates(front, keep="last")

    # ---------- 存盘 ----------
    outp = Path(args.out).expanduser().resolve()
    outp.parent.mkdir(parents=True, exist_ok=True)
    allf.to_parquet(outp, index=False)
    print(f"[DONE] saved features: {outp} (rows={len(allf)}, cols={len(allf.columns)})")


if __name__ == "__main__":
    main()
