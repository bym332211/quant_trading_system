#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于特征产物快速生成“占位预测”文件（parquet），用于打通回测与扫描流程。

生成的文件格式：
  columns: [instrument, datetime, score]
  - instrument: 大写字符串
  - datetime: pandas 时间戳（保持原时区-naive）
  - score: 0~1 的伪随机分数（按日期固定种子，可复现），仅用于流程验证

用法示例：
  python scripts/make_dummy_preds.py \
    --features_path artifacts/features_day.parquet \
    --start 2017-01-01 --end 2024-12-31 \
    --out artifacts/preds/weekly/predictions.parquet \
    --per_day_limit 300
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_path", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per_day_limit", type=int, default=300, help="每个交易日最多取多少只标的")
    return ap.parse_args()


def main():
    args = parse_args()
    fpath = Path(args.features_path).expanduser().resolve()
    if not fpath.exists():
        raise FileNotFoundError(f"features_path not found: {fpath}")

    # 读取尽量少的列；若列名缺失则退化读取全部
    try:
        df = pd.read_parquet(fpath, columns=["instrument", "datetime"])
    except Exception:
        df = pd.read_parquet(fpath)

    # 标准化列名
    if "instrument" not in df.columns:
        for c in ["symbol", "ticker", "Symbol", "TICKER"]:
            if c in df.columns:
                df = df.rename(columns={c: "instrument"})
                break
    if "datetime" not in df.columns:
        for c in ["date", "Date", "timestamp", "Timestamp", "DATETIME"]:
            if c in df.columns:
                df = df.rename(columns={c: "datetime"})
                break

    df = df[["instrument", "datetime"]].copy()
    df["instrument"] = df["instrument"].astype(str).str.upper()
    df["datetime"] = pd.to_datetime(df["datetime"], utc=False)

    # 过滤时间窗
    mask = (df["datetime"] >= pd.Timestamp(args.start)) & (df["datetime"] <= pd.Timestamp(args.end))
    df = df.loc[mask]
    if df.empty:
        raise RuntimeError("no rows after date filtering; check start/end and features file")

    # 每日最多取 per_day_limit 支持
    def _sample_group(g: pd.DataFrame) -> pd.DataFrame:
        if len(g) <= args.per_day_limit:
            return g
        return g.sample(n=int(args.per_day_limit), random_state=42)

    df["dt_norm"] = df["datetime"].dt.normalize()
    # 兼容 pandas 未来版本：include_groups=False；旧版本回退不带该参数
    gb = df.groupby("dt_norm", group_keys=False)
    try:
        df = gb.apply(_sample_group, include_groups=False).reset_index(drop=True)
    except TypeError:
        df = gb.apply(_sample_group).reset_index(drop=True)

    # 生成可复现的伪随机 score（按日期播种）
    scores = []
    for d, g in df.groupby("dt_norm"):
        rs = np.random.RandomState(int(pd.Timestamp(d).strftime("%Y%m%d")))
        s = rs.rand(len(g))
        scores.append(pd.Series(s, index=g.index))
    df["score"] = pd.concat(scores).sort_index()

    # 输出 parquet
    outp = Path(args.out).expanduser().resolve()
    outp.parent.mkdir(parents=True, exist_ok=True)
    out_cols = ["instrument", "datetime", "score"]
    df[out_cols].to_parquet(outp, index=False)
    print(f"[saved] dummy preds -> {outp}  rows={len(df)}  days={df['dt_norm'].nunique()}")


if __name__ == "__main__":
    main()
