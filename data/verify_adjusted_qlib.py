#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_adjusted_qlib.py

目的：
- 验证 Qlib 读出的 $open/$high/$low/$close/$vwap 是否等于我们标准化CSV里的“复权价”；
- 同时确认 $factor 恒为 1.0。

用法示例：
python data/verify_adjusted_qlib.py \
  --qlib_dir ~/.qlib/qlib_data/us_data_5min \
  --source_dir ~/.qlib/source/us_from_yf_5min \
  --freq 5min --read-freq 1min --sample 3 --rows 400
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import math

import pandas as pd
import numpy as np

import qlib
from qlib.data import D

def pick_symbols(qlib_dir: Path, source_dir: Path, sample: int):
    # 优先从 instruments/all.txt 取
    inst = qlib_dir / "instruments" / "all.txt"
    syms = []
    if inst.exists():
        syms = [l.strip() for l in inst.read_text().splitlines() if l.strip()]
    if not syms:
        # 兜底：从 source_dir 的文件名推断
        for p in sorted(source_dir.glob("*.csv")):
            syms.append(p.stem)
    return syms[:sample]

def read_csv_slice(source_dir: Path, symbol: str, rows: int) -> pd.DataFrame:
    fp = source_dir / f"{symbol}.csv"
    df = pd.read_csv(fp)
    # 统一列名
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    if "date" not in df.columns:
        raise RuntimeError(f"{fp} 缺少 date 列")
    # 解析为 tz-naive
    dt = pd.to_datetime(df["date"], utc=True, errors="coerce").dt.tz_convert("UTC").dt.tz_localize(None)
    df["date"] = dt
    df = df.sort_values("date").tail(rows).reset_index(drop=True)
    return df[["date","open","high","low","close","vwap","volume"]]

def compare_series(name: str, a: pd.Series, b: pd.Series, atol=1e-4, rtol=1e-4):
    """返回(通过?, 最大绝对误差, 最大相对误差)"""
    a = a.astype(float); b = b.astype(float)
    if len(a) == 0:
        return True, 0.0, 0.0
    abs_err = (a - b).abs()
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_err = abs_err / np.maximum(b.abs(), 1e-12)
    ok = bool((abs_err <= atol).all() or (rel_err <= rtol).all())
    return ok, float(abs_err.max()), float(rel_err.max())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qlib_dir", required=True)
    ap.add_argument("--source_dir", required=True)
    ap.add_argument("--freq", required=True, choices=["day","1min","5min","15min","30min","60min"])
    ap.add_argument("--read-freq", default="", help="Qlib 读取频率；分钟库统一用 1min（默认自动映射）")
    ap.add_argument("--sample", type=int, default=3)
    ap.add_argument("--rows", type=int, default=400)
    args = ap.parse_args()

    qlib_dir = Path(args.qlib_dir).expanduser().resolve()
    source_dir = Path(args.source_dir).expanduser().resolve()

    # Qlib 读取频率映射
    freq = args.freq.lower()
    qfreq = args.read_freq.lower() if args.read_freq else ( "1min" if freq in {"1min","5min","15min","30min","60min"} else "day" )

    print(f"[INIT] qlib_dir={qlib_dir} | source_dir={source_dir} | freq={freq} | read_freq={qfreq}")

    qlib.init(provider_uri=str(qlib_dir), region="us")

    symbols = pick_symbols(qlib_dir, source_dir, args.sample)
    if not symbols:
        print("[FATAL] 无可用标的", file=sys.stderr)
        sys.exit(2)
    print("Sampling symbols:", symbols)

    # 时间范围：取第一个标的 CSV 末尾 rows 的起止
    df0 = read_csv_slice(source_dir, symbols[0], args.rows)
    if df0.empty:
        print("[FATAL] CSV 无数据", file=sys.stderr)
        sys.exit(2)
    start, end = df0["date"].iloc[0], df0["date"].iloc[-1]
    start_s, end_s = pd.Timestamp(start).strftime("%Y-%m-%d %H:%M:%S"), pd.Timestamp(end).strftime("%Y-%m-%d %H:%M:%S")
    print("Time range:", start_s, "->", end_s)

    # 读取 Qlib 特征
    fields = ["$open","$high","$low","$close","$vwap","$factor"]
    qdf = D.features(symbols, fields, start_time=start_s, end_time=end_s, freq=qfreq)
    if qdf.empty:
        print("[FATAL] Qlib 返回空数据", file=sys.stderr)
        sys.exit(2)
    qdf = qdf.reset_index().rename(columns={"instrument":"symbol","datetime":"date"})
    # 统一为 tz-naive
    qdf["date"] = pd.to_datetime(qdf["date"], utc=True).dt.tz_convert("UTC").dt.tz_localize(None)

    # 校验每只标的
    all_ok = True
    for sym in symbols:
        csv = read_csv_slice(source_dir, sym, args.rows)
        qsub = qdf[qdf["symbol"]==sym].copy()

        # 对齐时间交集
        common = pd.Index(sorted(set(csv["date"]).intersection(set(qsub["date"]))))
        csv = csv[csv["date"].isin(common)].set_index("date").sort_index()
        qsub = qsub[qsub["date"].isin(common)].set_index("date").sort_index()

        if len(common)==0:
            print(f"[{sym}] 无公共时间交集，跳过")
            continue

        # 检查 factor==1
        fac_unique = set(np.round(qsub["$factor"].dropna().astype(float).unique(), 8).tolist())
        fac_ok = (len(fac_unique) <= 1 and (len(fac_unique)==0 or list(fac_unique)[0] == 1.0))
        print(f"[{sym}] factor唯一值: {fac_unique} -> {'OK' if fac_ok else 'FAIL'}")

        checks = [
            ("open",  qsub["$open"],  csv["open"]),
            ("high",  qsub["$high"],  csv["high"]),
            ("low",   qsub["$low"],   csv["low"]),
            ("close", qsub["$close"], csv["close"]),
            ("vwap",  qsub["$vwap"],  csv["vwap"]),
        ]
        sym_ok = fac_ok
        for name, qa, ca in checks:
            ok, amax, rmax = compare_series(name, qa, ca, atol=1e-4, rtol=1e-4)
            print(f"[{sym}] {name:<5} -> {'OK ' if ok else 'FAIL'} | max_abs_err={amax:.6g} max_rel_err={rmax:.6g}")
            sym_ok = sym_ok and ok

        all_ok = all_ok and sym_ok
        print(f"[{sym}] RESULT => {'PASS ✅' if sym_ok else 'FAIL ❌'}  (rows={len(common)})")
        print("-"*60)

    if not all_ok:
        print("OVERALL RESULT: FAIL ❌", file=sys.stderr)
        sys.exit(3)
    print("OVERALL RESULT: PASS ✅")
    sys.exit(0)

if __name__ == "__main__":
    main()
