#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_qlib_us.py

功能：
1) 将每个标的的原始 CSV（示例列：Date, open, high, low, close, adj_close, volume, ...）
   标准化为 Qlib 认可的第三方格式：
   date, symbol, open, high, low, close, volume, factor[, vwap]
   并输出到 --source_dir
2) 若安装了 qlib，自动调用 dump_bin 将标准化后的 CSV 写入 --qlib_dir 生成 .bin

用法（以日级别为例）：
python build_qlib_us.py \
  --in_dir ~/data/sp500 \
  --source_dir ~/.qlib/source/us_from_yf \
  --qlib_dir ~/.qlib/qlib_data/us_data \
  --freq day \
  --n_jobs 8
"""

import argparse
import concurrent.futures as futures
import os
import re
import sys
import traceback
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import subprocess

# --------------------------------
# 频率映射
# --------------------------------
VALID_FREQS = {
    "day": "day",
    "1min": "1min",
    "5min": "5min",
    "15min": "15min",
    "30min": "30min",
    "60min": "60min",
    "1h": "60min",  # 兼容写法
}

REQUIRED_COLS = ["date", "symbol", "open", "high", "low", "close", "volume", "factor"]
OPTIONAL_COLS = ["vwap"]  # 若能计算则包含

# 支持从文件名中识别并移除的频率后缀（例如 A_5min.csv -> A）
FREQ_SUFFIX_RE = re.compile(r"_(?:1min|5min|15min|30min|60min|1h|1m|5m|15m|30m|60m)$", re.IGNORECASE)

# --------------------------------
# 工具函数
# --------------------------------
def expanduser_mkdir(p: str) -> Path:
    path = Path(os.path.expanduser(p)).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path

def infer_symbol_from_path(p: Path) -> str:
    """
    从文件名推断 ticker。
    - 允许形如 A.csv、BRK-B.csv、RDS.A.csv
    - 若带时间粒度后缀（如 A_5min.csv），去掉后缀部分
    """
    name = p.stem.strip()
    # 去除 _{freq} 后缀
    name = FREQ_SUFFIX_RE.sub("", name)
    return name

def to_utc_naive(series: pd.Series, assume_utc: bool = True) -> pd.Series:
    """
    统一把时间戳转为 UTC 且 tz-naive。
    - 对于含 +00:00 或任意 tz 的字符串：解析为 tz-aware -> 转 UTC -> 去 tz
    - 对于不含 tz 的字符串：直接解析；若 assume_utc=True，则视作 UTC（无 tz）
    """
    if assume_utc:
        dt = pd.to_datetime(series, utc=True, errors="coerce")
        # 此时 dt 为 tz-aware（UTC），去 tz：
        return dt.dt.tz_convert("UTC").dt.tz_localize(None)
    else:
        # 少数场景希望保留本地时间的“名义值”，但仍转为 tz-naive
        dt = pd.to_datetime(series, errors="coerce")
        # 若本来是 aware，则转为 UTC 后去 tz
        if getattr(dt.dtype, "tz", None) is not None:
            return dt.dt.tz_convert("UTC").dt.tz_localize(None)
        return dt

def normalize_one_csv(csv_path: Path, out_dir: Path, freq_key: str) -> Tuple[str, Optional[str]]:
    """
    处理单个标的 CSV，输出标准化 CSV 到 out_dir。
    - open/high/low/close/vwap：写入 **复权价**（优先用 adj_*，否则用 adj_close/close 的比例统一缩放）
    - volume：原样
    - factor：固定为 1.0（防止后续忘记乘因子）
    返回 (symbol, error_msg)，error_msg 为 None 表示成功。
    """
    symbol = infer_symbol_from_path(csv_path)
    try:
        df = pd.read_csv(csv_path)
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

        # 日期列
        date_col = next((c for c in ("date", "datetime", "time", "timestamp") if c in df.columns), None)
        if date_col is None:
            return symbol, "缺少日期列（date/datetime/time/timestamp）"

        # 取列工具
        def pick(*names):
            for n in names:
                if n in df.columns:
                    return n
            return None

        # 原始价列
        c_open  = pick("open")
        c_high  = pick("high")
        c_low   = pick("low")
        c_close = pick("close")
        c_volume = pick("volume", "vol")

        # 复权价列（若存在，优先使用）
        c_aopen  = pick("adj_open", "adjopen")
        c_ahigh  = pick("adj_high", "adjhigh")
        c_alow   = pick("adj_low", "adjlow")
        c_aclose = pick("adj_close", "adjclose", "adjusted_close")

        # 必要列检查
        missing = [n for n, c in [
            ("open", c_open), ("high", c_high), ("low", c_low),
            ("close", c_close), ("volume", c_volume)
        ] if c is None]
        if missing:
            return symbol, f"缺少必要列：{missing}"

        # 数值化
        o_raw = pd.to_numeric(df[c_open], errors="coerce")
        h_raw = pd.to_numeric(df[c_high], errors="coerce")
        l_raw = pd.to_numeric(df[c_low],  errors="coerce")
        c_raw = pd.to_numeric(df[c_close], errors="coerce")
        v_raw = pd.to_numeric(df[c_volume], errors="coerce")

        # 计算缩放比例 f（用于在没有 adj_* 时把原始 OHLC 等比调整成复权价）
        if c_aclose is not None and c_close is not None:
            aclose = pd.to_numeric(df[c_aclose], errors="coerce")
            with np.errstate(divide="ignore", invalid="ignore"):
                f = (aclose / c_raw).replace([np.inf, -np.inf], np.nan)
        else:
            f = pd.Series(1.0, index=df.index, dtype="float64")
        f = f.ffill().bfill().fillna(1.0).clip(0.05, 20.0)

        # 复权后的 OHLC：优先直接用 adj_*；否则用 f * 原价
        def use_adj_or_scale(raw_s: pd.Series, adj_col: Optional[str]) -> pd.Series:
            if adj_col is not None and adj_col in df.columns:
                return pd.to_numeric(df[adj_col], errors="coerce")
            return raw_s * f

        o_adj = use_adj_or_scale(o_raw, c_aopen)
        h_adj = use_adj_or_scale(h_raw, c_ahigh)
        l_adj = use_adj_or_scale(l_raw, c_alow)
        c_adj = use_adj_or_scale(c_raw, c_aclose)

        # 时间统一：UTC tz-naive；日线仅保留日期
        dt = to_utc_naive(df[date_col], assume_utc=True)
        if freq_key == "day":
            dt = pd.to_datetime(dt.dt.date)

        # vwap 用复权价近似
        vwap = (h_adj + l_adj + c_adj) / 3.0

        out = pd.DataFrame({
            "date": dt,
            "symbol": symbol,
            "open":   o_adj,
            "high":   h_adj,
            "low":    l_adj,
            "close":  c_adj,
            "volume": v_raw,
            "factor": 1.0,     # 复权价已写入，固定 1.0 避免后续误用
            "vwap":   vwap,
        })

        # 清洗与排序
        out = out.dropna(subset=["date", "open", "high", "low", "close", "volume"])
        out = out.sort_values("date").drop_duplicates(subset=["date"])

        # 列顺序
        cols = [*REQUIRED_COLS]
        for c in OPTIONAL_COLS:
            if c in out.columns:
                cols.append(c)
        out = out[cols]

        # 输出
        out_file = out_dir / f"{symbol}.csv"
        out.to_csv(out_file, index=False)
        return symbol, None

    except Exception as e:
        return symbol, f"{type(e).__name__}: {e}\n{traceback.format_exc()}"


# def run_qlib_dump(source_dir: Path, qlib_dir: Path, freq_key: str) -> None:
#     """
#     调用 Qlib 官方脚本将标准化 CSV 转为 .bin。
#     需确保已安装 pyqlib： pip install pyqlib
#     """
#     qlib_freq = VALID_FREQS.get(freq_key, "day")
#     cmd = [
#         sys.executable, "-m", "qlib.scripts.dump_bin", "dump_all",
#         "--csv_path", str(source_dir),
#         "--qlib_dir", str(qlib_dir),
#         "--freq", qlib_freq,
#         "--date_field_name", "date",
#         "--symbol_field_name", "symbol",
#         "--exclude_fields", "symbol",
#     ]
#     print("[Qlib] 执行：", " ".join(cmd))
#     try:
#         subprocess.run(cmd, check=True)
#         print("[Qlib] dump_all 成功，数据已写入：", qlib_dir)
#     except subprocess.CalledProcessError as e:
#         print("\n[Qlib] dump_all 失败，请在终端单独执行上面命令查看详细报错。")
#         raise e
def run_qlib_dump(source_dir: Path, qlib_dir: Path, freq_key: str, n_jobs: int) -> None:
    # 1st: pip 源码版/带 scripts 的 pyqlib
    try:
        from qlib.scripts.dump_bin import DumpDataAll
    except Exception:
        # 2nd: 项目本地的 scripts/dump_bin.py（你可以把官方脚本放进仓库的 scripts/ 目录）
        try:
            from scripts.dump_bin import DumpDataAll   # 注意：这是你仓库里的 scripts 目录
        except Exception as e:
            raise ImportError(
                "找不到 DumpDataAll。请采用以下任一做法：\n"
                "A) pip 安装源码版 pyqlib（包含 qlib.scripts）\n"
                "B) 把官方 scripts/dump_bin.py 放到你项目的 scripts/ 目录，并重试"
            ) from e

    qlib_freq = {"1h":"60min"}.get(freq_key, freq_key)
    include_fields = "open,high,low,close,volume,factor,vwap"

    dump = DumpDataAll(
        data_path=str(source_dir),   # ← 这里从 csv_path 改为 data_path
        qlib_dir=str(qlib_dir),
        freq=qlib_freq,
        date_field_name="date",
        symbol_field_name="symbol",
        include_fields=include_fields,
    )

    for attr in ("works", "max_workers"):
        if hasattr(dump, attr):
            setattr(dump, attr, int(max(1, n_jobs)))
    print(f"[Qlib] DumpDataAll >>> csv_path={source_dir} qlib_dir={qlib_dir} freq={qlib_freq}")
    dump.dump()

# --------------------------------
# 主流程
# --------------------------------
def main():
    ap = argparse.ArgumentParser(description="Normalize CSVs and build Qlib US data (bin).")
    ap.add_argument("--in_dir", default="~/data/sp500", help="Input per-ticker CSV dir.")
    ap.add_argument("--source_dir", default="~/.qlib/source/us_from_yf", help="Normalized CSV dir (intermediate).")
    ap.add_argument("--qlib_dir", default="~/.qlib/qlib_data/us_data", help="Output qlib .bin dir.")
    ap.add_argument("--freq", default="day",
                    choices=list(VALID_FREQS.keys()),
                    help="Frequency: day / 1min / 5min / 15min / 30min / 60min / 1h")
    ap.add_argument("--n_jobs", type=int, default=8, help="并行处理 CSV 的进程数")
    ap.add_argument("--skip_dump_bin", action="store_true", help="只做标准化，不执行 qlib dump_bin")
    args = ap.parse_args()

    in_dir = Path(os.path.expanduser(args.in_dir)).resolve()
    source_dir = expanduser_mkdir(args.source_dir)
    qlib_dir = expanduser_mkdir(args.qlib_dir)
    freq_key = args.freq

    if not in_dir.exists():
        print(f"[ERROR] 输入目录不存在：{in_dir}")
        sys.exit(1)

    csv_files = sorted([p for p in in_dir.glob("*.csv") if p.is_file()])
    if not csv_files:
        print(f"[WARN] 未在 {in_dir} 找到任何 CSV 文件。")
        sys.exit(0)

    print(f"[INFO] 标准化 {len(csv_files)} 个标的文件，频率={freq_key}")
    successes, failures = 0, []

    # 并行处理
    with futures.ProcessPoolExecutor(max_workers=max(1, args.n_jobs)) as ex:
        tasks = [ex.submit(normalize_one_csv, p, source_dir, freq_key) for p in csv_files]
        for fut in futures.as_completed(tasks):
            symbol, err = fut.result()
            if err is None:
                successes += 1
                if successes % 50 == 0:
                    print(f"[OK] 已完成 {successes} 个…（示例：{symbol}）")
            else:
                failures.append((symbol, err))
                print(f"[FAIL] {symbol}: {err.splitlines()[0]}")

    print(f"[SUMMARY] 成功 {successes}；失败 {len(failures)}；输出：{source_dir}")
    if failures:
        print("失败样例（首行错误）：")
        for sym, err in failures[:20]:
            print(f"  - {sym}: {err.splitlines()[0]}")
        if len(failures) > 20:
            print(f"  ... 其余 {len(failures) - 20} 个省略")

    # dump_bin
    if not args.skip_dump_bin:
        try:
            run_qlib_dump(source_dir, qlib_dir, freq_key)
        except Exception:
            sys.exit(2)

if __name__ == "__main__":
    main()
