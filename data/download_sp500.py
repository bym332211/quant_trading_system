#!/usr/bin/env python3
"""
download_sp500.py
- 读取 tickers 文本（每行一个）
- yfinance 下载：原始(未复权) + 自动复权(OHLC)
- 并发下载、失败重试
- 可选生成合并面板（各票 adj_close 宽表）

依赖：
  pip install -U yfinance pandas numpy tqdm
"""
from __future__ import annotations

import os
import sys
import time
import argparse
from typing import List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import yfinance as yf
from tqdm import tqdm


def read_tickers(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        raw = [line.strip() for line in f]
    seen, out = set(), []
    for s in raw:
        if not s or s.startswith("#"):
            continue
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def sanitize_filename(ticker: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in ticker)


def select_col(df: pd.DataFrame, name: str) -> Optional[pd.Series]:
    """稳健地从 df 中取出名为 name 的列；若得到的是 DataFrame，则取第一列并转为 Series。"""
    if name not in df.columns:
        # MultiIndex/重复列场景：尽量用 equals 判断第一层匹配
        try:
            if hasattr(df.columns, "get_level_values"):
                lvl0 = df.columns.get_level_values(0)
                if name in set(lvl0):
                    sub = df.xs(name, axis=1, level=0, drop_level=False)
                    if isinstance(sub, pd.DataFrame):
                        return sub.iloc[:, 0].rename(name)
                    return sub.rename(name)
        except Exception:
            return None
        return None

    colobj = df[name]
    if isinstance(colobj, pd.DataFrame):
        # 有重复列名时，df[name] 可能还是 DataFrame
        colobj = colobj.iloc[:, 0]
    # 确保索引与df一致
    colobj = pd.Series(colobj, index=df.index, name=name)
    return colobj

def download_one(
    ticker: str, start: str, end: str, interval: str, outdir: str,
    retries: int = 3, sleep: float = 1.0
) -> Tuple[str, bool, Optional[str], Optional[str]]:
    """支持不同时间级别的下载：raw(未复权) + adjusted(自动复权)，合并后落盘"""
    os.makedirs(outdir, exist_ok=True)
    csv_path = os.path.join(outdir, f"{sanitize_filename(ticker)}_{interval}.csv")

    for attempt in range(1, retries + 1):
        try:
            # 高频数据限制：只能下载最近几天的数据
            if interval in ["1m", "15m", "1h", "4h"]:
                end = pd.to_datetime("today").strftime("%Y-%m-%d")  # 结束日期为今天
                start = pd.to_datetime("today") - pd.Timedelta(days=7)  # 仅下载最近7天的数据

            # 下载原始数据（未复权）
            df_raw = yf.download(
                ticker, start=start, end=end, interval=interval,
                auto_adjust=False, actions=True, progress=False, threads=False,
                group_by="column"
            )
            if df_raw is None or df_raw.empty:
                raise RuntimeError("Empty df_raw")

            # 下载复权数据
            df_adj = yf.download(
                ticker, start=start, end=end, interval=interval,
                auto_adjust=True, actions=True, progress=False, threads=False,
                group_by="column"
            )
            if df_adj is None or df_adj.empty:
                # 若没有复权数据，则使用原始数据
                df_adj = df_raw[["Open", "High", "Low", "Close"]].copy()

            # 对齐索引并构建输出
            idx = df_raw.index.union(df_adj.index).sort_values()
            df_raw = df_raw.reindex(idx)
            df_adj = df_adj.reindex(idx)

            out = pd.DataFrame(index=idx)

            # 原始列（未复权）
            for col in ["Open", "High", "Low", "Close", "Adj Close", "Volume", "Dividends", "Stock Splits"]:
                s = select_col(df_raw, col)
                if s is not None:
                    out[col.replace(" ", "_").lower()] = s

            # 复权后的 OHLC
            for col in ["Open", "High", "Low", "Close"]:
                s = select_col(df_adj, col)
                if s is not None:
                    out[f"adj_{col.lower()}"] = s

            out.index.name = "Date"
            out.dropna(how="all", inplace=True)
            out.to_csv(csv_path)
            return ticker, True, csv_path, None

        except Exception as e:
            if attempt >= retries:
                return ticker, False, None, f"{type(e).__name__}: {e}"
            time.sleep(sleep * attempt)

    return ticker, False, None, "Unknown error"




def build_merged_csv(success_items, merged_csv: str) -> None:
    frames = []
    for ticker, ok, path, _ in success_items:
        if not ok or not path:
            continue
        try:
            df = pd.read_csv(path, parse_dates=["Date"]).set_index("Date")
            col = "adj_close" if "adj_close" in df.columns else "close"
            frames.append(df[col].rename(ticker))
        except Exception:
            pass
    if not frames:
        return
    panel = pd.concat(frames, axis=1).sort_index()
    os.makedirs(os.path.dirname(merged_csv) or ".", exist_ok=True)
    panel.to_csv(merged_csv)


def main():
    ap = argparse.ArgumentParser(description="Download Yahoo Finance data for a list of tickers.")
    ap.add_argument("--symbols-file", default="sp500_tickers.txt")
    ap.add_argument("--start", default="2000-01-01")
    ap.add_argument("--end", default="2100-01-01")
    ap.add_argument("--outdir", default="data/sp500")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--merged-csv", default="")
    ap.add_argument("--interval", default="1d", choices=["1d", "1wk", "1mo", "1m", "5m", "15m", "1h", "4h"], help="Data interval.")

    args = ap.parse_args()

    tickers = read_tickers(args.symbols_file)
    if not tickers:
        print(f"No tickers found in {args.symbols_file}", file=sys.stderr)
        sys.exit(2)

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(download_one, t, args.start, args.end, args.interval, args.outdir) for t in tickers]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="Downloading"):
            results.append(fut.result())

    ok = [r for r in results if r[1]]
    fail = [r for r in results if not r[1]]
    print(f"Done. Success: {len(ok)}, Fail: {len(fail)}")
    if fail:
        for t, _, _, e in fail[:10]:
            print(f"  Fail {t}: {e}", file=sys.stderr)

    if args.merged_csv:
        build_merged_csv(ok, args.merged_csv)
        print("Merged CSV:", args.merged_csv)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)

