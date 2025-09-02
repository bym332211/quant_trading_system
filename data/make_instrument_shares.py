#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_instrument_shares.py
从 Qlib instruments/all.txt 读取美股代码，用 yfinance 抓“历史流通股本”并固化成 CSV：
  instrument,date,shares,source,ts

特点
- 优先使用 yfinance.Ticker.get_shares_full() 拿“历史时间序列”；
- 若拿不到，回退到 fast_info.shares_outstanding / get_info()['sharesOutstanding']（单值）；
- 可断点续跑（--resume，会跳过已抓到的 instrument）；
- 防限流：--sleep 控制每次请求间隔；--workers 建议保持 1（或很小）。

用法：
  python data/make_instrument_shares.py \
    --qlib_dir ~/.qlib/qlib_data/us_data \
    --out data/instrument_shares.csv \
    --sleep 0.6 --retries 3 --resume
"""

from __future__ import annotations
import argparse, sys, time
from pathlib import Path
from datetime import datetime

import pandas as pd

try:
    import yfinance as yf
except Exception as e:
    print("请先: pip install yfinance --upgrade", file=sys.stderr)
    raise

# 兼容 Py3.8+ 的 UTC
try:
    from datetime import UTC as _UTC
except Exception:
    from datetime import timezone as _tz
    _UTC = _tz.utc


def _utc_now_iso() -> str:
    return datetime.now(_UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _load_instruments(qlib_dir: str) -> list[str]:
    inst_file = Path(qlib_dir).expanduser().resolve() / "instruments" / "all.txt"
    if not inst_file.exists():
        raise FileNotFoundError(f"未找到 instruments 文件: {inst_file}")
    syms = [l.strip().upper() for l in inst_file.read_text().splitlines() if l.strip()]
    return syms


def _normalize_shares_df(df: pd.DataFrame, sym: str, source: str) -> pd.DataFrame:
    """
    将 yfinance 返回的 shares 表规整为: instrument,date,shares,source,ts
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["instrument","date","shares","source","ts"])

    # 常见列名兼容
    cols = {c.lower(): c for c in df.columns}
    if "shares outstanding" in cols:
        shares_col = cols["shares outstanding"]
    elif "shares_outstanding" in cols:
        shares_col = cols["shares_outstanding"]
    elif "shares" in cols:
        shares_col = cols["shares"]
    else:
        # 可能是 Series
        if hasattr(df, "name") and ("share" in str(getattr(df, "name", "")).lower()):
            df = df.to_frame(name="shares")
            shares_col = "shares"
        else:
            # 最后尝试将唯一一列当 shares
            if df.shape[1] == 1:
                shares_col = df.columns[0]
            else:
                return pd.DataFrame(columns=["instrument","date","shares","source","ts"])

    out = df.copy()
    # 处理 index 为日期的情况
    if "Date" in out.columns:
        out["date"] = pd.to_datetime(out["Date"], utc=False)
    elif out.index.name and "date" in out.index.name.lower():
        out = out.reset_index().rename(columns={out.index.name: "date"})
    elif not ("date" in out.columns):
        # 如果 index 是 DatetimeIndex
        if isinstance(out.index, pd.DatetimeIndex):
            out = out.reset_index().rename(columns={"index": "date"})
        else:
            # 无日期列情况下，无法拼 as-of；直接返回空
            return pd.DataFrame(columns=["instrument","date","shares","source","ts"])

    out = out[["date", shares_col]].rename(columns={shares_col: "shares"})
    out["instrument"] = sym
    out["source"] = source
    out["ts"] = _utc_now_iso()

    out["date"] = pd.to_datetime(out["date"], utc=False)
    out["shares"] = pd.to_numeric(out["shares"], errors="coerce")
    out = out.dropna(subset=["date", "shares"])
    out = out[out["shares"] > 0]
    out = out[["instrument","date","shares","source","ts"]]
    out = out.sort_values(["instrument","date"]).drop_duplicates(["instrument","date"], keep="last")
    return out


def fetch_shares_for_symbol(sym: str, retries: int = 3, sleep: float = 0.6) -> pd.DataFrame:
    """
    依次尝试：
      1) Ticker.get_shares_full(start="1990-01-01")
      2) 单值 shares：fast_info.shares_outstanding 或 get_info()['sharesOutstanding']
    返回统一列：instrument,date,shares,source,ts
    """
    err_last = None
    for _ in range(max(1, retries)):
        try:
            tk = yf.Ticker(sym)
            # 1) 历史时间序列
            hist = None
            if hasattr(tk, "get_shares_full"):
                try:
                    hist = tk.get_shares_full(start="1990-01-01")
                except Exception:
                    hist = None
            if hist is not None and not getattr(hist, "empty", True):
                df = _normalize_shares_df(hist.reset_index(), sym, "yfinance.get_shares_full")
                if not df.empty:
                    return df

            # 2) 单值回退
            val = None
            try:
                val = getattr(tk.fast_info, "shares_outstanding", None)
            except Exception:
                val = None
            if val is None:
                try:
                    info = tk.get_info()
                    val = info.get("sharesOutstanding")
                except Exception:
                    val = None

            if val:
                df = pd.DataFrame(
                    [{"instrument": sym,
                      "date": pd.Timestamp("1990-01-01"),
                      "shares": float(val),
                      "source": "yfinance.single_value",
                      "ts": _utc_now_iso()}]
                )
                return df

            # 若都没有，记错误后重试
            err_last = RuntimeError("no shares from yfinance")
        except Exception as e:
            err_last = e
        time.sleep(max(0.0, sleep))

    # 最终失败 → 返回空表（由上层统计）
    return pd.DataFrame(columns=["instrument","date","shares","source","ts"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qlib_dir", required=True)
    ap.add_argument("--out", default="data/instrument_shares.csv")
    ap.add_argument("--tickers_file", default="", help="可选：自定义 tickers 列表，每行一个")
    ap.add_argument("--sample", type=int, default=0, help="仅抓前 N 个；0=全部")
    ap.add_argument("--sleep", type=float, default=0.6, help="每个请求之间的秒数（防限流）")
    ap.add_argument("--retries", type=int, default=3, help="单票重试次数")
    ap.add_argument("--resume", action="store_true", help="若已存在 CSV，则跳过其中已抓到的 instrument")
    ap.add_argument("--flush_every", type=int, default=50, help="每抓多少只就落盘一次（避免长跑中断丢数据）")
    args = ap.parse_args()

    # 1) 读取 tickers
    if args.tickers_file and Path(args.tickers_file).expanduser().exists():
        syms = [l.strip().upper() for l in Path(args.tickers_file).expanduser().read_text().splitlines() if l.strip()]
    else:
        syms = _load_instruments(args.qlib_dir)

    if args.sample > 0:
        syms = syms[:args.sample]

    outp = Path(args.out).expanduser().resolve()
    outp.parent.mkdir(parents=True, exist_ok=True)

    # 2) 断点续跑：已抓到的 instrument 直接跳过
    done_syms: set[str] = set()
    if args.resume and outp.exists():
        try:
            prev = pd.read_csv(outp)
            if not prev.empty and {"instrument","date","shares"} <= set(prev.columns):
                done_syms = set(str(x).upper() for x in prev["instrument"].unique())
                print(f"[resume] 检测到已抓 {len(done_syms)} 个 instrument，将跳过这些。")
        except Exception:
            pass

    todo = [s for s in syms if s not in done_syms]
    print(f"[start] 计划抓取 {len(todo)} / {len(syms)} 只标的。输出：{outp}")

    # 3) 主循环
    buf = []
    grabbed = 0
    empty_cnt = 0
    t0 = time.time()
    for i, sym in enumerate(todo, 1):
        df = fetch_shares_for_symbol(sym, retries=args.retries, sleep=args.sleep)
        if df.empty:
            empty_cnt += 1
        else:
            buf.append(df)
            grabbed += 1

        # 进度日志
        if (i % 25 == 0) or (i == len(todo)):
            ok_rate = 0.0 if i == 0 else grabbed / i
            print(f"[{i}/{len(todo)}] last={sym} grabbed={grabbed} empty={empty_cnt} ok_rate={ok_rate:.1%}")

        # 定期落盘
        if (len(buf) >= args.flush_every) or (i == len(todo)):
            if buf:
                part = pd.concat(buf, axis=0, ignore_index=True)
                if outp.exists():
                    old = pd.read_csv(outp)
                    part = pd.concat([old, part], axis=0, ignore_index=True)
                # 去重 & 排序
                part["instrument"] = part["instrument"].astype(str).str.upper()
                part["date"] = pd.to_datetime(part["date"], utc=False)
                part["shares"] = pd.to_numeric(part["shares"], errors="coerce")
                part = part.dropna(subset=["instrument","date","shares"])
                part = part[part["shares"] > 0]
                part = part.sort_values(["instrument","date"]).drop_duplicates(["instrument","date"], keep="last")
                part.to_csv(outp, index=False)
                buf = []
        # 节流
        time.sleep(max(0.0, args.sleep))

    dt = time.time() - t0
    print(f"[done] 写出 {outp} | instruments={len(pd.read_csv(outp)['instrument'].unique()) if outp.exists() else 0} | "
          f"耗时 {dt/60:.1f} 分钟 | 抓取成功 {grabbed}，空 {empty_cnt}")
    

if __name__ == "__main__":
    import time
    main()
