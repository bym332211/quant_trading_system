#!/usr/bin/env python3
# make_sp500_list.py — 生成 S&P 500 列表（原始符号 & Yahoo风格），并额外加入 SPY
import os
import re
import sys
import time
import pandas as pd
from io import StringIO

# 为了尽量避免证书问题，优先用 certifi
try:
    import certifi
    CERT_PATH = certifi.where()
except Exception:
    CERT_PATH = True  # 退化为系统默认

import requests

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
BACKUP_URLS = [
    # 这些备份源可能略滞后，但足够生成列表；按顺序尝试
    "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv",
    "https://datahub.io/core/s-and-p-500-companies/r/constituents.csv",
]

OUT_ORIGINAL = "sp500_symbols_original.txt"  # 原始符号（保留点号，如 BRK.B）
OUT_TICKERS  = "sp500_tickers.txt"           # 常用数据源符号（点号替换为短横，如 BRK-B）

UA = {"User-Agent": "Mozilla/5.0 (compatible; sp500-list-generator/1.0)"}

def _clean_original(sym: str) -> str:
    # 保留 A-Z 0-9 . - ；去掉其它字符；转大写
    s = re.sub(r"[^A-Za-z0-9.\-]", "", str(sym).strip()).upper()
    return s

def _to_yahoo(sym: str) -> str:
    # 将原始符号转换为 Yahoo 风格（. → -）
    s = _clean_original(sym).replace(".", "-")
    # 再做一次清洗，确保只有 A-Z0-9 和 -
    s = re.sub(r"[^A-Z0-9\-]", "", s)
    return s

def fetch_from_wikipedia() -> list[str]:
    # 先用 requests 拿到 HTML，再用 pandas 解析，避免 pandas.read_html 直连的证书问题
    resp = requests.get(WIKI_URL, headers=UA, timeout=20, verify=CERT_PATH)
    resp.raise_for_status()
    tables = pd.read_html(StringIO(resp.text))
    # 找包含 Symbol 列的表
    sym_col = None
    candidates = []
    for t in tables:
        cols = [str(c).strip().lower() for c in t.columns]
        if any(c in ("symbol", "ticker", "ticker symbol") for c in cols):
            candidates.append(t)
    if not candidates:
        raise RuntimeError("No table with Symbol column found on Wikipedia page.")
    # 取第一张符合的表
    df = candidates[0]
    # 兼容不同列名
    for col in df.columns:
        if str(col).strip().lower() in ("symbol", "ticker", "ticker symbol"):
            sym_col = col
            break
    if sym_col is None:
        raise RuntimeError("Symbol column not found after filtering.")
    syms = [str(x) for x in df[sym_col].dropna().tolist()]
    return syms

def fetch_from_backups() -> list[str]:
    last_err = None
    for url in BACKUP_URLS:
        try:
            resp = requests.get(url, headers=UA, timeout=20, verify=CERT_PATH)
            resp.raise_for_status()
            # pandas>=2.0 移除了 pd.compat.StringIO；统一使用标准库 io.StringIO
            df = pd.read_csv(StringIO(resp.text))
            # 常见列名兼容
            col = None
            for c in df.columns:
                if str(c).strip().lower() in ("symbol", "ticker", "ticker symbol"):
                    col = c
                    break
            if col is None:
                continue
            return [str(x) for x in df[col].dropna().tolist()]
        except Exception as e:
            last_err = e
            time.sleep(1)
    if last_err:
        raise last_err
    return []

def get_sp500_symbols() -> list[str]:
    # 优先 Wikipedia，失败再试备用源
    try:
        return fetch_from_wikipedia()
    except Exception as e:
        print(f"[WARN] fetch Wikipedia failed: {e}", file=sys.stderr)
        print("[INFO] trying backup sources ...", file=sys.stderr)
        return fetch_from_backups()

def main():
    syms_raw = get_sp500_symbols()
    if not syms_raw:
        print("[ERROR] failed to fetch S&P 500 symbols from all sources.", file=sys.stderr)
        sys.exit(1)

    # 清洗 & 去重
    original_set = set()
    ticker_set   = set()
    for s in syms_raw:
        o = _clean_original(s)
        if not o:
            continue
        t = _to_yahoo(o)
        if not t:
            continue
        original_set.add(o)
        ticker_set.add(t)

    # ★ 额外加入基准：SPY
    original_set.add("SPY")
    ticker_set.add("SPY")

    original = sorted(original_set)
    tickers  = sorted(ticker_set)

    with open(OUT_ORIGINAL, "w", encoding="utf-8") as f:
        f.write("\n".join(original) + "\n")
    with open(OUT_TICKERS, "w", encoding="utf-8") as f:
        f.write("\n".join(tickers) + "\n")

    print(f"Wrote {OUT_ORIGINAL}  ({len(original)} lines)")
    print(f"Wrote {OUT_TICKERS}   ({len(tickers)} lines)")
    print("Example (first 10 tickers):", tickers[:10])

if __name__ == "__main__":
    main()

