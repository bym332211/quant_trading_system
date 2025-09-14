#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_instrument_sector.py
从 Qlib instruments/all.txt 读取美股代码，使用 yfinance 获取 sector/industry，
生成 data/instrument_sector.csv（列：instrument,sector,industry,source,ts）。

用法：
  python data/make_instrument_sector.py \
    --qlib_dir ~/.qlib/qlib_data/us_data \
    --out data/instrument_sector.csv \
    --sleep 0.6 --sample 0
"""

from __future__ import annotations
import argparse, time, sys
from pathlib import Path
import pandas as pd
from datetime import datetime
import requests
try:
    import certifi
    _CERT_PATH = certifi.where()
except Exception:
    _CERT_PATH = True

# 兼容 Py3.8+：优先使用 datetime.UTC（3.11+），否则回退到 timezone.utc
try:
    from datetime import UTC as _UTC
except Exception:
    from datetime import timezone as _tz
    _UTC = _tz.utc

def norm_str(x):
    if x is None: return None
    s = str(x).strip()
    return s if s else None

def fetch_sector_industry(ticker: str):
    import yfinance as yf
    try:
        tk = yf.Ticker(ticker)
        info = tk.get_info()  # 可能较慢；遇到429时适当sleep
        sector   = info.get("sector") or info.get("sectorKey")
        industry = info.get("industry") or info.get("industryKey")
        return norm_str(sector), norm_str(industry), "yfinance"
    except Exception as e:
        return None, None, f"error:{type(e).__name__}"

def _fallback_sector_from_sp500() -> dict:
    """Fallback sector mapping using public S&P500 constituents dataset.
    Returns dict {SYMBOL_UPPER: SECTOR}.
    """
    urls = [
        "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv",
        "https://datahub.io/core/s-and-p-500-companies/r/constituents.csv",
    ]
    for u in urls:
        try:
            r = requests.get(u, headers={"User-Agent": "sector-fetch/1.0"}, timeout=15, verify=_CERT_PATH)
            r.raise_for_status()
            import io
            df = pd.read_csv(io.StringIO(r.text))
            cols = {c.lower(): c for c in df.columns}
            sym_col = cols.get("symbol") or cols.get("ticker") or list(df.columns)[0]
            sec_col = cols.get("sector") or cols.get("gics sector") or cols.get("gics_sector")
            if not sym_col or not sec_col:
                continue
            d = {}
            for _, row in df.iterrows():
                sym = str(row[sym_col]).strip().upper()
                sec = norm_str(row.get(sec_col)) if hasattr(row, 'get') else norm_str(row[sec_col])
                if not sym or not sec:
                    continue
                d[sym] = sec
                d[sym.replace(".", "-")] = sec
                d[sym.replace("-", ".")] = sec
            if d:
                return d
        except Exception:
            continue
    return {}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qlib_dir", required=True)
    ap.add_argument("--out", default="data/instrument_sector.csv")
    ap.add_argument("--sample", type=int, default=0, help="仅抓前N个；0=全部")
    ap.add_argument("--sleep", type=float, default=0.6, help="每次请求之间的秒数（防限流）")
    args = ap.parse_args()

    inst_file = Path(args.qlib_dir).expanduser().resolve() / "instruments" / "all.txt"
    if not inst_file.exists():
        print(f"[ERR] instruments 文件不存在: {inst_file}", file=sys.stderr)
        sys.exit(1)
    # all.txt format: CODE\tSTART\tEND -> only take first column (CODE)
    raw_lines = [l.strip() for l in inst_file.read_text(encoding="utf-8", errors="ignore").splitlines() if l.strip()]
    codes = []
    seen = set()
    for l in raw_lines:
        c = l.split("\t")[0].strip().upper()
        if c and c not in seen:
            seen.add(c)
            codes.append(c)
    symbols = codes
    if args.sample > 0:
        symbols = symbols[:args.sample]

    rows = []
    t0 = time.time()
    sp500_map = _fallback_sector_from_sp500()
    for i, sym in enumerate(symbols, 1):
        # prefer public S&P500 mapping to avoid rate limit
        sec_fb = sp500_map.get(sym)
        sector, industry, src = (sec_fb, None, "sp500") if sec_fb else (None, None, None)
        if not sector:
            sector, industry, src = fetch_sector_industry(sym)
            if not sector and sec_fb:
                sector = sec_fb
                src = (src + "+sp500") if src else "sp500"
        # 修正：使用 timezone-aware 的 UTC 时间；规范成 RFC3339 的 'Z' 结尾
        ts = datetime.now(_UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        rows.append({
            "instrument": sym,
            "sector": sector if sector else "Unknown",
            "industry": industry if industry else "Unknown",
            "source": src,
            "ts": ts,
        })
        if i % 25 == 0:
            print(f"[{i}/{len(symbols)}] last={sym} sector={sector} industry={industry}")
        time.sleep(max(0.0, args.sleep))

    df = pd.DataFrame(rows)
    outp = Path(args.out).expanduser().resolve()
    outp.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(outp, index=False)
    ok = (df["sector"]!="Unknown").mean()
    print(f"[DONE] wrote {outp} (tickers={len(df)}, sector_known={ok:.0%}, cost={time.time()-t0:.1f}s)")

if __name__ == "__main__":
    main()
