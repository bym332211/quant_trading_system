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
    symbols = [l.strip().upper() for l in inst_file.read_text().splitlines() if l.strip()]
    if args.sample > 0:
        symbols = symbols[:args.sample]

    rows = []
    t0 = time.time()
    for i, sym in enumerate(symbols, 1):
        sector, industry, src = fetch_sector_industry(sym)
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
