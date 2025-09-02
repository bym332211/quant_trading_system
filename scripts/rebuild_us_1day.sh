#!/usr/bin/env bash
set -euo pipefail

RAW_DIR="$HOME/workspace/quant_trading_system/data/data/sp500"
SOURCE_DIR="$HOME/.qlib/source/us_from_yf"
QLIB_DIR="$HOME/.qlib/qlib_data/us_data"
N_JOBS=8
CLEAN_MODE="backup"
PY="${PYTHON:-/home/ec2-user/.pyenv/versions/3.12.5/bin/python}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
[ -x "$PY" ] || PY="$(command -v python)"
ts(){ date +%Y%m%d_%H%M%S; }
backup_or_clean(){ local d="$1"; if [ -e "$d" ]; then
  if [ "$CLEAN_MODE" = "delete" ]; then rm -rf "$d"; else mv "$d" "${d}.bak.$(ts)"; fi; fi; mkdir -p "$d"; }

echo "== Clean =="
backup_or_clean "$SOURCE_DIR"; backup_or_clean "$QLIB_DIR"

echo "== Normalize (freq=day) =="
"$PY" "$REPO_ROOT/data/build_qlib_us.py" \
  --in_dir "$RAW_DIR" --source_dir "$SOURCE_DIR" --qlib_dir "$QLIB_DIR" \
  --freq day --n_jobs "$N_JOBS" --skip_dump_bin

echo "== Dump (day) =="
"$PY" "$REPO_ROOT/scripts/dump_bin.py" dump_all \
  --data_path "$SOURCE_DIR" --qlib_dir "$QLIB_DIR" --freq day \
  --date_field_name date --symbol_field_name symbol \
  --include_fields open,high,low,close,volume,factor,vwap --max_workers "$N_JOBS"

[ -d "$QLIB_DIR/features" ] || { echo "[FATAL] 未找到 $QLIB_DIR/features"; exit 2; }

echo "== Rebuild instruments（*.day.bin） =="
env QLIB_DIR="$QLIB_DIR" "$PY" - <<'PY'
from pathlib import Path
import os
qlib_dir = Path(os.environ["QLIB_DIR"]); feat_dir = qlib_dir/"features"
cands = [p for p in feat_dir.iterdir() if p.is_dir() and p.name.lower() not in {"day","1min","minute"}]
keep=[]
for d in cands:
    if any(d.glob("*.day.bin")): keep.append(d.name)
inst = qlib_dir/"instruments"; inst.mkdir(parents=True, exist_ok=True)
with open(inst/"all.txt","w") as f:
    for name in sorted(set(keep)): f.write(name.upper()+"\n")
print(f"[OK] instruments/all.txt 写入 {len(keep)} 个标的")
PY

echo "== Sanity read (freq='day') =="
env QLIB_DIR="$QLIB_DIR" "$PY" - <<'PY'
import os, qlib
from qlib.data import D
from pathlib import Path
QLIB_DIR=os.environ["QLIB_DIR"]; qlib.init(provider_uri=QLIB_DIR, region="us")
symbols=[l.strip() for l in open(Path(QLIB_DIR)/"instruments"/"all.txt") if l.strip()][:2]
print("Using tickers:", symbols)
df=D.features(symbols,["$close","$open","$volume"],start_time="2019-01-02",end_time="2019-01-15",freq="day")
print(df.head()); print("rows:",len(df))
PY

# =========================
# Step X: 复权校验（OK/NG）
# =========================
echo "== Verify adjusted prices (day) =="

set +e  # 我们手动接管返回码
env QLIB_DIR="$QLIB_DIR" SOURCE_DIR="$SOURCE_DIR" "$PY" - <<'PY'
# 校验逻辑：
# - 从 instruments/all.txt 取前 N 个标的（默认 5 个）
# - 读取其标准化 CSV（open/high/low/close/vwap 已为复权价，factor=1.0）
# - 同时间窗从 Qlib 读取 $open/$high/$low/$close/$vwap/$factor
# - 对齐时间并比较，打印每票 OK/FAIL；最终打印 “复权校验结果: OK/NG”
from pathlib import Path
import os, sys
import pandas as pd
import numpy as np
import qlib
from qlib.data import D

QLIB_DIR = Path(os.environ["QLIB_DIR"]).expanduser().resolve()
SRC_DIR  = Path(os.environ["SOURCE_DIR"]).expanduser().resolve()
SAMPLE_N = int(os.environ.get("VERIFY_SAMPLE", "5"))
ROWS_N   = int(os.environ.get("VERIFY_ROWS", "1000"))

def pick_symbols():
    inst = QLIB_DIR / "instruments" / "all.txt"
    syms=[]
    if inst.exists():
        syms=[l.strip() for l in inst.read_text().splitlines() if l.strip()]
    if not syms:
        syms=[p.stem.upper() for p in sorted(SRC_DIR.glob("*.csv"))]
    return syms[:SAMPLE_N]

def read_csv_tail(sym: str, n: int) -> pd.DataFrame:
    fp = SRC_DIR / f"{sym}.csv"
    df = pd.read_csv(fp)
    df.columns=[c.strip().lower().replace(" ","_") for c in df.columns]
    if "date" not in df.columns: raise RuntimeError(f"{fp} 缺少 date 列")
    dt = pd.to_datetime(df["date"], utc=True, errors="coerce").dt.tz_convert("UTC").dt.tz_localize(None)
    df["date"]=dt
    df=df.sort_values("date").tail(n)
    return df[["date","open","high","low","close","vwap","volume"]]

def cmp(a: pd.Series, b: pd.Series, atol=1e-4, rtol=1e-4):
    a=a.astype(float); b=b.astype(float)
    abs_err=(a-b).abs()
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_err = abs_err/np.maximum(b.abs(),1e-12)
    ok = bool((abs_err<=atol).all() or (rel_err<=rtol).all())
    return ok, float(abs_err.max() if len(abs_err) else 0.0), float(rel_err.max() if len(rel_err) else 0.0)

qlib.init(provider_uri=str(QLIB_DIR), region="us")
symbols = pick_symbols()
if not symbols:
    print("[NG] 无可用标的")
    sys.exit(3)

# 时间窗取第一票 CSV 的最后 ROWS_N 行
csv0 = read_csv_tail(symbols[0], ROWS_N)
if csv0.empty:
    print("[NG] CSV 无数据")
    sys.exit(3)

start = csv0["date"].iloc[0].strftime("%Y-%m-%d")
end   = csv0["date"].iloc[-1].strftime("%Y-%m-%d")

fields=["$open","$high","$low","$close","$vwap","$factor"]
qdf = D.features(symbols, fields, start_time=start, end_time=end, freq="day")
if qdf.empty:
    print("[NG] Qlib 读取为空")
    sys.exit(3)
qdf = qdf.reset_index().rename(columns={"instrument":"symbol","datetime":"date"})
qdf["date"]=pd.to_datetime(qdf["date"], utc=True).dt.tz_convert("UTC").dt.tz_localize(None)
overall_ok=True

for sym in symbols:
    csv = read_csv_tail(sym, ROWS_N)
    qsub = qdf[qdf["symbol"]==sym].copy()
    # 对齐交集
    common = pd.Index(sorted(set(csv["date"]).intersection(set(qsub["date"]))))
    if len(common)==0:
        print(f"[{sym}] 无公共时间交集 -> SKIP")
        continue
    csv = csv.set_index("date").loc[common].sort_index()
    qsub = qsub.set_index("date").loc[common].sort_index()

    # factor==1 检查
    fac = qsub["$factor"].dropna().astype(float).round(8).unique().tolist()
    fac_ok = (len(fac)==0) or (len(fac)==1 and fac[0]==1.0)
    sym_ok = fac_ok
    print(f"[{sym}] factor唯一值: {fac} -> {'OK' if fac_ok else 'NG'}")

    for name, qcol, ccol in [
        ("open",  qsub["$open"],  csv["open"]),
        ("high",  qsub["$high"],  csv["high"]),
        ("low",   qsub["$low"],   csv["low"]),
        ("close", qsub["$close"], csv["close"]),
        ("vwap",  qsub["$vwap"],  csv["vwap"]),
    ]:
        ok, amax, rmax = cmp(qcol, ccol)
        print(f"[{sym}] {name:<5} -> {'OK ' if ok else 'NG '} | max_abs_err={amax:.6g} max_rel_err={rmax:.6g}")
        sym_ok = sym_ok and ok

    print(f"[{sym}] 校验结果 => {'OK ✅' if sym_ok else 'NG ❌'}  (rows={len(common)})\n" + "-"*60)
    overall_ok = overall_ok and sym_ok

print(f"复权校验结果: {'OK ✅' if overall_ok else 'NG ❌'}")
sys.exit(0 if overall_ok else 3)
PY
RC=$?
set -e

if [ $RC -ne 0 ]; then
  echo "[FATAL] 复权校验结果: NG ❌"
  exit $RC
else
  echo "[OK] 复权校验结果: OK ✅"
fi

echo "[DONE] 1day 重建完成：$QLIB_DIR"
