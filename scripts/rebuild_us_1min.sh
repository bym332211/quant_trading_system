#!/usr/bin/env bash
set -euo pipefail

RAW_DIR="$HOME/workspace/quant_trading_system/data/data/sp500_1min"
SOURCE_DIR="$HOME/.qlib/source/us_from_yf_1min"
QLIB_DIR="$HOME/.qlib/qlib_data/us_data_1min"
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
rm -f "$QLIB_DIR/calendars/5min.txt" "$QLIB_DIR/features/5min" 2>/dev/null || true

echo "== Normalize (freq=1min) =="
"$PY" "$REPO_ROOT/data/build_qlib_us.py" \
  --in_dir "$RAW_DIR" --source_dir "$SOURCE_DIR" --qlib_dir "$QLIB_DIR" \
  --freq 1min --n_jobs "$N_JOBS" --skip_dump_bin

echo "== Dump (1min -> fallback minute) =="
set +e
"$PY" "$REPO_ROOT/scripts/dump_bin.py" dump_all \
  --data_path "$SOURCE_DIR" --qlib_dir "$QLIB_DIR" --freq 1min \
  --date_field_name date --symbol_field_name symbol \
  --include_fields open,high,low,close,volume,factor,vwap --max_workers "$N_JOBS"
rc=$?
if [ $rc -ne 0 ]; then
  "$PY" "$REPO_ROOT/scripts/dump_bin.py" dump_all \
    --data_path "$SOURCE_DIR" --qlib_dir "$QLIB_DIR" --freq minute \
    --date_field_name date --symbol_field_name symbol \
    --include_fields open,high,low,close,volume,factor,vwap --max_workers "$N_JOBS"
  rc=$?
fi
set -e
[ $rc -eq 0 ] || { echo "[FATAL] dump_bin 失败"; exit 2; }

echo "== 检查分钟特征文件是否存在 =="
env QLIB_DIR="$QLIB_DIR" "$PY" - <<'PY'
from pathlib import Path
import os
feat = Path(os.environ["QLIB_DIR"])/"features"
if not feat.exists(): raise SystemExit("[FATAL] 未找到 features 目录")
hit = any(feat.rglob("*.1min.bin")) or any(feat.rglob("*.minute.bin"))
print("[OK] 检测到分钟特征文件") if hit else (_ for _ in ()).throw(SystemExit("[FATAL] 未检测到 *.1min.bin / *.minute.bin"))
PY

CALDIR="$QLIB_DIR/calendars"; mkdir -p "$CALDIR"
[ -f "$CALDIR/minute.txt" ] && [ ! -f "$CALDIR/1min.txt" ] && ln -s minute.txt "$CALDIR/1min.txt" || true

echo "== Rebuild instruments =="
env QLIB_DIR="$QLIB_DIR" "$PY" - <<'PY'
from pathlib import Path
import os
qlib_dir = Path(os.environ["QLIB_DIR"]); feat_dir = qlib_dir/"features"
cands = [p for p in feat_dir.iterdir() if p.is_dir() and p.name.lower() not in {"day","1min","minute"}]
keep=[]
for d in cands:
    if any(d.glob("*.1min.bin")) or any(d.glob("*.minute.bin")): keep.append(d.name)
inst = qlib_dir/"instruments"; inst.mkdir(parents=True, exist_ok=True)
with open(inst/"all.txt","w") as f:
    for name in sorted(set(keep)): f.write(name.upper()+"\n")
print(f"[OK] instruments/all.txt 写入 {len(keep)} 个标的")
PY

echo "== Sanity read (freq='1min') =="
env QLIB_DIR="$QLIB_DIR" "$PY" - <<'PY'
import os, qlib
from qlib.data import D
from pathlib import Path
QLIB_DIR=os.environ["QLIB_DIR"]; qlib.init(provider_uri=QLIB_DIR, region="us")
symbols=[l.strip() for l in open(Path(QLIB_DIR)/"instruments"/"all.txt") if l.strip()][:2]
print("Using tickers:", symbols)
cal=[l.strip() for l in open(Path(QLIB_DIR)/"calendars"/"1min.txt") if l.strip()]
start,end=cal[max(0,len(cal)-200)],cal[-1]
df=D.features(symbols,["$close","$open","$volume"],start_time=start,end_time=end,freq="1min")
print(df.head()); print(df.tail()); print("rows:",len(df))
PY

echo "[DONE] 1min 源重建完成：$QLIB_DIR"
