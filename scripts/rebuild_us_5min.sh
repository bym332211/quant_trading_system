#!/usr/bin/env bash
set -euo pipefail

###################### 配置区：按需修改 ######################
RAW_DIR="$HOME/workspace/quant_trading_system/data/data/sp500_5min"   # 5min 源CSV
SOURCE_DIR="$HOME/.qlib/source/us_from_yf_5min"                       # 标准化CSV输出
QLIB_DIR="$HOME/.qlib/qlib_data/us_data_5min"                         # Qlib分钟库
N_JOBS=8
CLEAN_MODE="backup"   # backup 或 delete
PY="${PYTHON:-/home/ec2-user/.pyenv/versions/3.12.5/bin/python}"
#############################################################

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
[ -x "$PY" ] || PY="$(command -v python)"
ts() { date +%Y%m%d_%H%M%S; }

backup_or_clean() {
  local d="$1"
  if [ -e "$d" ]; then
    if [ "$CLEAN_MODE" = "delete" ]; then
      rm -rf "$d"; echo "[CLEAN] deleted $d"
    else
      local bak="${d}.bak.$(ts)"; mv "$d" "$bak"; echo "[BACKUP] moved $d -> $bak"
    fi
  fi
  mkdir -p "$d"
}

echo "== Step 0. 清理旧产物（$CLEAN_MODE） =="
backup_or_clean "$SOURCE_DIR"
backup_or_clean "$QLIB_DIR"
rm -f "$QLIB_DIR/calendars/5min.txt" "$QLIB_DIR/features/5min" 2>/dev/null || true

echo "== Step 1. 标准化 CSV（freq=5min，仅写入 SOURCE_DIR） =="
"$PY" "$REPO_ROOT/data/build_qlib_us.py" \
  --in_dir "$RAW_DIR" \
  --source_dir "$SOURCE_DIR" \
  --qlib_dir "$QLIB_DIR" \
  --freq 5min \
  --n_jobs "$N_JOBS" \
  --skip_dump_bin

echo "== Step 1.1 自检：确认时间戳无 +00:00，文件名无频率后缀 =="
if grep -m1 -n "+00:00" "$SOURCE_DIR"/*.csv >/dev/null 2>&1; then
  echo "[WARN] 标准化CSV仍含 +00:00（建议检查 to_utc_naive）"
else
  echo "[OK] 未检测到 +00:00 时区后缀"
fi
if ls "$SOURCE_DIR" | grep -Ei '_(1m|5m|15m|30m|60m|1min|5min|15min|30min|60min)\.csv$' >/dev/null 2>&1; then
  echo "[WARN] 文件名仍带频率后缀；确认 FREQ_SUFFIX_RE 是否覆盖简写"
else
  echo "[OK] 文件名看起来已去掉频率后缀"
fi

echo "== Step 2. dump_bin（优先 --freq 1min，失败回退 minute） =="
set +e
"$PY" "$REPO_ROOT/scripts/dump_bin.py" dump_all \
  --data_path "$SOURCE_DIR" \
  --qlib_dir "$QLIB_DIR" \
  --freq 1min \
  --date_field_name date \
  --symbol_field_name symbol \
  --include_fields open,high,low,close,volume,factor,vwap \
  --max_workers "$N_JOBS"
rc=$?
if [ $rc -ne 0 ]; then
  echo "[INFO] 1min 失败，尝试 minute 兼容参数……"
  "$PY" "$REPO_ROOT/scripts/dump_bin.py" dump_all \
    --data_path "$SOURCE_DIR" \
    --qlib_dir "$QLIB_DIR" \
    --freq minute \
    --date_field_name date \
    --symbol_field_name symbol \
    --include_fields open,high,low,close,volume,factor,vwap \
    --max_workers "$N_JOBS"
  rc=$?
fi
set -e
[ $rc -eq 0 ] || { echo "[FATAL] dump_bin 失败"; exit 2; }

echo "== Step 2.1 检查分钟特征文件是否存在 =="
env QLIB_DIR="$QLIB_DIR" "$PY" - <<'PY'
from pathlib import Path
import os
qlib_dir = Path(os.environ["QLIB_DIR"])
feat = qlib_dir/"features"
if not feat.exists():
    raise SystemExit("[FATAL] 未找到 features 目录: "+str(feat))
hit = any(feat.rglob("*.1min.bin")) or any(feat.rglob("*.minute.bin"))
print("[OK] 检测到分钟特征文件") if hit else (_ for _ in ()).throw(SystemExit("[FATAL] 未检测到 *.1min.bin / *.minute.bin"))
PY

echo "== Step 2.2 修复日历别名（1min.txt） =="
CALDIR="$QLIB_DIR/calendars"
mkdir -p "$CALDIR"
if [ -f "$CALDIR/minute.txt" ] && [ ! -f "$CALDIR/1min.txt" ]; then
  ln -s minute.txt "$CALDIR/1min.txt" || cp "$CALDIR/minute.txt" "$CALDIR/1min.txt"
fi

echo "== Step 3. 用 features 反推 instruments =="
env QLIB_DIR="$QLIB_DIR" "$PY" - <<'PY'
from pathlib import Path
import os
qlib_dir = Path(os.environ["QLIB_DIR"])
feat_dir = qlib_dir/"features"
cands = [p for p in feat_dir.iterdir() if p.is_dir() and p.name.lower() not in {"day","1min","minute"}]
keep=[]
for d in cands:
    if any(d.glob("*.1min.bin")) or any(d.glob("*.minute.bin")):
        keep.append(d.name)
inst = qlib_dir/"instruments"; inst.mkdir(parents=True, exist_ok=True)
with open(inst/"all.txt","w") as f:
    for name in sorted(set(keep)): f.write(name.upper()+"\n")
print(f"[OK] instruments/all.txt 写入 {len(keep)} 个标的")
PY

echo "== Step 4. 验收读取（freq='1min'） =="
env QLIB_DIR="$QLIB_DIR" "$PY" - <<'PY'
import os, qlib
from qlib.data import D
from pathlib import Path
QLIB_DIR = os.environ["QLIB_DIR"]
qlib.init(provider_uri=QLIB_DIR, region="us")
inst_path = Path(QLIB_DIR)/"instruments"/"all.txt"
symbols = [l.strip() for l in open(inst_path) if l.strip()][:2]
print("Using tickers:", symbols)
cal = [l.strip() for l in open(Path(QLIB_DIR)/"calendars"/"1min.txt") if l.strip()]
start, end = cal[max(0,len(cal)-200)], cal[-1]
df = D.features(symbols, ["$close","$open","$volume"], start_time=start, end_time=end, freq="1min")
print(df.head()); print(df.tail())
print("rows:", len(df), "per-instrument:", df.groupby('instrument').size().to_dict())
PY

echo "[DONE] 5min 源重建完成：$QLIB_DIR"
