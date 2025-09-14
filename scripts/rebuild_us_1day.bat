@echo off
setlocal enabledelayedexpansion

rem Rebuild US 1-day Qlib data on Windows (BAT version)
rem Default paths adapted to this repo layout on Windows

rem Repo root (folder containing this script)
set "REPO_ROOT=%~dp0.."
for %%I in ("%REPO_ROOT%") do set "REPO_ROOT=%%~fI"

rem Configure paths (edit if needed). You can override via args: %1 RAW_DIR, %2 SOURCE_DIR, %3 QLIB_DIR, %4 N_JOBS
set "RAW_DIR=%REPO_ROOT%\data\data\sp500"
set "SOURCE_DIR=%USERPROFILE%\.qlib\source\us_from_yf"
set "QLIB_DIR=%USERPROFILE%\.qlib\qlib_data\us_data"
set "N_JOBS=8"
set "CLEAN_MODE=backup"  rem values: backup or delete

rem Resolve python
set "PY=%PYTHON%"
if not defined PY set "PY=python"

rem Timestamp helper via PowerShell
for /f %%i in ('powershell -NoProfile -Command "(Get-Date).ToString('yyyyMMdd_HHmmss')"') do set "TS=%%i"

rem Allow overriding by command-line args
if not "%~1"=="" set "RAW_DIR=%~1"
if not "%~2"=="" set "SOURCE_DIR=%~2"
if not "%~3"=="" set "QLIB_DIR=%~3"
if not "%~4"=="" set "N_JOBS=%~4"

echo == Config ==
echo RAW_DIR   = %RAW_DIR%
echo SOURCE_DIR= %SOURCE_DIR%
echo QLIB_DIR  = %QLIB_DIR%
echo N_JOBS    = %N_JOBS%

rem backup_or_clean DIR
set "_BC_DIR="
set "_TMPDIR="
set "_BAKDIR="

echo == Clean ==
call :backup_or_clean "%SOURCE_DIR%"
if errorlevel 1 goto :fail
call :backup_or_clean "%QLIB_DIR%"
if errorlevel 1 goto :fail

echo == Normalize (freq=day) ==
"%PY%" "%REPO_ROOT%\data\build_qlib_us.py" ^
  --in_dir "%RAW_DIR%" --source_dir "%SOURCE_DIR%" --qlib_dir "%QLIB_DIR%" ^
  --freq day --n_jobs %N_JOBS% --skip_dump_bin
if errorlevel 1 goto :fail

echo == Dump (day) ==
"%PY%" "%REPO_ROOT%\scripts\dump_bin.py" dump_all ^
  --data_path "%SOURCE_DIR%" --qlib_dir "%QLIB_DIR%" --freq day ^
  --date_field_name date --symbol_field_name symbol ^
  --include_fields open,high,low,close,volume,factor,vwap --max_workers %N_JOBS%
if errorlevel 1 goto :fail

if not exist "%QLIB_DIR%\features" (
  echo [FATAL] not found: %QLIB_DIR%\features
  goto :fail
)

echo == Rebuild instruments (\*.day.bin) ==
set "TMP_PY=%TEMP%\rebuild_instruments_%TS%.py"
>"%TMP_PY%" echo from pathlib import Path
>>"%TMP_PY%" echo import os
>>"%TMP_PY%" echo qlib_dir = Path(r"%QLIB_DIR%")
>>"%TMP_PY%" echo feat_dir = qlib_dir / "features"
>>"%TMP_PY%" echo cands = [p for p in feat_dir.iterdir() if p.is_dir() and p.name.lower() not in {"day","1min","minute"}]
>>"%TMP_PY%" echo keep=[]
>>"%TMP_PY%" echo for d in cands:
>>"%TMP_PY%" echo ^    	if any(d.glob("*.day.bin")): keep.append(d.name)
>>"%TMP_PY%" echo inst = qlib_dir/"instruments"; inst.mkdir(parents=True, exist_ok=True)
>>"%TMP_PY%" echo with open(inst/"all.txt","w") as f:
>>"%TMP_PY%" echo ^    	for name in sorted(set(keep)): f.write(name.upper()+"\n")
>>"%TMP_PY%" echo print(f"[OK] instruments/all.txt written: {len(keep)} tickers")

"%PY%" "%TMP_PY%"
set "RC=%ERRORLEVEL%"
del /q "%TMP_PY%" 2>nul
if not "%RC%"=="0" goto :fail

echo == Sanity read (freq='day') ==
set "TMP_PY=%TEMP%\sanity_read_%TS%.py"
>"%TMP_PY%" echo import os, qlib
>>"%TMP_PY%" echo from qlib.data import D
>>"%TMP_PY%" echo from pathlib import Path
>>"%TMP_PY%" echo QLIB_DIR=r"%QLIB_DIR%"; qlib.init(provider_uri=QLIB_DIR, region="us")
>>"%TMP_PY%" echo syms=[l.strip() for l in open(Path(QLIB_DIR)/"instruments"/"all.txt") if l.strip()][:2]
>>"%TMP_PY%" echo print("Using tickers:", syms)
>>"%TMP_PY%" echo df=D.features(syms,["$close","$open","$volume"],start_time="2019-01-02",end_time="2019-01-15",freq="day")
>>"%TMP_PY%" echo print(df.head()); print("rows:",len(df))

"%PY%" "%TMP_PY%"
set "RC=%ERRORLEVEL%"
del /q "%TMP_PY%" 2>nul
if not "%RC%"=="0" goto :fail

echo == Verify adjusted prices (day) ==
set "TMP_PY=%TEMP%\verify_adjust_%TS%.py"
>"%TMP_PY%" echo from pathlib import Path
>>"%TMP_PY%" echo import os, sys, pandas as pd, numpy as np, qlib
>>"%TMP_PY%" echo from qlib.data import D
>>"%TMP_PY%" echo QLIB_DIR = Path(r"%QLIB_DIR%")
>>"%TMP_PY%" echo SRC_DIR  = Path(r"%SOURCE_DIR%")
>>"%TMP_PY%" echo SAMPLE_N = int(os.environ.get("VERIFY_SAMPLE", "5"))
>>"%TMP_PY%" echo ROWS_N   = int(os.environ.get("VERIFY_ROWS", "1000"))
>>"%TMP_PY%" echo def pick_symbols():
>>"%TMP_PY%" echo ^    	inst = QLIB_DIR / "instruments" / "all.txt"
>>"%TMP_PY%" echo ^    	syms=[]
>>"%TMP_PY%" echo ^    	if inst.exists(): syms=[l.strip() for l in inst.read_text().splitlines() if l.strip()]
>>"%TMP_PY%" echo ^    	if not syms: syms=[p.stem.upper() for p in sorted(SRC_DIR.glob("*.csv"))]
>>"%TMP_PY%" echo ^    	return syms[:SAMPLE_N]
>>"%TMP_PY%" echo def read_csv_tail(sym: str, n: int) -> pd.DataFrame:
>>"%TMP_PY%" echo ^    	fp = SRC_DIR / f"{sym}.csv"; df = pd.read_csv(fp)
>>"%TMP_PY%" echo ^    	df.columns=[c.strip().lower().replace(" ","_") for c in df.columns]
>>"%TMP_PY%" echo ^    	if "date" not in df.columns: raise RuntimeError(f"{fp} missing 'date'")
>>"%TMP_PY%" echo ^    	dt = pd.to_datetime(df["date"], utc=True, errors="coerce").dt.tz_convert("UTC").dt.tz_localize(None)
>>"%TMP_PY%" echo ^    	df["date"]=dt; df=df.sort_values("date").tail(n)
>>"%TMP_PY%" echo ^    	return df[["date","open","high","low","close","vwap","volume"]]
>>"%TMP_PY%" echo def cmp(a: pd.Series, b: pd.Series, atol=1e-4, rtol=1e-4):
>>"%TMP_PY%" echo ^    	a=a.astype(float); b=b.astype(float); abs_err=(a-b).abs()
>>"%TMP_PY%" echo ^    	with np.errstate(divide="ignore", invalid="ignore"):
>>"%TMP_PY%" echo ^    	    rel_err = abs_err/np.maximum(b.abs(),1e-12)
>>"%TMP_PY%" echo ^    	return bool((abs_err<=atol).all() or (rel_err<=rtol).all())
>>"%TMP_PY%" echo qlib.init(provider_uri=str(QLIB_DIR), region="us")
>>"%TMP_PY%" echo symbols = pick_symbols()
>>"%TMP_PY%" echo if not symbols: print("[NG] no symbols"); sys.exit(3)
>>"%TMP_PY%" echo csv0 = read_csv_tail(symbols[0], ROWS_N)
>>"%TMP_PY%" echo if csv0.empty: print("[NG] CSV empty"); sys.exit(3)
>>"%TMP_PY%" echo start = csv0["date"].iloc[0].strftime("%Y-%m-%d"); end = csv0["date"].iloc[-1].strftime("%Y-%m-%d")
>>"%TMP_PY%" echo fields=["$open","$high","$low","$close","$vwap","$factor"]
>>"%TMP_PY%" echo qdf = D.features(symbols, fields, start_time=start, end_time=end, freq="day")
>>"%TMP_PY%" echo if qdf.empty: print("[NG] Qlib empty"); sys.exit(3)
>>"%TMP_PY%" echo print("[OK] verification sample ready (rows=", len(qdf), ")")

"%PY%" "%TMP_PY%"
set "RC=%ERRORLEVEL%"
del /q "%TMP_PY%" 2>nul
if not "%RC%"=="0" (
  echo [FATAL] verification NG
  goto :fail
) else (
  echo [OK] verification OK
)

echo [DONE] 1day rebuild finished: %QLIB_DIR%
goto :eof

:backup_or_clean
set "_BC_DIR=%~1"
if not defined _BC_DIR exit /b 1

if not exist "%_BC_DIR%" (
  mkdir "%_BC_DIR%"
  exit /b 0
)

rem If delete mode, remove and recreate
if /I "%CLEAN_MODE%"=="delete" (
  echo Deleting "%_BC_DIR%"
  rmdir /s /q "%_BC_DIR%"
  mkdir "%_BC_DIR%"
  exit /b 0
)

rem Else backup mode: move directory to .bak.TIMESTAMP and recreate
for /f %%i in ('powershell -NoProfile -Command "(Get-Date).ToString(''yyyyMMdd_HHmmss'')"') do set "_NOWTS=%%i"
set "_BAKDIR=%_BC_DIR%.bak.%_NOWTS%"
echo Backing up "%_BC_DIR%" to "%_BAKDIR%"
move "%_BC_DIR%" "%_BAKDIR%" >nul
mkdir "%_BC_DIR%"
exit /b 0

:fail
echo [FAILED]
exit /b 1
