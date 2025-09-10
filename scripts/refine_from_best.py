#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
refine_from_best.py

读取先前 sweep 的 best_params.json，围绕其邻域生成 9 组（top_k±step_topk, tv±step_tv；其余保持中心值），
运行回测并输出 summary 与 BEST，便于对比“细化矩阵”。

示例：
  python scripts/refine_from_best.py \
    --best_json backtest/reports/hier_sweep/20250909_045816/best_params.json \
    --qlib_dir ~/.qlib/qlib_data/us_data \
    --preds artifacts/preds/weekly/preds_20240101_20241231.parquet \
    --features_path artifacts/features_day.parquet \
    --start 2024-01-01 --end 2024-12-31 \
    --out_root backtest/reports/hier_sweep_refine
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import csv


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--best_json", required=True, help="之前 sweep 的 best_params.json 路径")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--strategy_key", default="sharpe_focus")
    ap.add_argument("--qlib_dir", required=True)
    ap.add_argument("--preds", required=True)
    ap.add_argument("--features_path", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--out_root", default="backtest/reports/hier_sweep_refine")
    # 邻域设置：只在 top_k 和 target_vol 上做 ±1 步，得到 3×3=9 组
    ap.add_argument("--step_topk", type=int, default=5)
    ap.add_argument("--step_tv", type=float, default=0.01)
    ap.add_argument("--trade_at", choices=["open","close"], default="open")
    ap.add_argument("--commission_bps", type=float, default=1.0)
    ap.add_argument("--slippage_bps", type=float, default=5.0)
    ap.add_argument("--cash", type=float, default=1_000_000.0)
    return ap.parse_args()


def neighborhood(center: float, step: float, is_int: bool) -> list:
    vals = [center - step, center, center + step]
    if is_int:
        return sorted(set(int(round(v)) for v in vals))
    return sorted(set(round(v, 6) for v in vals))


def run_one(args: argparse.Namespace, out_dir: Path, p: dict) -> int:
    cmd = [
        sys.executable, "backtest/engine/run_backtest.py",
        "--config", args.config,
        "--strategy_key", args.strategy_key,
        "--qlib_dir", args.qlib_dir,
        "--preds", args.preds,
        "--features_path", args.features_path,
        "--start", args.start,
        "--end", args.end,
        "--out_dir", str(out_dir),
        "--trade_at", args.trade_at,
        "--commission_bps", str(args.commission_bps),
        "--slippage_bps", str(args.slippage_bps),
        "--cash", str(args.cash),
        "--adv_limit_pct", str(p.get("adv_limit_pct", 0.0)),
        "--top_k", str(p["top_k"]),
        "--short_k", str(p.get("short_k", 0)),
        "--membership_buffer", str(p["membership_buffer"]),
        "--weight_scheme", p["weight_scheme"],
        "--max_pos_per_name", str(p["max_pos_per_name"]),
        "--smooth_eta", str(p["smooth_eta"]),
        "--neutralize", p.get("neutralize", "beta,sector"),
        "--target_vol", str(p["target_vol"]),
        "--hard_cap",
        "--long_exposure", str(p.get("long_exposure", 1.0)),
        "--short_exposure", str(p.get("short_exposure", 0.0)),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(proc.stdout)
    return proc.returncode


def main():
    args = parse_args()
    best = json.loads(Path(args.best_json).read_text())

    tk_center = int(best.get("top_k", 30))
    tv_center = float(best.get("target_vol", 0.10))
    buf = float(best.get("membership_buffer", 0.30))
    cap = float(best.get("max_pos_per_name", 0.04))
    eta = float(best.get("smooth_eta", 0.60))
    ws = str(best.get("weight_scheme", "equal"))
    neu = str(best.get("neutralize", "beta,sector"))

    tk_vals = neighborhood(tk_center, args.step_topk, is_int=True)
    tv_vals = neighborhood(tv_center, args.step_tv, is_int=False)

    ts = time.strftime("%Y%m%d_%H%M%S")
    root = Path(args.out_root) / ts
    root.mkdir(parents=True, exist_ok=True)

    results = []
    idx = 0
    for tk in tk_vals:
        for tv in tv_vals:
            idx += 1
            tag = f"tk{tk}_ws{ws}_buf{buf}_cap{cap}_eta{eta}_tv{tv}"
            out_dir = root / tag
            out_dir.mkdir(parents=True, exist_ok=True)
            p = {
                "top_k": tk,
                "short_k": 0,
                "membership_buffer": buf,
                "weight_scheme": ws,
                "max_pos_per_name": cap,
                "smooth_eta": eta,
                "neutralize": neu,
                "target_vol": tv,
                "hard_cap": True,
                "long_exposure": 1.0,
                "short_exposure": 0.0,
            }
            print(f"[{idx}/9] running -> {out_dir}")
            rc = run_one(args, out_dir, p)
            if rc != 0:
                print(f"[warn] run failed rc={rc}: {out_dir}")
                continue
            # read kpis
            kpath = out_dir / "kpis.json"
            try:
                kpis = json.loads(kpath.read_text())
            except Exception:
                kpis = {}
            row = {
                **{kk: p[kk] for kk in ["top_k","membership_buffer","weight_scheme","max_pos_per_name","smooth_eta","target_vol"]},
                **{kk: kpis.get(kk) for kk in ["sharpe","cagr","mdd_pct"]},
                "out_dir": str(out_dir),
            }
            results.append(row)

    # write summary
    if results:
        keys = list(results[0].keys())
        csvp = root / "refine_summary.csv"
        with open(csvp, "w", newline="") as f:
            import csv as _csv
            w = _csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in results:
                w.writerow(r)
        # best
        results.sort(key=lambda r: (-(r.get("sharpe") or -1e9)))
        bestp = root / "best_params.json"
        (root / "BEST").write_text(results[0]["out_dir"])  # 快捷定位
        with open(bestp, "w") as f:
            json.dump(results[0], f, indent=2)
        print(f"[saved] summary -> {csvp}")
        print(f"[saved] best -> {bestp}")
    else:
        print("[warn] no results collected")


if __name__ == "__main__":
    main()
