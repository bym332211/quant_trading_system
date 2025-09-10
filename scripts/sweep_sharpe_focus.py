#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量参数扫描脚本（Sharpe 优先）

说明：
- 基于 backtest/engine/run_backtest.py，通过 CLI 覆盖参数进行网格搜索；
- 每个组合输出到独立目录，解析 kpis.json 汇总指标，按 Sharpe 降序排序；
- 保存 sweep_summary.csv 与 best_params.json，方便快速挑选最优组合。

用法示例：
  python scripts/sweep_sharpe_focus.py \
    --qlib_dir "/home/ec2-user/.qlib/qlib_data/us_data" \
    --preds "artifacts/preds/weekly/predictions.parquet" \
    --features_path "artifacts/features_day.parquet" \
    --start "2017-01-01" --end "2024-12-31" \
    --out_root "backtest/reports/sweep_sharpe_focus"

注意：
- config/config.yaml 的策略键使用 "sharpe_focus"；
- 为绕过当前 ConfigLoader 对 entry 子配置的限制，本脚本通过 CLI 显式覆盖入场参数。
"""
from __future__ import annotations
import argparse
import itertools
import json
from pathlib import Path
import subprocess
import sys
import time
import csv


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--strategy_key", default="sharpe_focus")
    ap.add_argument("--qlib_dir", required=True)
    ap.add_argument("--preds", required=True)
    ap.add_argument("--features_path", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--out_root", default="backtest/reports/sweep_sharpe_focus")
    ap.add_argument("--grid_preset", choices=["coarse", "medium", "fine", "focused"], default="coarse",
                    help="预设网格粒度：coarse(快筛) / medium / fine(精扫)")
    ap.add_argument("--trade_at", choices=["open", "close"], default="open")
    ap.add_argument("--commission_bps", type=float, default=1.0)
    ap.add_argument("--slippage_bps", type=float, default=5.0)
    ap.add_argument("--cash", type=float, default=1_000_000.0)
    ap.add_argument("--adv_limit_pct", type=float, default=0.0)
    ap.add_argument("--dry", action="store_true", help="只打印命令，不执行")
    ap.add_argument("--limit", type=int, default=None, help="最多运行多少组（用于抽样预跑）")
    return ap.parse_args()


def safe_tag(x) -> str:
    if isinstance(x, float):
        return ("%.2f" % x).rstrip("0").rstrip(".")
    return str(x)


def build_grid(preset: str):
    """根据预设粒度构建网格空间"""
    if preset == "coarse":
        # 快速粗筛：32 组（2x2x1x2x2x1x2）
        top_k_list = [30, 60]
        weight_schemes = ["icdf", "equal"]
        membership_buffers = [0.30]
        max_pos_list = [0.02, 0.04]
        smooth_eta_list = [0.40, 0.60]
        neutralize_list = ["beta,sector"]
        target_vol_list = [0.08, 0.12]
    elif preset == "medium":
        # 中等规模：128 组，围绕粗筛方向（top_k≈30/45，tv含0.08）
        top_k_list = [30, 45]
        weight_schemes = ["icdf", "equal"]
        membership_buffers = [0.30, 0.35]
        max_pos_list = [0.02, 0.03]
        smooth_eta_list = [0.50, 0.60]
        neutralize_list = ["beta,sector", "beta,sector,size"]
        target_vol_list = [0.08, 0.10]
    elif preset == "focused":
        # 聚焦扫描：64 组（2x1x2x2x2x2x2），更贴近粗筛最优（icdf/eta高/低tv）
        top_k_list = [30, 45]
        weight_schemes = ["icdf"]
        membership_buffers = [0.30, 0.35]
        max_pos_list = [0.02, 0.03]
        smooth_eta_list = [0.50, 0.60]
        neutralize_list = ["beta,sector", "beta,sector,size"]
        target_vol_list = [0.08, 0.10]
    else:  # fine
        # 精细扫描：432 组（3x2x2x3x2x2x3）
        top_k_list = [30, 45, 60]
        weight_schemes = ["icdf", "equal"]
        membership_buffers = [0.25, 0.35]
        max_pos_list = [0.02, 0.03, 0.04]
        smooth_eta_list = [0.40, 0.60]
        neutralize_list = ["beta,sector", "beta,sector,size"]
        target_vol_list = [0.08, 0.10, 0.12]

    grid = []
    for (tk, ws, buf, cap, eta, neu, tv) in itertools.product(
        top_k_list, weight_schemes, membership_buffers, max_pos_list,
        smooth_eta_list, neutralize_list, target_vol_list
    ):
        grid.append({
            "top_k": tk,
            "short_k": 0,
            "membership_buffer": buf,
            "weight_scheme": ws,
            "max_pos_per_name": cap,
            "smooth_eta": eta,
            "neutralize": neu,
            "target_vol": tv,
            "hard_cap": True,
            # 长多设定
            "long_exposure": 1.0,
            "short_exposure": 0.0,
        })
    return grid


def build_outdir(root: Path, params: dict) -> Path:
    tag = "tk%s_ws%s_buf%s_cap%s_eta%s_neu%s_tv%s" % (
        safe_tag(params["top_k"]),
        params["weight_scheme"],
        safe_tag(params["membership_buffer"]),
        safe_tag(params["max_pos_per_name"]),
        safe_tag(params["smooth_eta"]),
        params["neutralize"].replace(",", "-").replace(" ", ""),
        safe_tag(params["target_vol"]),
    )
    return root / tag


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
        "--adv_limit_pct", str(args.adv_limit_pct),
        # 选股与入场（显式覆盖）
        "--top_k", str(p["top_k"]),
        "--short_k", str(p["short_k"]),
        "--membership_buffer", str(p["membership_buffer"]),
        "--weight_scheme", p["weight_scheme"],
        "--max_pos_per_name", str(p["max_pos_per_name"]),
        "--smooth_eta", str(p["smooth_eta"]),
        "--neutralize", p["neutralize"],
        "--target_vol", str(p["target_vol"]),
        "--hard_cap",
        "--long_exposure", str(p["long_exposure"]),
        "--short_exposure", str(p["short_exposure"]),
    ]
    if args.dry:
        print("DRY:", " ".join(cmd))
        return 0
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(proc.stdout)
    return proc.returncode


def load_kpis(out_dir: Path) -> dict:
    kpath = out_dir / "kpis.json"
    if not kpath.exists():
        return {}
    try:
        return json.loads(kpath.read_text())
    except Exception:
        return {}


def main():
    args = parse_args()
    # 预检：路径存在性
    from pathlib import Path
    preds_p = Path(args.preds).expanduser().resolve()
    feat_p = Path(args.features_path).expanduser().resolve()
    qlib_p = Path(args.qlib_dir).expanduser().resolve()
    missing = []
    if not preds_p.exists():
        missing.append(f"preds missing: {preds_p}")
    if not feat_p.exists():
        missing.append(f"features missing: {feat_p}")
    if not qlib_p.exists():
        missing.append(f"qlib_dir missing: {qlib_p}")
    if missing:
        print("[error] required inputs not found:\n - " + "\n - ".join(missing))
        print("hint: you can generate a dummy predictions parquet for pipeline testing:")
        print("  python scripts/make_dummy_preds.py --features_path \"%s\" --start \"%s\" --end \"%s\" --out \"%s\" --per_day_limit 300" % (
            str(feat_p), args.start, args.end, str(preds_p)))
        sys.exit(2)
    ts = time.strftime("%Y%m%d_%H%M%S")
    root = Path(args.out_root) / ts
    root.mkdir(parents=True, exist_ok=True)

    grid = build_grid(args.grid_preset)
    if args.limit is not None:
        grid = grid[: max(1, int(args.limit))]

    print(f"[info] grid_preset={args.grid_preset}  combos={len(grid)}")

    results = []
    for i, p in enumerate(grid, 1):
        out_dir = build_outdir(root, p)
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{i}/{len(grid)}] running -> {out_dir}")
        rc = run_one(args, out_dir, p)
        if rc != 0:
            print(f"[warn] run failed rc={rc}: {out_dir}")
            continue
        kpis = load_kpis(out_dir)
        row = {
            **{k: p[k] for k in [
                "top_k","short_k","membership_buffer","weight_scheme",
                "max_pos_per_name","smooth_eta","neutralize","target_vol"
            ]},
            **{k: kpis.get(k) for k in [
                "sharpe","cagr","mdd_pct","turnover_mean","turnover_p90",
                "adv_clip_days_frac","adv_clip_ratio_avg","gross_long_avg","gross_short_avg"
            ]},
            "out_dir": str(out_dir),
        }
        results.append(row)

    # 排序：Sharpe 降序，turnover / adv_clip 升序，回撤升序
    def sort_key(r: dict):
        # 处理缺失
        s = r.get("sharpe")
        s = -1e9 if s is None else s
        return (
            -(s),
            r.get("turnover_mean") or 1e9,
            r.get("adv_clip_ratio_avg") or 1e9,
            r.get("mdd_pct") or 1e9,
        )

    results.sort(key=sort_key)

    # 写 CSV 汇总
    csv_path = root / "sweep_summary.csv"
    if results:
        keys = list(results[0].keys())
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in results:
                w.writerow(r)
        print(f"[saved] summary -> {csv_path}")

        # 最优参数
        best_path = root / "best_params.json"
        (root / "BEST").write_text(results[0]["out_dir"])  # 快捷定位
        with open(best_path, "w") as f:
            json.dump(results[0], f, indent=2)
        print(f"[saved] best -> {best_path}")
    else:
        print("[warn] no results collected; check runs and logs")


if __name__ == "__main__":
    main()
