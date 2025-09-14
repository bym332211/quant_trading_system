#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分阶段网格搜索（粗 → 细），以 Sharpe 最优为目标。

特性：
- 第1阶段：使用粗粒度预设快速扫方向；
- 第2阶段：围绕第1阶段最优参数自动细化局部网格；
- 输出每阶段 summary CSV、最优参数 JSON，并指向最优 run 目录。

用法示例：
  python scripts/hierarchical_sweep_sharpe.py \
    --qlib_dir "/home/ec2-user/.qlib/qlib_data/us_data" \
    --preds "artifacts/preds/weekly/predictions.parquet" \
    --features_path "artifacts/features_day.parquet" \
    --start "2017-01-01" --end "2024-12-31" \
    --out_root "backtest/reports/hier_sweep" \
    --stage1_preset coarse --run_stage2

备注：默认策略 key 使用 config 中的 "sharpe_focus"；可根据需要调整。
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
from typing import Dict, Any, List, Tuple


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--strategy_key", default="xsec_rebalance")
    ap.add_argument("--qlib_dir", required=True)
    ap.add_argument("--preds", required=True)
    ap.add_argument("--features_path", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--out_root", default="backtest/reports/hier_sweep")
    ap.add_argument("--trade_at", choices=["open", "close"], default="open")
    ap.add_argument("--commission_bps", type=float, default=1.0)
    ap.add_argument("--slippage_bps", type=float, default=5.0)
    ap.add_argument("--cash", type=float, default=1_000_000.0)
    ap.add_argument("--adv_limit_pct", type=float, default=0.0)
    ap.add_argument("--dry", action="store_true", help="只打印命令，不执行")
    ap.add_argument("--limit_stage1", type=int, default=None, help="第1阶段最多运行多少组（抽样）")
    ap.add_argument("--limit_stage2", type=int, default=None, help="第2阶段最多运行多少组（抽样）")

    # 分阶段控制
    ap.add_argument("--stage1_preset", choices=["coarse", "medium", "fine", "focused"], default="coarse",
                    help="第1阶段粗粒度预设")
    ap.add_argument("--run_stage2", action="store_true", help="启用第2阶段细化扫描")

    # 第2阶段邻域宽度（围绕最优点）
    ap.add_argument("--n_topk_neighbors", type=int, default=2, help="top_k 左右邻点数量（步长见下）")
    ap.add_argument("--step_topk", type=int, default=5, help="top_k 邻域步长")
    ap.add_argument("--n_tv_neighbors", type=int, default=2, help="target_vol 左右邻点数量")
    ap.add_argument("--step_tv", type=float, default=0.01, help="target_vol 邻域步长（绝对值）")
    ap.add_argument("--n_buf_neighbors", type=int, default=1, help="membership_buffer 左右邻点数量")
    ap.add_argument("--step_buf", type=float, default=0.05, help="membership_buffer 邻域步长")
    ap.add_argument("--n_eta_neighbors", type=int, default=1, help="smooth_eta 左右邻点数量")
    ap.add_argument("--step_eta", type=float, default=0.05, help="smooth_eta 邻域步长")
    ap.add_argument("--n_cap_neighbors", type=int, default=1, help="max_pos_per_name 左右邻点数量")
    ap.add_argument("--step_cap", type=float, default=0.01, help="max_pos_per_name 邻域步长")

    # 离散备选是否扩展
    ap.add_argument("--try_both_weight_schemes", action="store_true", help="第2阶段同时尝试 equal 与 icdf")
    ap.add_argument("--try_alt_neutralize", action="store_true", help="第2阶段同时尝试另一组去因子项")
    return ap.parse_args()


def safe_tag(x) -> str:
    if isinstance(x, float):
        return ("%.3f" % x).rstrip("0").rstrip(".")
    return str(x)


def build_grid_stage1(preset: str) -> List[Dict[str, Any]]:
    """根据预设粒度构建第1阶段粗粒度网格。"""
    if preset == "coarse":
        top_k_list = [10, 30, 60]
        weight_schemes = ["icdf", "equal"]
        membership_buffers = [0.30]
        max_pos_list = [0.02, 0.04]
        smooth_eta_list = [0.40, 0.60]
        neutralize_list = ["beta,sector"]
        target_vol_list = [0.08, 0.12]
    elif preset == "medium":
        top_k_list = [30, 45]
        weight_schemes = ["icdf", "equal"]
        membership_buffers = [0.30, 0.35]
        max_pos_list = [0.02, 0.03]
        smooth_eta_list = [0.50, 0.60]
        neutralize_list = ["beta,sector", "beta,sector,size"]
        target_vol_list = [0.08, 0.10]
    elif preset == "focused":
        top_k_list = [30, 45]
        weight_schemes = ["icdf"]
        membership_buffers = [0.30, 0.35]
        max_pos_list = [0.02, 0.03]
        smooth_eta_list = [0.50, 0.60]
        neutralize_list = ["beta,sector", "beta,sector,size"]
        target_vol_list = [0.08, 0.10]
    else:  # fine
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


def neighborhood(center: float, n: int, step: float, vmin: float = None, vmax: float = None, is_int: bool = False) -> List[float]:
    vals = [center + i * step for i in range(-n, n + 1)]
    if vmin is not None:
        vals = [v for v in vals if v >= vmin - 1e-12]
    if vmax is not None:
        vals = [v for v in vals if v <= vmax + 1e-12]
    if is_int:
        vals = sorted({int(round(v)) for v in vals})
    else:
        vals = sorted({round(v, 6) for v in vals})
    return vals


def build_grid_stage2(best: Dict[str, Any], args: argparse.Namespace) -> List[Dict[str, Any]]:
    """基于第1阶段最优参数，围绕其邻域构建细粒度网格。"""
    tk_center = int(best.get("top_k", 40))
    tv_center = float(best.get("target_vol", 0.10))
    buf_center = float(best.get("membership_buffer", 0.30))
    eta_center = float(best.get("smooth_eta", 0.55))
    cap_center = float(best.get("max_pos_per_name", 0.03))
    ws_center = str(best.get("weight_scheme", "icdf"))
    neu_center = str(best.get("neutralize", "beta,sector"))

    top_k_list = neighborhood(tk_center, args.n_topk_neighbors, args.step_topk, vmin=10, vmax=200, is_int=True)
    target_vol_list = neighborhood(tv_center, args.n_tv_neighbors, args.step_tv, vmin=0.0, vmax=0.5)
    membership_buffers = neighborhood(buf_center, args.n_buf_neighbors, args.step_buf, vmin=0.0, vmax=0.8)
    smooth_eta_list = neighborhood(eta_center, args.n_eta_neighbors, args.step_eta, vmin=0.0, vmax=0.99)
    max_pos_list = neighborhood(cap_center, args.n_cap_neighbors, args.step_cap, vmin=0.005, vmax=0.2)

    weight_schemes = [ws_center]
    if args.try_both_weight_schemes and ws_center in ("icdf", "equal"):
        alt = "equal" if ws_center == "icdf" else "icdf"
        weight_schemes = [ws_center, alt]

    neutralize_list = [neu_center]
    if args.try_alt_neutralize:
        alt_neu = "beta,sector,size" if neu_center == "beta,sector" else "beta,sector"
        neutralize_list = [neu_center, alt_neu]

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
            "long_exposure": 1.0,
            "short_exposure": 0.0,
        })
    return grid


def build_outdir(root: Path, stage: str, params: dict) -> Path:
    tag = "s%s_tk%s_ws%s_buf%s_cap%s_eta%s_neu%s_tv%s" % (
        stage,
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


def sort_key(r: dict):
    s = r.get("sharpe")
    s = -1e9 if s is None else s
    return (
        -(s),
        r.get("turnover_mean") or 1e9,
        r.get("adv_clip_ratio_avg") or 1e9,
        r.get("mdd_pct") or 1e9,
    )


def sweep(root: Path, stage_tag: str, args: argparse.Namespace, grid: List[Dict[str, Any]], limit: int | None) -> Tuple[List[dict], Path | None]:
    if limit is not None:
        grid = grid[: max(1, int(limit))]
    results: List[dict] = []
    for i, p in enumerate(grid, 1):
        out_dir = build_outdir(root, stage_tag, p)
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{stage_tag}] [{i}/{len(grid)}] running -> {out_dir}")
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

    if not results:
        return [], None

    results.sort(key=sort_key)
    csv_path = root / f"s{stage_tag}_summary.csv"
    keys = list(results[0].keys())
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in results:
            w.writerow(r)
    print(f"[saved] summary -> {csv_path}")

    best_dir = Path(results[0]["out_dir"]) if results else None
    if best_dir is not None:
        (root / f"BEST_s{stage_tag}").write_text(str(best_dir))
        with open(root / f"best_params_s{stage_tag}.json", "w") as f:
            json.dump(results[0], f, indent=2)
        print(f"[saved] best -> {root / f'best_params_s{stage_tag}.json'}")
    return results, best_dir


def main():
    args = parse_args()
    # 预检输入（dry 模式跳过存在性约束）
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
    if missing and not args.dry:
        print("[error] required inputs not found:\n - " + "\n - ".join(missing))
        print("hint: generate dummy predictions if needed:")
        print("  python scripts/make_dummy_preds.py --features_path \"%s\" --start \"%s\" --end \"%s\" --out \"%s\" --per_day_limit 300" % (
            str(feat_p), args.start, args.end, str(preds_p)))
        sys.exit(2)
    elif missing and args.dry:
        print("[warn] inputs missing but continuing due to --dry:\n - " + "\n - ".join(missing))

    ts = time.strftime("%Y%m%d_%H%M%S")
    root = Path(args.out_root) / ts
    root.mkdir(parents=True, exist_ok=True)

    # 阶段1：粗粒度
    grid1 = build_grid_stage1(args.stage1_preset)
    print(f"[info] stage1 preset={args.stage1_preset}  combos={len(grid1)}")
    res1, best1_dir = sweep(root, "1", args, grid1, args.limit_stage1)
    if not res1:
        print("[fatal] stage1 produced no results; aborting")
        sys.exit(3)
    best1 = res1[0]

    # 阶段2：细化邻域
    if args.run_stage2:
        grid2 = build_grid_stage2(best1, args)
        print(f"[info] stage2 refine around best1  combos={len(grid2)}")
        res2, best2_dir = sweep(root, "2", args, grid2, args.limit_stage2)
        if res2:
            (root / "BEST").write_text(res2[0]["out_dir"])  # 指向最终最优
            with open(root / "best_params.json", "w") as f:
                json.dump(res2[0], f, indent=2)
            print(f"[done] final best saved -> {root / 'best_params.json'}")
        else:
            # 若细化失败，保留阶段1结果
            (root / "BEST").write_text(best1["out_dir"])
            with open(root / "best_params.json", "w") as f:
                json.dump(best1, f, indent=2)
            print("[warn] stage2 had no results; kept stage1 best")
    else:
        (root / "BEST").write_text(best1["out_dir"])  # 仅阶段1
        with open(root / "best_params.json", "w") as f:
            json.dump(best1, f, indent=2)
        print(f"[done] best (stage1 only) -> {root / 'best_params.json'}")


if __name__ == "__main__":
    main()
