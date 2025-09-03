#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
models/lgbm/evaluate.py

对已训练的 LGBM 模型做 OOS 评估：
- 构建 (X, y, panel)（与训练同口径 winsor/zscore/标签中性化）
- 计算整体 RMSE/R2、逐日 IC/RankIC、分桶（10 桶）表现
- 写出 per_day.csv / deciles.csv / summary.json

用法示例：
python models/lgbm/evaluate.py \
  --model artifacts/models/lgbm_reg_y5.pkl \
  --features_path artifacts/features_day.parquet \
  --label y_fwd_5 \
  --start 2019-01-01 --end 2019-12-31 \
  --features "mom_20,vol_20,vwap_spread,ma20_gap,logv_zn_20,rng_hl,oc" \
  --winsor 0.01 --zscore \
  --neutralize_label \
  --out_dir artifacts/reports/eval_y5_2019
"""

from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

from dataset import DatasetBuilder, Config, DataConfig, PreprocessConfig, NeutralizeConfig

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--features_path", default="artifacts/features_day.parquet")
    ap.add_argument("--label", default="y_fwd_5")
    ap.add_argument("--features", default="mom_20,vol_20,vwap_spread,ma20_gap,logv_zn_20,rng_hl,oc")
    ap.add_argument("--winsor", type=float, default=0.01)
    ap.add_argument("--zscore", action="store_true", default=True)
    ap.add_argument("--no-zscore", dest="zscore", action="store_false")
    ap.add_argument("--neutralize_label", action="store_true", help="评估期 y 使用残差标签（与训练一致）")

    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--out_dir", required=True)
    return ap.parse_args()

def load_model(path: str):
    p = Path(path).expanduser().resolve()
    try:
        import joblib
        obj = joblib.load(p)
    except Exception:
        import pickle
        with open(p, "rb") as f:
            obj = pickle.load(f)
    return obj

def daily_corr(a: pd.Series, b: pd.Series, method="pearson") -> float:
    if method == "spearman":
        return a.rank().corr(b.rank())
    return a.corr(b)

def main():
    args = parse_args()
    model_obj = load_model(args.model)
    model = model_obj["model"]
    trained_feats = model_obj.get("features", None)

    feats = tuple([s.strip() for s in args.features.split(",") if s.strip()])
    cfg = Config(
        data=DataConfig(
            features_path=args.features_path,
            label=args.label,
            features=feats,
            start=args.start,
            end=args.end,
            min_price=1.0,
            min_adv_usd=1e6,
            min_names_per_day=50,
        ),
        preprocess=PreprocessConfig(
            winsor=args.winsor,
            zscore=args.zscore,
            neutralize=NeutralizeConfig(enable=args.neutralize_label, on="label",
                                        exposures=("mkt_beta_60","ln_dollar_vol_20","ind_*","liq_bucket_*"),
                                        ridge_lambda=1e-8, center_exposures=True, drop_dummy=True),
        ),
    )
    ds = DatasetBuilder(cfg)
    df = ds.load_features()

    # 构建评估期数据（与训练同口径）
    X, y, panel = ds.build_train_xy(
        df, start=args.start, end=args.end,
        use_resid_label=args.neutralize_label,
        feature_cols=cfg.data.features,
        drop_na=True,
    )
    # 特征列顺序对齐模型
    if trained_feats is not None:
        feat_cols = [c for c in trained_feats if c in X.columns]
        X = X[feat_cols]
    preds = model.predict(X)

    # 整体指标
    y_np = y.to_numpy()
    pred_np = np.asarray(preds, dtype=float)
    rmse = float(np.sqrt(np.mean((pred_np - y_np) ** 2)))
    rmse_baseline0 = float(np.sqrt(np.mean((0.0 - y_np) ** 2)))  # 预测全 0 的基线
    ss_res = float(np.sum((y_np - pred_np) ** 2))
    ss_tot = float(np.sum((y_np - y_np.mean()) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    # 逐日 IC / RankIC
    eval_df = panel.copy()
    eval_df["y"] = y.values
    eval_df["pred"] = preds
    per_day = []
    for dt, g in eval_df.groupby("datetime"):
        if g["y"].nunique() < 2:  # 无法计算相关
            continue
        ic = g["pred"].corr(g["y"])
        ric = g["pred"].rank().corr(g["y"].rank())
        per_day.append({"datetime": dt, "IC": ic, "RankIC": ric, "n": len(g)})
    per_day_df = pd.DataFrame(per_day).sort_values("datetime")

    # 分桶（10 桶）表现：日内分桶 => 当期平均 y，横跨日期再平均
    eval_df["bucket"] = eval_df.groupby("datetime")["pred"].transform(
        lambda s: pd.qcut(s.rank(method="first"), q=10, labels=False, duplicates="drop")
    )
    deciles = (eval_df.dropna(subset=["bucket"])
                      .groupby(["datetime", "bucket"])["y"].mean()
                      .reset_index())
    # 每桶的时序平均收益
    decile_avg = deciles.groupby("bucket")["y"].mean()
    top_bottom = float(decile_avg.get(decile_avg.index.max(), np.nan) -
                       decile_avg.get(decile_avg.index.min(), np.nan))

    # 汇总
    ic_mean = float(per_day_df["IC"].mean()) if not per_day_df.empty else float("nan")
    ic_std  = float(per_day_df["IC"].std(ddof=1)) if not per_day_df.empty else float("nan")
    ric_mean = float(per_day_df["RankIC"].mean()) if not per_day_df.empty else float("nan")
    ric_std  = float(per_day_df["RankIC"].std(ddof=1)) if not per_day_df.empty else float("nan")
    ic_pos_rate = float((per_day_df["IC"] > 0).mean()) if not per_day_df.empty else float("nan")
    ric_pos_rate = float((per_day_df["RankIC"] > 0).mean()) if not per_day_df.empty else float("nan")
    # 简单 t 统计（未做自相关调整）
    ic_t = float(ic_mean / (ic_std / np.sqrt(len(per_day_df)))) if per_day_df.shape[0] > 5 and np.isfinite(ic_std) and ic_std > 0 else float("nan")
    ric_t = float(ric_mean / (ric_std / np.sqrt(len(per_day_df)))) if per_day_df.shape[0] > 5 and np.isfinite(ric_std) and ric_std > 0 else float("nan")

    summary = {
        "rows": int(len(eval_df)),
        "rmse": rmse,
        "rmse_baseline0": rmse_baseline0,
        "r2": r2,
        "ic_mean": ic_mean,
        "ic_std": ic_std,
        "ic_pos_rate": ic_pos_rate,
        "ic_tstat": ic_t,
        "rankic_mean": ric_mean,
        "rankic_std": ric_std,
        "rankic_pos_rate": ric_pos_rate,
        "rankic_tstat": ric_t,
        "decile_top_minus_bottom": top_bottom,
    }

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    per_day_df.to_csv(out_dir / "per_day.csv", index=False)
    decile_avg.to_csv(out_dir / "deciles.csv", header=["avg_y"])
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=lambda x: str(x))

    print("[summary]", json.dumps(summary, indent=2))
    print(f"[saved] {out_dir}/per_day.csv")
    print(f"[saved] {out_dir}/deciles.csv")
    print(f"[saved] {out_dir}/summary.json")

if __name__ == "__main__":
    main()
