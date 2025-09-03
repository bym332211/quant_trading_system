#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
models/lgbm/inference.py

- 载入已训练模型
- 按日期区间，逐日截面构建特征（与训练同口径：winsor/zscore）
- 输出预测 score（可选逐日标准化/排名），写入 parquet

示例：
python models/lgbm/inference.py \
  --model artifacts/models/lgbm_reg_y5.pkl \
  --features_path artifacts/features_day.parquet \
  --start 2020-01-01 --end 2020-12-31 \
  --features "mom_20,vol_20,vwap_spread,ma20_gap,logv_zn_20,rng_hl,oc" \
  --winsor 0.01 --zscore \
  --out artifacts/preds/weekly/preds_20200101_20201231.parquet \
  --xsec_standardize --rank_within_day
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
    ap.add_argument("--features", default="mom_20,vol_20,vwap_spread,ma20_gap,logv_zn_20,rng_hl,oc")
    ap.add_argument("--label", default="y_fwd_5")
    ap.add_argument("--winsor", type=float, default=0.01)
    ap.add_argument("--zscore", action="store_true", default=True)
    ap.add_argument("--no-zscore", dest="zscore", action="store_false")

    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)

    ap.add_argument("--out", required=True)
    ap.add_argument("--xsec_standardize", action="store_true", help="逐日对 score 做标准化")
    ap.add_argument("--rank_within_day", action="store_true", help="输出 rank（1=最好）")
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


def main():
    args = parse_args()
    model_obj = load_model(args.model)
    model = model_obj["model"]
    task = model_obj.get("task", "reg")
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
            neutralize=NeutralizeConfig(enable=False),  # 推理不做 label 中性化
        ),
    )
    ds = DatasetBuilder(cfg)
    df = ds.load_features()

    # 只取需要列，避免无关列占内存
    need_cols = set(["instrument","datetime"]) | set(feats)
    df = df[[c for c in df.columns if c in need_cols]].copy()

    # 逐日截面：winsor & zscore（与训练一致）
    # 用 build_cross_section 的逻辑批量处理（直接在全样本上处理更快）
    # 这里直接复用 ds.cs
    df = ds.cs.winsorize(df, feats)
    df = ds.cs.zscore(df, feats)

    # 特征列顺序要与训练一致
    if trained_feats is not None:
        feat_cols = [c for c in trained_feats if c in df.columns]
    else:
        feat_cols = [c for c in feats if c in df.columns]

    X = df[feat_cols].astype(float)
    scores = model.predict(X)

    out = df[["instrument","datetime"]].copy()
    out["score"] = scores.astype(float)

    # 可选：逐日标准化 & rank
    if args.xsec_standardize:
        g = out.groupby("datetime")
        mu = g["score"].transform("mean")
        sd = g["score"].transform("std")
        out["score"] = ((out["score"] - mu) / sd.replace(0, np.nan)).fillna(0.0)

    if args.rank_within_day:
        out["rank"] = out.groupby("datetime")["score"].rank(ascending=False, method="first").astype(int)

    # 写盘
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.sort_values(["datetime","score"], ascending=[True, False]).to_parquet(out_path, index=False)
    print(f"[done] saved predictions -> {out_path} | rows={len(out)} | cols={list(out.columns)}")


if __name__ == "__main__":
    main()
