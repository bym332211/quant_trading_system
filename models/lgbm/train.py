#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
models/lgbm/train.py  (rank 版修复：连续收益 -> 逐日离散等级)

- 读 features parquet
- 构建训练/验证集（winsor/zscore；可选 label 中性化）
- 训练 LightGBM：reg（回归）或 rank（lambdarank）
- rank 模式下：将每个交易日的连续 y 映射为 rank_bins 个整型等级

示例（rank）：
python models/lgbm/train.py \
  --features_path artifacts/features_day.parquet \
  --label y_fwd_5 \
  --train_start 2014-01-01 --train_end 2018-12-31 \
  --valid_start 2019-01-01 --valid_end 2019-12-31 \
  --features "mom_20,vol_20,vwap_spread,ma20_gap,logv_zn_20,rng_hl,oc" \
  --winsor 0.01 --zscore \
  --neutralize_on label --exposures "mkt_beta_60,ln_dollar_vol_20,ind_*,liq_bucket_*" \
  --task rank --rank_bins 5 \
  --out_model artifacts/models/lgbm_rank_y5.pkl
"""

from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

from dataset import DatasetBuilder, Config, DataConfig, PreprocessConfig, NeutralizeConfig


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_path", default="artifacts/features_day.parquet")
    ap.add_argument("--label", default="y_fwd_5")
    ap.add_argument("--features", default="mom_20,vol_20,vwap_spread,ma20_gap,logv_zn_20,rng_hl,oc",
                    help="逗号分隔；支持通配符（在 dataset 中展开）")
    ap.add_argument("--winsor", type=float, default=0.01)
    ap.add_argument("--zscore", action="store_true", default=True)
    ap.add_argument("--no-zscore", dest="zscore", action="store_false")

    ap.add_argument("--neutralize_on", default="label", choices=["off","label"])
    ap.add_argument("--exposures", default="mkt_beta_60,ln_dollar_vol_20,ind_*,liq_bucket_*")
    ap.add_argument("--ridge_lambda", type=float, default=1e-8)

    ap.add_argument("--train_start", required=True)
    ap.add_argument("--train_end", required=True)
    ap.add_argument("--valid_start", required=True)
    ap.add_argument("--valid_end", required=True)

    ap.add_argument("--task", choices=["reg","rank"], default="reg")
    ap.add_argument("--rank_bins", type=int, default=5, help="rank 模式下每日日内等级桶数（>=2）")

    ap.add_argument("--lgb_params", default="",
                    help='JSON 字符串；覆盖默认参数。例如：\'{"n_estimators":2000,"learning_rate":0.05}\'')
    ap.add_argument("--out_model", required=True)
    ap.add_argument("--out_info", default="")
    return ap.parse_args()


def build_cfg(args) -> Config:
    feats = tuple([s.strip() for s in args.features.split(",") if s.strip()])
    expos = tuple([s.strip() for s in args.exposures.split(",") if s.strip()])

    cfg = Config(
        data=DataConfig(
            features_path=args.features_path,
            label=args.label,
            features=feats,
            start=min(args.train_start, args.valid_start),
            end=max(args.train_end, args.valid_end),
            min_price=1.0,
            min_adv_usd=1e6,
            min_names_per_day=100,
        ),
        preprocess=PreprocessConfig(
            winsor=args.winsor,
            zscore=args.zscore,
            neutralize=NeutralizeConfig(
                enable=(args.neutralize_on != "off"),
                on=args.neutralize_on,
                exposures=expos,
                ridge_lambda=args.ridge_lambda,
                center_exposures=True,
                drop_dummy=True,
            ),
        ),
    )
    return cfg


def make_groups(panel: pd.DataFrame) -> list[int]:
    """给 LGBMRanker 构造逐日 group sizes；要求数据按日期块连续."""
    sizes = panel.groupby("datetime", sort=False)["instrument"].size().tolist()
    return [int(x) for x in sizes]


def labels_to_daily_grades(panel: pd.DataFrame, y: pd.Series, bins: int = 5) -> pd.Series:
    """
    将连续 y 转成逐日离散等级（0..bins-1，int）。
    - 优先用 qcut 分位桶；若当天样本太少或重复分位，则回退到按百分位等宽分桶。
    - 返回与 panel 行顺序对齐的 Int64（可能含 NA；调用方需据此过滤）。
    """
    assert len(panel) == len(y)
    df = panel.copy()
    df["y"] = y.values

    def _per_day(s: pd.Series) -> pd.Series:
        # 先对 y 做稳定 rank（避免大量相等值）
        r = s.rank(method="first")
        try:
            q = pd.qcut(r, q=max(2, bins), labels=False, duplicates="drop")
            return q.astype("float")
        except ValueError:
            # 回退：按等宽百分位
            n = len(r)
            if n <= 1:
                return pd.Series([np.nan] * n, index=s.index, dtype="float")
            pr = (r - 1) / (n)  # [0,1)
            b = np.floor(pr * max(2, bins)).astype(float)
            b[b == bins] = bins - 1  # 边界保护
            return pd.Series(b, index=s.index, dtype="float")

    lab = df.groupby("datetime", sort=False)["y"].apply(_per_day)
    # groupby-apply 会改变索引，重建与原顺序对齐
    lab.index = df.index
    return lab.astype("float")


def main():
    args = parse_args()
    cfg = build_cfg(args)
    ds = DatasetBuilder(cfg)

    df = ds.load_features()

    # 训练集
    X_tr, y_tr, p_tr = ds.build_train_xy(
        df, start=args.train_start, end=args.train_end,
        use_resid_label=(args.neutralize_on == "label"),
        feature_cols=cfg.data.features,
        drop_na=True,
    )
    # 验证集
    X_va, y_va, p_va = ds.build_train_xy(
        df, start=args.valid_start, end=args.valid_end,
        use_resid_label=(args.neutralize_on == "label"),
        feature_cols=cfg.data.features,
        drop_na=True,
    )

    print(f"[data] train: {X_tr.shape}, valid: {X_va.shape}")

    # LightGBM
    import lightgbm as lgb

    # 默认参数
    base_params = dict(
        n_estimators=4000,
        learning_rate=0.05,
        max_depth=-1,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=0.0,
        random_state=2025,
        n_jobs=-1,
    )
    if args.lgb_params:
        base_params.update(json.loads(args.lgb_params))

    if args.task == "reg":
        model = lgb.LGBMRegressor(**base_params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric="rmse",
            callbacks=[lgb.early_stopping(stopping_rounds=200, verbose=True)]
        )
    else:
        # ------- 将连续标签转为逐日离散等级（int） -------
        if args.rank_bins < 2:
            raise ValueError("--rank_bins 必须 >= 2")
        y_tr_grade = labels_to_daily_grades(p_tr, y_tr, bins=args.rank_bins)
        y_va_grade = labels_to_daily_grades(p_va, y_va, bins=args.rank_bins)

        # 过滤当天无法分桶（样本过少导致 NA）的样本
        m_tr = y_tr_grade.notna().to_numpy()
        m_va = y_va_grade.notna().to_numpy()

        X_tr2 = X_tr.loc[m_tr]
        y_tr2 = y_tr_grade.loc[m_tr].astype(int).to_numpy()
        p_tr2 = p_tr.loc[m_tr]

        X_va2 = X_va.loc[m_va]
        y_va2 = y_va_grade.loc[m_va].astype(int).to_numpy()
        p_va2 = p_va.loc[m_va]

        g_tr = make_groups(p_tr2)
        g_va = make_groups(p_va2)

        print(f"[rank] filtered train: {X_tr2.shape}, valid: {X_va2.shape} | bins={args.rank_bins}")

        # 强制使用 lambdarank；评估 NDCG
        model = lgb.LGBMRanker(objective="lambdarank", **base_params)
        model.fit(
            X_tr2, y_tr2,
            group=g_tr,
            eval_set=[(X_va2, y_va2)],
            eval_group=[g_va],
            eval_at=[5, 10, 20],
            callbacks=[lgb.early_stopping(stopping_rounds=200, verbose=True)]
        )

        # 用经过过滤/排序后的特征名保存（与 reg 一致）
        X_tr = X_tr2
        X_va = X_va2

    # 保存
    out_model = Path(args.out_model).expanduser().resolve()
    out_model.parent.mkdir(parents=True, exist_ok=True)
    try:
        import joblib
        joblib.dump({"model": model, "features": list(X_tr.columns), "task": args.task}, out_model)
    except Exception:
        import pickle
        with open(out_model, "wb") as f:
            pickle.dump({"model": model, "features": list(X_tr.columns), "task": args.task}, f)

    print(f"[done] saved model -> {out_model}")

    # 记录 info
    info = {
        "task": args.task,
        "params": base_params,
        "features": list(X_tr.columns),
        "train_rows": int(X_tr.shape[0]),
        "valid_rows": int(X_va.shape[0]),
        "best_iteration": getattr(model, "best_iteration_", None),
        "feature_importances_": {c:int(v) for c,v in zip(X_tr.columns, getattr(model, "feature_importances_", np.zeros(len(X_tr.columns))))},
    }
    out_info = args.out_info or (str(out_model).replace(".pkl", "_info.json"))
    with open(out_info, "w") as f:
        json.dump(info, f, indent=2)
    print(f"[info] saved -> {out_info}")


if __name__ == "__main__":
    main()
