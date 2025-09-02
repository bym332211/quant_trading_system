# strategies/lgbm_weekly/predict.py
import argparse, os, json, warnings
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# try to import lightgbm; fallback to sklearn HGB if not present
try:
    import lightgbm as lgb
    _USE_LGB = True
except Exception:
    from sklearn.ensemble import HistGradientBoostingRegressor
    _USE_LGB = False

def load_config(path:str)->dict:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _weekday(d: pd.Timestamp)->int:
    return int(pd.Timestamp(d).weekday())

def _to_ts(x): return pd.Timestamp(x).tz_localize(None)

def _winsorize_zscore(df: pd.DataFrame, cols, by="datetime", winsor=0.01, do_z=True):
    if winsor is None and not do_z:
        return df
    dfs = []
    for dt, g in df.groupby(by):
        gg = g.copy()
        for c in cols:
            s = gg[c].astype(float)
            if winsor is not None and 0 < winsor < 0.5:
                lo, hi = s.quantile([winsor, 1 - winsor])
                s = s.clip(lo, hi)
            if do_z:
                mu, std = s.mean(), s.std(ddof=0)
                if std == 0 or np.isnan(std):
                    gg[c] = 0.0
                else:
                    gg[c] = (s - mu) / std
            else:
                gg[c] = s
        dfs.append(gg)
    return pd.concat(dfs, axis=0)

def _neutralize(df: pd.DataFrame, cols, exposures, by="datetime"):
    """Cross-section neutralization via OLS residuals: col ~ exposures"""
    if not exposures:
        return df
    import numpy.linalg as LA
    dfs = []
    for dt, g in df.groupby(by):
        X = g[exposures].astype(float).values
        if X.ndim != 2 or X.shape[1] == 0:
            dfs.append(g); continue
        X = np.nan_to_num(X, nan=0.0)
        # add intercept
        X = np.c_[np.ones(len(g)), X]
        XtX = X.T @ X
        try:
            XtX_inv = LA.pinv(XtX)
        except Exception:
            dfs.append(g); continue
        for c in cols:
            y = g[c].astype(float).values
            y = np.nan_to_num(y, nan=np.nanmean(y) if np.isnan(y).any() else 0.0)
            beta = XtX_inv @ (X.T @ y)
            y_hat = X @ beta
            resid = y - y_hat
            g[c] = resid
        dfs.append(g)
    return pd.concat(dfs, axis=0)

def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def _week_rebalances(trading_dates: pd.DatetimeIndex, first_rebalance: pd.Timestamp, weekday:int):
    first = pd.Timestamp(first_rebalance)
    first = first.tz_localize(None)
    ds = [d for d in trading_dates if (d >= first and _weekday(d) == weekday)]
    return pd.DatetimeIndex(ds)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)

    paths = cfg["paths"]
    feats_path = os.path.expanduser(paths["features_day"])
    preds_dir = os.path.expanduser(paths["preds_dir"])
    _ensure_dir(preds_dir)

    # Load features/labels
    df = pd.read_parquet(feats_path)
    # enforce schema
    for col in ["instrument", "datetime"]:
        assert col in df.columns, f"missing column: {col}"
    df["datetime"] = pd.to_datetime(df["datetime"], utc=False)
    df.sort_values(["datetime","instrument"], inplace=True)
    df = df.drop_duplicates(["datetime","instrument"])

    # filter universe by ADV if needed
    min_adv = cfg.get("universe", {}).get("min_adv", None)
    if min_adv:
        adv_col = "adv_20"
        if adv_col in df.columns:
            df = df[df[adv_col] >= float(min_adv)]

    # select feature columns
    y_col = cfg["data"]["label"]
    feat_cols = cfg["data"]["features"]
    available = [c for c in feat_cols if c in df.columns]
    missing = set(feat_cols) - set(available)
    if missing:
        print(f"[warn] missing features ignored: {sorted(list(missing))}")
    feat_cols = available
    assert y_col in df.columns, f"label {y_col} not in dataframe"

    # pre-process per cross section
    pp = cfg.get("preprocess", {})
    winsor = pp.get("winsor", 0.01)
    do_z = bool(pp.get("zscore", True))
    df_proc = _winsorize_zscore(df[["datetime","instrument"]+feat_cols].copy(),
                                feat_cols, winsor=winsor, do_z=do_z)
    if pp.get("neutralize", {}).get("enable", False):
        exposures = pp["neutralize"].get("exposures", [])
        exposures = [e for e in exposures if e in df.columns]
        if exposures:
            # attach exposures
            df_proc = df_proc.merge(df[["datetime","instrument"]+exposures],
                                    on=["datetime","instrument"], how="left")
            df_proc = _neutralize(df_proc, feat_cols, exposures)
            df_proc = df_proc[["datetime","instrument"]+feat_cols]

    # merge label (label不做标准化)
    df_proc = df_proc.merge(df[["datetime","instrument", y_col]],
                            on=["datetime","instrument"], how="left")

    # derive trading calendar
    all_dates = pd.DatetimeIndex(df_proc["datetime"].drop_duplicates().sort_values())
    wf = cfg["walk_forward"]
    train_start = _to_ts(wf["train_start"])
    first_reb = _to_ts(wf["first_rebalance"])
    end_date = _to_ts(wf["end_date"])
    weekday = int(wf.get("rebalance_weekday", 4))
    max_train_years = wf.get("max_train_years", None)
    min_train_obs = int(wf.get("min_train_obs", 0))

    cal = all_dates[(all_dates>=train_start) & (all_dates<=end_date)]
    rebalances = _week_rebalances(cal, first_reb, weekday)

    # rolling train → predict on each rebalance date
    rows = []
    for i, reb_dt in enumerate(rebalances):
        # time guards: last train date < rebalance date
        trn_end = reb_dt - pd.Timedelta(days=1)
        trn_df = df_proc[(df_proc["datetime"]>=train_start) & (df_proc["datetime"]<=trn_end)]

        # optional sliding window by years
        if max_train_years:
            lo = trn_end - pd.DateOffset(years=int(max_train_years))
            trn_df = trn_df[trn_df["datetime"]>=lo]

        # drop NA label rows
        trn_df = trn_df.dropna(subset=[y_col])
        if len(trn_df) < max(min_train_obs, 1000):
            print(f"[skip] {reb_dt.date()} insufficient train obs: {len(trn_df)}")
            continue

        X = trn_df[feat_cols].values.astype(np.float32)
        y = trn_df[y_col].values.astype(np.float32)

        # fit model
        if _USE_LGB:
            params = cfg["model"]["params"]
            model = lgb.LGBMRegressor(**params, random_state=cfg.get("seed", 42))
            # LightGBM 直接fit（横截面堆叠后的panel），早停对时序意义不大，先不启用
            model.fit(X, y)
        else:
            model = HistGradientBoostingRegressor(random_state=cfg.get("seed",42))
            model.fit(X, y)

        # predict cross-section at rebalance date
        cs = df_proc[df_proc["datetime"]==reb_dt].copy()
        if cs.empty:
            print(f"[warn] no cross-section at {reb_dt.date()}")
            continue
        X_cs = cs[feat_cols].values.astype(np.float32)
        score = model.predict(X_cs)
        cs["score"] = score

        # rank (1=best)
        cs["rank"] = cs["score"].rank(ascending=False, method="first").astype(int)

        # optional weight hint（等权Top-K作为提示）
        top_k = int(cfg["ranking"].get("top_k", 20))
        if cfg["ranking"].get("use_weight_hint", False):
            w = np.zeros(len(cs), dtype=float)
            idx = cs.sort_values("score", ascending=False).index[:top_k]
            w[np.isin(cs.index, idx)] = 1.0 / top_k
            cs["weight_hint"] = w

        # meta
        if cfg.get("outputs",{}).get("write_meta", True):
            meta = {
                "rebalance_dt": str(reb_dt.date()),
                "model": "lgbm" if _USE_LGB else "sklearn_hgbr",
                "features": feat_cols,
                "train_start": str(train_start.date()),
                "train_end": str(trn_end.date()),
                "rows": int(len(trn_df)),
            }
            cs["meta"] = json.dumps(meta, ensure_ascii=False)

        cs_out = cs[["instrument"] + (["score","rank"] if cfg["outputs"].get("write_rank", True) else ["score"])
                   + (["weight_hint"] if "weight_hint" in cs.columns else [])
                   + (["meta"] if "meta" in cs.columns else [])].copy()
        cs_out.insert(0, "rebalance_dt", reb_dt)

        # write one parquet per rebalance (便于回测按日读取/切片)
        out_path = os.path.join(preds_dir, f"preds_{reb_dt.strftime('%Y%m%d')}.parquet")
        cs_out.to_parquet(out_path, index=False)
        rows.append(len(cs_out))
        print(f"[ok] {reb_dt.date()} -> {out_path} ({len(cs_out)} rows)")

    print(f"[done] total rebalance files: {len(rows)}")

if __name__ == "__main__":
    main()
