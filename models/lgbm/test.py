from dataset import DatasetBuilder

cfg = {
  "data": {
    "features_path": "artifacts/features_day.parquet",
    "label": "y_fwd_5",
    "features": ["mom_20","vol_20","vwap_spread","ma20_gap","logv_zn_20","rng_hl","oc"],
    "start": "2014-01-01", "end": "2024-12-31",
    "min_price": 1.0, "min_adv_usd": 1e6, "min_names_per_day": 100,
  },
  "preprocess": {
    "winsor": 0.01,
    "zscore": True,
    "neutralize": {
      "enable": True,
      "on": "label",
      "exposures": ["mkt_beta_60","ln_dollar_vol_20","ind_*","liq_bucket_*"],
      "ridge_lambda": 1e-6,
      "center_exposures": True,
      "drop_dummy": True
    }
  }
}

ds = DatasetBuilder(cfg)
df = ds.load_features()
X, y, panel = ds.build_train_xy(df, start="2016-01-01", end="2019-12-31", use_resid_label=True)

# 推理某一天
X_t, panel_t = ds.build_cross_section(df, "2020-06-30")
