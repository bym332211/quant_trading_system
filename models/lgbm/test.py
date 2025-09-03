# models/lgbm/test.py
from dataset import DatasetBuilder, Config, DataConfig, PreprocessConfig, NeutralizeConfig

cfg = Config(
    data=DataConfig(
        features_path="artifacts/features_day.parquet",
        label="y_fwd_5",
        # 也可用通配符，比如 ("mom_*","vol_*","ma*_gap","vwap_spread","oc","rng_hl")
        features=("mom_20","vol_20","vwap_spread","ma20_gap","logv_zn_20","rng_hl","oc"),
        min_price=1.0,
        min_adv_usd=1e6,
        min_names_per_day=100,
        start="2014-01-01",
        end="2024-12-31",
    ),
    preprocess=PreprocessConfig(
        winsor=0.01,
        zscore=True,
        neutralize=NeutralizeConfig(
            enable=True,
            on="label",
            exposures=("mkt_beta_60","ln_dollar_vol_20","ind_*","liq_bucket_*"),
            ridge_lambda=1e-8,
            center_exposures=True,
            drop_dummy=True,
        )
    )
)

ds = DatasetBuilder(cfg)

# 1) 读取特征（此时一定包含 instrument / datetime）
df = ds.load_features()
print("[check] columns has datetime?", "datetime" in df.columns, "| rows:", len(df))

# 2) （可选）若你真的要筛列，一定把 instrument / datetime / label + features + 暴露 都带上
label = cfg.data.label
feat_cols = list(cfg.data.features)
expos_cols = ["mkt_beta_60","ln_dollar_vol_20"] + \
             [c for c in df.columns if c.startswith("ind_")] + \
             [c for c in df.columns if c.startswith("liq_bucket_")]
keep = (["instrument","datetime",label] + feat_cols + expos_cols)
keep = [c for c in keep if c in df.columns]  # 防御：有些列可能不存在
df = df[keep].copy()
print("before build_train_xy -> has datetime?", "datetime" in df.columns)
print("first 20 cols:", list(df.columns)[:20])
# 3) 构建训练集
X, y, panel = ds.build_train_xy(
    df,
    start="2016-01-01",
    end="2019-12-31",
    use_resid_label=True
)
print("[ok] X/y shapes:", X.shape, y.shape, "| panel:", panel.shape)
print(panel.head())
