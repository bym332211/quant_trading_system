# README — Quant Trading System (US Equities → Crypto-ready)

本项目旨在构建一个**可扩展的量化交易系统**，以 **美股（S&P 500）高夏普** 为首要目标，并平滑扩展到**加密货币**。  
整体分四层：**数据 → 特征/标签 → 模型/选股 → 回测/执行**。

- **数据层**：下载并清洗多频数据（day/1min/5min…），以 Qlib `.bin` 为统一存储；**OHLC/VWAP 统一前复权口径**。  
- **特征层**：从 Qlib 读取（已复权）生成核心因子与监督标签（兼容日/分钟）。  
- **模型层**：优先落地 **LGBM 周/月节奏的横截面选股**（可扩展 XGB/Ridge/Linear）。  
- **回测层**：基于 **Backtrader** 的统一回测框架 + 诊断 KPI（Sharpe/CAGR/MDD/IC/Turnover/成本等）+ **Walk-Forward**；执行侧预留 **Kronos**（分钟级入场/成本建模）。

---

## 当前状态（v3.4.x）

- ✅ **可用回测引擎**：支持收盘/次日开盘成交、`exec_lag`、中性化（beta/行业/市值/流动性）、平滑、目标波动（年化）、杠杆上限、**%ADV 限速**、**单票上限**。  
- ✅ **硬上限 Water-filling**：多/空两腿分别做“水位法”精确分配，严格满足**单票上限**且不二次超限（替代“clip+再归一”）。  
- ✅ **短腿择时（基于 SPY 动量）**：使用 **T-1** 信息；**规则说明已与实现一致：`动量 ≤ 阈值` 时允许做空**。  
- ✅ **诊断输出**：每日换手（限速前/后）、%ADV 命中率与裁剪占比、长短腿日度贡献、累计佣金、平均毛多/毛空敞口、长腿/短腿 Sharpe 等。  
- ✅ **成本选项**：基础 bps 佣金与滑点；可扩展 per-share/交易所费/集合竞价点差/SEC fee/借券费（扩展版插件化）。  
- ✅ **数据合约**：Predictions/Features/行情读取的 schema 已固定；避免时间泄漏的流程与注意事项写入本文档。

---

## 目录结构（当前 + 目标形态）

```plaintext
quant_trading_system/
├── data/
│   ├── download_sp500.py
│   ├── build_qlib_us.py
│   ├── build_factors.py                 # 输出 features_{day|1min}.parquet
│   └── sp500_tickers.txt
├── scripts/
│   ├── dump_bin.py
│   ├── rebuild_us_1day.sh
│   ├── rebuild_us_1min.sh
│   └── rebuild_us_5min.sh
├── backtest/
│   ├── engine/
│   │   └── run_backtest.py             # v3.4.x（诊断+water-filling+择时）
│   └── reports/                        # 回测产物
├── strategies/
│   └── lgbm_weekly/                    # 训练/推理（WIP）
├── models/
│   └── lgbm/
│       ├── dataset.py
│       ├── train.py
│       └── inference.py
├── artifacts/
│   ├── features_day.parquet
│   ├── features_1min.parquet           # 预留
│   └── preds/weekly/preds_*.parquet
├── config/
│   └── config.yaml                     # 全局路径/窗口/成本/约束（WIP）
└── tests/                              # 单测（WIP）
```

## 数据结构（Data Contracts，关键字段）
1) 特征/标签（artifacts/features_day.parquet）

- 主键：(instrument, datetime)；均为 UTC tz-naive。

- 必要列（节选）：
instrument, datetime, $open,$high,$low,$close,$vwap,$volume, ret_1, adv_20, ret_5, mom_20, vol_20, mkt_beta_60, ln_dollar_vol_20, ind_*（行业 one-hot）, liq_bucket_*（流动性分桶） …

- 生成策略：所有价格字段为前复权；窗口热身期出现 NA 属正常。

2) 预测（回测读取）

- 文件：artifacts/preds/weekly/preds_YYYY_YYYY.parquet（或多文件合集）

- 列：instrument, datetime (=源预测日), score, [rank]

- 约束：必须为完全样本外（OOS）预测；回测会将“周频锚日” as-of 映射到最近的源预测日，再套用 exec_lag 生成“执行日”。

**防泄漏提示**：训练/验证/测试严格按时间切分，回测区间内不得使用未来信息训练。建议按滑窗/扩窗做 Walk-Forward 训练→预测→回测。

## 注意事项（强烈建议）

- 样本外：确保 --preds 仅包含回测区间的 OOS 预测。

- 锚与日历：周频锚点来自 anchor_symbol（默认 SPY）的交易日历；换标的会改变调仓表。

- 短腿择时：规则为 mom ≤ threshold 时允许做空（已与代码注释一致）。

- 单票上限：若不开 --hard_cap，回退为“归一 + 裁剪”的近似；建议开启以避免二次超限。

- %ADV 限速：过严会显著降低换手与收益率；可观察 per_day_ext.csv 的 adv_clip_* 指标来调参

## 变更记录（最近）

- v3.4.1

  - 新增：短腿择时（SPY 动量，T-1；mom ≤ 阈值 允许做空）

  - 新增：硬上限 Water-filling（多/空两腿精确分配）

  - 新增：逐日诊断扩展（turnover pre/post、ADV 裁剪、腿贡献、累计佣金）

  - 修复：--hard_cap 开关逻辑（不再被强制为 True）

  - 稳健：缺列/空截面/边界情况的容错与回退

- v3.3.x → v3.4.0

  - 目标波动、%ADV 限速、平滑/缓冲、行业/流动性/β中性化

  - 成本参数化与导出、数据契约固化、周频锚点与 as-of 映射

## FAQ

- 为什么“动量 ≤ 阈值时允许做空”？
这是一个宏观/风险基调过滤：弱势/下行环境更鼓励使用空头腿；强势时减少对空头的依赖（降低做空的机会成本与踏空风险）。你也可以改成相反逻辑，只需切换阈值判定。

- ICDF 与 Equal 的选择？
icdf 更强调头部、对尾部分配更克制；equal 更分散，换手较低。结合 max_pos_per_name + hard_cap 可取得稳健分散。

- 结果“看起来不错”是否因含样本期？
若 --preds 覆盖 2020–2024 的训练内预测，确实会乐观。务必只用 OOS 预测做回测；推荐 Walk-Forward。

## 层级网格搜索（Sharpe）

- 目标：先用粗粒度网格快速探索方向，再在最优点附近细化，最大化 Sharpe。
- 提供两个脚本：
  - `scripts/sweep_sharpe_focus.py`：单阶段扫参数（`coarse|medium|focused|fine`）。
  - `scripts/hierarchical_sweep_sharpe.py`：两阶段（粗 → 细化）自动围绕最优点构建邻域。

示例（两阶段 + 邻域扩展）：

```
python scripts/hierarchical_sweep_sharpe.py \
  --qlib_dir "/home/ec2-user/.qlib/qlib_data/us_data" \
  --preds "artifacts/preds/weekly/predictions.parquet" \
  --features_path "artifacts/features_day.parquet" \
  --start "2017-01-01" --end "2024-12-31" \
  --out_root "backtest/reports/hier_sweep" \
  --stage1_preset coarse --run_stage2 \
  --try_both_weight_schemes --try_alt_neutralize
```

细化阶段可通过以下参数调节邻域大小：
- `--n_topk_neighbors`/`--step_topk`（`top_k` 左右邻点与步长）
- `--n_tv_neighbors`/`--step_tv`（`target_vol`）
- `--n_buf_neighbors`/`--step_buf`（`membership_buffer`）
- `--n_eta_neighbors`/`--step_eta`（`smooth_eta`）
- `--n_cap_neighbors`/`--step_cap`（`max_pos_per_name`）

输出（目录：`backtest/reports/hier_sweep/<timestamp>/`）：
- `s1_summary.csv`、`best_params_s1.json`（阶段1）
- `s2_summary.csv`、`best_params_s2.json`（阶段2，如启用）
- `BEST` 文件指向最终最优 run 目录

## 任务看板（简版）

- P0：端到端闭环 + 周频 LGBM 基线 + 报告 ✅（回测+诊断已就绪，训练/预测脚本 WIP）

- P1：协方差/TE 约束、冲击/成交回填、报告增强（进行中）

- P2：执行器（分钟）、监控/CI、Crypto 适配（计划中）
