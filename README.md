# README — Quant Trading System (US Equities → Crypto-ready)

本项目旨在构建一个**可扩展的量化交易系统**，以 **美股（S&P 500）高夏普** 为首要目标，并平滑扩展到**加密货币**。  
整体分四层：**数据 → 特征/标签 → 模型/选股 → 回测/执行**。

- **数据层**：下载并清洗多频数据（day/1min/5min…），以 Qlib `.bin` 为统一存储；**OHLC/VWAP 写入前复权价**。  
- **特征层**：从 Qlib 读入（已复权）生成核心因子与监督标签（兼容日/分钟）。  
- **模型层**：优先落地 **LGBM 周/月节奏的横截面选股**（可扩展 XGB/Ridge/Linear）。  
- **回测层**：基于 **Backtrader** 的统一回测框架 + KPI（Sharpe/CAGR/MDD/IC 等）+ **Walk-Forward**；**Kronos 执行器**负责分钟级入场出场与成本建模。

---

## 目录结构（目标形态）

```plaintext
quant_trading_system/
├── data/
│   ├── download_sp500.py          # YahooFinance 多频下载
│   ├── build_qlib_us.py           # 标准化为 Qlib CSV，并 dump 到 .bin（OHLC/VWAP=前复权）
│   ├── build_factors.py           # 从 Qlib 读取生成因子/标签（day/1min 兼容）
│   └── sp500_tickers.txt
├── scripts/
│   ├── dump_bin.py                # 本地化 Qlib dump_bin（已验证）
│   ├── rebuild_us_1day.sh         # 重建日频（含复权一致性校验）
│   ├── rebuild_us_1min.sh         # 重建 1min（含日历/别名/校验）
│   └── rebuild_us_5min.sh         # 重建 5min（含校验与降级）
├── backtest/
│   ├── engine/
│   │   ├── bt_adapter.py          # 数据适配（行情/预测→Backtrader feeds）
│   │   ├── broker_costs.py        # 成本/点差/冲击/滑点模型
│   │   ├── position_sizing.py     # Score→Weights、目标波动、Risk-Parity
│   │   ├── risk_manager.py        # 约束（单票/行业/净毛敞口/β中性/MDD/VaR/ES）
│   │   ├── analyzers.py           # KPI/IC/RankIC/分层表现
│   │   ├── walk_forward.py        # 训练→预测→回测（滑窗/扩窗）
│   │   └── run_backtest.py        # 统一入口
│   ├── kpis.py                    # KPI 汇总导出（csv/json/md/html）
│   └── reports/                   # 回测产物
├── strategies/
│   ├── lgbm_weekly/               # 周频选股
│   ├── lgbm_monthly/              # 月频选股
│   └── kronos_exec/               # 分钟执行（TWAP/VWAP/MOO/MOC）
├── models/
│   └── lgbm/
│       ├── dataset.py             # 横截面构造 / winsor / zscore / 中性化
│       ├── train.py               # 训练
│       └── inference.py           # 预测
├── artifacts/
│   ├── features_day.parquet       # 日频因子与标签
│   ├── features_1min.parquet      # 分钟因子与标签（执行/短线用）
│   └── preds/                     # 预测结果（回测读取）
├── config/
│   └── config.yaml                # 全局路径/窗口/成本/约束
└── tests/                         # 单测（复权/因子/窗口/回测可复现）
```
## 数据结构说明（Data Contracts）

> 约定所有时间均为 **UTC、tz-naive**；日频为 `YYYY-MM-DD` 的日期型，分钟频为 `YYYY-MM-DD HH:MM:SS` 的时间型。  
> 频率键值统一：`day`、`1min`、`5min`、`15min`、`30min`、`60min`（读取分钟库一律以 `1min` 进入 Qlib）。

---

### 1) 原始下载 CSV（per-ticker，来自 yfinance）

- **路径**：`data/data/sp500/<TICKER>_<interval>.csv`（例如 `AAPL_5m.csv`）  
- **索引/列**：
  | 字段           | 类型        | 说明 |
  |----------------|-------------|------|
  | `Date`         | datetime64  | 可能带 `+00:00`；分钟/日频混在此列 |
  | `open,high,low,close` | float | 原始（**未复权**）价格 |
  | `adj_close`    | float       | yfinance 自动复权收盘价 |
  | `volume`       | float/int   | 成交量 |
  | `dividends`    | float       | 分红（若有） |
  | `stock_splits` | float       | 拆合股（若有） |
  | `adj_open/adj_high/adj_low` | float |（我们下载脚本额外保存）复权 OH/L |

- **不变式**：
  - 行时间戳严格递增；单文件仅一个标的；允许缺失行（停牌/非交易时段）。
  - 价格非负，`volume ≥ 0`。

---

### 2) 标准化 CSV（供 dump_bin，per-ticker）

- **路径**：`~/.qlib/source/us_from_yf[_<freq>]/<SYMBOL>.csv`（如 `AAPL.csv`）  
- **列**（**已前复权**写入 OHLC/VWAP）：
  | 字段      | 类型       | 说明 |
  |-----------|------------|------|
  | `date`    | datetime64 | 日频为日期，分钟为秒分辨率；**UTC tz-naive** |
  | `symbol`  | string     | 统一大写（`AAPL`） |
  | `open,high,low,close` | float | **前复权价格**（由 `adj_*` 回推） |
  | `volume`  | float/int  | 成交量 |
  | `factor`  | float      | 复权因子（`adj_close/raw_close` 的平滑版本），仅保留以兼容生态，不再参与后续计算 |
  | `vwap`    | float      | **前复权** VWAP（缺额用 `(H+L+C)/3` 近似） |

- **不变式**：
  - `(symbol, date)` 唯一；`date` 严格递增；价格非负、`volume ≥ 0`；
  - 若原始存在 `adj_*`，则标准化后的 `open/high/low/close/vwap` 必与复权口径一致。

---

### 3) Qlib `.bin` 目录结构（只读接口约定）

- **根目录**：`~/.qlib/qlib_data/us_data[_<freq>]`
```
qlib_data/
├── calendars/
│ ├── day.txt
│ ├── 1min.txt # 有的版本名为 minute.txt（我们在 rebuild 脚本中做了兼容/别名）
├── features/
│ └── aapl/
│ ├── open.day.bin
│ ├── close.day.bin
│ ├── open.1min.bin
│ └── ...
└── instruments/
└── all.txt # 每行一个大写代码
```

- **不变式**：
- `calendars/<freq>.txt` 与 `features/*.<freq>.bin` 频率一致；
- `instruments/all.txt` 的成分必须能在 `features/<sym>/...` 中找到对应字段；
- 分钟库读取频率统一用 `freq='1min'`（Qlib 内部向上聚合）。

---

### 4) 因子/标签产物（Parquet，按频率区分）

- **路径**：
- 日频：`artifacts/features_day.parquet`
- 分钟：`artifacts/features_1min.parquet`（如需）
- **模式**：
| 字段         | 类型       | 说明 |
|--------------|------------|------|
| `instrument` | string     | 证券代码（大写） |
| `datetime`   | datetime64 | UTC tz-naive；为主键之一 |
| `$open,$high,$low,$close,$vwap,$volume` | float | 来自 Qlib 读数（**前复权**） |
| `ret_1,ret_5,ret_20` | float | 基础收益 |
| `mom_5,mom_20,mom_60` | float | 动量 |
| `vol_20,vol_60` | float | 波动率（滚动 std） |
| `rng_hl`     | float | (H-L)/C |
| `logv,logv_zn_20,adv_20` | float | 量能与标准化 |
| `vwap_spread`| float | (C-VWAP)/VWAP |
| `oc,upper_shadow,lower_shadow` | float | 微结构/形态 |
| `ma5_gap,ma20_gap` | float | 均线偏离 |
| `dow`        | int   | 周内序（0-6） |
| `tod_sin,tod_cos` 或 `dom_sin,dom_cos` | float | 分钟/日频季节性编码 |
| `y_fwd_1,y_fwd_5,y_fwd_20` | float | 未来收益标签 |
- **主键**：`(instrument, datetime)` 唯一  
- **不变式**：
- 窗口热身导致的 NA 仅出现在窗口前段与标签尾部；其余特征应为数值（winsorize 后）。
- 所有价格字段已为**前复权**口径。

---

### 5) 预测产物（Predictions，用于回测/执行）

- **路径建议**：
- 周频：`artifacts/preds/weekly/preds_YYYYMMDD.parquet`
- 月频：`artifacts/preds/monthly/preds_YYYYMMDD.parquet`
- **模式**：
| 字段           | 类型       | 说明 |
|----------------|------------|------|
| `rebalance_dt` | datetime64 | 调仓生效日（日频索引） |
| `instrument`   | string     | 证券代码（大写） |
| `score`        | float      | 越大越优的相对得分 |
| `rank`         | int        | 可选；横截面内排序（1=最好） |
| `weight_hint`  | float      | 可选；模型侧建议权重（-1~1） |
| `meta`         | string/json| 可选；版本、特征视图、训练窗信息 |
- **不变式**：
- 对同一 `rebalance_dt`，每个 `instrument` 至多一行；
- 回测引擎将基于 `score`/`weight_hint` + 约束/风险模型生成**目标权重**。

---

### 6) 回测输入/输出（与引擎接口）

- **输入**：
- **行情**：由 Qlib 动态读取 `$open,$high,$low,$close,$vwap,$volume`（前复权）；
- **预测**：见上文 Predictions 合约；
- **配置**：`config/config.yaml`（数据窗、交易成本、目标波动、约束、调仓节奏等）。
- **输出**（`backtest/reports/`）：
- `equity_curve.csv`、`trades.csv`、`weights.csv`
- `kpis.json/md`（Sharpe/CAGR/MDD/Calmar/Turnover/Costs/HitRatio/IC 等）
- Walk-Forward 汇总表（每窗口 KPI 与稳定性指标）

---

### 7) 复权一致性校验（内建于 rebuild 脚本）

- **原则**：  
从 **源 CSV 的 `adj_*`** 与 **Qlib 读出的 OHLC/VWAP** 做等时比对，误差阈值 `tol=1e-4`。
- **判定**：  
`OK`：误差率中位数 < `tol` 且 P95 < `5*tol`；否则 `NG` 并列出样本。
- **适用**：  
日频与分钟库均适用（分钟库抽样若干标的、若干交易日）。

---

### 8) 命名/大小写/缺失处理规范

- **代码/目录**：标的符号在文件系统中按 **小写目录** 存储（Qlib 习惯），在数据行字段 `instrument/symbol` 中为 **大写**。
- **频率**：写入与读取遵循 `day` 与 `1min`；`5min/15min...` 通过 Qlib 的上层聚合或另建库。
- **缺失**：
- 因子窗口热身期允许 NA；模型训练阶段需做填充或剔除；
- 行情读数若缺（停牌）则在回测层禁买/清仓或保留权重（由策略参数决定）。

---

### 9) 加密资产扩展（预留字段）

- **行情扩展字段（后续版本可加入）**：
| 字段           | 类型   | 说明 |
|----------------|--------|------|
| `base_ccy`     | string | 交易对基础币（如 BTC） |
| `quote_ccy`    | string | 计价币（如 USDT） |
| `contract_mult`| float  | 合约乘数/面值（合约品种） |
| `maker_fee,taker_fee` | float | 费率（万分比） |
- **日历**：24/7（不交易日为空集）；分钟库直接以自然分钟为日历。

---



## 特征/标签结构说明（`data/build_factors.py` 输出）

- **文件**：`artifacts/features_{day|1min}.parquet`
- **Schema**
  - `instrument`（string，**大写 TICKER**）
  - `datetime`（timestamp，UTC-naive）
  - 数值特征/标签（float）
- **基础收益**：`ret_1, ret_5, ret_20`
- **动量**：`mom_5, mom_20, mom_60`（价格相对窗口首）
- **波动**：`vol_20, vol_60`（1步收益滚动 std）、`rng_hl = (H - L) / C`
- **价量**：`logv`、`logv_zn_20`（rolling zscore of logV）、`adv_20`（均量）
- **微结构**：`vwap_spread = (C - VWAP)/VWAP`、`oc = (C - O)/O`、`upper_shadow`、`lower_shadow`
- **均值回归**：`ma5_gap, ma20_gap`（价格相对短/长均线偏离）
- **季节性**：
  - **day**：`dom_sin, dom_cos`（月内位置）
  - **1min**：`tod_sin, tod_cos`（交易日内分钟位置，6.5 小时周期）
  - `dow`：周内日（0=周一）
- **标签**：`y_fwd_{1|5|20}`（未来 k 步相对收益）
- **处理**：
  - 窗口热身 NA 合理存在，可选 `winsor`（默认每边 1%）
  - 下游 **横截面处理器** 再做 winsor/zscore/行业/市值中性化

---

## 快速使用

```bash
# 1) 重建数据（举例：日频 / 1min / 5min）
bash scripts/rebuild_us_1day.sh
bash scripts/rebuild_us_1min.sh
bash scripts/rebuild_us_5min.sh

# rebuild 脚本中已包含：
# - 标准化（OHLC/VWAP = 前复权）
# - dump_bin（freq 命名兼容）
# - instruments 自动重建
# - 复权一致性校验（源 CSV adj_* vs Qlib 读出 OHLC）：OK/NG

# 2) 生成因子与标签（日频举例）
python data/build_factors.py --qlib_dir ~/.qlib/qlib_data/us_data \
  --freq day --start 2010-01-01 --end 2100-01-01 \
  --out artifacts/features_day.parquet --winsor 0.01

# 3)（即将提供）训练/预测/回测入口
# python strategies/lgbm_weekly/train.py ...
# python strategies/lgbm_weekly/predict.py ...
# python backtest/engine/run_backtest.py --config config/config.yaml

## # 1) 重建数据（举例：日频 / 1min / 5min）
bash scripts/rebuild_us_1day.sh
bash scripts/rebuild_us_1min.sh
bash scripts/rebuild_us_5min.sh

# rebuild 脚本中已包含：
# - 标准化（OHLC/VWAP=前复权）
# - dump_bin（freq 命名兼容）
# - instruments 自动重建
# - 复权一致性校验（源CSV adj_* vs Qlib 读出 OHLC）：OK/NG

# 2) 生成因子与标签（日频举例）
python data/build_factors.py --qlib_dir ~/.qlib/qlib_data/us_data \
  --freq day --start 2010-01-01 --end 2100-01-01 \
  --out artifacts/features_day.parquet --winsor 0.01

# 3)（即将提供）训练/预测/回测入口
# python strategies/lgbm_weekly/train.py ...
# python strategies/lgbm_weekly/predict.py ...
# python backtest/engine/run_backtest.py --config config/config.yaml
```

## 回测/执行框架（设计要点）

- Backtrader 统一引擎（配置驱动）

  - bt_adapter.py：行情与预测（Parquet）拼接对齐，映射到 feeds

  - Walk-Forward：每窗口 训练→预测→回测，滚动重估协方差与风险预算

- 仓位管理（Position Sizing）

  - ScoreToTargetWeights（分层打分→权重，温度/尾部抑制）

  - VolTargetSizer（目标波动率），RiskParitySizer（行业/桶等风险分配）

- 风险控制（Risk & Constraints）

  - 约束盒：单票/行业/净毛/杠杆；β、风格中性；停牌/低流动性限额

  - 动态守护：MDD/VaR/ES 触发降杠杆；ATR 止损/跟踪止盈

- 成本模型（broker_costs.py）

  - 点差 + 冲击（平方根/成交量函数）+ 佣金，ADV 限速，部分成交

- KPI/分析（analyzers.py / kpis.py）

 - Sharpe/CAGR/MDD/Calmar/Turnover/Costs/HitRatio、IC/RankIC、分层业绩

## 仓位管理与风险控制（实现与接口）

### 1) 仓位管理 (Position Sizing)
- **目标波动率/风险预算**: Vol targeting、Inverse-Vol、Risk-Parity  
- **打分到权重**: 分层打分→权重（带温度/尾部抑制）；不确定度折减  
- **流动性与换手**: L1 换手率；|△权重| ≤ x% ADV  

### 2) 风险与约束 (Risk & Constraints)
- **持仓约束**: 单票/行业上限、净/毛敞口、杠杆、做空比例  
- **中性化**: 对基准β、行业、风险暴露做回归中性化  
- **停牌/流动性**: 停牌屏蔽、低流动性票权重上限  

### 3) 成本与执行 (Broker Costs & Execution)
- **成本**: 点差 + 冲击（平方根/成交量函数）+ 佣金  
- **执行**: TWAP/VWAP/MOO/MOC，ADV 限速，部分成交回填  

### 4) 指标与报告 (KPI/Analytics)
- **收益类**: Sharpe、CAGR、MDD、Calmar、超额收益/TE  
- **交易类**: Turnover、Costs、成交率  
- **信号类**: IC/RankIC、分层表现、稳定性  

# 量化系统任务总表（合并版 · 含优先级）

> 原则（贯彻到全链路）
- **时序切分**：严格按时间滚动（例：2015–2019 训练，2020 验证，2021–2025 测试），Walk-Forward 串联训练→预测→回测。
- **横截面标准化**：每个截面内做去极值（winsor）+ 标准化（zscore），形成统一的 `CrossSectionProcessor`。
- **基准与暴露控制**：标签/特征对行业、β、市值做中性化（训练前回归中性化或训练时加入约束），目标是 **提升夏普**。
- **模型路线**：先上轻量 **Ridge / LGBM / XGBoost** 做 IC&回测基线，再评估序列/多任务模型。

---

## P0（必须优先 · 跑通端到端）
**目标：最小可用闭环，支持周/月调仓，产出可复现实验与 KPI 报告**

- 数据层
  - [x] 多源下载器（yfinance 限制处理/重试）
  - [x] 标准化为 **Qlib**（前复权 OHLC/VWAP，因子补充）
  - [x] `dump_bin` 重建（1min/分钟合并，自动 instruments）
  - [x] 一致性校验（CSV adj_* vs Qlib 读出 OHLC）
  - [ ] 数据更新策略（Append/Replace + 快照）

- 特征&标签
  - [x] **日频** 因子与监督标签（Parquet）
  - [x] `CrossSectionProcessor` v1：winsor/zscore/行业&市值中性化
  - [ ] 特征/标签 **schema 契约** 与版本记录（Data Contracts）

- 模型&推理
  - [x] **LGBM Weekly Baseline**：`models/lgbm/dataset.py → train.py → inference.py`
  - [x] `strategies/lgbm_weekly/predict.py` 输出：`artifacts/preds/weekly/*.parquet`
  - [ ] Ridge / XGBoost 基线（方便 sanity-check 与对比）

- 回测/执行
  - [x] Backtrader 最小闭环：`bt_adapter.py + run_backtest.py`（可读 Parquet 预测 & Qlib 行情）
  - [x] **Position Sizing v1**：`ScoreToTargetWeights`（分层打分→权重，温度/上限）+ `VolTargetSizer`
  - [x] **Risk Constraints v1**：单票/行业/净毛/杠杆约束，β 对基准回归中性
  - [x] **Costs v1**：点差 + 线性冲击 + 佣金；**%ADV 限速**
  - [x] **Walk-Forward v1**：时间切分，训练→预测→回测 **严格时序滚动**

- 评估&报告
  - [x] KPI 汇总：Sharpe/CAGR/MDD/Calmar/TE、成本、Turnover、IC/RankIC、分层表现
  - [ ] 报告落地：`reports/run_YYYYMMDD/`（csv/json/md/html）

**交付物（P0 完成判据）**
- `run_backtest.py` 一键产出：预测→回测→KPI
- README：数据→特征→模型→回测链路与参数样例
- 一个完整实验目录（含日志、快照、KPI 导出）

---

## P1（性能与稳健性提升）
**目标：更强表现 + 更靠近实盘的风险/协方差与报告体系**

- 模型与验证
  - [ ] LGBM 参数网格 / 学习曲线；OOS IC 稳定性监控
  - [ ] Ridge / XGBoost 对比报告；分层 IC（行业、市值分组）
  - [ ] 月频 Baseline（与周频并行），统一切分与评估

- 风险与协方差
  - [ ] **EWMA 协方差** 与 **Shrinkage**；**TE/风险预算** 约束（可选）
  - [ ] 权重平滑 & 换手惩罚（惩罚项或启发式）

- 成本与执行
  - [ ] 冲击模型升级（平方根/成交量函数）与成交回填策略
  - [ ] 未成交残单处理（收盘 MOC/MOO 兜底；限价/市价切换阈值）

- 报告与可视化
  - [ ] KPI 报告增强：时序图、分桶收益、稳定性雷达
  - [ ] 实验管理：配置/版本/数据契约快照（便于复现实验）

---

## P2（执行深化与工程化）
**目标：更贴近生产的执行、分钟级特征、监控与多市场扩展**

- 执行系统
  - [ ] **Kronos 执行器 v1**：TWAP/VWAP/MOO/MOC，自适应节奏与预期成交监控
  - [ ] 成交不及预期的动态限速/再撮合与风控联动

- 高频特征
  - [ ] **1 分钟** 特征产线：`features_1min.parquet`（执行侧特征）

- 监控与稳定性
  - [ ] 数据质量仪表盘（缺 bar/重复/时间跳跃/复权漂移）
  - [ ] 重建日志与快照汇总（数据→特征→模型→回测全链路）

- 多市场&CI
  - [ ] Crypto 接口（CCXT）、24/7 日历、费率/资金费与风险参数差异
  - [ ] 单测覆盖（复权/窗口一致性/回测回现）、CI、FAQ（timezone/freq 命名/yfinance 限制/复权）

---

## 看板（跨层遗漏检查）
- [x] 数据下载与 Qlib 标准化、dump_bin 重建与一致性校验
- [x] 日频因子/标签流水线；分钟级（P2）
- [x] Walk-Forward 串联训练-预测-回测（严格时序）
- [x] Position Sizing / Risk Constraints / Costs v1
- [x] Backtrader 最小闭环；预测文件 → 回测桥接
- [ ] 协方差/TE 约束、权重平滑（P1）
- [ ] 执行器 Kronos v1 与动态节奏（P2）
- [ ] 报告导出与可视化增强、实验管理（P1）

---

## 里程碑（建议）
- **M0（P0 完成）**：端到端闭环 + 周频 LGBM 基线 + KPI 报告
- **M1（P1 完成）**：协方差/TE 约束、成本/执行增强、报告增强
- **M2（P2 完成）**：Kronos 执行器、分钟特征、监控/CI、Crypto 适配



 