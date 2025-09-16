# 回测交易问题：2020-2024 回测但只有 2024 有交易（已定位根因）

## 问题描述

- 运行区间：2020-01-01 至 2024-12-31
- 实际输出：仅 2024 年产生日级净值与订单（per_day/equity 最早 2024-03-27，orders 最早 2024-04-02）
- 预期：2020-2024 每年均应产生交易

## 结论（根因）

Backtrader 会对齐所有数据源的起始日期；当股票池中存在“较晚上市/较晚开始有行情数据”的股票时，系统会以“所有数据源中最晚的首个交易日”作为全局起点。本次股票池包含如 `GEV`、`SOLV` 等 2024 年才有首日行情的标的，导致全局共同起点被推迟到 2024-03-27，从而只有 2024 年产生交易记录。

## 证据

- 预测数据覆盖完整（2020-2024）：`scripts/inspect_preds.py`
  - 年份分布：2020:124927, 2021:125635, 2022:125480, 2023:125209, 2024:126997
- 执行日（周频）映射正常：`scripts/inspect_exec_dates_ascii.py`
  - 每年约 52~53 个执行日，2020-2024 均存在执行日
- 回测产出仅在 2024：`scripts/inspect_out_dir.py`
  - per_day/equity 最早 2024-03-27，orders 最早 2024-04-02
- 所有行情数据源的“共同起始日”为 2024-03-27：`scripts/inspect_latest_feed_start.py`
  - 最晚首日的标的包括：CEG(2022-01-19), KVUE(2023-05-04), VLTO(2023-10-04), SOLV(2024-03-26), GEV(2024-03-27)
- 候选数量健康，排除“过滤过严”导致零交易的可能：`scripts/check_candidate_counts.py`
  - 2020-2024 各年每个执行日映射后平均候选数 ~500，应用行业×流动性过滤后仍 ~105

## 解决方案（按推荐度排序）

1) 在价格数据准备后，过滤“起始日晚于回测开始日”的标的（推荐，改动最小）
- 位置：`strategies/xsecrebalance/data_preparation.py: prepare_price_data`
- 规则示例：仅保留 `df.index.min() <= args['start']` 的标的；或允许小幅宽限（如 ≤ 20 个交易日）
- 影响：保证 Backtrader 全局起点 ≤ 回测开始日，从 2020 年开始产生交易

2) 固定 Universe 为“在 2020-01-01 前已有历史”的静态股票池
- 直接使用预先清洗的名单（如 S&P 500 在役名单的一个时间点截面），并与预测集/价格数据取交集

3) 分段回测
- 将 2020–2023 与 2024 拆分回测，避免 2024 年新上市/分拆标的拉高全局起点

### 方案 C 细化：分段回测与结果合并

- 分段策略：按自然年切分（示例：2020、2021、2022、2023、2024 五段）。
- 合并目标：把各段的订单、持仓、日收益、净值串接，并在“全部区间上”计算 KPI。

实现与注意点：
- 输入与运行
  - 每段使用相同的参数与数据路径，仅调整 `--start/--end`。
  - 推荐每段内仍按“映射后的 exec 日”执行（保持周频/执行逻辑一致）。

- 合并规则（文件级）
  - `per_day_ext.csv`/`equity_curve.csv`：逐段按日期升序拼接；边界日避免重复（上一段包含的最后一天，不再纳入下一段）。
  - `orders.csv`/`positions.csv`：直接按 `datetime` 升序拼接；若边界日同时存在上一段的平仓与下一段的建仓，保留时间顺序（如需可在导出中增加 `seq` 序号以稳定排序）。

- KPI 计算（关键）
  - 不逐段加权或简单平均；而是在“合并后的完整日收益序列”上一次性计算。
  - 年化夏普示例：`sharpe = sqrt(252) * mean(daily_ret) / std(daily_ret)`；年化收益：对合并后的 `cumprod(1 + daily_ret)` 取首尾。
  - MDD/回撤曲线：基于合并后的累计净值计算。

- 资产与头寸的衔接（两种模式）
  - 模式 A（连续组合，推荐但需增强引擎）：
    - 在段末导出“持仓与现金”（或权重），段首读取并作为初始状态继续交易。
    - 引擎需要支持：`--init_cash` 与 `--init_positions`（或通过配置/文件注入）。
    - 优点：最贴近真实连续回测；缺点：需在引擎中实现“导入初始头寸”的能力（当前未实现）。
  - 模式 B（年末清仓、用期末权益作为下一段初始现金，易落地）：
    - 段与段之间假定“在前一段末清仓”，下一段以 `cash = 前段 summary.cash_end` 启动，头寸从空开始。
    - 成本处理：
      - 基础实现：不显式计提边界清仓/重建的交易成本（实现简单，但低估了交易成本）。
      - 改进实现：在段末最后一个 exec 日，强制 `target=0` 触发平仓（计入手续费/滑点）；或者在合并时对边界权重按 `commission_bps+slippage_bps` 估算一次性成本并从边界日收益中扣减。
    - 当前引擎最少改动建议：先实现“读写 cash_end → 下一段 --cash”的现金承接；清仓成本后续补充。

- 边界日处理
  - 推荐“闭区间-开区间”拼接：如 2020 段 `start=2020-01-01,end=2020-12-31`，下一段从 `2021-01-01` 开始，避免重叠。
  - 若使用模式 B 的“强制清仓”，应确保清仓发生在上一段最后一个可执行日。

- 宇宙（universe）差异与一致性
  - 分段策略天然减少“新股拉高共同起点”的问题，但各段 universe 可不同（由各段 preds+价格可用性决定）。
  - KPI 在合并时不需要 universe 一致，只需日收益连续可比。

- 工具化建议
  - 新增脚本 `scripts/run_segmented_backtest.py`：
    - 输入：全局 `start/end`、按年切段或自定义分段列表；初始现金；模式选择（A/B）；是否强制清仓；输出目录。
    - 逻辑：循环调用 `run_backtest.py`，存档每段输出；按规则合并 csv；在合并后的 `per_day` 上重算 KPI，生成“全区间 summary”。
  - 引擎增强（后续）：支持 `--init_positions` 与 `--force_flatten_on_end` 以实现模式 A 或改进的模式 B。

4) 支持动态股票池/动态数据源（架构增强）
- 回测中允许标的在出现数据后再加入数据源（Backtrader 默认强对齐，需要较大工程改造）

## 推荐实施

- 先落地方案 1：在 `prepare_price_data` 载入 `price_map` 后，按最早日期过滤，并打印日志列出被剔除标的及其首日日期，便于复核
- 如需保持 2024 年新股，也可以采用“宽限阈值”与“按需分段”的组合

（补充）若短期内不改引擎，方案 C 建议先采用“模式 B：现金承接 + 年末清仓（可先不计清仓成本）”，并在合并时统一计算 KPI；后续逐步增强到“模式 A：携带头寸连续回测”。

## 验证计划

- 执行一次 2020-2024 全区间回测，确认 per_day/equity 最早日期≤2020-01-02，orders 覆盖 2020 全年
- Spot check：抽样查看 2020Q1 的执行日是否下单；检查 `positions.csv`/`orders.csv` 日期分布

## 环境与路径

- 预测数据：`artifacts/preds/preds_y5_2020_2024.parquet`
- 特征数据：`artifacts/features_day.parquet`
- QLib：`~/.qlib/qlib_data/us_data`
- 回测入口：`backtest/engine/run_backtest.py`

## 调试与辅助脚本（已加入仓库）

- `scripts/inspect_preds.py`：检查预测数据年份覆盖
- `scripts/inspect_qlib.py`：检查 QLib 日历与单标行情范围
- `scripts/inspect_exec_dates_ascii.py`：统计各年执行日数量
- `scripts/check_candidate_counts.py`：统计执行日前后候选数量
- `scripts/inspect_latest_feed_start.py`：找出最晚首日的标的与日期（关键定位）
- `scripts/inspect_out_dir.py`：读取最近回测输出的日期范围

## TODO（落地改造）

- [ ] 在 `prepare_price_data` 增加“起始日过滤 + 被剔除标的日志”
- [ ] 新增配置项：`min_history_start` 或 `max_late_start_days`（可选，默认严格）
- [ ] 回归测试：跑通 2020-2024，确认 2020 年产出交易

—— 每次调查或改动后，更新此 Issue 记录新的证据与结果 ——

## 方案 C 执行结果（分段回测 + 合并KPI）

- 执行方式：`scripts/run_segmented_backtest.py`（模式 B：年末清仓，现金承接；KPI 在合并后的完整日收益上统一计算）
- 参数要点：`--years 2020-2024 --hard_cap --qlib_dir ~/.qlib/qlib_data/us_data --preds artifacts/preds/preds_y5_2020_2024.parquet --features_path artifacts/features_day.parquet`
- 输出目录：`artifacts/segmented/{2020..2024}/` 与 `artifacts/segmented/combined/`

合并区间 KPI（2020-2024）
- 合并日数（trading days）：692
- Sharpe（年化）：0.5920
- CAGR：9.64%
- 最大回撤：24.21%
- 期末现金：974,907.13（初始 1,000,000）

分段摘要（每年单段 summary.cash_end 为“下一段初始现金”）
- 2020：cash 1,000,000 → 1,032,213.12，Sharpe 1.6469，CAGR 70.34%，MDD 0.79%
- 2021：cash 1,032,213.12 → 1,062,474.25，Sharpe 0.4378，CAGR 4.06%，MDD 5.85%
- 2022：cash 1,062,474.25 → 892,715.17，Sharpe -1.2355，CAGR -16.71%，MDD 20.91%
- 2023：cash 892,715.17 → 982,897.10，Sharpe 1.4313，CAGR 48.82%，MDD 7.09%
- 2024：cash 982,897.10 → 974,907.13，Sharpe -0.0698，CAGR -1.06%，MDD 5.87%

说明与后续改进
- 本次采用模式 B（年末清仓、现金承接）。边界清仓/重建的成本暂未显式计入；如需更保守，可在边界日按 `commission_bps+slippage_bps` 估算一次性成本并从当日收益扣减。
- 若要实现“携带头寸连续回测”（模式 A），需增强引擎支持 `--init_positions` 与 `--force_flatten_on_end`。在此之前，方案 B 可作为稳定替代。
