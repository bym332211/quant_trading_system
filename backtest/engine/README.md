v3.4.1 使用说明 & 推荐参数（"当前最优解"）

## 版本要点

- **短腿择时**：基于锚标的（默认 SPY）63 日动量，用 T-1 信息；当 动量 ≤ 阈值 时，允许做空（文档与实现一致）。
- **硬上限 Water-filling**：在单票上限约束下精确把多/空腿分配到目标暴露（不会因裁剪再失衡）。
- **权重流水线**：中性化 → 平滑 → 目标波动缩放 → 硬上限 → %ADV 限速。
- **稳健性**：特征缺列自动降级；短腿择时在历史不足/缺值时"默认放行"避免误关空腿。

## 1) 依赖 & 输入

**依赖**：python>=3.9, pandas, numpy, backtrader, pyqlib（US region）

**--preds（Parquet）**：列需含
- instrument（或 symbol/ticker/... 会自动识别并改名）
- datetime（或 date/timestamp/...）
- score（必需）
- rank（可选；存在则优先按 rank 做 membership）

**--features_path（Parquet）**：用于中性化/σ/ADV 的特征，推荐包含
- mkt_beta_60, ln_dollar_vol_20, ind_*, liq_bucket_*
- ret_1（估 σ），adv_20 与 $vwap（估 ADV$）

**缺列会自动降级**：无 ret_1 则不做目标波动缩放；无 adv_20/$vwap 则不做 %ADV 限速

**Universe**：窗口内 preds 出现过的股票 + --anchor_symbol（默认 SPY）

**调仓日**：对 锚标的交易日做 周频最早日 采样 → 与预测日 as-of 对齐 → --exec_lag 推进到执行日

**成本**：当前仅使用 commission_bps & slippage_bps（Backtrader 侧）；借券费不在本版内置

## 2) 关键参数（常用）

### 选股与权重
- `--top_k / --short_k`：长/短持仓数
- `--weight_scheme {equal,icdf}`：等权或分位正态逆(CDF^-1)
- `--membership_buffer`：入池缓冲（减少换手）
- `--long_exposure / --short_exposure`：多/空腿目标暴露（例如 1/-1 或 1/-0.6）
- `--max_pos_per_name`：单票上限（与 --hard_cap 联用时生效最佳）

### 风险与限制
- `--neutralize "beta,sector,liq,size"`：线性（岭）中性化
- `--smooth_eta`：权重平滑系数 η（越大越"钝化"）
- `--target_vol`：目标年化波动（先缩放，再做硬上限）
- `--leverage_cap`：目标波动缩放时的杠杆上限
- `--adv_limit_pct`：单票一次换手的 %ADV 限速（例：0.01=1% ADV）
- `--hard_cap`：启用硬上限 water-filling 以精确分配并满足单票上限

### 短腿择时（用 T-1 信息）
- `--short_timing_mom63`：启用择时
- `--short_timing_threshold`（默认 0.0）：当 SPY 的 63d 动量 ≤ 阈值 时允许做空
- `--short_timing_lookback`（默认 63）

## 3) 流程（每日执行日）

1. **Membership（含缓冲区）**：从 score/rank 决定 long/short 池；择时关空腿时，短腿池清空
2. **Raw 权重**：equal / icdf 生成
3. **中性化**：回归残差法（beta/sector/liq/size 维度，dummy 自动 drop 一列）
4. **平滑**：w = η·w_prev + (1-η)·w_tgt
5. **目标波动**：EWM σ（T-1 截止）估计组合年化波动，按 target_vol 缩放（受 leverage_cap 限定）
6. **硬上限（建议开启）**：对多/空腿分开做 water-filling，使
   - 单票 |w_i| ≤ max_pos_per_name
   - 多腿和= long_exposure、空腿和= -short_exposure（择时不许做空则空腿=0）
7. **%ADV 限速**：按 |Δw| ≤ adv_limit_pct · ADV$ / PortValue 裁剪调仓
8. **下单**：order_target_percent
9. **诊断**：拆腿收益、turnover（pre/post-ADV）、ADV 命中率、累计佣金等

## 4) 输出文件

- `equity_curve.csv`：datetime, value, cash, ret
- `portfolio_returns.csv`：逐日组合收益（Backtrader Analyzer 或回填）
- `per_day_ext.csv`：逐日诊断
  - ret, ret_long, ret_short, turnover_pre, turnover_post, adv_clip_ratio, adv_clip_names, gross_long, gross_short, commission_cum
- `orders.csv / positions.csv`：成交与持仓快照
- `summary.json`：整体指标（CAGR/Sharpe/MDD、参数与开关、rebalance 天数……）
- `kpis.json`：补充 KPI（turnover_mean/p90、adv_clip 命中率、gross_*_avg、长短腿 Sharpe）

## 5) 快速上手（命令示例）

### A. 多空（低换手、较高 Sharpe 的平衡解）

结合你近期回测结果，表现更稳的部分做空方案（TV≈8%、低换手）：

```bash
python run_backtest.py \
  --qlib_dir ~/qlib/us_data \
  --preds ~/data/preds.parquet \
  --features_path ~/data/features.parquet \
  --start 2020-01-01 --end 2024-12-31 \
  --trade_at close --exec_lag 0 \
  --top_k 40 --short_k 8 \
  --long_exposure 1.0 --short_exposure -0.6 \
  --weight_scheme equal \
  --membership_buffer 0.40 --smooth_eta 0.85 \
  --neutralize "beta,sector,liq,size" \
  --target_vol 0.08 --leverage_cap 2.0 \
  --max_pos_per_name 0.05 --hard_cap \
  --adv_limit_pct 0.01 \
  --out_dir ./runs/ls_tv08_eq_cap

```

**为何推荐**  
你给出的 [ls_tv08_equal_adv1_q10_liq3_sh8_diag] 组合：Sharpe≈0.77、CAGR≈28.5%、MDD≈8.3%、turnover≈3.9%，在成本/限速下权衡较好。硬上限进一步保证风控一致性。

### B. Long-only（T+1、简洁稳健基线）

```bash
python run_backtest.py
–qlib_dir ~/qlib/us_data
–preds ~/data/preds.parquet
–features_path ~/data/features.parquet
–start 2020-01-01 --end 2024-12-31
–trade_at open --exec_lag 1
–top_k 50 --short_k 0
–long_exposure 1.0 --short_exposure 0.0
–weight_scheme equal
–membership_buffer 0.25 --smooth_eta 0.70
–neutralize “beta,sector,liq,size”
–target_vol 0.12 --leverage_cap 2.0
–max_pos_per_name 0.03 --hard_cap
–adv_limit_pct 0.005
–out_dir ./runs/long_Tplus1_eq_cap
```

**为何推荐**  
交易路径简单、实现友好；exec_lag=1（T+1 开盘）规避信号-成交同日偏差；单票上限 3% + 硬上限保证集中度。

### C. 短腿择时（可选）

若你希望在"市场偏弱"时才做空（按本版定义：SPY 63D 动量 ≤ 阈值 即可做空），示例：

```bash
…（承上多空命令）
–short_timing_mom63
–short_timing_threshold 0.0
–short_timing_lookback 63
```

**注**：择时只门控空腿（当日允许/不允许做空），不影响多腿；信号使用 T-1 信息，避免未来函数。

## 6) 实用提示

- **硬上限与再平衡**：Water-filling 在多/空腿分别分配，确保腿级总暴露与单票上限同时满足，避免"裁剪后再归一"的二次偏差。
- **%ADV 限速**：是下单前的最终一道闸；诊断上有 turnover_pre/post 与 adv_clip_ratio。
- **σ/ADV 缺列降级**：features 缺列不报错，功能自动降级（但效果会变）；尽量提供 ret_1 / adv_20 / $vwap。
- **rank 优先**：如果 preds 里有 rank，membership 会优先用 rank 做进入/退出缓冲。
- **trade_at 与 exec_lag**：
  - trade_at=open 常与 exec_lag=1 组合（T+1 开盘执行）。
  - trade_at=close 可与 exec_lag=0（当日收盘执行）。

## 7) 版本记录（本次与 v3.3.1 相比）

- **短腿择时**：明确文档为「动量 ≤ 阈值时允许做空」（与实现一致）；T-1 生效。
- **硬上限 Water-filling**：新增 --hard_cap 控制；当开启且 max_pos_per_name>0 时启用。
- **稳健性增强**
  - features 缺列时自动降级（不抛错）；
  - 短腿择时在历史不足/缺值时"全放行"（不误关空腿）。
- **行为修正**：--hard_cap 不再默认 True；完全由 CLI 控制。

## 8) "当前最优解"小结（基于你给的近期结果）

- **多空（部分做空）**：TV≈8%、top_k=40/short_k=8、short_exposure=-0.6、buffer=0.4、eta=0.85、close、ADV 1%、建议开启硬上限、max_pos=5%。
  - 兼顾 Sharpe、回撤与换手，诊断表现均衡。
  
- **Long-only（T+1 开盘）**：top_k=50、TV≈12%、buffer≈0.25、eta≈0.7、硬上限 + 单票 3%、ADV 0.5%。
  - 路径简单、交易稳健，适合做"上线基线"。

> 以上为参数模版；不同数据与成本下仍建议网格微调（buffer/eta/TV/short_k/max_pos/%ADV 是最敏感的几个旋钮）。
