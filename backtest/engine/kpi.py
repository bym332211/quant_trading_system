# -*- coding: utf-8 -*-
turn_mean = float(diagdf["turnover_post"].mean()) if "turnover_post" in diagdf else float('nan')
turn_p90 = float(diagdf["turnover_post"].quantile(0.9)) if "turnover_post" in diagdf else float('nan')
adv_hit_days = float((diagdf["adv_clip_names"] > 0).mean()) if "adv_clip_names" in diagdf else 0.0
adv_clip_avg = float(diagdf.get("adv_clip_ratio", pd.Series(dtype=float)).replace([np.inf, -np.inf], np.nan).fillna(0.0).mean())
gross_long_avg = float(diagdf["gross_long"].mean()) if "gross_long" in diagdf else float('nan')
gross_short_avg = float(diagdf["gross_short"].mean()) if "gross_short" in diagdf else float('nan')


# === SPY Information Ratio ===
spy_ret = _spy_returns_from_price(price_map, anchor_symbol or getattr(args, 'anchor_symbol', None), eq["datetime"])
merged = pd.DataFrame({
"datetime": pd.to_datetime(retdf["datetime"]).values,
"ret": retdf["ret"].values,
})
merged = merged.sort_values("datetime")
merged["spy_ret"] = spy_ret.reindex(pd.to_datetime(merged["datetime"]).normalize()).values
merged["active"] = merged["ret"] - merged["spy_ret"].fillna(0.0)
ir_spy = _annualize_sharpe(merged["active"].to_numpy()) # mean/TE * sqrt(252)


# Build summary.json (parameters + headline metrics)
neutral_list = [s for s in [s.strip() for s in getattr(args, 'neutralize', '').split(',')] if s]
summary: Dict[str, Any] = {
"start": args.start, "end": args.end,
"cash_init": float(eq["value"].iloc[0]) if len(eq) else float(getattr(args, 'cash', np.nan)),
"cash_end": float(eq["value"].iloc[-1]) if len(eq) else float(getattr(args, 'cash', np.nan)),
"top_k": int(getattr(args, 'top_k', np.nan)) if getattr(args, 'top_k', None) is not None else None,
"short_k": int(getattr(args, 'short_k', np.nan)) if getattr(args, 'short_k', None) is not None else None,
"long_exposure": float(args.long_exposure),
"short_exposure": float(args.short_exposure),
"commission_bps": float(args.commission_bps),
"slippage_bps": float(args.slippage_bps),
"trade_at": args.trade_at,
"neutralize": neutral_list,
"membership_buffer": getattr(args, 'membership_buffer', None),
"smooth_eta": float(args.smooth_eta),
"days": int(len(eq)),
"CAGR": cagr,
"Sharpe": sharpe,
"MDD_pct": mdd,
"rebalance_days": len(set(list(strat.p.preds_by_exec.keys()))) if hasattr(strat.p, 'preds_by_exec') else None,
"universe_size": len(price_map) if price_map else None,
"avg_candidates_per_reb": float(np.mean([len(g) for g in getattr(strat.p, 'preds_by_exec', {}).values()])) if getattr(strat.p, 'preds_by_exec', None) else None,
"exec_lag": int(args.exec_lag),
"target_vol": float(args.target_vol),
"weight_scheme": args.weight_scheme,
"adv_limit_pct": float(args.adv_limit_pct),
"hard_cap": bool(args.hard_cap),
"short_timing_mom63": bool(args.short_timing_mom63),
"short_timing_threshold": float(args.short_timing_threshold),
"short_timing_lookback": int(args.short_timing_lookback),
"strategy_key": strategy_key,
"SPY_IR": ir_spy,
}
with open(out_dir / "summary.json", "w") as f:
json.dump(summary, f, indent=2)


# Additional KPIs
kpis = {
"turnover_mean": turn_mean,
"turnover_p90": turn_p90,
"adv_clip_days_frac": adv_hit_days,
"adv_clip_ratio_avg": adv_clip_avg,
"gross_long_avg": gross_long_avg,
"gross_short_avg": gross_short_avg,
"commission_total": float(getattr(strat, "_commission_cum", 0.0)),
"sharpe_long": sharpe_long,
"sharpe_short": sharpe_short,
"spy_ir": ir_spy,
}
with open(out_dir / "kpis.json", "w") as f:
json.dump(kpis, f, indent=2)


print(f"[saved KPI & diagnostics] -> {out_dir}")