import argparse
from pathlib import Path
import pandas as pd


def collect_yearly_long(run_dir: Path) -> pd.DataFrame:
    records = []
    for seg in sorted(run_dir.glob('seg_*')):
        if not seg.is_dir():
            continue
        try:
            y = int(seg.name.replace('seg_', ''))
        except Exception:
            continue
        f = seg / 'pnl_by_sector_liq_raw_long.csv'
        if not f.exists():
            continue
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        if not {'sector','liq_bucket','value'} <= set(df.columns):
            continue
        # ensure types
        df = df[['sector','liq_bucket','value']].copy()
        df['year'] = y
        records.append(df)
    if records:
        return pd.concat(records, ignore_index=True)
    return pd.DataFrame(columns=['sector','liq_bucket','value','year'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run_dir', required=True, help='A single parameter run dir containing seg_YYYY')
    ap.add_argument('--exclude_unknown', action='store_true', help='Exclude Unknown sector and liq_unknown bucket')
    ap.add_argument('--out', default=None, help='Output directory for rankings (default: <run_dir>/RANKS)')
    args = ap.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve() if args.out else (run_dir / 'RANKS')
    out_dir.mkdir(parents=True, exist_ok=True)

    long_df = collect_yearly_long(run_dir)
    if long_df.empty:
        print('NO_DATA')
        return
    if args.exclude_unknown:
        long_df = long_df[(long_df['sector'] != 'Unknown') & (long_df['liq_bucket'] != 'liq_unknown')]

    # Profitability: total across all years
    prof = (long_df.groupby(['sector','liq_bucket'], as_index=False)['value'].sum()
                     .rename(columns={'value':'total'}))
    prof_sorted = prof.sort_values('total', ascending=False)
    prof_sorted.to_csv(out_dir / 'sector_liq_profit_rank.csv', index=False, float_format='%.6f')

    # Stability: compute std across years and mean
    pvt = long_df.pivot_table(index=['sector','liq_bucket'], columns='year', values='value', aggfunc='sum').fillna(0.0)
    stats = pd.DataFrame({'mean': pvt.mean(axis=1), 'std': pvt.std(axis=1)})
    # stability rank: std asc, then mean desc
    stab_sorted = stats.sort_values(['std','mean'], ascending=[True, False]).reset_index()
    stab_sorted.to_csv(out_dir / 'sector_liq_stability_rank.csv', index=False, float_format='%.6f')

    # Print top rows
    print('Top 10 by profitability (total):')
    print(prof_sorted.head(10).to_string(index=False))
    print('\nTop 10 by stability (std asc, mean desc):')
    print(stab_sorted.head(10).to_string(index=False))
    print('\nSaved rankings to', out_dir)


if __name__ == '__main__':
    main()

