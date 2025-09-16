import argparse
from pathlib import Path
import pandas as pd

def collect_years(run_dir: Path):
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
        yield y, pd.read_csv(f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run_dir', required=True)
    ap.add_argument('--exclude_unknown', action='store_true')
    args = ap.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    records = []
    for y, df in collect_years(run_dir):
        if not {'sector','liq_bucket','value'} <= set(df.columns):
            continue
        grp = df.groupby(['sector','liq_bucket'], as_index=False)['value'].sum()
        for _, r in grp.iterrows():
            sec = str(r['sector'])
            lb = str(r['liq_bucket'])
            val = float(r['value'])
            records.append({'sector': sec, 'liq_bucket': lb, 'year': y, 'value': val})
    if not records:
        print('NO_DATA')
        return
    tab = pd.DataFrame(records)
    if args.exclude_unknown:
        tab = tab[(tab['sector'] != 'Unknown') & (tab['liq_bucket'] != 'liq_unknown')]
    pvt = tab.pivot_table(index=['sector','liq_bucket'], columns='year', values='value', aggfunc='sum').fillna(0.0)
    stats = pd.DataFrame({'mean': pvt.mean(axis=1), 'std': pvt.std(axis=1)})
    # Top5 stable profitable
    top = stats[(stats['mean'] > 0)].sort_values(['std','mean'], ascending=[True, False]).head(5)
    worst = stats.sort_values('std', ascending=False).head(5)
    print('TOP5_STABLE_PROFITABLE (excl_unknown=%s)' % args.exclude_unknown)
    for (sec, lb), r in top.iterrows():
        print(f"{sec} | {lb} | std={r['std']:.4f} | mean={r['mean']:.4f}")
    print('\nWORST5_BY_STD (excl_unknown=%s)' % args.exclude_unknown)
    for (sec, lb), r in worst.iterrows():
        print(f"{sec} | {lb} | std={r['std']:.4f} | mean={r['mean']:.4f}")

if __name__ == '__main__':
    main()

