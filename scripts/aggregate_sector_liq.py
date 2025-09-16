import argparse
from pathlib import Path
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--roots', nargs='+', required=True, help='One or more run directories or parent dirs to scan')
    ap.add_argument('--out', required=True, help='Output directory for aggregated CSVs')
    args = ap.parse_args()

    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    files = []
    for r in args.roots:
        p = Path(r).expanduser().resolve()
        if p.is_file():
            continue
        if (p / 'pnl_by_sector_liq_raw_long.csv').exists():
            files.append(p / 'pnl_by_sector_liq_raw_long.csv')
        else:
            # scan children
            for sub in p.rglob('pnl_by_sector_liq_raw_long.csv'):
                files.append(sub)

    if not files:
        print('no raw_long files found under roots')
        return

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            if {'sector','liq_bucket','value'} <= set(df.columns):
                dfs.append(df[['sector','liq_bucket','value']])
        except Exception:
            continue

    if not dfs:
        print('no valid data loaded')
        return

    all_long = pd.concat(dfs, ignore_index=True)
    all_long = all_long.groupby(['sector','liq_bucket'], as_index=False)['value'].sum()
    all_long.to_csv(out_dir / 'pnl_by_sector_liq_agg_long.csv', index=False)

    wide = all_long.pivot_table(index='sector', columns='liq_bucket', values='value', aggfunc='sum').fillna(0.0)
    # reorder columns: numeric buckets ascending, then others, put liq_unknown last
    cols = list(wide.columns)
    num_cols = [c for c in cols if isinstance(c, str) and c.startswith('liq_') and c.split('_')[-1].isdigit()]
    num_cols_sorted = sorted(num_cols, key=lambda x: int(x.split('_')[-1]))
    other_cols = [c for c in cols if c not in num_cols]
    col_order = num_cols_sorted + [c for c in other_cols if c != 'liq_unknown'] + (['liq_unknown'] if 'liq_unknown' in other_cols else [])
    wide = wide.reindex(columns=[c for c in col_order if c in wide.columns])
    wide.to_csv(out_dir / 'pnl_by_sector_liq_agg_wide.csv', float_format='%.2f')
    print('aggregated ->', out_dir)


if __name__ == '__main__':
    main()
