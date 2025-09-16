import argparse
from pathlib import Path
import pandas as pd


def collect_yearly_matrices(run_dir: Path) -> dict[int, pd.DataFrame]:
    """Return {year: wide_df} built from seg_YYYY/pnl_by_sector_liq_raw_long.csv"""
    out: dict[int, pd.DataFrame] = {}
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
            d = pd.read_csv(f)
        except Exception:
            continue
        if not {'sector','liq_bucket','value'} <= set(d.columns):
            continue
        w = d.pivot_table(index='sector', columns='liq_bucket', values='value', aggfunc='sum').fillna(0.0)
        out[y] = w
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run_dir', required=True, help='A single parameter run dir containing seg_YYYY')
    args = ap.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    year2wide = collect_yearly_matrices(run_dir)
    if not year2wide:
        print('NO_DATA')
        return
    years = sorted(year2wide.keys())

    # build long table of (sector, liq_bucket, year, value)
    records = []
    for y in years:
        w = year2wide[y]
        w = w.reset_index()  # sector column
        for _, row in w.iterrows():
            sector = row['sector']
            for col in w.columns:
                if col == 'sector':
                    continue
                val = float(row[col]) if pd.notna(row[col]) else 0.0
                records.append({'sector': sector, 'liq_bucket': col, 'year': y, 'value': val})
    df = pd.DataFrame(records)
    pvt = df.pivot_table(index=['sector','liq_bucket'], columns='year', values='value', aggfunc='sum').fillna(0.0)

    mean = pvt.mean(axis=1)
    std = pvt.std(axis=1)
    res = pd.DataFrame({'mean': mean, 'std': std})

    # Top5 stable profitable: mean>0, sort by std asc then mean desc
    top = res[res['mean'] > 0].sort_values(['std','mean'], ascending=[True, False]).head(5)
    # Worst5 unstable: sort by std desc; show mean for context
    worst = res.sort_values('std', ascending=False).head(5)

    print('TOP5_STABLE_PROFITABLE (sector | liq_bucket | std | mean)')
    for (sec, lb), row in top.iterrows():
        print(f"{sec} | {lb} | {row['std']:.4f} | {row['mean']:.4f}")

    print('\nWORST5_BY_STD (sector | liq_bucket | std | mean)')
    for (sec, lb), row in worst.iterrows():
        print(f"{sec} | {lb} | {row['std']:.4f} | {row['mean']:.4f}")


if __name__ == '__main__':
    main()

