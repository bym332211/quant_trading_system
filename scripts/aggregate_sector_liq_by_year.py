import argparse
from pathlib import Path
import pandas as pd


def load_wide_or_build(seg_dir: Path) -> pd.DataFrame | None:
    wide_p = seg_dir / 'pnl_by_sector_liq_raw_wide.csv'
    if wide_p.exists():
        try:
            df = pd.read_csv(wide_p)
            if 'sector' in df.columns:
                df = df.set_index('sector')
            return df
        except Exception:
            pass
    long_p = seg_dir / 'pnl_by_sector_liq_raw_long.csv'
    if long_p.exists():
        try:
            d = pd.read_csv(long_p)
            if {'sector','liq_bucket','value'} <= set(d.columns):
                w = d.pivot_table(index='sector', columns='liq_bucket', values='value', aggfunc='sum').fillna(0.0)
                return w
        except Exception:
            pass
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run_dir', required=True, help='A single sweep run dir that contains seg_YYYY subdirs')
    ap.add_argument('--out', default=None, help='Output dir (default: <run_dir>/AGG_BY_YEAR)')
    args = ap.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve() if args.out else (run_dir / 'AGG_BY_YEAR')
    out_dir.mkdir(parents=True, exist_ok=True)

    segs = sorted([p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith('seg_')])
    years_done = []
    for seg in segs:
        try:
            y = int(seg.name.replace('seg_', ''))
        except Exception:
            continue
        w = load_wide_or_build(seg)
        if w is None or w.empty:
            continue
        # Save per-year wide CSV
        w.to_csv(out_dir / f'pnl_by_sector_liq_{y}.csv', float_format='%.2f')
        years_done.append(y)

    if years_done:
        with open(out_dir / 'years.txt', 'w') as f:
            f.write("\n".join(str(y) for y in sorted(years_done)))
        print('aggregated years:', ', '.join(str(y) for y in sorted(years_done)))
        print('output dir:', out_dir)
    else:
        print('no segment year matrices generated (missing raw files?)')


if __name__ == '__main__':
    main()
