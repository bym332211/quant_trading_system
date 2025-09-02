# 下载并处理标准普尔500成分股的日线数据（daily）
python download_sp500.py --symbols-file ~/sp500_tickers.txt --start "2023-01-01" --end "2023-12-31" --interval "1d" --outdir "data/sp500"

# 使用 daily 数据生成 .bin 格式
python data/build_qlib_us.py \
  --in_dir  ~/.qlib/source_raw/sp500_day \
  --source_dir ~/.qlib/source/us_from_yf \
  --qlib_dir ~/.qlib/qlib_data/us_data \
  --freq day \
  --n_jobs 8

/home/ec2-user/.pyenv/versions/3.12.5/bin/python ./scripts/dump_bin.py dump_all \
  --data_path ~/.qlib/source/us_from_yf \
  --qlib_dir ~/.qlib/qlib_data/us_data \
  --freq day \
  --date_field_name date \
  --symbol_field_name symbol \
  --include_fields open,high,low,close,volume,factor,vwap \
  --max_workers 8

# 下载并处理标准普尔500成分股的5分钟数据（5min）
python download_sp500.py --symbols-file ~/sp500_tickers.txt --start "2025-07-01" --end "2025-07-31" --interval "5m" --outdir "data/sp500_5min"


# 使用 5min 数据生成 .bin 格式
python data/build_qlib_us.py \
  --in_dir ~/workspace/quant_trading_system/data/data/sp500_5min \
  --source_dir ~/.qlib/source/us_from_yf_5min \
  --qlib_dir ~/.qlib/qlib_data/us_data_5min \
  --freq 5min \
  --n_jobs 8 \
  --skip_dump_bin

  /home/ec2-user/.pyenv/versions/3.12.5/bin/python \
  ./scripts/dump_bin.py dump_all \
  --data_path ~/.qlib/source/us_from_yf_5min \
  --qlib_dir ~/.qlib/qlib_data/us_data_5min \
  --freq 1min \
  --date_field_name date \
  --symbol_field_name symbol \
  --include_fields open,high,low,close,volume,factor,vwap \
  --max_workers 8


  python data/build_qlib_us.py \
  --in_dir ~/workspace/quant_trading_system/data/data/sp500_1min \
  --source_dir ~/.qlib/source/us_from_yf_1min \
  --qlib_dir ~/.qlib/qlib_data/us_data_1min \
  --freq 1min \
  --n_jobs 8 \
  --skip_dump_bin

  /home/ec2-user/.pyenv/versions/3.12.5/bin/python \
  ./scripts/dump_bin.py dump_all \
  --data_path ~/.qlib/source/us_from_yf_1min \
  --qlib_dir ~/.qlib/qlib_data/us_data_1min \
  --freq 1min \
  --date_field_name date \
  --symbol_field_name symbol \
  --include_fields open,high,low,close,volume,factor,vwap \
  --max_workers 8