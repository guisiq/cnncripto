import pandas as pd
from pathlib import Path
from src.features.builder import FeatureBuilder
from train_micronet_recurrent import prepare_micro_data

p = Path('data/timeframe=5m/symbol=BTCUSDT/candles.parquet')
if not p.exists():
    print('file not found:', p)
    raise SystemExit(1)

df = pd.read_parquet(p, engine='pyarrow')
from datetime import datetime
# filter same as train

df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)
df_filtered = df[(df['timestamp'] >= datetime(2023,1,1)) & (df['timestamp'] < datetime(2025,1,1))].copy()

builder = FeatureBuilder()
df_feat = builder.add_features(df_filtered)
print('feature columns count:', len([c for c in df_feat.columns if c not in ['timestamp','open','high','low','close','volume','quote_volume','symbol','date']]))

prices, micro = prepare_micro_data(df_feat, micro_window=60)
print('prices len:', len(prices))
print('micro shape:', micro.shape)
print('sample micro vector length:', micro.shape[1])
print('first micro vector (truncated):', micro[0][:20])
