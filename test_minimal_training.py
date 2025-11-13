#!/usr/bin/env python3
"""
Versão mínima do script de treinamento para identificar o problema exato
"""

import numpy as np
import torch
import sys
from datetime import datetime

# Force CPU
torch.set_default_device("cpu")

print("=== Script de treinamento mínimo ===")

try:
    from src.ingest.binance import BinanceIngestor
    from src.features.builder import FeatureBuilder
    from src.config import config
    
    # Force CPU na config
    config.device = "cpu"
    print(f"Device forçado para: {config.device}")
    
    print("\n1. Carregando dados...")
    ingestor = BinanceIngestor()
    df_all = ingestor.load_from_parquet(symbol="BTCUSDT", interval="5m")
    
    print(f"Dados shape: {df_all.shape}")
    
    if len(df_all) < 1000:
        print("Poucos dados, baixando mais...")
        df_all = ingestor.ingest_symbol(symbol="BTCUSDT", interval="5m", days_back=30)
    
    # Use a smaller sample for testing
    df_train = df_all.tail(1000).copy()
    print(f"Using {len(df_train)} samples for testing")
    
    print("\n2. Building features...")
    builder = FeatureBuilder()
    df_train = builder.build_features(df_train)
    print(f"Features built: {df_train.shape}")
    
    if df_train.empty:
        print("DataFrame vazio após features!")
        exit(1)
    
    print("\n3. Preparando dados simples...")
    # Simplified data preparation
    numeric_cols = []
    for col in df_train.columns:
        if np.issubdtype(df_train[col].dtype, np.number):
            numeric_cols.append(col)
    
    features = df_train[numeric_cols].fillna(0).values.astype(np.float32)
    prices = df_train['close'].values.astype(np.float32)
    
    print(f"Features shape: {features.shape}")
    print(f"Prices shape: {prices.shape}")
    
    # Simple window creation (much smaller than original)
    window_size = 50  # Much smaller than 492
    samples = []
    price_samples = []
    
    for i in range(window_size, len(features)):
        window_data = features[i-window_size:i]
        agg = np.concatenate([
            window_data.mean(axis=0),
            window_data.std(axis=0),
            window_data[-1]
        ])
        samples.append(agg)
        price_samples.append(prices[i])
    
    samples = np.array(samples, dtype=np.float32)
    price_samples = np.array(price_samples, dtype=np.float32)
    
    print(f"Final samples shape: {samples.shape}")
    print(f"Final prices shape: {price_samples.shape}")
    
    print("\n4. Teste de conversão para tensor...")
    
    # Test tensor creation step by step
    print("Creating price tensor...")
    price_tensor = torch.tensor(price_samples, dtype=torch.float32, device="cpu")
    print(f"✅ Price tensor: {price_tensor.shape}")
    
    print("Creating features tensor...")
    features_tensor = torch.tensor(samples, dtype=torch.float32, device="cpu")
    print(f"✅ Features tensor: {features_tensor.shape}")
    
    print("\n5. Teste de ambiente simples...")
    
    # Minimal environment test
    class SimpleEnv:
        def __init__(self, prices, features, num_envs=2):
            self.num_envs = num_envs
            self.prices = torch.tensor(prices, dtype=torch.float32, device="cpu")
            self.features = torch.tensor(features, dtype=torch.float32, device="cpu")
            print(f"SimpleEnv created with {num_envs} envs")
    
    env = SimpleEnv(price_samples, samples, num_envs=2)
    print("✅ Environment created successfully!")
    
    print("\n🎉 Teste mínimo concluído com sucesso!")
    
except Exception as e:
    print(f"❌ ERRO: {e}")
    import traceback
    traceback.print_exc()