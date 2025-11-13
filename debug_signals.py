"""Debug: Verificar sinais gerados pela rede"""
import numpy as np
from src.pipeline import TradingPipeline

# Criar pipeline
pipeline = TradingPipeline()

# Fetch dados
print("Baixando dados...")
long_data, short_data, full_df = pipeline.fetch_and_prepare_data("BTCUSDT", days_back=5)

print(f"Dados: {len(full_df)} candles")

# Gerar embedding macro
print("\nGerando macro embedding...")
macro_emb = pipeline.generate_macro_embedding("BTCUSDT", days_back=5)
print(f"Macro embedding shape: {macro_emb.shape}")
print(f"Macro stats: mean={macro_emb.mean():.3f}, std={macro_emb.std():.3f}")

# Testar predict signal
print("\nTestando sinais...")
signals = []
for i in range(60, min(len(full_df), 100)):
    signal = pipeline.predict_signal(full_df, "BTCUSDT", days_back=5, current_idx=i)
    signals.append(signal)
    if i < 70:
        print(f"  Candle {i}: signal={signal:.4f}")

signals = np.array(signals)
print(f"\nEstatísticas dos sinais (n={len(signals)}):")
print(f"  mean:  {signals.mean():.4f}")
print(f"  std:   {signals.std():.4f}")
print(f"  min:   {signals.min():.4f}")
print(f"  max:   {signals.max():.4f}")
print(f"  |s|>0.1: {np.sum(np.abs(signals) > 0.1)}")
print(f"  |s|>0.2: {np.sum(np.abs(signals) > 0.2)}")
print(f"  |s|>0.5: {np.sum(np.abs(signals) > 0.5)}")
