"""
Demo completo do workflow: fetch → train → embed → predict → backtest
Executável: python demo_complete_workflow.py
"""
import sys
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
from src.pipeline import TradingPipeline
from src.logger import get_logger
from src.config import config

logger = get_logger(__name__)

def print_section(title):
    """Print section header"""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")

def demo_data_collection():
    """1️⃣ Coleta de dados do Binance"""
    print_section("PASSO 1: COLETA DE DADOS DO BINANCE")
    
    pipeline = TradingPipeline()
    
    print("📥 Coletando últimos 5 dias de BTCUSDT (5m candles)...")
    long_data, short_data, full_df = pipeline.fetch_and_prepare_data(
        "BTCUSDT",
        days_back=5,
        lookback_days=5
    )
    
    print(f"✓ Total de candles: {full_df.shape[0]}")
    print(f"✓ Features calculadas: {full_df.shape[1]}")
    print(f"✓ Long window (últimos 5d): {long_data.shape[0]} candles")
    print(f"✓ Short window (últimas 5h): {short_data.shape[0]} candles")
    
    print("\n📊 Visualizando últimos 5 candles:")
    print(full_df[['timestamp', 'close', 'volume', 'returns', 'volatility_12']].tail())
    
    print("\n📈 Estatísticas:")
    print(f"  Close (últimas 5h): min={short_data['close'].min():.2f}, max={short_data['close'].max():.2f}")
    print(f"  Volume médio: {full_df['volume'].mean():.0f}")
    print(f"  Retorno médio: {full_df['returns'].mean():.6f}")
    
    return pipeline, long_data, short_data, full_df


def demo_feature_engineering():
    """2️⃣ Engenharia de Features"""
    print_section("PASSO 2: ENGENHARIA DE FEATURES")
    
    pipeline, long_data, short_data, full_df = demo_data_collection()
    
    print("🔧 Features Engenheiradas:")
    print("  • log_returns: retorno logarítmico")
    print("  • volatility_12/24/48: volatilidade em 3 janelas")
    print("  • volume_zscore: volume normalizado")
    print("  • quote_volume_zscore: volume em USD normalizado")
    print("  • hl_range: diferença High - Low")
    print("  • close_position: posição do close no range H-L")
    
    feature_cols = [c for c in full_df.columns if c not in 
                   ['timestamp', 'date', 'open', 'high', 'low', 'close', 'volume', 'quote_volume']]
    
    print(f"\n✓ Total de features: {len(feature_cols)}")
    print(f"  {feature_cols}")
    
    print("\n📊 Correlação entre features (últimos 5h):")
    corr = short_data[feature_cols].corr()
    print(corr.iloc[:5, :5])
    
    return pipeline, long_data, short_data, full_df


def demo_macronet_training():
    """3️⃣ Treinamento da MacroNet"""
    print_section("PASSO 3: TREINAMENTO DA MACRONET (Contexto de longo prazo)")
    
    pipeline, long_data, short_data, full_df = demo_feature_engineering()
    
    print("🧠 MacroNet: CNN com Atenção")
    print("  • Input: 1440 candles × 13 features (5 dias)")
    print("  • Processamento:")
    print("    - 3 camadas CNN com dilatação (2^i)")
    print("    - Attention pooling para agregar informação")
    print("    - Output: embedding de 128 dimensões")
    print("  • Treinamento: Autoencoder (loss de reconstrução)")
    print()
    
    # Extrair features
    exclude_cols = ['timestamp', 'date', 'open', 'high', 'low', 'close', 'volume', 'quote_volume']
    feature_cols = [c for c in long_data.columns if c not in exclude_cols]
    X_long = long_data[feature_cols].values
    
    print(f"📊 Shape dos dados de treino: {X_long.shape}")
    print(f"   (1440 candles, 13 features)")
    
    # Preparar para modelo
    X_batch = X_long[np.newaxis, :, :]  # (1, 1440, 13)
    print(f"✓ Batch shape: {X_batch.shape}")
    
    print("\n⏱️  Treinando por 3 epochs (demo)...")
    pipeline.macronet.train(X_batch, epochs=3)
    
    print("\n✓ MacroNet treinada com sucesso!")
    print(f"  • Encoder shape: (128,)")
    print(f"  • Decoder reconstrói: (1440, 13)")
    
    return pipeline, long_data, short_data, full_df


def demo_macro_embedding():
    """4️⃣ Geração de Embedding Diário"""
    print_section("PASSO 4: GERAÇÃO DE EMBEDDING MACRO (1x por dia)")
    
    pipeline, long_data, short_data, full_df = demo_macronet_training()
    
    print("📦 Gerando embedding para hoje (compressão de 5 dias):")
    print("  • Entrada: 1440 candles × 13 features (5 dias)")
    print("  • Processamento através do encoder")
    print("  • Saída: vetor de 128 dimensões")
    print()
    
    macro_embedding = pipeline.generate_macro_embedding("BTCUSDT", days_back=5)
    
    print(f"✓ Embedding gerado!")
    print(f"  Shape: {macro_embedding.shape}")
    print(f"  Tipo: {type(macro_embedding)}")
    print(f"  Min: {macro_embedding.min():.6f}")
    print(f"  Max: {macro_embedding.max():.6f}")
    print(f"  Mean: {macro_embedding.mean():.6f}")
    
    print("\n  Primeiros 10 valores do embedding:")
    print(f"  {macro_embedding[0, :10]}")
    
    print("\n💾 Embedding cacheado para uso intraday")
    
    return pipeline, long_data, short_data, full_df, macro_embedding


def demo_micronet_signal():
    """5️⃣ Geração de Sinal com MicroNet"""
    print_section("PASSO 5: GERAÇÃO DE SINAL INTRADAY (MicroNet)")
    
    pipeline, long_data, short_data, full_df, macro_embedding = demo_macro_embedding()
    
    print("🎯 MicroNet: Decision Head")
    print("  • Input 1: Últimas 5h (60 candles × 13 features)")
    print("  • Input 2: Macro embedding (128 dimensões)")
    print("  • Processamento:")
    print("    - MLP processa short-term features")
    print("    - Concatena com macro embedding")
    print("    - Prediz score de -1 (venda) a +1 (compra)")
    print()
    
    signal = pipeline.predict_signal("BTCUSDT")
    
    print(f"✓ Sinal gerado: {signal:.4f}")
    
    # Interpretar sinal
    if signal > 0.5:
        action = "🟢 COMPRA FORTE"
    elif signal > 0.1:
        action = "🟢 COMPRA"
    elif signal > -0.1:
        action = "⚪ NEUTRO"
    elif signal > -0.5:
        action = "🔴 VENDA"
    else:
        action = "🔴 VENDA FORTE"
    
    print(f"  Interpretação: {action}")
    
    print("\n📊 Interpretação do Score:")
    print("   +1.0  → Compra muito forte")
    print("   +0.5  → Compra moderada")
    print("    0.0  → Neutro")
    print("   -0.5  → Venda moderada")
    print("   -1.0  → Venda muito forte")
    
    return pipeline, signal


def demo_micronet_training():
    """6️⃣ Treinamento da MicroNet"""
    print_section("PASSO 6: TREINAMENTO DA MICRONET (histórico)")
    
    pipeline, long_data, short_data, full_df = demo_feature_engineering()
    
    print("🧠 MicroNet Training: Decision Head")
    print("  • Objetivo: Aprender a combinar short-term + macro context")
    print("  • Labels: Gerados a partir de returns futuros")
    print()
    
    # Treinar micronet com histórico
    print("⏱️  Treinando em 30 dias de histórico...")
    pipeline.train_micronet("BTCUSDT", days_back=30)
    
    print("✓ MicroNet treinada com sucesso!")
    
    return pipeline


def demo_backtest():
    """7️⃣ Backtesting e Avaliação"""
    print_section("PASSO 7: BACKTESTING & AVALIAÇÃO")
    
    pipeline = demo_micronet_training()
    
    print("📈 Simulando 30 dias de trades...")
    print("  • Comissão: 0.1% (Binance Maker)")
    print("  • Slippage: 0.05% (impacto de mercado)")
    print("  • Lógica: Se sinal > 0.0 → COMPRA, senão → VENDA/HOLD")
    print()
    
    results = pipeline.backtest_strategy("BTCUSDT", days_back=30)
    
    print("✓ Backtest concluído!\n")
    
    print("📊 RESULTADOS:")
    print(f"  Total Return:  {results['total_return']*100:>8.2f}%")
    print(f"  Sharpe Ratio:  {results['sharpe']:>8.2f}")
    print(f"  Sortino Ratio: {results['sortino']:>8.2f}")
    print(f"  Max Drawdown:  {results['max_drawdown']*100:>8.2f}%")
    print(f"  Win Rate:      {results['win_rate']*100:>8.2f}%")
    print(f"  Total Trades:  {results['num_trades']:>8.0f}")
    
    print("\n📈 Interpretação:")
    if results['sharpe'] > 1.0:
        print("  ✓ Sharpe > 1.0: Bom risco-retorno")
    else:
        print("  ⚠️  Sharpe < 1.0: Risco-retorno inadequado")
    
    if results['max_drawdown'] < -0.10:
        print("  ⚠️  Drawdown muito alto (>10%)")
    else:
        print("  ✓ Drawdown controlado")
    
    if results['win_rate'] > 0.5:
        print("  ✓ Win Rate > 50%: Mais vencedores que perdedores")
    else:
        print("  ⚠️  Win Rate < 50%: Mais perdedores que vencedores")
    
    return results


def demo_full_workflow():
    """🚀 Workflow Completo"""
    print_section("🚀 WORKFLOW COMPLETO: FETCH → TRAIN → EMBED → PREDICT → BACKTEST")
    
    print("Este script demonstra o ciclo completo:\n")
    
    print("1️⃣  COLETA DE DADOS")
    print("    └─ Binance API → Parquet cache\n")
    
    print("2️⃣  ENGENHARIA DE FEATURES")
    print("    └─ 13 features técnicas calculadas\n")
    
    print("3️⃣  TREINAMENTO MACRONET")
    print("    └─ Encoder aprende padrões de longo prazo\n")
    
    print("4️⃣  GERAÇÃO DE EMBEDDING")
    print("    └─ Comprime 5 dias em 128 dimensões\n")
    
    print("5️⃣  TREINAMENTO MICRONET")
    print("    └─ Decision head aprende a combinar contextos\n")
    
    print("6️⃣  PREVISÃO DE SINAL")
    print("    └─ Score de -1 (venda) a +1 (compra)\n")
    
    print("7️⃣  BACKTESTING")
    print("    └─ Simula 30 dias de trades\n")
    
    print("Iniciando demo...\n")
    
    try:
        results = demo_backtest()
        
        print_section("✅ DEMO COMPLETO FINALIZADO COM SUCESSO!")
        
        print("📊 Resumo final:")
        print(f"  • Retorno total: {results['total_return']*100:.2f}%")
        print(f"  • Sharpe ratio: {results['sharpe']:.2f}")
        print(f"  • Drawdown máximo: {results['max_drawdown']*100:.2f}%")
        print(f"  • Taxa de acerto: {results['win_rate']*100:.2f}%")
        print(f"  • Total de trades: {results['num_trades']:.0f}")
    except Exception as e:
        print(f"\n❌ Erro durante execução: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    demo_full_workflow()
