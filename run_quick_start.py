"""
Quick Start Script - Começa aqui para entender o sistema
Executável: python run_quick_start.py
"""
import sys
import os
sys.path.insert(0, '.')
os.environ['PYTHONIOENCODING'] = 'utf-8'

from src.pipeline import TradingPipeline
from src.config import config
import numpy as np

def print_title(text):
    print(f"\n{'='*70}")
    print(f"  {text}")
    print(f"{'='*70}\n")

def step1_fetch_data():
    """Passo 1: Coletar dados"""
    print_title("PASSO 1: COLETA DE DADOS DO BINANCE")
    
    print("Inicializando pipeline...")
    pipeline = TradingPipeline()
    
    print("📥 Coletando últimos 2 dias de BTCUSDT (teste rápido)...")
    long_data, short_data, full_df = pipeline.fetch_and_prepare_data("BTCUSDT", days_back=2)
    
    print(f"\n✅ Dados coletados com sucesso!")
    print(f"   • Total de candles: {len(full_df)}")
    print(f"   • Período: {full_df['timestamp'].min()} a {full_df['timestamp'].max()}")
    print(f"   • Features: {full_df.shape[1]}")
    print(f"   • Preço atual: ${full_df['close'].iloc[-1]:.2f}")
    
    return pipeline, long_data, short_data, full_df


def step2_train_macronet(pipeline, long_data):
    """Passo 2: Treinar MacroNet"""
    print_title("PASSO 2: TREINO DA MACRONET")
    
    print("Preparando dados para MacroNet...")
    X_features = pipeline.extract_feature_arrays(long_data)
    X = X_features[np.newaxis, :, :]
    
    print(f"Shape dos dados: {X.shape}")
    print("   • Batch size: 1")
    print(f"   • Candles: {X.shape[1]} (últimos 2 dias)")
    print(f"   • Features: {X.shape[2]}")
    
    print("\n⏱️  Treinando MacroNet por 2 epochs (rápido)...")
    pipeline.macronet.train(X, epochs=2)
    
    print("\n✅ MacroNet treinada!")
    print(f"   • Embedding shape: (1, {config.macronet.embedding_dim})")
    
    return pipeline


def step3_generate_embedding(pipeline):
    """Passo 3: Gerar embedding"""
    print_title("PASSO 3: GERAÇÃO DE EMBEDDING DIÁRIO")
    
    print("Gerando embedding (comprimindo 2 dias em 128 dimensões)...")
    embedding = pipeline.generate_macro_embedding("BTCUSDT", days_back=2)
    
    print(f"\n✅ Embedding gerado!")
    print(f"   • Shape: {embedding.shape}")
    print(f"   • Min: {embedding.min():.6f}")
    print(f"   • Max: {embedding.max():.6f}")
    print(f"   • Mean: {embedding.mean():.6f}")
    
    return pipeline, embedding


def step4_predict_signal(pipeline):
    """Passo 4: Gerar sinal"""
    print_title("PASSO 4: GERAÇÃO DE SINAL INTRADAY")
    
    print("Gerando sinal de compra/venda...")
    signal = pipeline.predict_signal("BTCUSDT")
    
    print(f"\n✅ Sinal gerado!")
    print(f"   • Valor: {signal:.4f}")
    
    # Interpretar
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
    
    print(f"   • Ação: {action}")
    
    return pipeline, signal


def step5_backtest(pipeline):
    """Passo 5: Backtesting"""
    print_title("PASSO 5: BACKTESTING")
    
    print("Simulando últimos 10 dias de trades...")
    results = pipeline.backtest_strategy("BTCUSDT", days_back=10)
    
    print(f"\n✅ Backtest concluído!")
    print(f"\n📊 RESULTADOS:")
    print(f"   • Total Return:  {results['total_return']*100:>8.2f}%")
    print(f"   • Sharpe Ratio:  {results['sharpe']:>8.2f}")
    print(f"   • Max Drawdown:  {results['max_drawdown']*100:>8.2f}%")
    print(f"   • Win Rate:      {results['win_rate']*100:>8.2f}%")
    print(f"   • Total Trades:  {results['num_trades']:>8.0f}")
    
    return results


def print_summary():
    """Resumo final"""
    print_title("✅ QUICK START COMPLETO!")
    
    print("""
🎯 O que você aprendeu:
   1. Coleta dados do Binance (API)
   2. Calcula 13 features técnicas
   3. Treina MacroNet (encoder)
   4. Gera embedding diário (128-dim)
   5. Treina MicroNet (decision head)
   6. Prediz sinal (-1 a +1)
   7. Executa backtest

📊 Próximos passos:
   1. Editar config.py para ajustar hiperparâmetros
   2. Executar python quick_tests.py para testes completos
   3. Executar python interactive_analysis.py para análise
   4. Integrar com API REST para produção

📚 Documentação:
   • COMO_EXECUTAR.md - Guia completo
   • SETUP.md - Instalação
   • ROADMAP.md - Plano de desenvolvimento
   • README.md - Visão geral do projeto
    """)


def main():
    """Executar quick start completo"""
    print("\n" + "="*70)
    print("=" + " "*68 + "=")
    print("=" + "  QUICK START - SISTEMA DE TRADING COM NEURAL NETWORKS".center(68) + "=")
    print("=" + " "*68 + "=")
    print("="*70)
    
    print(f"\n📋 Configuração:")
    print(f"   • Device: {config.device}")
    print(f"   • Python: 3.12")
    print(f"   • PyTorch: 2.2.0")
    print(f"   • Polars: 0.20.3")
    
    try:
        # Executar passos
        pipeline, long_data, short_data, full_df = step1_fetch_data()
        pipeline = step2_train_macronet(pipeline, long_data)
        pipeline, embedding = step3_generate_embedding(pipeline)
        pipeline, signal = step4_predict_signal(pipeline)
        results = step5_backtest(pipeline)
        
        print_summary()
        
    except KeyboardInterrupt:
        print("\n\n👋 Interrompido pelo usuário")
    except Exception as e:
        print(f"\n❌ Erro: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
