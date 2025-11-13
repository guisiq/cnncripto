"""
Quick Start Script - Versao ASCII compativel com Windows CP1252
Executável: python run_quick_start_ascii.py
"""
import sys
import os
sys.path.insert(0, '.')

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
    
    print(f"Buscando ultimos 5 dias de BTCUSDT (5m candles)...")
    try:
        long_data, short_data, full_df = pipeline.fetch_and_prepare_data(symbol="BTCUSDT")
        
        print(f"\n>>> Sucesso!")
        print(f"    Total de candles: {full_df.shape[0]}")
        print(f"    Features: {full_df.shape[1]}")
        print(f"    Preco atual: ${full_df['close'].iloc[-1]:.2f}")
        return full_df, long_data, short_data
        
    except Exception as e:
        print(f"\n>>> Erro: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def step2_train_macronet(full_df):
    """Passo 2: Treinar MacroNet"""
    print_title("PASSO 2: TREINAMENTO DA REDE MACRONET")
    
    if full_df is None:
        print(">>> Pulando passo 2 (dados nao disponiveis)")
        return
    
    pipeline = TradingPipeline()
    
    print(f"Shape dos dados: {full_df.shape}")
    print(f"Treinando MacroNet por 2 epochs (rapido)...")
    
    try:
        pipeline.train_macronet("BTCUSDT", days_back=5)
        print(f"\n>>> MacroNet treinada com sucesso!")
        
    except Exception as e:
        print(f"\n>>> Erro: {e}")

def step3_generate_embedding(long_data):
    """Passo 3: Gerar embeddings"""
    print_title("PASSO 3: GERACAO DE EMBEDDINGS")
    
    if long_data is None:
        print(">>> Pulando passo 3 (dados nao disponiveis)")
        return None
    
    pipeline = TradingPipeline()
    
    try:
        print(f"Gerando macro embedding da ultima janela...")
        macro_emb = pipeline.generate_macro_embedding("BTCUSDT", days_back=5)
        
        print(f"\n>>> Embedding gerado com sucesso!")
        print(f"    Shape: {macro_emb.shape}")
        print(f"    Dimensoes: {macro_emb.shape[-1]}")
        return macro_emb
        
    except Exception as e:
        print(f"\n>>> Erro: {e}")
        import traceback
        traceback.print_exc()
        return None

def step4_predict_signal(short_data, macro_emb):
    """Passo 4: Gerar sinal de predicao"""
    print_title("PASSO 4: PREDICAO DE SINAL")
    
    if short_data is None or macro_emb is None:
        print(">>> Pulando passo 4 (dados nao disponiveis)")
        return
    
    pipeline = TradingPipeline()
    
    try:
        print(f"Processando ultimos {short_data.shape[0]} candles...")
        signal = pipeline.predict_signal("BTCUSDT")
        
        print(f"\n>>> Sinal gerado com sucesso!")
        print(f"    Valor: {signal:.4f}")
        print(f"    Interpretacao: ", end="")
        
        if signal > 0.5:
            print("FORTE BUY (sinal positivo muito alto)")
        elif signal > 0.0:
            print("COMPRA (sinal positivo)")
        elif signal > -0.5:
            print("VENDA (sinal negativo)")
        else:
            print("FORTE SELL (sinal negativo muito alto)")
            
    except Exception as e:
        print(f"\n>>> Erro: {e}")

def step5_backtest():
    """Passo 5: Backtesting"""
    print_title("PASSO 5: BACKTESTING")
    
    pipeline = TradingPipeline()
    
    try:
        print(f"Executando backtest nos ultimos 5 dias...")
        results = pipeline.backtest_strategy("BTCUSDT", days_back=5)
        
        print(f"\n>>> Backtest concluido!")
        print(f"    Total return: {results.get('total_return', 0):.2%}")
        print(f"    Sharpe ratio: {results.get('sharpe_ratio', 0):.2f}")
        print(f"    Win rate: {results.get('win_rate', 0):.2%}")
        
    except Exception as e:
        print(f"\n>>> Erro: {e}")

def main():
    """Executar quick start completo"""
    print("\n" + "="*70)
    print("=" + " "*68 + "=")
    print("=" + "  QUICK START - SISTEMA DE TRADING COM NEURAL NETWORKS".center(68) + "=")
    print("=" + " "*68 + "=")
    print("="*70)
    
    print(f"\nConfiguracao:")
    print(f"  * Device: {config.device}")
    print(f"  * Python: 3.12")
    print(f"  * PyTorch: 2.2.0")
    
    # Executar os 5 passos
    full_df, long_data, short_data = step1_fetch_data()
    step2_train_macronet(full_df)
    macro_emb = step3_generate_embedding(long_data)
    step4_predict_signal(short_data, macro_emb)
    step5_backtest()
    
    print("\n" + "="*70)
    print("  QUICK START CONCLUIDO!")
    print("="*70)
    print("\nProximos passos:")
    print("  1. Ler: COMO_EXECUTAR.md")
    print("  2. Executar: python quick_tests.py")
    print("  3. Explorar: python interactive_analysis.py")
    print("\nDocumentacao:")
    print("  * COMO_EXECUTAR.md - Guia completo")
    print("  * SETUP.md - Instalacao")
    print("  * ROADMAP.md - Plano de desenvolvimento")
    print("  * README.md - Visao geral do projeto")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrompido pelo usuario.")
    except Exception as e:
        print(f"\n\nErro geral: {e}")
        import traceback
        traceback.print_exc()
