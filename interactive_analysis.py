"""
Script de Análise Interativa - Mostra dados, features e resultados em tabelas
Executável: python interactive_analysis.py
"""
import sys
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
from src.pipeline import TradingPipeline
from src.logger import get_logger
from src.config import config

logger = get_logger(__name__)

def print_header(text):
    """Imprimir cabeçalho formatado"""
    print(f"\n{'█'*80}")
    print(f"█ {text:<76} █")
    print(f"{'█'*80}\n")

def format_table(df, title, max_rows=10):
    """Formatar e imprimir tabela"""
    print(f"\n📊 {title}")
    print("-" * 100)
    
    # Limitar a número de rows
    if len(df) > max_rows:
        display_df = pd.concat([df.head(max_rows//2), df.tail(max_rows//2)])
        print(display_df.to_string())
        print(f"... (mostrando {max_rows} de {len(df)} linhas)")
    else:
        print(df.to_string())
    print()

def analyze_data():
    """Analisar dados coletados"""
    print_header("1. ANÁLISE DE DADOS COLETADOS")
    
    pipeline = TradingPipeline()
    
    print("📥 Coletando dados de BTCUSDT (últimos 5 dias)...")
    long_data, short_data, full_df = pipeline.fetch_and_prepare_data(
        "BTCUSDT",
        days_back=5
    )
    
    # Tabela de resumo
    summary = pd.DataFrame({
        'Dataset': ['Long Window (5d)', 'Short Window (5h)', 'Full Data'],
        'Candles': [len(long_data), len(short_data), len(full_df)],
        'Features': [long_data.shape[1], short_data.shape[1], full_df.shape[1]],
        'Período': ['5 dias', '5 horas', f'{len(full_df)} candles × 5m']
    })
    format_table(summary, "Resumo dos Datasets")
    
    # Mostrar dados brutos
    print("\n📈 Dados Brutos (últimos 5 candles):")
    cols_to_show = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    format_table(full_df[cols_to_show].tail(), "OHLCV", max_rows=5)
    
    # Estatísticas
    print("\n📊 Estatísticas de Preço:")
    stats_df = pd.DataFrame({
        'Métrica': ['Mínimo', 'Q1 (25%)', 'Mediana', 'Q3 (75%)', 'Máximo', 'Média', 'Desvio Padrão'],
        'Close': [
            full_df['close'].min(),
            full_df['close'].quantile(0.25),
            full_df['close'].median(),
            full_df['close'].quantile(0.75),
            full_df['close'].max(),
            full_df['close'].mean(),
            full_df['close'].std()
        ]
    })
    format_table(stats_df, "Estatísticas do Close", max_rows=10)
    
    # Features calculadas
    feature_cols = [c for c in full_df.columns if c not in 
                   ['timestamp', 'date', 'open', 'high', 'low', 'close', 'volume', 'quote_volume']]
    
    print(f"\n🔧 Features Engenheiradas ({len(feature_cols)} total):")
    features_desc = pd.DataFrame({
        'Feature': feature_cols,
        'Descrição': [
            'Retorno logarítmico',
            'Volatilidade (12 períodos)',
            'Volatilidade (24 períodos)',
            'Volatilidade (48 períodos)',
            'Volume normalizado (Z-score)',
            'Volume em USD normalizado (Z-score)',
            'Range High-Low',
            'Posição do Close no Range',
            'Volume / Close',
            'Quote Volume / Close',
            'Volume × Close',
            'Returns × Volume',
            'Volume SMA'
        ][:len(feature_cols)]
    })
    format_table(features_desc, "Features Disponíveis", max_rows=20)
    
    # Correlação
    print("\n🔗 Matriz de Correlação (primeiras 5 features):")
    corr_df = full_df[feature_cols[:5]].corr()
    print(corr_df.to_string())
    
    return pipeline, long_data, short_data, full_df


def analyze_features():
    """Analisar features e distribuições"""
    print_header("2. ANÁLISE DE FEATURES")
    
    pipeline, long_data, short_data, full_df = analyze_data()
    
    feature_cols = [c for c in full_df.columns if c not in 
                   ['timestamp', 'date', 'open', 'high', 'low', 'close', 'volume', 'quote_volume']]
    
    print("📊 Distribuição das Features (últimas 5h):")
    feature_stats = pd.DataFrame({
        'Feature': feature_cols,
        'Min': [short_data[f].min() for f in feature_cols],
        'Max': [short_data[f].max() for f in feature_cols],
        'Mean': [short_data[f].mean() for f in feature_cols],
        'Std': [short_data[f].std() for f in feature_cols]
    })
    format_table(feature_stats, "Estatísticas das Features", max_rows=20)
    
    # Top correlações
    print("\n🔗 Top 10 Correlações (valores absolutos):")
    corr_matrix = full_df[feature_cols].corr().abs()
    
    # Pegar upper triangle (evitar duplicatas)
    corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_pairs.append({
                'Feature 1': corr_matrix.columns[i],
                'Feature 2': corr_matrix.columns[j],
                'Correlação': corr_matrix.iloc[i, j]
            })
    
    corr_df_sorted = pd.DataFrame(corr_pairs).sort_values('Correlação', ascending=False)
    format_table(corr_df_sorted.head(10), "Top 10 Correlações", max_rows=10)
    
    return pipeline, long_data, short_data, full_df


def analyze_models():
    """Analisar arquitetura dos modelos"""
    print_header("3. ARQUITETURA DOS MODELOS")
    
    print("🧠 MacroNet Architecture:")
    print(f"""
    Input: (1, 1440, 13)
    ├─ TemporalEncoder
    │  ├─ Conv1D(13→32, kernel=3, dilation=1) + ReLU
    │  ├─ Conv1D(32→64, kernel=3, dilation=2) + ReLU
    │  ├─ Conv1D(64→128, kernel=3, dilation=4) + ReLU
    │  └─ AttentionPooling → ({config.macronet.embedding_dim},)
    └─ Output: (1, {config.macronet.embedding_dim}) [Embedding]
    
    Autoencoder Loss: MSE(input, reconstructed)
    
    Parâmetros configuráveis (config.py):
      • embedding_dim: {config.macronet.embedding_dim}
      • encoder_layers: {config.macronet.encoder_layers}
      • hidden_dim: {config.macronet.hidden_dim}
      • learning_rate: {config.macronet.learning_rate}
      • epochs: {config.macronet.epochs}
    """)
    
    print("\n🎯 MicroNet Architecture:")
    print(f"""
    Input 1: (1, {config.micronet.lookback_candles}, 13)   [Últimas 5h]
    Input 2: (1, {config.macronet.embedding_dim})      [Macro embedding]
    ├─ Short Processor: Conv1D + Flatten
    ├─ Concatenation: [short_features, macro_embedding]
    └─ DecisionHead
       ├─ Dense(128→64) + ReLU + Dropout
       ├─ Dense(64→32) + ReLU + Dropout
       └─ Dense(32→1) + Tanh → [-1, 1]
    
    Training Loss: MSE(signal, future_returns)
    
    Parâmetros configuráveis:
      • lookback_candles: {config.micronet.lookback_candles} (5h @ 5m)
      • decision_dropout: {config.micronet.dropout}
      • learning_rate: {config.micronet.learning_rate}
      • epochs: {config.micronet.epochs}
    """)
    
    print("\n📊 Feature Engineering Pipeline:")
    print("""
    Raw OHLCV (Open, High, Low, Close, Volume)
    ├─ Technical Indicators
    │  ├─ log_returns: ln(close_t / close_t-1)
    │  ├─ volatility_N: rolling std dev (windows 12,24,48)
    │  ├─ volume_zscore: (volume - mean) / std
    │  ├─ hl_range: high - low
    │  └─ close_position: (close - low) / (high - low)
    └─ Normalization: MinMax or ZScore
    """)


def interactive_menu():
    """Menu interativo"""
    print_header("🚀 ANALISADOR INTERATIVO - SISTEMA DE TRADING")
    
    while True:
        print("\n📋 Escolha uma opção:")
        print("  1. Analisar Dados Coletados")
        print("  2. Analisar Features e Correlações")
        print("  3. Ver Arquitetura dos Modelos")
        print("  4. Executar Demo Completo (longo)")
        print("  5. Sair")
        
        choice = input("\n👉 Opção (1-5): ").strip()
        
        if choice == '1':
            pipeline, long_data, short_data, full_df = analyze_data()
        elif choice == '2':
            pipeline, long_data, short_data, full_df = analyze_features()
        elif choice == '3':
            analyze_models()
        elif choice == '4':
            print("\n⏳ Executando demo completo (5-10 minutos)...")
            from demo_complete_workflow import demo_full_workflow
            demo_full_workflow()
        elif choice == '5':
            print("\n👋 Até logo!")
            break
        else:
            print("❌ Opção inválida!")


if __name__ == "__main__":
    try:
        interactive_menu()
    except KeyboardInterrupt:
        print("\n\n👋 Interrompido pelo usuário")
    except Exception as e:
        print(f"\n❌ Erro: {e}")
        import traceback
        traceback.print_exc()
