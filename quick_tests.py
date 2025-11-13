"""
Script de Testes Rápidos - Validar cada componente individualmente
Executável: python quick_tests.py
"""
import sys
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
from src.pipeline import TradingPipeline
from src.config import config
from src.features.builder import FeatureBuilder
from src.macronet.model import MacroNet
from src.micronet.model import MicroNet
from src.evaluation.backtest import SimpleBacktester
from src.logger import get_logger

logger = get_logger(__name__)

def test_header(name):
    """Imprimir cabeçalho de teste"""
    print(f"\n{'='*70}")
    print(f"  TEST: {name}")
    print(f"{'='*70}")

def test_config():
    """Teste 1: Verificar Configurações"""
    test_header("1. Verificar Configurações")
    
    print(f"✓ Device: {config.device}")
    print(f"✓ MacroNet embedding_dim: {config.macronet.embedding_dim}")
    print(f"✓ MicroNet lookback_candles: {config.micronet.lookback_candles}")
    print(f"✓ Backtest commission: {config.backtest.commission*100}%")
    print(f"✓ Backtest slippage: {config.backtest.slippage*100}%")
    
    return True


def test_data_ingestion():
    """Teste 2: Ingestão de Dados"""
    test_header("2. Ingestão de Dados (Binance)")
    
    pipeline = TradingPipeline()
    
    print("⏳ Coletando 2 dias de BTCUSDT...")
    long_data, short_data, full_df = pipeline.fetch_and_prepare_data("BTCUSDT", days_back=2)
    
    assert len(full_df) > 0, "Nenhum dado coletado!"
    assert 'close' in full_df.columns, "Coluna 'close' não encontrada!"
    assert 'volume' in full_df.columns, "Coluna 'volume' não encontrada!"
    
    print(f"✓ Total de candles: {len(full_df)}")
    print(f"✓ Features: {full_df.shape[1]}")
    print(f"✓ Data range: {full_df['timestamp'].min()} a {full_df['timestamp'].max()}")
    
    return True


def test_feature_engineering():
    """Teste 3: Engenharia de Features"""
    test_header("3. Engenharia de Features")
    
    # Dados dummy
    df = pd.DataFrame({
        'close': np.random.randn(100).cumsum() + 100,
        'high': np.random.randn(100).cumsum() + 105,
        'low': np.random.randn(100).cumsum() + 95,
        'volume': np.random.rand(100) * 1000000,
        'quote_volume': np.random.rand(100) * 50000000
    })
    
    print("⏳ Calculando features em dados dummy...")
    features_df = FeatureBuilder.build_features(df)
    
    assert features_df.shape[0] > 0, "Nenhuma feature calculada!"
    assert features_df.shape[1] > df.shape[1], "Features não adicionadas!"
    
    feature_cols = [c for c in features_df.columns if c not in df.columns]
    
    print(f"✓ Features calculadas: {len(feature_cols)}")
    print(f"  {feature_cols}")
    
    # Teste normalização
    features_norm = FeatureBuilder.normalize_features(features_df, method='minmax')
    print(f"✓ Features normalizadas (minmax)")
    
    return True


def test_macronet():
    """Teste 4: MacroNet Training"""
    test_header("4. MacroNet Training")
    
    print("⏳ Criando dados dummy (batch_size=1, seq_len=100, features=13)...")
    X = np.random.randn(1, 100, 13).astype(np.float32)
    
    print("⏳ Inicializando MacroNet...")
    macronet = MacroNet(config_obj=config.macronet)

    print("⏳ Treinando por 2 epochs...")
    macronet.train(X, epochs=2)
    
    print("⏳ Gerando embedding...")
    embedding = macronet.encode(X)
    
    assert embedding.shape == (1, config.macronet.embedding_dim), f"Embedding shape errado: {embedding.shape}"
    
    print(f"✓ MacroNet funcionando!")
    print(f"  Input: {X.shape}")
    print(f"  Embedding: {embedding.shape}")
    
    return True


def test_micronet():
    """Teste 5: MicroNet Training"""
    test_header("5. MicroNet Training")
    
    print("⏳ Criando dados dummy...")
    X_short = np.random.randn(1, 60, 13).astype(np.float32)
    X_macro = np.random.randn(1, 128).astype(np.float32)
    y = np.random.randn(1, 1).astype(np.float32)
    
    print("⏳ Inicializando MicroNet...")
    micronet = MicroNet(config_obj=config.micronet)
    
    print("⏳ Treinando por 2 epochs...")
    micronet.train(X_short, X_macro, y, epochs=2)
    
    print("⏳ Gerando sinal...")
    signal = micronet.predict(X_short, X_macro)
    # predict may return array; take scalar for display/assert
    signal_val = float(signal.flatten()[0]) if hasattr(signal, 'flatten') else float(signal)

    assert -1.0 <= signal_val <= 1.0, f"Signal fora do range: {signal_val}"

    print(f"✓ MicroNet funcionando!")
    print(f"  Short input: {X_short.shape}")
    print(f"  Macro input: {X_macro.shape}")
    print(f"  Signal: {signal_val:.4f}")
    
    return True


def test_backtest():
    """Teste 6: Backtesting"""
    test_header("6. Backtesting")
    
    print("⏳ Criando dados simulados...")
    prices = np.random.randn(100).cumsum() + 100
    signals = np.random.rand(100) * 2 - 1  # [-1, 1]
    
    print("⏳ Executando backtest...")
    backtester = SimpleBacktester(
        initial_cash=10000,
        commission=config.backtest.commission,
    )
    
    results = backtester.backtest(prices, signals)
    
    assert 'total_return' in results, "total_return não encontrado!"
    assert 'sharpe_ratio' in results, "sharpe_ratio não encontrado!"
    
    print(f"✓ Backtest funcionando!")
    print(f"  Total return: {results['total_return']*100:.2f}%")
    print(f"  Sharpe ratio: {results['sharpe_ratio']:.2f}")
    print(f"  Max drawdown: {results['max_drawdown']*100:.2f}%")
    print(f"  Trades: {results['num_trades']:.0f}")
    
    return True


def test_pipeline():
    """Teste 7: Pipeline Completo"""
    test_header("7. Pipeline Completo")
    
    pipeline = TradingPipeline()
    
    print("⏳ Passo 1: Fetch data...")
    long_data, short_data, full_df = pipeline.fetch_and_prepare_data("BTCUSDT", days_back=2)
    print(f"  ✓ {len(full_df)} candles")
    
    print("⏳ Passo 2: Extract features...")
    X_features = pipeline.extract_feature_arrays(long_data)
    print(f"  ✓ Shape: {X_features.shape}")
    
    print("⏳ Passo 3: Train macronet...")
    X = X_features[np.newaxis, :, :]
    pipeline.macronet.train(X, epochs=1)
    print(f"  ✓ Trained")
    
    print("⏳ Passo 4: Generate embedding...")
    embedding = pipeline.generate_macro_embedding("BTCUSDT", days_back=2)
    print(f"  ✓ Embedding: {embedding.shape}")
    
    print("⏳ Passo 5: Train micronet...")
    # Train micronet with actual feature dimensions from pipeline
    short_data_features = pipeline.extract_feature_arrays(short_data)  # (60, 10)
    # Expand to batch: (10, 60, 10)
    X_short_train = np.repeat(short_data_features[np.newaxis, :, :], 10, axis=0).astype(np.float32)
    X_macro_train = np.random.randn(10, 128).astype(np.float32)
    y_train = np.random.uniform(-1, 1, (10, 1)).astype(np.float32)
    pipeline.micronet.train(X_short_train, X_macro_train, y_train, epochs=1)
    print(f"  ✓ Trained")
    
    print("⏳ Passo 6: Generate signal...")
    signal = pipeline.predict_signal("BTCUSDT")
    print(f"  ✓ Signal: {signal:.4f}")
    
    print("\n✓ Pipeline completo funcionando!")
    
    return True


def run_all_tests():
    """Executar todos os testes"""
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + "  🧪 QUICK TESTS - Validar Componentes".center(68) + "█")
    print("█" + " "*68 + "█")
    print("█"*70)
    
    tests = [
        ("Config", test_config),
        ("Data Ingestion", test_data_ingestion),
        ("Feature Engineering", test_feature_engineering),
        ("MacroNet", test_macronet),
        ("MicroNet", test_micronet),
        ("Backtest", test_backtest),
        ("Pipeline", test_pipeline),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, "✅ PASS"))
            print(f"\n✅ {name}: PASSED")
        except Exception as e:
            results.append((name, f"❌ FAIL: {str(e)[:40]}"))
            print(f"\n❌ {name}: FAILED")
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
    
    # Resumo
    print("\n" + "="*70)
    print("  RESUMO DOS TESTES")
    print("="*70)
    
    passed = sum(1 for _, r in results if "✅" in r)
    total = len(results)
    
    for name, result in results:
        print(f"{name:30} {result}")
    
    print(f"\nTotal: {passed}/{total} ✅")
    
    if passed == total:
        print("\n🎉 Todos os testes passaram!")
    else:
        print(f"\n⚠️  {total - passed} teste(s) falharam")


if __name__ == "__main__":
    try:
        run_all_tests()
    except KeyboardInterrupt:
        print("\n👋 Interrompido")
    except Exception as e:
        print(f"\n❌ Erro geral: {e}")
        import traceback
        traceback.print_exc()
