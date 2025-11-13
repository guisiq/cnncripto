"""
Tests for cppncripto modules
"""
import pytest
import numpy as np
import pandas as pd
from src.features.builder import FeatureBuilder
from src.macronet.model import MacroNet
from src.micronet.model import MicroNet
from src.evaluation.backtest import SimpleBacktester, calculate_metrics

class TestFeatureBuilder:
    """Test feature engineering"""
    
    def test_log_returns(self):
        close = pd.Series([100, 101, 102, 103])
        returns = FeatureBuilder.log_returns(close)
        assert len(returns) == len(close)
        assert np.isnan(returns.iloc[0])
    
    def test_rolling_volatility(self):
        close = pd.Series(np.random.randn(100).cumsum() + 100)
        vol = FeatureBuilder.rolling_volatility(close, windows=[12, 24])
        assert len(vol) == len(close)
        assert 'volatility_12' in vol.columns
    
    def test_build_features(self):
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='5min'),
            'open': np.random.uniform(100, 105, 100),
            'high': np.random.uniform(100, 106, 100),
            'low': np.random.uniform(99, 105, 100),
            'close': np.random.uniform(100, 105, 100),
            'volume': np.random.uniform(1, 100, 100),
            'quote_volume': np.random.uniform(1000, 10000, 100),
        })
        
        features = FeatureBuilder.build_features(df)
        assert len(features) < len(df)  # NaNs dropped
        assert 'log_return' in features.columns
        assert 'volatility_12' in features.columns

class TestMacroNet:
    """Test MacroNet"""
    
    def test_macronet_encoding(self):
        macronet = MacroNet()
        
        # Generate dummy data
        X = np.random.randn(10, 100, 8).astype(np.float32)
        
        macronet.train(X, epochs=2)
        embeddings = macronet.encode(X)
        
        assert embeddings.shape == (10, 128)
    
    def test_macronet_save_load(self, tmp_path):
        macronet = MacroNet()
        X = np.random.randn(5, 100, 8).astype(np.float32)
        macronet.train(X, epochs=1)
        
        path = tmp_path / "test_model.pt"
        macronet.save_model(str(path))
        assert path.exists()

class TestMicroNet:
    """Test MicroNet"""
    
    def test_micronet_prediction(self):
        micronet = MicroNet()
        
        X_short = np.random.randn(50, 15).astype(np.float32)
        macro_emb = np.random.randn(50, 128).astype(np.float32)
        y = np.random.uniform(-1, 1, (50, 1)).astype(np.float32)
        
        micronet.train(X_short, macro_emb, y, epochs=2)
        scores = micronet.predict(X_short, macro_emb)
        
        assert scores.shape == (50,)
        assert np.all((scores >= -1) & (scores <= 1))

class TestBacktester:
    """Test backtesting"""
    
    def test_simple_backtest(self):
        backtester = SimpleBacktester()
        
        prices = np.linspace(100, 110, 100)
        signals = np.sin(np.linspace(0, 2*np.pi, 100))
        
        results = backtester.backtest(prices, signals, signal_threshold=0.5)
        
        assert 'total_return' in results
        assert 'sharpe_ratio' in results
        assert 'max_drawdown' in results
    
    def test_calculate_metrics(self):
        equity_curve = np.array([10000, 10100, 10200, 10050, 10500])
        metrics = calculate_metrics(equity_curve)
        
        assert 'total_return' in metrics
        assert 'sharpe_ratio' in metrics
        assert 'max_drawdown' in metrics

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
