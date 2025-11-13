"""
Main training and inference pipeline
"""
import numpy as np
import pandas as pd
from pathlib import Path
import json
from datetime import datetime

from src.ingest.binance import BinanceIngestor
from src.features.builder import FeatureBuilder
from src.macronet.model import MacroNet
from src.micronet.model import MicroNet
from src.evaluation.backtest import SimpleBacktester, calculate_metrics
from src.logger import get_logger
from src.config import config

logger = get_logger(__name__)

class TradingPipeline:
    """Complete training and inference pipeline"""
    
    def __init__(self, config_obj=None):
        self.config = config_obj or config
        self.ingestor = BinanceIngestor()
        self.feature_builder = FeatureBuilder()
        self.macronet = MacroNet()
        self.micronet = MicroNet()
        self.backtester = SimpleBacktester()
        
        # Cache
        self.current_macro_embedding = None
        self.last_macro_embedding_date = None
    
    def fetch_and_prepare_data(
        self,
        symbol: str,
        days_back: int = 30,
        lookback_days: int = 5
    ) -> tuple:
        """
        Fetch data and prepare for training/inference
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            days_back: Days to fetch
            lookback_days: Days for MacroNet lookback
        
        Returns:
            (long_data, short_data, features_df)
        """
        logger.info("fetching_data", symbol=symbol, days_back=days_back)
        
        # Try to load from cache first
        try:
            df = self.ingestor.load_from_parquet(symbol)
            if len(df) == 0:
                raise FileNotFoundError("Empty cache")
        except:
            # Fetch from API
            df = self.ingestor.ingest_symbol(symbol, days_back=days_back)
        
        # Build features
        df = self.feature_builder.build_features(df, drop_na=True)
        df = self.feature_builder.normalize_features(df, method='minmax')
        
        # Split long (macro) and short (micro) data
        lookback_candles = lookback_days * (24 * 60) // 5  # Convert days to 5m candles
        
        long_data = df.iloc[-lookback_candles:].copy()
        short_data = df.tail(60).copy()  # Last 5 hours
        
        logger.info(
            "data_prepared",
            symbol=symbol,
            total_rows=len(df),
            long_rows=len(long_data),
            short_rows=len(short_data)
        )
        
        return long_data, short_data, df
    
    def extract_feature_arrays(self, df: pd.DataFrame) -> np.ndarray:
        """Extract feature matrix from DataFrame"""
        exclude_cols = ['timestamp', 'date', 'open', 'high', 'low', 'close', 'volume', 'quote_volume']
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        return df[feature_cols].values
    
    def train_macronet(self, symbol: str, days_back: int = 30):
        """Train MacroNet on historical data"""
        logger.info("training_macronet", symbol=symbol)
        
        long_data, _, _ = self.fetch_and_prepare_data(symbol, days_back=days_back)
        
        X_features = self.extract_feature_arrays(long_data)
        
        # Reshape for sequence: (1, seq_len, num_features)
        X = X_features[np.newaxis, :, :]
        
        self.macronet.train(X, epochs=self.config.macronet.epochs)
    
    def generate_macro_embedding(self, symbol: str, days_back: int = 5) -> np.ndarray:
        """Generate macro embedding for current day"""
        logger.info("generating_macro_embedding", symbol=symbol)
        
        long_data, _, _ = self.fetch_and_prepare_data(symbol, days_back=days_back)
        
        X_features = self.extract_feature_arrays(long_data)
        X = X_features[np.newaxis, :, :]
        
        embedding = self.macronet.encode(X)
        self.current_macro_embedding = embedding[0]
        self.last_macro_embedding_date = datetime.now().date()
        
        logger.info("macro_embedding_generated", shape=embedding.shape)
        return embedding[0]
    
    def train_micronet(
        self,
        symbol: str,
        days_back: int = 30,
        lookback_short: int = 60
    ):
        """Train MicroNet on historical signals"""
        logger.info("training_micronet", symbol=symbol)
        
        _, _, full_df = self.fetch_and_prepare_data(symbol, days_back=days_back)
        
        # Generate macro embedding on historical data
        if self.macronet.model is None:
            self.train_macronet(symbol, days_back=days_back)
        
        # Create training dataset
        X_short_list = []
        macro_embeddings_list = []
        y_list = []
        
        # For each day, generate embedding + short windows
        daily_groups = full_df.groupby(full_df['timestamp'].dt.date)
        
        for date, day_df in daily_groups:
            if len(day_df) < lookback_short + 5:
                continue
            
            # Macro embedding for the day (using first 80% of day)
            idx_split = int(len(day_df) * 0.8)
            day_long = day_df.iloc[:idx_split]
            
            X_long_feats = self.extract_feature_arrays(day_long)
            if len(X_long_feats) > 100:
                X_long = X_long_feats[np.newaxis, :, :]
                macro_emb = self.macronet.encode(X_long)[0]
            else:
                continue
            
            # Short windows on remaining 20%
            day_short = day_df.iloc[idx_split:].reset_index(drop=True)
            
            for i in range(lookback_short, len(day_short) - 5):
                window = day_short.iloc[i - lookback_short:i]
                X_short_feats = self.extract_feature_arrays(window)
                
                # Label: future return (next 5 candles)
                future_price = day_short.iloc[i + 5]['close']
                current_price = day_short.iloc[i]['close']
                future_return = np.log(future_price / current_price)
                
                # Convert to score [-1, 1]
                label = np.tanh(future_return * 20)
                
                X_short_list.append(X_short_feats[-1])  # Last feature vector
                macro_embeddings_list.append(macro_emb)
                y_list.append(label)
        
        if len(X_short_list) == 0:
            logger.warning("no_training_samples")
            return
        
        X_short = np.array(X_short_list)
        macro_embeddings = np.array(macro_embeddings_list)
        y = np.array(y_list).reshape(-1, 1)
        
        self.micronet.train(
            X_short,
            macro_embeddings,
            y,
            epochs=self.config.micronet.epochs
        )
    
    def predict_signal(self, symbol: str) -> float:
        """Generate prediction for current time"""
        _, short_data, _ = self.fetch_and_prepare_data(symbol)
        
        if self.current_macro_embedding is None:
            self.generate_macro_embedding(symbol)
        
        X_short_feats = self.extract_feature_arrays(short_data)
        
        if len(X_short_feats) == 0:
            logger.warning("no_short_data")
            return 0.0
        
        X_short = X_short_feats[-1:].astype(np.float32)
        macro_emb = self.current_macro_embedding[np.newaxis, :].astype(np.float32)
        
        score = self.micronet.predict(X_short, macro_emb)[0]
        
        logger.info("signal_generated", signal=score)
        return float(score)
    
    def backtest_strategy(
        self,
        symbol: str,
        days_back: int = 30,
        signal_threshold: float = 0.5
    ) -> dict:
        """Run backtest"""
        logger.info("running_backtest", symbol=symbol)
        
        _, _, full_df = self.fetch_and_prepare_data(symbol, days_back=days_back)
        
        # Generate signals for full history
        signals = []
        close_prices = full_df['close'].values
        
        daily_groups = full_df.groupby(full_df['timestamp'].dt.date)
        
        for date, day_df in daily_groups:
            if len(day_df) < 100:
                signals.extend([0.0] * len(day_df))
                continue
            
            # Generate macro embedding for day
            idx_split = int(len(day_df) * 0.8)
            day_long = day_df.iloc[:idx_split]
            
            X_long_feats = self.extract_feature_arrays(day_long)
            if len(X_long_feats) > 100:
                X_long = X_long_feats[np.newaxis, :, :]
                macro_emb = self.macronet.encode(X_long)[0]
            else:
                signals.extend([0.0] * len(day_df))
                continue
            
            # Generate signals for day
            day_short = day_df.iloc[idx_split:].reset_index(drop=True)
            for i in range(len(day_short)):
                if i < 60:
                    signals.append(0.0)
                else:
                    window = day_short.iloc[i - 60:i]
                    X_short_feats = self.extract_feature_arrays(window)
                    X_short = X_short_feats[-1:].astype(np.float32)
                    macro_emb_single = macro_emb[np.newaxis, :].astype(np.float32)
                    
                    try:
                        score = self.micronet.predict(X_short, macro_emb_single)[0]
                        signals.append(score)
                    except:
                        signals.append(0.0)
        
        signals = np.array(signals)
        
        # Run backtest
        results = self.backtester.backtest(close_prices, signals, signal_threshold=signal_threshold)
        
        logger.info(
            "backtest_results",
            return_pct=f"{results['total_return']:.2%}",
            sharpe=f"{results['sharpe_ratio']:.2f}"
        )
        
        return results

if __name__ == "__main__":
    # Example usage
    pipeline = TradingPipeline()
    
    # Train
    # pipeline.train_macronet("BTCUSDT", days_back=10)
    # pipeline.train_micronet("BTCUSDT", days_back=10)
    
    # Predict
    # score = pipeline.predict_signal("BTCUSDT")
    # print(f"Signal: {score}")
    
    # Backtest
    # results = pipeline.backtest_strategy("BTCUSDT", days_back=10)
    # print(results)
