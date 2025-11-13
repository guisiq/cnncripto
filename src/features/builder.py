"""
Feature engineering for candle data
"""
import pandas as pd
import numpy as np
from src.logger import get_logger

logger = get_logger(__name__)

class FeatureBuilder:
    """Generate technical features from candle data"""
    
    @staticmethod
    def log_returns(close: pd.Series) -> pd.Series:
        """Log returns: log(close_t / close_{t-1})"""
        return np.log(close / close.shift(1))
    
    @staticmethod
    def rolling_volatility(
        close: pd.Series, 
        windows: list = [12, 24, 48]
    ) -> pd.DataFrame:
        """Calculate rolling volatility (std of log returns)"""
        log_ret = FeatureBuilder.log_returns(close)
        
        volatility = pd.DataFrame()
        for w in windows:
            volatility[f"volatility_{w}"] = log_ret.rolling(window=w).std()
        
        return volatility
    
    @staticmethod
    def volume_features(volume: pd.Series, quote_volume: pd.Series, window: int = 24) -> pd.DataFrame:
        """Generate volume-related features"""
        features = pd.DataFrame()
        
        # Normalized volume
        vol_mean = volume.rolling(window=window).mean()
        vol_std = volume.rolling(window=window).std()
        features['volume_zscore'] = (volume - vol_mean) / (vol_std + 1e-8)
        
        # Quote asset volume (proxy for USD volume in spot)
        qvol_mean = quote_volume.rolling(window=window).mean()
        features['quote_volume_norm'] = quote_volume / (qvol_mean + 1e-8)
        
        return features
    
    @staticmethod
    def price_features(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 24) -> pd.DataFrame:
        """Generate price-based features"""
        features = pd.DataFrame()
        
        # High-Low range
        hl_range = (high - low) / close
        features['hl_range'] = hl_range
        features['hl_range_ma'] = hl_range.rolling(window=window).mean()
        
        # Close position in range
        features['close_position'] = (close - low) / (high - low + 1e-8)
        
        return features
    
    @staticmethod
    def build_features(
        df: pd.DataFrame,
        volatility_windows: list = [12, 24, 48],
        volume_window: int = 24,
        price_window: int = 24,
        drop_na: bool = True
    ) -> pd.DataFrame:
        """
        Build complete feature set from candle data
        
        Args:
            df: DataFrame with columns: open, high, low, close, volume, quote_volume
            volatility_windows: Windows for volatility calculation
            volume_window: Window for volume normalization
            price_window: Window for price features
            drop_na: Whether to drop NaN rows
        
        Returns:
            DataFrame with original data + derived features
        """
        logger.info("building_features", rows=len(df), columns=len(df.columns))
        
        features = df.copy()
        
        # Log returns
        features['log_return'] = FeatureBuilder.log_returns(features['close'])
        
        # Volatility features
        vol_features = FeatureBuilder.rolling_volatility(
            features['close'], 
            windows=volatility_windows
        )
        features = pd.concat([features, vol_features], axis=1)
        
        # Volume features
        vol_feats = FeatureBuilder.volume_features(
            features['volume'], 
            features['quote_volume'],
            window=volume_window
        )
        features = pd.concat([features, vol_feats], axis=1)
        
        # Price features
        price_feats = FeatureBuilder.price_features(
            features['high'],
            features['low'],
            features['close'],
            window=price_window
        )
        features = pd.concat([features, price_feats], axis=1)
        
        if drop_na:
            features = features.dropna()
        
        logger.info("features_built", final_shape=features.shape)
        return features
    
    @staticmethod
    def normalize_features(
        df: pd.DataFrame,
        feature_cols: list = None,
        method: str = "minmax"
    ) -> pd.DataFrame:
        """
        Normalize features to [-1, 1] or [0, 1] range
        
        Args:
            df: DataFrame with features
            feature_cols: List of columns to normalize (default: all except timestamp-related)
            method: 'minmax' or 'zscore'
        
        Returns:
            Normalized DataFrame
        """
        if feature_cols is None:
            exclude = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'quote_volume', 'date']
            feature_cols = [c for c in df.columns if c not in exclude and df[c].dtype in [np.float64, np.float32]]
        
        normalized = df.copy()
        
        for col in feature_cols:
            if col not in normalized.columns:
                continue
            
            if method == "minmax":
                col_min = normalized[col].min()
                col_max = normalized[col].max()
                normalized[col] = 2 * (normalized[col] - col_min) / (col_max - col_min + 1e-8) - 1
            
            elif method == "zscore":
                col_mean = normalized[col].mean()
                col_std = normalized[col].std()
                normalized[col] = (normalized[col] - col_mean) / (col_std + 1e-8)
        
        return normalized

if __name__ == "__main__":
    # Example usage
    sample_data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=100, freq='5min'),
        'open': np.random.uniform(40000, 50000, 100),
        'high': np.random.uniform(40100, 50100, 100),
        'low': np.random.uniform(39900, 49900, 100),
        'close': np.random.uniform(40000, 50000, 100),
        'volume': np.random.uniform(10, 100, 100),
        'quote_volume': np.random.uniform(400000, 5000000, 100),
    })
    
    builder = FeatureBuilder()
    features = builder.build_features(sample_data)
    print(features.head())
