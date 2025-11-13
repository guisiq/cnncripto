"""
Binance data ingestion module
"""
import os
import pandas as pd
import polars as pl
from binance.client import Client
from pathlib import Path
from datetime import datetime, timedelta
import time
from src.logger import get_logger
from src.config import config

logger = get_logger(__name__)

class BinanceIngestor:
    """Fetch and store Binance candle data"""
    
    def __init__(self, api_key: str = None, api_secret: str = None):
        self.api_key = api_key or os.getenv("BINANCE_API_KEY")
        self.api_secret = api_secret or os.getenv("BINANCE_API_SECRET")
        
        if self.api_key and self.api_secret:
            self.client = Client(api_key=self.api_key, api_secret=self.api_secret)
        else:
            self.client = Client()
        
        self.data_dir = Path(config.data.data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def fetch_candles(
        self, 
        symbol: str, 
        interval: str = "5m", 
        days_back: int = 30
    ) -> pd.DataFrame:
        """
        Fetch historical candles from Binance
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            interval: Kline interval (default '5m')
            days_back: Number of days to fetch (default 30)
        
        Returns:
            DataFrame with columns: open, high, low, close, volume, etc.
        """
        logger.info("fetching_candles", symbol=symbol, days_back=days_back, interval=interval)
        
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=days_back)
        
        all_candles = []
        current_time = start_time
        
        while current_time < end_time:
            try:
                # Fetch up to 1000 candles
                klines = self.client.get_historical_klines(
                    symbol=symbol,
                    interval=interval,
                    start_str=int(current_time.timestamp() * 1000),
                    end_str=int(end_time.timestamp() * 1000),
                    limit=1000
                )
                
                if not klines:
                    break
                
                all_candles.extend(klines)
                
                # Move to end of last batch
                last_time = datetime.fromtimestamp(klines[-1][0] / 1000)
                if last_time >= end_time:
                    break
                
                current_time = last_time + timedelta(minutes=5)
                time.sleep(0.1)  # Rate limiting
                
            except Exception as e:
                logger.error("fetch_error", symbol=symbol, error=str(e))
                raise
        
        # Convert to DataFrame
        df = pd.DataFrame(all_candles, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades_count', 
            'taker_buy_volume', 'taker_buy_quote_volume', 'ignore'
        ])
        
        # Clean up
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_volume']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col])
        
        df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
        df = df.drop(['open_time', 'close_time', 'ignore', 'taker_buy_quote_volume'], axis=1)
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        logger.info("candles_fetched", symbol=symbol, count=len(df))
        return df
    
    def save_to_parquet(self, df: pd.DataFrame, symbol: str, interval: str = "5m"):
        """Save candle data to Parquet (partitioned by symbol)"""
        symbol_dir = self.data_dir / f"timeframe={interval}" / f"symbol={symbol}"
        symbol_dir.mkdir(parents=True, exist_ok=True)
        
        # Partition by date
        df['date'] = df['timestamp'].dt.date
        
        pl_df = pl.from_pandas(df)
        
        # Write partitioned
        file_path = symbol_dir / "candles.parquet"
        pl_df.write_parquet(str(file_path))
        
        logger.info("saved_to_parquet", path=str(file_path), rows=len(df))
        return file_path
    
    def load_from_parquet(self, symbol: str, interval: str = "5m") -> pd.DataFrame:
        """Load candle data from Parquet"""
        symbol_dir = self.data_dir / f"timeframe={interval}" / f"symbol={symbol}"
        file_path = symbol_dir / "candles.parquet"
        
        if not file_path.exists():
            logger.warning("file_not_found", path=str(file_path))
            return pd.DataFrame()
        
        df = pl.read_parquet(str(file_path)).to_pandas()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df.sort_values('timestamp').reset_index(drop=True)
    
    def ingest_symbol(self, symbol: str, interval: str = "5m", days_back: int = 30):
        """Full ingestion pipeline: fetch -> save"""
        df = self.fetch_candles(symbol=symbol, interval=interval, days_back=days_back)
        self.save_to_parquet(df, symbol=symbol, interval=interval)
        return df

if __name__ == "__main__":
    ingestor = BinanceIngestor()
    
    # Example: fetch and save
    for symbol in ["BTCUSDT", "ETHUSDT"]:
        try:
            ingestor.ingest_symbol(symbol, days_back=5)
        except Exception as e:
            logger.error("ingest_failed", symbol=symbol, error=str(e))
