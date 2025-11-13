"""
Configuration management for cppncripto
"""
import os
from dataclasses import dataclass
from pathlib import Path
import yaml
from dotenv import load_dotenv

load_dotenv()

@dataclass
class DataConfig:
    """Data configuration"""
    data_dir: str = os.getenv("DATA_DIR", "./data")
    symbols: list = None
    timeframe: str = "5m"
    max_candles_per_request: int = 1000
    
    def __post_init__(self):
        if self.symbols is None:
            symbols_str = os.getenv("DEFAULT_SYMBOLS", "BTCUSDT,ETHUSDT")
            self.symbols = [s.strip() for s in symbols_str.split(",")]
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)

@dataclass
class MacroNetConfig:
    """MacroNet (Encoder) configuration"""
    embedding_dim: int = 128
    encoder_layers: int = 3
    hidden_dim: int = 256
    dropout: float = 0.1
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 50
    lookback_days: int = 5
    model_path: str = "./models/macronet"
    
    def __post_init__(self):
        Path(self.model_path).mkdir(parents=True, exist_ok=True)

@dataclass
class MicroNetConfig:
    """MicroNet (Decision Head) configuration"""
    hidden_dim: int = 64
    dropout: float = 0.1
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    lookback_candles: int = 60  # 5 horas em 5m
    decision_threshold: float = 0.5
    adaptive_threshold: bool = True
    threshold_percentile: float = 15.0
    model_path: str = "./models/micronet"
    
    def __post_init__(self):
        Path(self.model_path).mkdir(parents=True, exist_ok=True)

@dataclass
class EmbeddingConfig:
    """Embedding cache configuration"""
    embedding_dir: str = "./embeddings"
    cache_ttl_hours: int = 24
    
    def __post_init__(self):
        Path(self.embedding_dir).mkdir(parents=True, exist_ok=True)

@dataclass
class BacktestConfig:
    """Backtesting configuration"""
    backtest_dir: str = "./backtests"
    initial_cash: float = 10000.0
    commission: float = 0.001
    slippage: float = 0.0005
    
    def __post_init__(self):
        Path(self.backtest_dir).mkdir(parents=True, exist_ok=True)

@dataclass
class Config:
    """Main configuration"""
    data: DataConfig = None
    macronet: MacroNetConfig = None
    micronet: MicroNetConfig = None
    embedding: EmbeddingConfig = None
    backtest: BacktestConfig = None
    device: str = "cpu"
    
    def __post_init__(self):
        if self.data is None:
            self.data = DataConfig()
        if self.macronet is None:
            self.macronet = MacroNetConfig()
        if self.micronet is None:
            self.micronet = MicroNetConfig()
        if self.embedding is None:
            self.embedding = EmbeddingConfig()
        if self.backtest is None:
            self.backtest = BacktestConfig()
        
        # Auto-detect best device: CUDA > MPS (Apple) > XPU (Intel) > CPU
        self.device = self.detect_device()
    
    def detect_device(self) -> str:
        """Auto-detect best available device (GPU/MPS/CPU)"""
        try:
            import torch
            # Try to detect Intel Extension for PyTorch (ipex) - provides XPU support
            try:
                import intel_extension_for_pytorch as ipex  # type: ignore
                # If import succeeds, prefer xpu device name
                # Note: actual runtime device may still be 'cpu' but ipex will accelerate
                return "xpu"
            except Exception:
                # ipex not installed; continue with other backends
                pass
            
            # Priority: CUDA > MPS (Apple) > CPU
            if torch.cuda.is_available():
                return "cuda"
            
            # Apple Silicon (M1, M2, M3, etc) - Metal Performance Shaders
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            
            # Intel GPU (Arc, Iris Xe) - via oneAPI
            # Some builds expose torch.backends.xpu for Intel GPUs
            if hasattr(torch.backends, "xpu") and getattr(torch.backends.xpu, "is_available", lambda: False)():
                return "xpu"
            
            # Fallback: CPU
            return "cpu"
        except:
            return "cpu"
    
    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load configuration from YAML file"""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        
        config = cls()
        if "data" in data:
            config.data = DataConfig(**data["data"])
        if "macronet" in data:
            config.macronet = MacroNetConfig(**data["macronet"])
        if "micronet" in data:
            config.micronet = MicroNetConfig(**data["micronet"])
        if "embedding" in data:
            config.embedding = EmbeddingConfig(**data["embedding"])
        if "backtest" in data:
            config.backtest = BacktestConfig(**data["backtest"])
        
        return config
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "data": {
                "data_dir": self.data.data_dir,
                "symbols": self.data.symbols,
                "timeframe": self.data.timeframe,
            },
            "macronet": {
                "embedding_dim": self.macronet.embedding_dim,
                "encoder_layers": self.macronet.encoder_layers,
                "hidden_dim": self.macronet.hidden_dim,
                "lookback_days": self.macronet.lookback_days,
            },
            "micronet": {
                "hidden_dim": self.micronet.hidden_dim,
                "lookback_candles": self.micronet.lookback_candles,
                "decision_threshold": self.micronet.decision_threshold,
            },
            "device": self.device,
        }

# Global config instance
config = Config()
