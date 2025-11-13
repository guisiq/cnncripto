"""
MicroNet: Decision head for intraday signals (executed every 5m evaluation)
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
from src.logger import get_logger
from src.config import config

logger = get_logger(__name__)

class DecisionHead(nn.Module):
    """Decision head combining short-term features + macro embedding"""
    
    def __init__(
        self,
        short_window_features: int,
        macro_embedding_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Process short-term window
        self.short_processor = nn.Sequential(
            nn.Linear(short_window_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        # Combine short-term + macro embedding
        combined_dim = (hidden_dim // 2) + macro_embedding_dim
        
        self.decision_head = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh()  # Output in [-1, 1]
        )
    
    def forward(self, x_short, macro_embedding):
        """
        Args:
            x_short: (batch_size, short_window_features)
            macro_embedding: (batch_size, macro_embedding_dim)
        
        Returns:
            score: (batch_size, 1) in [-1, 1]
        """
        short_feat = self.short_processor(x_short)
        combined = torch.cat([short_feat, macro_embedding], dim=1)
        score = self.decision_head(combined)
        return score

class MicroNet:
    """MicroNet training and inference wrapper"""
    
    def __init__(self, config_obj=None):
        self.config = config_obj or config.micronet
        self.device = config.device
        self.model_dir = Path(self.config.model_path)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.optimizer = None
        self.device_obj = torch.device(self.device)
    
    def build_model(self, short_features_dim: int, macro_embedding_dim: int):
        """Build decision head model"""
        logger.info(
            "building_micronet_model",
            short_features=short_features_dim,
            macro_embedding=macro_embedding_dim
        )
        
        self.model = DecisionHead(
            short_window_features=short_features_dim,
            macro_embedding_dim=macro_embedding_dim,
            hidden_dim=self.config.hidden_dim,
            dropout=self.config.dropout
        ).to(self.device_obj)
        
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate
        )
        
        logger.info("model_built", model_params=sum(p.numel() for p in self.model.parameters()))
    
    def train(
        self,
        X_short: np.ndarray,
        macro_embeddings: np.ndarray,
        y: np.ndarray,
        epochs: int = None,
        batch_size: int = None
    ):
        """
        Train decision head
        
        Args:
            X_short: (num_samples, short_window_features)
            macro_embeddings: (num_samples, macro_embedding_dim)
            y: (num_samples, 1) targets in [-1, 1]
            epochs: Number of training epochs
            batch_size: Batch size
        """
        epochs = epochs or self.config.epochs
        batch_size = batch_size or self.config.batch_size
        
        if self.model is None:
            short_dim = X_short.shape[1]
            macro_dim = macro_embeddings.shape[1]
            self.build_model(short_dim, macro_dim)
        
        # Convert to tensors
        X_short_tensor = torch.from_numpy(X_short).float().to(self.device_obj)
        macro_tensor = torch.from_numpy(macro_embeddings).float().to(self.device_obj)
        y_tensor = torch.from_numpy(y).float().to(self.device_obj)
        
        dataset = TensorDataset(X_short_tensor, macro_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        criterion = nn.MSELoss()
        
        logger.info("starting_micronet_training", epochs=epochs, batch_size=batch_size, samples=len(X_short))
        
        for epoch in range(epochs):
            total_loss = 0
            for batch_X_short, batch_macro, batch_y in loader:
                self.optimizer.zero_grad()
                
                pred = self.model(batch_X_short, batch_macro)
                loss = criterion(pred, batch_y)
                
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(loader)
            
            if (epoch + 1) % 20 == 0:
                logger.info("micronet_training_progress", epoch=epoch + 1, loss=avg_loss)
        
        logger.info("micronet_training_complete", final_loss=avg_loss)
        self.save_model()
    
    def predict(self, X_short: np.ndarray, macro_embeddings: np.ndarray) -> np.ndarray:
        """
        Generate predictions
        
        Args:
            X_short: (num_samples, short_window_features)
            macro_embeddings: (num_samples, macro_embedding_dim)
        
        Returns:
            scores: (num_samples,) in [-1, 1]
        """
        if self.model is None:
            raise ValueError("Model not built or loaded")
        
        X_short_tensor = torch.from_numpy(X_short).float().to(self.device_obj)
        macro_tensor = torch.from_numpy(macro_embeddings).float().to(self.device_obj)
        
        with torch.no_grad():
            scores = self.model(X_short_tensor, macro_tensor)
        
        return scores.cpu().numpy().flatten()
    
    def save_model(self, path: str = None):
        """Save model to disk"""
        if self.model is None:
            logger.warning("no_micronet_model_to_save")
            return
        
        path = path or str(self.model_dir / "micronet_latest.pt")
        torch.save(self.model.state_dict(), path)
        logger.info("micronet_model_saved", path=path)
    
    def load_model(self, path: str, short_features_dim: int, macro_embedding_dim: int):
        """Load model from disk"""
        self.build_model(short_features_dim, macro_embedding_dim)
        state_dict = torch.load(path, map_location=self.device_obj)
        self.model.load_state_dict(state_dict)
        logger.info("micronet_model_loaded", path=path)

if __name__ == "__main__":
    # Example usage
    micronet = MicroNet()
    
    # Generate dummy data
    X_short_dummy = np.random.randn(1000, 15).astype(np.float32)
    macro_dummy = np.random.randn(1000, 128).astype(np.float32)
    y_dummy = np.random.uniform(-1, 1, (1000, 1)).astype(np.float32)
    
    micronet.train(X_short_dummy, macro_dummy, y_dummy, epochs=5)
    scores = micronet.predict(X_short_dummy, macro_dummy)
    print(f"Scores shape: {scores.shape}, range: [{scores.min():.3f}, {scores.max():.3f}]")
