"""
MacroNet: Encoder for long-term patterns (executed once per day)
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

class TemporalEncoder(nn.Module):
    """Temporal CNN encoder"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        embedding_dim: int = 128,
        num_layers: int = 3,
        kernel_size: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        
        # Dilated convolutions for temporal patterns
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        
        in_channels = input_dim
        for i in range(num_layers):
            dilation = 2 ** i
            self.conv_layers.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=hidden_dim,
                    kernel_size=kernel_size,
                    padding=(kernel_size - 1) * dilation // 2,
                    dilation=dilation
                )
            )
            self.bn_layers.append(nn.BatchNorm1d(hidden_dim))
            in_channels = hidden_dim
        
        # Attention pooling
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Softmax(dim=1)
        )
        
        # Output embedding
        self.embedding_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, embedding_dim)
        )
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, input_dim)
        
        Returns:
            embedding: (batch_size, embedding_dim)
        """
        # Conv expects (batch, channels, seq_len)
        x = x.transpose(1, 2)
        
        # Apply dilated convolutions
        for conv, bn in zip(self.conv_layers, self.bn_layers):
            x = conv(x)
            x = bn(x)
            x = self.relu(x)
            x = self.dropout(x)
        
        # Transpose back to (batch, seq_len, channels)
        x = x.transpose(1, 2)
        
        # Attention-based pooling
        attention_weights = self.attention(x)  # (batch, seq_len, 1)
        context = (x * attention_weights).sum(dim=1)  # (batch, hidden_dim)
        
        # Generate embedding
        embedding = self.embedding_layer(context)
        return embedding

class MacroNetAutoencoder(nn.Module):
    """Autoencoder for unsupervised macro embedding"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        embedding_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.encoder = TemporalEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # Decoder: reconstruct original sequence
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, input_dim)
        )
    
    def encode(self, x):
        """Get embedding"""
        return self.encoder(x)
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, input_dim)
        
        Returns:
            reconstruction: (batch_size, seq_len, input_dim)
            embedding: (batch_size, embedding_dim)
        """
        embedding = self.encoder(x)
        
        # Reconstruct using embedding
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        # Expand embedding to sequence length
        expanded = embedding.unsqueeze(1).expand(batch_size, seq_len, -1)
        
        # Decode
        reconstruction = self.decoder(expanded)
        
        return reconstruction, embedding

class MacroNet:
    """MacroNet training and inference wrapper"""
    
    def __init__(self, config_obj=None):
        self.config = config_obj or config.macronet
        self.device = config.device
        self.model_dir = Path(self.config.model_path)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.optimizer = None
        self.device_obj = torch.device(self.device)
    
    def build_model(self, input_dim: int):
        """Build autoencoder model"""
        logger.info("building_macronet_model", input_dim=input_dim, embedding_dim=self.config.embedding_dim)
        
        self.model = MacroNetAutoencoder(
            input_dim=input_dim,
            hidden_dim=self.config.hidden_dim,
            embedding_dim=self.config.embedding_dim,
            num_layers=self.config.encoder_layers,
            dropout=self.config.dropout
        ).to(self.device_obj)
        
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate
        )
        
        logger.info("model_built", model_params=sum(p.numel() for p in self.model.parameters()))
    
    def train(self, X: np.ndarray, epochs: int = None, batch_size: int = None):
        """
        Train autoencoder on data
        
        Args:
            X: (num_sequences, seq_len, num_features)
            epochs: Number of training epochs
            batch_size: Batch size
        """
        epochs = epochs or self.config.epochs
        batch_size = batch_size or self.config.batch_size
        
        if self.model is None:
            input_dim = X.shape[2]
            self.build_model(input_dim)
        
        # Convert to tensors
        X_tensor = torch.from_numpy(X).float().to(self.device_obj)
        dataset = TensorDataset(X_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        criterion = nn.MSELoss()
        
        logger.info("starting_training", epochs=epochs, batch_size=batch_size, samples=len(X))
        
        for epoch in range(epochs):
            total_loss = 0
            for batch_X, in loader:
                self.optimizer.zero_grad()
                
                recon, _ = self.model(batch_X)
                loss = criterion(recon, batch_X)
                
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(loader)
            
            if (epoch + 1) % 10 == 0:
                logger.info("training_progress", epoch=epoch + 1, loss=avg_loss)
        
        logger.info("training_complete", final_loss=avg_loss)
        self.save_model()
    
    def encode(self, X: np.ndarray) -> np.ndarray:
        """
        Generate embeddings for data
        
        Args:
            X: (num_sequences, seq_len, num_features)
        
        Returns:
            embeddings: (num_sequences, embedding_dim)
        """
        if self.model is None:
            raise ValueError("Model not built or loaded")
        
        X_tensor = torch.from_numpy(X).float().to(self.device_obj)
        
        with torch.no_grad():
            embeddings = self.model.encode(X_tensor)
        
        return embeddings.cpu().numpy()
    
    def save_model(self, path: str = None):
        """Save model to disk"""
        if self.model is None:
            logger.warning("no_model_to_save")
            return
        
        path = path or str(self.model_dir / "macronet_latest.pt")
        torch.save(self.model.state_dict(), path)
        logger.info("model_saved", path=path)
    
    def load_model(self, path: str, input_dim: int):
        """Load model from disk"""
        self.build_model(input_dim)
        state_dict = torch.load(path, map_location=self.device_obj)
        self.model.load_state_dict(state_dict)
        logger.info("model_loaded", path=path)

if __name__ == "__main__":
    # Example usage
    macronet = MacroNet()
    
    # Generate dummy data (100 sequences, 1440 timesteps, 10 features)
    X_dummy = np.random.randn(100, 1440, 10).astype(np.float32)
    
    macronet.train(X_dummy, epochs=10)
    embeddings = macronet.encode(X_dummy)
    print(f"Embeddings shape: {embeddings.shape}")
