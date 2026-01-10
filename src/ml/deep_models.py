"""
Deep Neural Network Models for Premier League Match Prediction.

This module contains PyTorch-based deep learning models separate from
the traditional ML models in models.py.

Models:
    - MatchOutcomeNN: Multi-layer neural network for outcome prediction
    - ScorePredictorNN: Neural network for exact score prediction
    - DualHeadNN: Combined model for both outcome and score prediction
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not installed. Install with: pip install torch")

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, log_loss, classification_report

logger = logging.getLogger(__name__)


def check_torch():
    """Check if PyTorch is available."""
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for deep learning models. "
            "Install with: pip install torch"
        )


class MatchOutcomeNN(nn.Module):
    """
    Deep Neural Network for match outcome prediction.
    
    Architecture:
        - Input layer with batch normalization
        - Multiple hidden layers with ReLU activation
        - Dropout for regularization
        - Softmax output for 3-class classification
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [256, 128, 64, 32],
        dropout_rate: float = 0.3,
        use_batch_norm: bool = True,
    ):
        """
        Initialize the neural network.
        
        Args:
            input_dim: Number of input features
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout probability
            use_batch_norm: Whether to use batch normalization
        """
        check_torch()
        super(MatchOutcomeNN, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        
        layers = []
        prev_dim = input_dim
        
        # Input batch normalization
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(input_dim))
        
        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # Output layer (3 classes: Away Win, Draw, Home Win)
        layers.append(nn.Linear(prev_dim, 3))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self.network(x)
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get probability predictions."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=1)
        return probs


class ScorePredictorNN(nn.Module):
    """
    Neural Network for predicting match scores.
    
    Uses two output heads:
        - Home goals (regression)
        - Away goals (regression)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [256, 128, 64],
        dropout_rate: float = 0.2,
    ):
        """
        Initialize the score prediction network.
        
        Args:
            input_dim: Number of input features
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout probability
        """
        check_torch()
        super(ScorePredictorNN, self).__init__()
        
        self.input_dim = input_dim
        
        # Shared layers
        shared_layers = []
        prev_dim = input_dim
        
        shared_layers.append(nn.BatchNorm1d(input_dim))
        
        for hidden_dim in hidden_dims:
            shared_layers.append(nn.Linear(prev_dim, hidden_dim))
            shared_layers.append(nn.BatchNorm1d(hidden_dim))
            shared_layers.append(nn.ReLU())
            shared_layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        self.shared = nn.Sequential(*shared_layers)
        
        # Separate heads for home and away goals
        self.home_head = nn.Sequential(
            nn.Linear(prev_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.ReLU()  # Goals are non-negative
        )
        
        self.away_head = nn.Sequential(
            nn.Linear(prev_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.ReLU()
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning (home_goals, away_goals)."""
        shared_out = self.shared(x)
        home_goals = self.home_head(shared_out)
        away_goals = self.away_head(shared_out)
        return home_goals.squeeze(), away_goals.squeeze()


class DualHeadNN(nn.Module):
    """
    Combined Neural Network for both outcome and score prediction.
    
    Architecture:
        - Shared feature extraction layers
        - Outcome classification head (3 classes)
        - Score regression head (2 outputs)
    """
    
    def __init__(
        self,
        input_dim: int,
        shared_dims: List[int] = [256, 128],
        outcome_dims: List[int] = [64, 32],
        score_dims: List[int] = [64, 32],
        dropout_rate: float = 0.3,
    ):
        """
        Initialize the dual-head network.
        
        Args:
            input_dim: Number of input features
            shared_dims: Dimensions of shared layers
            outcome_dims: Dimensions of outcome-specific layers
            score_dims: Dimensions of score-specific layers
            dropout_rate: Dropout probability
        """
        check_torch()
        super(DualHeadNN, self).__init__()
        
        self.input_dim = input_dim
        
        # Shared feature extraction
        shared_layers = [nn.BatchNorm1d(input_dim)]
        prev_dim = input_dim
        
        for dim in shared_dims:
            shared_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = dim
        
        self.shared = nn.Sequential(*shared_layers)
        
        # Outcome classification head
        outcome_layers = []
        for dim in outcome_dims:
            outcome_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = dim
        outcome_layers.append(nn.Linear(prev_dim, 3))
        self.outcome_head = nn.Sequential(*outcome_layers)
        
        # Score regression head
        score_layers = []
        prev_dim = shared_dims[-1]
        for dim in score_dims:
            score_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate * 0.5)  # Less dropout for regression
            ])
            prev_dim = dim
        self.score_shared = nn.Sequential(*score_layers)
        
        self.home_goals = nn.Sequential(
            nn.Linear(prev_dim, 1),
            nn.ReLU()
        )
        self.away_goals = nn.Sequential(
            nn.Linear(prev_dim, 1),
            nn.ReLU()
        )
        
    def forward(
        self, 
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Returns:
            Tuple of (outcome_logits, home_goals, away_goals)
        """
        shared_out = self.shared(x)
        
        # Outcome prediction
        outcome_logits = self.outcome_head(shared_out)
        
        # Score prediction
        score_features = self.score_shared(shared_out)
        home = self.home_goals(score_features).squeeze()
        away = self.away_goals(score_features).squeeze()
        
        return outcome_logits, home, away


class DeepMatchPredictor:
    """
    Wrapper class for training and using deep learning models.
    
    Handles:
        - Data preprocessing (scaling)
        - Training with early stopping
        - Model persistence
        - Prediction interface
    """
    
    def __init__(
        self,
        model_type: str = "outcome",  # "outcome", "score", or "dual"
        hidden_dims: List[int] = [256, 128, 64, 32],
        dropout_rate: float = 0.3,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 200,
        early_stopping_patience: int = 20,
        device: Optional[str] = None,
    ):
        """
        Initialize the deep match predictor.
        
        Args:
            model_type: Type of model ("outcome", "score", or "dual")
            hidden_dims: Hidden layer dimensions
            dropout_rate: Dropout probability
            learning_rate: Adam learning rate
            batch_size: Training batch size
            epochs: Maximum training epochs
            early_stopping_patience: Epochs without improvement before stopping
            device: Device to use ("cuda", "mps", or "cpu")
        """
        check_torch()
        
        self.model_type = model_type
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        
        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_fitted = False
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
    def _create_model(self, input_dim: int) -> nn.Module:
        """Create the appropriate model based on model_type."""
        if self.model_type == "outcome":
            return MatchOutcomeNN(
                input_dim=input_dim,
                hidden_dims=self.hidden_dims,
                dropout_rate=self.dropout_rate
            )
        elif self.model_type == "score":
            return ScorePredictorNN(
                input_dim=input_dim,
                hidden_dims=self.hidden_dims,
                dropout_rate=self.dropout_rate
            )
        elif self.model_type == "dual":
            return DualHeadNN(
                input_dim=input_dim,
                shared_dims=self.hidden_dims[:2],
                outcome_dims=self.hidden_dims[2:],
                score_dims=self.hidden_dims[2:],
                dropout_rate=self.dropout_rate
            )
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")
    
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        home_goals_train: Optional[pd.Series] = None,
        away_goals_train: Optional[pd.Series] = None,
        home_goals_val: Optional[pd.Series] = None,
        away_goals_val: Optional[pd.Series] = None,
    ) -> "DeepMatchPredictor":
        """
        Train the deep learning model.
        
        Args:
            X_train: Training features
            y_train: Training labels (match outcomes)
            X_val: Validation features
            y_val: Validation labels
            home_goals_train: Home goals for score prediction
            away_goals_train: Away goals for score prediction
            home_goals_val: Validation home goals
            away_goals_val: Validation away goals
            
        Returns:
            self
        """
        self.feature_names = list(X_train.columns)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        if X_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
        else:
            # Split training data for validation
            val_size = int(0.15 * len(X_train))
            X_val_scaled = X_train_scaled[-val_size:]
            X_train_scaled = X_train_scaled[:-val_size]
            y_val = y_train.iloc[-val_size:]
            y_train = y_train.iloc[:-val_size]
            
            if home_goals_train is not None:
                home_goals_val = home_goals_train.iloc[-val_size:]
                away_goals_val = away_goals_train.iloc[-val_size:]
                home_goals_train = home_goals_train.iloc[:-val_size]
                away_goals_train = away_goals_train.iloc[:-val_size]
        
        # Create tensors
        X_train_tensor = torch.FloatTensor(X_train_scaled)
        y_train_tensor = torch.LongTensor(y_train.values)
        X_val_tensor = torch.FloatTensor(X_val_scaled)
        y_val_tensor = torch.LongTensor(y_val.values)
        
        # Create model
        self.model = self._create_model(X_train_scaled.shape[1])
        self.model = self.model.to(self.device)
        
        # Loss and optimizer
        if self.model_type == "outcome":
            criterion = nn.CrossEntropyLoss()
        elif self.model_type == "score":
            criterion = nn.MSELoss()
        else:  # dual
            outcome_criterion = nn.CrossEntropyLoss()
            score_criterion = nn.MSELoss()
        
        optimizer = optim.Adam(
            self.model.parameters(), 
            lr=self.learning_rate,
            weight_decay=1e-4  # L2 regularization
        )
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )
        
        # Data loaders
        if self.model_type in ["score", "dual"] and home_goals_train is not None:
            train_dataset = TensorDataset(
                X_train_tensor, 
                y_train_tensor,
                torch.FloatTensor(home_goals_train.values),
                torch.FloatTensor(away_goals_train.values)
            )
            val_dataset = TensorDataset(
                X_val_tensor, 
                y_val_tensor,
                torch.FloatTensor(home_goals_val.values),
                torch.FloatTensor(away_goals_val.values)
            )
        else:
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False
        )
        
        # Training loop
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        
        logger.info(f"Training {self.model_type} model on {self.device}...")
        
        for epoch in range(self.epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for batch in train_loader:
                if self.model_type in ["score", "dual"] and len(batch) == 4:
                    X_batch, y_batch, home_batch, away_batch = batch
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    home_batch = home_batch.to(self.device)
                    away_batch = away_batch.to(self.device)
                else:
                    X_batch, y_batch = batch
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                
                optimizer.zero_grad()
                
                if self.model_type == "outcome":
                    outputs = self.model(X_batch)
                    loss = criterion(outputs, y_batch)
                elif self.model_type == "score":
                    home_pred, away_pred = self.model(X_batch)
                    loss = criterion(home_pred, home_batch) + criterion(away_pred, away_batch)
                else:  # dual
                    outcome_logits, home_pred, away_pred = self.model(X_batch)
                    outcome_loss = outcome_criterion(outcome_logits, y_batch)
                    score_loss = score_criterion(home_pred, home_batch) + \
                                 score_criterion(away_pred, away_batch)
                    loss = outcome_loss + 0.5 * score_loss  # Weight score loss lower
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    if self.model_type in ["score", "dual"] and len(batch) == 4:
                        X_batch, y_batch, home_batch, away_batch = batch
                        X_batch = X_batch.to(self.device)
                        y_batch = y_batch.to(self.device)
                        home_batch = home_batch.to(self.device)
                        away_batch = away_batch.to(self.device)
                    else:
                        X_batch, y_batch = batch
                        X_batch = X_batch.to(self.device)
                        y_batch = y_batch.to(self.device)
                    
                    if self.model_type == "outcome":
                        outputs = self.model(X_batch)
                        loss = criterion(outputs, y_batch)
                        _, predicted = torch.max(outputs, 1)
                        val_correct += (predicted == y_batch).sum().item()
                        val_total += y_batch.size(0)
                    elif self.model_type == "score":
                        home_pred, away_pred = self.model(X_batch)
                        loss = criterion(home_pred, home_batch) + criterion(away_pred, away_batch)
                    else:  # dual
                        outcome_logits, home_pred, away_pred = self.model(X_batch)
                        outcome_loss = outcome_criterion(outcome_logits, y_batch)
                        score_loss = score_criterion(home_pred, home_batch) + \
                                     score_criterion(away_pred, away_batch)
                        loss = outcome_loss + 0.5 * score_loss
                        _, predicted = torch.max(outcome_logits, 1)
                        val_correct += (predicted == y_batch).sum().item()
                        val_total += y_batch.size(0)
                    
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            val_accuracy = val_correct / val_total if val_total > 0 else 0
            
            # Update scheduler
            scheduler.step(val_loss)
            
            # Track history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_accuracy'].append(val_accuracy)
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Log progress
            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{self.epochs} - "
                    f"Train Loss: {train_loss:.4f}, "
                    f"Val Loss: {val_loss:.4f}, "
                    f"Val Acc: {val_accuracy:.4f}"
                )
            
            if patience_counter >= self.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        self.is_fitted = True
        logger.info(f"Training complete. Best Val Loss: {best_val_loss:.4f}")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict match outcomes."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            if self.model_type == "outcome":
                outputs = self.model(X_tensor)
                _, predictions = torch.max(outputs, 1)
            elif self.model_type == "dual":
                outcome_logits, _, _ = self.model(X_tensor)
                _, predictions = torch.max(outcome_logits, 1)
            else:
                raise ValueError("Score model doesn't support outcome prediction")
        
        return predictions.cpu().numpy()
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            if self.model_type == "outcome":
                outputs = self.model(X_tensor)
            elif self.model_type == "dual":
                outputs, _, _ = self.model(X_tensor)
            else:
                raise ValueError("Score model doesn't support probability prediction")
            
            probs = torch.softmax(outputs, dim=1)
        
        return probs.cpu().numpy()
    
    def predict_score(
        self, 
        X: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict home and away goals."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if self.model_type not in ["score", "dual"]:
            raise ValueError("Model doesn't support score prediction")
        
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            if self.model_type == "score":
                home_goals, away_goals = self.model(X_tensor)
            else:  # dual
                _, home_goals, away_goals = self.model(X_tensor)
        
        return home_goals.cpu().numpy(), away_goals.cpu().numpy()
    
    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        home_goals: Optional[pd.Series] = None,
        away_goals: Optional[pd.Series] = None,
    ) -> Dict[str, float]:
        """
        Evaluate model on test data.
        
        Returns:
            Dictionary with evaluation metrics
        """
        metrics = {}
        
        if self.model_type in ["outcome", "dual"]:
            y_pred = self.predict(X)
            y_proba = self.predict_proba(X)
            
            metrics['accuracy'] = accuracy_score(y, y_pred)
            metrics['log_loss'] = log_loss(y, y_proba)
            
            # Per-class metrics
            report = classification_report(y, y_pred, output_dict=True)
            for cls in ['0', '1', '2']:
                if cls in report:
                    metrics[f'class_{cls}_f1'] = report[cls]['f1-score']
        
        if self.model_type in ["score", "dual"] and home_goals is not None:
            home_pred, away_pred = self.predict_score(X)
            
            metrics['home_goals_mae'] = np.mean(np.abs(home_pred - home_goals))
            metrics['away_goals_mae'] = np.mean(np.abs(away_pred - away_goals))
            metrics['home_goals_rmse'] = np.sqrt(np.mean((home_pred - home_goals) ** 2))
            metrics['away_goals_rmse'] = np.sqrt(np.mean((away_pred - away_goals) ** 2))
            
            # Exact score accuracy
            home_rounded = np.round(home_pred).astype(int)
            away_rounded = np.round(away_pred).astype(int)
            exact_match = (home_rounded == home_goals) & (away_rounded == away_goals)
            metrics['exact_score_accuracy'] = np.mean(exact_match)
        
        return metrics
    
    def save(self, path: str) -> None:
        """Save the model to disk."""
        save_dict = {
            'model_state': self.model.state_dict() if self.model else None,
            'model_type': self.model_type,
            'hidden_dims': self.hidden_dims,
            'dropout_rate': self.dropout_rate,
            'input_dim': self.model.input_dim if self.model else None,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted,
            'training_history': self.training_history,
        }
        
        with open(path, 'wb') as f:
            pickle.dump(save_dict, f)
        
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str) -> "DeepMatchPredictor":
        """Load a model from disk."""
        with open(path, 'rb') as f:
            save_dict = pickle.load(f)
        
        self.model_type = save_dict['model_type']
        self.hidden_dims = save_dict['hidden_dims']
        self.dropout_rate = save_dict['dropout_rate']
        self.scaler = save_dict['scaler']
        self.feature_names = save_dict['feature_names']
        self.is_fitted = save_dict['is_fitted']
        self.training_history = save_dict['training_history']
        
        if save_dict['model_state'] is not None:
            self.model = self._create_model(save_dict['input_dim'])
            self.model.load_state_dict(save_dict['model_state'])
            self.model = self.model.to(self.device)
        
        logger.info(f"Model loaded from {path}")
        return self


class ResidualBlock(nn.Module):
    """Residual block with skip connection."""
    
    def __init__(self, dim: int, dropout_rate: float = 0.2):
        check_torch()
        super(ResidualBlock, self).__init__()
        
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )
        self.relu = nn.ReLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.block(x)
        out = out + residual
        out = self.relu(out)
        return out


class ResNetMatchPredictor(nn.Module):
    """
    ResNet-style architecture for match prediction.
    
    Uses residual connections to enable training of deeper networks.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_blocks: int = 4,
        dropout_rate: float = 0.2,
    ):
        check_torch()
        super(ResNetMatchPredictor, self).__init__()
        
        self.input_dim = input_dim
        
        # Input projection
        self.input_layer = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Residual blocks
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, dropout_rate) for _ in range(num_blocks)]
        )
        
        # Output head
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 3)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        x = self.residual_blocks(x)
        x = self.output(x)
        return x


def train_deep_model(
    model_type: str = "outcome",
    data_dir: str = "data/processed",
    save_path: str = "models/deep_model.pkl",
    epochs: int = 200,
    batch_size: int = 32,
    learning_rate: float = 0.001,
) -> Dict[str, float]:
    """
    Train a deep learning model on processed data.
    
    Args:
        model_type: "outcome", "score", or "dual"
        data_dir: Directory with processed data
        save_path: Path to save the trained model
        epochs: Training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        
    Returns:
        Dictionary with evaluation metrics
    """
    check_torch()
    import json
    
    data_path = Path(data_dir)
    
    # Load data
    train_df = pd.read_csv(data_path / "train.csv")
    val_df = pd.read_csv(data_path / "val.csv")
    test_df = pd.read_csv(data_path / "test.csv")
    
    # Load feature columns from feature_info.json (same as ML models)
    feature_info_path = data_path / "feature_info.json"
    if feature_info_path.exists():
        with open(feature_info_path, 'r') as f:
            feature_info = json.load(f)
        feature_cols = feature_info.get('feature_columns', [])
        # Filter to only columns that exist in data
        feature_cols = [c for c in feature_cols if c in train_df.columns]
    else:
        # Fallback: Use numeric columns excluding targets
        target_col = 'match_result'
        exclude_cols = [
            'match_result', 'home_goals', 'away_goals', 'season', 
            'home_team', 'away_team', 'date', 'day_of_week', 'kickoff_time',
            'score', 'venue', 'referee', 'match_report', 'gameweek',
            'home_xg', 'away_xg', 'total_goals', 'goal_difference'
        ]
        numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [
            c for c in numeric_cols 
            if c not in exclude_cols and c not in ['home_goals', 'away_goals']
        ]
    
    target_col = 'match_result'
    
    X_train = train_df[feature_cols].fillna(0)
    y_train = train_df[target_col]
    X_val = val_df[feature_cols].fillna(0)
    y_val = val_df[target_col]
    X_test = test_df[feature_cols].fillna(0)
    y_test = test_df[target_col]
    
    # Goals for score prediction
    home_goals_train = train_df.get('home_goals')
    away_goals_train = train_df.get('away_goals')
    home_goals_val = val_df.get('home_goals')
    away_goals_val = val_df.get('away_goals')
    home_goals_test = test_df.get('home_goals')
    away_goals_test = test_df.get('away_goals')
    
    print(f"\n{'='*60}")
    print(f"TRAINING DEEP NEURAL NETWORK ({model_type.upper()})")
    print(f"{'='*60}")
    print(f"Train samples: {len(X_train)}")
    print(f"Val samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Features: {len(feature_cols)}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print()
    
    # Create and train model
    predictor = DeepMatchPredictor(
        model_type=model_type,
        hidden_dims=[256, 128, 64, 32],
        dropout_rate=0.3,
        learning_rate=learning_rate,
        batch_size=batch_size,
        epochs=epochs,
        early_stopping_patience=20,
    )
    
    predictor.fit(
        X_train, y_train,
        X_val, y_val,
        home_goals_train, away_goals_train,
        home_goals_val, away_goals_val,
    )
    
    # Evaluate
    print(f"\n{'='*60}")
    print("EVALUATION RESULTS")
    print(f"{'='*60}")
    
    test_metrics = predictor.evaluate(
        X_test, y_test,
        home_goals_test, away_goals_test
    )
    
    for metric, value in test_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Save model
    predictor.save(save_path)
    print(f"\nModel saved to: {save_path}")
    
    return test_metrics


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train deep learning model for match prediction")
    parser.add_argument("--model-type", type=str, default="outcome",
                       choices=["outcome", "score", "dual"],
                       help="Type of model to train")
    parser.add_argument("--epochs", type=int, default=200,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Training batch size")
    parser.add_argument("--lr", type=float, default=0.001,
                       help="Learning rate")
    parser.add_argument("--data-dir", type=str, default="data/processed",
                       help="Directory with processed data")
    parser.add_argument("--save-path", type=str, default=None,
                       help="Path to save model")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    save_path = args.save_path or f"models/deep_{args.model_type}_model.pkl"
    
    train_deep_model(
        model_type=args.model_type,
        data_dir=args.data_dir,
        save_path=save_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )
