# type: ignore


from typing import Tuple, List, Optional, Dict, Any
import tensorflow as tf
import pandas as pd
import numpy as np
import logging
import os
from datetime import datetime

from tensorflow import keras
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)

# Check for required packages
REQUIRED_PACKAGES = {
    'tensorflow': False,
    'sklearn': False
}

try:
    import tensorflow as tf
    REQUIRED_PACKAGES['tensorflow'] = True
except ImportError:
    logger.warning("TensorFlow not found. Install with: pip install tensorflow")

try:
    from sklearn.preprocessing import MinMaxScaler
    REQUIRED_PACKAGES['sklearn'] = True
except ImportError:
    logger.warning("scikit-learn not found. Install with: pip install scikit-learn")

class PricePredictor:
    def __init__(
        self, 
        lookback: int = 60, 
        forecast_horizon: int = 24,
        model_path: Optional[str] = None
    ):
        # Validate dependencies before proceeding
        missing_packages = [pkg for pkg, installed in REQUIRED_PACKAGES.items() if not installed]
        if missing_packages:
            raise ImportError(
                f"Required packages missing: {', '.join(missing_packages)}. "
                "Please install required packages before using PricePredictor."
            )

        self.lookback = lookback
        self.forecast_horizon = forecast_horizon
        self.scaler = MinMaxScaler()
        self.features = ['price', 'volume', 'rsi', 'ma_fast', 'ma_slow']

        self.Model = Model
        self.Sequential = Sequential
        self.LSTM = LSTM
        self.Dense = Dense
        self.Dropout = Dropout
        self.EarlyStopping = EarlyStopping
        self.load_model = load_model

        self.model = self._load_model(model_path) if model_path else self._build_model()

    def _build_model(self) -> 'Model':
        """Build LSTM model architecture"""
        model = self.Sequential([
            self.LSTM(
                50, 
                return_sequences=True, 
                input_shape=(self.lookback, len(self.features))
            ),
            self.Dropout(0.2),
            self.LSTM(50),
            self.Dense(self.forecast_horizon)
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        return model

    def _load_model(self, path: Optional[str]) -> 'Model':
        """Load model from file"""
        if path is None:
            return self._build_model()

        try:
            if not os.path.exists(path):
                logger.warning(f"Model path {path} not found, building new model")
                return self._build_model()
                
            model = self.load_model(path)
            if not isinstance(model, self.Model):
                raise ValueError("Loaded object is not a Keras Model")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model from {path}: {e}")
            return self._build_model()

    def save_model(self, path: str) -> None:
        """Save model to file"""
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self.model.save(path)
            logger.info(f"Model saved successfully to {path}")
        except Exception as e:
            logger.error(f"Failed to save model to {path}: {e}")
            raise

    def _validate_data(self, df: pd.DataFrame) -> None:
        """Validate input data"""
        missing_features = [f for f in self.features if f not in df.columns]
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")

        if df.isnull().any().any():
            raise ValueError("Data contains null values")

        if len(df) < self.lookback + self.forecast_horizon:
            raise ValueError(
                f"Insufficient data points. Need at least {self.lookback + self.forecast_horizon}, "
                f"got {len(df)}"
            )

    def _prepare_data(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare and scale data"""
        self._validate_data(df)
        return self.scaler.fit_transform(df[self.features])

    def _create_sequences(
        self, 
        data: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for training"""
        X, y = [], []
        for i in range(len(data) - self.lookback - self.forecast_horizon + 1):
            X.append(data[i:(i + self.lookback)])
            y.append(data[i + self.lookback:i + self.lookback + self.forecast_horizon, 0])
        return np.array(X), np.array(y)

    def _create_prediction_sequence(self, data: np.ndarray) -> np.ndarray:
        """Create sequence for prediction"""
        if len(data) < self.lookback:
            raise ValueError(f"Not enough data points. Need at least {self.lookback}")
        return np.array([data[-self.lookback:]])

    async def train(
        self, 
        historical_data: pd.DataFrame, 
        epochs: int = 50, 
        batch_size: int = 32,
        validation_split: float = 0.1
    ) -> Dict[str, List[float]]:
        """Train the model with historical data"""
        try:
            processed_data = self._prepare_data(historical_data)
            X_train, y_train = self._create_sequences(processed_data)

            early_stopping = self.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            )

            history = self.model.fit(
                X_train, 
                y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                callbacks=[early_stopping]
            )

            training_metrics = history.history
            final_loss = training_metrics.get('loss', [0.0])[-1]
            logger.info(f"Training completed. Final loss: {final_loss:.4f}")
            return training_metrics

        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise

    async def predict(self, recent_data: pd.DataFrame) -> np.ndarray:
        """Predict future prices"""
        try:
            processed = self._prepare_data(recent_data)
            sequence = self._create_prediction_sequence(processed)
            prediction = self.model.predict(sequence)
            
            padded_prediction = np.concatenate([
                prediction, 
                np.zeros((prediction.shape[0], len(self.features)-1))
            ], axis=1)
            
            return self.scaler.inverse_transform(padded_prediction)[:, 0]

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise

    def evaluate(self, test_data: pd.DataFrame) -> Dict[str, float]:
        """Evaluate model performance"""
        try:
            processed_data = self._prepare_data(test_data)
            X_test, y_test = self._create_sequences(processed_data)

            metrics = self.model.evaluate(X_test, y_test, return_dict=True)
            return metrics

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise