"""Machine Learning Integration System"""

import logging
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import joblib

logger = logging.getLogger(__name__)

@dataclass
class MLConfig:
    """ML system configuration"""
    sequence_length: int = 60  # Length of input sequences
    prediction_length: int = 12  # Number of periods to predict
    training_epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    model_path: str = "models/"
    retrain_interval: int = 24  # Hours between retraining

@dataclass
class ModelMetrics:
    """Model performance metrics"""
    mse: float
    mae: float
    accuracy: float
    last_training: datetime
    predictions: List[float]
    confidence: float

class MLSystem:
    """Advanced ML system for trading"""
    
    def __init__(self, config: MLConfig):
        self.config = config
        self.price_scaler = MinMaxScaler()
        self.volume_scaler = MinMaxScaler()
        self.models: Dict[str, Any] = {}
        self.metrics: Dict[str, ModelMetrics] = {}
        self.last_training: Dict[str, datetime] = {}
        
        # Initialize TensorFlow settings
        tf.keras.backend.set_floatx('float64')

    def create_price_model(self) -> Sequential:
        """Create price prediction model"""
        model = Sequential([
            LSTM(100, return_sequences=True, 
                 input_shape=(self.config.sequence_length, 5)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(self.config.prediction_length)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=self.config.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        return model

    def create_pattern_model(self) -> Sequential:
        """Create pattern recognition model"""
        model = Sequential([
            LSTM(64, return_sequences=True, 
                 input_shape=(self.config.sequence_length, 5)),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(3, activation='softmax')  # Bullish, Bearish, Neutral
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=self.config.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model

    async def train_models(
        self,
        token_address: str,
        price_history: List[float],
        volume_history: List[float],
        additional_features: Optional[List[List[float]]] = None
    ) -> bool:
        """Train ML models for a token"""
        try:
            if len(price_history) < self.config.sequence_length + self.config.prediction_length:
                logger.warning(f"Insufficient data for {token_address}")
                return False

            # Prepare training data
            X, y = self._prepare_training_data(
                price_history,
                volume_history,
                additional_features
            )

            # Train price prediction model
            price_model = self.create_price_model()
            price_history = price_model.fit(
                X, y,
                epochs=self.config.training_epochs,
                batch_size=self.config.batch_size,
                validation_split=0.2,
                verbose=0
            )

            # Train pattern recognition model
            pattern_model = self.create_pattern_model()
            pattern_labels = self._generate_pattern_labels(price_history)
            pattern_history = pattern_model.fit(
                X, pattern_labels,
                epochs=self.config.training_epochs,
                batch_size=self.config.batch_size,
                validation_split=0.2,
                verbose=0
            )

            # Save models
            self.models[token_address] = {
                'price': price_model,
                'pattern': pattern_model
            }

            # Update metrics
            self.metrics[token_address] = ModelMetrics(
                mse=price_history.history['loss'][-1],
                mae=price_history.history['mae'][-1],
                accuracy=pattern_history.history['accuracy'][-1],
                last_training=datetime.now(),
                predictions=[],
                confidence=self._calculate_confidence(price_history, pattern_history)
            )

            # Save models to disk
            self._save_models(token_address)
            
            return True

        except Exception as e:
            logger.error(f"Error training models for {token_address}: {str(e)}")
            return False

    async def predict(
        self,
        token_address: str,
        current_data: List[float],
        additional_features: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """Generate predictions for a token"""
        try:
            if token_address not in self.models:
                logger.warning(f"No models found for {token_address}")
                return {}

            # Check if retraining is needed
            if self._needs_retraining(token_address):
                logger.info(f"Models for {token_address} need retraining")
                return {}

            # Prepare input data
            X = self._prepare_prediction_data(
                current_data,
                additional_features
            )

            # Generate predictions
            price_predictions = self.models[token_address]['price'].predict(X)
            pattern_probs = self.models[token_address]['pattern'].predict(X)

            # Process predictions
            processed_predictions = self._process_predictions(
                price_predictions[0],
                pattern_probs[0]
            )

            # Update metrics
            self.metrics[token_address].predictions = processed_predictions['prices']
            
            return processed_predictions

        except Exception as e:
            logger.error(f"Error generating predictions for {token_address}: {str(e)}")
            return {}

    def _prepare_training_data(
        self,
        price_history: List[float],
        volume_history: List[float],
        additional_features: Optional[List[List[float]]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for model training"""
        try:
            # Scale the data
            scaled_prices = self.price_scaler.fit_transform(
                np.array(price_history).reshape(-1, 1)
            )
            scaled_volumes = self.volume_scaler.fit_transform(
                np.array(volume_history).reshape(-1, 1)
            )

            # Combine features
            features = [scaled_prices, scaled_volumes]
            if additional_features:
                features.extend([
                    np.array(feat).reshape(-1, 1)
                    for feat in additional_features
                ])

            # Create sequences
            X, y = [], []
            for i in range(len(scaled_prices) - self.config.sequence_length - self.config.prediction_length + 1):
                # Input sequence
                seq = np.column_stack([
                    feat[i:(i + self.config.sequence_length)]
                    for feat in features
                ])
                X.append(seq)
                
                # Target sequence (future prices)
                target = scaled_prices[
                    (i + self.config.sequence_length):
                    (i + self.config.sequence_length + self.config.prediction_length)
                ]
                y.append(target)

            return np.array(X), np.array(y)

        except Exception as e:
            logger.error(f"Error preparing training data: {str(e)}")
            raise

    def _prepare_prediction_data(
        self,
        current_data: List[float],
        additional_features: Optional[List[float]] = None
    ) -> np.ndarray:
        """Prepare data for prediction"""
        try:
            # Scale current data
            scaled_data = [
                self.price_scaler.transform(
                    np.array(current_data).reshape(-1, 1)
                )
            ]

            if additional_features:
                scaled_data.extend([
                    np.array(feat).reshape(-1, 1)
                    for feat in additional_features
                ])

            # Create sequence
            X = np.column_stack(scaled_data)
            return np.expand_dims(X[-self.config.sequence_length:], 0)

        except Exception as e:
            logger.error(f"Error preparing prediction data: {str(e)}")
            raise

    def _process_predictions(
        self,
        price_preds: np.ndarray,
        pattern_probs: np.ndarray
    ) -> Dict[str, Any]:
        """Process model predictions"""
        try:
            # Inverse transform price predictions
            predicted_prices = self.price_scaler.inverse_transform(
                price_preds.reshape(-1, 1)
            ).flatten()

            # Process pattern probabilities
            pattern_map = ['bearish', 'neutral', 'bullish']
            pattern = pattern_map[np.argmax(pattern_probs)]
            confidence = float(np.max(pattern_probs))

            return {
                'prices': predicted_prices.tolist(),
                'pattern': pattern,
                'confidence': confidence,
                'probabilities': {
                    pattern_map[i]: float(prob)
                    for i, prob in enumerate(pattern_probs)
                }
            }

        except Exception as e:
            logger.error(f"Error processing predictions: {str(e)}")
            return {}

    def _generate_pattern_labels(
        self,
        price_history: List[float]
    ) -> np.ndarray:
        """Generate pattern recognition labels"""
        try:
            returns = np.diff(price_history) / price_history[:-1]
            labels = []

            for i in range(len(returns) - self.config.sequence_length + 1):
                seq_returns = returns[i:i + self.config.sequence_length]
                trend = np.mean(seq_returns)
                volatility = np.std(seq_returns)

                if trend > volatility:
                    labels.append([0, 0, 1])  # Bullish
                elif trend < -volatility:
                    labels.append([1, 0, 0])  # Bearish
                else:
                    labels.append([0, 1, 0])  # Neutral

            return np.array(labels)

        except Exception as e:
            logger.error(f"Error generating pattern labels: {str(e)}")
            raise

    def _calculate_confidence(
        self,
        price_history: Any,
        pattern_history: Any
    ) -> float:
        """Calculate model confidence score"""
        try:
            # Combine different metrics
            price_loss = price_history.history['val_loss'][-1]
            pattern_accuracy = pattern_history.history['val_accuracy'][-1]

            # Normalize price loss (lower is better)
            norm_loss = 1 / (1 + price_loss)

            # Combine metrics (weighted average)
            confidence = (0.6 * norm_loss + 0.4 * pattern_accuracy)
            return float(confidence)

        except Exception as e:
            logger.error(f"Error calculating confidence: {str(e)}")
            return 0.0

    def _needs_retraining(self, token_address: str) -> bool:
        """Check if models need retraining"""
        if token_address not in self.last_training:
            return True

        hours_since_training = (
            datetime.now() - self.last_training[token_address]
        ).total_seconds() / 3600

        return hours_since_training >= self.config.retrain_interval

    def _save_models(self, token_address: str) -> None:
        """Save models to disk"""
        try:
            # Create model directory if it doesn't exist
            import os
            os.makedirs(self.config.model_path, exist_ok=True)

            # Save models
            base_path = os.path.join(self.config.model_path, token_address)
            self.models[token_address]['price'].save(f"{base_path}_price.h5")
            self.models[token_address]['pattern'].save(f"{base_path}_pattern.h5")

            # Save scalers
            joblib.dump(self.price_scaler, f"{base_path}_price_scaler.pkl")
            joblib.dump(self.volume_scaler, f"{base_path}_volume_scaler.pkl")

        except Exception as e:
            logger.error(f"Error saving models: {str(e)}")

    def load_models(self, token_address: str) -> bool:
        """Load models from disk"""
        try:
            base_path = os.path.join(self.config.model_path, token_address)

            # Load models
            self.models[token_address] = {
                'price': tf.keras.models.load_model(f"{base_path}_price.h5"),
                'pattern': tf.keras.models.load_model(f"{base_path}_pattern.h5")
            }

            # Load scalers
            self.price_scaler = joblib.load(f"{base_path}_price_scaler.pkl")
            self.volume_scaler = joblib.load(f"{base_path}_volume_scaler.pkl")

            return True

        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            return False