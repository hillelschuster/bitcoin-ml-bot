"""LSTM-based machine learning model for the Bitcoin ML Trading Bot.

This module provides the primary prediction model used by the trading bot.
It includes functions for loading models and generating trading signals,
as well as a PredictionModel class that handles model training and prediction.

The implementation uses a Keras LSTM neural network for sequence-based prediction
and supports hot reloading of models when they are updated on disk.

Key components:
- load_model(): Loads model and scaler from disk (used by monitoring and backtest)
- predict_signal(): Generates trading signals from market data (used by backtest)
- PredictionModel: Main class for model management, training, and prediction

Integration points:
- model_retrain_loop.py: Provides hot reloading functionality
- model_monitor.py: Monitors model performance and triggers retraining
- model_evaluator.py: Evaluates and compares model performance
"""
import os
import logging
import joblib
import threading
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from src.models.model_retrain_loop import ModelFileWatcher  # Hot reloader deprecated — this is final class name for migration


def load_model_metadata(path="model_artifacts/model_metadata.json"):
    """Load model metadata from JSON file.

    Args:
        path (str): Path to the metadata JSON file

    Returns:
        dict: Model metadata, or default metadata if file not found
    """
    try:
        with open(path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        # Create default metadata file if it doesn't exist
        default_metadata = {
            "trained": False,
            "version": "untrained",
            "training_samples": 0
        }

        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Write default metadata file
        with open(path, "w") as f:
            json.dump(default_metadata, f, indent=2)

        return default_metadata

# Configure logging
logger = logging.getLogger('model')

# Default sequence length for LSTM
DEFAULT_SEQUENCE_LENGTH = 30

# Number of input features for multivariate LSTM model
num_features = 7

def prepare_sequences(df, sequence_length=DEFAULT_SEQUENCE_LENGTH):
    """Prepare sequence data for LSTM model from a DataFrame.

    This function converts a DataFrame with features into sequence data suitable
    for LSTM model training or prediction. It handles different column formats
    and extracts the target variable.

    Args:
        df (pandas.DataFrame): DataFrame with features and target
        sequence_length (int): Length of sequences to create

    Returns:
        tuple: (X, y) where X is a numpy array of shape [samples, time_steps, features]
               and y is a numpy array of target values
    """
    try:
        # Remove timestamp column if present
        if 'timestamp' in df.columns:
            df = df.drop(columns=['timestamp'])

        # Extract target variable
        if 'close' in df.columns:
            # Use price movement as target
            target = (df['close'].shift(-1) > df['close']).astype(int)
            df = df.drop(columns=['close'])
        elif 'target' in df.columns:
            # Use explicit target column
            target = df['target']
            df = df.drop(columns=['target'])
        else:
            logger.error("No target or close column found in DataFrame.")
            return None, None

        # Create sequences
        X, y = [], []
        for i in range(len(df) - sequence_length):
            X.append(df.iloc[i:i+sequence_length].values)
            y.append(target.iloc[i+sequence_length])

        return np.array(X), np.array(y)
    except Exception as e:
        logger.error(f"Error preparing sequences: {e}")
        return None, None

def load_model(model_path="model_artifacts/lstm_model.h5", scaler_path="model_artifacts/lstm_scaler.pkl"):
    """Load the trained machine learning model and its scaler from disk.

    This function is used by the backtest and monitoring modules to load
    the model without instantiating the full PredictionModel class.

    It first tries to use the hot reloader if available, then falls back to direct loading.

    Args:
        model_path (str): Path to the model file
        scaler_path (str): Path to the scaler file

    Returns:
        tuple: (model, scaler) where model is the trained Keras LSTM model and
               scaler is the StandardScaler used to normalize features
    """
    try:
        # Try to get model from model watcher first
        try:
            from src.models.model_retrain_loop import get_model, get_scaler
            from src.monitoring.model_monitor import initialize_model_watcher

            # Initialize model watcher if not already initialized
            initialize_model_watcher(model_path, scaler_path)

            # Get model and scaler from file watcher
            model = get_model()
            scaler = get_scaler()

            if model is not None and scaler is not None:
                logger.debug("Model and scaler loaded from model file watcher")
                return model, scaler
            else:
                logger.warning("Model file watcher returned None for model or scaler, falling back to direct loading")
        except Exception as e:
            logger.warning(f"Model file watcher not available, falling back to direct loading: {e}")

        # Fall back to direct loading
        model = keras.models.load_model(model_path)
        scaler = joblib.load(scaler_path)
        logger.info(f"Model loaded from {model_path} and scaler from {scaler_path}")
        return model, scaler
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None, None

def predict_signal(df, model, scaler):
    """Generate a trading signal prediction using the trained model.

    This function is used by the backtest module to generate trading signals
    without instantiating the full PredictionModel class.

    Args:
        df (pandas.DataFrame): Market data with features
        model: Trained Keras LSTM model
        scaler: Feature scaler for normalization

    Returns:
        int: Predicted signal (1 for buy, -1 for sell, 0 for hold)

    Note:
        This function expects an LSTM model that outputs a single probability value.
        The prediction is converted to a signal as follows:
        - probability > 0.5: UP signal (1)
        - probability <= 0.5: DOWN signal (-1)
        - Error or missing model: HOLD signal (0)
    """
    try:
        if model is None or scaler is None:
            logger.error("Model or scaler is None")
            return 0  # Return HOLD signal if model or scaler is missing

        # Prepare sequence data for LSTM (last 30 time steps)
        sequence_length = 30
        if len(df) < sequence_length:
            logger.warning(f"Not enough data points for LSTM. Need {sequence_length}, got {len(df)}")
            return 0

        # Extract features, dropping non-feature columns
        features = df.drop(columns=["timestamp"] if "timestamp" in df.columns else [])
        if "close" in features.columns:
            features = features.drop(columns=["close"])

        # Get the last sequence_length rows
        sequence_data = features.iloc[-sequence_length:].values

        # Scale the features
        scaled_data = scaler.transform(sequence_data)

        # Reshape for LSTM input: [samples, time steps, features]
        X = np.array([scaled_data])

        # Make prediction with LSTM model
        prediction = model.predict(X, verbose=0)

        # Convert to signal: 1 (UP), -1 (DOWN), 0 (HOLD)
        if prediction[0][0] > 0.5:  # Probability of UP
            return 1  # UP signal
        else:
            return -1  # DOWN signal
    except Exception as e:
        logger.error(f"Error in predict_signal: {e}")
        return 0  # Return HOLD signal on error

class PredictionModel:
    """Machine learning model for price prediction.

    This is the main prediction model used by the trading bot. It handles
    loading, training, and making predictions with the Keras LSTM model.

    Features (inputs):
    - Price data: open, high, low, close
    - Volume data
    - Technical indicators: RSI, MACD, Bollinger Bands, etc.
    - Derived features: price momentum, volatility, etc.
    - Sequence data: Last N time steps for temporal pattern recognition

    Target (output):
    - Classification: UP or DOWN prediction for next period
    - Confidence: Probability of the prediction being correct

    Model type:
    - LSTM: Long Short-Term Memory neural network
    - Specialized for sequence/time series data
    - Captures temporal patterns and dependencies
    """

    def __init__(self, config):
        """Initialize the prediction model.

        Args:
            config: Configuration instance containing model settings
        """
        self.config = config
        model_config = config.get('model', {})

        # Get model configuration parameters
        self.prediction_horizon = model_config.get('prediction_horizon', 24)
        self.conf_threshold = model_config.get('confidence_threshold', 0.65)

        # Create model_artifacts directory if it doesn't exist
        os.makedirs('model_artifacts', exist_ok=True)

        # Set model file paths
        model_path = model_config.get('path', 'model_artifacts/lstm_model.h5')
        self.model_path = model_path
        self.scaler_path = os.path.join(os.path.dirname(model_path), 'lstm_scaler.pkl')

        # Initialize model and scaler
        self.model = None
        self.scaler = StandardScaler()

        # Load model metadata
        self.metadata = load_model_metadata()
        self.model_trained = self.metadata.get("trained", False)
        logger.info(f"Loaded model metadata: {self.metadata}")

        # Configure hot reloading
        self.hot_reload_enabled = model_config.get('hot_reload', {}).get('enabled', True)
        self.reloader = None

        if self.hot_reload_enabled:
            try:
                # Try to use model watcher first
                try:
                    from src.monitoring.model_monitor import initialize_model_watcher
                    initialize_model_watcher(self.model_path, self.scaler_path)
                    logger.info(f"Using model watcher for {self.model_path}")
                except Exception as e:
                    logger.warning(f"Model watcher initialization failed: {e}")

                # Initialize instance-level file watcher as backup
                self.reloader = ModelFileWatcher(self.model_path, self.scaler_path)

                # Start watching for model changes in a separate thread
                self.watch_thread = threading.Thread(target=self.reloader.watch, daemon=True)
                self.watch_thread.start()

                logger.info(f"Model file watching enabled for {self.model_path}")
            except Exception as e:
                logger.error(f"Failed to initialize model file watcher: {e}")
                self.hot_reload_enabled = False
                # Fall back to direct loading
                self.load_model()
        else:
            # Load model if available (traditional way)
            self.load_model()

        logger.info("Initialized Keras LSTM prediction model")

    def load_model(self):
        """Load model from file."""
        try:
            # Check if model file exists
            if not os.path.exists(self.model_path):
                logger.info("No saved model found")
                self._build_model()  # Create a new model
                return False

            try:
                # Load the model
                self.model = keras.models.load_model(self.model_path)
                logger.info(f"Model loaded from {self.model_path}")
            except Exception as e:
                logger.warning(f"Error loading model: {e}. Creating a new model.")
                self._build_model()  # Create a new model

            # Load scaler if available
            try:
                if os.path.exists(self.scaler_path):
                    self.scaler = joblib.load(self.scaler_path)
                    logger.info(f"Scaler loaded from {self.scaler_path}")
            except Exception as e:
                logger.warning(f"Error loading scaler: {e}. Creating a new scaler.")
                self.scaler = StandardScaler()

            return True

        except Exception as e:
            logger.error(f"Error in load_model: {e}")
            self._build_model()  # Create a new model as fallback
            self.scaler = StandardScaler()
            return False

    def save_model(self):
        """Save model and scaler to file.

        This method saves the model and scaler to disk, ensuring they can be
        loaded by the file watcher and other components.

        Returns:
            bool: True if save was successful, False otherwise
        """
        try:
            if self.model is None:
                logger.error("Cannot save model: model is None")
                return False

            # Create models directory if it doesn't exist
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

            # Save model
            self.model.save(self.model_path)
            logger.info(f"Model saved to {self.model_path}")

            # Save scaler
            joblib.dump(self.scaler, self.scaler_path)
            logger.info(f"Scaler saved to {self.scaler_path}")

            # Notify model_health logger
            try:
                model_health_logger = logging.getLogger('model_health')
                model_health_logger.info(f"Model saved to {self.model_path}")
            except Exception:
                pass  # Ignore errors in logging

            return True
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False

    def train(self, X, y):
        """
        Train the LSTM model.

        Args:
            X: Input features (DataFrame or numpy array)
                If DataFrame: Will be converted to sequence data
                If numpy array: Should be in shape [samples, time steps, features]
            y: Target values (1 for UP, 0 for DOWN)

        Returns:
            dict: Training metrics including accuracy and loss, or None if training fails

        Note:
            This method is compatible with both direct calls and calls from model_monitor.py.
            It handles both DataFrame and numpy array inputs, and properly saves the model
            for file watching.
        """
        if X is None or y is None:
            logger.warning("No data for training")
            return

        try:
            logger.info(f"Training Keras LSTM model")

            # Prepare features
            if hasattr(X, 'drop') and callable(X.drop):
                # If X is a DataFrame, drop timestamp and close columns if they exist
                columns_to_drop = []
                if "timestamp" in X.columns:
                    columns_to_drop.append("timestamp")
                if "close" in X.columns:
                    columns_to_drop.append("close")
                X_prepared = X.drop(columns=columns_to_drop) if columns_to_drop else X.copy()

                # Convert DataFrame to numpy array
                X_prepared = X_prepared.values
            else:
                # If X is a numpy array, use it as is
                X_prepared = X

            # Prepare sequence data for LSTM
            if len(X_prepared.shape) == 2:
                # Data is 2D [samples, features], need to reshape for sequence
                logger.warning("Input data is 2D, but LSTM requires 3D sequence data. Using sliding window approach.")

                # Create sequences using sliding window
                sequence_length = 30  # Use last 30 time steps
                X_sequences = []
                y_sequences = []

                for i in range(len(X_prepared) - sequence_length):
                    X_sequences.append(X_prepared[i:i+sequence_length])
                    y_sequences.append(y[i+sequence_length])

                X_prepared = np.array(X_sequences)
                y_prepared = np.array(y_sequences)
            else:
                # Data is already 3D [samples, time steps, features]
                y_prepared = y

            # Fit scaler on the flattened sequence data
            n_samples, seq_len, n_features = X_prepared.shape
            X_flat = X_prepared.reshape(n_samples * seq_len, n_features)
            self.scaler.fit(X_flat)

            # Scale each sequence
            X_scaled = np.zeros_like(X_prepared)
            for i in range(n_samples):
                X_scaled[i] = self.scaler.transform(X_prepared[i])

            # Convert y to binary classification target if needed
            if hasattr(y_prepared, 'values'):
                y_binary = np.array((y_prepared.values > 0), dtype=int)
            else:
                y_binary = np.array((y_prepared > 0), dtype=int)

            # Build LSTM model
            input_shape = (X_scaled.shape[1], X_scaled.shape[2])  # (time steps, features)
            self.model = self._build_lstm_model(input_shape)

            # Train model
            history = self.model.fit(
                X_scaled, y_binary,
                epochs=50,
                batch_size=32,
                validation_split=0.2,
                callbacks=[
                    keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                    keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
                ],
                verbose=1
            )

            # Get final metrics
            final_loss = history.history['loss'][-1] if 'loss' in history.history else None
            final_acc = history.history['accuracy'][-1] if 'accuracy' in history.history else None
            val_loss = history.history['val_loss'][-1] if 'val_loss' in history.history else None
            val_acc = history.history['val_accuracy'][-1] if 'val_accuracy' in history.history else None

            logger.info(f"LSTM model trained successfully: loss={final_loss:.4f}, acc={final_acc:.4f}")

            # Save model
            self.save_model()

            # Log training metrics to model_health.log
            try:
                model_health_logger = logging.getLogger('model_health')
                model_health_logger.info(f"Model trained: loss={final_loss:.4f}, acc={final_acc:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
            except Exception as e:
                logger.warning(f"Could not log to model_health.log: {e}")

            # Return metrics
            return {
                'loss': final_loss,
                'accuracy': final_acc,
                'val_loss': val_loss,
                'val_accuracy': val_acc
            }

        except Exception as e:
            logger.error(f"Error training LSTM model: {e}")
            # Log training failure to model_health.log
            try:
                model_health_logger = logging.getLogger('model_health')
                model_health_logger.error(f"Model training failed: {e}")
            except Exception:
                pass  # Ignore errors in logging
            return None

    def _build_model(self):
        """Build a default model if needed.

        Creates a simple LSTM model with default architecture.
        """
        logger.warning("Building a default LSTM model. This should be trained before use.")
        # Default input shape: 30 time steps, num_features features
        input_shape = (30, num_features)
        self.model = self._build_lstm_model(input_shape)

    def _build_lstm_model(self, input_shape):
        """Build an LSTM model with the given input shape.

        Args:
            input_shape: Tuple of (time_steps, features)

        Returns:
            A compiled Keras LSTM model
        """
        model = keras.Sequential([
            keras.layers.LSTM(64, input_shape=input_shape, return_sequences=True),
            keras.layers.Dropout(0.2),
            keras.layers.LSTM(32),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')
        ])

        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        logger.info(f"Built LSTM model with input shape {input_shape}")
        return model

    def predict(self, df):
        """Make a prediction using the LSTM model.

        Args:
            df (pandas.DataFrame): Market data with features

        Returns:
            tuple: (prediction, confidence) where prediction is 'UP', 'DOWN', or 'HOLD'
                  and confidence is a float between 0 and 1
        """
        # Check if model is untrained or missing
        if not self.model_trained or self.model is None:
            logger.warning("⚠️ Prediction skipped: LSTM model is untrained or missing.")
            return "HOLD", 0.0

        # Validate input data - LSTM needs enough data for a sequence
        sequence_length = 30  # Same as used in training
        if df is None or df.empty or len(df) < sequence_length:
            logger.warning(f"Insufficient data for LSTM prediction. Need at least {sequence_length} data points.")
            return "HOLD", 0.0

        # Get the model - either from hot reloader or directly
        model = None
        scaler = None

        if self.hot_reload_enabled:
            try:
                # Try to use the instance reloader first
                if self.reloader:
                    model = self.reloader.get_model()
                    scaler = self.reloader.get_scaler()

                # If instance reloader failed, try the module-level reloader
                if model is None or scaler is None:
                    try:
                        from src.models.model_retrain_loop import get_model, get_scaler
                        model = get_model()
                        scaler = get_scaler()
                        if model is not None:
                            logger.debug("Using model from module-level file watcher")
                    except Exception as e:
                        logger.debug(f"Module-level file watcher not available: {e}")
            except Exception as e:
                logger.warning(f"File watcher error: {e}, falling back to direct model")

        # Fall back to instance variables if file watching failed
        if model is None:
            model = self.model
            scaler = self.scaler
            logger.debug("Using directly loaded model (file watching failed or disabled)")

        # Check if model is available
        if model is None:
            logger.error("No model loaded. Please train or load a model first.")
            return "HOLD", 0.0

        try:
            # Prepare features by dropping non-feature columns
            columns_to_drop = []
            if "timestamp" in df.columns:
                columns_to_drop.append("timestamp")
            if "close" in df.columns:
                columns_to_drop.append("close")

            features = df.drop(columns=columns_to_drop) if columns_to_drop else df.copy()

            # Get the last sequence_length rows for LSTM input
            sequence_data = features.iloc[-sequence_length:].values

            # Check if we have a shape mismatch between data and scaler
            n_features = sequence_data.shape[1]
            if hasattr(scaler, 'n_features_in_') and scaler.n_features_in_ != n_features:
                logger.warning(f"Feature count mismatch: Data has {n_features} features, but scaler expects {scaler.n_features_in_}")

                # Log the feature names and first few values for debugging
                if hasattr(df, 'columns'):
                    logger.info(f"Available features: {list(df.columns)}")
                    logger.info(f"First row of features: {df.iloc[0].to_dict()}")

                # Log the sequence data shape and first few values
                logger.info(f"Sequence data shape: {sequence_data.shape}")
                logger.info(f"First row of sequence data: {sequence_data[0]}")

                # Extract just the first feature (usually price) as a fallback
                price_data = sequence_data[:, 0].reshape(-1, 1)
                logger.info(f"Using only the first feature (price) for prediction due to mismatch")
                logger.info(f"Price data shape: {price_data.shape}")

                # Scale the price data
                scaled_sequence = scaler.transform(price_data)
                logger.info(f"Scaled price data shape: {scaled_sequence.shape}")

                # Reshape for LSTM input: [samples, time steps, features] = [1, sequence_length, 1]
                X = scaled_sequence.reshape(1, sequence_length, 1)
                logger.info(f"Final input shape for model: {X.shape}")
            else:
                # Normal case - feature counts match
                if sequence_data.shape[1] == 1:
                    # Single feature case (just price)
                    price_data = sequence_data.reshape(-1, 1)
                    # Scale the price data
                    scaled_sequence = scaler.transform(price_data)
                    # Reshape for LSTM input: [samples, time steps, features] = [1, sequence_length, 1]
                    X = scaled_sequence.reshape(1, sequence_length, 1)
                else:
                    # Multi-feature case
                    # Reshape to 2D for scaling
                    original_shape = sequence_data.shape
                    flattened = sequence_data.reshape(-1, original_shape[-1])
                    # Scale all features at once
                    scaled_flat = scaler.transform(flattened)
                    # Reshape back to original shape
                    scaled_sequence = scaled_flat.reshape(original_shape)
                    # Reshape for LSTM input: [samples, time steps, features]
                    X = np.array([scaled_sequence])

            # Make prediction with LSTM model
            # Model outputs probability for class 1 (UP)
            prediction_result = model.predict(X, verbose=0)
            proba = prediction_result[0][0]  # Get the probability value

            # Log the raw prediction result
            logger.info(f"Raw model prediction: {prediction_result}, shape: {prediction_result.shape}")
            logger.info(f"Extracted probability: {proba:.6f}")

            # Set confidence and prediction
            confidence = proba if proba > 0.5 else 1 - proba
            prediction = "UP" if proba > 0.5 else "DOWN"

            # Check if prediction is too close to 0.5 (indecisive)
            if abs(proba - 0.5) < 0.05:
                logger.warning(f"Prediction is very close to 0.5 (indecisive): {proba:.6f}")

            logger.info(f"LSTM prediction: {prediction}, Probability: {proba:.6f}, Confidence: {confidence:.6f}")

            # Apply confidence threshold
            if confidence < self.conf_threshold:
                return "HOLD", confidence

            return prediction, confidence

        except Exception as e:
            logger.error(f"LSTM prediction failed: {e}")
            return "HOLD", 0.0


