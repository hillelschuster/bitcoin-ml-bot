#!/usr/bin/env python
"""
Verify that the LSTM model and scaler files are valid.

This script checks:
1. If the model files exist
2. If they can be loaded correctly
3. If the model can make predictions

Usage:
    python scripts/verify_model_files.py
"""

import os
import sys
import pickle
import logging
import numpy as np
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('verify_model')

def verify_model_files():
    """Verify that the LSTM model and scaler files are valid."""
    # Check if model files exist
    model_path = 'model_artifacts/lstm_model.h5'
    scaler_path = 'model_artifacts/lstm_scaler.pkl'
    
    logger.info("Verifying model files...")
    
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        return False
    
    if not os.path.exists(scaler_path):
        logger.error(f"Scaler file not found: {scaler_path}")
        return False
    
    logger.info("Model files exist. Checking if they can be loaded...")
    
    try:
        # Try to import TensorFlow
        import tensorflow as tf
        logger.info(f"TensorFlow version: {tf.__version__}")
    except ImportError as e:
        logger.error(f"Failed to import TensorFlow: {e}")
        return False
    
    try:
        # Try to load the model
        model = tf.keras.models.load_model(model_path)
        logger.info(f"Model loaded successfully: {model_path}")
        logger.info(f"Model summary: {model.summary()}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return False
    
    try:
        # Try to load the scaler
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        logger.info(f"Scaler loaded successfully: {scaler_path}")
    except Exception as e:
        logger.error(f"Failed to load scaler: {e}")
        return False
    
    logger.info("Testing model prediction...")
    
    try:
        # Create a dummy sequence for testing
        sequence_length = model.input_shape[1]
        dummy_sequence = np.random.random((1, sequence_length, 1))
        
        # Scale the dummy sequence
        dummy_sequence_scaled = dummy_sequence  # Already between 0 and 1
        
        # Make a prediction
        prediction = model.predict(dummy_sequence_scaled)
        logger.info(f"Model prediction shape: {prediction.shape}")
        logger.info(f"Model prediction value: {prediction[0][0]}")
        
        # Interpret the prediction
        predicted_direction = "UP" if prediction[0][0] > 0.5 else "DOWN"
        logger.info(f"Predicted direction: {predicted_direction} (confidence: {prediction[0][0]:.4f})")
    except Exception as e:
        logger.error(f"Failed to make prediction: {e}")
        return False
    
    logger.info("✅ Model verification successful! The model and scaler are valid and can make predictions.")
    return True

if __name__ == "__main__":
    logger.info("Starting model verification...")
    
    if verify_model_files():
        logger.info("✅ Model verification passed!")
        sys.exit(0)
    else:
        logger.error("❌ Model verification failed!")
        sys.exit(1)
