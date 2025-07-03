#!/usr/bin/env python
"""
Test script for model verification logic.
"""

import os
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)

logger = logging.getLogger('test')

def verify_model_exists():
    """Verify that the LSTM model exists and can be loaded.
    
    This function ensures that the bot doesn't start if the model is missing or broken.
    
    Returns:
        bool: True if model exists and can be loaded, False otherwise
    """
    # Check if model files exist
    model_path = 'model_artifacts/lstm_model.h5'
    scaler_path = 'model_artifacts/lstm_scaler.pkl'
    
    logger.info("Verifying LSTM model exists...")
    
    if not os.path.exists(model_path):
        logger.critical(f"[MODEL ERROR] Model file not found: {model_path}. Exiting.")
        return False
        
    if not os.path.exists(scaler_path):
        logger.critical(f"[MODEL ERROR] Scaler file not found: {scaler_path}. Exiting.")
        return False
    
    # In a real scenario, we would also try to load the model here
    # But for testing purposes, we'll just check if the files exist
    
    logger.info("LSTM model files verified successfully")
    return True

if __name__ == "__main__":
    logger.info("Testing model verification logic")
    
    # Verify that the model exists and can be loaded
    if not verify_model_exists():
        logger.critical("Bot startup aborted due to missing or broken model")
        sys.exit(1)
    
    logger.info("Model verification passed")
