"""Test module for model monitoring functionality.

This module provides test functions for the model monitoring components.
"""
import os
import logging
import sys

# Add the src directory to the path so we can import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.monitoring.model_monitor import start_monitoring
from src.models.model_evaluator import ModelEvaluator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('test_model_monitor')

def test_model_monitor():
    """Test the ModelEvaluator class."""
    # Create a model evaluator
    evaluator = ModelEvaluator('model_artifacts/synthetic_data.csv')

    # Evaluate models
    results = evaluator.evaluate_models(
        old_model_path='model_artifacts/lstm_model.h5',
        new_model_path='model_artifacts/lstm_model.h5'
    )

    if results:
        print("Model comparison results:")
        print(f"Old model accuracy: {results['old_model']['accuracy']:.4f}")
        print(f"New model accuracy: {results['new_model']['accuracy']:.4f}")
        print(f"Accuracy delta: {results['delta']['accuracy']:.4f}")

        # Check if the new model should replace the old model
        should_replace = evaluator.should_replace_model(results)
        print(f"Should replace model: {should_replace}")
    else:
        print("Model evaluation failed")

def test_monitoring():
    """Test the monitoring functionality."""
    print("Starting model monitoring...")
    # This will run indefinitely, so we'll just call it and let it run for a bit
    try:
        start_monitoring(interval=10)  # Use a short interval for testing
    except KeyboardInterrupt:
        print("Monitoring stopped by user")

if __name__ == "__main__":
    # Choose which test to run
    import argparse
    parser = argparse.ArgumentParser(description='Test model monitoring functionality')
    parser.add_argument('--test', choices=['monitor', 'compare'], default='compare',
                        help='Which test to run (monitor or compare)')
    args = parser.parse_args()

    if args.test == 'monitor':
        test_monitoring()
    else:
        test_model_monitor()
