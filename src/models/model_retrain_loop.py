"""Model retraining and file watching engine for LSTM model management.

This module provides file-based model watching and reloading via the ModelFileWatcher class.

It enables automatic detection and loading of updated model files,
ensuring the trading bot always uses the latest trained model without
requiring a restart.
"""
import os
import time
import logging
import threading
import joblib
import numpy as np
from datetime import datetime
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from tensorflow import keras
from src.core.utils.indicator_features import compute_atr, compute_rsi, compute_macd, compute_bollinger_width

# Configure logging
logger = logging.getLogger('model_retrain_loop')

# Initialize global model file watcher
_default_reloader = None


class ModelFileHandler(FileSystemEventHandler):
    """
    File system event handler for model file changes.

    This handler watches for modifications to model files and triggers
    the reload process when changes are detected.
    """

    def __init__(self, reloader):
        """
        Initialize the model file handler.

        Args:
            reloader: ModelFileWatcher instance that handles the actual reloading
        """
        self.reloader = reloader
        self.last_modified_time = 0
        self.cooldown_period = 2  # seconds to prevent multiple reloads for the same change

    def on_modified(self, event):
        """
        Handle file modification events.

        Args:
            event: File system event containing information about the changed file
        """
        try:
            # Only process non-directory events for the specific model file we're watching
            if not event.is_directory and os.path.abspath(event.src_path) == os.path.abspath(self.reloader.model_path):
                # Implement cooldown to prevent multiple reloads for the same change
                current_time = time.time()
                if current_time - self.last_modified_time > self.cooldown_period:
                    self.last_modified_time = current_time
                    logger.info(f"Model file changed: {event.src_path}")
                    time.sleep(0.5)  # Wait a moment for file to be fully written
                    self.reloader.reload_model()
        except Exception as e:
            logger.error(f"Error in on_modified: {e}")

class ModelFileWatcher:
    """
    File watcher for machine learning models.

    This class watches model files for changes and automatically reloads them
    when modifications are detected. It provides thread-safe access to the
    current model and scaler objects.
    """

    def __init__(self, model_path, scaler_path=None):
        """
        Initialize the model file watcher.

        Args:
            model_path: Path to the model file (typically a .h5 file for LSTM)
            scaler_path: Path to the scaler file (optional, will use default if None)
        """
        self.model_path = model_path
        self.scaler_path = scaler_path or os.path.join(os.path.dirname(model_path), 'lstm_scaler.pkl')

        # Initialize model and scaler
        self.model = None
        self.scaler = None

        # Initialize lock for thread safety when accessing model/scaler
        self.lock = threading.RLock()

        # Load model initially
        self.reload_model()

        # Initialize file system observer
        self.observer = None
        self.event_handler = ModelFileHandler(self)

        logger.info(f"Model file watcher initialized for {model_path}")

    def reload_model(self):
        """
        Reload the model and scaler from files.

        Returns:
            bool: True if reload was successful, False otherwise
        """
        try:
            # Acquire lock to ensure thread safety during reload
            with self.lock:
                # Check if model file exists
                if not os.path.exists(self.model_path):
                    logger.warning(f"Model file not found: {self.model_path}")
                    return False

                # Get file modification time for logging
                mod_time = datetime.fromtimestamp(os.path.getmtime(self.model_path))

                # Load the model
                try:
                    # Wait a moment to ensure file is fully written
                    time.sleep(0.5)

                    # Load the Keras LSTM model
                    new_model = keras.models.load_model(self.model_path)
                    logger.info(f"LSTM model loaded from {self.model_path} (modified: {mod_time})")

                    # Verify the model works by running a simple forward pass
                    try:
                        # Create a small random input to test the model
                        # Assuming model expects input shape [batch, time_steps, features]
                        # Default to 30 time steps and 10 features if we can't determine
                        input_shape = new_model.input_shape
                        if input_shape and len(input_shape) == 3:
                            batch_size, time_steps, n_features = 1, input_shape[1], input_shape[2]
                        else:
                            batch_size, time_steps, n_features = 1, 30, 10
                            logger.warning(f"Could not determine input shape from model, using default: (1, 30, 10)")

                        test_input = np.random.random((batch_size, time_steps, n_features))
                        test_output = new_model.predict(test_input, verbose=0)
                        logger.info(f"Model verification successful. Output shape: {test_output.shape}")

                        # If we get here, the model is working
                        self.model = new_model
                    except Exception as e:
                        logger.error(f"Model verification failed: {e}")
                        return False
                except Exception as e:
                    logger.error(f"Error loading model: {e}")
                    return False

                # Load scaler if available
                if os.path.exists(self.scaler_path):
                    try:
                        # Wait a moment to ensure file is fully written
                        time.sleep(0.5)

                        # Load the scaler using joblib
                        self.scaler = joblib.load(self.scaler_path)
                        logger.info(f"Scaler loaded from {self.scaler_path}")
                    except Exception as e:
                        logger.warning(f"Error loading scaler: {e}")

                return True

        except Exception as e:
            logger.error(f"Error in reload_model: {e}")
            return False

    def get_model(self):
        """
        Get the current model in a thread-safe manner.

        Returns:
            The loaded model object or None if no model is loaded
        """
        with self.lock:
            return self.model

    def get_scaler(self):
        """
        Get the current scaler in a thread-safe manner.

        Returns:
            The loaded scaler object or None if no scaler is loaded
        """
        with self.lock:
            return self.scaler

    def watch(self, interval=1.0):
        """
        Watch the model file for changes and reload when modifications are detected.

        This method starts a watchdog observer that monitors the directory containing
        the model file. When changes are detected, the model is automatically reloaded.

        Args:
            interval: Polling interval in seconds for the watchdog observer
        """
        try:
            # Create observer
            self.observer = Observer()

            # Schedule watching the directory containing the model file
            model_dir = os.path.dirname(os.path.abspath(self.model_path))
            self.observer.schedule(self.event_handler, model_dir, recursive=False)

            # Start observer
            self.observer.start()
            logger.info(f"Started watching {self.model_path} for changes")

            try:
                # Keep the thread alive
                while True:
                    time.sleep(interval)
            except KeyboardInterrupt:
                self.stop()

        except Exception as e:
            logger.error(f"Error in watch: {e}")

    def stop(self):
        """
        Stop watching the model file and clean up resources.
        """
        if self.observer:
            self.observer.stop()
            self.observer.join()
            logger.info("Stopped watching model file")

# Module-level functions for easier access to model and scaler via file watcher
def initialize_file_watcher(model_path='model_artifacts/lstm_model.h5', scaler_path=None):
    """
    Initialize the default file watcher for the module.

    Args:
        model_path: Path to the model file
        scaler_path: Path to the scaler file (optional)

    Returns:
        The initialized ModelFileWatcher instance
    """
    global _default_reloader
    _default_reloader = ModelFileWatcher(model_path, scaler_path)

    # Start watching in a background thread
    watch_thread = threading.Thread(target=_default_reloader.watch, daemon=True)
    watch_thread.start()

    return _default_reloader

def get_model():
    """
    Get the current model from the default file watcher.

    Returns:
        The loaded model object or None if no model is loaded
    """
    if _default_reloader is None:
        logger.warning("Model file watcher not initialized. Call initialize_file_watcher() first.")
        return None
    return _default_reloader.get_model()

def get_scaler():
    """
    Get the current scaler from the default file watcher.

    Returns:
        The loaded scaler object or None if no scaler is loaded
    """
    if _default_reloader is None:
        logger.warning("Model file watcher not initialized. Call initialize_file_watcher() first.")
        return None
    return _default_reloader.get_scaler()

def trigger_manual_retrain():
    """Trigger a manual model retraining process.

    This function manually triggers the model retraining process,
    which uses the latest trade data from the database to retrain
    the LSTM model. It logs the process and results.

    Returns:
        bool: True if retraining was successful, False otherwise.
    """
    try:
        from src.monitoring.model_monitor import retrain_model
        import hashlib

        # Calculate model hash before retraining
        def calculate_file_hash(file_path):
            sha256_hash = hashlib.sha256()
            with open(file_path, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()

        model_path = 'model_artifacts/lstm_model.h5'

        # Check if model exists
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            return False

        # Calculate hash before retraining
        original_hash = calculate_file_hash(model_path)
        logger.info(f"Original model hash: {original_hash[:8]}...")

        # Trigger retraining
        logger.info("Starting manual model retraining...")
        success = retrain_model()

        if success:
            # Calculate hash after retraining
            new_hash = calculate_file_hash(model_path)
            logger.info(f"New model hash: {new_hash[:8]}...")

            # Check if model was actually updated
            if original_hash == new_hash:
                logger.warning("Model was not updated (hash unchanged)")
            else:
                logger.info("Model was successfully updated")

            # Reload the model in the file watcher
            if _default_reloader:
                _default_reloader.reload_model()
                logger.info("Model file watcher updated with new model")

            return True
        else:
            logger.error("Manual retraining failed")
            return False
    except Exception as e:
        logger.error(f"Error in manual retraining: {e}")
        return False

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='LSTM Model Management')
    parser.add_argument('--mode', type=str, choices=['watch', 'retrain'], default='watch',
                      help='Mode: watch (monitor model file) or retrain (trigger manual retraining)')
    parser.add_argument('--model', type=str, default='model_artifacts/lstm_model.h5', help='Path to model file')
    parser.add_argument('--scaler', type=str, help='Path to scaler file')

    args = parser.parse_args()

    # Configure logging for standalone execution
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    if args.mode == 'retrain':
        # Trigger manual retraining
        print("Triggering manual model retraining...")
        success = trigger_manual_retrain()
        print(f"Retraining {'successful' if success else 'failed'}")
        print("Check logs/model_health.log for details")
    else:
        # Initialize the default file watcher
        reloader = initialize_file_watcher(args.model, args.scaler)

        print(f"Watching {args.model} for changes. Press Ctrl+C to stop.")

        try:
            while True:
                # Check model every 5 seconds
                model = get_model()
                if model:
                    print(f"Model loaded: {type(model).__name__}")
                else:
                    print("No model loaded")

                time.sleep(5)
        except KeyboardInterrupt:
            print("Stopping...")
            reloader.stop()
