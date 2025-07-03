"""Model monitoring utilities for the Bitcoin ML Trading Bot.

This module provides utilities for monitoring model performance,
comparing models, and evaluating model quality.
"""
import os
import logging
import sqlite3
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import joblib
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from src.monitoring.model_monitor import prepare_sequences

# Configure logging
logger = logging.getLogger('model_evaluator')

# Configure model health logger
model_health_logger = logging.getLogger('model_health')
if not model_health_logger.handlers:
    model_health_handler = logging.FileHandler('logs/model_health.log')
    model_health_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    model_health_logger.addHandler(model_health_handler)
    model_health_logger.setLevel(logging.INFO)

class ModelEvaluator:
    """Class for monitoring and evaluating machine learning models.

    This class provides methods for evaluating model performance,
    comparing models, and archiving models.
    """

    def __init__(self, dataset_path, target_column='target'):
        """Initialize the model monitor.

        Args:
            dataset_path (str): Path to the dataset for model evaluation
            target_column (str): Name of the target column in the dataset
        """
        self.dataset_path = dataset_path
        self.target_column = target_column
        logger.info(f"ModelEvaluator initialized with dataset: {dataset_path}")

    def evaluate_models(self, old_model_path, new_model_path):
        """Compare the performance of two models on the same dataset.

        Args:
            old_model_path (str): Path to the old model file
            new_model_path (str): Path to the new model file

        Returns:
            dict: Dictionary containing performance metrics for both models
                  and the delta between them
        """
        try:
            logger.info(f"Comparing models: {old_model_path} vs {new_model_path}")

            # Load dataset
            df = pd.read_csv(self.dataset_path)

            # Prepare features and target
            X = df.drop(columns=["timestamp", "close", self.target_column]
                       if "timestamp" in df.columns else ["close", self.target_column])
            y = df[self.target_column]

            # Load models
            logger.info(f"Loading old model from {old_model_path}")
            old_model = keras.models.load_model(old_model_path)

            logger.info(f"Loading new model from {new_model_path}")
            new_model = keras.models.load_model(new_model_path)

            # Prepare sequences for LSTM
            X_sequences, y = prepare_sequences(df)
            if X_sequences is None or y is None:
                logger.error("Failed to prepare sequences for model evaluation")
                return None

            # Load scaler
            scaler_path = os.path.join(os.path.dirname(old_model_path), 'lstm_scaler.pkl')
            scaler = joblib.load(scaler_path)

            # Scale sequences
            n_samples, seq_len, n_features = X_sequences.shape
            X_scaled = np.zeros_like(X_sequences)
            for i in range(n_samples):
                for j in range(seq_len):
                    X_scaled[i, j] = scaler.transform([X_sequences[i, j]])[0]

            # Make predictions
            old_preds_prob = old_model.predict(X_scaled, verbose=0)
            new_preds_prob = new_model.predict(X_scaled, verbose=0)

            old_preds = (old_preds_prob > 0.5).astype(int).flatten()
            new_preds = (new_preds_prob > 0.5).astype(int).flatten()

            # Calculate metrics
            old_acc = accuracy_score(y, old_preds)
            new_acc = accuracy_score(y, new_preds)

            old_precision = precision_score(y, old_preds, zero_division=0)
            new_precision = precision_score(y, new_preds, zero_division=0)

            old_recall = recall_score(y, old_preds, zero_division=0)
            new_recall = recall_score(y, new_preds, zero_division=0)

            old_f1 = f1_score(y, old_preds, zero_division=0)
            new_f1 = f1_score(y, new_preds, zero_division=0)

            # Log to model health log
            model_health_logger.info(f"Model comparison - Old: acc={old_acc:.4f}, f1={old_f1:.4f} | New: acc={new_acc:.4f}, f1={new_f1:.4f}")

            # Calculate confusion matrices
            old_cm = confusion_matrix(y, old_preds)
            new_cm = confusion_matrix(y, new_preds)

            # Calculate deltas
            acc_delta = new_acc - old_acc
            precision_delta = new_precision - old_precision
            recall_delta = new_recall - old_recall
            f1_delta = new_f1 - old_f1

            # Prepare results
            results = {
                "old_model": {
                    "accuracy": old_acc,
                    "precision": old_precision,
                    "recall": old_recall,
                    "f1_score": old_f1,
                    "confusion_matrix": old_cm.tolist()
                },
                "new_model": {
                    "accuracy": new_acc,
                    "precision": new_precision,
                    "recall": new_recall,
                    "f1_score": new_f1,
                    "confusion_matrix": new_cm.tolist()
                },
                "delta": {
                    "accuracy": acc_delta,
                    "precision": precision_delta,
                    "recall": recall_delta,
                    "f1_score": f1_delta
                }
            }

            # Log results
            logger.info(f"Model comparison results:")
            logger.info(f"Old model - Accuracy: {old_acc:.4f}, F1: {old_f1:.4f}")
            logger.info(f"New model - Accuracy: {new_acc:.4f}, F1: {new_f1:.4f}")
            logger.info(f"Delta - Accuracy: {acc_delta:.4f}, F1: {f1_delta:.4f}")

            return results

        except Exception as e:
            logger.error(f"Error comparing models: {e}")
            return None

    def should_replace_model(self, comparison_results, min_improvement=0.01):
        """Determine if the new model should replace the old model.

        This method implements strict model upgrade conditions to ensure
        that the new model is actually better than the old one before
        replacing it.

        Args:
            comparison_results (dict): Results from evaluate_models
            min_improvement (float): Minimum improvement in accuracy required

        Returns:
            bool: True if the new model should replace the old model
        """
        if comparison_results is None:
            logger.warning("Cannot make replacement decision: comparison results are None")
            return False

        # Extract metrics
        acc_delta = comparison_results["delta"]["accuracy"]
        f1_delta = comparison_results["delta"]["f1_score"]
        precision_delta = comparison_results["delta"]["precision"]
        recall_delta = comparison_results["delta"]["recall"]

        # Get absolute values
        old_acc = comparison_results["old_model"]["accuracy"]
        new_acc = comparison_results["new_model"]["accuracy"]

        # Log detailed metrics
        logger.info(f"Model comparison metrics:")
        logger.info(f"Accuracy: {old_acc:.4f} → {new_acc:.4f} (Δ: {acc_delta:.4f})")
        logger.info(f"F1 Score: {comparison_results['old_model']['f1_score']:.4f} → {comparison_results['new_model']['f1_score']:.4f} (Δ: {f1_delta:.4f})")
        logger.info(f"Precision: {comparison_results['old_model']['precision']:.4f} → {comparison_results['new_model']['precision']:.4f} (Δ: {precision_delta:.4f})")
        logger.info(f"Recall: {comparison_results['old_model']['recall']:.4f} → {comparison_results['new_model']['recall']:.4f} (Δ: {recall_delta:.4f})")

        # Decision logic: accuracy must improve by min_improvement OR
        # accuracy must not decrease AND f1 must improve by min_improvement
        should_replace = (acc_delta >= min_improvement or
                         (acc_delta >= 0 and f1_delta >= min_improvement))

        # Additional safety check: new accuracy must be at least 0.6
        if new_acc < 0.6:
            logger.warning(f"New model accuracy ({new_acc:.4f}) is below minimum threshold (0.6)")
            should_replace = False

        logger.info(f"Model replacement decision: {should_replace} (acc_delta={acc_delta:.4f}, f1_delta={f1_delta:.4f})")

        # Send alert if model will be replaced
        if should_replace:
            try:
                from src.utils.model_alert import send_model_upgrade_alert
                send_model_upgrade_alert(
                    comparison_results["old_model"],
                    comparison_results["new_model"]
                )
            except Exception as e:
                logger.error(f"Failed to send model upgrade alert: {e}")

        return should_replace

    def archive_model(self, model_path, archive_dir="model_artifacts/history"):
        """Archive a model file with a timestamp.

        This method creates a backup of the model file with a timestamp
        before it is replaced. It also validates the integrity of the
        backup using SHA256 checksums.

        Args:
            model_path (str): Path to the model file
            archive_dir (str): Directory to store archived models

        Returns:
            str: Path to the archived model file
        """
        try:
            # Create archive directory if it doesn't exist
            os.makedirs(archive_dir, exist_ok=True)

            # Generate timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Create archive filename
            model_filename = os.path.basename(model_path)
            archive_filename = f"{os.path.splitext(model_filename)[0]}_{timestamp}{os.path.splitext(model_filename)[1]}"
            archive_path = os.path.join(archive_dir, archive_filename)

            # Calculate hash of original model for integrity verification
            import hashlib

            def calculate_file_hash(file_path):
                sha256_hash = hashlib.sha256()
                with open(file_path, "rb") as f:
                    for byte_block in iter(lambda: f.read(4096), b""):
                        sha256_hash.update(byte_block)
                return sha256_hash.hexdigest()

            # Calculate hash of original model
            original_hash = calculate_file_hash(model_path)

            # Copy the model file to archive
            import shutil
            shutil.copy2(model_path, archive_path)

            # Verify the integrity of the archived model
            archive_hash = calculate_file_hash(archive_path)

            if original_hash == archive_hash:
                logger.info(f"Model archived to {archive_path} (hash verified: {original_hash[:8]}...)")
            else:
                logger.warning(f"Model archived but hash mismatch: {original_hash[:8]}... vs {archive_hash[:8]}...")

            # Also archive the scaler if it exists
            scaler_path = os.path.join(os.path.dirname(model_path), 'lstm_scaler.pkl')
            if os.path.exists(scaler_path):
                scaler_archive_path = os.path.join(archive_dir, f"lstm_scaler_{timestamp}.pkl")
                shutil.copy2(scaler_path, scaler_archive_path)
                logger.info(f"Scaler archived to {scaler_archive_path}")

            return archive_path

        except Exception as e:
            logger.error(f"Error archiving model: {e}")
            return None

    def evaluate_model_on_dataset(self, model_path, dataset_path=None, target_column=None):
        """Evaluate a model on a dataset.

        Args:
            model_path (str): Path to the model file
            dataset_path (str, optional): Path to the dataset
            target_column (str, optional): Name of the target column

        Returns:
            dict: Dictionary containing performance metrics or None if evaluation fails
        """
        try:
            # Use provided dataset or default
            dataset_path = dataset_path or self.dataset_path
            target_column = target_column or self.target_column

            logger.info(f"Evaluating model {model_path} on dataset {dataset_path}")

            # Load dataset
            df = pd.read_csv(dataset_path)

            # Check if target column exists in the dataset
            if target_column not in df.columns:
                logger.warning(f"Target column '{target_column}' not found in dataset. Available columns: {df.columns.tolist()}")
                return None

            # Prepare features and target
            X = df.drop(columns=["timestamp", "close", target_column]
                       if "timestamp" in df.columns else ["close", target_column])
            y = df[target_column]

            # Prepare sequences for LSTM
            X_sequences, y = prepare_sequences(df)
            if X_sequences is None or y is None:
                logger.error("Failed to prepare sequences for model evaluation")
                return None

            # Load model and scaler
            logger.info(f"Loading model from {model_path}")
            model = keras.models.load_model(model_path)

            scaler_path = os.path.join(os.path.dirname(model_path), 'lstm_scaler.pkl')
            scaler = joblib.load(scaler_path)

            # Scale sequences
            n_samples, seq_len, n_features = X_sequences.shape
            X_scaled = np.zeros_like(X_sequences)
            for i in range(n_samples):
                for j in range(seq_len):
                    X_scaled[i, j] = scaler.transform([X_sequences[i, j]])[0]

            # Make predictions
            preds_prob = model.predict(X_scaled, verbose=0)
            preds = (preds_prob > 0.5).astype(int).flatten()

            # Calculate metrics
            acc = accuracy_score(y, preds)
            precision = precision_score(y, preds, zero_division=0)
            recall = recall_score(y, preds, zero_division=0)
            f1 = f1_score(y, preds, zero_division=0)
            cm = confusion_matrix(y, preds)

            # Prepare results
            results = {
                "accuracy": acc,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "confusion_matrix": cm.tolist()
            }

            # Log results
            logger.info(f"Model evaluation results:")
            logger.info(f"Accuracy: {acc:.4f}")
            logger.info(f"Precision: {precision:.4f}")
            logger.info(f"Recall: {recall:.4f}")
            logger.info(f"F1 Score: {f1:.4f}")

            # Log to model health log
            model_health_logger.info(f"Model evaluation: acc={acc:.4f}, f1={f1:.4f}, precision={precision:.4f}, recall={recall:.4f}")

            return results

        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            return None

def create_model_comparator_cli():
    """Create a command-line interface for model comparison."""
    import argparse

    parser = argparse.ArgumentParser(description='Compare machine learning models')
    parser.add_argument('--old-model', type=str, required=True, help='Path to the old model')
    parser.add_argument('--new-model', type=str, required=True, help='Path to the new model')
    parser.add_argument('--dataset', type=str, required=True, help='Path to the evaluation dataset')
    parser.add_argument('--target', type=str, default='target', help='Name of the target column')
    parser.add_argument('--min-improvement', type=float, default=0.01, help='Minimum improvement required')
    parser.add_argument('--archive', action='store_true', help='Archive the old model if replaced')
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                      help='Set the logging level (default: INFO)')

    args = parser.parse_args()

    # Create model evaluator
    evaluator = ModelEvaluator(args.dataset, args.target)

    # Compare models
    try:
        results = evaluator.evaluate_models(args.old_model, args.new_model)

        print("\nModel Comparison Results:")
        if results:
            print(f"Old model - Accuracy: {results['old_model']['accuracy']:.4f}, F1: {results['old_model']['f1_score']:.4f}")
            print(f"New model - Accuracy: {results['new_model']['accuracy']:.4f}, F1: {results['new_model']['f1_score']:.4f}")
            print(f"Delta - Accuracy: {results['delta']['accuracy']:.4f}, F1: {results['delta']['f1_score']:.4f}")

            # Determine if the new model should replace the old model
            should_replace = evaluator.should_replace_model(results, args.min_improvement)
            print(f"\nRecommendation: {'Replace' if should_replace else 'Keep'} the current model")

            # Archive the old model if requested
            if args.archive and should_replace:
                archive_path = evaluator.archive_model(args.old_model)
                if archive_path:
                    print(f"Old model archived to {archive_path}")
        else:
            print("Model comparison failed to produce valid results")
            print("Please check the logs for more details")
    except Exception as e:
        print(f"\nError during model comparison: {e}")
        print("Please check the logs for more details")

if __name__ == "__main__":
    import sys

    # Run the CLI
    try:
        # Parse arguments first to get log level before configuring logging
        import argparse
        parser = argparse.ArgumentParser(description='Compare machine learning models')
        parser.add_argument('--log-level', type=str, default='INFO',
                          choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                          help='Set the logging level (default: INFO)')
        # Parse just the log level argument
        args, _ = parser.parse_known_args()

        # Configure logging for CLI usage with the specified level
        log_level = getattr(logging, args.log_level.upper())
        logging.basicConfig(level=log_level,
                           format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        logger.info(f"Logging level set to {args.log_level}")

        # Run the CLI
        create_model_comparator_cli()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
