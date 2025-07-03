#!/usr/bin/env python
"""Command-line tool for comparing machine learning models.

This tool allows users to compare the performance of two models on a dataset
and determine if the new model should replace the old model.

Safety features:
1. Model compatibility validation
2. Automatic backup before replacement
3. Scaler validation
4. Enhanced model evaluation
5. Detailed logging
"""
import os
import sys
import logging
import shutil
import hashlib
import json
from datetime import datetime

# Add the src directory to the path so we can import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.models.model_evaluator import ModelEvaluator
import joblib
from tensorflow import keras

def calculate_file_hash(file_path):
    """Calculate the SHA-256 hash of a file.

    Args:
        file_path (str): Path to the file

    Returns:
        str: SHA-256 hash of the file
    """
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        # Read and update hash in chunks of 4K
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def validate_scaler_compatibility(model_path, scaler_path):
    """Validate that a scaler is compatible with a model.

    Args:
        model_path (str): Path to the model file
        scaler_path (str): Path to the scaler file

    Returns:
        bool: True if the scaler is compatible, False otherwise
    """
    try:
        # Load the model and get its input shape
        model = keras.models.load_model(model_path)
        input_shape = model.input_shape

        # Load the scaler and check its feature count
        scaler = joblib.load(scaler_path)

        # For StandardScaler, check scale_ attribute which has one value per feature
        if hasattr(scaler, 'scale_'):
            scaler_feature_count = len(scaler.scale_)

            # For LSTM models, input shape is (None, sequence_length, features)
            # For other models, input shape might be (None, features)
            if len(input_shape) == 3:  # LSTM model
                model_feature_count = input_shape[2]
            else:  # Other model types
                model_feature_count = input_shape[1]

            # Check if feature counts match
            if scaler_feature_count != model_feature_count:
                logging.warning(f"Scaler feature count ({scaler_feature_count}) does not match model input features ({model_feature_count})")
                return False

            logging.info(f"Scaler validation passed: {scaler_feature_count} features")
            return True
        else:
            logging.warning("Scaler does not have expected attributes for validation")
            return False
    except Exception as e:
        logging.error(f"Error validating scaler compatibility: {e}")
        return False

def create_model_backup(model_path, scaler_path=None, backup_dir=None):
    """Create a backup of a model and its scaler.

    Args:
        model_path (str): Path to the model file
        scaler_path (str, optional): Path to the scaler file
        backup_dir (str, optional): Directory to store the backup

    Returns:
        dict: Dictionary with paths to the backup files
    """
    try:
        # Generate timestamp for the backup
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create backup directory if not specified
        if backup_dir is None:
            backup_dir = os.path.join(os.path.dirname(model_path), 'backups')

        # Create the backup directory if it doesn't exist
        os.makedirs(backup_dir, exist_ok=True)

        # Create backup filenames
        model_filename = os.path.basename(model_path)
        model_backup_path = os.path.join(backup_dir, f"{os.path.splitext(model_filename)[0]}_{timestamp}{os.path.splitext(model_filename)[1]}")

        # Copy the model file
        shutil.copy2(model_path, model_backup_path)
        logging.info(f"Model backed up to {model_backup_path}")

        # Copy the scaler file if provided
        scaler_backup_path = None
        if scaler_path and os.path.exists(scaler_path):
            scaler_filename = os.path.basename(scaler_path)
            scaler_backup_path = os.path.join(backup_dir, f"{os.path.splitext(scaler_filename)[0]}_{timestamp}{os.path.splitext(scaler_filename)[1]}")
            shutil.copy2(scaler_path, scaler_backup_path)
            logging.info(f"Scaler backed up to {scaler_backup_path}")

        # Create a metadata file with checksums
        metadata = {
            'timestamp': timestamp,
            'original_model_path': model_path,
            'backup_model_path': model_backup_path,
            'model_checksum': calculate_file_hash(model_path),
            'backup_model_checksum': calculate_file_hash(model_backup_path)
        }

        if scaler_path and os.path.exists(scaler_path) and scaler_backup_path:
            metadata.update({
                'original_scaler_path': scaler_path,
                'backup_scaler_path': scaler_backup_path,
                'scaler_checksum': calculate_file_hash(scaler_path),
                'backup_scaler_checksum': calculate_file_hash(scaler_backup_path)
            })

        # Save metadata
        metadata_path = os.path.join(backup_dir, f"backup_metadata_{timestamp}.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        return {
            'model_backup_path': model_backup_path,
            'scaler_backup_path': scaler_backup_path,
            'metadata_path': metadata_path
        }
    except Exception as e:
        logging.error(f"Error creating model backup: {e}")
        return None

def main():
    """Run the model comparator CLI."""
    import argparse

    parser = argparse.ArgumentParser(description='Compare machine learning models')
    parser.add_argument('--old-model', type=str, required=True, help='Path to the old model')
    parser.add_argument('--new-model', type=str, required=True, help='Path to the new model')
    parser.add_argument('--dataset', type=str, required=True, help='Path to the evaluation dataset')
    parser.add_argument('--target', type=str, default='target', help='Name of the target column')
    parser.add_argument('--min-improvement', type=float, default=0.01, help='Minimum improvement required')
    parser.add_argument('--archive', action='store_true', help='Archive the old model if replaced')
    parser.add_argument('--backup', action='store_true', help='Create a backup before replacing the model')
    parser.add_argument('--min-samples', type=int, default=100, help='Minimum number of samples required for evaluation')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('--log-file', type=str, help='Path to log file (default: logs/model_comparator.log)')

    args = parser.parse_args()

    # Set up log file
    log_file = args.log_file if args.log_file else 'logs/model_comparator.log'
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),  # Log to console
            logging.FileHandler(log_file)  # Log to file
        ]
    )

    logging.info(f"Model comparator started with arguments: {args}")

    # Validate that the model files exist
    if not os.path.exists(args.old_model):
        logging.error(f"Old model file not found: {args.old_model}")
        return

    if not os.path.exists(args.new_model):
        logging.error(f"New model file not found: {args.new_model}")
        return

    # Validate that the dataset file exists
    if not os.path.exists(args.dataset):
        logging.error(f"Dataset file not found: {args.dataset}")
        return

    # Check if the dataset has enough samples
    try:
        import pandas as pd
        df = pd.read_csv(args.dataset)
        if len(df) < args.min_samples:
            logging.warning(f"Dataset has only {len(df)} samples, which is less than the minimum required ({args.min_samples})")
            print(f"Warning: Dataset has only {len(df)} samples, which may not be sufficient for reliable evaluation")
    except Exception as e:
        logging.error(f"Error reading dataset: {e}")
        return

    # Create model evaluator
    evaluator = ModelEvaluator(args.dataset, args.target)

    # Compare models
    logging.info(f"Comparing models: {args.old_model} vs {args.new_model}")
    results = evaluator.evaluate_models(args.old_model, args.new_model)

    if results:
        # Log detailed results
        logging.info("\nModel Comparison Results:")
        logging.info(f"Old model - Accuracy: {results['old_model']['accuracy']:.4f}, F1: {results['old_model']['f1_score']:.4f}, Precision: {results['old_model']['precision']:.4f}, Recall: {results['old_model']['recall']:.4f}")
        logging.info(f"New model - Accuracy: {results['new_model']['accuracy']:.4f}, F1: {results['new_model']['f1_score']:.4f}, Precision: {results['new_model']['precision']:.4f}, Recall: {results['new_model']['recall']:.4f}")
        logging.info(f"Delta - Accuracy: {results['delta']['accuracy']:.4f}, F1: {results['delta']['f1_score']:.4f}, Precision: {results['delta']['precision']:.4f}, Recall: {results['delta']['recall']:.4f}")

        # Print results to console
        print("\nModel Comparison Results:")
        print(f"Old model - Accuracy: {results['old_model']['accuracy']:.4f}, F1: {results['old_model']['f1_score']:.4f}")
        print(f"New model - Accuracy: {results['new_model']['accuracy']:.4f}, F1: {results['new_model']['f1_score']:.4f}")
        print(f"Delta - Accuracy: {results['delta']['accuracy']:.4f}, F1: {results['delta']['f1_score']:.4f}")

        # Determine if the new model should replace the old model
        should_replace = evaluator.should_replace_model(results, args.min_improvement)
        logging.info(f"Recommendation: {'Replace' if should_replace else 'Keep'} the current model")
        print(f"\nRecommendation: {'Replace' if should_replace else 'Keep'} the current model")

        # Check if we should proceed with replacement
        if should_replace:
            # Validate scaler compatibility
            new_scaler_path = os.path.join(os.path.dirname(args.new_model), 'lstm_scaler.pkl')
            old_scaler_path = os.path.join(os.path.dirname(args.old_model), 'lstm_scaler.pkl')

            if os.path.exists(new_scaler_path):
                logging.info(f"Validating scaler compatibility: {new_scaler_path}")
                scaler_compatible = validate_scaler_compatibility(args.new_model, new_scaler_path)
                if not scaler_compatible:
                    logging.error("Scaler validation failed - aborting model replacement")
                    print("Error: Scaler validation failed - aborting model replacement")
                    return

            # Create backup before replacement (always backup regardless of archive flag)
            backup_created = False
            if args.backup or True:  # Always create backup for safety
                logging.info("Creating backup of current model before replacement")
                backup_result = create_model_backup(
                    args.old_model,
                    old_scaler_path if os.path.exists(old_scaler_path) else None
                )
                if backup_result:
                    backup_created = True
                    print(f"Backup created: {backup_result['model_backup_path']}")
                else:
                    logging.warning("Failed to create backup - proceeding with caution")
                    print("Warning: Failed to create backup - proceeding with caution")

            # Archive the old model if requested (using ModelEvaluator's archive method)
            if args.archive:
                archive_path = evaluator.archive_model(args.old_model)
                if archive_path:
                    logging.info(f"Old model archived to {archive_path}")
                    print(f"Old model archived to {archive_path}")

            # Copy the new model to replace the old model
            try:
                # Create a copy with the original filename preserved
                shutil.copy2(args.new_model, args.old_model)
                logging.info(f"New model copied to {args.old_model}")
                print(f"New model copied to {args.old_model}")

                # Copy the scaler if it exists
                if os.path.exists(new_scaler_path):
                    shutil.copy2(new_scaler_path, old_scaler_path)
                    logging.info(f"New scaler copied to {old_scaler_path}")
                    print(f"New scaler copied to {old_scaler_path}")

                # Verify the copied files
                if os.path.exists(args.old_model):
                    new_hash = calculate_file_hash(args.new_model)
                    replaced_hash = calculate_file_hash(args.old_model)
                    if new_hash == replaced_hash:
                        logging.info("File integrity verified after replacement")
                    else:
                        logging.warning("File integrity check failed after replacement")
                        print("Warning: File integrity check failed after replacement")
            except Exception as e:
                logging.error(f"Error replacing model: {e}")
                print(f"Error replacing model: {e}")

                # Attempt to restore from backup if available
                if backup_created and backup_result:
                    try:
                        logging.warning("Attempting to restore from backup")
                        shutil.copy2(backup_result['model_backup_path'], args.old_model)
                        if 'scaler_backup_path' in backup_result and backup_result['scaler_backup_path']:
                            shutil.copy2(backup_result['scaler_backup_path'], old_scaler_path)
                        logging.info("Restored from backup successfully")
                        print("Restored from backup successfully")
                    except Exception as restore_error:
                        logging.error(f"Error restoring from backup: {restore_error}")
                        print(f"Error restoring from backup: {restore_error}")
    else:
        logging.error("Model comparison failed to produce valid results")
        print("Model comparison failed")

if __name__ == "__main__":
    main()
