"""Model monitoring module for the Bitcoin ML Trading Bot.

This module monitors the LSTM model's accuracy on recent trades,
and retrains it automatically if accuracy drops or new trades exceed a threshold.
"""

import os
import logging
import sqlite3
import numpy as np
import pandas as pd
from time import sleep
from datetime import datetime
from tensorflow import keras
from sklearn.metrics import accuracy_score
from src.models.model_lstm_core import prepare_sequences, load_model
from src.models.model_retrain_loop import get_model, get_scaler

# === Config ===
DB_PATH = "logs/trades.db"
MODEL_PATH = "model_artifacts/lstm_model.h5"
SCALER_PATH = "model_artifacts/lstm_scaler.pkl"
SEQUENCE_LENGTH = 30
ACCURACY_THRESHOLD = 0.68  # Only retrain if accuracy drops below this threshold
TRADE_COUNT_THRESHOLD = 50   # Trigger retraining evaluation every 50 trades
CHECK_INTERVAL = 600  # seconds

# === Logging Setup ===
logger = logging.getLogger("model_monitor")
logger.setLevel(logging.INFO)
fh = logging.FileHandler("logs/model_health.log")
fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(fh)

def prepare_sequences(df, sequence_length=SEQUENCE_LENGTH):
    try:
        if 'timestamp' in df.columns:
            df = df.drop(columns=['timestamp'])
        if 'close' in df.columns:
            target = (df['close'].shift(-1) > df['close']).astype(int)
            df = df.drop(columns=['close'])
        elif 'target' in df.columns:
            target = df['target']
            df = df.drop(columns=['target'])
        else:
            logger.error("No target or close column found.")
            return None, None

        X, y = [], []
        for i in range(len(df) - sequence_length):
            X.append(df.iloc[i:i+sequence_length].values)
            y.append(target.iloc[i+sequence_length])
        return np.array(X), np.array(y)
    except Exception as e:
        logger.error(f"Error preparing sequences: {e}")
        return None, None

def retrain_model(trades_df=None):
    """Retrain the LSTM model using recent trade data.

    This function retrains the model using the latest trade data from the database,
    including all available features. It logs the retraining timestamp and ensures
    the model is properly saved with validation.

    Args:
        trades_df (DataFrame, optional): DataFrame containing trade data. If None,
                                        data will be loaded from the database.

    Returns:
        bool: True if retraining was successful, False otherwise.
    """
    try:
        # Record start time for logging
        start_time = datetime.now()
        logger.info(f"[RETRAIN] Starting model retraining at {start_time}")

        # Load trade data if not provided
        if trades_df is None:
            if not os.path.exists(DB_PATH):
                logger.error(f"[RETRAIN] Missing DB at {DB_PATH}")
                return False
            conn = sqlite3.connect(DB_PATH)
            # Get more samples for better training
            trades_df = pd.read_sql("SELECT * FROM trades ORDER BY timestamp DESC LIMIT 1000", conn)
            conn.close()

        if trades_df.empty:
            logger.warning("[RETRAIN] No trades for retraining.")
            return False

        # Extract all available features for better model training
        # Include all numeric columns except ID and timestamp-related ones
        feature_cols = [col for col in trades_df.columns
                      if trades_df[col].dtype in ['float64', 'int64']
                      and col not in ['id', 'timestamp', 'trade_id']]

        logger.info(f"[RETRAIN] Using features: {feature_cols}")

        # Create a copy to avoid modifying the original dataframe
        df = trades_df[['timestamp'] + feature_cols].copy()

        # Define target based on PnL
        if 'pnl' in df.columns:
            df['target'] = (df['pnl'] > 0).astype(int)
        else:
            # Fallback if PnL not available
            df['target'] = ((df['exit_price'] > df['entry_price']).astype(int)
                           if 'exit_price' in df.columns and 'entry_price' in df.columns
                           else None)
            if df['target'] is None:
                logger.error("[RETRAIN] Cannot determine target variable")
                return False

        # Prepare sequences for LSTM
        X, y = prepare_sequences(df)
        if X is None or y is None or len(X) < 50:  # Ensure enough samples
            logger.error(f"[RETRAIN] Insufficient data: {len(X) if X is not None else 0} samples")
            return False

        # Load current model and scaler
        model, scaler = load_model()
        if model is None or scaler is None:
            logger.error("[RETRAIN] Failed to load current model or scaler")
            return False

        # Scale the input features
        X_scaled = np.zeros_like(X)
        for i in range(len(X)):
            for j in range(SEQUENCE_LENGTH):
                X_scaled[i, j] = scaler.transform([X[i, j]])[0]

        # Log training details
        logger.info(f"[RETRAIN] Training on {len(X_scaled)} samples with {X.shape[2]} features")

        # Train the model with early stopping and learning rate reduction
        history = model.fit(
            X_scaled, y,
            epochs=30,  # More epochs with early stopping
            batch_size=32,
            validation_split=0.2,
            callbacks=[
                keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3)
            ],
            verbose=1
        )

        # Calculate training duration
        duration = (datetime.now() - start_time).total_seconds()

        # Get final metrics
        final_loss = history.history['loss'][-1]
        final_val_loss = history.history['val_loss'][-1] if 'val_loss' in history.history else None

        # Create backup of current model before saving new one
        import shutil
        import hashlib

        # Calculate hash of current model for integrity verification
        def calculate_file_hash(file_path):
            sha256_hash = hashlib.sha256()
            with open(file_path, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()

        # Create backup directory if it doesn't exist
        backup_dir = os.path.join(os.path.dirname(MODEL_PATH), 'history')
        os.makedirs(backup_dir, exist_ok=True)

        # Create backup filename with timestamp
        timestamp_str = start_time.strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(backup_dir, f"lstm_model_{timestamp_str}.h5")

        # Calculate hash of current model
        original_hash = calculate_file_hash(MODEL_PATH)

        # Create backup
        shutil.copy2(MODEL_PATH, backup_path)
        logger.info(f"[RETRAIN] Backed up current model to {backup_path}")

        # Save new model
        model.save(MODEL_PATH)

        # Verify new model integrity
        new_hash = calculate_file_hash(MODEL_PATH)

        # Log retraining results
        logger.info(f"[RETRAIN] Model saved to {MODEL_PATH}")
        logger.info(f"[RETRAIN] Training duration: {duration:.2f} seconds")
        logger.info(f"[RETRAIN] Final loss: {final_loss:.4f}, Val loss: {final_val_loss:.4f if final_val_loss else 'N/A'}")
        logger.info(f"[RETRAIN] Original model hash: {original_hash[:8]}...")
        logger.info(f"[RETRAIN] New model hash: {new_hash[:8]}...")

        # Save retraining metadata
        metadata = {
            "timestamp": timestamp_str,
            "samples": len(X_scaled),
            "features": X.shape[2],
            "duration": duration,
            "final_loss": float(final_loss),
            "final_val_loss": float(final_val_loss) if final_val_loss else None,
            "original_hash": original_hash,
            "new_hash": new_hash
        }

        # Save metadata to file
        import json
        metadata_path = os.path.join(backup_dir, f"retrain_metadata_{timestamp_str}.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        # Update model_metadata.json to indicate model is now trained
        model_metadata = {
            "trained": True,
            "version": "v1.0",
            "training_samples": len(X),
            "retrained_at": datetime.now().isoformat()
        }

        model_metadata_path = "model_artifacts/model_metadata.json"
        with open(model_metadata_path, "w") as f:
            json.dump(model_metadata, f, indent=2)

        logger.info(f"[RETRAIN] Updated model metadata: {model_metadata}")

        logger.info(f"[RETRAIN] Completed successfully at {datetime.now()}")
        return True
    except Exception as e:
        logger.error(f"[RETRAIN] Failed: {e}")
        return False

def initialize_model_watcher(model_path=MODEL_PATH, scaler_path=SCALER_PATH):
    """
    Initialize model file watching and automatic reloading.

    This function provides model file watching functionality
    with a more integrated approach that works with the model monitoring system.

    Args:
        model_path: Path to the model file
        scaler_path: Path to the scaler file

    Returns:
        bool: True if initialization was successful
    """
    try:
        from src.models.model_retrain_loop import initialize_file_watcher
        initialize_file_watcher(model_path, scaler_path)
        logger.info(f"Model watcher initialized for {model_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize model watcher: {e}")
        return False

def start_monitoring(interval=CHECK_INTERVAL):
    logger.info("Starting model monitor...")
    last_trade_count = 0

    # Initialize the model watcher
    initialize_model_watcher(MODEL_PATH, SCALER_PATH)

    while True:
        try:
            # Try to get model from file watcher first
            model = get_model()
            scaler = get_scaler()

            # Fall back to direct loading if file watcher not initialized
            if model is None or scaler is None:
                model, scaler = load_model()
                if model is None or scaler is None:
                    sleep(interval)
                    continue

            # Check if model is trained by loading metadata
            from src.models.model_lstm_core import load_model_metadata
            metadata = load_model_metadata()
            if not metadata.get("trained", False):
                logger.warning("⚠️ Model Monitor: Detected untrained placeholder model. No live trading signals will be generated until retraining.")

            if not os.path.exists(DB_PATH):
                sleep(interval)
                continue

            conn = sqlite3.connect(DB_PATH)
            trades_df = pd.read_sql("SELECT * FROM trades ORDER BY timestamp DESC LIMIT 200", conn)
            conn.close()

            if trades_df.empty:
                sleep(interval)
                continue

            current_trade_count = len(trades_df)
            new_trades = current_trade_count - last_trade_count

            df = trades_df[['timestamp', 'entry_price', 'exit_price', 'amount', 'pnl']].copy()
            df['target'] = (df['pnl'] > 0).astype(int)
            X, y_true = prepare_sequences(df)

            if X is None or y_true is None or len(X) == 0:
                sleep(interval)
                continue

            X_scaled = np.zeros_like(X)
            for i in range(len(X)):
                for j in range(SEQUENCE_LENGTH):
                    X_scaled[i, j] = scaler.transform([X[i, j]])[0]

            y_pred = (model.predict(X_scaled, verbose=0) > 0.5).astype(int).flatten()
            acc = accuracy_score(y_true, y_pred)

            logger.info(f"LSTM accuracy: {acc:.4f} on {len(y_true)} samples")

            if acc < ACCURACY_THRESHOLD:
                logger.warning("Accuracy low — triggering retrain.")
                retrain_model(trades_df)
                last_trade_count = current_trade_count
            elif new_trades >= TRADE_COUNT_THRESHOLD:
                logger.info(f"{new_trades} new trades — triggering retrain.")
                retrain_model(trades_df)
                last_trade_count = current_trade_count

        except Exception as e:
            logger.error(f"Monitor loop error: {e}")
        sleep(interval)


def sanity_check_last_trade():
    """Check if the last trade in the database has valid data.

    Returns:
        bool: True if the trade data is valid, False otherwise
    """
    try:
        db_path = os.path.join("data", "db", "trades.db")
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query("SELECT * FROM trades ORDER BY timestamp DESC LIMIT 1", conn)
        conn.close()

        if df.empty:
            logger.warning("🟡 No recent trades found in DB.")
            return False

        trade = df.iloc[0]
        pnl = trade.get("pnl")
        qty = trade.get("amount")
        entry = trade.get("entry_price")
        exit = trade.get("exit_price")

        if any(x is None or x == 0 for x in [pnl, qty, entry, exit]):
            logger.error(f"❌ Invalid trade data detected: {trade.to_dict()}")
            return False

        if abs(pnl) > (entry * qty * 2):
            logger.error(f"⚠️ Unrealistic PnL detected: {pnl} (Check entry/exit logic)")
            return False

        logger.info("✅ Last trade passed sanity check.")
        return True

    except Exception as e:
        logger.error(f"❌ Error in sanity_check_last_trade: {e}")
        return False

def check_model_health():
    """Check if the model performance metrics are within acceptable ranges.

    Returns:
        bool: True if the model health is good, False otherwise
    """
    try:
        db_path = os.path.join("data", "db", "trades.db")
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query("SELECT * FROM model_performance ORDER BY timestamp DESC LIMIT 1", conn)
        conn.close()

        if df.empty:
            logger.warning("🟡 No model performance logs found.")
            return False

        row = df.iloc[0]
        loss = row.get("loss")
        if loss is not None and loss > 0.2:
            logger.warning(f"⚠️ Model loss high: {loss:.4f}")
            return False

        logger.info("✅ Model performance OK.")
        return True

    except Exception as e:
        logger.error(f"❌ Error in check_model_health: {e}")
        return False

def run_integrity_checks():
    """Run all integrity checks.

    Returns:
        bool: True if all checks pass, False otherwise
    """
    logger.info("🔍 Running integrity checks...")
    trade_ok = sanity_check_last_trade()
    model_ok = check_model_health()

    if not trade_ok or not model_ok:
        logger.warning("🛑 Integrity check failed. Alerting operator.")
        return False

    return True

def start_integrity_monitoring(interval=3600):
    """Start a background process that periodically runs integrity checks.

    This function continuously monitors the integrity of the trading system
    by checking trade data validity and model health metrics.

    Args:
        interval (int): Time in seconds between checks (default: 3600 - 1 hour)

    Note:
        This function runs in an infinite loop and should be executed in a separate thread.
    """
    logger.info(f"[Integrity Monitor] Starting with check interval of {interval} seconds")

    while True:
        try:
            result = run_integrity_checks()
            if result:
                logger.info("[Integrity Monitor] All checks passed successfully")
            else:
                logger.warning("[Integrity Monitor] One or more checks failed")
        except Exception as e:
            logger.error(f"[Integrity Monitor] Error during checks: {e}")

        # Sleep until next check
        sleep(interval)
