#!/usr/bin/env python
"""
Main entry point for the Bitcoin ML Trading Bot.

This script starts four main components in parallel:
1. Model auto-reloader - periodically reloads the model module
2. Model monitor - checks model performance metrics
3. Telegram reporter - sends periodic status reports via Telegram
4. Trading bot - main trading loop that executes the strategy
"""

from src.core.bot.orchestrator import run_trading_bot
# Model hot reloader removed - model_monitor.py handles retraining
from src.monitoring.model_monitor import start_monitoring
from src.monitoring.reporter import start_periodic_reporting
from src.utils.notifier import send_notification
from src.core.exchange.exchange_base import get_client
from src.core.exchange.position_utils import get_current_position_size
from src.core.exchange.price_utils import get_market_price
from src.utils.config import Config
from threading import Thread
import logging
import json
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/bot.log')
    ]
)

logger = logging.getLogger('run')

def check_for_open_positions():
    """
    Check if there are any open positions when the bot starts.
    This is part of the crash recovery logic to handle bot restarts with open positions.

    Returns:
        bool: True if open positions were detected, False otherwise
    """
    try:
        # Initialize exchange client
        config = Config('config/config.json')
        client = get_client(test_mode=config.get('test_mode', True))

        # Check for open positions
        position_size = get_current_position_size(client, "BTCUSDT")

        if position_size > 0.0001:  # Position exists
            # Get position details
            current_price = get_market_price(client, "BTCUSDT")
            position_value = position_size * current_price
            position_side = "LONG" if position_size > 0 else "SHORT"

            # Log the detected position
            logger.warning(f"[CRASH RECOVERY] Detected open {position_side} position: {position_size:.6f} BTC (${position_value:.2f})")

            # Store the position size in active_position.json for tracking
            try:
                # Get position details
                position_side = "long" if position_size > 0 else "short"

                with open("logs/active_position.json", "w") as f:
                    json.dump({
                        "size": abs(position_size),
                        "closed": False,
                        "entry_price": current_price,  # We don't know the actual entry price, so use current price
                        "entry_time": time.time(),
                        "position_side": position_side,
                        "confidence": 0.85,  # Default confidence
                        "recovered": True  # Mark as recovered from crash
                    }, f)
                logger.info(f"[CRASH RECOVERY] Stored position in active_position.json: {position_size} BTC, side: {position_side}")
            except Exception as e:
                logger.error(f"[CRASH RECOVERY] Failed to store position size: {e}")

            # Send notification
            message = f"🔄 Bot restarted with open {position_side} position: {position_size:.6f} BTC (${position_value:.2f}). Resuming monitoring..."
            send_notification(message)

            return True
        else:
            logger.info("[CRASH RECOVERY] No open positions detected on startup")
            return False

    except Exception as e:
        logger.error(f"[CRASH RECOVERY] Error checking for open positions: {e}")
        return False

def verify_model_exists():
    """Verify that the LSTM model exists and can be loaded.

    This function ensures that the bot doesn't start if the model is missing or broken.
    It uses the real TensorFlow model loading to verify the model is valid.

    Returns:
        bool: True if model exists and can be loaded, False otherwise
    """
    from src.models.model_lstm_core import load_model

    logger.info("Verifying LSTM model exists and can be loaded...")
    model, scaler = load_model()

    if model is None or scaler is None:
        logger.critical("[MODEL ERROR] Could not load LSTM model or scaler. Exiting.")
        return False

    logger.info("LSTM model verified successfully")
    return True

if __name__ == "__main__":
    logger.info("Starting Bitcoin ML Trading Bot")

    # Verify that the model exists and can be loaded
    if not verify_model_exists():
        logger.critical("Bot startup aborted due to missing or broken model")
        send_notification("🚨 CRITICAL: Bot startup aborted - LSTM model missing or broken")
        exit(1)

    # Check for open positions (crash recovery)
    open_position_detected = check_for_open_positions()
    if open_position_detected:
        logger.info("[CRASH RECOVERY] Bot will resume monitoring the open position")

    # NOTE: model_hot_reloader removed - now using model_monitor.py

    # Start model performance monitoring
    logger.info("Starting model performance monitoring")
    Thread(target=start_monitoring, daemon=True).start()

    # Start periodic Telegram reporting (every 30 minutes)
    logger.info("Starting periodic Telegram reporting")
    Thread(target=start_periodic_reporting, args=(1800,), daemon=True).start()

    # Start the bot
    logger.info("Starting trading bot")
    run_trading_bot()
