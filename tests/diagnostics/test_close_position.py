#!/usr/bin/env python
"""
Test script for closing an open position - FOR TESTING EXIT LOGIC ONLY. NOT USED IN LIVE TRADING.

This script demonstrates how to properly close a position using the modular exchange functionality.
It can be used to manually close a position if the bot fails to do so.
"""

import logging
import json
import time
from src.core.exchange.exchange_base import get_client
from src.core.exchange.order_manager import close_position
from src.core.exchange.price_utils import get_market_price
from src.utils.config import Config
from src.utils.notifier import send_notification

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/test_close.log')
    ]
)

logger = logging.getLogger('test_close')

def test_close_position():
    """Test closing an open position."""
    logger.info("Starting position close test")

    try:
        # Load position data
        with open("logs/active_position.json") as f:
            position_data = json.load(f)

        size = position_data.get("size")
        position_side = position_data.get("position_side", "long")

        if not size or size <= 0:
            logger.error("Invalid position size in active_position.json")
            return

        logger.info(f"Found position in file: {size} BTC, side: {position_side}")

        # Initialize exchange client
        config = Config()
        client = get_client(test_mode=config.get('test_mode', True))

        # Determine side for closing
        side = "SELL" if position_side.lower() == "long" else "BUY"

        # IMPORTANT: Use the correct symbol format - "BTCUSDT" for Binance API
        symbol = "BTCUSDT"  # Binance API format

        # Close the position
        logger.info(f"[EXIT-TRACE] Closing position with size: {size} BTC, side: {side}, symbol: {symbol}")
        close_result = close_position(client, symbol, size=size, market=True)

        if close_result:
            logger.info(f"[EXIT-VERIFY] Position closed successfully: {close_result}")
            send_notification(f"✅ Test close successful: {side} {size} BTC")

            # Update the active_position.json file with full position state
            current_time = time.time()
            exit_price = get_market_price(client, "BTCUSDT")

            # Try to get existing position data
            position_data = {
                "size": 0,
                "closed": True,
                "entry_price": 0,
                "position_side": position_side,
                "symbol": "BTCUSDT",
                "confidence": 0,
                "timestamp": 0,
                "exit_price": exit_price,
                "exit_time": current_time,
                "exit_reason": "test_close"
            }

            try:
                with open("logs/active_position.json", "r") as f:
                    existing_data = json.load(f)
                    # Preserve existing fields
                    for key in existing_data:
                        if key != "closed" and key != "size" and key != "exit_price" and key != "exit_time" and key != "exit_reason":
                            position_data[key] = existing_data[key]
            except Exception as e:
                logger.warning(f"[POSITION-FILE] Could not read existing position data: {e}")

            with open("logs/active_position.json", "w") as f:
                json.dump(position_data, f, indent=2)

            logger.info("[EXIT-VERIFY] active_position.json updated to closed state")
        else:
            logger.error("[EXIT-ERROR] Failed to close position")
            send_notification("❌ Test close failed")

    except Exception as e:
        logger.error(f"[EXIT-ERROR] Error closing position: {e}")
        send_notification(f"❌ Test close error: {str(e)[:100]}")

if __name__ == "__main__":
    test_close_position()
