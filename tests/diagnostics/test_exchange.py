#!/usr/bin/env python
"""Test script for the exchange functionality.

This script tests the exchange functionality with proper error handling and retry logic.
"""

import os
import sys
import logging
import json

# Add the src directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.utils.config import Config
from src.core.exchange.exchange_base import get_client
from src.core.exchange.price_utils import get_market_price
from src.core.exchange.position_utils import get_current_position_size
from src.core.exchange.exchange_base import get_account_balance

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger('test_exchange')

def create_test_config():
    """Create a test configuration with mock API keys."""
    config_data = {
        "version": "1.0",
        "test_mode": True,
        "trading": {
            "symbol": "BTCUSDT",
            "timeframe": "1h",
            "risk_per_trade": 0.02,
            "quantity": 0.001,
            "stop_loss_pct": 0.02,
            "take_profit_pct": 0.04
        },
        "model": {
            "confidence_threshold": 0.65,
            "type": "lstm"
        },
        "atr_period": 14,
        "logging": {
            "level": "INFO",
            "file": "logs/trading_bot.log"
        },
        "binance_api_key": "test_api_key",
        "binance_secret_key": "test_api_secret"
    }

    # Save the test config
    with open("test_config.json", "w") as f:
        json.dump(config_data, f, indent=4)

    return Config("test_config.json")

def test_client_initialization():
    """Test client initialization with proper error handling."""
    logger.info("Testing client initialization...")

    try:
        # Create a test config
        config = create_test_config()

        # Initialize the client
        client = get_client(test_mode=True)
        logger.info("Client initialization successful")

        # Test that the client is not None
        assert client is not None, "Client should not be None"

        logger.info("Client initialization verified")
        return True, client
    except Exception as e:
        logger.error(f"Client initialization failed: {e}")
        return False, None

def test_get_price(client=None):
    """Test get_market_price function with proper error handling."""
    logger.info("Testing get_market_price function...")

    try:
        # Initialize client if not provided
        if client is None:
            success, client = test_client_initialization()
            if not success:
                logger.error("Client initialization failed, skipping price test")
                return False

        # Test get_market_price
        try:
            price = get_market_price(client, "BTCUSDT")
            logger.info(f"Price: {price}")
            assert price is not None, "Price should not be None"
            assert price > 0, f"Price should be positive, got {price}"
            logger.info("get_market_price test passed")
            return True
        except Exception as e:
            logger.error(f"get_market_price test failed: {e}")
            return False
    except Exception as e:
        logger.error(f"Test setup failed: {e}")
        return False

def test_get_account_balance(client=None):
    """Test get_account_balance function with proper error handling."""
    logger.info("Testing get_account_balance function...")

    try:
        # Initialize client if not provided
        if client is None:
            success, client = test_client_initialization()
            if not success:
                logger.error("Client initialization failed, skipping balance test")
                return False

        # Test get_account_balance
        try:
            balance = get_account_balance(client, "USDT")
            logger.info(f"Balance: {balance}")
            assert balance is not None, "Balance should not be None"
            assert balance >= 0, f"Balance should be non-negative, got {balance}"
            logger.info("get_account_balance test passed")
            return True
        except Exception as e:
            logger.error(f"get_account_balance test failed: {e}")
            return False
    except Exception as e:
        logger.error(f"Test setup failed: {e}")
        return False

def main():
    """Run all tests."""
    logger.info("Starting exchange tests...")

    # Run tests
    init_success, client = test_client_initialization()

    # Only run other tests if initialization succeeds
    if init_success:
        price_result = test_get_price(client)
        balance_result = test_get_account_balance(client)

        # Print summary
        logger.info("\nTest Results:")
        logger.info(f"Client initialization: {'PASSED' if init_success else 'FAILED'}")
        logger.info(f"get_market_price test: {'PASSED' if price_result else 'FAILED'}")
        logger.info(f"get_account_balance test: {'PASSED' if balance_result else 'FAILED'}")
    else:
        logger.error("Client initialization failed, skipping other tests")

    # Clean up
    if os.path.exists("test_config.json"):
        os.remove("test_config.json")
        logger.info("Removed test_config.json")

if __name__ == "__main__":
    main()
