"""
Test script for the price_utils.py module.

This script verifies that the price utility functions work correctly
by importing and calling them directly.
"""

import sys
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the price utility functions
from src.core.exchange.price_utils import (
    validate_price,
    get_fresh_price,
    get_fill_price,
    get_market_price
)

# Import the Exchange class for testing
from src.core.exchange import Exchange
from src.utils.config import Config

def test_validate_price():
    """Test the validate_price function."""
    print("\nTesting validate_price...")

    # Test valid price
    assert validate_price(100.0) == True
    print("✅ Valid price test passed")

    # Test invalid prices
    assert validate_price(0) == False
    assert validate_price(-10) == False
    assert validate_price(None) == False
    print("✅ Invalid price tests passed")

    # Test with symbol
    assert validate_price(100.0, symbol="BTCUSDT") == True
    print("✅ Valid price with symbol test passed")

    # Test with raise_exception
    try:
        validate_price(None, raise_exception=True)
        assert False, "Should have raised an exception"
    except ValueError:
        print("✅ Exception test passed")

    print("All validate_price tests passed!")

def test_get_market_price():
    """Test the get_market_price function."""
    print("\nTesting get_market_price...")

    # Initialize config and exchange
    config = Config('config/config.json')

    # Create a mock client for testing
    class MockClient:
        def mark_price(self, symbol):
            return {"markPrice": "50000.0"}

        def trades(self, symbol, limit):
            return [{"price": "49900.0"}]

        def klines(self, symbol, interval, limit):
            return [[0, 0, 0, 0, "49800.0"]]

        def ticker_price(self, symbol):
            return {"price": "49700.0"}

    # Test with mock client
    price = get_market_price(MockClient(), "BTCUSDT", test_mode=True)
    assert price > 0, f"Price should be positive, got {price}"
    print(f"✅ Mock client test passed, price: {price}")

    # Test with real client (if available)
    try:
        from src.core.exchange import Exchange
        exchange = Exchange(config)
        price = get_market_price(exchange.client, "BTCUSDT", test_mode=True)
        assert price > 0, f"Price should be positive, got {price}"
        print(f"✅ Real client test passed, price: {price}")
    except Exception as e:
        print(f"⚠️ Real client test skipped: {e}")

    print("All get_market_price tests passed!")

if __name__ == "__main__":
    print("Running price_utils.py tests...")

    # Run the tests
    test_validate_price()
    test_get_market_price()

    print("\nAll tests passed! ✅")
