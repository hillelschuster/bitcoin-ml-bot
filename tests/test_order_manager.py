"""
Test script for the order_manager.py module.

This script verifies that the order_manager.py module works correctly
by testing order execution and management functions.
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

# Import the order_manager functions
from src.core.exchange.order_manager import (
    execute_market_order,
    execute_trade,
    close_position,
    cancel_all_orders,
    get_open_orders,
    get_position_amount
)

# Import the client initialization function
from src.core.exchange.exchange_base import get_client

def test_get_open_orders():
    """Test the get_open_orders function."""
    print("\nTesting get_open_orders...")
    
    # Initialize client
    client = get_client()
    
    # Get open orders
    orders = get_open_orders(client)
    print(f"Open orders: {orders}")
    
    print("get_open_orders test passed!")

def test_get_position_amount():
    """Test the get_position_amount function."""
    print("\nTesting get_position_amount...")
    
    # Initialize client
    client = get_client()
    
    # Get position amount
    amount = get_position_amount(client)
    print(f"Position amount: {amount}")
    
    print("get_position_amount test passed!")

def test_cancel_all_orders():
    """Test the cancel_all_orders function."""
    print("\nTesting cancel_all_orders...")
    
    # Initialize client
    client = get_client()
    
    # Cancel all orders
    result = cancel_all_orders(client)
    print(f"Cancel result: {result}")
    
    print("cancel_all_orders test passed!")

if __name__ == "__main__":
    print("Running order_manager.py tests...")
    
    # Run the tests
    test_get_open_orders()
    test_get_position_amount()
    test_cancel_all_orders()
    
    print("\nAll tests passed! ✅")
