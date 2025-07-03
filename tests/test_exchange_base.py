"""
Test script for the exchange_base.py module.

This script verifies that the exchange_base.py module works correctly
by testing client initialization and balance fetching.
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

# Import the exchange_base functions
from src.core.exchange.exchange_base import (
    get_client,
    get_account_balance,
    get_total_balance
)

def test_client_initialization():
    """Test client initialization."""
    print("\nTesting client initialization...")
    
    # Initialize client
    client = get_client()
    assert client is not None, "Client should not be None"
    
    # Test ping
    try:
        ping_result = client.ping()
        print(f"Ping result: {ping_result}")
        assert ping_result == {}, "Ping should return an empty dict"
    except Exception as e:
        print(f"Ping failed: {e}")
        assert False, f"Ping should not fail: {e}"
    
    print("Client initialization test passed!")

def test_balance_fetching():
    """Test balance fetching."""
    print("\nTesting balance fetching...")
    
    # Initialize client
    client = get_client()
    
    # Test get_account_balance
    balance = get_account_balance(client)
    print(f"Account balance: {balance}")
    assert balance >= 0, "Balance should be non-negative"
    
    # Test get_total_balance
    total_balance = get_total_balance(client)
    print(f"Total balance: {total_balance}")
    assert total_balance >= 0, "Total balance should be non-negative"
    
    print("Balance fetching test passed!")

if __name__ == "__main__":
    print("Running exchange_base.py tests...")
    
    # Run the tests
    test_client_initialization()
    test_balance_fetching()
    
    print("\nAll tests passed! ✅")
