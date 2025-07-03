"""
Test script for the modular exchange structure.

This script verifies that all the modular exchange components work correctly
and can be imported from the exchange package.
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

# Import from the exchange package
from src.core.exchange import (
    # From exchange_base
    get_client,
    get_account_balance,
    get_total_balance,
    
    # From price_utils
    get_fresh_price,
    get_market_price,
    
    # From order_manager
    execute_trade,
    close_position,
    cancel_all_orders,
    
    # From risk_utils
    ensure_minimum_notional,
    place_stop_if_needed,
    
    # From position_utils
    get_position_amount,
    get_current_position_size,
    get_unrealized_pnl
)

def test_imports():
    """Test that all imports work correctly."""
    print("\nTesting imports...")
    
    # Check that all imported functions exist
    assert callable(get_client), "get_client should be callable"
    assert callable(get_account_balance), "get_account_balance should be callable"
    assert callable(get_total_balance), "get_total_balance should be callable"
    assert callable(get_fresh_price), "get_fresh_price should be callable"
    assert callable(get_market_price), "get_market_price should be callable"
    assert callable(execute_trade), "execute_trade should be callable"
    assert callable(close_position), "close_position should be callable"
    assert callable(cancel_all_orders), "cancel_all_orders should be callable"
    assert callable(ensure_minimum_notional), "ensure_minimum_notional should be callable"
    assert callable(place_stop_if_needed), "place_stop_if_needed should be callable"
    assert callable(get_position_amount), "get_position_amount should be callable"
    assert callable(get_current_position_size), "get_current_position_size should be callable"
    assert callable(get_unrealized_pnl), "get_unrealized_pnl should be callable"
    
    print("All imports work correctly!")

def test_client_initialization():
    """Test client initialization."""
    print("\nTesting client initialization...")
    
    # Initialize client
    client = get_client()
    assert client is not None, "Client should not be None"
    
    print("Client initialization works correctly!")

if __name__ == "__main__":
    print("Running exchange modules tests...")
    
    # Run the tests
    test_imports()
    test_client_initialization()
    
    print("\nAll tests passed! ✅")
