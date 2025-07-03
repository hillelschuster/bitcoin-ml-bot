#!/usr/bin/env python
"""
Test script for the TradeLogger class.

This script tests the trade logging functionality by:
1. Creating a test database
2. Logging a test trade
3. Retrieving the trade from the database
4. Verifying that all fields are correctly stored
"""

import os
import sys
import sqlite3
import logging
from datetime import datetime

# Add the parent directory to the path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the TradeLogger class
from src.utils.logger import TradeLogger, TradeType, TradeSide, PositionSide

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('test_trade_logger')

def test_trade_logger():
    """Test the TradeLogger class."""
    # Create a test database path
    test_db_path = "logs/test_trades.db"
    
    # Delete the test database if it exists
    if os.path.exists(test_db_path):
        os.remove(test_db_path)
        logger.info(f"Deleted existing test database: {test_db_path}")
    
    # Create a logger instance with the test database
    trade_logger = TradeLogger(db_path=test_db_path)
    logger.info(f"Created TradeLogger with test database: {test_db_path}")
    
    # Test data for a trade
    test_trade = {
        "symbol": "BTCUSDT",
        "price": 50000.0,
        "quantity": 0.01,
        "side": TradeSide.BUY.value,
        "position_side": PositionSide.LONG.value,
        "type": TradeType.OPEN.value,
        "confidence": 0.75,
        "entry_price": 50000.0,
        "model_used": "LSTM v1.0",
        "reason": "Strong uptrend detected"
    }
    
    # Log the test trade
    logger.info("Logging test trade...")
    success = trade_logger.log_trade(**test_trade)
    
    if not success:
        logger.error("Failed to log test trade")
        return False
    
    logger.info("Test trade logged successfully")
    
    # Verify the trade was logged correctly
    logger.info("Verifying trade was logged correctly...")
    
    # Connect to the database directly
    conn = sqlite3.connect(test_db_path)
    c = conn.cursor()
    
    # Get the trade from the database
    c.execute("SELECT * FROM trades")
    rows = c.fetchall()
    
    # Get column names
    c.execute("PRAGMA table_info(trades)")
    columns = [col[1] for col in c.fetchall()]
    
    # Check if we have exactly one trade
    if len(rows) != 1:
        logger.error(f"Expected 1 trade, found {len(rows)}")
        conn.close()
        return False
    
    # Convert row to dictionary
    trade = {}
    for i, col in enumerate(columns):
        trade[col] = rows[0][i]
    
    # Verify key fields
    logger.info("Verifying key fields...")
    
    # Check symbol
    if trade.get('symbol') != test_trade['symbol']:
        logger.error(f"Symbol mismatch: expected {test_trade['symbol']}, got {trade.get('symbol')}")
        conn.close()
        return False
    
    # Check price
    if float(trade.get('price')) != test_trade['price']:
        logger.error(f"Price mismatch: expected {test_trade['price']}, got {trade.get('price')}")
        conn.close()
        return False
    
    # Check quantity
    if float(trade.get('quantity')) != test_trade['quantity']:
        logger.error(f"Quantity mismatch: expected {test_trade['quantity']}, got {trade.get('quantity')}")
        conn.close()
        return False
    
    # Check side
    if trade.get('side') != test_trade['side']:
        logger.error(f"Side mismatch: expected {test_trade['side']}, got {trade.get('side')}")
        conn.close()
        return False
    
    # Check position_side
    if trade.get('position_side') != test_trade['position_side']:
        logger.error(f"Position side mismatch: expected {test_trade['position_side']}, got {trade.get('position_side')}")
        conn.close()
        return False
    
    # Check type
    if trade.get('type') != test_trade['type']:
        logger.error(f"Type mismatch: expected {test_trade['type']}, got {trade.get('type')}")
        conn.close()
        return False
    
    # Check confidence
    if float(trade.get('confidence')) != test_trade['confidence']:
        logger.error(f"Confidence mismatch: expected {test_trade['confidence']}, got {trade.get('confidence')}")
        conn.close()
        return False
    
    # Close the connection
    conn.close()
    
    logger.info("All fields verified successfully")
    logger.info("Trade logging test passed!")
    
    return True

if __name__ == "__main__":
    logger.info("Starting trade logger test...")
    
    if test_trade_logger():
        logger.info("✅ Trade logger test passed!")
        sys.exit(0)
    else:
        logger.error("❌ Trade logger test failed!")
        sys.exit(1)
