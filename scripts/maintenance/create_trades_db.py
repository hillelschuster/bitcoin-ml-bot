#!/usr/bin/env python
"""
Script to create the main trades.db file.

This script creates the main trades.db file in the logs directory
and logs a sample trade to verify that the database is working correctly.
"""

import os
import sys
import logging

# Add the parent directory to the path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the TradeLogger class
from src.utils.logger import TradeLogger, TradeType, TradeSide, PositionSide

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('create_trades_db')

def create_trades_db():
    """Create the main trades.db file and log a sample trade."""
    # Create a logger instance with the main database
    trade_logger = TradeLogger()
    logger.info(f"Created TradeLogger with main database at {trade_logger.db_path}")
    
    # Log a sample trade
    logger.info("Logging sample trade...")
    success = trade_logger.log_trade(
        symbol="BTCUSDT",
        price=50000.0,
        quantity=0.01,
        side=TradeSide.BUY.value,
        position_side=PositionSide.LONG.value,
        type=TradeType.OPEN.value,
        confidence=0.75,
        entry_price=50000.0,
        model_used="LSTM v1.0",
        reason="Sample trade for database initialization"
    )
    
    if success:
        logger.info("Sample trade logged successfully")
    else:
        logger.error("Failed to log sample trade")
        return False
    
    logger.info(f"Main trades database created at {trade_logger.db_path}")
    return True

if __name__ == "__main__":
    logger.info("Creating main trades database...")
    
    if create_trades_db():
        logger.info("✅ Main trades database created successfully!")
        sys.exit(0)
    else:
        logger.error("❌ Failed to create main trades database!")
        sys.exit(1)
