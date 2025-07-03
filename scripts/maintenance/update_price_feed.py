#!/usr/bin/env python
"""
Script to update the live_price_feed.csv file with the latest price data.
This script should be run regularly (e.g., every minute) to keep the data up-to-date.
"""
import pandas as pd
import time
import os
import logging
import sys

# Add the root directory to the Python path to allow imports from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.core.exchange.exchange_base import get_client
from src.utils.config import Config
from src.utils.data import fetch_live_ohlcv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('price_feed')

def update_price_feed():
    """Update the live_price_feed.csv file with the latest price data."""
    try:
        # Initialize exchange client
        config = Config('config/config.json')
        client = get_client(test_mode=config.get('test_mode', True))

        # Get historical data (last 200 candles)
        df = fetch_live_ohlcv(symbol="BTCUSDT", timeframe='1m', limit=200, client=client)

        # Ensure data directory exists
        os.makedirs('data', exist_ok=True)

        # Save to CSV
        csv_path = 'data/price_feed.csv'
        df.to_csv(csv_path, index=True)

        logger.info(f"Updated {csv_path} with {len(df)} candles. Latest price: {df['close'].iloc[-1]:.2f}")
        return True
    except Exception as e:
        logger.error(f"Error updating price feed: {e}")
        return False

if __name__ == "__main__":
    update_price_feed()
