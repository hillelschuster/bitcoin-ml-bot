"""
Price utility functions for the Bitcoin ML Trading Bot.

This module provides functions for fetching, validating, and processing
price data from the Binance Futures API with comprehensive error handling
and fallback mechanisms.
"""

import logging
import random
import time
from functools import wraps

# Import from exchange_base
from .exchange_base import get_module_logger

# Setup module logger
logger = get_module_logger('price_utils')

def validate_price(price, symbol=None, raise_exception=False):
    """
    Validate that a price value is valid (positive and non-zero).

    Args:
        price (float): Price value to validate
        symbol (str, optional): Symbol for error messages
        raise_exception (bool): Whether to raise an exception for invalid prices

    Returns:
        bool: True if price is valid, False otherwise

    Raises:
        ValueError: If price is invalid and raise_exception is True
    """
    symbol_str = f" for {symbol}" if symbol else ""

    if price is None:
        error_msg = f"Price is None{symbol_str}"
        logger.error(f"[PRICE VALIDATION] {error_msg}")
        if raise_exception:
            raise ValueError(error_msg)
        return False

    if price <= 0:
        error_msg = f"Invalid price value: {price}{symbol_str}"
        logger.error(f"[PRICE VALIDATION] {error_msg}")
        if raise_exception:
            raise ValueError(error_msg)
        return False

    return True

def get_fresh_price(client, symbol):
    """
    Get a fresh price from multiple sources with fallback mechanisms.

    This function tries multiple methods to get the most recent price,
    falling back to alternative methods if the primary ones fail.

    Args:
        client: The exchange client
        symbol: Trading pair symbol

    Returns:
        float: Fresh price or None if all methods fail
    """
    try:
        # Method 1: Try mark price (most accurate for futures)
        if hasattr(client, 'mark_price'):
            try:
                price_data = client.mark_price(symbol=symbol)
                price = float(price_data["markPrice"])
                if validate_price(price):
                    logger.info(f"[PRICE] Got fresh price from mark_price: {price:.2f}")
                    return price
            except Exception as e:
                logger.warning(f"[PRICE] Failed to get mark price: {e}")

        # Method 2: Try ticker
        if hasattr(client, 'fetch_ticker'):
            try:
                ticker = client.fetch_ticker(symbol)
                if ticker and 'last' in ticker:
                    price = float(ticker['last'])
                    if validate_price(price):
                        logger.info(f"[PRICE] Got fresh price from ticker: {price:.2f}")
                        return price
            except Exception as e:
                logger.warning(f"[PRICE] Failed to get ticker price: {e}")

        # Method 3: Try orderbook
        if hasattr(client, 'fetch_order_book'):
            try:
                orderbook = client.fetch_order_book(symbol)
                if orderbook and 'asks' in orderbook and len(orderbook['asks']) > 0:
                    price = float(orderbook['asks'][0][0])
                    if validate_price(price):
                        logger.info(f"[PRICE] Got fresh price from orderbook: {price:.2f}")
                        return price
            except Exception as e:
                logger.warning(f"[PRICE] Failed to get orderbook price: {e}")

        # Method 4: Try recent trades
        if hasattr(client, 'trades'):
            try:
                trades = client.trades(symbol=symbol, limit=1)
                if trades and len(trades) > 0:
                    price = float(trades[0]['price'])
                    if validate_price(price):
                        logger.info(f"[PRICE] Got fresh price from recent trades: {price:.2f}")
                        return price
            except Exception as e:
                logger.warning(f"[PRICE] Failed to get recent trades price: {e}")

        # Method 5: Try klines (candlestick data)
        if hasattr(client, 'klines'):
            try:
                klines = client.klines(symbol=symbol, interval='1m', limit=1)
                if klines and len(klines) > 0:
                    price = float(klines[0][4])  # Close price is at index 4
                    if validate_price(price):
                        logger.info(f"[PRICE] Got fresh price from klines: {price:.2f}")
                        return price
            except Exception as e:
                logger.warning(f"[PRICE] Failed to get klines price: {e}")

        logger.error(f"[PRICE] All methods to get fresh price failed for {symbol}")
        return None

    except Exception as e:
        logger.error(f"[PRICE] Failed to get fresh price: {e}")
        return None

def get_fill_price(client, order, symbol):
    """
    Get the fill price from an order, with fallback to current price if needed.

    Args:
        client: The exchange client
        order: Order object from the exchange
        symbol: Trading pair symbol

    Returns:
        float: Fill price or 0 if all methods fail
    """
    fill_price = 0

    # Try to get price from order object
    if order and isinstance(order, dict):
        # Method 1: Direct price field
        if 'price' in order and order['price']:
            try:
                fill_price = float(order['price'])
                if validate_price(fill_price):
                    logger.info(f"[FILL PRICE] Using order price: {fill_price:.2f}")
                    return fill_price
            except (ValueError, TypeError) as e:
                logger.warning(f"[FILL PRICE] Invalid price in order: {e}")

        # Method 2: Average price field
        if 'average' in order and order['average']:
            try:
                fill_price = float(order['average'])
                if validate_price(fill_price):
                    logger.info(f"[FILL PRICE] Using order average price: {fill_price:.2f}")
                    return fill_price
            except (ValueError, TypeError) as e:
                logger.warning(f"[FILL PRICE] Invalid average price in order: {e}")

        # Method 3: avgPrice field (Binance-specific)
        if 'avgPrice' in order and order['avgPrice']:
            try:
                fill_price = float(order['avgPrice'])
                if validate_price(fill_price):
                    logger.info(f"[FILL PRICE] Using order avgPrice: {fill_price:.2f}")
                    return fill_price
            except (ValueError, TypeError) as e:
                logger.warning(f"[FILL PRICE] Invalid avgPrice in order: {e}")

    # If no valid price found in order, get current price as fallback
    if not validate_price(fill_price):
        try:
            fill_price = get_fresh_price(client, symbol)
            if validate_price(fill_price):
                logger.warning(f"[FILL PRICE] Order response missing price. Using current price: {fill_price:.2f}")
                return fill_price
        except Exception as e:
            logger.error(f"[FILL PRICE] Failed to get fallback price: {e}")

    # If all methods fail, return 0 (caller should handle this case)
    if not validate_price(fill_price):
        logger.error(f"[FILL PRICE] Could not determine fill price for order")
        return 0

    return fill_price

def get_market_price(client, symbol, test_mode=False):
    """
    Get current market price with proper error handling and validation.

    Args:
        client: The exchange client
        symbol: Trading pair symbol
        test_mode (bool): Whether the bot is running in test mode

    Returns:
        float: Current market price

    Raises:
        Exception: If price fetch fails and test_mode is False
    """
    # Ensure symbol is in the correct format (no slashes)
    symbol = symbol.replace("/", "") if "/" in symbol else symbol

    # Get fresh price using our utility function
    price = get_fresh_price(client, symbol)

    # Validate price
    if not validate_price(price):
        # If in test mode, return a default price
        if test_mode:
            default_price = 50000.0
            logger.warning(f"[PRICE] Test mode - Using default price: ${default_price:.2f}")
            return default_price

        # If not in test mode, raise an exception - no fake fallbacks
        error_msg = f"[PRICE ERROR] CRITICAL: All price fetch methods failed for {symbol}"
        logger.critical(error_msg)
        raise Exception(error_msg)

    logger.info(f"[PRICE] Market price for {symbol}: ${price:.2f}")
    return price

# Export the functions
__all__ = ['validate_price', 'get_fresh_price', 'get_fill_price', 'get_market_price']
