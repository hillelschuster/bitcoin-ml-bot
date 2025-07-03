"""
Position utilities for tracking open size, direction, and PnL on Binance.
"""

import logging
from src.core.exchange.exchange_base import get_client, get_module_logger

logger = get_module_logger("position_utils")


def get_position_amount(client, symbol="BTCUSDT"):
    """
    Get the current position amount for a symbol.

    Args:
        client: Binance UMFutures client
        symbol: Trading pair symbol (default: "BTCUSDT")

    Returns:
        float: Position amount (positive for long, negative for short, 0 for no position)
    """
    try:
        # Ensure symbol format
        symbol = symbol.replace("/", "")

        # Get position information
        try:
            positions = client.get_position_risk(symbol=symbol)
        except Exception as e:
            # If there's a timestamp error, try using account information instead
            if "Timestamp" in str(e):
                logger.warning(f"[POSITION] Timestamp error, using account information instead")
                account_info = client.account()
                positions = account_info.get('positions', [])
            else:
                raise e

        # Find the position for the specified symbol
        for position in positions:
            if position['symbol'] == symbol:
                # Convert position amount to float
                position_amount = float(position['positionAmt'])
                return position_amount

        # If no position found for the symbol
        return 0.0
    except Exception as e:
        logger.error(f"[POSITION] Failed to get size: {e}")
        return 0.0


def get_current_position_size(client, symbol="BTCUSDT"):
    """
    Get the current position size (absolute value) for a symbol.

    Args:
        client: Binance UMFutures client
        symbol: Trading pair symbol (default: "BTCUSDT")

    Returns:
        float: Position size (absolute value)
    """
    try:
        # Get position amount
        position_amount = get_position_amount(client, symbol)

        # Return absolute value
        return abs(position_amount)
    except Exception as e:
        logger.error(f"[POSITION] Failed to get size: {e}")
        return 0.0


def get_unrealized_pnl(client, symbol="BTCUSDT"):
    """
    Get the unrealized PnL for a symbol.

    Args:
        client: Binance UMFutures client
        symbol: Trading pair symbol (default: "BTCUSDT")

    Returns:
        float: Unrealized PnL
    """
    try:
        # Ensure symbol format
        symbol = symbol.replace("/", "")

        # Get position information
        try:
            positions = client.get_position_risk(symbol=symbol)
        except Exception as e:
            # If there's a timestamp error, try using account information instead
            if "Timestamp" in str(e):
                logger.warning(f"[PNL] Timestamp error, using account information instead")
                account_info = client.account()
                positions = account_info.get('positions', [])
            else:
                raise e

        # Find the position for the specified symbol
        for position in positions:
            if position['symbol'] == symbol:
                # Convert unrealized PnL to float
                unrealized_pnl = float(position.get('unRealizedProfit', 0.0))
                return unrealized_pnl

        # If no position found for the symbol
        return 0.0
    except Exception as e:
        logger.error(f"[PNL] Failed to get unrealized PnL: {e}")
        return 0.0
