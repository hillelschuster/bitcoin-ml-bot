"""
Exchange base client initialization and config helpers.
"""

import logging
from binance.um_futures import UMFutures
from config.Config import BINANCE_API_KEY, BINANCE_API_SECRET, TEST_MODE

logger = logging.getLogger('exchange.exchange_base')


def get_module_logger(module_name):
    """
    Get a consistently configured logger for exchange modules.

    Args:
        module_name (str): Name of the module (without the 'exchange.' prefix)

    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger(f'exchange.{module_name}')
    return logger


def get_client(test_mode=TEST_MODE, api_key=BINANCE_API_KEY, api_secret=BINANCE_API_SECRET):
    """Return a configured Binance UMFutures client."""
    base_url = "https://testnet.binancefuture.com" if test_mode else None
    try:
        client = UMFutures(key=api_key, secret=api_secret, base_url=base_url)
        logger.info(f"[EXCHANGE_BASE] Client initialized (test={test_mode})")
        return client
    except Exception as e:
        logger.error(f"[EXCHANGE_BASE] Failed to create client: {e}")
        raise


def get_account_balance(client, asset="USDT"):
    """Return the available balance for the given asset."""
    try:
        balances = client.balance()
        for balance in balances:
            if balance["asset"] == asset:
                return float(balance["balance"])
    except Exception as e:
        logger.error(f"[EXCHANGE_BASE] Failed to fetch account balance: {e}")
    return 0.0


def get_total_balance(client, asset="USDT"):
    """Return the total wallet balance (cross) for a given asset."""
    try:
        balances = client.account()
        for balance in balances['assets']:
            if balance['asset'] == asset:
                return float(balance["walletBalance"])
    except Exception as e:
        logger.error(f"[EXCHANGE_BASE] Failed to fetch total balance: {e}")
    return 0.0
