"""
Risk and stop-loss management utilities for Binance trades.
"""

import time
import json
import logging
from src.core.exchange.exchange_base import get_client, get_module_logger
from src.core.exchange.price_utils import get_fresh_price

logger = get_module_logger("risk_utils")


def ensure_minimum_notional(exchange, symbol, amount):
    """Ensure order meets exchange minimum notional requirements."""
    try:
        market = exchange.market(symbol)
        precision = market['precision']['amount']
        min_notional = market.get('limits', {}).get('cost', {}).get('min', 200.0)

        price = get_fresh_price(exchange, symbol)
        notional = amount * price

        if notional < min_notional:
            adjusted = round(min_notional / price, precision)
            logger.warning(f"Amount too small for {symbol}. Adjusted from {amount:.4f} → {adjusted:.4f}")
            return adjusted

        return round(amount, precision)
    except Exception as e:
        logger.error(f"[ensure_minimum_notional] Failed for {symbol}: {e}")
        return amount


def place_stop_loss_once(exchange, symbol, stop_price, amount, current_price=None):
    """Place a reduce-only stop-market order. Cancels any existing SLs."""
    try:
        client_order_id = f"sl_{int(time.time())}"

        if not current_price:
            current_price = get_fresh_price(exchange, symbol)

        if not current_price or current_price <= 0:
            raise ValueError(f"[STOP-LOSS] Invalid current price: {current_price}")

        stop_loss_pct = abs((stop_price - current_price) / current_price)
        if stop_loss_pct > 0.2:
            logger.error("[STOP-LOSS] Stop loss too wide. Adjusting to 2% fallback.")
            stop_price = current_price * 0.98

        open_orders = exchange.fetch_open_orders(symbol)
        for o in open_orders:
            if 'STOP' in o['type'] and o['side'] == 'SELL':
                exchange.cancel_order(o['id'], symbol)
        time.sleep(1)

        params = {
            'stopPrice': stop_price,
            'reduceOnly': True,
            'timeInForce': 'GTC',
            'clientOrderId': client_order_id
        }

        stop_order = exchange.create_order(symbol=symbol, type='stop_market', side='sell', amount=amount, params=params)
        logger.info(f"[STOP-LOSS] Placed stop-market: {stop_order}")
        return stop_order["id"]
    except Exception as e:
        logger.error(f"[STOP-LOSS] Failed to place SL: {e}")
        return None


def place_stop_if_needed(exchange, symbol, side, stop_loss, amount, fill_price):
    """Call stop-loss logic only for long positions with valid inputs."""
    if side != "buy" or not stop_loss or not fill_price:
        logger.info("[STOP-LOSS] Skipped: invalid side or price.")
        return None

    return place_stop_loss_once(exchange, symbol, stop_loss, amount, fill_price)


def is_stop_loss_triggered(reason):
    """Heuristic check for SL trigger."""
    return reason and "stop" in reason.lower()
