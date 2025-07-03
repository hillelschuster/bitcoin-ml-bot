"""
Exchange module for the Bitcoin ML Trading Bot.

This package provides a modular interface to the Binance Futures API,
with comprehensive error handling, retry logic, and safety measures.
"""

# Re-export price utility functions
from .price_utils import (
    validate_price,
    get_fresh_price,
    get_fill_price,
    get_market_price
)

# Exchange class has been fully modularized
# No need to import the monolithic Exchange class anymore

# Re-export exchange_base functions
from .exchange_base import (
    get_module_logger,
    get_client,
    get_account_balance,
    get_total_balance
)

# Re-export order_manager functions
from .order_manager import (
    execute_market_order,
    execute_trade,
    close_position,
    cancel_all_orders,
    get_open_orders
)

# Re-export risk_utils functions
from .risk_utils import (
    ensure_minimum_notional,
    place_stop_loss_once,
    place_stop_if_needed,
    is_stop_loss_triggered
)

# Re-export position_utils functions
from .position_utils import (
    get_position_amount,
    get_current_position_size,
    get_unrealized_pnl
)
