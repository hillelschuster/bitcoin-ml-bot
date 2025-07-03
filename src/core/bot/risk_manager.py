"""
Risk management module for the Bitcoin ML Trading Bot.
Handles position sizing, trade timing, and risk adjustments.
"""
import time
import logging
from src.core.utils.sentiment_utils import fetch_sentiment_score, get_sentiment_adjustment_factor
from config.Config import RISK_PER_TRADE

# Configure logging
logger = logging.getLogger('core.bot.risk_manager')

# Configuration is now loaded from config.Config

class RiskManager:
    """
    Risk management class for the Bitcoin ML Trading Bot.
    Handles position sizing, trade timing, and risk adjustments.
    """

    def __init__(self, exchange=None, trade_logger=None, config=None):
        """
        Initialize the risk manager.

        Args:
            exchange: Exchange instance for market data and account info
            trade_logger: TradeLogger instance for accessing trade history
            config: Configuration dictionary
        """
        self.exchange = exchange
        self.trade_logger = trade_logger
        self.config = config or {}
        self.last_trade_time = 0
        self.last_trade_confidence = 0.0

        # Get risk management parameters from config
        trading_config = self.config.get('trading', {})

        # Risk parameters
        self.base_risk_pct = trading_config.get('base_risk_pct', RISK_PER_TRADE)  # Default from Config
        self.max_consecutive_losses = trading_config.get('max_consecutive_losses', 3)
        self.drawdown_threshold = trading_config.get('drawdown_threshold', 0.05)
        self.leverage = trading_config.get('leverage', 20)  # Default leverage on Binance Futures

        logger.info(f"Risk manager initialized with base risk: {self.base_risk_pct*100:.2f}%")

    # ===== Cooldown Management =====

    def get_adaptive_cooldown(self, confidence):
        """
        Return cooldown period in seconds based on confidence level.

        Args:
            confidence (float): Prediction confidence (0.0-1.0)

        Returns:
            int: Cooldown period in seconds
        """
        if confidence >= 0.95:
            return 30  # Very high confidence = short cooldown
        elif confidence >= 0.85:
            return 60  # High confidence = medium cooldown
        elif confidence >= 0.75:
            return 90  # Medium confidence = longer cooldown
        else:
            return 120  # Lower confidence = longest cooldown

    def can_trade(self, confidence=None, override=False):
        """
        Determine if enough time has passed since the last trade.
        Uses adaptive cooldown based on the confidence of the last trade.

        Args:
            confidence (float): Current prediction confidence (optional, for logging)
            override (bool): If True, skips cooldown checks entirely

        Returns:
            tuple: (can_trade, reason)
                can_trade: True if trading is allowed, False otherwise
                reason: Reason for not trading if can_trade is False
        """
        # Skip cooldown check if override is True
        if override:
            logger.info("[COOLDOWN OVERRIDE] Skipping cooldown due to strong signal.")
            return True, ""

        # Get the adaptive cooldown based on the last trade's confidence
        cooldown_seconds = self.get_adaptive_cooldown(self.last_trade_confidence)

        # Check if enough time has passed
        time_since_last_trade = time.time() - self.last_trade_time
        can_trade = time_since_last_trade > cooldown_seconds

        if not can_trade:
            remaining = cooldown_seconds - time_since_last_trade
            reason = f"Cooldown period: {remaining:.1f} seconds remaining"
            if confidence is not None:
                logger.info(f"[COOLDOWN] Confidence: {confidence:.4f} → {reason}")
            else:
                logger.info(f"[COOLDOWN] {reason} (last confidence: {self.last_trade_confidence:.4f})")
            return False, reason

        return True, ""

    def record_trade_time(self, confidence=0.75):
        """
        Record the current time and confidence as the last trade time.

        Args:
            confidence (float): Prediction confidence of the trade (0.0-1.0)
        """
        self.last_trade_time = time.time()
        self.last_trade_confidence = confidence

        # Calculate the adaptive cooldown for this confidence level
        cooldown_seconds = self.get_adaptive_cooldown(confidence)

        logger.info(f"[COOLDOWN] Confidence: {confidence:.4f} → Cooldown set to {cooldown_seconds} seconds")
        logger.info(f"[COOLDOWN] Trade time recorded: {self.last_trade_time}")

    # ===== Risk Checks =====

    def check_consecutive_losses(self):
        """
        Check for consecutive losses.

        Returns:
            tuple: (can_trade, reason)
                can_trade: True if risk management rules allow trading, False otherwise
                reason: Reason for not trading if can_trade is False
        """
        # If no trade logger is available, allow trading
        if self.trade_logger is None:
            return True, ""

        # Get recent PnLs
        recent_pnls = self.trade_logger.get_recent_pnls(10)

        # If no recent trades, allow trading
        if not recent_pnls:
            return True, ""

        # Check for consecutive losses
        consecutive_losses = 0
        for pnl in recent_pnls:
            if pnl < 0:
                consecutive_losses += 1
            else:
                break

        if consecutive_losses >= self.max_consecutive_losses:
            reason = f"Too many consecutive losses ({consecutive_losses})"
            logger.warning(f"[RISK CHECK] {reason}")
            return False, reason

        return True, ""

    def check_drawdown(self):
        """
        Check for excessive drawdown.

        Returns:
            tuple: (can_trade, reason)
                can_trade: True if risk management rules allow trading, False otherwise
                reason: Reason for not trading if can_trade is False
        """
        # If no trade logger or exchange is available, allow trading
        if self.trade_logger is None or self.exchange is None:
            return True, ""

        # Get recent PnLs
        recent_pnls = self.trade_logger.get_recent_pnls(10)

        # If no recent trades, allow trading
        if not recent_pnls:
            return True, ""

        # Check for drawdown
        total_pnl = sum(recent_pnls)

        try:
            # Get current balance
            balance = self.exchange.get_balance()

            # Calculate initial balance before recent trades
            initial_balance = balance - total_pnl

            if initial_balance > 0:
                drawdown = abs(total_pnl) / initial_balance if total_pnl < 0 else 0
                if drawdown > self.drawdown_threshold:
                    reason = f"Drawdown threshold exceeded ({drawdown:.2%})"
                    logger.warning(f"[RISK CHECK] {reason}")
                    return False, reason
        except (ValueError, KeyError, AttributeError) as e:
            logger.error(f"[RISK CHECK] Error calculating drawdown: {e}")

        return True, ""

    def can_trade_risk_check(self, confidence=None, override=False):
        """
        Perform all risk checks to determine if trading is allowed.

        Args:
            confidence (float): Current prediction confidence (optional)
            override (bool): If True, skips cooldown checks but still performs other risk checks

        Returns:
            tuple: (can_trade, reason)
                can_trade: True if all risk checks pass, False otherwise
                reason: Reason for not trading if can_trade is False
        """
        # Check cooldown (can be overridden)
        if not self.can_trade(confidence, override):
            return False, "Trade cooldown in effect"

        # Check consecutive losses (cannot be overridden)
        can_trade, reason = self.check_consecutive_losses()
        if not can_trade:
            return False, reason

        # Check drawdown (cannot be overridden)
        can_trade, reason = self.check_drawdown()
        if not can_trade:
            return False, reason

        return True, ""

    def evaluate_position(self, position_data, current_price=None):
        """
        Evaluate an existing position to determine if it should be closed.

        Args:
            position_data (dict): Position data including entry price, size, etc.
            current_price (float, optional): Current market price. If None, will fetch from exchange.

        Returns:
            tuple: (should_close, reason)
                should_close: True if position should be closed, False otherwise
                reason: Reason for closing if should_close is True
        """
        try:
            # If no position data, nothing to evaluate
            if not position_data or not isinstance(position_data, dict):
                return False, "No position data provided"

            # Get current price if not provided
            if current_price is None and self.exchange is not None:
                try:
                    current_price = self.exchange.get_price()
                except (ValueError, KeyError, AttributeError) as e:
                    logger.error(f"[POSITION EVAL] Error getting current price: {e}")
                    return False, "Could not get current price"

            # If still no price, can't evaluate
            if not current_price or current_price <= 0:
                return False, "Invalid current price"

            # Get position details
            entry_price = position_data.get('entry_price')
            position_size = position_data.get('size')
            stop_loss = position_data.get('stop_loss')
            take_profit = position_data.get('take_profit')

            if not entry_price or not position_size:
                return False, "Incomplete position data"

            # Calculate unrealized PnL
            pnl_pct = (current_price - entry_price) / entry_price

            # Check stop loss
            if stop_loss and current_price <= stop_loss:
                return True, f"Stop loss triggered: {current_price:.2f} <= {stop_loss:.2f}"

            # Check take profit
            if take_profit and current_price >= take_profit:
                return True, f"Take profit reached: {current_price:.2f} >= {take_profit:.2f}"

            # Check max drawdown
            if pnl_pct <= -self.drawdown_threshold:
                return True, f"Max drawdown exceeded: {pnl_pct:.2%} <= -{self.drawdown_threshold:.2%}"

            # Position is still valid
            return False, ""

        except (ValueError, KeyError, AttributeError) as e:
            logger.error(f"[POSITION EVAL] Error evaluating position: {e}")
            return False, f"Evaluation error: {str(e)}"

    # ===== Position Sizing =====

    def calculate_position_size(self, confidence, balance=None, price=None):
        """
        Calculate position size based on account balance, confidence, and risk parameters.
        Includes all necessary validations and caps in a single function.

        Args:
            confidence (float): Prediction confidence (0.0-1.0)
            balance (float, optional): Account balance in USDT. If None, will attempt to fetch from exchange.
            price (float, optional): Current price. If None, will attempt to fetch from exchange.

        Returns:
            float: Position size in base currency (e.g., BTC)

        Raises:
            RuntimeError: If balance or price cannot be fetched
        """
        # Log the start of position sizing with all inputs
        logger.info(f"[POSITION SIZING] Starting calculation with confidence={confidence:.4f}")

        # Get account balance if not provided with retry logic
        if balance is None and self.exchange is not None:
            for attempt in range(3):
                try:
                    balance = self.exchange.get_balance()
                    if balance and balance > 0:
                        break
                    logger.warning(f"[POSITION SIZING] Invalid balance returned: {balance} (attempt {attempt+1}/3)")
                except (RuntimeError, ValueError) as e:
                    logger.error(f"[POSITION SIZING] Error getting balance (attempt {attempt+1}/3): {e}")

                # Don't sleep on the last attempt
                if attempt < 2:
                    sleep_time = 2 ** attempt
                    logger.info(f"[POSITION SIZING] Retrying balance fetch in {sleep_time}s...")
                    time.sleep(sleep_time)
            else:
                # If we get here, all attempts failed
                error_msg = "[POSITION SIZING] Failed to fetch valid balance after 3 attempts"
                logger.critical(error_msg)
                raise RuntimeError("Cannot fetch balance – aborting trade.")

        # Verify balance is valid (no fallbacks in live trading)
        if not balance or balance <= 0:
            error_msg = f"[POSITION SIZING] Invalid balance: {balance}"
            logger.critical(error_msg)
            raise RuntimeError("Invalid balance – aborting trade.")

        logger.info(f"[POSITION SIZING] Account balance: ${balance:.2f}")

        # Get current price if not provided with retry logic
        if price is None and self.exchange is not None:
            for attempt in range(3):
                try:
                    price = self.exchange.get_price()
                    if price and price > 0:
                        break
                    logger.warning(f"[POSITION SIZING] Invalid price returned: {price} (attempt {attempt+1}/3)")
                except (RuntimeError, ValueError) as e:
                    logger.error(f"[POSITION SIZING] Error getting price (attempt {attempt+1}/3): {e}")

                # Don't sleep on the last attempt
                if attempt < 2:
                    sleep_time = 2 ** attempt
                    logger.info(f"[POSITION SIZING] Retrying price fetch in {sleep_time}s...")
                    time.sleep(sleep_time)
            else:
                # If we get here, all attempts failed
                error_msg = "[POSITION SIZING] Failed to fetch valid price after 3 attempts"
                logger.critical(error_msg)
                raise RuntimeError("Cannot fetch price – aborting trade.")

        # Verify price is valid (no fallbacks in live trading)
        if not price or price <= 0:
            error_msg = f"[POSITION SIZING] Invalid price: {price}"
            logger.critical(error_msg)
            raise RuntimeError("Invalid price – aborting trade.")

        logger.info(f"[POSITION SIZING] Current BTC price: ${price:.2f}")

        # Determine risk percentage based on confidence level
        if confidence >= 0.9:  # Very high confidence
            risk_pct = 0.05  # 5% of balance
        elif confidence >= 0.8:  # High confidence
            risk_pct = 0.04  # 4% of balance
        elif confidence >= 0.7:  # Medium-high confidence
            risk_pct = 0.03  # 3% of balance
        else:  # Lower confidence
            risk_pct = 0.02  # 2% of balance

        # Calculate margin amount directly from balance
        margin_amount = balance * risk_pct
        logger.info(f"[POSITION SIZING] Direct margin calculation: {risk_pct:.2%} of ${balance:.2f} = ${margin_amount:.2f}")

        # Apply sentiment adjustment if available
        try:
            sentiment_adjustment = get_sentiment_adjustment_factor()
            logger.info(f"[POSITION SIZING] Sentiment adjustment: {sentiment_adjustment:.2f}")
            margin_amount = margin_amount * sentiment_adjustment
            logger.info(f"[POSITION SIZING] After sentiment: ${margin_amount:.2f}")
        except (ValueError, KeyError, AttributeError) as e:
            logger.warning(f"[POSITION SIZING] Error getting sentiment adjustment: {e}. Using unadjusted amount.")
            sentiment_adjustment = 1.0

        # Apply absolute maximum cap (10% of balance)
        max_margin = balance * 0.10  # Maximum 10% of balance as margin
        if margin_amount > max_margin:
            logger.warning(f"[POSITION SIZING] Margin ${margin_amount:.2f} exceeds 10% cap (${max_margin:.2f}). Capping.")
            margin_amount = max_margin

        # Calculate BTC quantity based on margin and current price
        notional_value = margin_amount * self.leverage
        quantity = round(notional_value / price, 5)  # 5 decimal places for BTC
        logger.info(f"[POSITION SIZING] Margin: ${margin_amount:.2f}, Leverage: {self.leverage}x, Notional: ${notional_value:.2f}")
        logger.info(f"[POSITION SIZING] Initial quantity: {quantity:.6f} BTC (${quantity * price:.2f})")

        # Ensure minimum notional value of $100 (Binance requirement)
        min_notional = 100.0
        min_btc_quantity = min_notional / price if price > 0 else 0.002

        if quantity < min_btc_quantity:
            logger.warning(f"[POSITION SIZING] Quantity {quantity:.6f} BTC below min notional (${min_notional}). Increasing to {min_btc_quantity:.6f} BTC.")
            quantity = round(min_btc_quantity, 5)

        # Final position size logging
        final_value = quantity * price
        pct_of_balance = (final_value / balance) * 100 if balance > 0 else 0

        # Get actual sentiment score if possible
        try:
            sentiment_score = fetch_sentiment_score()
        except (ValueError, KeyError, AttributeError):
            sentiment_score = 50.0  # Default value if not available

        # Log comprehensive position sizing details
        logger.info(f"[POSITION SIZING] FINAL: {quantity:.6f} BTC (${final_value:.2f}, {pct_of_balance:.2f}% of balance)")
        logger.info(f"[POSITION SIZING] Amount=${final_value:.2f}, Qty={quantity:.6f}, Balance=${balance:.2f}, Confidence={confidence:.4f}")
        logger.info(f"[POSITION SIZING] Sentiment={sentiment_score:.2f}, Risk Adj={sentiment_adjustment:.2f}")

        return quantity

    # ===== Drawdown Monitoring =====

    def check_drawdown_exceeded(self, balance, pnl, max_drawdown_pct=0.01, allow_high_risk=False):
        """
        Check if the current drawdown exceeds the maximum allowed percentage.

        Args:
            balance (float): Current account balance
            pnl (float): Current unrealized PnL
            max_drawdown_pct (float): Maximum allowed drawdown as a percentage of balance
            allow_high_risk (bool): If True, allows higher drawdown percentages

        Returns:
            bool: True if drawdown is exceeded, False otherwise

        Raises:
            RuntimeError: If balance or PnL is None
        """
        if balance is None or pnl is None:
            error_msg = "[DRAWDOWN] Cannot check drawdown: missing balance or PnL"
            logger.critical(error_msg)
            raise RuntimeError("Cannot check drawdown: missing balance or PnL – aborting trade.")

        # Enforce 7% drawdown cap unless allow_high_risk=True
        if max_drawdown_pct > 0.1 and not allow_high_risk:
            original_pct = max_drawdown_pct * 100
            max_drawdown_pct = 0.07  # Cap at 7%
            logger.warning(f"[DRAWDOWN] Max drawdown too high ({original_pct:.1f}%). Using 7% instead.")
        # Ensure max_drawdown_pct is reasonable
        elif max_drawdown_pct <= 0 or max_drawdown_pct > 0.2:
            logger.warning(f"[DRAWDOWN] Invalid max_drawdown_pct: {max_drawdown_pct}. Using default 0.01 (1%)")
            max_drawdown_pct = 0.01

        # Calculate maximum allowed loss
        max_loss = -abs(balance * max_drawdown_pct)

        # Check if PnL exceeds maximum loss
        drawdown_exceeded = pnl <= max_loss

        if drawdown_exceeded:
            logger.warning(f"[DRAWDOWN] Exceeded: {pnl:.2f} USDT vs limit {max_loss:.2f} USDT (balance: {balance:.2f})")
        else:
            logger.debug(f"[DRAWDOWN] Within limits: {pnl:.2f} USDT vs limit {max_loss:.2f} USDT (balance: {balance:.2f})")

        return drawdown_exceeded

    # ===== Stop Loss Calculation =====

    def calculate_stop_loss(self, price, confidence=None, side='long'):
        """
        Calculate the stop loss price based on current price, confidence, and trade side.
        Includes sanity checks to prevent invalid stop-loss prices.

        Args:
            price (float): Current market price
            confidence (float, optional): Prediction confidence (0.0-1.0)
            side (str): 'long' or 'short'

        Returns:
            float: Stop loss price

        Raises:
            ValueError: If price is invalid
        """
        # Validate price
        if not price or price <= 0:
            error_msg = f"[STOP-LOSS ERROR] Invalid price: {price}"
            logger.critical(error_msg)
            raise ValueError("Invalid price for stop-loss calculation – aborting trade.")

        # Get stop loss percentage from config or use default
        trading_config = self.config.get('trading', {})
        base_stop_loss_pct = trading_config.get('stop_loss_pct', 0.02)  # Default 2%

        # Adjust stop loss based on confidence if provided
        if confidence is not None:
            # Higher confidence = tighter stop loss
            if confidence >= 0.9:  # Very high confidence
                stop_loss_pct = base_stop_loss_pct * 0.8  # 80% of base (tighter)
            elif confidence >= 0.8:  # High confidence
                stop_loss_pct = base_stop_loss_pct * 0.9  # 90% of base
            elif confidence >= 0.7:  # Medium-high confidence
                stop_loss_pct = base_stop_loss_pct  # 100% of base (unchanged)
            else:  # Lower confidence
                stop_loss_pct = base_stop_loss_pct * 1.2  # 120% of base (wider)
        else:
            stop_loss_pct = base_stop_loss_pct

        # Enforce minimum and maximum stop loss percentages
        min_stop_loss_pct = 0.005  # 0.5% minimum
        max_stop_loss_pct = 0.05   # 5% maximum

        if stop_loss_pct < min_stop_loss_pct:
            logger.warning(f"[STOP-LOSS] Calculated stop loss {stop_loss_pct:.2%} below minimum {min_stop_loss_pct:.2%}. Using minimum.")
            stop_loss_pct = min_stop_loss_pct
        elif stop_loss_pct > max_stop_loss_pct:
            logger.warning(f"[STOP-LOSS] Calculated stop loss {stop_loss_pct:.2%} above maximum {max_stop_loss_pct:.2%}. Using maximum.")
            stop_loss_pct = max_stop_loss_pct

        # Calculate stop loss price based on side
        if side == 'long':
            stop_loss_price = price * (1 - stop_loss_pct)
        else: # short
            stop_loss_price = price * (1 + stop_loss_pct)


        # Log stop loss details
        logger.info(f"[STOP-LOSS] Calculated stop loss for {side.upper()} position: ${stop_loss_price:.2f} ({stop_loss_pct:.2%} from ${price:.2f})")

        return stop_loss_price
