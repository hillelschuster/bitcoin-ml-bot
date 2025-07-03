"""
Position management for the trading bot.

This module handles position entry logic, including long/short entries,
position sizing, and stop-loss placement.
"""

import time
import logging

logger = logging.getLogger("core.bot.position_manager")


class PositionManager:
    def __init__(self, exchange, state_manager, exit_handler, notifier, config, risk_manager=None):
        """
        Initialize the position manager.
        
        Args:
            exchange: Exchange client
            state_manager: State manager for position state
            exit_handler: Exit handler for emergency exits
            notifier: Notifier for sending alerts
            config: Configuration object
            risk_manager: Risk manager for position sizing (optional)
        """
        self.exchange = exchange
        self.state_manager = state_manager
        self.exit_handler = exit_handler
        self.notifier = notifier
        self.config = config
        self.risk_manager = risk_manager
        
    def enter_long(self, confidence, current_price, symbol=None):
        """
        Enter a long position with proper error handling and stop-loss placement.
        
        Args:
            confidence: Prediction confidence (0.0-1.0)
            current_price: Current market price
            symbol: Trading pair symbol (optional)
            
        Returns:
            bool: True if position was successfully entered, False otherwise
        """
        # Use the state manager's trade lock to prevent concurrent entry/exit operations
        if self.state_manager._trade_lock:
            with self.state_manager._trade_lock:
                return self._enter_long_internal(confidence, current_price, symbol)
        else:
            return self._enter_long_internal(confidence, current_price, symbol)
            
    def _enter_long_internal(self, confidence, current_price, symbol=None):
        """
        Internal implementation of enter_long with proper error handling and stop-loss placement.
        
        Args:
            confidence: Prediction confidence (0.0-1.0)
            current_price: Current market price
            symbol: Trading pair symbol (optional)
            
        Returns:
            bool: True if position was successfully entered, False otherwise
        """
        # Get symbol from config if not provided
        if symbol is None:
            symbol = self.config.get('symbol', 'BTC/USDT')
            
        # Check if trading is suspended due to circuit breaker
        if self.state_manager.trading_suspended:
            logger.warning("🚨 Cannot enter LONG - trading suspended by circuit breaker")
            return False

        # Verify we're not already in a position
        if self.state_manager.position:
            logger.warning(f"Cannot enter LONG - already in {self.state_manager.position.upper()} position")
            return False

        # Calculate position size with risk adjustment
        size = self.calculate_position_size(confidence, current_price)
        if size <= 0:
            logger.error(f"Invalid position size calculated: {size}")
            return False
            
        logger.info(f"Entering LONG position with size: {size} at price: {current_price}")

        # Execute the order
        try:
            order_result = self.exchange.place_order('BUY', quantity=size)

            if not order_result:
                logger.error("Failed to execute LONG entry")
                return False
        except (RuntimeError, ValueError) as e:
            logger.error(f"Exception during LONG order placement: {e}")
            return False

        # Update position state
        self.state_manager.position = "long"
        self.state_manager.position_entry_time = time.time()
        self.state_manager.save_position_state()

        # Place stop loss
        stop_price = self.risk_manager.calculate_stop_loss(current_price, confidence, side='long')
        logger.info(f"Placing LONG stop-loss at ${stop_price:.2f}")
        
        try:
            stop_result = self.place_stop_loss(symbol, size, stop_price)
            if stop_result:
                logger.info(f"Successfully placed LONG stop-loss at ${stop_price:.2f}")
            else:
                logger.warning(f"Failed to place stop-loss for LONG position")
        except (RuntimeError, ValueError) as e:
            error_msg = f"❌ CRITICAL: Exception placing stop loss for LONG position: {e}"
            logger.critical(error_msg)
            self.notifier.send(f"⚠️ CRITICAL: {error_msg}")
            # Emergency exit the position since it's unprotected
            self.exit_handler.emergency_exit(symbol, "Exception placing stop loss")
            return False

        # Log the trade
        try:
            from src.utils.logger import TradeLogger
            trade_logger = TradeLogger()
            trade_logger.log_trade(
                type="open",
                symbol=symbol,
                price=current_price,
                quantity=size,
                side="BUY",
                confidence=confidence
            )
        except (IOError, OSError, json.JSONDecodeError) as log_error:
            logger.error(f"Error logging trade: {log_error}")
            
        self.notifier.send(f"✅ LONG position opened at {current_price}")
        return True
        
    def enter_short(self, confidence, current_price, symbol=None):
        """
        Enter a short position with proper error handling and stop-loss placement.
        
        Args:
            confidence: Prediction confidence (0.0-1.0)
            current_price: Current market price
            symbol: Trading pair symbol (optional)
            
        Returns:
            bool: True if position was successfully entered, False otherwise
        """
        # Use the state manager's trade lock to prevent concurrent entry/exit operations
        if self.state_manager._trade_lock:
            with self.state_manager._trade_lock:
                return self._enter_short_internal(confidence, current_price, symbol)
        else:
            return self._enter_short_internal(confidence, current_price, symbol)
            
    def _enter_short_internal(self, confidence, current_price, symbol=None):
        """
        Internal implementation of enter_short with proper error handling and stop-loss placement.
        
        Args:
            confidence: Prediction confidence (0.0-1.0)
            current_price: Current market price
            symbol: Trading pair symbol (optional)
            
        Returns:
            bool: True if position was successfully entered, False otherwise
        """
        # Get symbol from config if not provided
        if symbol is None:
            symbol = self.config.get('symbol', 'BTC/USDT')
            
        # Check if trading is suspended due to circuit breaker
        if self.state_manager.trading_suspended:
            logger.warning("🚨 Cannot enter SHORT - trading suspended by circuit breaker")
            return False

        # Verify we're not already in a position
        if self.state_manager.position:
            logger.warning(f"Cannot enter SHORT - already in {self.state_manager.position.upper()} position")
            return False

        # Calculate position size with risk adjustment
        size = self.calculate_position_size(confidence, current_price)
        if size <= 0:
            logger.error(f"Invalid position size calculated: {size}")
            return False
            
        logger.info(f"Entering SHORT position with size: {size} at price: {current_price}")

        # Execute the order
        try:
            order_result = self.exchange.place_order('SELL', quantity=size)

            if not order_result:
                logger.error("Failed to execute SHORT entry")
                return False
        except (RuntimeError, ValueError) as e:
            logger.error(f"Exception during SHORT order placement: {e}")
            return False

        # Update position state
        self.state_manager.position = "short"
        self.state_manager.position_entry_time = time.time()
        self.state_manager.save_position_state()

        # Place stop loss
        stop_price = self.risk_manager.calculate_stop_loss(current_price, confidence, side='short')
        logger.info(f"Placing SHORT stop-loss at ${stop_price:.2f}")
        
        try:
            stop_result = self.place_stop_loss(symbol, size, stop_price, is_long=False)
            if stop_result:
                logger.info(f"Successfully placed SHORT stop-loss at ${stop_price:.2f}")
            else:
                logger.warning(f"Failed to place stop-loss for SHORT position")
        except (RuntimeError, ValueError) as e:
            error_msg = f"❌ CRITICAL: Exception placing stop loss for SHORT position: {e}"
            logger.critical(error_msg)
            self.notifier.send(f"⚠️ CRITICAL: {error_msg}")
            # Emergency exit the position since it's unprotected
            self.exit_handler.emergency_exit(symbol, "Exception placing stop loss")
            return False

        # Log the trade
        try:
            from src.utils.logger import TradeLogger
            trade_logger = TradeLogger()
            trade_logger.log_trade(
                type="open",
                symbol=symbol,
                price=current_price,
                quantity=size,
                side="SELL",
                confidence=confidence
            )
        except (IOError, OSError, json.JSONDecodeError) as log_error:
            logger.error(f"Error logging trade: {log_error}")
            
        self.notifier.send(f"✅ SHORT position opened at {current_price}")
        return True
        
    def calculate_position_size(self, confidence, current_price=None, balance=None):
        """
        Calculate position size based on account balance, confidence, and risk parameters.
        
        Args:
            confidence: Prediction confidence (0.0-1.0)
            current_price: Current market price (optional)
            balance: Account balance (optional)
            
        Returns:
            float: Position size in base currency (e.g., BTC)
        """
        # If risk manager is available, use it
        if self.risk_manager:
            return self.risk_manager.calculate_position_size(confidence, balance, current_price)
            
        # Otherwise, implement basic position sizing logic
        logger.info(f"[POSITION SIZING] Starting calculation with confidence={confidence:.4f}")
        
        # Get current price if not provided
        if current_price is None:
            try:
                current_price = self.exchange.get_market_price(self.config.get('symbol', 'BTC/USDT'))
            except (RuntimeError, ValueError) as e:
                logger.error(f"[POSITION SIZING] Failed to get price: {e}")
                return 0
                
        # Get account balance if not provided
        if balance is None:
            try:
                balance = self.exchange.get_account_balance()
            except (RuntimeError, ValueError) as e:
                logger.error(f"[POSITION SIZING] Failed to get balance: {e}")
                return 0
                
        # Validate inputs
        if balance <= 0 or current_price <= 0 or confidence <= 0:
            logger.error(f"[POSITION SIZING] Invalid inputs: balance={balance}, price={current_price}, confidence={confidence}")
            return 0
            
        # Get risk parameters from config
        risk_per_trade = self.config.get('risk_per_trade', 0.01)  # Default 1%
        max_position_pct = self.config.get('max_position_pct', 0.1)  # Default 10%
        leverage = self.config.get('leverage', 1)  # Default 1x
        
        # Adjust risk based on confidence
        adjusted_risk = risk_per_trade * confidence
        
        # Calculate margin amount
        margin_amount = balance * adjusted_risk
        
        # Calculate position size
        notional_value = margin_amount * leverage
        quantity = round(notional_value / current_price, 5)  # 5 decimal places for BTC
        
        # Ensure minimum notional value of $100 (Binance requirement)
        min_notional = 100.0
        min_btc_quantity = min_notional / current_price if current_price > 0 else 0.002
        
        if quantity < min_btc_quantity:
            logger.warning(f"[POSITION SIZING] Quantity {quantity:.6f} BTC below min notional (${min_notional}). Increasing to {min_btc_quantity:.6f} BTC.")
            quantity = round(min_btc_quantity, 5)
            
        # Ensure maximum position size
        max_quantity = (balance * max_position_pct) / current_price
        if quantity > max_quantity:
            logger.warning(f"[POSITION SIZING] Quantity {quantity:.6f} BTC above max position size. Reducing to {max_quantity:.6f} BTC.")
            quantity = round(max_quantity, 5)
            
        # Log final position size
        final_value = quantity * current_price
        pct_of_balance = (final_value / balance) * 100 if balance > 0 else 0
        logger.info(f"[POSITION SIZING] FINAL: {quantity:.6f} BTC (${final_value:.2f}, {pct_of_balance:.2f}% of balance)")
        
        return quantity
        
    def place_stop_loss(self, symbol, size, stop_price, is_long=True):
        """
        Place a stop-loss order for a position.
        
        Args:
            symbol: Trading pair symbol
            size: Position size
            stop_price: Stop-loss price
            is_long: Whether the position is long (True) or short (False)
            
        Returns:
            bool: True if stop-loss was successfully placed, False otherwise
        """
        try:
            # Cancel any existing stop orders first
            self.exchange.cancel_all_orders(symbol)
            
            # Place the stop-loss order
            side = "SELL" if is_long else "BUY"
            stop_result = self.exchange.place_stop_loss_once(symbol, stop_price, size)
            
            if stop_result:
                logger.info(f"Successfully placed {side} stop-loss at ${stop_price:.2f}")
                return True
            else:
                logger.warning(f"Failed to place stop-loss for {side} position")
                return False
        except (RuntimeError, ValueError) as e:
            logger.error(f"Error placing stop-loss: {e}")
            return False
