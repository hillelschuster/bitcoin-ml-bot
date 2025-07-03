"""
Safety monitoring for the trading bot.

This module handles drawdown checks, emergency exit retry counters,
and circuit breaker logic to ensure safe trading operations.
"""

import time
import logging

from src.utils.config import Config

logger = logging.getLogger("core.bot.safety_monitor")

# Constants
MAX_EMERGENCY_EXIT_ATTEMPTS = Config().get('risk_management', {}).get('emergency_exit_max_attempts', 5)


class SafetyMonitor:
    def __init__(self, exchange, config, state_manager, notifier):
        """
        Initialize the safety monitor.
        
        Args:
            exchange: Exchange client
            config: Configuration object
            state_manager: State manager for position and circuit breaker state
            notifier: Notifier for sending alerts
        """
        self.exchange = exchange
        self.config = config
        self.state_manager = state_manager
        self.notifier = notifier
        self.position_closing = False
        
    def monitor_drawdown(self, symbol, entry_time, max_drawdown_pct=None):
        """
        Monitor position for excessive drawdown and exit if necessary.
        
        Args:
            symbol: Trading pair symbol
            entry_time: Position entry time
            max_drawdown_pct: Maximum allowed drawdown percentage (optional)
            
        Returns:
            bool: True if drawdown limit is exceeded, False otherwise
        """
        # Skip if no position or already in process of closing
        if not self.state_manager.position or self.position_closing:
            return False

        # Give position some time to breathe before monitoring drawdown
        # (prevents immediate exit on normal price fluctuations)
        current_time = time.time()
        grace_period = self.config.get('drawdown_grace_period_seconds', 30)
        if current_time - entry_time < grace_period:
            return False

        # Fetch current balance and unrealized PnL
        try:
            balance = self.exchange.get_account_balance()
            pnl = self.exchange.get_unrealized_pnl(symbol)

            # Validate the data
            if balance <= 0:
                logger.error(f"Invalid balance for drawdown monitoring: {balance}")
                return False

            # Log PnL for monitoring
            logger.debug(f"Current PnL: {pnl:.2f} USDT on {self.state_manager.position} position")

            # Get max drawdown percentage from config if not provided
            if max_drawdown_pct is None:
                max_drawdown_pct = self.config.get('max_drawdown_pct', 0.01)

            # Check if drawdown is exceeded
            if self.check_drawdown_exceeded(balance, pnl, max_drawdown_pct):
                logger.warning(f"⚠️ Drawdown limit exceeded - initiating emergency exit")
                return True
                
            return False
        except (RuntimeError, ValueError) as e:
            logger.error(f"Error in drawdown monitoring: {e}")
            return False
            
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
        """
        if balance is None or pnl is None:
            error_msg = "[DRAWDOWN] Cannot check drawdown: missing balance or PnL"
            logger.critical(error_msg)
            return False

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
        
    def record_exit_failure(self, reason, position=None, attempts=0):
        """
        Record an emergency exit failure and check if circuit breaker should be activated.
        
        Args:
            reason: Reason for the failure
            position: Position side (optional)
            attempts: Number of exit attempts (optional)
            
        Returns:
            bool: True if circuit breaker was activated, False otherwise
        """
        current_time = time.time()

        # Create a failure record
        failure = {
            'timestamp': current_time,
            'reason': reason,
            'position': position or self.state_manager.position,
            'attempts': attempts
        }

        # Add to the list of failures
        self.state_manager.emergency_exit_failures.append(failure)
        self.state_manager.emergency_exit_failures_last_24h += 1

        # Save metrics to file
        self.state_manager.save_emergency_exit_metrics()

        # Check if circuit breaker should be activated
        if self.check_circuit_breaker_condition():
            self.activate_circuit_breaker()
            return True
            
        return False
        
    def check_circuit_breaker_condition(self):
        """
        Check if circuit breaker should be activated.
        
        Returns:
            bool: True if circuit breaker should be activated, False otherwise
        """
        # Check if we have 3 or more failures in the last 15 minutes
        current_time = time.time()
        fifteen_minutes_ago = current_time - (15 * 60)

        # Filter failures to only include those from the last 15 minutes
        recent_failures = [f for f in self.state_manager.emergency_exit_failures 
                          if f.get('timestamp', 0) > fifteen_minutes_ago]

        if len(recent_failures) >= 3:
            logger.critical(f"🚨 Circuit breaker condition met: {len(recent_failures)} failures in the last 15 minutes")
            return True

        return False
        
    def activate_circuit_breaker(self, duration_seconds=900):
        """
        Activate the circuit breaker to prevent new trades.
        
        Args:
            duration_seconds: Duration in seconds for the circuit breaker to remain active
            
        Returns:
            bool: True if circuit breaker was activated, False if it was already active
        """
        if not self.state_manager.trading_suspended:
            self.state_manager.trading_suspended = True
            self.state_manager.save_circuit_breaker_state(active=True, duration_seconds=duration_seconds)

            # Log and notify
            logger.critical("🚨 CIRCUIT BREAKER ACTIVATED: Too many failed exits")
            self.notifier.send("🚨 CIRCUIT BREAKER ACTIVATED: Too many failed exits. Trading suspended for 15 minutes.")
            return True
            
        return False
        
    def is_trading_allowed(self):
        """
        Check if trading is allowed based on circuit breaker state.
        
        Returns:
            bool: True if trading is allowed, False otherwise
        """
        return not self.state_manager.trading_suspended
