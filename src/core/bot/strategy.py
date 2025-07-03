"""Trading strategy module for the Bitcoin ML Trading Bot.

Provides functions and classes for generating trading signals and executing trades.
Includes Smart Confidence Selector for fallback and LSTM-based prediction.
"""

import logging
import time
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, Union
from src.core.utils.sentiment_utils import fetch_sentiment_score

logger = logging.getLogger('core.bot.strategy')

def should_enter_trade(signal, df):
    """Determine if a trade entry should be executed based on signal and volume confirmation.

    Args:
        signal (int): The predicted signal from the model (1 for buy)
        df (pandas.DataFrame): Market data with OHLCV information

    Returns:
        bool: True if a buy trade should be executed, False otherwise
    """
    return signal == 1 and df["volume"].iloc[-1] > df["volume"].rolling(10).mean().iloc[-1]

def should_enter_long(signal, df):
    """Determine if a long trade entry should be executed based on signal and volume confirmation.

    Args:
        signal (int): The predicted signal from the model (1 for buy)
        df (pandas.DataFrame): Market data with OHLCV information

    Returns:
        bool: True if a long trade should be executed, False otherwise
    """
    if signal == 1 and df["volume"].iloc[-1] > df["volume"].rolling(10).mean().iloc[-1]:
        logger.info("Long entry signal detected with volume confirmation")
        return True
    return False

def should_enter_short(signal, df):
    """Determine if a short trade entry should be executed based on signal and volume confirmation.

    Args:
        signal (int): The predicted signal from the model (-1 for sell)
        df (pandas.DataFrame): Market data with OHLCV information

    Returns:
        bool: True if a short trade should be executed, False otherwise
    """
    if signal == -1 and df["volume"].iloc[-1] > df["volume"].rolling(10).mean().iloc[-1]:
        logger.info("Short entry signal detected with volume confirmation")
        return True
    return False

def should_exit_trade(signal, df=None):
    """Determine if a trade exit should be executed based on signal.

    Args:
        signal (int): The predicted signal from the model
        df (pandas.DataFrame, optional): Market data (not used in this implementation)

    Returns:
        bool: True if the trade should be exited, False otherwise
    """
    # df parameter is kept for compatibility but not used in this simple implementation
    return signal == -1

def should_exit(signal, position=None, df=None):
    """Determine if a position should be exited based on signal and current position.

    Args:
        signal (int): The predicted signal from the model
        position (str): Current position ('long', 'short', or None)
        df (pandas.DataFrame, optional): Market data (not used in this implementation)

    Returns:
        bool: True if the position should be exited, False otherwise
    """
    # Exit on neutral signal
    if signal == 0:
        return True
    # Exit long position on sell signal
    if position == "long" and signal == -1:
        return True
    # Exit short position on buy signal
    if position == "short" and signal == 1:
        return True
    return False

class TradingStrategy:
    """Trading strategy implementation."""

    def __init__(self, config, exchange, trade_logger=None, risk_manager=None):
        """Initialize the trading strategy.

        Args:
            config: Configuration object or dictionary
            exchange: Exchange interface for executing trades
            trade_logger: Optional logger for trade data
            risk_manager: Optional external risk manager instance
        """
        self.config = config
        self.exchange = exchange
        self.trade_logger = trade_logger
        self.risk_manager = risk_manager

        # Get trading configuration
        trading_config = config.get('trading', {})
        self.symbol = trading_config.get('symbol', 'BTCUSDT')
        self.position_size = trading_config.get('position_size', 0.01)
        self.max_positions = trading_config.get('max_positions', 3)
        self.stop_loss_pct = trading_config.get('stop_loss_pct', 0.01)
        self.take_profit_pct = trading_config.get('take_profit_pct', 0.02)

        # Risk management parameters
        self.max_consecutive_losses = trading_config.get('max_consecutive_losses', 3)
        self.drawdown_threshold = trading_config.get('drawdown_threshold', 0.05)

        # Strategy parameters
        self.min_change_threshold = 1.0  # Minimum price change percentage to trigger a trade

        logger.info(f"Initialized trading strategy for {self.symbol}")
        if self.risk_manager:
            logger.info("Using external risk manager for risk checks")

    def generate_signal(self, current_price, predicted_price):
        """Generate trading signal based on price prediction."""
        try:
            # Calculate price change percentage
            change_pct = (predicted_price - current_price) / current_price * 100

            # Determine direction
            direction = 'up' if change_pct > 0 else 'down'

            # Log prediction
            logger.info(f"Prediction: {direction.upper()} {abs(change_pct):.2f}%")

            # Check if change percentage exceeds threshold
            if abs(change_pct) < self.min_change_threshold:
                logger.info(f"Change {abs(change_pct):.2f}% below threshold {self.min_change_threshold}%")
                return 'hold'

            # Generate signal based on direction
            return 'buy' if direction == 'up' else 'sell' if direction == 'down' else 'hold'

        except Exception as e:
            logger.error(f"Error generating signal: {e}")
            return 'hold'

    def execute(self, signal, confidence=None, market_volatility=None):
        """Execute the trading strategy.

        Args:
            signal (str): Trading signal ('buy', 'sell', 'hold')
            confidence (float, optional): Signal confidence level (0.0-1.0)
            market_volatility (float, optional): Current market volatility metric

        Returns:
            dict: Execution result with action and details
        """
        try:
            # Check risk management rules
            can_trade, reason = self._check_risk_management()
            if not can_trade:
                logger.info(f"Signal: {signal.upper()} - Risk management prevented trade: {reason}")
                return {'action': 'hold', 'reason': f'risk_management: {reason}'}

            if signal == 'hold':
                logger.info("Signal: HOLD - No action taken")
                return {'action': 'hold'}

            # Get current market price
            current_price = self._get_current_price()
            if current_price is None or current_price <= 0:
                logger.error("Could not get current price")
                return {'action': 'hold', 'reason': 'price_unavailable'}

            # Calculate dynamic SL/TP based on confidence and volatility if provided
            stop_loss_pct = None
            take_profit_pct = None

            if confidence is not None or market_volatility is not None:
                # Adjust SL/TP based on confidence and volatility
                stop_loss_pct, take_profit_pct = self._calculate_dynamic_risk_params(
                    confidence, market_volatility
                )

            # Execute signal
            signal_lower = signal.lower()
            if signal_lower in ('buy', 'up'):
                return self._execute_buy(
                    current_price,
                    stop_loss_pct=stop_loss_pct,
                    take_profit_pct=take_profit_pct
                ) or {'action': 'hold', 'reason': 'buy_failed'}
            elif signal_lower in ('sell', 'down'):
                return self._execute_sell(current_price) or {'action': 'hold', 'reason': 'sell_failed'}
            else:
                logger.info(f"Signal: {signal} - No matching action, treating as HOLD")
                return {'action': 'hold', 'reason': 'unknown_signal'}

        except Exception as e:
            logger.error(f"Error executing strategy: {e}")
            return {'action': 'hold', 'reason': f'error: {str(e)}'}

    def _get_current_price(self):
        """Get current market price.

        Returns:
            float or None: Current market price or None if price fetch fails
        """
        try:
            price = self.exchange.get_price(self.symbol)
            if price is None or price <= 0:
                logger.error(f"Invalid price returned: {price}")
                return None
            return price
        except Exception as e:
            logger.error(f"Error getting current price: {e}")
            return None  # Return None instead of a fallback price

    def _check_risk_management(self):
        """Check risk management rules based on recent PnLs.

        This method will use an external RiskManager if available, otherwise
        it will perform basic risk checks internally.

        Returns:
            tuple: (can_trade, reason)
                can_trade: True if risk management rules allow trading, False otherwise
                reason: Reason for not trading if can_trade is False
        """
        # If we have a risk_manager attribute, use it for risk checks
        if hasattr(self, 'risk_manager') and self.risk_manager is not None:
            logger.debug("Using external RiskManager for risk checks")
            return self.risk_manager.can_trade()

        # Otherwise, perform basic risk checks internally
        logger.debug("Using internal risk checks (no external RiskManager available)")

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
            logger.warning(f"Risk management: {reason}")
            return False, reason

        # Check for drawdown
        total_pnl = sum(recent_pnls)
        initial_balance = self.exchange.get_balance() - total_pnl
        if initial_balance > 0:
            drawdown = abs(total_pnl) / initial_balance if total_pnl < 0 else 0
            if drawdown > self.drawdown_threshold:
                reason = f"Drawdown threshold exceeded ({drawdown:.2%})"
                logger.warning(f"Risk management: {reason}")
                return False, reason

        return True, ""

    def _execute_buy(self, current_price, stop_loss_pct=None, take_profit_pct=None):
        """Execute buy order with dynamic stop loss and take profit levels.

        Args:
            current_price (float): Current market price
            stop_loss_pct (float, optional): Override for stop loss percentage
            take_profit_pct (float, optional): Override for take profit percentage

        Returns:
            dict: Order execution details or None if execution failed
        """
        try:
            # Cancel any existing orders
            self.exchange.cancel_all_orders(self.symbol)

            # Calculate position size
            account_balance = self.exchange.get_balance()
            if account_balance is None or account_balance <= 0:
                logger.warning(f"Invalid account balance: {account_balance}")
                return {'action': 'hold', 'reason': 'invalid_balance'}

            # Calculate raw quantity
            raw_quantity = self._calculate_position_size(account_balance, current_price)

            # Round quantity to 3 decimal places for Binance compatibility
            quantity = round(raw_quantity, 3)

            # Log both pre and post rounding
            logger.info(f"Position sizing: raw={raw_quantity}, rounded={quantity}")

            if quantity <= 0:
                logger.warning("Calculated quantity is zero or negative")
                return {'action': 'hold', 'reason': 'invalid_quantity'}

            # Place market buy order
            order = self.exchange.create_market_buy_order(self.symbol, quantity)
            if not order:
                logger.error("Failed to place market buy order")
                return {'action': 'hold', 'reason': 'order_failed'}

            order_id = order.get('orderId')
            logger.info(f"Market buy order placed: {order_id}")

            # Use provided SL/TP percentages or fall back to defaults
            actual_sl_pct = stop_loss_pct if stop_loss_pct is not None else self.stop_loss_pct
            actual_tp_pct = take_profit_pct if take_profit_pct is not None else self.take_profit_pct

            # Log if using dynamic values
            if stop_loss_pct is not None or take_profit_pct is not None:
                logger.info(f"Using dynamic risk parameters: SL={actual_sl_pct:.2%}, TP={actual_tp_pct:.2%}")

            # Calculate stop loss and take profit levels
            stop_loss = current_price * (1 - actual_sl_pct)
            take_profit = current_price * (1 + actual_tp_pct)

            # Place stop loss order
            try:
                stop_loss_order = self.exchange.place_stop_loss_once(stop_loss, quantity, current_price)
                stop_loss_id = stop_loss_order.get('orderId') if stop_loss_order else None

                if not stop_loss_order:
                    logger.warning("Failed to place stop loss order")
            except Exception as sl_error:
                logger.error(f"Error placing stop loss order: {sl_error}")
                stop_loss_order = None
                stop_loss_id = None

            # Place take profit order with retry
            take_profit_order = None
            take_profit_id = None
            max_tp_attempts = 2  # Allow 1 retry

            for attempt in range(max_tp_attempts):
                try:
                    take_profit_order = self.exchange.create_take_profit_order(self.symbol, quantity, take_profit)
                    take_profit_id = take_profit_order.get('orderId') if take_profit_order else None

                    if take_profit_order:
                        if attempt > 0:
                            logger.info(f"Take profit order placed successfully on retry attempt {attempt}")
                        break
                    else:
                        logger.warning(f"Failed to place take profit order (attempt {attempt+1}/{max_tp_attempts})")
                        if attempt < max_tp_attempts - 1:
                            logger.info("Retrying take profit order placement...")
                            time.sleep(1)  # Brief pause before retry
                except Exception as tp_error:
                    logger.error(f"Error placing take profit order (attempt {attempt+1}/{max_tp_attempts}): {tp_error}")
                    if attempt < max_tp_attempts - 1:
                        logger.info("Retrying take profit order placement after error...")
                        time.sleep(1)  # Brief pause before retry

            return {
                'action': 'buy',
                'symbol': self.symbol,
                'quantity': quantity,
                'price': current_price,
                'value': quantity * current_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'order_id': order_id,
                'stop_loss_id': stop_loss_id,
                'take_profit_id': take_profit_id,
                'stop_loss_pct': actual_sl_pct,
                'take_profit_pct': actual_tp_pct
            }
        except Exception as e:
            logger.error(f"Error in buy execution: {e}")
            return {'action': 'hold', 'reason': f'execution_error: {str(e)}'}

    def _execute_sell(self, current_price):
        """Execute sell order."""
        try:
            # Cancel any existing orders
            self.exchange.cancel_all_orders(self.symbol)

            # Get current position
            position = self.exchange.get_position_amount()
            if position is None:
                logger.warning("Failed to get position amount")
                return {'action': 'hold', 'reason': 'position_fetch_failed'}

            if position <= 0:
                logger.warning("No position to sell")
                return {'action': 'hold', 'reason': 'no_position'}

            # Round quantity to 3 decimal places for Binance compatibility
            raw_quantity = abs(position)
            quantity = round(raw_quantity, 3)

            # Log both pre and post rounding
            logger.info(f"Sell quantity: raw={raw_quantity}, rounded={quantity}")

            # Place market sell order
            order = self.exchange.create_market_sell_order(self.symbol, quantity)
            if not order:
                logger.error("Failed to place market sell order")
                return {'action': 'hold', 'reason': 'order_failed'}

            order_id = order.get('orderId')
            logger.info(f"Market sell order placed: {order_id}")

            return {
                'action': 'sell',
                'symbol': self.symbol,
                'quantity': quantity,
                'price': current_price,
                'value': quantity * current_price,
                'order_id': order_id
            }
        except Exception as e:
            logger.error(f"Error in sell execution: {e}")
            return {'action': 'hold', 'reason': f'execution_error: {str(e)}'}

    def _calculate_dynamic_risk_params(self, confidence=None, volatility=None):
        """Calculate dynamic stop loss and take profit percentages based on market conditions.

        Args:
            confidence (float, optional): Signal confidence level (0.0-1.0)
            volatility (float, optional): Market volatility metric

        Returns:
            tuple: (stop_loss_pct, take_profit_pct)
        """
        # Start with default values
        stop_loss_pct = self.stop_loss_pct
        take_profit_pct = self.take_profit_pct

        # Adjust based on confidence if provided
        if confidence is not None:
            # Higher confidence = tighter stop loss, higher take profit
            confidence_factor = min(max(confidence, 0.1), 1.0)  # Clamp between 0.1 and 1.0

            # Adjust stop loss (lower = tighter)
            stop_loss_pct = self.stop_loss_pct * (1.5 - confidence_factor * 0.5)  # Range: 50% to 100% of default

            # Adjust take profit (higher = more ambitious)
            take_profit_pct = self.take_profit_pct * (0.8 + confidence_factor * 0.4)  # Range: 80% to 120% of default

        # Adjust based on volatility if provided
        if volatility is not None:
            # Higher volatility = wider stop loss to avoid noise
            volatility_factor = min(max(volatility, 0.5), 2.0)  # Clamp between 0.5 and 2.0

            # Widen stop loss in high volatility
            stop_loss_pct = stop_loss_pct * volatility_factor

            # Increase take profit in high volatility
            take_profit_pct = take_profit_pct * volatility_factor

        # Log the adjustments
        logger.info(f"Dynamic risk parameters: SL={stop_loss_pct:.2%} (default: {self.stop_loss_pct:.2%}), TP={take_profit_pct:.2%} (default: {self.take_profit_pct:.2%})")

        return stop_loss_pct, take_profit_pct

    def _calculate_position_size(self, account_balance, current_price):
        """Calculate position size based on risk management rules."""
        # Get risk per trade from config
        trading_config = self.config.get('trading', {})
        risk_per_trade = trading_config.get('risk_per_trade', 0.02)  # Default 2%

        # Calculate risk amount
        risk_amount = account_balance * risk_per_trade

        # Calculate position size based on risk and stop loss
        position_size = risk_amount / (current_price * self.stop_loss_pct)

        # Limit position size to max percentage of account
        max_position_size = account_balance * self.position_size
        if position_size > max_position_size:
            position_size = max_position_size

        # Log position sizing details
        logger.info(f"Position sizing: risk_amount=${risk_amount:.2f}, max_size=${max_position_size:.2f}, final_size=${position_size:.2f}")

        # Return raw position size (will be rounded at the execution level)
        return position_size

    # ===== Smart Confidence Selector Methods =====

    def model_trained(self, features: dict) -> bool:
        """Check if the model is trained and ready for prediction."""
        # Check if we have a model attribute
        if not hasattr(self, 'model') or self.model is None:
            logger.warning("No model available for prediction")
            return False

        # Check if the model has a predict method
        if not hasattr(self.model, 'predict'):
            logger.warning("Model does not have a predict method")
            return False

        # Check if we have enough features for prediction
        if not features or not isinstance(features, dict):
            logger.warning("Invalid features for prediction")
            return False

        # Check if we have price data
        if 'price_data' not in features or not features['price_data']:
            logger.warning("No price data in features")
            return False

        return True

    def get_confidence(self, features: dict) -> Tuple[str, float]:
        """Get prediction confidence using the Smart Confidence Selector.

        This method implements a fallback mechanism for confidence calculation:
        1. Try to use the model's predict method if available
        2. If model prediction fails, use sentiment-based confidence
        3. If sentiment is unavailable, use a conservative default

        Args:
            features (dict): Feature dictionary with price_data, sentiment_score, etc.

        Returns:
            tuple: (signal, confidence)
                signal: 'UP', 'DOWN', or 'HOLD'
                confidence: Confidence score (0.0-1.0)
        """
        # Default values
        signal = 'HOLD'
        confidence = 0.5

        # Try model-based prediction first
        if self.model_trained(features):
            try:
                # Extract price data for prediction
                price_data = features.get('price_data', [])

                # Make prediction
                model_signal, model_confidence = self.model.predict(price_data)

                # Log prediction
                logger.info(f"Model prediction: {model_signal} with confidence {model_confidence:.4f}")

                # Return model prediction if valid
                if model_signal in ('UP', 'DOWN', 'HOLD') and 0.0 <= model_confidence <= 1.0:
                    return model_signal, model_confidence
                else:
                    logger.warning(f"Invalid model prediction: {model_signal}, {model_confidence}")
            except Exception as e:
                logger.error(f"Error in model prediction: {e}")
                # Continue to fallback methods

        # Fallback to sentiment-based confidence
        try:
            # Get sentiment score from features or fetch it
            sentiment_score = features.get('sentiment_score')
            if sentiment_score is None:
                sentiment_score = fetch_sentiment_score()

            # Normalize sentiment score to 0-1 range
            normalized_sentiment = sentiment_score / 100.0

            # Determine signal based on sentiment
            if normalized_sentiment > 0.6:  # Bullish sentiment
                signal = 'UP'
                confidence = min(normalized_sentiment, 0.75)  # Cap at 0.75 for sentiment-based
            elif normalized_sentiment < 0.4:  # Bearish sentiment
                signal = 'DOWN'
                confidence = min(1.0 - normalized_sentiment, 0.75)  # Cap at 0.75 for sentiment-based
            else:  # Neutral sentiment
                signal = 'HOLD'
                confidence = 0.5

            logger.info(f"Sentiment-based prediction: {signal} with confidence {confidence:.4f}")
            return signal, confidence
        except Exception as e:
            logger.error(f"Error in sentiment-based confidence: {e}")

        # Final fallback to conservative default
        logger.warning("Using conservative default prediction: HOLD with confidence 0.5")
        return 'HOLD', 0.5

    def predict(self, market_data: pd.DataFrame) -> Dict[str, Union[str, float]]:
        """Make a prediction based on market data.

        Args:
            market_data (pd.DataFrame): Market data with OHLCV columns

        Returns:
            dict: Prediction result with signal, confidence, and additional info
        """
        try:
            # Check if model is untrained
            if not self.model.model_trained:
                logger.warning("⚠️ Strategy skipped: LSTM model is untrained or placeholder. No trading decision made.")
                return None

            # Prepare features for prediction
            features = {
                'price_data': market_data['close'].values if 'close' in market_data.columns else [],
                'sentiment_score': None  # Will be fetched in get_confidence if needed
            }

            # Get prediction and confidence
            signal, confidence = self.get_confidence(features)

            # Get current price
            current_price = market_data['close'].iloc[-1] if 'close' in market_data.columns else None

            # Return prediction result
            return {
                'signal': signal,
                'confidence': confidence,
                'price': current_price,
                'timestamp': pd.Timestamp.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            return {
                'signal': 'HOLD',
                'confidence': 0.5,
                'price': None,
                'timestamp': pd.Timestamp.now().isoformat(),
                'error': str(e)
            }
